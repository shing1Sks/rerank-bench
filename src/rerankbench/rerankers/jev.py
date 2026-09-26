"""Jev arms: SystemOne client + listwise and pointwise rerankers.

Wire format mirrors JevCity's proven client: POST {typesafe_url} with
Authorization: Bearer {typesafe_key} and body {state, model, questions}.
A `choice` question returns answers[name] = {choice, confidence, probabilities};
an `noul` question returns answers[name] = {noul} in 0..1.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import httpx

from rerankbench.rerankers import RerankResult

RETRY_STATUSES = {429, 529}


class JevError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _default_post(url: str, body: dict, headers: dict) -> dict:
    resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
    if resp.status_code != 200:
        raise JevError(f"http {resp.status_code}: {resp.text[:500]}", status=resp.status_code)
    return resp.json()


def post_systemone(
    cfg,
    state: str,
    questions: dict,
    post: Callable | None = None,
    attempts: int = 5,
    sleep: Callable = time.sleep,
) -> dict:
    """POST to SystemOne with backoff on 429/529; other statuses fail fast."""
    post = post or _default_post
    headers = {"Authorization": f"Bearer {cfg.typesafe_key}"}
    body = {"state": state, "model": cfg.jev_model, "questions": questions}
    for attempt in range(attempts):
        try:
            return post(cfg.typesafe_url, body, headers)
        except JevError as e:
            if e.status not in RETRY_STATUSES or attempt == attempts - 1:
                raise
            sleep(min(30, 2**attempt))
    raise JevError("unreachable")  # pragma: no cover


def truncate_for_listwise(text: str, cap_tokens: int) -> str:
    """Word-truncate to ~3/4 of the token cap; appends ' ...' when cut."""
    max_words = cap_tokens * 3 // 4
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


class JevListwise:
    """One SystemOne call per query: a single choice question over all candidates."""

    arm = "jev_listwise"

    def __init__(self, cfg, post: Callable):
        self._cfg = cfg
        self._post = post

    def rerank(self, query: str, candidates: list, query_meta: dict) -> RerankResult:
        t0 = time.perf_counter()
        criteria = {}
        truncated = False
        for c in candidates:
            text = truncate_for_listwise(c.text, self._cfg.listwise_chunk_cap)
            if len(text.split()) < len(c.text.split()):
                truncated = True
            criteria[c.id] = text
        state = (
            "You are ranking passages for a retrieval benchmark.\n"
            f"Query: {query}"
        )
        questions = {
            "q_rank": {
                "type": "choice",
                "instructions": "Which passage best answers the query?",
                "criteria": criteria,
            }
        }
        resp = post_systemone(self._cfg, state, questions, post=self._post)
        try:
            probabilities = resp["answers"]["q_rank"]["probabilities"]
            input_tokens = resp["usage"]["input_tokens"]
        except (KeyError, TypeError) as e:
            raise JevError(f"unexpected SystemOne response shape: {e}") from e
        missing = [c.id for c in candidates if c.id not in probabilities]
        if missing:
            raise JevError(f"missing probabilities for {missing}")
        ranked = sorted(candidates, key=lambda c: probabilities[c.id], reverse=True)
        return RerankResult(
            arm=self.arm,
            query_id=query_meta.get("query_id", ""),
            ranked_ids=[c.id for c in ranked],
            scores=[probabilities[c.id] for c in ranked],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=input_tokens,
            calls=1,
            cost_usd=input_tokens / 1e6 * self._cfg.jev_price_per_mtok,
            flags={"truncated": True} if truncated else {},
        )


class JevPointwise:
    """One noul SystemOne call per (query, candidate), fanned out on a thread pool."""

    arm = "jev_pointwise"

    def __init__(self, cfg, post: Callable, max_workers: int = 8):
        self._cfg = cfg
        self._post = post
        self._max_workers = max_workers

    def rerank(self, query: str, candidates: list, query_meta: dict) -> RerankResult:
        t0 = time.perf_counter()

        def score_one(c):
            text = truncate_for_listwise(c.text, self._cfg.listwise_chunk_cap)
            state = f"Query: {query}"
            questions = {
                "q_rel": {
                    "type": "noul",
                    "instructions": f"Does this passage answer the query? Passage: {text}",
                }
            }
            resp = post_systemone(self._cfg, state, questions, post=self._post)
            noul = resp.get("answers", {}).get("q_rel", {}).get("noul")
            tokens = resp.get("usage", {}).get("input_tokens", 0) or 0
            return c.id, noul, tokens

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            outcomes = list(ex.map(score_one, candidates))

        scores = {}
        missing = 0
        total_tokens = 0
        for cid, noul, tokens in outcomes:
            total_tokens += tokens
            if noul is None:
                missing += 1
                scores[cid] = 0.0
            else:
                scores[cid] = float(noul)

        ranked = sorted(candidates, key=lambda c: scores[c.id], reverse=True)
        return RerankResult(
            arm=self.arm,
            query_id=query_meta.get("query_id", ""),
            ranked_ids=[c.id for c in ranked],
            scores=[scores[c.id] for c in ranked],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=total_tokens,
            calls=len(candidates),
            cost_usd=total_tokens / 1e6 * self._cfg.jev_price_per_mtok,
            flags={"missing_noul": missing} if missing else {},
        )
