"""Shared test fakes for reranker/judge tests (no network anywhere)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FakeCfg:
    typesafe_url: str = "https://fake.typesafe/v1/systemone"
    jev_model: str = "jev-latest"
    typesafe_key: str = "fake-key"
    listwise_chunk_cap: int = 800
    jev_price_per_mtok: float = 0.042
    openrouter_rerank_url: str = "https://fake.openrouter/v1/rerank"
    openrouter_key: str = "fake-or-key"
    nemotron_slug: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    nemotron_slug_free: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
    gemini_model: str = "gemini-flash-latest"
    gemini_key: str = "fake-gem-key"
    judge_top_n: int = 5
    embed_model: str = "nomic-embed-text"
    top_k: int = 30
    data_dir: Path | None = None
    results_dir: Path | None = None


class CapturingPost:
    """Returns `reply` (dict) on 200; pops `statuses` first, raising JevError per code."""

    def __init__(self, reply, statuses=()):
        self.reply = reply
        self.statuses = list(statuses)
        self.calls = 0
        self.last_body = None
        self.last_headers = None

    def __call__(self, url, body, headers):
        from rerankbench.rerankers.jev import JevError

        self.calls += 1
        self.last_body = body
        self.last_headers = headers
        if self.statuses:
            code = self.statuses.pop(0)
            if code != 200:
                raise JevError(f"http {code}", status=code)
        return self.reply


class BodyRoutedPost:
    """Per-call reply from reply_fn(body) — deterministic under ThreadPoolExecutor."""

    def __init__(self, reply_fn):
        self.reply_fn = reply_fn
        self.calls = 0
        self.bodies = []

    def __call__(self, url, body, headers):
        self.calls += 1
        self.bodies.append(body)
        return self.reply_fn(body)
