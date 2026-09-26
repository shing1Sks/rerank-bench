"""Embed-only arm: the retrieval baseline, scores candidates by cosine similarity.

The run script wires embed_fn to a closure over the built index (query embedding
+ cosine_top_k against the candidate ids); the arm itself just consumes a
text -> score mapping so it stays interchangeable with the hosted arms.
"""
from __future__ import annotations

import time
from typing import Callable

from rerankbench.rerankers import RerankResult


class EmbedOnlyReranker:
    arm = "embed_only"

    def __init__(self, embed_fn: Callable[[list[str]], dict[str, float]]):
        self._embed_fn = embed_fn

    def rerank(self, query: str, candidates: list, query_meta: dict) -> RerankResult:
        t0 = time.perf_counter()
        scores_by_text = self._embed_fn([c.text for c in candidates])
        missing = sum(1 for c in candidates if c.text not in scores_by_text)
        scored = sorted(
            candidates,
            key=lambda c: scores_by_text.get(c.text, float("-inf")),
            reverse=True,  # stable: ties keep candidate order
        )
        return RerankResult(
            arm=self.arm,
            query_id=query_meta.get("query_id", ""),
            ranked_ids=[c.id for c in scored],
            scores=[scores_by_text.get(c.text, float("-inf")) for c in scored],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            flags={"missing_scores": missing} if missing else {},
        )
