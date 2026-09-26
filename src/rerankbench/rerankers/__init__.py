"""Reranker arms: shared result type + protocol, one module per arm."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RerankResult:
    arm: str
    query_id: str
    ranked_ids: list[str]
    scores: list[float] | None
    latency_ms: float
    input_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    flags: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class Reranker(Protocol):
    arm: str

    def rerank(self, query: str, candidates: list, query_meta: dict) -> RerankResult: ...
