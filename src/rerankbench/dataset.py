"""Benchmark dataset: schema, quotas, validation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

QUERY_TYPES = ("verbatim", "paraphrase", "reasoning", "negative", "structural", "graphical")

# Per-type minimums (spec section 6); total 175 sits inside the 150-250 range.
TYPE_QUOTAS = {
    "verbatim": 40,
    "paraphrase": 40,
    "reasoning": 30,
    "negative": 20,
    "structural": 20,
    "graphical": 25,
}


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    query_type: str
    gold_chunk_ids: list[str]
    source_doc: str
    notes: str


def load_queries(path: Path) -> list[Query]:
    with path.open("r", encoding="utf-8") as f:
        return [Query(**json.loads(line)) for line in f if line.strip()]


def validate_queries(queries: list[Query], chunk_ids: set[str]) -> list[str]:
    """All errors, one per line; empty list means the dataset is valid."""
    errs: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if q.query_type not in QUERY_TYPES:
            errs.append(f"{q.id}: bad query_type {q.query_type!r}")
        if q.query_type == "negative" and q.gold_chunk_ids:
            errs.append(f"{q.id}: negative query must have empty gold")
        if q.query_type != "negative" and not q.gold_chunk_ids:
            errs.append(f"{q.id}: non-negative query needs gold")
        for gid in q.gold_chunk_ids:
            if gid not in chunk_ids:
                errs.append(f"{q.id}: unknown gold chunk {gid}")
        if not 20 <= len(q.query) <= 400:
            errs.append(f"{q.id}: query length {len(q.query)} outside 20-400")
        if q.id in seen:
            errs.append(f"{q.id}: duplicate id")
        seen.add(q.id)
    return errs
