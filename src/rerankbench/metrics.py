"""Ranking metrics and judge-agreement stats.

All gold-conditioned metrics return None when gold is empty (a query with no
gold chunk is unscorable, never a zero); latency percentiles are nearest-rank.
"""
from __future__ import annotations

import math
from collections import Counter


def recall_at_k(gold: set, ranked: list, k: int) -> float | None:
    if not gold:
        return None
    hits = sum(1 for chunk_id in ranked[:k] if chunk_id in gold)
    return hits / len(gold)


def mrr(gold: set, ranked: list, k: int) -> float | None:
    if not gold:
        return None
    for i, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold: set, ranked: list, k: int = 10) -> float | None:
    if not gold:
        return None
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def p50(values) -> float:
    return _nearest_rank(values, 0.50)


def p95(values) -> float:
    return _nearest_rank(values, 0.95)


def _nearest_rank(values, q: float) -> float:
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of empty sequence")
    rank = max(1, math.ceil(q * len(xs)))
    return xs[rank - 1]


def win_tie_loss(verdicts: list) -> dict:
    counts = Counter(v for v in verdicts if v in ("A", "B", "TIE"))
    return {"A": counts["A"], "B": counts["B"], "TIE": counts["TIE"]}


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must have equal length")
    if not labels_a:
        raise ValueError("kappa of empty label lists")
    n = len(labels_a)
    classes = ("A", "B", "TIE")
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in classes)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
