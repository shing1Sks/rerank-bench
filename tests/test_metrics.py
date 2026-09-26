import pytest

from rerankbench.metrics import cohens_kappa, mrr, ndcg_at_k, p50, p95, recall_at_k, win_tie_loss


def test_recall_at_k():
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], 2) == 0.5
    assert recall_at_k({"a", "b"}, ["b", "x", "a"], 10) == 1.0
    assert recall_at_k({"a"}, ["x", "y"], 5) == 0.0
    assert recall_at_k(set(), ["a"], 5) is None


def test_mrr():
    assert mrr({"a"}, ["x", "a", "b"], 10) == 0.5
    assert mrr({"a", "b"}, ["x", "b", "a"], 10) == 0.5  # first hit wins
    assert mrr({"a"}, ["x", "y"], 10) == 0.0
    assert mrr(set(), ["a"], 10) is None


def test_ndcg_at_k_binary_gains():
    # gold a at rank 1: DCG = 1/log2(2) = 1; IDCG with 1 gold = 1
    assert ndcg_at_k({"a"}, ["a", "x", "b"], 10) == pytest.approx(1.0)
    # gold a at rank 2: DCG = 1/log2(3); IDCG = 1
    assert ndcg_at_k({"a"}, ["x", "a", "b"], 10) == pytest.approx(1 / _log2(3))
    # two golds both found: perfect
    assert ndcg_at_k({"a", "b"}, ["a", "b", "x"], 10) == pytest.approx(1.0)
    assert ndcg_at_k(set(), ["a"], 10) is None


def _log2(x):
    import math

    return math.log2(x)


def test_percentiles_nearest_rank():
    assert p50(range(1, 21)) == 10
    assert p95(range(1, 21)) == 19
    assert p50([5]) == 5


def test_win_tie_loss():
    assert win_tie_loss(["A", "B", "TIE", "A"]) == {"A": 2, "B": 1, "TIE": 1}
    assert win_tie_loss([]) == {"A": 0, "B": 0, "TIE": 0}


def test_cohens_kappa():
    labels = ["A", "B", "TIE", "A"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)
    assert cohens_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    with pytest.raises(ValueError):
        cohens_kappa(["A"], ["A", "B"])
