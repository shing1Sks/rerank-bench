from rerankbench.chunk import Chunk
from rerankbench.rerankers import RerankResult
from rerankbench.rerankers.embed_only import EmbedOnlyReranker


def chunks():
    return [Chunk(f"c{i}", "d", "s", f"t{i}", 1, 1) for i in range(5)]


def test_embed_only_ranks_by_embed_fn_scores():
    # embed order: c3 best, then c0, c4, c1, c2
    emb = lambda texts: {"t3": 0.9, "t0": 0.8, "t4": 0.7, "t1": 0.6, "t2": 0.5}
    r = EmbedOnlyReranker(embed_fn=emb).rerank("q", chunks(), {"query_id": "q1"})
    assert r.arm == "embed_only"
    assert r.ranked_ids == ["c3", "c0", "c4", "c1", "c2"]
    assert r.scores == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert r.calls == 0 and r.cost_usd == 0.0 and r.input_tokens == 0
    assert r.query_id == "q1"
    assert isinstance(r, RerankResult)


def test_embed_only_stable_on_ties_preserving_candidate_order():
    emb = lambda texts: {"t0": 0.5, "t1": 0.5, "t2": 0.5, "t3": 0.5, "t4": 0.5}
    r = EmbedOnlyReranker(embed_fn=emb).rerank("q", chunks(), {"query_id": "q1"})
    assert r.ranked_ids == ["c0", "c1", "c2", "c3", "c4"]


def test_embed_only_ignores_unknown_texts_and_flags_missing():
    emb = lambda texts: {"t0": 0.5}
    r = EmbedOnlyReranker(embed_fn=emb).rerank("q", chunks(), {"query_id": "q1"})
    assert r.flags.get("missing_scores") == 4
    assert r.ranked_ids[0] == "c0"
    assert set(r.ranked_ids) == {"c0", "c1", "c2", "c3", "c4"}
