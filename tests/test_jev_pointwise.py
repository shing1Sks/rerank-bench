from rerankbench.chunk import Chunk
from rerankbench.rerankers.jev import JevPointwise
from tests.fakes import BodyRoutedPost, FakeCfg


def chunks():
    return [Chunk("c1", "d", "s", "alpha", 1, 1), Chunk("c2", "d", "s", "beta", 1, 1)]


def test_pointwise_scores_each_candidate_and_ranks():
    def reply_fn(body):
        blob = str(body)
        noul = 0.9 if "alpha" in blob else 0.1
        return {"answers": {"q_rel": {"noul": noul}}, "usage": {"input_tokens": 105}}

    post = BodyRoutedPost(reply_fn)
    r = JevPointwise(cfg=FakeCfg(), post=post).rerank("q", chunks(), {"query_id": "q1"})
    assert r.arm == "jev_pointwise"
    assert r.ranked_ids == ["c1", "c2"]
    assert r.scores == [0.9, 0.1]
    assert r.calls == 2 and r.input_tokens == 210
    assert r.flags == {}


def test_pointwise_missing_noul_scores_zero_and_flags():
    def reply_fn(body):
        if "alpha" in str(body):
            return {"answers": {"q_rel": {"noul": 0.7}}, "usage": {"input_tokens": 10}}
        return {"answers": {}, "usage": {"input_tokens": 10}}

    r = JevPointwise(cfg=FakeCfg(), post=BodyRoutedPost(reply_fn)).rerank(
        "q", chunks(), {"query_id": "q1"}
    )
    assert r.ranked_ids == ["c1", "c2"]
    assert r.flags["missing_noul"] == 1
    assert r.scores == [0.7, 0.0]


def test_pointwise_truncates_long_passages():
    long_chunk = Chunk("c3", "d", "s", " ".join(["w"] * 5000), 2, 2)
    post = BodyRoutedPost(
        lambda body: {"answers": {"q_rel": {"noul": 0.5}}, "usage": {"input_tokens": 1}}
    )
    r = JevPointwise(cfg=FakeCfg(), post=post).rerank("q", [long_chunk], {"query_id": "q1"})
    question = post.bodies[0]["questions"]["q_rel"]
    assert question["type"] == "noul"
    chunk_text = question["instructions"].split("Passage:", 1)[1]
    assert len(chunk_text.split()) <= 800 * 3 // 4 + 2
    assert r.calls == 1
