import pytest

from rerankbench.chunk import Chunk
from rerankbench.rerankers.jev import JevError, JevListwise, post_systemone, truncate_for_listwise
from tests.fakes import CapturingPost, FakeCfg


def chunks():
    return [Chunk("c1", "d", "s", "alpha", 1, 1), Chunk("c2", "d", "s", "beta", 1, 1)]


def reply(c1=0.2, c2=0.8, tokens=1000):
    return {
        "answers": {"q_rank": {"choice": "c2", "probabilities": {"c1": c1, "c2": c2}}},
        "usage": {"input_tokens": tokens, "output_tokens": 0},
    }


def test_listwise_ranks_by_probabilities():
    post = CapturingPost(reply())
    r = JevListwise(cfg=FakeCfg(), post=post).rerank("what?", chunks(), {"query_id": "q1"})
    assert r.ranked_ids == ["c2", "c1"]
    assert r.scores == [0.8, 0.2]
    assert r.calls == 1 and r.input_tokens == 1000
    assert abs(r.cost_usd - 1000 / 1e6 * 0.042) < 1e-12
    assert r.flags == {}
    q = post.last_body["questions"]["q_rank"]
    assert q["criteria"] == {"c1": "alpha", "c2": "beta"}
    assert post.last_body["model"] == "jev-latest"
    assert "what?" in post.last_body["state"]
    assert post.last_headers["Authorization"] == "Bearer fake-key"


def test_listwise_truncates_long_chunks_and_flags():
    long_chunk = Chunk("c3", "d", "s", " ".join(["w"] * 5000), 2, 2)
    post = CapturingPost(
        {"answers": {"q_rank": {"probabilities": {"c3": 1.0}}}, "usage": {"input_tokens": 10, "output_tokens": 0}}
    )
    r = JevListwise(cfg=FakeCfg(), post=post).rerank("q", [long_chunk], {"query_id": "q2"})
    assert r.flags["truncated"] is True
    body_text = post.last_body["questions"]["q_rank"]["criteria"]["c3"]
    assert len(body_text.split()) <= 800 * 3 // 4 + 2


def test_listwise_missing_probability_raises():
    post = CapturingPost({"answers": {"q_rank": {"probabilities": {"c1": 0.5}}}, "usage": {"input_tokens": 1, "output_tokens": 0}})
    with pytest.raises(JevError):
        JevListwise(cfg=FakeCfg(), post=post).rerank("q", chunks(), {"query_id": "q1"})


def test_post_systemone_retries_on_429_then_succeeds():
    post = CapturingPost({"answers": {}, "usage": {}}, statuses=[429, 200])
    out = post_systemone(FakeCfg(), "state", {}, post=post, sleep=lambda s: None)
    assert post.calls == 2
    assert out == {"answers": {}, "usage": {}}


def test_post_systemone_fails_fast_on_500():
    post = CapturingPost({"answers": {}, "usage": {}}, statuses=[500, 200])
    with pytest.raises(JevError):
        post_systemone(FakeCfg(), "state", {}, post=post, attempts=5, sleep=lambda s: None)
    assert post.calls == 1  # non-429/529 never retried


def test_post_systemone_gives_up_after_attempts_on_persistent_429():
    post = CapturingPost({"answers": {}, "usage": {}}, statuses=[429] * 6)
    with pytest.raises(JevError):
        post_systemone(FakeCfg(), "state", {}, post=post, attempts=5, sleep=lambda s: None)
    assert post.calls == 5


def test_truncate_keeps_word_boundary():
    out = truncate_for_listwise(" ".join(["word"] * 100), cap_tokens=10)
    assert len(out.split()) <= 8  # 10 * 3/4 words + ellipsis marker
    assert out.split()[0] == "word"


def test_default_post_error_includes_response_body(monkeypatch):
    class FakeResp:
        status_code = 400
        text = '{"error":"criteria too large"}'

        def json(self):
            return {}

    monkeypatch.setattr("rerankbench.rerankers.jev.httpx.post", lambda *a, **k: FakeResp())
    with pytest.raises(JevError) as excinfo:
        post_systemone(FakeCfg(), "state", {}, post=None)
    assert "criteria too large" in str(excinfo.value)
