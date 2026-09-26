import json
import types

import pytest

import rerankbench.judges
from rerankbench.judges import (
    format_ranking_block,
    judge_orders_jev,
    judge_pair_gemini,
    judge_pair_jev,
    run_judge_pass,
    spot_check_gemini,
)
from rerankbench.rerankers.jev import JevError
from tests.fakes import BodyRoutedPost, CapturingPost, FakeCfg


def id_to_text():
    return {"x1": "alpha one", "x2": "beta two", "x3": "gamma three"}


def test_format_ranking_block_numbers_and_truncates():
    block = format_ranking_block(["x1", "x2"], id_to_text(), top_n=2)
    assert block == "[1] x1: alpha one\n[2] x2: beta two"
    assert format_ranking_block(["x1", "x2"], id_to_text(), top_n=1) == "[1] x1: alpha one"


def test_judge_pair_jev_blind_state_and_verdict():
    post = CapturingPost({"answers": {"q_judge": {"choice": "A"}}, "usage": {"input_tokens": 5}})
    verdict = judge_pair_jev(FakeCfg(), "what is alpha?", ["x1", "x2"], ["x2", "x1"], id_to_text(), post)
    assert verdict == "A"
    state = post.last_body["state"]
    assert "what is alpha?" in state
    assert "Ranking A" in state and "Ranking B" in state
    assert "jev" not in state.lower()
    criteria = post.last_body["questions"]["q_judge"]["criteria"]
    assert set(criteria) == {"A", "B", "TIE"}


def test_judge_orders_jev_consistent_and_flipped():
    calls = []

    def consistent(body):
        calls.append(1)
        choice = "A" if len(calls) % 2 == 1 else "B"  # x-rank always better
        return {"answers": {"q_judge": {"choice": choice}}, "usage": {"input_tokens": 5}}

    verdict, flipped = judge_orders_jev(
        FakeCfg(), "q", ["x1", "x2"], ["x2", "x1"], id_to_text(), BodyRoutedPost(consistent)
    )
    assert (verdict, flipped) == ("A", False)

    def contradictory(body):
        calls.append(1)
        return {"answers": {"q_judge": {"choice": "A"}}, "usage": {"input_tokens": 5}}

    verdict, flipped = judge_orders_jev(
        FakeCfg(), "q", ["x1", "x2"], ["x2", "x1"], id_to_text(), BodyRoutedPost(contradictory)
    )
    assert (verdict, flipped) == ("TIE", True)


def test_judge_pair_gemini_parses_first_line_and_flags_unknown():
    post = lambda url, body: {"candidates": [{"content": {"parts": [{"text": "A"}]}}]}
    assert judge_pair_gemini(FakeCfg(), "q", ["x1"], ["x2"], id_to_text(), post) == ("A", False)
    post = lambda url, body: {"candidates": [{"content": {"parts": [{"text": "TIE"}]}}]}
    assert judge_pair_gemini(FakeCfg(), "q", ["x1"], ["x2"], id_to_text(), post) == ("TIE", False)
    post = lambda url, body: {"candidates": [{"content": {"parts": [{"text": "C nonsense"}]}}]}
    assert judge_pair_gemini(FakeCfg(), "q", ["x1"], ["x2"], id_to_text(), post) == ("TIE", True)


PAIRINGS = [
    ("jev_listwise", "jev_pointwise"),
    ("nemotron_vl", "jev_listwise"),
    ("nemotron_vl", "jev_pointwise"),
    ("nemotron_vl", "embed_only"),
    ("jev_listwise", "embed_only"),
    ("jev_pointwise", "embed_only"),
]


def make_run(tmp_path, arms=("jev_listwise", "jev_pointwise", "nemotron_vl", "embed_only")):
    data = tmp_path / "data"
    (data / "chunks").mkdir(parents=True)
    (data / "dataset").mkdir()
    with (data / "chunks" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for cid, text in id_to_text().items():
            f.write(json.dumps({"id": cid, "text": text}) + "\n")
    with (data / "dataset" / "queries.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "q1", "query": "what is alpha?"}) + "\n")
        f.write(json.dumps({"id": "q2", "query": "what is beta?"}) + "\n")
        f.write(json.dumps({"id": "q3", "query": "what is gamma?"}) + "\n")  # no run rows
    rows = [
        {"arm": arm, "query_id": qid, "ranked_ids": ["x1", "x2", "x3"]}
        for arm in arms
        for qid in ("q1", "q2")
    ]
    run_dir = tmp_path / "results" / "smoke"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"meta": {}, "rows": rows}), encoding="utf-8")
    return run_dir


def test_run_judge_pass_default_post_parses_json(monkeypatch, tmp_path):
    run_dir = make_run(tmp_path, arms=("jev_listwise", "embed_only"))
    calls = []

    class FakeResp:
        status_code = 200

        def json(self):
            calls.append(1)
            choice = "A" if len(calls) % 2 == 1 else "B"
            return {"answers": {"q_judge": {"choice": choice}}, "usage": {"input_tokens": 5}}

    monkeypatch.setattr("rerankbench.rerankers.jev.httpx.post", lambda *a, **k: FakeResp())
    cfg = FakeCfg(data_dir=tmp_path / "data", results_dir=tmp_path / "results")
    summary = run_judge_pass(cfg, run_dir, judge="jev", post=None)
    assert len(calls) == 4  # 1 pairing x 2 judgeable queries x 2 orders
    assert summary["jev_listwise>embed_only"]["wins"] == 2


def test_run_judge_pass_jev_writes_verdicts_and_summary(tmp_path):
    run_dir = make_run(tmp_path)
    calls = []

    def routed(body):
        calls.append(1)
        choice = "A" if len(calls) % 2 == 1 else "B"  # consistent: A-rank always better
        return {"answers": {"q_judge": {"choice": choice}}, "usage": {"input_tokens": 5}}

    cfg = FakeCfg(data_dir=tmp_path / "data")
    summary = run_judge_pass(cfg, run_dir, judge="jev", post=BodyRoutedPost(routed))
    rows = [json.loads(l) for l in (run_dir / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12  # 6 pairings x 2 queries
    assert rows[0]["pairing"] == "jev_listwise>embed_only" or rows[0]["pairing"].count(">") == 1
    assert all(r["verdict"] in ("A", "B", "TIE") and isinstance(r["flipped"], bool) for r in rows)
    assert summary["jev_listwise>embed_only"]["wins"] == 2
    assert summary["jev_listwise>embed_only"]["flips"] == 0
    assert summary["jev_listwise>embed_only"]["swap_consistency"] == 1.0
    assert len(summary) == 6


def test_spot_check_gemini_kappa_against_jev(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "verdicts.jsonl").write_text(
        "\n".join(
            json.dumps({"pairing": f"{a}>{b}", "query_id": qid, "verdict": "A", "flipped": False})
            for a, b in PAIRINGS
            for qid in ("q1", "q2")
        ),
        encoding="utf-8",
    )
    cfg = FakeCfg(data_dir=tmp_path / "data")
    post = lambda url, body: {"candidates": [{"content": {"parts": [{"text": "A"}]}}]}
    summary = spot_check_gemini(cfg, run_dir, post=post, subset=50)
    assert len(summary) == 6
    assert summary["jev_listwise>embed_only"]["kappa"] == 1.0
    assert summary["jev_listwise>embed_only"]["n"] == 2
    spot = [json.loads(l) for l in (run_dir / "spotcheck.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(spot) == 12
    assert all(r["gemini"] == "A" and r["flagged"] is False for r in spot)


def test_spot_check_gemini_default_post_parses_json(monkeypatch, tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "verdicts.jsonl").write_text(
        "\n".join(
            json.dumps({"pairing": f"{a}>{b}", "query_id": qid, "verdict": "A", "flipped": False})
            for a, b in PAIRINGS
            for qid in ("q1", "q2")
        ),
        encoding="utf-8",
    )
    calls = []

    class FakeResp:
        status_code = 200

        def json(self):
            calls.append(1)
            return {"candidates": [{"content": {"parts": [{"text": "A"}]}}]}

    monkeypatch.setattr(
        rerankbench.judges,
        "httpx",
        types.SimpleNamespace(post=lambda *a, **k: FakeResp()),
        raising=False,
    )
    cfg = FakeCfg(data_dir=tmp_path / "data")
    summary = spot_check_gemini(cfg, run_dir, subset=50)
    assert len(calls) == 12  # 6 pairings x 2 judgeable queries, single order each
    assert summary["jev_listwise>embed_only"]["kappa"] == 1.0
    assert summary["jev_listwise>embed_only"]["flagged"] == 0


def test_spot_check_gemini_flags_transport_failures(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "verdicts.jsonl").write_text(
        "\n".join(
            json.dumps({"pairing": f"{a}>{b}", "query_id": qid, "verdict": "A", "flipped": False})
            for a, b in PAIRINGS
            for qid in ("q1", "q2")
        ),
        encoding="utf-8",
    )

    def crash(url, body):
        raise JevError("http 429", status=429)

    cfg = FakeCfg(data_dir=tmp_path / "data")
    summary = spot_check_gemini(cfg, run_dir, post=crash, subset=50)
    assert all(s["flagged"] == s["n"] == 2 for s in summary.values())
    spot = [json.loads(l) for l in (run_dir / "spotcheck.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["gemini"] == "TIE" and r["flagged"] is True for r in spot)


def test_judges_cli_dispatches_jev_pass_and_spot_check(tmp_path, monkeypatch):
    import rerankbench.judges as judges_mod

    seen = {}
    monkeypatch.setattr(judges_mod, "load_config", lambda: FakeCfg())

    def fake_pass(cfg, run_path, judge="jev", post=None, subset=None):
        seen["pass"] = (str(run_path), judge, subset)
        return {"x>y": {"wins": 1, "losses": 0, "ties": 0, "flips": 0, "swap_consistency": 1.0}}

    def fake_spot(cfg, run_path, post=None, subset=None):
        seen["spot"] = (str(run_path), subset)
        return {"x>y": {"kappa": 0.5, "n": 1, "flagged": 0}}

    monkeypatch.setattr(judges_mod, "run_judge_pass", fake_pass)
    monkeypatch.setattr(judges_mod, "spot_check_gemini", fake_spot)

    run_dir = tmp_path / "results" / "run1"
    judges_mod.main(["--run", str(run_dir), "--judge", "jev", "--subset", "10"])
    assert seen["pass"] == (str(run_dir), "jev", 10)

    judges_mod.main(["--run", str(run_dir), "--judge", "gemini"])
    assert seen["spot"] == (str(run_dir), None)
