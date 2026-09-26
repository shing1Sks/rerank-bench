import json

from rerankbench.report import build_report


def make_run_dir(tmp_path, rows):
    run_dir = tmp_path / "results" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"meta": {"tag": "run1"}, "rows": rows}), encoding="utf-8")
    return run_dir


def row(arm, qid, gold, ranked, latency=100.0, cost=0.0, qtype=None):
    return {
        "arm": arm,
        "query_id": qid,
        "query_type": qtype or ("verbatim" if gold else "negative"),
        "gold": gold,
        "ranked_ids": ranked,
        "latency_ms": latency,
        "cost_usd": cost,
        "input_tokens": 0,
        "calls": 1,
        "flags": {},
        "scores": None,
        "model_version": "m",
    }


def test_build_report_computes_per_arm_metrics_with_gold_scope(tmp_path):
    rows = [
        row("a", "q1", ["g1"], ["g1", "x", "y"], latency=10.0),
        row("a", "q2", ["g1", "g2"], ["x", "g1", "g2"], latency=30.0),
        row("a", "n1", [], ["x", "y", "z"], latency=20.0),  # negative: excluded
        row("b", "q1", ["g1"], ["x", "y", "g1"], latency=50.0, cost=0.01),
        row("b", "q2", ["g1", "g2"], ["g1", "g2", "x"], latency=70.0, cost=0.03),
    ]
    report = build_report(make_run_dir(tmp_path, rows), judge_summary={}, spot_check={})
    a = report.per_arm["a"]
    b = report.per_arm["b"]
    assert a["recall@1"] == (1.0 + 0.0) / 2
    assert a["recall@5"] == 1.0
    assert a["mrr"] == (1.0 + 0.5) / 2
    assert b["recall@1"] == (0.0 + 0.5) / 2  # q2 has two golds, one at rank 1
    assert report.per_arm["a"]["p50_latency_ms"] == 20.0
    assert report.per_arm["a"]["p95_latency_ms"] == 30.0
    assert abs(b["cost_per_1k_usd"] - 20.0) < 1e-9  # 0.04 per 2 queries -> 20 per 1k
    assert "4 of 5" in report.gold_metric_scope  # one negative row excluded


def test_report_to_markdown_includes_quality_and_spot_check_tables(tmp_path):
    rows = [row("a", "q1", ["g1"], ["g1"])]
    judge_summary = {"nemotron_vl>jev_listwise": {"wins": 7, "losses": 2, "ties": 1, "flips": 1, "swap_consistency": 0.9}}
    spot_check = {"nemotron_vl>jev_listwise": {"kappa": 0.62, "n": 50, "flagged": 2}}
    report = build_report(make_run_dir(tmp_path, rows), judge_summary=judge_summary, spot_check=spot_check)
    md = report.to_markdown()
    assert "nemotron_vl>jev_listwise" in md
    assert "7" in md and "0.62" in md
    assert "recall@1" in md
    assert "4 of 5" in md or "gold" in md.lower()


def test_report_per_type_metrics_cover_types_and_exclude_negatives(tmp_path):
    rows = [
        row("a", "q1", ["g1"], ["g1", "x"], qtype="verbatim"),
        row("a", "q2", ["g1"], ["x", "g1"], qtype="reasoning"),
        row("a", "g1", ["g1"], ["g1", "x"], qtype="graphical"),
        row("a", "n1", [], ["x", "y"], qtype="negative"),
        row("b", "q1", ["g1"], ["x", "g1"], qtype="verbatim"),
    ]
    report = build_report(make_run_dir(tmp_path, rows), judge_summary={}, spot_check={})
    pt = report.per_type
    assert pt[("a", "verbatim")]["recall@1"] == 1.0
    assert pt[("a", "reasoning")]["recall@1"] == 0.0
    assert pt[("a", "graphical")]["recall@1"] == 1.0
    assert pt[("b", "verbatim")]["recall@1"] == 0.0
    assert all((arm, qt) not in pt for arm in ("a", "b") for qt in ("negative",))
    assert pt[("a", "verbatim")]["n"] == 1
    md = report.to_markdown()
    assert "Per query type" in md
    assert "graphical" in md and "reasoning" in md


def test_report_cli_writes_report_md(tmp_path):
    import json as json_mod

    import rerankbench.report as report_mod

    run_dir = tmp_path / "results" / "run1"
    run_dir.mkdir(parents=True)
    rows = [row("a", "q1", ["g1"], ["g1"])]
    (run_dir / "run.json").write_text(json_mod.dumps({"meta": {"tag": "run1"}, "rows": rows}), encoding="utf-8")

    report_mod.main(["--run", str(run_dir)])

    out = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Per-arm results" in out and "recall@1" in out
