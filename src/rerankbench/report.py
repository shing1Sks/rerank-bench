"""Final report: per-arm quality/latency/cost tables + judge verdicts, as markdown."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from rerankbench.metrics import mrr, ndcg_at_k, p50, p95, recall_at_k

METRIC_KEYS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@dataclass
class Report:
    per_arm: dict = field(default_factory=dict)
    per_type: dict = field(default_factory=dict)
    judge_summary: dict = field(default_factory=dict)
    spot_check: dict = field(default_factory=dict)
    gold_metric_scope: str = ""
    tables: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        parts = ["## Per-arm results", ""]
        parts.append(self.tables["metrics"])
        parts.append("")
        parts.append(f"_Gold scope: {self.gold_metric_scope}_")
        if self.per_type:
            parts += ["", "## Per query type", "", self.tables["per_type"]]
        if self.judge_summary:
            parts += ["", "## Blind pairwise judging (Jev, position-swapped)", ""]
            parts.append(self.tables["judging"])
        if self.spot_check:
            parts += ["", "## Gemini Flash spot check", ""]
            parts.append(self.tables["spot_check"])
        return "\n".join(parts)


def build_report(run_dir, judge_summary: dict, spot_check: dict) -> Report:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = run["rows"]
    arms = list(dict.fromkeys(r["arm"] for r in rows))  # row order, deduped

    per_arm: dict[str, dict] = {}
    scorable_total = sum(1 for r in rows if r["gold"])
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        scorable = [r for r in arm_rows if r["gold"]]
        gold_sets = [set(r["gold"]) for r in scorable]
        per_arm[arm] = {
            "n": len(arm_rows),
            **{
                key: _mean(
                    recall_at_k(g, r["ranked_ids"], int(key.split("@")[1]))
                    for g, r in zip(gold_sets, scorable)
                )
                for key in ("recall@1", "recall@5", "recall@10")
            },
            "mrr": _mean(mrr(g, r["ranked_ids"], 10) for g, r in zip(gold_sets, scorable)),
            "ndcg@10": _mean(ndcg_at_k(g, r["ranked_ids"], 10) for g, r in zip(gold_sets, scorable)),
            "p50_latency_ms": p50([r["latency_ms"] for r in arm_rows]) if arm_rows else 0.0,
            "p95_latency_ms": p95([r["latency_ms"] for r in arm_rows]) if arm_rows else 0.0,
            "calls": sum(r.get("calls", 0) for r in arm_rows),
            "cost_usd": sum(r.get("cost_usd", 0.0) for r in arm_rows),
            "errors": sum(1 for r in arm_rows if r.get("flags", {}).get("error")),
        }
        n = len(arm_rows)
        per_arm[arm]["cost_per_1k_usd"] = per_arm[arm]["cost_usd"] / n * 1000 if n else 0.0

    n_rows = len(rows)
    scope = (
        f"quality metrics over {scorable_total} of {n_rows} rows "
        f"(negative queries have no gold and are excluded, never scored as zero)"
    )

    # per (arm, query_type) quality; negative rows carry no gold and stay out here too
    per_type: dict[tuple, dict] = {}
    type_keys = sorted({(r["arm"], r["query_type"]) for r in rows if r["gold"]})
    for arm, qtype in type_keys:
        t_rows = [
            r for r in rows if r["arm"] == arm and r["query_type"] == qtype and r["gold"]
        ]
        gold_sets = [set(r["gold"]) for r in t_rows]
        per_type[(arm, qtype)] = {
            "n": len(t_rows),
            **{
                key: _mean(
                    recall_at_k(g, r["ranked_ids"], int(key.split("@")[1]))
                    for g, r in zip(gold_sets, t_rows)
                )
                for key in ("recall@1", "recall@5", "recall@10")
            },
            "mrr": _mean(mrr(g, r["ranked_ids"], 10) for g, r in zip(gold_sets, t_rows)),
            "ndcg@10": _mean(ndcg_at_k(g, r["ranked_ids"], 10) for g, r in zip(gold_sets, t_rows)),
        }

    pt_header = "| arm | query type | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |"
    pt_sep = "|---|---|---|---|---|---|---|---|"
    pt_lines = [pt_header, pt_sep]
    for (arm, qtype), t in per_type.items():
        pt_lines.append(
            f"| {arm} | {qtype} | {t['n']} | {t['recall@1']:.3f} | {t['recall@5']:.3f} "
            f"| {t['recall@10']:.3f} | {t['mrr']:.3f} | {t['ndcg@10']:.3f} |"
        )

    header = (
        "| arm | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms | $/1k queries | errors |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    metric_lines = [header, sep]
    for arm in arms:
        a = per_arm[arm]
        metric_lines.append(
            f"| {arm} | {a['n']} | {a['recall@1']:.3f} | {a['recall@5']:.3f} | {a['recall@10']:.3f} "
            f"| {a['mrr']:.3f} | {a['ndcg@10']:.3f} | {a['p50_latency_ms']:.0f} | {a['p95_latency_ms']:.0f} "
            f"| {a['cost_per_1k_usd']:.3f} | {a['errors']} |"
        )

    judge_header = "| pairing | A wins | B wins | ties | flips | swap consistency |"
    judge_sep = "|---|---|---|---|---|---|"
    judge_lines = [judge_header, judge_sep]
    for pairing, s in judge_summary.items():
        judge_lines.append(
            f"| {pairing} | {s['wins']} | {s['losses']} | {s['ties']} | {s['flips']} | {s['swap_consistency']:.2f} |"
        )

    spot_header = "| pairing | kappa | n | flagged |"
    spot_sep = "|---|---|---|---|"
    spot_lines = [spot_header, spot_sep]
    for pairing, s in spot_check.items():
        spot_lines.append(f"| {pairing} | {s['kappa']:.2f} | {s['n']} | {s['flagged']} |")

    report = Report(
        per_arm=per_arm,
        per_type=per_type,
        judge_summary=judge_summary,
        spot_check=spot_check,
        gold_metric_scope=scope,
        tables={
            "metrics": "\n".join(metric_lines),
            "per_type": "\n".join(pt_lines),
            "judging": "\n".join(judge_lines),
            "spot_check": "\n".join(spot_lines),
        },
    )
    return report


def main(argv=None):
    """CLI: rebuild report.md from a completed run directory; no new API calls."""
    parser = argparse.ArgumentParser(description="build the markdown report for a run")
    parser.add_argument("--run", default="results/run1", help="run directory (holds run.json)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)

    def load(name):
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    report = build_report(run_dir, judge_summary=load("judge_summary.json"), spot_check=load("spotcheck_summary.json"))
    md = report.to_markdown()
    (run_dir / "report.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
