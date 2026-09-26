"""LLM-as-judge: blind pairwise comparisons between arms, plus the Gemini spot check.

Jev judges both orders of every pair and reconciles: agreement gives the verdict,
disagreement degrades to TIE with flipped=True. Gemini judges a single order per
pair for the spot check; unknown answers become flagged TIEs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

import httpx

from rerankbench.config import load_config
from rerankbench.metrics import cohens_kappa
from rerankbench.rerankers.jev import JevError, _default_post, post_systemone

PAIRINGS = [
    ("jev_listwise", "jev_pointwise"),
    ("nemotron_vl", "jev_listwise"),
    ("nemotron_vl", "jev_pointwise"),
    ("nemotron_vl", "embed_only"),
    ("jev_listwise", "embed_only"),
    ("jev_pointwise", "embed_only"),
]


def format_ranking_block(ranked_ids: list, id_to_text: dict, top_n: int) -> str:
    return "\n".join(
        f"[{i}] {cid}: {id_to_text.get(cid, '(missing text)')}"
        for i, cid in enumerate(ranked_ids[:top_n], start=1)
    )


def _judge_state(query: str, rank_a: list, rank_b: list, id_to_text: dict, top_n: int) -> str:
    return (
        f"Query: {query}\n\n"
        f"Ranking A:\n{format_ranking_block(rank_a, id_to_text, top_n)}\n\n"
        f"Ranking B:\n{format_ranking_block(rank_b, id_to_text, top_n)}"
    )


_JUDGE_QUESTIONS = {
    "q_judge": {
        "type": "choice",
        "instructions": (
            "Which ranking better answers the query? "
            "Compare the two ranked passage lists and pick one."
        ),
        "criteria": {
            "A": "Ranking A is better",
            "B": "Ranking B is better",
            "TIE": "Roughly equal quality",
        },
    }
}


def judge_pair_jev(cfg, query, rank_a, rank_b, id_to_text, post) -> str:
    state = _judge_state(query, rank_a, rank_b, id_to_text, cfg.judge_top_n)
    resp = post_systemone(cfg, state, _JUDGE_QUESTIONS, post=post)
    try:
        choice = resp["answers"]["q_judge"]["choice"]
    except (KeyError, TypeError) as e:
        raise JevError(f"judge response missing choice: {e}") from e
    if choice not in ("A", "B", "TIE"):
        raise JevError(f"judge returned invalid choice: {choice!r}")
    return choice


def judge_orders_jev(cfg, query, rank_a, rank_b, id_to_text, post) -> tuple[str, bool]:
    v1 = judge_pair_jev(cfg, query, rank_a, rank_b, id_to_text, post)
    v2 = judge_pair_jev(cfg, query, rank_b, rank_a, id_to_text, post)
    inverted = {"A": "B", "B": "A", "TIE": "TIE"}[v2]
    if v1 == inverted:
        return v1, False
    return "TIE", True


def judge_pair_gemini(cfg, query, rank_a, rank_b, id_to_text, post) -> tuple[str, bool]:
    prompt = (
        f"{_judge_state(query, rank_a, rank_b, id_to_text, cfg.judge_top_n)}\n\n"
        "Which ranking better answers the query? Reply with exactly one word: A, B, or TIE."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.gemini_model}:generateContent?key={cfg.gemini_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = post(url, body)
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, TypeError, IndexError) as e:
        return "TIE", True
    first_line = text.strip().splitlines()[0].strip().upper() if text.strip() else ""
    if first_line in ("A", "B", "TIE"):
        return first_line, False
    return "TIE", True


def _load_run(run_path, data_dir):
    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    id_to_text = {}
    with (data_dir / "chunks" / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            id_to_text[c["id"]] = c["text"]
    queries = {}
    with (data_dir / "dataset" / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["id"]] = q["query"]
    rankings = {(r["arm"], r["query_id"]): r["ranked_ids"] for r in run["rows"]}
    return rankings, id_to_text, queries


def _query_subset(queries: dict, subset) -> list:
    qids = sorted(queries)
    if subset is None or subset >= len(qids):
        return qids
    return sorted(random.Random(13).sample(qids, subset))


def run_judge_pass(cfg, run_path, judge: str = "jev", post: Callable | None = None, subset=None) -> dict:
    """Blind pairwise judging over all pairings; writes verdicts.jsonl + judge_summary.json."""
    rankings, id_to_text, queries = _load_run(run_path, cfg.data_dir)
    qids = _query_subset(queries, subset)
    if judge == "jev" and post is None:
        post = _default_post
    summary = {}
    with (run_path / "verdicts.jsonl").open("w", encoding="utf-8") as out:
        for arm_a, arm_b in PAIRINGS:
            verdicts = []
            flips = 0
            judgeable = [q for q in qids if (arm_a, q) in rankings and (arm_b, q) in rankings]
            for qid in judgeable:
                rank_a = rankings[(arm_a, qid)]
                rank_b = rankings[(arm_b, qid)]
                if judge == "jev":
                    verdict, flipped = judge_orders_jev(
                        cfg, queries[qid], rank_a, rank_b, id_to_text, post
                    )
                else:
                    raise ValueError(f"unsupported judge: {judge} (use spot_check_gemini)")
                flips += int(flipped)
                verdicts.append(verdict)
                out.write(
                    json.dumps(
                        {"pairing": f"{arm_a}>{arm_b}", "query_id": qid, "verdict": verdict, "flipped": flipped}
                    )
                    + "\n"
                )
            wins = verdicts.count("A")
            losses = verdicts.count("B")
            ties = verdicts.count("TIE")
            summary[f"{arm_a}>{arm_b}"] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "flips": flips,
                "swap_consistency": 1.0 if not verdicts else 1.0 - flips / len(verdicts),
            }
    (run_path / "judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _default_gemini_post(url: str, body: dict) -> dict:
    resp = httpx.post(url, json=body, timeout=120.0)
    if resp.status_code != 200:
        raise JevError(f"http {resp.status_code}: {resp.text[:500]}", status=resp.status_code)
    return resp.json()


def spot_check_gemini(cfg, run_path, post: Callable | None = None, subset=None) -> dict:
    """Gemini re-judges one order per pair; kappa against the stored jev verdicts."""
    post = post or _default_gemini_post
    rankings, id_to_text, queries = _load_run(run_path, cfg.data_dir)
    qids = _query_subset(queries, subset)
    jev_verdicts = {}
    with (run_path / "verdicts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            jev_verdicts[(row["pairing"], row["query_id"])] = row["verdict"]
    summary = {}
    with (run_path / "spotcheck.jsonl").open("w", encoding="utf-8") as out:
        for arm_a, arm_b in PAIRINGS:
            pairing = f"{arm_a}>{arm_b}"
            gemini_labels, jev_labels, flagged = [], [], 0
            judgeable = [q for q in qids if (arm_a, q) in rankings and (arm_b, q) in rankings]
            for qid in judgeable:
                try:
                    verdict, is_flagged = judge_pair_gemini(
                        cfg, queries[qid], rankings[(arm_a, qid)], rankings[(arm_b, qid)], id_to_text, post
                    )
                except JevError:
                    verdict, is_flagged = "TIE", True
                flagged += int(is_flagged)
                gemini_labels.append(verdict)
                jev_labels.append(jev_verdicts[(pairing, qid)])
                out.write(
                    json.dumps(
                        {
                            "pairing": pairing,
                            "query_id": qid,
                            "gemini": verdict,
                            "jev": jev_verdicts[(pairing, qid)],
                            "flagged": bool(is_flagged),
                        }
                    )
                    + "\n"
                )
            summary[pairing] = {
                "kappa": cohens_kappa(gemini_labels, jev_labels),
                "n": len(gemini_labels),
                "flagged": flagged,
            }
    (run_path / "spotcheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    """CLI: --judge jev runs the full position-swapped pass; --judge gemini the spot check."""
    parser = argparse.ArgumentParser(description="judge a completed run's arm pairings")
    parser.add_argument("--run", required=True, help="run directory (holds run.json)")
    parser.add_argument("--judge", choices=("jev", "gemini"), default="jev")
    parser.add_argument("--subset", type=int, default=None, help="judge only the first N sampled queries")
    args = parser.parse_args(argv)

    cfg = load_config()
    run_path = Path(args.run)
    if args.judge == "jev":
        summary = run_judge_pass(cfg, run_path, judge="jev", subset=args.subset)
        for pairing, s in summary.items():
            print(
                f"{pairing:28s} A={s['wins']:3d} B={s['losses']:3d} TIE={s['ties']:3d} "
                f"flips={s['flips']} consistency={s['swap_consistency']:.2f}"
            )
    else:
        summary = spot_check_gemini(cfg, run_path, subset=args.subset)
        for pairing, s in summary.items():
            print(f"{pairing:28s} kappa={s['kappa']:.2f} n={s['n']} flagged={s['flagged']}")


if __name__ == "__main__":
    main()
