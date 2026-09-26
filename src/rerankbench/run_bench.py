"""Benchmark runner: identical top-K candidates per query, per-arm rerank, disk cache."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rerankbench.chunk import load_chunks
from rerankbench.dataset import load_queries
from rerankbench.embed import OllamaEmbedder, cosine_top_k, load_index
from rerankbench.rerankers import RerankResult
from rerankbench.rerankers.embed_only import EmbedOnlyReranker
from rerankbench.rerankers.jev import JevListwise, JevPointwise
from rerankbench.rerankers.nemotron import (
    NemotronReranker,
    default_openrouter_post,
    page_image_data_uri,
    probe_slugs,
)

ALL_ARMS = ("embed_only", "nemotron_vl", "jev_listwise", "jev_pointwise")


def make_embed_fn(candidates: list, score_by_cid: dict):
    scores_by_text = {c.text: score_by_cid[c.id] for c in candidates}
    return lambda texts: scores_by_text


def run_bench(
    cfg,
    dataset_path,
    arms: list[str],
    tag: str,
    limit: int | None = None,
    embedder=None,
    posts: dict | None = None,
) -> Path:
    posts = posts or {}
    embedder = embedder or OllamaEmbedder(cfg)
    run_dir = cfg.results_dir / tag
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(dataset_path)
    if limit:
        queries = queries[:limit]
    chunks_by_id = {c.id: c for c in load_chunks(cfg.data_dir / "chunks" / "chunks.jsonl")}
    index_ids, matrix = load_index(cfg)
    typesafe_post = posts.get("typesafe")
    openrouter_post = posts.get("openrouter") or default_openrouter_post
    current = {"query_type": ""}

    def image_for(c):
        if current["query_type"] != "graphical":
            return None
        return page_image_data_uri(cfg, c.doc_id, c.page_start)

    versions: dict[str, str] = {}
    arm_objs: dict[str, object] = {}
    for arm in arms:
        if arm == "embed_only":
            versions[arm] = cfg.embed_model
        elif arm == "jev_listwise":
            arm_objs[arm] = JevListwise(cfg, typesafe_post)
            versions[arm] = cfg.jev_model
        elif arm == "jev_pointwise":
            arm_objs[arm] = JevPointwise(cfg, typesafe_post)
            versions[arm] = cfg.jev_model
        elif arm == "nemotron_vl":
            probed = probe_slugs(cfg, openrouter_post)
            if probed is None:
                print("warning: nemotron unreachable via base and :free slugs; its rows will degrade to candidate order")
            arm_objs[arm] = NemotronReranker(cfg, openrouter_post, image_for=image_for, slug=probed)
            versions[arm] = probed or f"{cfg.nemotron_slug} (unreachable)"
        else:
            raise ValueError(f"unknown arm: {arm}")

    rows: list[dict] = []
    for q in queries:
        current["query_type"] = q.query_type
        qv = embedder.embed([q.query])[0]
        top = cosine_top_k(qv, index_ids, matrix, cfg.top_k)
        score_by_cid = dict(top)
        candidates = [chunks_by_id[cid] for cid, _ in top]
        for arm in arms:
            if arm == "embed_only":
                reranker = EmbedOnlyReranker(make_embed_fn(candidates, score_by_cid))
            else:
                reranker = arm_objs[arm]
            cache_file = cache_dir / arm / f"{q.id}.json"
            result = None
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                # error rows are retried on resume: a cached failure must not stick
                if not cached.get("flags", {}).get("error"):
                    result = RerankResult(**cached)
            if result is None:
                result = reranker.rerank(q.query, candidates, {"query_id": q.id})
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(asdict(result)), encoding="utf-8")
            rows.append(
                {
                    **asdict(result),
                    "query_type": q.query_type,
                    "gold": list(q.gold_chunk_ids),
                    "model_version": versions[arm],
                }
            )

    meta = {
        "tag": tag,
        "created": datetime.now(timezone.utc).isoformat(),
        "arms": list(arms),
        "n_queries": len(queries),
        "limit": limit,
        "top_k": cfg.top_k,
        "model_versions": versions,
    }
    (run_dir / "run.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    return run_dir


def _make_image_for(cfg):
    holder = {"query_type": ""}

    def image_for(c):
        if holder["query_type"] != "graphical":
            return None
        return page_image_data_uri(cfg, c.doc_id, c.page_start)

    def set_query_type(query_meta: dict):
        holder["query_type"] = query_meta.get("query_type", "")

    return image_for


def main():
    parser = argparse.ArgumentParser(description="run the rerank benchmark")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--arms", default="all", help="comma list or 'all'")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from rerankbench.config import load_config

    cfg = load_config()
    arms = list(ALL_ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    t0 = time.perf_counter()
    run_dir = run_bench(
        cfg,
        cfg.data_dir / "dataset" / "queries.jsonl",
        arms=arms,
        tag=args.tag,
        limit=args.limit,
    )
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    print(f"ran {run['meta']['n_queries']} queries x {len(arms)} arms in {time.perf_counter() - t0:.1f}s -> {run_dir}")
    for arm in arms:
        arm_rows = [r for r in run["rows"] if r["arm"] == arm]
        mean_latency = sum(r["latency_ms"] for r in arm_rows) / max(len(arm_rows), 1)
        total_cost = sum(r["cost_usd"] for r in arm_rows)
        print(f"  {arm:14s} mean latency {mean_latency:8.1f} ms   cost ${total_cost:.4f}")


if __name__ == "__main__":
    main()
