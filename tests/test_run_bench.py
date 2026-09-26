import json

import numpy as np

from rerankbench.judges import PAIRINGS  # noqa: F401  (re-export sanity)
from rerankbench.rerankers.jev import JevError
from rerankbench.run_bench import run_bench
from tests.fakes import BodyRoutedPost, FakeCfg


VECTORS = {"alpha one": [1.0, 0.0], "beta two": [0.0, 1.0], "gamma three": [0.5, 0.5]}


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        m = np.array([VECTORS.get(t, [1.0, 0.0]) for t in texts], dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def make_world(tmp_path):
    data = tmp_path / "data"
    (data / "chunks").mkdir(parents=True)
    (data / "dataset").mkdir()
    with (data / "chunks" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for i, (cid, text) in enumerate(
            [("t_c0000", "alpha one"), ("t_c0001", "beta two"), ("t_c0002", "gamma three")]
        ):
            f.write(
                json.dumps(
                    {"id": cid, "doc_id": "t", "source": "s", "text": text,
                     "page_start": i + 1, "page_end": i + 1}
                )
                + "\n"
            )
    with (data / "dataset" / "queries.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"id": "ver_01", "query": "find alpha one", "query_type": "verbatim",
                 "gold_chunk_ids": ["t_c0000"], "source_doc": "t", "notes": ""}
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {"id": "ver_02", "query": "find beta two", "query_type": "verbatim",
                 "gold_chunk_ids": ["t_c0001"], "source_doc": "t", "notes": ""}
            )
            + "\n"
        )
    ids = np.array(["t_c0000", "t_c0001", "t_c0002"])
    matrix = np.array(list(VECTORS.values()), dtype=np.float32)
    np.savez(data / "index.npz", ids=ids, matrix=matrix)
    return FakeCfg(data_dir=data, results_dir=tmp_path / "results")


def jev_reply(body):
    return {
        "answers": {"q_rank": {"choice": "t_c0000", "probabilities": {"t_c0000": 0.6, "t_c0001": 0.3, "t_c0002": 0.1}}},
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def test_run_bench_writes_run_json_rows_and_cache(tmp_path):
    cfg = make_world(tmp_path)
    post = BodyRoutedPost(jev_reply)
    run_dir = run_bench(
        cfg,
        cfg.data_dir / "dataset" / "queries.jsonl",
        arms=["embed_only", "jev_listwise"],
        tag="smoke",
        limit=1,
        embedder=FakeEmbedder(),
        posts={"typesafe": post, "openrouter": BodyRoutedPost(lambda b: {"results": []})},
    )
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(run["rows"]) == 2  # 1 query x 2 arms
    by_arm = {r["arm"]: r for r in run["rows"]}
    assert by_arm["embed_only"]["ranked_ids"][0] == "t_c0000"
    assert by_arm["jev_listwise"]["ranked_ids"][0] == "t_c0000"
    row = by_arm["jev_listwise"]
    assert row["query_type"] == "verbatim"
    assert row["gold"] == ["t_c0000"]
    assert row["model_version"] == "jev-latest"
    assert (run_dir / "cache" / "jev_listwise" / "ver_01.json").exists()
    assert (run_dir / "cache" / "embed_only" / "ver_01.json").exists()


def test_run_bench_cache_hit_skips_api(tmp_path):
    cfg = make_world(tmp_path)
    post = BodyRoutedPost(jev_reply)
    kwargs = dict(
        dataset_path=cfg.data_dir / "dataset" / "queries.jsonl",
        arms=["jev_listwise"],
        tag="smoke",
        limit=1,
        embedder=FakeEmbedder(),
        posts={"typesafe": post, "openrouter": BodyRoutedPost(lambda b: {"results": []})},
    )
    run_bench(cfg, **kwargs)
    calls_after_first = post.calls
    assert calls_after_first == 1
    run_bench(cfg, **kwargs)
    assert post.calls == calls_after_first  # cache hit: no second API call


def test_run_bench_limit_controls_query_count(tmp_path):
    cfg = make_world(tmp_path)
    run_dir = run_bench(
        cfg,
        cfg.data_dir / "dataset" / "queries.jsonl",
        arms=["embed_only"],
        tag="limited",
        limit=2,
        embedder=FakeEmbedder(),
        posts={},
    )
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(run["rows"]) == 2


def test_run_bench_resume_retries_cached_error_rows(tmp_path):
    cfg = make_world(tmp_path)

    def failing(url, body, headers):
        raise JevError("http 500", status=500)  # non-retryable: fail fast, degrade

    calls = []

    def ok(url, body, headers):
        calls.append(1)
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    kwargs = dict(
        dataset_path=cfg.data_dir / "dataset" / "queries.jsonl",
        arms=["nemotron_vl"],
        tag="retry",
        limit=1,
        embedder=FakeEmbedder(),
    )
    run_dir = run_bench(cfg, **kwargs, posts={"openrouter": failing})
    rows = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["rows"]
    assert rows[0]["flags"].get("error")

    run_dir = run_bench(cfg, **kwargs, posts={"openrouter": ok})
    rows = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["rows"]
    assert calls, "error row was served from cache instead of retried"
    assert rows[0]["flags"] == {}
