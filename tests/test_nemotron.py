import json
from pathlib import Path

import numpy as np
import pytest

from rerankbench.chunk import Chunk
from rerankbench.rerankers.jev import JevError
from rerankbench.rerankers.nemotron import NemotronReranker, page_image_data_uri, probe_slugs
from tests.fakes import BodyRoutedPost, CapturingPost, FakeCfg


def chunks():
    return [Chunk("c1", "d", "s", "alpha", 1, 1), Chunk("c2", "d", "s", "beta", 1, 1)]


def reply():
    return {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.3}]}


def test_nemotron_maps_scores_by_index():
    post = CapturingPost(reply())
    r = NemotronReranker(cfg=FakeCfg(), post=post).rerank("q", chunks(), {"query_id": "q1"})
    assert r.arm == "nemotron_vl"
    assert r.ranked_ids == ["c2", "c1"]
    assert r.scores == [0.9, 0.3]
    assert r.calls == 1
    assert post.last_body["model"] == FakeCfg.nemotron_slug
    assert post.last_body["query"] == "q"
    assert post.last_body["top_n"] == 2
    assert post.last_body["documents"] == [{"text": "alpha"}, {"text": "beta"}]
    assert post.last_headers["Authorization"] == "Bearer fake-or-key"


def test_nemotron_sends_image_when_image_for_returns_uri():
    post = CapturingPost(reply())
    r = NemotronReranker(
        cfg=FakeCfg(),
        post=post,
        image_for=lambda c: "data:image/png;base64,AAA" if c.id == "c1" else None,
    ).rerank("q", chunks(), {"query_id": "q1"})
    assert r.ranked_ids == ["c2", "c1"]
    assert post.last_body["documents"][0] == {"image": "data:image/png;base64,AAA", "text": "alpha"}
    assert "image" not in post.last_body["documents"][1]


def test_nemotron_accepts_slug_override():
    post = CapturingPost(reply())
    r = NemotronReranker(cfg=FakeCfg(), post=post, slug="override-slug").rerank(
        "q", chunks(), {"query_id": "q1"}
    )
    assert post.last_body["model"] == "override-slug"
    assert r.ranked_ids == ["c2", "c1"]


def test_run_bench_uses_probed_slug_for_requests(tmp_path):
    from rerankbench.run_bench import run_bench

    def routed(url, body, headers):
        from rerankbench.rerankers.jev import JevError

        if body["model"] == FakeCfg.nemotron_slug:
            raise JevError("http 404", status=404)  # base slug: no endpoint
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    cfg = make_world(tmp_path)
    run_dir = run_bench(
        cfg,
        cfg.data_dir / "dataset" / "queries.jsonl",
        arms=["nemotron_vl"],
        tag="slugprobe",
        limit=1,
        embedder=FakeEmbedder(),
        posts={"openrouter": routed},
    )
    rows = [r for r in json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["rows"]]
    assert rows[0]["flags"] == {}  # probed slug actually used: no fallback
    assert rows[0]["ranked_ids"][0] == "t_c0000"
    assert rows[0]["model_version"] == FakeCfg.nemotron_slug_free


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
    vectors = {"alpha one": [1.0, 0.0], "beta two": [0.0, 1.0], "gamma three": [0.5, 0.5]}
    ids = np.array(["t_c0000", "t_c0001", "t_c0002"])
    matrix = np.array(list(vectors.values()), dtype=np.float32)
    np.savez(data / "index.npz", ids=ids, matrix=matrix)
    return FakeCfg(data_dir=data, results_dir=tmp_path / "results")


class FakeEmbedder:
    def embed(self, texts):
        vectors = {"alpha one": [1.0, 0.0], "beta two": [0.0, 1.0], "gamma three": [0.5, 0.5]}
        m = np.array([vectors.get(t, [1.0, 0.0]) for t in texts], dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def test_nemotron_retries_on_429_then_succeeds():
    post = CapturingPost(reply(), statuses=[429, 200])
    r = NemotronReranker(cfg=FakeCfg(), post=post, sleep=lambda s: None).rerank(
        "q", chunks(), {"query_id": "q1"}
    )
    assert post.calls == 2
    assert r.ranked_ids == ["c2", "c1"]
    assert r.flags == {}


def test_nemotron_degrades_to_candidate_order_on_total_failure():
    post = CapturingPost(reply(), statuses=[500, 200])
    r = NemotronReranker(cfg=FakeCfg(), post=post, sleep=lambda s: None).rerank(
        "q", chunks(), {"query_id": "q1"}
    )
    assert post.calls == 1  # non-retryable: no crash, no retry
    assert r.ranked_ids == ["c1", "c2"]
    assert "error" in r.flags


def test_nemotron_gives_up_after_attempts_on_persistent_429():
    post = CapturingPost(reply(), statuses=[429] * 6)
    r = NemotronReranker(cfg=FakeCfg(), post=post, sleep=lambda s: None).rerank(
        "q", chunks(), {"query_id": "q1"}
    )
    assert post.calls == 5
    assert r.ranked_ids == ["c1", "c2"]
    assert "error" in r.flags


def test_probe_slugs_prefers_base_then_free():
    ok = probe_slugs(FakeCfg(), post=BodyRoutedPost(lambda body: {"results": []}))
    assert ok == FakeCfg.nemotron_slug

    def base_404(url, body, headers):
        if body["model"] == FakeCfg.nemotron_slug:
            raise JevError("http 404", status=404)
        return {"results": []}

    assert probe_slugs(FakeCfg(), post=base_404) == FakeCfg.nemotron_slug_free
    assert probe_slugs(FakeCfg(), post=CapturingPost({"results": []}, statuses=[404, 404])) is None


def _tiny_pdf(path: Path):
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << >> >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


def test_page_image_data_uri_renders_caches_and_degrades(tmp_path):
    pdf_path = tmp_path / "cache" / "corpus" / "testdoc.pdf"
    pdf_path.parent.mkdir(parents=True)
    _tiny_pdf(pdf_path)
    cfg = FakeCfg(data_dir=tmp_path)

    uri = page_image_data_uri(cfg, "testdoc", 1)
    assert uri is not None and uri.startswith("data:image/png;base64,")
    cached = tmp_path / "cache" / "pages" / "testdoc" / "p1.png"
    assert cached.exists() and cached.stat().st_size > 0

    assert page_image_data_uri(cfg, "nosuchdoc", 1) is None


def test_page_image_data_uri_walks_dpi_ladder_when_oversized(tmp_path, monkeypatch):
    import base64

    import rerankbench.rerankers.nemotron as nem

    pdf_path = tmp_path / "cache" / "corpus" / "testdoc.pdf"
    pdf_path.parent.mkdir(parents=True)
    _tiny_pdf(pdf_path)
    cfg = FakeCfg(data_dir=tmp_path)
    monkeypatch.setattr(nem, "MAX_IMAGE_BYTES", 8)

    def render_by_dpi(pdf_path, page, dpi):
        if dpi == 150:
            return b"x" * 100  # top rung "too big"
        return b"small"

    monkeypatch.setattr(nem, "_render_png", render_by_dpi)
    uri = page_image_data_uri(cfg, "testdoc", 1)
    assert uri == "data:image/png;base64," + base64.b64encode(b"small").decode("ascii")
    assert (tmp_path / "cache" / "pages" / "testdoc" / "p1.png").read_bytes() == b"small"


def test_page_image_data_uri_returns_none_when_all_rungs_oversized(tmp_path, monkeypatch, capsys):
    import rerankbench.rerankers.nemotron as nem

    pdf_path = tmp_path / "cache" / "corpus" / "testdoc.pdf"
    pdf_path.parent.mkdir(parents=True)
    _tiny_pdf(pdf_path)
    cfg = FakeCfg(data_dir=tmp_path)
    monkeypatch.setattr(nem, "MAX_IMAGE_BYTES", 8)
    monkeypatch.setattr(nem, "_render_png", lambda p, pg, dpi: b"x" * 100)

    assert page_image_data_uri(cfg, "testdoc", 1) is None
    assert not (tmp_path / "cache" / "pages" / "testdoc" / "p1.png").exists()
    assert "oversized" in capsys.readouterr().out.lower()
