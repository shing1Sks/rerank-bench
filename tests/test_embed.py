from pathlib import Path

import numpy as np

from rerankbench.config import Config
from rerankbench.embed import OllamaEmbedder, build_index, cosine_top_k, load_index


def fake_post(url, body, headers=None):
    texts = body["input"]
    return {"embeddings": [[float(len(t)), 1.0] for t in texts]}


def cfg_with(tmp_path: Path) -> Config:
    return Config(typesafe_key="k", openrouter_key="k", gemini_key="k", data_dir=tmp_path)


def test_embedder_normalizes():
    emb = OllamaEmbedder(cfg_with(Path("unused")), post=fake_post)
    m = emb.embed(["aa", "bbb"])
    assert m.shape == (2, 2)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)


def test_embedder_batches_calls():
    calls = []

    def counting_post(url, body, headers=None):
        calls.append(len(body["input"]))
        return fake_post(url, body, headers)

    emb = OllamaEmbedder(cfg_with(Path("unused")), post=counting_post)
    emb.embed([f"t{i}" for i in range(70)])
    assert calls == [32, 32, 6]


def test_cosine_top_k_orders_descending():
    ids = ["a", "b", "c"]
    matrix = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    top = cosine_top_k(q, ids, matrix, k=2)
    assert [i for i, _ in top] == ["a", "b"]
    assert top[0][1] >= top[1][1]


def test_index_roundtrip(tmp_path: Path):
    from rerankbench.chunk import Chunk

    cfg = cfg_with(tmp_path)
    chunks = [Chunk(f"c{i}", "d", "s", f"text{i}", 1, 1) for i in range(3)]
    ids, matrix = build_index(cfg, chunks, OllamaEmbedder(cfg, post=fake_post))
    ids2, matrix2 = load_index(cfg)
    assert ids2 == ids == ["c0", "c1", "c2"]
    assert np.allclose(matrix2, matrix)
