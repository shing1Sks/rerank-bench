"""Local embeddings via Ollama + a flat cosine index (no vector DB at this corpus size)."""
from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np

from rerankbench.config import Config

BATCH = 32


def _default_post(url: str, body: dict, headers=None) -> dict:
    r = httpx.post(url, json=body, timeout=httpx.Timeout(600))
    r.raise_for_status()
    return r.json()


class OllamaEmbedder:
    def __init__(self, cfg: Config, post=None):
        self.url = cfg.ollama_url
        self.model = cfg.embed_model
        self._post = post or _default_post

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            data = self._post(self.url, {"model": self.model, "input": texts[i : i + BATCH]}, None)
            out.extend(data["embeddings"])
        m = np.array(out, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(norms, 1e-9)


def build_index(cfg: Config, chunks: list, embedder: OllamaEmbedder) -> tuple[list[str], np.ndarray]:
    ids = [c.id for c in chunks]
    matrix: list[list[float]] = []
    for i in range(0, len(ids), 200):
        matrix.extend(embedder.embed([c.text for c in chunks[i : i + 200]]).tolist())
        print(f"embedded {min(i + 200, len(ids))}/{len(ids)}", flush=True)
    m = np.array(matrix, dtype=np.float32)
    np.savez(cfg.data_dir / "index.npz", ids=np.array(ids), matrix=m)
    return ids, m


def load_index(cfg: Config) -> tuple[list[str], np.ndarray]:
    data = np.load(cfg.data_dir / "index.npz")
    return [str(i) for i in data["ids"]], data["matrix"]


def cosine_top_k(query_vec: np.ndarray, ids: list[str], matrix: np.ndarray, k: int) -> list[tuple[str, float]]:
    q = query_vec / max(float(np.linalg.norm(query_vec)), 1e-9)
    scores = matrix @ q
    order = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in order]


if __name__ == "__main__":
    from rerankbench.chunk import load_chunks
    from rerankbench.config import load_config

    cfg = load_config()
    chunks = load_chunks(cfg.data_dir / "chunks" / "chunks.jsonl")
    ids, m = build_index(cfg, chunks, OllamaEmbedder(cfg))
    print(f"indexed {len(ids)} chunks, dim {m.shape[1]}")
