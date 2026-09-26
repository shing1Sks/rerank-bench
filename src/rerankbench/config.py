"""Central config: keys from .env, pinned model slugs and benchmark constants."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    typesafe_key: str
    openrouter_key: str
    gemini_key: str
    typesafe_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"
    openrouter_rerank_url: str = "https://openrouter.ai/api/v1/rerank"
    nemotron_slug: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    nemotron_slug_free: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
    gemini_model: str = "gemini-flash-latest"
    ollama_url: str = "http://localhost:11434/api/embed"
    embed_model: str = "nomic-embed-text"
    top_k: int = 30
    chunk_target_tokens: int = 500
    chunk_overlap_tokens: int = 50
    listwise_chunk_cap: int = 800
    jev_price_per_mtok: float = 0.042
    judge_top_n: int = 5
    spot_check_pairs: int = 50
    spot_check_seed: int = 13
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    results_dir: Path = field(default_factory=lambda: ROOT / "results")


def load_config(env_file: Path | None = None) -> Config:
    load_dotenv(env_file if env_file else ROOT / ".env")
    cfg = Config(
        typesafe_key=os.environ.get("TYPESAFE_API_KEY", ""),
        openrouter_key=os.environ.get("OPENROUTER_RERANK_KEY", ""),
        gemini_key=os.environ.get("GEMINI_API_KEY", ""),
    )
    missing = [
        n
        for n, v in [
            ("TYPESAFE_API_KEY", cfg.typesafe_key),
            ("OPENROUTER_RERANK_KEY", cfg.openrouter_key),
            ("GEMINI_API_KEY", cfg.gemini_key),
        ]
        if not v
    ]
    if missing:
        raise SystemExit(f"missing keys in .env: {', '.join(missing)}")
    return cfg
