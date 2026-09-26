import textwrap
from pathlib import Path

from rerankbench.config import load_config

KEY_VARS = ("TYPESAFE_API_KEY", "OPENROUTER_RERANK_KEY", "GEMINI_API_KEY")


def test_load_config_reads_env_file(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(textwrap.dedent("""
        TYPESAFE_API_KEY=k1
        OPENROUTER_RERANK_KEY=k2
        GEMINI_API_KEY=k3
    """).strip() + "\n", encoding="utf-8")
    cfg = load_config(env)
    assert cfg.typesafe_key == "k1"
    assert cfg.top_k == 30
    assert cfg.jev_price_per_mtok == 0.042
    assert "llama-nemotron-rerank-vl-1b-v2" in cfg.nemotron_slug


def test_load_config_fails_loud_on_missing_keys(tmp_path: Path, monkeypatch):
    # earlier tests leak dotenv-loaded keys into the real environ; clear them
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("TYPESAFE_API_KEY=k1\n", encoding="utf-8")
    try:
        load_config(env)
        raised = False
    except SystemExit as e:
        raised = "OPENROUTER_RERANK_KEY" in str(e)
    assert raised
