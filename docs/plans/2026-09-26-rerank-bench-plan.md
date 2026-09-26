# rerank-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a public benchmark comparing 4 rerank arms (embed-only, Nemotron VL, Jev listwise, Jev pointwise) on real documents, scored by gold IR metrics + blind swap-corrected LLM judging.

**Architecture:** Offline pipeline of plain Python modules: corpus → chunks → local embeddings → top-30 candidates per query → 4 reranker arms over identical candidates → cached run artifacts → judge pass → markdown/JSON report. Every external API is called via plain `httpx` with the transport injectable, so tests never touch the network.

**Tech Stack:** Python 3.12, uv, httpx, pypdfium2, numpy, python-dotenv, beautifulsoup4, pytest.

**Spec:** `docs/specs/2026-09-26-rerank-bench-design.md` (approved 2026-09-26 — this plan argues from it; executors read both).

## Global Constraints

- Python 3.12 via `.python-version`; dependency groups only: `httpx`, `pypdfium2`, `numpy`, `python-dotenv`, `beautifulsoup4`; dev: `pytest`. No LLM SDKs — all four APIs (TypeSafe, OpenRouter, Gemini, Ollama) go through plain `httpx` with injectable transport.
- **Contract wall:** git commit/push steps below run ONLY after Shreyash's explicit approval (batched at phase gates P1/P5). Never push without a separate yes.
- User gates are hard stops: **GATE-P1** (repo create + push), **GATE-P3** (before authoring the full dataset), **GATE-P4** (before the full run), **GATE-P5** (before preserving/publishing results).
- pytest must make zero network calls — live API contact only through `rerankbench.probe` and the run scripts, outside tests.
- All file I/O: `pathlib` + explicit `encoding="utf-8"` (Windows host, PowerShell 5.1). JSON files written via the Write tool or `json.dump`, never `Set-Content`.
- Constants pinned by spec: K=30 candidates; judge sees top-5; chunks ~500 tokens, ~50 overlap; listwise truncates chunk text to 800 tokens; Jev $0.042/Mtok input, output free; spot-check subset 50 pairs, seed 13.
- No emojis anywhere in the repo (README, code comments, report output).
- Keys live in `.env` (gitignored), copied from `Desktop/JevCity/.env` into the repo root; `.env.example` committed with empty values.

## Review Focus

1. **Negative queries leaking into gold metrics** — a query with `gold_chunk_ids: []` must yield `None` from `recall_at_k`/`mrr`/`ndcg_at_k` and be excluded from those aggregates, not scored as 0. Test in Task 12.
2. **Listwise state overflow** — 30 long chunks can exceed the 32k-token state limit; the listwise arm must truncate each candidate to 800 tokens, set a `truncated: true` flag, and still produce a full ranking. Test in Task 10.
3. **Position-swap flips silently dropped** — inconsistent double-judgments must be counted as `flips` and reported in the judge summary; headline win/tie/loss aggregates both orders. Test in Task 13.
4. **Nemotron API failure modes** — 429/529 retried with exponential backoff (5 attempts), 413 on image payloads triggers DPI fallback (150→110→80) and a per-query error row instead of crashing the run. Tests in Task 9.
5. **Run resumability** — rerunning the same tag must skip queries whose raw responses are already cached (call-count must not increase), so rate limits or crashes never force a full re-run. Test in Task 14.

---

### Task 1: Repo scaffold (P1 — after GATE-P1 approval)

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `.env` (uncommitted), `README.md`, `CHANGELOG.md`, `src/rerankbench/__init__.py`, `src/rerankbench/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass + `load_config(env_file: Path | None) -> Config` — every later task imports these from `rerankbench.config`.

- [ ] **Step 1: Init uv project files**

`pyproject.toml`:

```toml
[project]
name = "rerank-bench"
version = "0.1.0"
description = "Benchmark: Jev as a reranker vs NVIDIA Nemotron VL rerank, on real documents, judged blind."
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pypdfium2>=4",
    "numpy>=2.0",
    "python-dotenv>=1.0",
    "beautifulsoup4>=4.12",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rerankbench"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`.python-version`: `3.12`

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.env
data/cache/
data/corpus/
results/**/cache/
.pytest_cache/
```

`.env.example` (committed, empty values):

```
TYPESAFE_API_KEY=
OPENROUTER_RERANK_KEY=
GEMINI_API_KEY=
```

- [ ] **Step 2: Write config.py**

```python
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
    missing = [n for n, v in [("TYPESAFE_API_KEY", cfg.typesafe_key), ("OPENROUTER_RERANK_KEY", cfg.openrouter_key), ("GEMINI_API_KEY", cfg.gemini_key)] if not v]
    if missing:
        raise SystemExit(f"missing keys in .env: {', '.join(missing)}")
    return cfg
```

- [ ] **Step 3: Write failing test**

`tests/test_config.py`:

```python
import textwrap
from pathlib import Path

from rerankbench.config import load_config


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


def test_load_config_fails_loud_on_missing_keys(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("TYPESAFE_API_KEY=k1\n", encoding="utf-8")
    try:
        load_config(env)
        raised = False
    except SystemExit as e:
        raised = "OPENROUTER_RERANK_KEY" in str(e)
    assert raised
```

- [ ] **Step 4: `uv sync` then `uv run pytest`** — expect PASS (2 tests).
- [ ] **Step 5: Create real `.env`** — copy the three key lines from `C:\Users\SHREYASH KUMAR SINGH\Desktop\JevCity\.env` (TYPESAFE_API_KEY, OPENROUTER_RERANK_KEY, GEMINI_API_KEY). Verify `git check-ignore .env` prints the file (ignored).
- [ ] **Step 6: README stub** (title, one-paragraph description from spec §1, "work in progress" note, no emoji) + empty `CHANGELOG.md` header.
- [ ] **Step 7: GATE-P1** — present to Shreyash, then on explicit go:
  `git init -b main`, commit scaffold, `gh repo create shing1Sks/rerank-bench --public --source . --push` (push is itself the approved action; no further pushes without new approval).

---

### Task 2: Chunking (pure logic)

**Files:**
- Create: `src/rerankbench/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Produces: `Chunk` dataclass (`id, doc_id, source, text, page_start, page_end`), `approx_tokens(text) -> int`, `chunk_pages(pages: list[tuple[int, str]], doc_id, source, *, target_tokens, overlap_tokens) -> list[Chunk]`, `save_chunks(chunks, path)`, `load_chunks(path) -> list[Chunk]` (JSONL, utf-8).

- [ ] **Step 1: Failing tests**

```python
from pathlib import Path

from rerankbench.chunk import Chunk, approx_tokens, chunk_pages, load_chunks, save_chunks


def test_approx_tokens_is_words_times_four_thirds():
    assert approx_tokens("one two three") == 4
    assert approx_tokens("") == 1


def test_chunks_respect_page_boundaries_and_overlap():
    pages = [(1, " ".join(["alpha"] * 400)), (2, " ".join(["beta"] * 400))]
    chunks = chunk_pages(pages, "doc1", "test", target_tokens=100, overlap_tokens=20)
    assert chunks, "expected chunks"
    assert all(c.page_start == c.page_end for c in chunks), "no chunk may span pages"
    assert all(c.doc_id == "doc1" and c.id.startswith("doc1_c") for c in chunks)
    # overlap: consecutive chunk texts in the same page share trailing/leading words
    same_page = [c for c in chunks if c.page_start == 1]
    assert len(same_page) > 1
    tail = same_page[0].text.split()[-5:]
    head = same_page[1].text.split()[:5]
    assert tail == head


def test_chunk_jsonl_roundtrip(tmp_path: Path):
    pages = [(1, "hello world " * 100)]
    chunks = chunk_pages(pages, "d", "s", target_tokens=50, overlap_tokens=10)
    path = tmp_path / "c.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
```

- [ ] **Step 2: Run** `uv run pytest tests/test_chunk.py` — expect FAIL (module missing).
- [ ] **Step 3: Implement**

```python
"""Page-aware chunking: word-window chunks that never span pages."""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    page_start: int
    page_end: int


def approx_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)) * 4 // 3)


def _window_page(text: str, doc_id: str, page: int, source: str, target: int, overlap: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    win, ov = int(target * 3 / 4), max(1, int(overlap * 3 / 4))
    step = max(1, win - ov)
    chunks, i = [], 0
    while i < len(words):
        piece = " ".join(words[i : i + win])
        chunks.append(Chunk(f"{doc_id}_c{len(chunks):04d}", doc_id, source, piece, page, page))
        if i + win >= len(words):
            break
        i += step
    return chunks


def chunk_pages(pages, doc_id, source, *, target_tokens=500, overlap_tokens=50) -> list[Chunk]:
    out: list[Chunk] = []
    for page, text in pages:
        # counter continues across pages so ids stay unique without page numbers
        base = len(out)
        made = _window_page(text, doc_id, page, source, target_tokens, overlap_tokens)
        out.extend(Chunk(f"{doc_id}_c{base + n:04d}", made[n].doc_id if made else doc_id, source, made[n].text if made else "", made[n].page_start if made else page, made[n].page_end if made else page) for n in range(len(made)))
    return out
```

Correction: the list-comp above re-indexes wrongly — write the loop plainly instead:

```python
def chunk_pages(pages, doc_id, source, *, target_tokens=500, overlap_tokens=50) -> list[Chunk]:
    out: list[Chunk] = []
    for page, text in pages:
        words = text.split()
        if not words:
            continue
        win, ov = int(target_tokens * 3 / 4), max(1, int(overlap_tokens * 3 / 4))
        step = max(1, win - ov)
        i = 0
        while i < len(words):
            piece = " ".join(words[i : i + win])
            out.append(Chunk(f"{doc_id}_c{len(out):04d}", doc_id, source, piece, page, page))
            if i + win >= len(words):
                break
            i += step
    return out


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_chunk.py` — expect PASS.
- [ ] **Step 5: Commit** (on approval at next gate): `git add src/rerankbench/chunk.py tests/test_chunk.py` → `feat: page-aware chunking`

---

### Task 3: Corpus ingest (download + extract)

**Files:**
- Create: `src/rerankbench/ingest.py`, `data/corpus_manifest.json` (template), `src/rerankbench/bs4_html.py` (tiny HTML→text helper)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `chunk_pages`, `save_chunks` (Task 2), `Config.data_dir` (Task 1).
- Produces: `SOURCES` (ordered list of source dicts), `fetch_all(cfg, client) -> dict[str, Path]` (doc_id → local file), `extract_pages(doc_id, path) -> list[tuple[int, str]]`, `build_corpus(cfg, client) -> list[Chunk]` (writes `data/chunks/chunks.jsonl` + updates `data/corpus_manifest.json` with sha256 of each fetched file).

- [ ] **Step 1: Source table + manifest template.** `SOURCES` in `ingest.py`, each entry `{"doc_id", "title", "kind", "license", "urls": [candidates in order]}`:

```python
SOURCES = [
    {"doc_id": "mobydick", "title": "Moby Dick", "kind": "gutenberg_txt", "license": "public domain",
     "urls": ["https://www.gutenberg.org/cache/epub/2701/pg2701.txt"]},
    {"doc_id": "constitution", "title": "US Constitution (transcript)", "kind": "html", "license": "public domain",
     "urls": ["https://www.archives.gov/founding-docs/constitution-transcript"]},
    {"doc_id": "scotus", "title": "SCOTUS opinion (long)", "kind": "html", "license": "public domain",
     "urls": [
        "https://supreme.justia.com/cases/federal/us/576/644/",
        "https://www.law.cornell.edu/supct/html/14-556.ZO.html",
     ]},
    {"doc_id": "openstax_phys", "title": "OpenStax University Physics vol 1 (ch 1-8 pages)", "kind": "pdf", "license": "CC-BY 4.0",
     "urls": ["https://openstax.org/api/pages/university-physics-volume-1?include=pdf"]},
    {"doc_id": "openstax_calc", "title": "OpenStax Calculus vol 1", "kind": "pdf", "license": "CC-BY 4.0",
     "urls": ["https://openstax.org/api/pages/calculus-volume-1?include=pdf"]},
    {"doc_id": "d2l", "title": "Dive into Deep Learning", "kind": "pdf", "license": "CC-BY-SA 4.0",
     "urls": ["https://d2l.ai/d2l-en.pdf"]},
]
WIKI_TITLES = ["Renaissance", "Photosynthesis", "Quantum mechanics", "Plate tectonics",
               "French Revolution", "Great Barrier Reef"]
```

URLs are candidates: at run time the fetcher tries each in order and pins the first that returns HTTP 200 plus a non-trivial body, recording the winning URL + sha256 into `data/corpus_manifest.json`. If every candidate for a source fails, print the failure and continue (run stays resumable); the manifest records `status: failed` for it. OpenStax's page-API JSON is a candidate mechanism — if the response does not contain a direct PDF URL, fall back to scraping the details page for the first `...pdf` href with bs4 (both mechanisms written, since the URL pattern is the one genuinely unknown thing).

- [ ] **Step 2: Failing tests** (fakes only — no network in tests):

```python
import json
from pathlib import Path

import httpx

from rerankbench.ingest import extract_pages, pick_url


class FakeResp:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status


def test_pick_url_prefers_first_success():
    calls = []

    def client_get(url):
        calls.append(url)
        return FakeResp(b"ok") if url.endswith("good") else FakeResp(b"", status=404)

    assert pick_url(["bad", "good", "later"], client_get) == "good"
    assert calls == ["bad", "good"]


def test_pick_url_returns_none_when_all_fail():
    assert pick_url(["a", "b"], lambda u: FakeResp(b"", status=500)) is None


def test_extract_pages_pdf(tmp_path: Path):
    # build a 2-page pdf with pypdfium2 itself: skip (fixture too heavy) -> use txt kind instead
    text_file = tmp_path / "doc.txt"
    text_file.write_text("page one words " * 50 + "\n\f" + "page two words " * 50, encoding="utf-8")
    pages = extract_pages("d", text_file)
    assert len(pages) == 2 and pages[0][0] == 1 and pages[1][0] == 2
```

- [ ] **Step 3: Run** `uv run pytest tests/test_ingest.py` — FAIL.
- [ ] **Step 4: Implement** — mechanisms: `httpx.Client(follow_redirects=True, timeout=120)`; kinds: `gutenberg_txt` (strip Gutenberg header/footer between `*** START`/`*** END` markers, split on `\f` → pages, else 3000-word page blocks with `page_start=page_end=0`); `html` (bs4: drop `script/style/nav/footer`, `get_text("\n")`, collapse blank lines, single page 0); `pdf` (`pypdfium2`: for each page `page.get_textpage().get_text_range()`); `wikipedia` (per title: `https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&redirects=1&titles=<t>` via GET, take `pages -> * -> extract`, each article = own doc_id `wiki_<slug>` + one `SOURCES`-style entry appended with license `CC-BY-SA 4.0`, attribution URL recorded). Chunk target/overlap from Config. Write `data/chunks/chunks.jsonl` via `save_chunks`. Manifest entry per doc: `{"doc_id", "title", "license", "url", "sha256", "pages", "chunks", "status"}`.

- [ ] **Step 5: Live fetch step (outside pytest, local + reversible):** `uv run python -m rerankbench.ingest`. Expected: per-doc status lines (`mobydick: OK url=... sha256=... pages=... chunks=...`); final corpus total within the spec's 1,500–3,000 chunks; print per-source chunk counts for balance. d2l PDF is ~15MB — expect a slow first fetch. If OpenStax cloudfront URLs 404, use the details-page scrape fallback and note the working URL in the manifest.
- [ ] **Step 6: Spot-check extraction** — print 2 random chunks from 3 different sources; confirm readable text (no tag soup, no mojibake). Math sources will show mangled equations — expected per spec §12.
- [ ] **Step 7: Commit** (batched on approval): `feat: corpus ingest with manifest and page-aware chunks`

---

### Task 4: Embeddings + vector index

**Files:**
- Create: `src/rerankbench/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `Chunk`, `load_chunks` (Task 2), `Config.ollama_url`, `Config.embed_model`.
- Produces: `OllamaEmbedder(cfg, post)` with `.embed(texts: list[str]) -> "np.ndarray"` (L2-normalized rows); `build_index(cfg, chunks, embedder) -> None` (saves `data/index.npz`: `ids` str array + `matrix` float32); `load_index(cfg) -> tuple[list[str], "np.ndarray"]`; `cosine_top_k(query_vec, ids, matrix, k) -> list[tuple[str, float]]` (descending).

- [ ] **Step 1: Failing tests**

```python
import numpy as np

from rerankbench.embed import OllamaEmbedder, cosine_top_k


def fake_post(url, json_body, headers=None):
    texts = json_body["input"]
    vecs = np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)
    return {"embeddings": vecs.tolist()}


def test_embedder_normalizes_and_batches():
    emb = OllamaEmbedder(cfg=None, post=fake_post)
    m = emb.embed(["aa", "bbb"])
    assert m.shape == (2, 2)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)


def test_cosine_top_k_orders_descending():
    ids = ["a", "b", "c"]
    matrix = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    top = cosine_top_k(q, ids, matrix, k=2)
    assert [i for i, _ in top] == ["a", "b"]
    assert top[0][1] >= top[1][1]
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** — `post(url, json_body)` does `httpx.post(url, json=json_body, timeout=300)`; request body `{"model": cfg.embed_model, "input": texts}`; parse `resp["embeddings"]`; batch in chunks of 32; normalize `m / norm`. `build_index` embeds all chunk texts (progress print every 200), `np.savez` the ids+matrix. `cosine_top_k`: `scores = matrix @ (q / np.linalg.norm(q))`, `np.argsort` descending, return `(ids[i], float(scores[i]))` for first k.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Local env step:** `ollama pull nomic-embed-text` (~275MB), then `uv run python -m rerankbench.embed` builds `data/index.npz`. Expected: "indexed N chunks, dim 768" where N == chunk count.
- [ ] **Step 6: Commit** (batched): `feat: local ollama embeddings + cosine index`

---

### Task 5: Dataset types + validator (code half of P3)

**Files:**
- Create: `src/rerankbench/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `Query` dataclass (`id, query, query_type, gold_chunk_ids, source_doc, notes`), `QUERY_TYPES = ("verbatim", "paraphrase", "reasoning", "negative", "structural", "graphical")`, `TYPE_QUOTAS` dict (verbatim 40, paraphrase 40, reasoning 30, negative 20, structural 20, graphical 25), `load_queries(path) -> list[Query]`, `validate_queries(queries, chunk_ids: set[str]) -> list[str]` (error strings, empty = valid).

- [ ] **Step 1: Failing tests**

```python
from pathlib import Path

from rerankbench.dataset import Query, load_queries, validate_queries


def q(**kw):
    base = dict(id="q1", query="x", query_type="verbatim", gold_chunk_ids=["c1"], source_doc="d", notes="")
    return Query(**{**base, **kw})


def test_validation_catches_bad_type_and_unknown_gold():
    errs = validate_queries([q(query_type="nope"), q(id="q2", gold_chunk_ids=["ghost"])], {"c1"})
    assert any("query_type" in e and "q1" in e for e in errs)
    assert any("gold" in e and "q2" in e for e in errs)


def test_negative_must_have_empty_gold_and_others_must_not():
    errs = validate_queries([q(query_type="negative", gold_chunk_ids=[]), q(id="q3", gold_chunk_ids=[])], set())
    assert not any("q1" in e for e in errs)
    assert any("q3" in e for e in errs)


def test_jsonl_roundtrip(tmp_path: Path):
    import json
    p = tmp_path / "qs.jsonl"
    p.write_text(json.dumps(q().__dict__) + "\n", encoding="utf-8")
    assert load_queries(p)[0] == q()
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** — validator rules: query_type in QUERY_TYPES; `negative` ⇔ empty gold; every gold id ∈ chunk_ids (skip for negatives); query text 20–400 chars; ids unique. **Step 4: Run** — PASS. **Step 5: Commit** (batched): `feat: dataset schema + validator`

---

### Task 6: GATE-P3 — dataset plan review, then author the dataset

**Files:**
- Create: `data/dataset/queries.jsonl` (after the gate), `src/rerankbench/dataset_stats.py` (counts per type/source, printed table)

- [ ] **Step 1: Build the GATE-P3 review pack** — corpus manifest summary (sources, licenses, chunk counts), type quotas table (Task 5), and **10 sample queries** (first 2 authored per type, gold-labeled by reading actual chunks from the built corpus). Present to Shreyash. STOP — proceed only on explicit go.
- [ ] **Step 2: Author the full dataset** (my manual work, per spec §6): for each of the 6 sources, read chunks and write queries meeting TYPE_QUOTAS (total ≈175, within the spec's 150–250); procedure per type — `verbatim`: answer near-verbatim in one chunk; `paraphrase`: rewrite the idea in different words; `reasoning`: answer needs two chunks, both gold; `negative`: topically adjacent, genuinely unanswerable from corpus, empty gold; `structural`: answer in a caption/footnote/small subsection; `graphical`: answer visible in a figure/table on the chunk's page (PDF sources only — record `source_doc` so the VL arm can render that page). Working notes stay in `data/dataset/curation_notes.md` (committed — provenance).
- [ ] **Step 3: Validate + stats:** `uv run python -m rerankbench.dataset_stats` — must print zero validation errors and per-type counts ≥ quotas. Fix until clean.
- [ ] **Step 4: Commit** (batched, after showing stats): `feat: benchmark dataset — 175 queries, 6 types, gold labels`

---

### Task 7: Reranker protocol + embed-only arm

**Files:**
- Create: `src/rerankbench/rerankers/__init__.py`, `src/rerankbench/rerankers/embed_only.py`
- Test: `tests/test_rerank_embed_only.py`

**Interfaces:**
- Consumes: `Chunk`, `Query`, `cosine_top_k`, `load_index`.
- Produces: `RerankResult` dataclass (`arm: str, query_id: str, ranked_ids: list[str], scores: list[float] | None, latency_ms: float, input_tokens: int, calls: int, cost_usd: float, flags: dict, raw: dict`), `Reranker` Protocol (`.arm: str`, `.rerank(query: str, candidates: list[Chunk], query_meta: dict) -> RerankResult`), `EmbedOnlyReranker(embed_fn)`.

- [ ] **Step 1: Failing test**

```python
from rerankbench.chunk import Chunk
from rerankbench.rerankers.embed_only import EmbedOnlyReranker


def test_embed_only_preserves_candidate_order():
    chunks = [Chunk(f"c{i}", "d", "s", f"t{i}", 1, 1) for i in range(5)]
    # embed order: c3 best, then c0, c4, c1, c2
    emb = lambda texts: {f"t{i}": 4 - i for i, _ in enumerate(texts)}
    r = EmbedOnlyReranker(embed_fn=emb).rerank("q", chunks, {"query_id": "q1"})
    assert r.arm == "embed_only"
    assert r.ranked_ids == ["c3", "c0", "c4", "c1", "c2"]
    assert r.calls == 0 and r.cost_usd == 0.0
```

(EmbedOnlyReranker takes `embed_fn: Callable[[list[str]], dict[str, float]]` returning a score per candidate text — the run script wires a real closure over `cosine_top_k` results; keeping the arm's own interface score-based makes all four arms interchangeable.)

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** both files. **Step 4: Run** — PASS. **Step 5: Commit** (batched): `feat: reranker protocol + embed-only arm`

---

### Task 8: Jev SystemOne client + listwise arm

**Files:**
- Create: `src/rerankbench/rerankers/jev.py`
- Test: `tests/test_jev_listwise.py`

**Interfaces:**
- Consumes: `Config.typesafe_url/.jev_model/.typesafe_key/.listwise_chunk_cap`, `Chunk`.
- Produces: `post_systemone(cfg, state: str, questions: dict, post=_default_post, attempts=5, sleep=time.sleep) -> dict` (returns parsed JSON; retries on 429/529 with `sleep(min(30, 2**n))` backoff; raises `JevError` otherwise); `truncate_for_listwise(text: str, cap_tokens: int) -> str` (word-truncate to cap·3/4 words + " …"); `JevListwise(cfg, post)` Reranker: builds one `choice` question `q_rank` ("Which passage best answers the query?"), candidates as `criteria` `{chunk_id: truncated_text}`, state = short header + query; ranks by `answers["q_rank"]["probabilities"]` descending; flags `{"truncated": bool}`; tokens from `usage.input_tokens`, cost = in/1e6·0.042; missing probabilities → `JevError`. Also creates **`tests/fakes.py`** holding `FakeCfg` (stub config) and `CapturingPost` — every later test file imports these instead of redefining them.

- [ ] **Step 1: Failing tests**

```python
import pytest

from rerankbench.chunk import Chunk
from rerankbench.rerankers.jev import JevError, JevListwise, post_systemone, truncate_for_listwise


class CapturingPost:
    def __init__(self, reply, statuses=()):
        self.reply, self.statuses, self.calls = reply, list(statuses), 0
        self.last_body = None

    def __call__(self, url, body, headers):
        self.calls += 1
        self.last_body = body
        if self.statuses:
            code = self.statuses.pop(0)
            if code != 200:
                raise JevError(f"http {code}")
        return self.reply


def chunks():
    return [Chunk("c1", "d", "s", "alpha", 1, 1), Chunk("c2", "d", "s", "beta", 1, 1)]


def test_listwise_ranks_by_probabilities():
    post = CapturingPost({"answers": {"q_rank": {"choice": "c2", "probabilities": {"c1": 0.2, "c2": 0.8}}},
                          "usage": {"input_tokens": 1000, "output_tokens": 0}})
    r = JevListwise(cfg=FakeCfg(), post=post).rerank("what?", chunks(), {"query_id": "q1"})
    assert r.ranked_ids == ["c2", "c1"]
    assert r.scores == [0.8, 0.2]
    assert r.calls == 1 and abs(r.cost_usd - 1000 / 1e6 * 0.042) < 1e-12
    assert post.last_body["questions"]["q_rank"]["criteria"] == {"c1": "alpha", "c2": "beta"}


def test_listwise_truncates_long_chunks_and_flags():
    long_chunk = Chunk("c3", "d", "s", " ".join(["w"] * 5000), 2, 2)
    post = CapturingPost({"answers": {"q_rank": {"probabilities": {"c3": 1.0}}},
                          "usage": {"input_tokens": 10, "output_tokens": 0}})
    r = JevListwise(cfg=FakeCfg(), post=post).rerank("q", [long_chunk], {"query_id": "q2"})
    assert r.flags["truncated"] is True
    assert len(post.last_body["questions"]["q_rank"]["criteria"]["c3"].split()) <= 800 * 3 // 4 + 2


def test_post_systemone_retries_on_429_then_succeeds():
    post = CapturingPost({"answers": {}, "usage": {}}, statuses=[429, 200])
    post_systemone(FakeCfg(), "state", {}, post=post, sleep=lambda s: None)
    assert post.calls == 2


def test_truncate_keeps_word_boundary():
    out = truncate_for_listwise(" ".join(["word"] * 100), cap_tokens=10)
    assert len(out.split()) <= 8  # 10 * 3/4
```

(`FakeCfg` is a tiny stub with the needed attrs, defined at top of the test file.)

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** Wire format mirrors JevCity's proven client (`Desktop/JevCity/src/server/jev.ts`): `POST cfg.typesafe_url`, headers `Authorization: Bearer`, body `{state, model, questions}`. Response parse: `data["answers"]["q_rank"]["probabilities"]` map. Retry: catch HTTP 429/529 → `time.sleep(min(30, 2**n))`, up to 5 attempts. **Step 4: Run** — PASS. **Step 5: Commit** (batched): `feat: systemone client + jev listwise reranker`

---

### Task 9: Jev pointwise arm (cookbook pattern)

**Files:**
- Modify: `src/rerankbench/rerankers/jev.py` (append pointwise)
- Test: `tests/test_jev_pointwise.py`

**Interfaces:**
- Produces: `JevPointwise(cfg, post, max_workers=8)` Reranker — one SystemOne call per (query, candidate): each call carries a single `noul` question `q_rel` ("Does this passage answer the query? Passage: {text}"); score = `answers["q_rel"]["noul"]` (missing → 0.0 + flag `missing_noul: n`); ranked descending; `calls == len(candidates)`; input tokens = sum across calls.

- [ ] **Step 1: Failing test**

```python
def test_pointwise_scores_each_candidate_separately():
    replies = iter([
        {"answers": {"q_rel": {"noul": 0.9}}, "usage": {"input_tokens": 100, "output_tokens": 0}},
        {"answers": {"q_rel": {"noul": 0.1}}, "usage": {"input_tokens": 110, "output_tokens": 0}},
    ])

    def post(url, body, headers):
        return next(replies)

    r = JevPointwise(cfg=FakeCfg(), post=post, max_workers=2).rerank("q", chunks(), {"query_id": "q1"})
    assert r.ranked_ids == ["c1", "c2"]
    assert r.calls == 2
    assert r.input_tokens == 210
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** with `concurrent.futures.ThreadPoolExecutor(max_workers)`; keep per-call text truncated to the same 800-token cap; latencies: record fan-out **wall time** as `latency_ms` (spec §8). **Step 4: Run** — PASS. **Step 5: Commit** (batched): `feat: jev pointwise reranker (cookbook noul pattern)`

---

### Task 10: Nemotron VL arm (OpenRouter) + page images

**Files:**
- Create: `src/rerankbench/rerankers/nemotron.py`
- Test: `tests/test_nemotron.py`

**Interfaces:**
- Consumes: `Config.openrouter_rerank_url/.openrouter_key/.nemotron_slug`, `Chunk.page_start`.
- Produces: `NemotronReranker(cfg, post, image_for=None)` + `NemotronError` — `image_for: Callable[[Chunk], str | None]` returns a base64 data URI for graphical queries else None; request `POST` `{"model": slug, "query": query, "documents": [{"text": t} | {"image": uri, "text": t}], "top_n": len}`; results `[{index, relevance_score}]` mapped back to chunk ids, ranked by score desc; 429/529 → backoff retry (5 attempts); 413 or image error → DPI fallback inside `page_image_data_uri` (150→110→80; exercised live by the graphical smoke queries since a real PDF is needed to render); a query that still fails yields `flags["error"]` + candidate-order fallback (does NOT crash the run); `probe_slugs(cfg, post) -> str | None` tries `nemotron_slug` then `nemotron_slug_free`, returns the first that 200s on a 2-document smoke payload.

- [ ] **Step 1: Failing tests**

```python
def test_nemotron_maps_scores_to_ids_in_order():
    def post(url, body, headers):
        assert body["model"] == "slug"
        return {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}

    r = NemotronReranker(cfg=FakeCfg(), post=post).rerank("q", chunks(), {"query_id": "q1", "query_type": "verbatim"})
    assert r.ranked_ids == ["c2", "c1"] and r.scores == [0.9, 0.1] and r.calls == 1


def test_nemotron_sends_image_for_graphical_queries():
    seen = {}

    def post(url, body, headers):
        seen["docs"] = body["documents"]
        return {"results": [{"index": 0, "relevance_score": 1.0}]}

    NemotronReranker(cfg=FakeCfg(), post=post, image_for=lambda c: "data:image/png;base64,AAA").rerank(
        "q", chunks(), {"query_id": "q1", "query_type": "graphical"})
    assert seen["docs"][0]["image"].startswith("data:image/png")


def test_nemotron_retries_on_429():
    codes = iter([429, 200])
    def post(url, body, headers):
        c = next(codes)
        if c != 200:
            raise NemotronError(f"http {c}")
        return {"results": [{"index": 0, "relevance_score": 1.0}]}
    r = NemotronReranker(cfg=FakeCfg(), post=post).rerank("q", chunks()[:1], {"query_id": "q1", "query_type": "verbatim"})
    assert r.flags.get("error") is None


def test_nemotron_total_failure_degrades_to_error_flag_not_crash():
    def post(url, body, headers):
        raise NemotronError("http 500")
    r = NemotronReranker(cfg=FakeCfg(), post=post).rerank("q", chunks(), {"query_id": "q1", "query_type": "verbatim"})
    assert "error" in r.flags
    assert r.ranked_ids == ["c1", "c2"]  # candidate order preserved as last resort
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement**, including `page_image_data_uri(cfg, doc_id, page, dpi=150) -> str` (pypdfium2 render → PNG bytes → base64; if encoded size > 1.8MB, retry at lower DPI 110 then 80, else give up with None; page PNGs cached under `data/cache/pages/{doc_id}/p{page}.png` — gitignored). **Step 4: Run** — PASS. **Step 5: Commit** (batched): `feat: nemotron vl reranker with page-image support and slug probe`

---

### Task 11: Metrics

**Files:**
- Create: `src/rerankbench/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `recall_at_k(ranked: list[str], gold: list[str], k: int) -> float | None`, `mrr(ranked, gold) -> float | None`, `ndcg_at_k(ranked, gold, k=10) -> float | None` (binary gains; all three return `None` when `gold` empty); `p50(xs) -> float`, `p95(xs) -> float` (nearest-rank); `win_tie_loss(verdicts: list[str]) -> dict` (verdicts "A"/"B"/"TIE" → counts); `cohens_kappa(a: list[str], b: list[str]) -> float` (3-class, raises on empty/mismatched length).

- [ ] **Step 1: Failing tests**

```python
import pytest

from rerankbench.metrics import cohens_kappa, mrr, ndcg_at_k, p50, p95, recall_at_k, win_tie_loss


def test_negative_gold_returns_none_not_zero():
    assert recall_at_k(["a", "b"], [], 10) is None
    assert mrr(["a"], []) is None
    assert ndcg_at_k(["a"], [], 10) is None


def test_recall_mrr_ndcg_known_values():
    ranked = ["x", "g1", "y", "g2"]
    assert recall_at_k(ranked, ["g1", "g2"], 2) == 0.5
    assert recall_at_k(ranked, ["g1", "g2"], 4) == 1.0
    assert mrr(ranked, ["g1"]) == 0.5
    ideal = ndcg_at_k(["g1", "g2", "x", "y"], ["g1", "g2"], 4)
    assert ndcg_at_k(ranked, ["g1", "g2"], 4) < ideal


def test_p50_p95_nearest_rank():
    xs = list(range(1, 21))
    assert p50(xs) == 10 and p95(xs) == 19


def test_win_tie_loss_and_kappa():
    assert win_tie_loss(["A", "B", "TIE", "A"]) == {"A": 2, "B": 1, "TIE": 1}
    assert cohens_kappa(["A", "A", "B"], ["A", "A", "B"]) == 1.0
    with pytest.raises(ValueError):
        cohens_kappa([], [])
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (nDCG: DCG = Σ relᵢ/log₂(i+1) for i≥2 positions with binary rel; IDCG from |gold| ideal; guard k≤0). **Step 4: Run** — PASS. **Step 5: Commit** (batched): `feat: ir metrics, win/tie/loss, cohens kappa`

---

### Task 12: Judges (Jev primary + Gemini spot-check)

**Files:**
- Create: `src/rerankbench/judges.py`
- Test: `tests/test_judges.py`

**Interfaces:**
- Consumes: `post_systemone` (Task 8), `Config.gemini_model/.gemini_key`, `Chunk` texts, `Config.judge_top_n`.
- Produces: `format_ranking_block(rank_letter: str, ranked_ids, id_to_text, top_n) -> str`; `judge_pair_jev(cfg, query, rank_a, rank_b, id_to_text, post) -> "A"|"B"|"TIE"` (one SystemOne call: state = query + both top-5 blocks labeled "Ranking A"/"Ranking B", single `choice` question with criteria `{A: description of A, B: ..., TIE: ...}`); `judge_orders_jev(cfg, query, rank_a, rank_b, id_to_text, post) -> dict` (judges both orders, reconciles: consistent → `{verdict, flipped: False}`; inconsistent → `{verdict, flipped: True}` where verdict is the both-orders aggregate; flips always counted); `judge_pair_gemini(cfg, query, rank_a, rank_b, id_to_text, post) -> str` (plain REST `POST {base}/models/{model}:generateContent?key=`, prompt asks for exactly `A`, `B`, or `TIE` on line 1 + one-line reason, parse first line, unknown → `TIE` + flag); `run_judge_pass(cfg, run_path, judge="jev", post=...) -> dict` — reads `run.json`, builds pairings `[("jev_listwise","jev_pointwise"), ("nemotron_vl","jev_listwise"), ("nemotron_vl","jev_pointwise"), ("nemotron_vl","embed_only"), ("jev_listwise","embed_only"), ("jev_pointwise","embed_only")]`, per pairing per query uses `judge_orders_jev`, writes `verdicts.jsonl` rows `{pairing, query_id, verdict, flipped}` and returns summary `{pairing: {wins, losses, ties, flips, swap_consistency}}`. Test fakes (`CapturingPost`, `FakeCfg`) come from `tests/fakes.py` (Task 8).

- [ ] **Step 1: Failing tests**

```python
def test_jev_judge_parses_choice_and_blind_labels():
    post = CapturingPost({"answers": {"verdict": {"choice": "A"}}, "usage": {"input_tokens": 5, "output_tokens": 0}})
    v = judge_pair_jev(FakeCfg(), "q?", ["c1"], ["c2"], {"c1": "t1", "c2": "t2"}, post)
    assert v == "A"
    state = post.last_body["state"]
    assert "Ranking A" in state and "Ranking B" in state and "jev" not in state.lower()


def test_swap_flip_is_counted_not_hidden():
    # order1 -> A picks first block; order2 (swapped) also picks first block => flipped
    replies = iter([
        {"answers": {"verdict": {"choice": "A"}}, "usage": {}},
        {"answers": {"verdict": {"choice": "A"}}, "usage": {}},
    ])
    def post(url, body, headers):
        return next(replies)
    # impl judges (rank_a, rank_b) then (rank_b, rank_a); "A" twice means different actual rankings won -> flipped
    summary = judge_orders_jev(FakeCfg(), "q?", ["c1"], ["c2"], {"c1": "t", "c2": "t"}, post)
    assert summary["flipped"] is True and summary["verdict"] in ("A", "B")
```

(Design note: with swapped orders, "consistent" means the same *actual ranking* wins both times — i.e. the letter flips when the block order flips. `judge_orders_jev` wraps two `judge_pair_jev` calls and reconciles: consistent → verdict + `flipped: False`; inconsistent → verdict "TIE-ish" recorded as the both-orders aggregate with `flipped: True`; flips always counted in summary.)

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** both judges + `run_judge_pass` (spot-check wiring: `--judge gemini --subset 50 --seed 13` re-judges a fixed random sample of the Jev verdicts for agreement; kappa computed by Task 11's `cohens_kappa`). **Step 4: Run** — PASS. **Step 5: Commit** (batched): `feat: blind swap-corrected jev judge + gemini spot-check`

---

### Task 13: Run orchestration + cache

**Files:**
- Create: `src/rerankbench/run_bench.py`
- Test: `tests/test_run_bench.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_bench(cfg, dataset_path, arms: list[str], tag: str, limit: int | None) -> Path` — loads chunks+index+queries, per query retrieves top-30, calls each arm, **caches raw per (arm, query) at `results/{tag}/cache/{arm}/{query_id}.json`** and skips the API call when cache exists, writes `results/{tag}/run.json` (`{"meta": {...config slugs/versions/date}, "rows": [per-query per-arm records]}`) and returns the run dir. CLI: `uv run python -m rerankbench.run_bench --tag smoke --arms all --limit 10`.

- [ ] **Step 1: Failing test**

```python
def test_cached_calls_are_skipped(tmp_path):
    calls = {"n": 0}
    arm = StubArm("stub", on_rerank=lambda: calls.__setitem__("n", calls["n"] + 1))
    # first run: 2 queries -> 2 calls; second run with warm cache: 0 calls
    run_bench(cfg=FakeCfg(data_dir=tmp_path, results_dir=tmp_path / "r"),
              dataset_path=two_query_file(tmp_path), arms=["stub"], tag="t", limit=None)
    first = calls["n"]
    run_bench(cfg=FakeCfg(data_dir=tmp_path, results_dir=tmp_path / "r"),
              dataset_path=two_query_file(tmp_path), arms=["stub"], tag="t", limit=None)
    assert first == 2 and calls["n"] == 2
```

(StubArm is a Reranker whose `rerank` returns canned results and counts invocations; cache write/read goes through the same code path real arms use.)

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** — also record per-row: `query_type`, gold ids, ranked ids, `latency_ms`, `input_tokens`, `cost_usd`, `flags`, `model_version` (Jev's returned `model` string / nemotron slug). Graphical queries: only the nemotron arm gets `image_for` wired (`page_image_data_uri` on the gold chunk's page). **Step 4: Run** — PASS.
- [ ] **Step 5: Smoke run (live, 10 queries — spec P4 smoke):** `uv run python -m rerankbench.run_bench --tag smoke --limit 10 --arms all`. Expected: all 4 arms produce 10 rows; print per-arm p50/p95 + token totals; no unhandled exceptions. Then `uv run python -m rerankbench.judges --run results/smoke --judge jev` — expect a summary table with win/tie/loss + flip counts.
- [ ] **Step 6: GATE-P4** — show Shreyash the smoke summary + projected full-run cost (tokens from smoke × 17.5). Proceed only on explicit go. Full run: `--tag run1` (all queries, all arms), then judge pass, then `--judge gemini --subset 50`.
- [ ] **Step 7: Commit** (batched): `feat: run orchestration with resumable cache`

---

### Task 14: Report + README + publish (P5)

**Files:**
- Create: `src/rerankbench/report.py` (tables from `run.json` + judge summaries: gold metrics per arm overall and per query_type; judge win/tie/loss + flips; latency p50/p95; tokens + cost per 1k queries), `README.md` (final: question, method, results tables, verdict, reproduce steps, license attributions for CC-BY/CC-BY-SA sources + Wikipedia attribution URLs, related-work section citing BRIGHT/ViDoRe/arXiv:2403.10407/MT-Bench), `CHANGELOG.md` (decision log per operating contract, written just before this approval gate)
- Test: `tests/test_report.py` (build tables from a 4-row fixture run.json; assert every arm appears, negatives excluded from gold-metric aggregates, kappa line present when spot-check file exists)

- [ ] **Step 1: Failing test**

```python
from pathlib import Path

from rerankbench.report import build_report


def test_report_covers_all_arms_excludes_negatives_and_includes_kappa(tmp_path: Path):
    run = {
        "meta": {"tag": "t"},
        "rows": [
            {"arm": "jev_listwise", "query_id": "q1", "query_type": "verbatim", "ranked_ids": ["g"], "gold": ["g"], "latency_ms": 10.0, "input_tokens": 100, "cost_usd": 0.0},
            {"arm": "embed_only", "query_id": "q1", "query_type": "verbatim", "ranked_ids": ["x"], "gold": ["g"], "latency_ms": 1.0, "input_tokens": 0, "cost_usd": 0.0},
            {"arm": "jev_listwise", "query_id": "q2", "query_type": "negative", "ranked_ids": ["x"], "gold": [], "latency_ms": 10.0, "input_tokens": 100, "cost_usd": 0.0},
        ],
    }
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    report = build_report(tmp_path, judge_summary=None, spot_check=None)
    assert "jev_listwise" in report and "embed_only" in report
    assert "negative" not in report.gold_metric_scope  # negatives excluded from gold aggregates
    assert report.tables["gold_metrics"]["jev_listwise"]["recall_at_1"] == 1.0
```

(The exact report object shape is settled when writing the failing test against the fixture above: `build_report(run_dir, judge_summary, spot_check) -> Report` with `.tables` (gold_metrics per arm, judge per pairing, latency, cost per 1k queries) and `.gold_metric_scope` recording which query types entered gold aggregates; `.to_markdown()` renders the README tables.)
- [ ] **Step 4: Generate** `results/run1/report.md` + update README tables from it.
- [ ] **Step 5: GATE-P5** — present full report + README to Shreyash. On explicit go: final commit + `git push` (the one push this project ends with, separately approved per contract).

## Cost guardrail (from spec §11)

Full-run budget: Jev input across 4 arms + judging ≈ **$0.15–0.25**; Nemotron $0 (free slug) — if the probe only finds a paid slug, STOP and report the projected cost before GATE-P4; Gemini spot-check pennies. Abort threshold: if projected full run exceeds **$1.00**, stop and escalate.
