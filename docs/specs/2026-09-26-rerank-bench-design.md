# rerank-bench — design spec

Date: 2026-09-26 · Status: draft for review · Owner: Shreyash (repo will live at github.com/shing1Sks/rerank-bench, public)

## 1. Motivation and research question

People increasingly use Jev (TypeSafe SystemOne, `jev-latest`) as a reranker. The working
hypothesis going in: that is an off-label use that likely underperforms a purpose-built
cross-encoder reranker. This benchmark settles it with evidence, and — more usefully —
answers *which* use of Jev as a reranker (if any) is the right one, at what latency and cost,
on real messy documents rather than toy corpora.

Research question: **On a candidate set produced by a fixed local embedder over real long
documents (books, law, math, physics, encyclopedic text), how do Jev (two call shapes) and
NVIDIA's Nemotron cross-encoder reranker compare on ranking quality, latency, and cost —
and does reranking beat the embedding order at all?**

## 2. Related work (novelty check)

- **BEIR / MS MARCO / MTEB(r)** — standard IR/rerank substrates; short clean text, gold qrels,
  no LLM judge, no images.
- **BRIGHT** (arXiv:2407.12886) — reasoning-intensive retrieval over real long docs (math, law,
  code, StackExchange); closest on corpus character; qrel metrics only, no reranker-vs-LLM-judge
  protocol, no images.
- **ViDoRe v1/v2** (arXiv:2505.17166) — visual document retrieval on real PDF page images;
  retrieval only (ColPali-style), no reranking comparison.
- **Cross-encoders vs LLM rerankers** (arXiv:2403.10407) — academic head-to-head on standard
  IR collections; no decision-model, no real-PDF/figures corpus, no judge.
- **LLM-as-judge practice** — Zheng et al. (arXiv:2306.05685, MT-Bench): pairwise comparison,
  position-swap correction, win/tie/loss, agreement stats. Self-preference bias (arXiv:2404.13076)
  is a known risk when a model judges outputs from its own family.

No existing public benchmark combines: decision-model reranker vs cross-encoder reranker, on
big real PDFs including figures, scored by both gold labels and an LLM judge with
swap-correction. This repo is that benchmark.

## 3. Arms under comparison

All arms receive the **identical candidate set** (top-K=30 chunks per query from the same
fixed embedder), isolating reranking ability.

| # | Arm | Mechanism | Calls / query |
|---|-----|-----------|---------------|
| 1 | `embed-only` | embedding cosine order, no rerank (baseline) | 0 |
| 2 | `nemotron-vl` | `POST https://openrouter.ai/api/v1/rerank`, model `nvidia/llama-nemotron-rerank-vl-1b-v2` (see §9 for exact slug verification), query + documents; relevance scores sorted | 1 |
| 3 | `jev-listwise` | one SystemOne call: `choice` question, 30 candidates as `criteria` options, rank by per-option `probabilities` | 1 |
| 4 | `jev-pointwise` | TypeSafe cookbook pattern: one `noul` question per query-candidate pair ("does this passage answer the query?"), fanned out concurrently; rank by noul probability | 30 concurrent |

Jev call shape (from JevCity's working client, `src/server/jev.ts`):
`POST https://api.typesafe.ai/v1/systemone`, Bearer `TYPESAFE_API_KEY`,
body `{state, model: "jev-latest", questions}`; answers keyed by question name carrying
`choice`, `confidence`, `noul`, `probabilities`. Constraints: choice questions max 255
options; 64k tokens/request (32k for state + longest question); text-only; $42/B input,
output free.

## 4. Corpus

License-clean, big, diverse — text committed to the repo only from public-domain or
CC-BY/CC-BY-SA sources; PDFs themselves are **not** committed (download scripts are).

| Source | Type | License | Why |
|---|---|---|---|
| Project Gutenberg: Moby Dick | long prose novel | public domain | long narrative, many paraphrase targets |
| US Constitution + one long SCOTUS opinion | legal | public domain | precise legal wording, structure-heavy |
| OpenStax University Physics (selected chapters) | physics textbook, many figures | CC-BY 4.0 | figure/table-heavy for the multimodal track |
| OpenStax Calculus or Precalculus (selected chapters) | math textbook | CC-BY 4.0 | formula-dense text, extraction is lossy — deliberately part of the test |
| Dive into Deep Learning (d2l.ai, selected chapters) | ML textbook | CC-BY-SA 4.0 | technical prose + figures |
| Wikipedia featured articles (10–20, via API) | encyclopedic | CC-BY-SA | topic diversity, entity-heavy queries |

Target: **8–12 documents, 1,500–3,000 chunks**, balanced across sources. Extraction via
`pypdfium2` (permissive, does text extraction + page rendering in one dependency). Every
chunk keeps `{doc_id, page_start, page_end}` metadata for the image track.

## 5. Pipeline

1. **Ingest** — download sources (script + manifest + checksums committed, files cached
   locally outside git), extract text per page, chunk to ~500 tokens with ~50-token overlap,
   never crossing a page boundary backwards (chunk maps to 1..n pages).
2. **Embed** — Ollama, `nomic-embed-text` (768-dim; needs a one-time `ollama pull` — no
   embedding model currently installed). Index persisted to disk (vectors + chunk text);
   brute-force cosine is fine at 3k chunks — no vector DB dependency.
3. **Retrieve** — per query, embed, take top-K=30 cosine. This fixed candidate set is the
   input to every arm.
4. **Rerank** — each of the 4 arms reranks the same 30 candidates (see §3).
5. **Score** — gold-label metrics + LLM judge (§6, §7), latency/token/cost capture per call.

## 6. Dataset

**150–250 handmade queries**, authored by me from the corpus (each query written while
looking at a specific chunk, then the chunk is un-seen during runs). Each entry:

```json
{"id": "q042", "query": "...", "query_type": "paraphrase",
 "gold_chunk_ids": ["d2l.ch3.c017"], "source_doc": "d2l", "notes": "..."}
```

Query-type taxonomy (counts roughly balanced, negatives smaller):

1. `verbatim` — answer stated near-verbatim in one chunk
2. `paraphrase` — answer present, substantially different wording
3. `reasoning` — answer requires combining two chunks (gold = both), BRIGHT-style
4. `negative` — topically related but unanswerable from the corpus (gold = ∅; tests
   discrimination; excluded from recall/MRR/nDCG, included in judging where "both bad"
   ties are informative)
5. `structural` — answer lives in a footnote, caption, or small subsection
6. `graphical` — answer lives in a figure or table; page image goes to the VL reranker,
   extracted text/captions to Jev (expected VL win — hypothesis under test, honestly
   framed as a modality asymmetry, not a fairness failure)

At the dataset gate (§10) you review the taxonomy, the corpus list, and 10 sample queries
before I create the full set.

## 7. Judging protocol

- **Primary judge: Jev** (per your call). Blind pairwise: given the query, the top-5 of
  ranking A and the top-5 of ranking B, judge returns {A better, B better, tie}. Arm labels
  hidden; A/B assignment randomized per pair.
- **Position-swap:** every pair judged twice (A-first and B-first). Swap-flip rate is
  reported as a judge-reliability metric; headline verdicts aggregate both orders.
- **Bias spot-check: Gemini Flash** (`GEMINI_API_KEY`, `gemini-flash-latest`) re-judges a
  ~50-pair subset under the identical blind+swap protocol; report percent agreement and
  Cohen's kappa between Jev-judge and Gemini-judge. This turns "Jev judging itself" from a
  hand-wave into a measured quantity.
- Pairing scheme: jev-listwise vs jev-pointwise, nemotron-vl vs each Jev shape, and each
  arm vs embed-only (the "does reranking help at all" pairings).

## 8. Metrics and outputs

Per arm, overall and **per query type**:

- **Gold metrics:** recall@1/@5/@10, MRR, nDCG@10 (negatives excluded; see §6). Relevance is
  binary: for `reasoning` queries every gold chunk counts as relevant (no graded qrels).
- **Judge metrics:** win/tie/loss per pairing, swap-flip rate, judge-agreement on the
  spot-check subset.
- **Efficiency:** latency p50/p95 per query (jev-pointwise measured as fan-out wall time,
  bounded concurrency), input tokens, calls, cost per 1k queries.
- **Modality slice:** graphical-query subset scored separately (nemotron-vl sees page
  images at 150 DPI; text arms see extracted text/captions).

Outputs: committed JSON run artifacts (including raw judge verdicts for auditability),
README results tables, and a written verdict: *is Jev a reranker, and which shape?*

## 9. Repo, stack, reproducibility

- **Repo:** public `rerank-bench` under `shing1Sks`. Layout:
  `docs/specs/` (this doc), `src/rerankbench/{ingest,embed,retrieve,rerankers/{embed_only,nemotron,jev_listwise,jev_pointwise},judge,run,report}.py`,
  `data/{corpus_manifest,chunks,dataset}/`, `results/`, `README.md`, `CHANGELOG.md`.
- **Stack:** Python 3.12 + `uv`, `pypdfium2`, `requests`/`httpx`, `ollama` for embeddings.
  No server, no frontend — scripts and committed artifacts.
- **Keys:** local `.env` (gitignored) with `TYPESAFE_API_KEY`, `OPENROUTER_RERANK_KEY`,
  `GEMINI_API_KEY` copied from the JevCity `.env`; `.env.example` committed, never real keys.
- **Reproducibility:** pinned `uv.lock`, recorded model versions (embedder, `jev-*` version
  string returned by the API, Nemotron slug), fixed seeds, cached raw API responses committed
  where size permits.
- **Slug contingency:** OpenRouter's free-variant slug for the Nemotron model is unverified.
  At first live contact we try `nvidia/llama-nemotron-rerank-vl-1b-v2` and its `:free`
  variant; if neither is servable we fall back to NVIDIA's NIM `POST ai.api.nvidia.com/v1/ranking`
  (text reranker; the multimodal track then degrades and I flag it to you at that point).

## 10. Phases and your gates

| Phase | Work | Gate |
|---|---|---|
| P0 | this spec | your review, now |
| P1 | scaffold repo locally, `gh repo create` + initial push | your explicit go (contract wall) |
| P2 | corpus download scripts, ingest, embed (incl. one-time `ollama pull nomic-embed-text`, tiny live sanity calls to both APIs to verify keys/slugs) | none — local, reversible |
| P3 | dataset curation (150–250 queries) | **gate:** taxonomy + corpus list + 10 samples before the full set |
| P4 | adapters, smoke run (10 queries), full run | **gate:** before the full run |
| P5 | results, README, verdict write-up | **gate:** before preserving/publishing; every push separately approved |

## 11. Cost estimate

- Jev: listwise ~4k input tokens/call, pointwise ~400/call → 200 queries ≈ **$0.10–0.15**.
  Judging adds ~$0.05. Output tokens free.
- Nemotron on OpenRouter: $0 if the free variant is live; the run fits inside normal rate
  limits with backoff.
- Gemini spot-check: pennies.
- Total: **well under $1**.

## 12. Risks and mitigations

- **OpenRouter free slug unavailable** → NIM fallback (§9); multimodal loss flagged at the gate.
- **Jev listwise 32k state limit** → 30 × ~500-token chunks ≈ 15k tokens, fits; if a source
  yields longer chunks, listwise truncates chunk text to 800 tokens with truncation recorded.
- **Position bias / judge noise** → swap-correction + flip-rate reporting (§7).
- **Math PDF extraction mangling equations** → deliberate; corpus character is part of the
  test, and the analysis will say so explicitly rather than hide it.
- **Free-tier rate limits (either API)** → exponential backoff, run resumability from cached
  responses, off-peak execution.
- **Self-preference bias** → judged objects are rankings, not Jev-generated text (lower risk
  than the classic case), plus the measured Gemini agreement spot-check.

## 13. Success criteria

1. Public repo with dataset, code, run artifacts, and README results tables.
2. A defensible verdict: quality (gold + judge, per query type), latency, cost — for all
   four arms, with judge-agreement evidence.
3. Per-type insights: where each arm wins (verbatim vs reasoning vs graphical).
4. The multimodal question answered honestly (VL reranker vs text-only arms on figure queries).
