# rerank-bench

Benchmarking Jev (TypeSafe SystemOne) as a reranker against NVIDIA's Nemotron VL
cross-encoder reranker (via OpenRouter), on real long documents — books, law, math,
physics, encyclopedic text — scored by gold IR metrics and blind, swap-corrected
LLM-as-judge pairwise comparison.

## The question

Decision-style models such as Jev are often pitched as rerankers, but the claim is
rarely measured. On a corpus of 12 real documents (a novel, the US Constitution, a
SCOTUS opinion, OpenStax physics and calculus, Dive into Deep Learning, six Wikipedia
articles — 1,992 chunks), how do Jev (two call shapes) and NVIDIA's Nemotron
cross-encoder reranker compare on ranking quality, latency, and cost — and does
reranking beat the embedding order at all?

## Method

**Shared retrieval.** All four arms rerank the same candidate set per query:
nomic-embed-text (Ollama, local, 768-dim) embeds every chunk once; for each query the
top 30 chunks by cosine similarity become the candidates. Differences between arms are
therefore pure reranking differences, not retrieval differences.

**Arms.**

| Arm | What it does |
|---|---|
| `embed_only` | The cosine order itself — the baseline every reranker must beat. |
| `nemotron_vl` | NVIDIA Llama Nemotron rerank 1B v2 via OpenRouter's rerank endpoint; graphical-structure queries attach the gold chunk's page image (rendered at 150 dpi, DPI-ladder fallback to 110/80 on oversized payloads). |
| `jev_listwise` | One SystemOne `choice` call: all 30 candidates (truncated to 800 tokens each) in the state, the model returns per-candidate probabilities, ranked by confidence. |
| `jev_pointwise` | 30 parallel SystemOne `noul` calls, one per candidate passage ("Does this passage answer the query?"), sorted by score. |

**Dataset.** 175 handmade queries over the corpus, six types: 40 verbatim, 40
paraphrase, 30 reasoning (gold = two chunks that must be combined), 20 negative
(topically related but unanswerable — gold is empty, and empty gold is scored as
*unknown*, never zero), 20 structural (footnotes, captions, small subsections),
25 graphical (figures and tables — the Nemotron image payload's home turf). Queries
and curation notes are committed (`data/dataset/`); authoring is reproducible from
`scripts/build_dataset.py`.

**Metrics.** recall@1/5/10, MRR, nDCG@10 over gold-bearing queries (negatives
excluded from these aggregates by construction, not scored as zero); latency p50/p95;
cost per 1k queries. Nemotron failures degrade to candidate order with an error flag
rather than crashing a run, and error counts are reported per arm.

**Judging.** Gold metrics reward matching a specific chunk; they cannot say which
*ranking reads better*. So a second protocol judges every arm pairing blind: two
rankings labeled A and B (no arm names in the prompt), Jev picks the better one for
each query — in both positions, with disagreement demoted to TIE and counted as a
flip (swap consistency is reported). Gemini Flash spot-checks 50 queries per pairing
in a single order, with Cohen's kappa against the Jev verdicts as an agreement score.
Both judges see the same top-5 passages per ranking.

## Results

Full run (`results/run1`): 175 queries x 4 arms, 0 errors in any arm, total cost
$0.26. Quality metrics cover the 155 gold-bearing queries per arm (620 of 700 rows);
the 20 negative queries are excluded there and appear only in judging, where
"both rankings bad" ties are informative.

| arm | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms | $/1k queries |
|---|---|---|---|---|---|---|---|---|
| embed_only | 0.532 | 0.842 | 0.913 | 0.716 | 0.750 | 0 | 0 | 0 |
| nemotron_vl | **0.797** | **0.923** | 0.942 | **0.918** | **0.905** | 1935 | 18609 | 0 |
| jev_listwise | 0.632 | 0.923 | 0.948 | 0.817 | 0.837 | 982 | 1421 | 0.58 |
| jev_pointwise | 0.597 | 0.890 | 0.932 | 0.781 | 0.805 | 7687 | 9320 | 0.92 |

Per query type (recall@1 / nDCG@10; full table in `results/run1/report.md`):

| query type | embed_only | nemotron_vl | jev_listwise | jev_pointwise |
|---|---|---|---|---|
| verbatim (40) | 0.725 / 0.837 | **0.950 / 0.950** | 0.750 / 0.876 | 0.750 / 0.876 |
| paraphrase (40) | 0.600 / 0.782 | **0.850 / 0.921** | 0.650 / 0.845 | 0.575 / 0.816 |
| reasoning (30) | 0.250 / 0.623 | **0.417 / 0.781** | 0.300 / 0.717 | 0.250 / 0.612 |
| structural (20) | 0.550 / 0.732 | **0.850 / 0.913** | 0.650 / 0.827 | 0.750 / 0.870 |
| graphical (25) | 0.440 / 0.724 | **0.880 / 0.948** | 0.800 / 0.915 | 0.680 / 0.851 |

Two reads jump out: **reasoning queries are the open problem** — combining two chunks
into one answer defeats every reranker (best recall@1 is 0.417), while everything
else sits at 0.6–0.95; and **the graphical slice is where the multimodal payload
pays** — Nemotron with page images reaches 0.880 where the text-only embed order
manages 0.440, and even the text-only Jev listwise arm beats embed by +0.36, so the
figure-adjacent chunks were retrievable but badly ordered without reranking.

Blind pairwise judging (Jev judge, both orders, disagreement demoted to TIE):

| pairing | A wins | B wins | ties | flips | swap consistency |
|---|---|---|---|---|---|
| jev_listwise > jev_pointwise | 27 | 5 | 143 | 81 | 0.54 |
| nemotron_vl > jev_listwise | 7 | 35 | 133 | 71 | 0.59 |
| nemotron_vl > jev_pointwise | 20 | 17 | 138 | 64 | 0.63 |
| nemotron_vl > embed_only | 55 | 7 | 113 | 66 | 0.62 |
| jev_listwise > embed_only | 69 | 3 | 103 | 51 | 0.71 |
| jev_pointwise > embed_only | 55 | 9 | 111 | 69 | 0.61 |

Gemini Flash spot check (50 queries per pairing, single order; Cohen's kappa vs the
Jev verdicts; flagged = unparsable or transport-failed answers, counted as TIE):

| pairing | kappa | n | flagged |
|---|---|---|---|
| jev_listwise > jev_pointwise | 0.22 | 50 | 3 |
| nemotron_vl > jev_listwise | 0.17 | 50 | 4 |
| nemotron_vl > jev_pointwise | 0.23 | 50 | 2 |
| nemotron_vl > embed_only | 0.30 | 50 | 0 |
| jev_listwise > embed_only | 0.37 | 50 | 1 |
| jev_pointwise > embed_only | 0.20 | 50 | 4 |

### Verdict

1. **Reranking beats the embedding order — this is the benchmark's most solid
   result.** Nemotron lifts recall@1 from 0.532 to 0.797; Jev listwise lifts it to
   0.632. The blind judge independently agrees by landslides (55–7 and 69–3), and
   those are the pairings where the two judges also agree best (kappa 0.30 / 0.37).
   If you currently serve embedding order directly, a reranker in front of it is
   worth the latency.
2. **On pure ranking quality, Nemotron VL wins decisively.** +0.165 recall@1 and
   +0.10 MRR over the best Jev arm. The cost is tail latency — p95 of 18.6s against
   listwise's 1.4s (free-endpoint queueing; zero errors, but a burst of traffic will
   feel that tail).
3. **Jev listwise is the credible decision-model shape; pointwise is not.** Listwise:
   ~1s median, $0.58/1k, beats embed_only on every metric. Pointwise: 30 calls per
   query buy *worse* quality at 8x the latency and 1.6x the cost. Per-query fan-out
   is the wrong call shape for this job — one structured listwise call is strictly
   better in this setup.
4. **The judge layer is weak evidence here, and we say so.** Kappa between Jev and
   Gemini spot checks is 0.17–0.37 (slight to fair). Where the judge agrees with the
   gold metrics (anything vs embed_only), it corroborates; where it contradicts them
   (it prefers jev_listwise over nemotron 35–7, against gold's opposite verdict),
   the kappa says do not trust it. Ties dominate every pairing (58–82%), a structural
   consequence of all arms reranking the same 30 candidates — top-5s overlap heavily.
   Gold metrics, not judge verdicts, carry this benchmark's conclusions; the judging
   protocol is reproducible and honest about its own agreement score.
5. **Per query type: reasoning is the open problem, graphical is the payoff.** No
   arm's recall@1 exceeds 0.417 on reasoning queries (answer requires combining two
   chunks) — listwise ranking over isolated candidates may be the wrong shape for
   multi-hop answers, not just a weak reranker. On graphical queries the Nemotron
   page-image payload wins (0.880 vs 0.440 embed-only), and Jev listwise's +0.36 over
   embed shows the figures were retrievable but badly ordered — a reranking failure,
   not a retrieval one.


## Reproduce

Prerequisites: Python 3.12 with [uv](https://docs.astral.sh/uv/), Ollama with
`nomic-embed-text` pulled, and API keys (TypeSafe/Jev, OpenRouter, Gemini) in a
repo-root `.env` — copy `.env.example` and fill in real values; the `.env` is
gitignored and must never be committed.

```bash
uv sync                                          # install dependencies
uv run pytest                                    # tests, no network needed
uv run python -m rerankbench.ingest              # fetch + chunk corpus (cached in data/cache/corpus)
uv run python -m rerankbench.embed               # build the vector index (needs Ollama running)
uv run python -m rerankbench.dataset_stats       # validate the committed dataset
uv run python -m rerankbench.run_bench --tag run1    # full 4-arm run (cache-aware, resumable)
uv run python -m rerankbench.judges --run results/run1 --judge jev      # position-swapped pairwise pass
uv run python -m rerankbench.judges --run results/run1 --judge gemini --subset 50   # spot check
uv run python -m rerankbench.report --run results/run1                  # rebuild report.md
```

`run.json` and the judge artifacts are committed under `results/run1/`. The bench and
judge stages call live APIs (cached per query, so reruns only pay for cache misses and
previously failed rows, which are retried); `report.md` itself rebuilds from those
artifacts with no API calls at all.

## Corpus and licenses

| Source | Used for | License |
|---|---|---|
| Moby Dick (Project Gutenberg) | long prose | Public domain |
| US Constitution (US National Archives transcript) | law, short reference | Public domain |
| Obergefell v. Hodges majority (Cornell LII) | case-law reasoning | Public domain |
| OpenStax University Physics vol 1 | physics, figures | CC-BY 4.0 |
| OpenStax Calculus vol 1 | math notation | CC-BY 4.0 |
| Dive into Deep Learning (d2l.ai) | ML prose + code | CC-BY-SA 4.0 |
| Wikipedia: Renaissance, Photosynthesis, Quantum mechanics, Plate tectonics, French Revolution, Great Barrier Reef | encyclopedic text | CC-BY-SA 4.0 |

OpenStax texts are used under CC-BY 4.0 (https://openstax.org/); Wikipedia articles
under CC-BY-SA 4.0 with attribution to their contributors
(https://en.wikipedia.org/wiki/&lt;Article&gt;); Dive into Deep Learning under CC-BY-SA 4.0
(https://d2l.ai). Chunk provenance (source URL + sha256 per document) is recorded in
`data/corpus_manifest.json`. No PDFs are committed; the ingest step re-fetches them.

## Related work

- **BEIR / MS MARCO / MTEB(r)** — standard IR/rerank substrates; short clean text,
  gold qrels, no LLM judge, no images.
- **BRIGHT** (arXiv:2407.12886) — reasoning-intensive retrieval over real long docs
  (math, law, code, StackExchange); closest on corpus character; qrel metrics only, no
  reranker-vs-LLM-judge protocol, no images.
- **ViDoRe v1/v2** (arXiv:2505.17166) — visual document retrieval on real PDF page
  images; retrieval only (ColPali-style), no reranking comparison.
- **Cross-encoders vs LLM rerankers** (arXiv:2403.10407) — academic head-to-head on
  standard IR collections; no decision-model, no real-PDF/figures corpus, no judge.
- **LLM-as-judge practice** — Zheng et al. (arXiv:2306.05685, MT-Bench): pairwise
  comparison, position-swap correction, win/tie/loss, agreement stats. Self-preference
  bias (arXiv:2404.13076) is a known risk when a model judges outputs from its own
  family — Jev judging Jev arms is the reason the Gemini spot check and kappa exist.

No existing public benchmark combines: decision-model reranker vs cross-encoder
reranker, on big real PDFs including figures, scored by both gold labels and an LLM
judge with position-swap correction.

## Reports

- `docs/reports/evaluation-method.md` — the full evaluation protocol: candidate
  sharing, arm call shapes, metric formulas and empty-gold semantics, judge design,
  fairness controls, reproducibility machinery, statistical caveats.
- `docs/reports/dataset-curation.md` — corpus construction, per-type authoring
  procedures with examples, negative-verification distractor log, balance tables,
  known limitations.
- `docs/reports/rerank-bench-paper.md` (+ `rerank-bench-paper.pdf`) — the
  technical report: the whole arc from wire verification and endpoint discovery
  through the smoke run to the final run1, all result tables, and the five
  findings. The PDF rebuilds from the markdown
  (`uv run --with markdown python scripts/build_paper_pdf.py`, then headless
  Chrome `--print-to-pdf`).
- `results/run1/report.md` — the raw per-run tables the above draw from.

## Status

Benchmark complete; results on `dev` awaiting merge to `main`. Methodology in
`docs/specs/2026-09-26-rerank-bench-design.md`, task breakdown in
`docs/plans/2026-09-26-rerank-bench-plan.md`, full artifacts in `results/run1/`
(run.json, verdicts.jsonl, judge_summary.json, spotcheck.jsonl, spotcheck_summary.json,
report.md — all regenerable from the committed data without new API calls).
