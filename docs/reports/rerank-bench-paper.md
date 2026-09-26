# Is a Decision Model a Reranker? A Four-Arm Benchmark on Real Documents

**rerank-bench technical report — 2026-09-27**

> This report is the narrative arc of the whole benchmark: why it exists, how it was
> built, what happened at every stage from the first 10-query smoke to the final
> 175-query run, and what the combined results say. Companion documents carry the
> full depth: `evaluation-method.md` (protocol and formulas),
> `dataset-curation.md` (corpus and queries), `results/run1/report.md` (raw tables).

## Abstract

Decision-style models (TypeSafe's SystemOne family, "Jev" here) are increasingly
pitched as rerankers, but the claim is rarely measured against a dedicated
cross-encoder. We built rerank-bench: 175 handmade queries over 1,992 chunks from 12
real documents (a novel, the US Constitution, a SCOTUS opinion, two OpenStax STEM
textbooks, Dive into Deep Learning, six Wikipedia articles), with six query types
including two-chunk reasoning, unanswerable negatives, document-furniture
structural queries, and figure-content graphical queries. Four arms rerank the same
top-30 nomic-embed-text candidates: the embedding order itself (baseline), NVIDIA's
Llama Nemotron rerank VL 1B v2 (via OpenRouter, with page-image payloads on
graphical queries), Jev in a one-call listwise shape, and Jev in a 30-call pointwise
fan-out. On gold metrics (recall@1/5/10, MRR@10, nDCG@10 over 155 gold-bearing
queries) reranking beats the embedding order decisively, Nemotron wins overall
quality (recall@1 0.797 vs 0.532 baseline), Jev listwise is the best
decision-model shape (0.632 at ~1s median and $0.58/1k queries), and pointwise is
dominated on every axis. Per type, all arms collapse on two-chunk reasoning queries
(best recall@1 0.417) while page-image payloads pay off on graphical queries
(0.880 vs 0.440 baseline). A position-swapped blind judge (Jev) with a Gemini Flash
spot check (Cohen's kappa 0.17-0.37) corroborates the gold metrics where the two
agree and is explicitly reported as too weak to overturn them where they disagree.
We conclude: reranking is worth it, a dedicated cross-encoder still leads on
quality, the listwise shape is the only sensible decision-model call pattern here,
and multi-hop synthesis is the open problem no reranker in this study solves.

## 1. Introduction

Production retrieval stacks increasingly bolt an LLM in front of a vector search to
reorder its candidates. Vendors of decision-style models — models whose native
output is a structured choice with probabilities rather than free text — pitch them
for exactly this job. Two claims deserve measurement:

1. Does LLM reranking beat the embedding order it sits on, by enough to justify its
   latency and cost?
2. Does a decision model (with a ranking-shaped prompt) match a purpose-built
   cross-encoder reranker?

Public benchmarks cover adjacent ground but not this comparison: BEIR/MS MARCO/
MTEB(r) score rerankers on short clean text with gold qrels and no LLM judge; BRIGHT
brings reasoning-intensive long documents but stops at retrieval metrics; ViDoRe
scores visual document retrieval but not reranking; the cross-encoder-vs-LLM-reranker
literature compares on standard IR collections without decision models, real PDFs,
or a judge protocol. rerank-bench fills that gap: decision model vs cross-encoder,
on big real documents including figures, scored by gold IR metrics and a
swap-corrected blind judge with an inter-judge agreement statistic.

## 2. Benchmark design

The design (spec: `docs/specs/2026-09-26-rerank-bench-design.md`) fixes four arms
behind one protocol — `rerank(query, candidates) -> (ranked_ids, scores, latency,
cost, flags)` — so every arm sees identical inputs:

- **embed_only** — the cosine order of the top-30 candidates. The baseline; every
  other arm is measured as lift over it.
- **nemotron_vl** — NVIDIA's rerank cross-encoder via OpenRouter. On graphical
  queries each candidate also carries its PDF page rendered to PNG (150 dpi, with a
  DPI fallback ladder for oversized payloads).
- **jev_listwise** — one SystemOne `choice` call: all 30 candidates in the state
  (800-token cap each, truncation flagged), ranked by returned probabilities.
- **jev_pointwise** — 30 parallel SystemOne `noul` calls ("does this passage answer
  the query?"), ranked by score. Tests the fan-out shape against the one-call shape.

Quality is gold IR metrics with strict empty-gold semantics (negative queries score
None, never zero, and are excluded from aggregates by construction). The judge layer
is blind (arm names never appear), position-swapped (disagreement between orders
demotes to TIE and counts a flip), and self-checked (Gemini Flash re-judges 50
queries per pairing; Cohen's kappa reports how much the judge layer can be trusted —
the designed counterweight to a Jev model judging Jev-produced rankings).

## 3. Dataset

175 queries, six types, over a 12-source corpus (licenses: 3 public domain, 2
CC-BY 4.0, 7 CC-BY-SA 4.0; provenance manifest committed):

| type | n | tests | example (abbreviated) |
|---|---|---|---|
| verbatim | 40 | lexical match | which passage contains the line "Who-e debel you..." |
| paraphrase | 40 | semantic match | where does the text describe the whaler's weathered look... |
| reasoning | 30 | two-chunk synthesis | why does the narrator hesitate ... and what do the part-owners say |
| negative | 20 | discrimination | which amendment prohibits quartering soldiers... (topic absent) |
| structural | 20 | document furniture | on what page does the Transformer treatment begin? |
| graphical | 25 | figure/table content | which figure shows a whirlpool-in-a-tank image... |

Negatives passed a two-pass verification (keyword co-occurrence scan, then
domain-term scan with manual review of every hit); decoy hits were judged and
kept as near-miss distractors where they make the query harder. Full procedure,
balance tables, and limitations: `dataset-curation.md`.

## 4. Execution history (first run to last)

The benchmark was built test-first (75 tests) and executed in stages; each stage's
finding shaped the next.

**Stage 1 — wire verification.** Both vendors' formats were verified live before
any arm was coded (SystemOne `choice`/`noul` shapes; OpenRouter's rerank endpoint).
This surfaced the first real finding: the pointwise wire format needs
`instructions`, not `text`, on a `noul` question — a wrong guess here fails with an
opaque 400.

**Stage 2 — endpoint discovery.** The base Nemotron slug returned 404 "No endpoints
found" at every payload size on this account; the `:free` variant worked fully
(verified to 62k-char payloads). The runner now probes both slugs before the first
call. This finding saved the nemotron arm from silently recording garbage: the first
smoke run had degraded all nemotron rows to candidate order before the probe was
added.

**Stage 3 — 10-query smoke.** All four arms plus a first judge pass, on a 10-query
slice: nemotron 0.900 recall@1 (vs embed 0.500), listwise 0.800, pointwise 0.600;
graphical image payloads accepted with zero flags. The smoke also shook out three
judge-layer bugs (chunk path, dataset key, partial-run handling) at 1/17th the cost
of finding them in the full run.

**Stage 4 — full run (run1).** 175 queries x 4 arms in 45 minutes, 0 errors, $0.26
total (budget: $0.15-0.25 projected, $1.00 abort). Judge pass: 33 minutes, all six
pairings at n=175. Gemini spot check: 50 per pairing. One tooling crash (a missing
default transport in the spot-check entry point) was fixed test-first and the stage
rerun alone against persisted artifacts — the per-stage artifact design meant zero
recomputation.

**Stage 5 — whole-branch review.** An independent reviewer recomputed every headline
number from raw artifacts (exact match) and found four important defects — a dead
DPI-ladder branch, error rows cached as successes, missing per-type reporting, no
committed entry points for the judge/report stages — all fixed test-first before
publication; none changed run1's numbers.

## 5. Results

### 5.1 Overall (155 gold-bearing queries per arm)

| arm | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms | $/1k |
|---|---|---|---|---|---|---|---|---|
| embed_only | 0.532 | 0.842 | 0.913 | 0.716 | 0.750 | 0 | 0 | 0 |
| nemotron_vl | **0.797** | **0.923** | 0.942 | **0.918** | **0.905** | 1935 | 18609 | 0 |
| jev_listwise | 0.632 | 0.923 | **0.948** | 0.817 | 0.837 | **982** | 1421 | 0.58 |
| jev_pointwise | 0.597 | 0.890 | 0.932 | 0.781 | 0.805 | 7687 | 9320 | 0.92 |

### 5.2 Per query type (recall@1 / nDCG@10)

| type | embed_only | nemotron_vl | jev_listwise | jev_pointwise |
|---|---|---|---|---|
| verbatim (40) | 0.725 / 0.837 | **0.950** / 0.950 | 0.750 / 0.876 | 0.750 / 0.876 |
| paraphrase (40) | 0.600 / 0.782 | **0.850** / 0.921 | 0.650 / 0.845 | 0.575 / 0.816 |
| reasoning (30) | 0.250 / 0.623 | **0.417** / 0.781 | 0.300 / 0.717 | 0.250 / 0.612 |
| structural (20) | 0.550 / 0.732 | **0.850** / 0.913 | 0.650 / 0.827 | 0.750 / 0.870 |
| graphical (25) | 0.440 / 0.724 | **0.880** / 0.948 | 0.800 / 0.915 | 0.680 / 0.851 |

### 5.3 Blind pairwise judging (Jev judge, both orders)

| pairing | A wins | B wins | ties | flips | swap consistency |
|---|---|---|---|---|---|
| jev_listwise > jev_pointwise | 27 | 5 | 143 | 81 | 0.54 |
| nemotron_vl > jev_listwise | 7 | 35 | 133 | 71 | 0.59 |
| nemotron_vl > jev_pointwise | 20 | 17 | 138 | 64 | 0.63 |
| nemotron_vl > embed_only | 55 | 7 | 113 | 66 | 0.62 |
| jev_listwise > embed_only | 69 | 3 | 103 | 51 | 0.71 |
| jev_pointwise > embed_only | 55 | 9 | 111 | 69 | 0.61 |

### 5.4 Gemini Flash spot check

| pairing | kappa | n | flagged |
|---|---|---|---|
| jev_listwise > jev_pointwise | 0.22 | 50 | 3 |
| nemotron_vl > jev_listwise | 0.17 | 50 | 4 |
| nemotron_vl > jev_pointwise | 0.23 | 50 | 2 |
| nemotron_vl > embed_only | 0.30 | 50 | 0 |
| jev_listwise > embed_only | 0.37 | 50 | 1 |
| jev_pointwise > embed_only | 0.20 | 50 | 4 |

## 6. What the combined results say

**Finding 1 — reranking beats the embedding order; this is the benchmark's most
solid result.** Nemotron lifts recall@1 from 0.532 to 0.797 (+0.265), Jev listwise
to 0.632 (+0.100). The blind judge independently agrees by landslides (55-7 and
69-3) — and those are exactly the pairings where the two judges agree best (kappa
0.30, 0.37). Two independent instruments, same direction, large margins. If a
retrieval stack serves cosine order directly today, a reranker in front of it is
worth its latency.

**Finding 2 — the dedicated cross-encoder still leads on quality.** Nemotron's
margin over the best decision-model arm is +0.165 recall@1, +0.10 MRR@10, and it
leads every per-type cell. The price is the tail: p95 18.6s (free-endpoint
queueing) against listwise's 1.4s. Quality-critical offline work should take the
cross-encoder; latency-sensitive serving has a real trade to make.

**Finding 3 — between the two decision-model shapes, listwise dominates.** One
structured call beats 30 fan-out calls on quality (0.632 vs 0.597 recall@1), median
latency (982 vs 7687 ms), and cost ($0.58 vs $0.92 per 1k). Pointwise is not
slightly worse; it is worse on every axis while making 30x the calls. The
decision-model reranker story, where it works at all, is a listwise story.

**Finding 4 — the judge layer is weak evidence, and the benchmark says so.** Kappa
between the Jev judge and its Gemini spot check is 0.17-0.37 — slight to fair
agreement. Where judge and gold agree (both vs embed_only) the judge corroborates.
Where they contradict — the judge prefers jev_listwise over nemotron 35-7 while
gold says the opposite by a wide margin — the kappa says do not trust the judge,
and the benchmark reports it that way rather than headlining it. Ties dominate
(58-82%) because all arms rerank the same 30 candidates; this is structural, and
the swap-consistency column (0.54-0.71) plus the spot check quantify how noisy the
instrument is. The design lesson generalizes: an LLM-as-judge layer without an
agreement statistic is unfalsifiable.

**Finding 5 — reasoning queries are the open problem; graphical queries are where
multimodality pays.** Two-chunk synthesis collapses every arm (best recall@1
0.417, nDCG@10 0.781): reranking isolated candidates cannot assemble an answer
that spans two passages — a retrieval-then-synthesize architecture problem, not a
reranker-quality problem. Conversely the graphical slice shows the page-image
payload earning its bytes: Nemotron with images reaches 0.880 recall@1 against
0.440 for the text-only baseline — and the text-only Jev listwise arm's 0.800
shows the figure captions were retrievable but badly ordered, i.e. the baseline's
graphical failure is a ranking failure that any strong reranker largely fixes, with
the image payload adding the last increment.

## 7. Threats to validity

- **Single run, no variance estimates.** With n=155 the headline gaps (0.10-0.27)
  are far beyond noise; ~0.05 differences (per-type cells, n=20-40) are within it.
- **Self-preference in the judge.** A Jev model judged Jev-produced rankings; the
  Gemini spot check and kappa exist precisely to price this, and Finding 4 prices
  it: low.
- **One author's gold labels.** No inter-annotator agreement; gold errors would
  shift absolute numbers but hit all arms identically (shared candidates).
- **Endpoint specifics.** Nemotron ran on the `:free` variant (the base slug has no
  endpoint on this account); its latency includes free-tier queueing. Jev pricing
  ($0.042/Mtok input) is TypeSafe's current rate.
- **Corpus blind spots.** Mangled PDF math extraction, English-only, no code or
  handwritten sources.

## 8. Conclusion

Measured on real documents with handmade queries and a self-checking judge layer:
(1) rerank your candidates — the lift over embedding order is large and doubly
confirmed; (2) a purpose-built cross-encoder reranker still beats a decision model
prompted into reranking, by a wide margin, at the cost of a long latency tail;
(3) if you do use a decision model, use one listwise call — the pointwise fan-out
shape is dominated everywhere; (4) no reranker in this study solves two-chunk
reasoning, which points at synthesis rather than ranking as the next bottleneck;
and (5) page-image payloads measurably pay for themselves on figure-grounded
queries. The full artifact chain — corpus manifest, dataset literals, run rows,
judge verdicts, spot checks, reports — is committed, and every stage rebuilds from
the previous one.

## References

- BRIGHT: A Realistic Benchmark for Reasoning-Intensive Retrieval. arXiv:2407.12886.
- ViDoRe: A Benchmark for Multimodal Document Retrieval. arXiv:2505.17166.
- Liu et al. Comparing Cross-Encoders and LLM Rerankers. arXiv:2403.10407.
- Zheng et al. Judging LLM-as-a-Judge with MT-Bench. arXiv:2306.05685.
- Self-Preference Bias in LLM Judges. arXiv:2404.13076.
- BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation. arXiv:2104.08663.
- MTEB: Massive Text Embedding Benchmark. arXiv:2210.07316.
- Nussbaum et al. Nomic Embed: Training a General Contextual Embedding with
  Unsupervised Contrastive Pretraining. nomic.ai, 2024.
- NVIDIA. Llama Nemotron Rerank VL 1B v2. openrouter.ai model documentation.
