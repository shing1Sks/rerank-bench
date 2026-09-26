# Evaluation method

This document specifies exactly how rerank-bench scores rerankers: the candidate
protocol, the metric definitions (with formulas and edge-case semantics), the judge
protocol, the fairness controls, and the reproducibility machinery. Everything here is
implemented in `src/rerankbench/` and pinned by the test suite.

## 1. Shared candidate protocol

Every arm reranks the same candidates, so differences between arms are pure reranking
differences:

1. All 1,992 corpus chunks are embedded once with `nomic-embed-text` (768-dim) via a
   local Ollama server; vectors are L2-normalized and stored in `data/index.npz`.
2. For each of the 175 queries, the query vector is compared to every chunk by cosine
   similarity and the top 30 chunks (`top_k = 30`) become that query's candidate list.
3. The `embed_only` arm's ranking IS this cosine order — the baseline every reranker
   must beat. The other three arms receive `(query, candidates)` and return a full
   permutation of the candidate ids.

## 2. Arms

### 2.1 embed_only

Cosine order, unchanged. Zero API calls, zero cost, sub-millisecond. It measures how
good raw retrieval already is — every other arm's value is measured as lift over this.

### 2.2 nemotron_vl

NVIDIA Llama Nemotron rerank 1B v2 via OpenRouter's rerank endpoint
(`POST /api/v1/rerank`), body `{model, query, documents, top_n}`. Each candidate is
one document `{text}`; for graphical queries every candidate additionally carries
`{image, text}` where image is the candidate chunk's own PDF page rendered to PNG.

- Slug probe: the base slug has no endpoint on the benchmarking account (HTTP 404 at
  every payload size); the runner probes base then `:free` with a 2-document smoke
  call and uses the reachable variant. A miss degrades rather than crashes (below).
- Page images: rendered at 150 dpi, cached on disk; a PNG above 1.8 MB re-renders down
  the DPI ladder (150 -> 110 -> 80) before giving up and sending text-only (logged).

### 2.3 jev_listwise

One SystemOne `choice` call per query. The state contains the query and all 30
candidates (each truncated to 800 tokens, ~3/4 of the cap in words, with a
`truncated` flag set when cut). The single question offers every candidate id as a
criterion; the model returns per-candidate probabilities and the arm sorts by them.
Cost model: input tokens billed at $0.042/Mtok, output free.

### 2.4 jev_pointwise

30 parallel SystemOne `noul` calls per query (thread pool, 8 workers), one per
candidate: "Does this passage answer the query? Passage: ..." Each returns a 0..1
score; the arm sorts by score. Missing `noul` answers score 0.0 and set a
`missing_noul` flag. This arm exists to test the fan-out call shape against the
one-call listwise shape.

## 3. Failure and degradation semantics

- Wire errors are status-carrying (`JevError(message, status)`); both Jev and
  OpenRouter transports raise them on non-200 with the response body excerpt.
- 429 and 529 are retried with exponential backoff (`min(30, 2^n)` seconds), 5
  attempts, then the error is raised.
- Every other status fails fast (no retry): a 400 is a wire-format bug, retrying
  cannot fix it.
- Nemotron total failure degrades: the row records candidate order plus
  `flags["error"]` — the run continues, other arms unaffected. Jev-arm errors
  propagate and stop the run (fail fast by design); the per-query cache bounds the
  lost work, and resume completes the run.

## 4. Metrics

Let `gold` be the query's gold chunk-id set and `ranked` the arm's ranking.
A **negative** query has `gold = {}` (empty by schema — unanswerable from this
corpus).

### 4.1 Gold metrics (quality)

- **recall@k** = |gold intersect top-k(ranked)| / |gold|
- **MRR@10** = mean over queries of 1/rank of the first gold hit within the top 10
  (0 if no hit in the top 10)
- **nDCG@10** with binary gains:
  DCG@10 = sum over positions i = 1..10 of rel_i / log2(i + 1), where rel_i = 1 iff
  ranked[i-1] is in gold; nDCG@10 = DCG@10 / IDCG@10 (ideal ordering of the same
  gold set). Two-chunk gold (reasoning queries) therefore requires both chunks high
  for a perfect score.
- **Empty-gold semantics**: for negative queries every gold metric returns `None` —
  never 0. Aggregates (per-arm and per-type means) exclude these rows by
  construction; the report's scope line states exactly how many rows entered
  (`quality metrics over 620 of 700 rows ...`). This matters: scoring negatives as
  zero would silently punish arms for a query where "no good answer exists" is the
  correct behavior.

### 4.2 Latency and cost

- **Latency** is wall-clock per query inside `rerank()` (ms), reported as p50/p95 by
  the nearest-rank method (sort, take index ceil(p*n)-1). Includes retries and
  (for nemotron) image rendering is cached on disk after first use, so image cost
  amortizes across a run.
- **Cost per 1k queries** = (sum of row costs) / n * 1000. Jev arms bill input
  tokens only ($0.042/Mtok); the nemotron `:free` endpoint and local embeddings
  bill 0.

### 4.3 Judging metrics

- **Wins/losses/ties** per pairing from the blind judge (section 5).
- **Flips** = judgments where the two orders disagreed (demoted to TIE).
- **Swap consistency** = 1 - flips/n.
- **Cohen's kappa** (3-class over A/B/TIE) between the Jev verdicts and the Gemini
  spot-check verdicts: kappa = (p_o - p_e) / (1 - p_e), where p_o is observed
  agreement and p_e is chance agreement from the two raters' label marginals.

## 5. Judge protocol

Gold metrics reward matching a labeled chunk; they cannot say which ranking *reads*
better, and they say nothing about negatives. The judge layer adds that view:

1. **Blind state**: for each pairing (arm_a, arm_b) and query, the prompt shows the
   query plus "Ranking A" and "Ranking B" — top-5 chunks each, formatted
   `[i] chunk_id: text`. Arm names never appear.
2. **Position swap (Jev judge)**: Jev answers the same question twice per pair, with
   the orders exchanged. The second answer is inverted (A<->B, TIE fixed); agreement
   yields the verdict, disagreement demotes to TIE and counts a flip. This is the
   standard position-bias correction from LLM-as-judge practice.
3. **Gemini spot check**: Gemini Flash judges one order per pair for 50 sampled
   queries per pairing (seeded sample, seed 13). Its first reply line must be exactly
   A, B, or TIE; anything else (or a transport failure) is a flagged TIE. Cohen's
   kappa against the stored Jev verdicts is the agreement score — the benchmark's
   own honesty check on its judge, and specifically on the self-preference risk of a
   Jev model judging Jev-produced rankings.
4. **Pairings judged**: all 6 pairs over 4 arms; a pairing is judged only on queries
   both arms ranked (full runs: all 175).

Known limitation, stated rather than hidden: ties dominate (58-82% per pairing)
because all arms rerank the same 30 candidates — top-5s overlap heavily. Where the
judge contradicts gold metrics, gold wins in this benchmark's conclusions (see the
results report); kappa 0.17-0.37 says the judge layer is weak evidence.

## 6. Fairness controls (what makes comparisons clean)

- Identical candidate sets per query across arms (section 1) — no retrieval noise.
- Identical truncation rules for Jev arms (same 800-token cap helper).
- Blind labels; no arm identity in any judge prompt.
- Degradation never silently zeroes an arm: error rows are flagged and counted in
  the report's errors column.
- One dataset, one candidate index, one run (`run1`); all artifacts committed.

## 7. Reproducibility machinery

- **Per-query cache**: raw rerank results cached at
  `results/{tag}/cache/{arm}/{query_id}.json`; reruns skip cache hits. Cached rows
  carrying `flags["error"]` are re-fetched on resume — a rate-limit failure never
  becomes permanent.
- **Slug probing** before the first nemotron call (section 2.2).
- **Artifact chain**: `run.json` (meta + rows) -> `verdicts.jsonl` +
  `judge_summary.json` -> `spotcheck.jsonl` + `spotcheck_summary.json` ->
  `report.md` -> README tables. Each stage rebuilds from the previous stage's files
  (`python -m rerankbench.report --run results/run1` needs no API access).
- **Tests**: 75 tests pin all of the above (metric edge cases, empty-gold None
  semantics, swap-flip accounting, retry ladders, cache-hit behavior, error-row
  retry, DPI ladder, blind state construction).

## 8. Statistical caveats (read before quoting numbers)

- Single run per configuration; no variance estimates. With n = 155 scorable
  queries, a recall@1 difference of ~0.05 is within noise — the headline gaps
  (0.10-0.27) are not.
- Per-type cells are smaller (n = 20-40); treat per-type rankings as suggestive,
  except where gaps are large (graphical: 0.44 vs 0.88).
- The judge is one model (Jev) with a weak-agreement spot check; judge verdicts are
  reported but do not carry conclusions on their own.
- Latency reflects one network path (residential, US endpoints) and, for nemotron,
  a free endpoint's queueing; treat absolute numbers as indicative, relative
  ordering as robust.
