# Changelog

Decision log — updated at good-to-go gates per the operating contract, not per edit.

## 2026-09-26 — P1: scaffold and public repo

- Repo created at `shing1Sks/rerank-bench`; scaffold committed on `main`, all work on `dev` (dev-to-main flow per operating contract).
- Config loads `.env` from repo root; `.env` gitignored, `.env.example` committed with empty values. Keys never leave `.env`.

## 2026-09-26 — P2: corpus

- Ingest caps: 250 pages max per PDF, 600 chunks max per doc — full d2l/OpenStax PDFs would blow past the spec's 1,500–3,000 chunk target otherwise.
- Scotus source moved to Cornell LII (`law.cornell.edu/supremecourt/text/14-556`) after justia (403), supremecourt.gov slip (404), and courtlistener (202 challenge) all failed.
- Wikipedia/403-hostile fetches fall back to a curl subprocess on TLS-fingerprint blocks.
- Fetch cache is first-class: `.url` sidecars record provenance; re-runs are resume-safe.
- Corpus: 12 sources, 1,992 chunks (spec aspired to 14 documents; 12 is what passed licensing and fetch checks). No PDFs committed; manifest carries URL + sha256 per source.

## 2026-09-26 — P3: dataset (GATE-P3 approved)

- 175 queries authored as committed Python literals (`scripts/build_dataset.py`) so provenance is in reviewable code, not a notebook.
- Negative queries verified topic-absent in two passes (`scripts/check_negatives.py`); distractor hits reviewed manually, rationale in `data/dataset/curation_notes.md`.
- Final mix: 40 verbatim / 40 paraphrase / 30 reasoning / 20 negative / 20 structural / 25 graphical.

## 2026-09-26 — P4: arms, metrics, judges, runner (GATE-P4 approved)

- Wire formats verified live against JevCity's proven client before coding: SystemOne `choice` + `noul` (`instructions` field, not `text` — a live 400 repro'd to this), OpenRouter rerank endpoint.
- Base Nemotron slug has no endpoint on this account (404 at any payload size); `:free` works fully. Runner probes both and passes the reachable one in — first smoke silently degraded to candidate order until this was found.
- Errors are status-carrying (`JevError`/`NemotronError` with `status`); retries on 429/529 only (5 attempts, capped backoff), everything else fails fast. Nemotron total failure degrades to candidate order with an error flag — a run never crashes on one arm.
- Graphical queries attach the gold chunk's page image to every Nemotron candidate (150 dpi, ladder to 110/80 if the payload is too big, disk-cached renders).
- Listwise truncates candidates to 800 tokens and flags truncation; pointwise fans out 30 `noul` calls on a thread pool.
- Negative queries score `None` on gold metrics (never zero) and are excluded from aggregates by construction.
- Jev judge runs both orders per pair; disagreement demotes to TIE and counts as a flip. Gemini spot check judges one order, unknown answers become flagged TIEs.
- Runner caches raw responses per (tag, arm, query) — reruns skip anything cached, so crashes never force a full re-run.

## 2026-09-27 — P5: full run, results, docs

- Full run `run1`: 175 queries x 4 arms in ~45 min, 0 errors, $0.26 total (under the $1.00 abort and the $0.15–0.25 projection's edge). Judge pass: 6 pairings x 175 x 2 orders in ~33 min. Gemini spot check: 50/pairing, 0–4 flagged each.
- `spot_check_gemini` shipped with a required `post` argument — a crash one hour into the run. Fixed test-first: default Gemini transport (httpx, status-carrying errors), argument now optional; the failed stage reruns alone against persisted artifacts instead of redoing the bench.
- Findings (README "Verdict" holds the full reading): reranking beats embedding order decisively; Nemotron wins quality by a wide margin with a brutal p95 tail (18.6s); Jev listwise is the right decision-model shape (~1s median, 0.632 recall@1, beats embed clearly) and pointwise is dominated; judge-to-judge kappa is only 0.17–0.37, so judge verdicts that contradict gold metrics (listwise > nemotron) are reported as unproven rather than headlined.
- README carries results tables, verdict, reproduce steps, license attributions (3 public-domain, OpenStax CC-BY 4.0, d2l + Wikipedia CC-BY-SA 4.0 with contributor attribution), and the related-work map (BEIR/MS MARCO/MTEB(r), BRIGHT, ViDoRe, cross-encoder-vs-LLM-reranker, MT-Bench).
- Whole-branch review (fresh reviewer, full working tree): 0 critical / 4 important / 7 minor. All four importants fixed test-first, suite 75/75: the DPI fallback ladder was dead code under a comment claiming it existed (now implemented; run1 unaffected — every needed page fit at 150 dpi); failed Nemotron rows were cached as successes so resume never retried them (now retried); per-query-type result tables were a spec success criterion that shipped missing (now in report.md and README — reasoning queries are the open problem, graphical is where page images pay); the judge/report stages had no committed entry point (now `python -m rerankbench.judges` and `python -m rerankbench.report`, and the README's regenerability claim is now accurate). The 7 minors are recorded as deferred in the plan ledger.

