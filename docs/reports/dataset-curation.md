# Dataset curation report

The 175-query dataset: what is in it, how each slice was authored and verified, how
balanced it is, and what it cannot test. Source of truth for the queries:
`scripts/build_dataset.py` (committed Python literals) writing
`data/dataset/queries.jsonl`; this report records the procedure and judgment calls.
`data/dataset/curation_notes.md` is the working log from authoring day.

## 1. Corpus the queries sit on

Built by `rerankbench.ingest` from 12 sources, chunked page-locally at
500-token target / 50-token overlap (no chunk spans a page), capped at 250 pages per
PDF and 600 chunks per document to hold the spec's 1,500-3,000 chunk target.
Provenance (URL + sha256 per source) lives in `data/corpus_manifest.json`.

| source | content | pages | chunks | license |
|---|---|---|---|---|
| mobydick (Gutenberg) | novel-length prose | 71 | 600 | public domain |
| constitution (archives.gov) | law, short reference | 1 | 14 | public domain |
| scotus (Cornell LII, Obergefell) | case-law reasoning | 1 | 94 | public domain |
| openstax_phys | physics textbook + figures | 248 | 431 | CC-BY 4.0 |
| openstax_calc | calculus textbook | 249 | 322 | CC-BY 4.0 |
| d2l (Dive into Deep Learning) | ML prose + code | 248 | 361 | CC-BY-SA 4.0 |
| 6 Wikipedia articles | encyclopedic text | 3-5 each | 170 total | CC-BY-SA 4.0 |

Total: **1,992 chunks**. The mix is deliberate: three text genres a reranker
plausibly meets in production (literary prose, legal reasoning, STEM textbook),
encyclopedic filler that generates plausible-but-wrong distractors, and two
figure-heavy PDFs for the multimodal slice.

## 2. The six query types

| type | n | what it tests | gold |
|---|---|---|---|
| verbatim | 40 | lexical match: query quotes a distinctive phrase from one chunk | 1 chunk |
| paraphrase | 40 | semantic match: same content, different wording | 1 chunk |
| reasoning | 30 | two-chunk synthesis (BRIGHT-style) | 2 chunks |
| negative | 20 | discrimination: plausible-sounding, unanswerable from corpus | empty |
| structural | 20 | document furniture: TOC, headings, captions, front matter | 1 chunk |
| graphical | 25 | figure/table content (the VL payload's home turf) | 1 chunk |

Example per type (first of each, from the committed dataset):

- **verbatim** — `which passage contains the line "Who-e debel you? - you no
  speak-e, dam-me, I kill-e"?` (gold: mobydick_c0040)
- **paraphrase** — `where does the text describe the whaler's old weathered look,
  with masts originally lost overboard in a gale ...` (gold: mobydick_c0089)
- **reasoning** — `why does the narrator hesitate at the tent door before shipping
  on the Pequod, and what do the part-owner and ...` (gold: mobydick_c0091 +
  mobydick_c0092 — each chunk alone is insufficient)
- **structural** — `in which chapter and on what page does the book's treatment of
  the Transformer architecture begin?` (gold: d2l_c0012, a TOC chunk)
- **graphical** — `which figure deliberately shows an image that could be a
  whirlpool in a tank of water or a collage of paint ...` (gold: openstax_phys_c0031,
  a figure caption)
- **negative** — `which amendment prohibits quartering soldiers in private homes
  during peacetime?` — third-amendment content; verified topic-absent (zero corpus
  hits for the amendment's language), so it is genuinely unanswerable here.

### Authoring procedure per type

- **verbatim**: every gold chunk was read from the built corpus; the quoted phrase
  was checked to appear in that chunk's text.
- **paraphrase**: authored from the gold chunk's content with the chunk's wording
  deliberately not reused.
- **reasoning**: pairs selected so that each chunk alone is insufficient —
  precedent + modern evidence, theorem + application, definition + worked example.
  Both chunks are gold, and nDCG@10's binary gains require both ranked high.
- **negative**: two-pass verification (`scripts/check_negatives.py`):
  (1) half-keyword co-occurrence scan over all 1,992 chunks, then (2)
  domain-critical-term scan with manual context review of every hit. 18 of 20 are
  term-scanned; 2 (neg_06, neg_20) have no reliable domain terms and were verified
  by full manual review instead.
- **structural**: gold is the furniture chunk itself (TOC entry, heading block,
  caption list, front matter).
- **graphical**: gold chunk contains the figure caption text; all 25 verified to
  carry "Figure" caption text at authoring time. 15 from openstax_phys, 10 from
  openstax_calc.

### Negative distractors reviewed and cleared

Corpus hits that look like answers but are not — several kept on purpose as
near-miss distractors (they make negatives harder and test discrimination):

| query | decoy hit | verdict |
|---|---|---|
| neg_03 refrigerator | whale bellies "are refrigerators" (metaphor); "pushing a refrigerator" force example | neither is a how-to |
| neg_04 cricket | "ye cricket-players"; "merry as a cricket" (mobydick idioms) | no cricket rules |
| neg_06 tokyo | "from Tokyo to Kyoto" maglev example (physics) | no population data — kept as distractor |
| neg_08 inception | GoogLeNet "Inception Blocks" TOC; "since its inception" idiom | no film content |
| neg_09 IRS | zero word-boundary hits for IRS; "1040", "schedule c" absent | clean |
| neg_13 react | "takes 0.5 s to react" (physics verb) | no React.js |
| neg_19 aperture | "apertures in its sounding-board" (whale anatomy) | no camera meaning |
| neg_20 trafalgar | "Admiral Nelson ... in Trafalgar Square" | no order of battle — kept as distractor |

## 3. Balance

- Type quotas exact: 40/40/30/20/20/25 (asserted by `rerankbench.dataset_stats`,
  which also enforces 20-400 char query length and id uniqueness).
- Source spread (non-negative queries): openstax_phys 39, openstax_calc 26,
  mobydick 24, scotus 20, d2l 17, constitution 15, wiki_* 12. Negatives carry
  `source_doc: none` (20).
- 155 gold-bearing queries (all but the 20 negatives) enter gold aggregates; the
  report's scope line pins this (`620 of 700 rows` = 155 x 4 arms).

## 4. Known limitations

- **PDF math extraction is mangled** (vector hats, equation layout) in the calculus
  and physics texts. This is documented in the spec and affects all arms equally —
  it lowers absolute numbers on math-heavy queries but does not bias comparisons.
- One author, one day: query difficulty was not calibrated against human solvers.
  Gold labels are the author's; no inter-annotator agreement exists.
- 20 structural and 25 graphical cells are small; per-type conclusions on those
  slices are suggestive, not definitive.
- Negatives test "unanswerable from this corpus", not "unanswerable" in general —
  several are trivially answerable on the open web (by design: the corpus is the
  universe).
- The corpus is English-only and excludes code-heavy and handwritten sources; BRIGHT
  covers code/StackExchange better.

## 5. Files

- `data/dataset/queries.jsonl` — 175 rows: `{id, query, query_type, gold_chunk_ids,
  source_doc, notes}`
- `scripts/build_dataset.py` — the queries as literals (provenance)
- `scripts/check_negatives.py` — the two-pass negative verifier
- `data/dataset/curation_notes.md` — authoring-day log
- `data/corpus_manifest.json` — corpus provenance
