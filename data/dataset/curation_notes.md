# Dataset curation notes

175 queries, authored 2026-09-26 against `data/chunks/chunks.jsonl` (1,992 chunks).
Builder: `scripts/build_dataset.py` (every query is a Python literal there - this file
records the procedure and judgment calls, the script is the source of truth).

## Procedure per type

- **verbatim (40)**: query quotes or near-quotes a distinctive phrase from one gold
  chunk. Every gold id read from the built corpus; quotes checked against chunk text.
- **paraphrase (40)**: same content as the gold chunk, rewritten without reusing its
  wording. One gold chunk each.
- **reasoning (30)**: query requires connecting two chunks; both are gold. Pairs chosen
  so each chunk alone is insufficient (precedent + modern evidence, theorem +
  application, definition + rewriting).
- **negative (20)**: topically adjacent or plausible-sounding, genuinely unanswerable
  from the corpus. Empty gold by schema. Verified two ways: 1) half-keyword co-occurrence
  scan over all chunks (`scripts/check_negatives.py`, loose pass), then 2) domain-critical
  term scan with manual context review of every hit (strict pass).
- **structural (20)**: answer lives in document furniture - TOC entries, section
  headings, syllabus headers, citation lists, front matter. Gold is the furniture chunk.
- **graphical (25)**: answer is in a figure caption or table on a PDF page
  (openstax_phys 15, openstax_calc 10). Gold chunk contains the caption text; the
  VL arm renders `page_start` of the gold chunk. All 25 verified to contain
  "Figure" caption text at authoring time.

## Negative-verification distractors found and cleared

These corpus hits were reviewed and judged NOT to answer the negative queries
(several are useful near-miss distractors that make the negatives harder):

- neg_03 refrigerator: whale bellies "are refrigerators" (metaphor, mobydick_c0352);
  "pushing a refrigerator" force example (openstax_phys_c0408). Neither is a how-to.
- neg_04 cricket: "ye cricket-players" (mobydick_c0197), "merry as a cricket"
  (mobydick_c0284). No cricket rules.
- neg_06 tokyo: "from Tokyo to Kyoto" maglev position example (openstax_phys_c0212).
  No population data. Kept as a near-miss distractor on purpose.
- neg_08 inception: GoogLeNet "Inception Blocks" TOC entry (d2l_c0008); "since its
  inception" idiom (wiki_quantum_mechanics). No film content.
- neg_09 IRS: zero word-boundary hits for IRS; "1040" and "schedule c" absent.
- neg_13 react: "takes 0.5 s to react" physics verb (openstax_phys_c0296). No React.js.
- neg_19 aperture: "apertures in its sounding-board" whale anatomy (mobydick_c0380).
  No camera meaning.
- neg_20 trafalgar: "Admiral Nelson... in Trafalgar Square" (mobydick_c0181). No order
  of battle. Kept as a near-miss distractor.

## Balance

- Type quotas met exactly: 40/40/30/20/20/25 (see `rerankbench.dataset.TYPE_QUOTAS`).
- source_doc spread: openstax_phys 39, openstax_calc 26, mobydick 24, scotus 20,
  d2l 17, constitution 15, wiki_* 12, none (negatives) 20.
- Query lengths all within the 20-400 char validator bound; ids unique.
- Known limitation: PDF math extracts are mangled (vector hats, equation layout);
  this is documented in the spec (section 12) and affects all arms equally.
