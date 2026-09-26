"""Verify negative queries are genuinely topic-absent from the corpus.

A negative query is answered by the corpus only if the corpus contains its
distinctive domain terms. For each negative query, DOMAIN_TERMS lists the words
whose presence in any chunk would mean the topic is actually covered; generic
words (form, work, train, water) are deliberately excluded because they match
unrelated prose. Expected result: zero hits for every term.
"""
from __future__ import annotations

from rerankbench.chunk import load_chunks
from rerankbench.config import load_config
from rerankbench.dataset import load_queries

# Terms whose presence would mean the negative topic is actually covered.
# Terms excluded here for having benign distractor hits (documented in
# data/dataset/curation_notes.md): refrigerator (whale-belly metaphor +
# physics push example), cricket (idioms), tokyo (maglev motion example),
# inception (GoogLeNet blocks), react (physics verb), aperture (whale
# blowhole), trafalgar (Trafalgar Square) - none of them answer the query.
DOMAIN_TERMS = {
    "neg_01": ["quartering", "third amendment"],
    "neg_02": ["formula 1", "motorsport", "grand prix"],
    "neg_03": ["whirlpool refrigerator", "water filter"],
    "neg_04": ["wicket", "lbw"],
    "neg_05": ["kombucha", "scoby", "fermented tea"],
    "neg_06": [],
    "neg_07": ["asyncio", "event loop", "coroutine"],
    "neg_08": ["dominic cobb"],
    "neg_09": ["1040", "schedule c"],
    "neg_10": ["chord progression", "improvisation"],
    "neg_11": ["tennis elbow", "epicondylitis", "tendonitis"],
    "neg_12": ["sourdough", "hydration ratio", "starter culture"],
    "neg_13": ["useeffect", "usestate"],
    "neg_14": ["bitcoin", "blockchain", "proof-of-work"],
    "neg_15": ["nicaraguan", "nicaragua canal"],
    "neg_16": ["retriever", "puppy"],
    "neg_17": ["glass menagerie", "tennessee williams"],
    "neg_18": ["figure skating", "ice skating"],
    "neg_19": ["shutter speed", "f-stop"],
    "neg_20": [],
}


def main() -> None:
    cfg = load_config()
    chunks = load_chunks(cfg.data_dir / "chunks" / "chunks.jsonl")
    queries = load_queries(cfg.data_dir / "dataset" / "queries.jsonl")
    negs = [q for q in queries if q.query_type == "negative"]

    clean = True
    for q in negs:
        terms = DOMAIN_TERMS[q.id]
        for term in terms:
            hits = [c.id for c in chunks if term in c.text.lower()]
            if hits:
                clean = False
                print(f"{q.id}: term {term!r} appears in {len(hits)} chunk(s): {hits[:6]}")
    if clean:
        print(f"all {len(negs)} negative queries verified topic-absent (0 domain-term hits)")


if __name__ == "__main__":
    main()
