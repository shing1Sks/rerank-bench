"""Print dataset stats: per-type counts vs quotas, per-source counts, validation result."""
from __future__ import annotations

from collections import Counter

from rerankbench.chunk import load_chunks
from rerankbench.config import load_config
from rerankbench.dataset import TYPE_QUOTAS, load_queries, validate_queries


def main() -> None:
    cfg = load_config()
    chunks = load_chunks(cfg.data_dir / "chunks" / "chunks.jsonl")
    queries = load_queries(cfg.data_dir / "dataset" / "queries.jsonl")

    errs = validate_queries(queries, {c.id for c in chunks})
    print(f"validation: {'OK' if not errs else f'{len(errs)} ERRORS'}")
    for e in errs:
        print(" ", e)

    by_type = Counter(q.query_type for q in queries)
    print(f"\n{'type':<12} {'count':>5} {'quota':>5} {'ok':>3}")
    for t, quota in TYPE_QUOTAS.items():
        n = by_type.get(t, 0)
        print(f"{t:<12} {n:>5} {quota:>5} {'yes' if n >= quota else 'NO':>3}")
    extra = set(by_type) - set(TYPE_QUOTAS)
    if extra:
        print("unexpected types:", extra)

    by_doc = Counter(q.source_doc for q in queries)
    print("\nper source_doc:")
    for doc, n in sorted(by_doc.items()):
        print(f"  {doc:<24} {n:>4}")
    print(f"\ntotal queries: {len(queries)}")


if __name__ == "__main__":
    main()
