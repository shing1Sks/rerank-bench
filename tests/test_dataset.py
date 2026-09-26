import json
from pathlib import Path

from rerankbench.dataset import QUERY_TYPES, TYPE_QUOTAS, Query, load_queries, validate_queries


def q(**kw):
    base = dict(
        id="q1",
        query="which passage explains the claimed effect?",
        query_type="verbatim",
        gold_chunk_ids=["c1"],
        source_doc="d",
        notes="",
    )
    return Query(**{**base, **kw})


def test_validation_catches_bad_type_and_unknown_gold():
    errs = validate_queries([q(query_type="nope"), q(id="q2", gold_chunk_ids=["ghost"])], {"c1"})
    assert any("query_type" in e and "q1" in e for e in errs)
    assert any("gold" in e and "q2" in e for e in errs)


def test_negative_must_have_empty_gold_and_others_must_not():
    errs = validate_queries([q(query_type="negative", gold_chunk_ids=[]), q(id="q3", gold_chunk_ids=[])], set())
    assert not any("q1" in e for e in errs)
    assert any("q3" in e for e in errs)


def test_query_length_bounds_and_unique_ids():
    errs = validate_queries([q(id="dup", query="word " * 100 + "end"), q(id="dup", query="tiny")], set())
    assert any("length" in e and "dup" in e for e in errs)
    assert any("duplicate id" in e for e in errs)


def test_jsonl_roundtrip(tmp_path: Path):
    p = tmp_path / "qs.jsonl"
    p.write_text(json.dumps(q().__dict__) + "\n", encoding="utf-8")
    assert load_queries(p)[0] == q()


def test_taxonomy_and_quotas_total_inside_spec_range():
    assert set(QUERY_TYPES) == {"verbatim", "paraphrase", "reasoning", "negative", "structural", "graphical"}
    assert 150 <= sum(TYPE_QUOTAS.values()) <= 250
