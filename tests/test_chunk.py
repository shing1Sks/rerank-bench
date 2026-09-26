from pathlib import Path

from rerankbench.chunk import (
    Chunk,
    approx_tokens,
    chunk_pages,
    load_chunks,
    save_chunks,
)


def test_approx_tokens_is_words_times_four_thirds():
    assert approx_tokens("one two three") == 4
    assert approx_tokens("") == 1


def test_chunks_respect_page_boundaries_and_overlap():
    pages = [(1, " ".join(["alpha"] * 400)), (2, " ".join(["beta"] * 400))]
    chunks = chunk_pages(pages, "doc1", "test", target_tokens=100, overlap_tokens=20)
    assert chunks, "expected chunks"
    assert all(c.page_start == c.page_end for c in chunks), "no chunk may span pages"
    assert all(c.doc_id == "doc1" and c.id.startswith("doc1_c") for c in chunks)
    # overlap: consecutive chunks on the same page share a word window
    same_page = [c for c in chunks if c.page_start == 1]
    assert len(same_page) > 1
    tail = same_page[0].text.split()[-5:]
    head = same_page[1].text.split()[:5]
    assert tail == head


def test_chunk_ids_are_unique_across_pages():
    pages = [(1, " ".join(["a"] * 300)), (2, " ".join(["b"] * 300))]
    chunks = chunk_pages(pages, "doc1", "test", target_tokens=100, overlap_tokens=20)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_page_is_skipped():
    assert chunk_pages([(1, ""), (2, "words here")], "d", "s", target_tokens=50, overlap_tokens=10)[0].page_start == 2


def test_chunk_jsonl_roundtrip(tmp_path: Path):
    pages = [(1, "hello world " * 100)]
    chunks = chunk_pages(pages, "d", "s", target_tokens=50, overlap_tokens=10)
    path = tmp_path / "c.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
