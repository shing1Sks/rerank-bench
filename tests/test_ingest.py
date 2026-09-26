from pathlib import Path

from rerankbench.ingest import (
    extract_pages,
    fetch_file,
    gutenberg_body,
    html_to_text,
    parse_wiki_extract,
    pick_url,
    sha256_bytes,
)


class FakeResp:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status


def test_pick_url_prefers_first_success():
    calls = []

    def client_get(url):
        calls.append(url)
        return FakeResp(b"ok") if url.endswith("good") else FakeResp(b"", status=404)

    assert pick_url(["bad", "good", "later"], client_get) == "good"
    assert calls == ["bad", "good"]


def test_pick_url_returns_none_when_all_fail():
    assert pick_url(["a", "b"], lambda u: FakeResp(b"", status=500)) is None


def test_gutenberg_body_strips_markers():
    raw = "Intro junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nreal text here\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense junk"
    assert gutenberg_body(raw) == "real text here"


def test_extract_pages_txt_real(tmp_path: Path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("page one words " * 50 + "\f" + "page two words " * 50, encoding="utf-8")
    pages = extract_pages("d", text_file)
    assert len(pages) == 2 and pages[0][0] == 1 and pages[1][0] == 2


def test_extract_pages_txt_without_form_feeds_makes_pseudo_pages(tmp_path: Path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text(" ".join(["word"] * 9000), encoding="utf-8")
    pages = extract_pages("d", text_file)
    assert len(pages) == 3  # 3000-word pseudo-pages
    assert [p[0] for p in pages] == [1, 2, 3]


def test_html_to_text_drops_script_and_style():
    html = "<html><head><style>b{}</style><script>evil()</script></head><body><h1>Title</h1><p>Body text.</p></body></html>"
    text = html_to_text(html)
    assert "Title" in text and "Body text." in text
    assert "evil" not in text and "b{}" not in text


def test_parse_wiki_extract_takes_first_page():
    data = {"query": {"pages": {"123": {"title": "Photosynthesis", "extract": "It is a process."}}}}
    title, text = parse_wiki_extract(data)
    assert title == "Photosynthesis" and text == "It is a process."


def test_sha256_bytes_stable():
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


class _Ok:
    status_code = 200
    content = b"<html><body>" + b"hello corpus body text " * 50 + b"</body></html>"  # >1KB: passes the size guard


def test_fetch_file_downloads_sniffs_suffix_and_hashes(tmp_path: Path):
    src = {"doc_id": "d1", "kind": "html", "urls": ["u1", "u2"]}
    path, sha, url = fetch_file(None, src, tmp_path, get=lambda u: _Ok())
    assert url == "u1"
    assert path.name == "d1.html" and path.read_bytes().startswith(b"<html>")
    assert sha == sha256_bytes(_Ok.content)


def test_fetch_file_reuses_cache_without_network(tmp_path: Path):
    cached = tmp_path / "d1.html"
    cached.write_bytes(b"cached bytes")
    (tmp_path / "d1.html.url").write_text("https://original/url", encoding="utf-8")

    def no_network(url):
        raise AssertionError("network touched on cache hit")

    path, sha, url = fetch_file(None, {"doc_id": "d1", "kind": "html", "urls": []}, tmp_path, get=no_network)
    assert path == cached and url == "https://original/url" and sha == sha256_bytes(b"cached bytes")
