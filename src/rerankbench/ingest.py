"""Corpus ingest: download license-clean sources, extract page-mapped text, chunk, manifest.

PDFs are cached locally (gitignored) and never committed; the manifest pins the
winning URL + sha256 of every fetched file so runs are reproducible from scripts.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pypdfium2 as pdfium
from bs4 import BeautifulSoup

from rerankbench.chunk import chunk_pages, save_chunks
from rerankbench.config import Config

MAX_PAGES_PDF = 250  # cap big textbooks so the corpus holds the spec's 1,500-3,000 chunk target
MAX_CHUNKS_PER_DOC = 600
PSEUDO_PAGE_WORDS = 3000  # txt without form feeds is blocked into pseudo-pages
WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "Mozilla/5.0 (rerank-bench research fetch; contact: repo README)"}

SOURCES: list[dict] = [
    {
        "doc_id": "mobydick",
        "title": "Moby Dick (Gutenberg)",
        "kind": "gutenberg_txt",
        "license": "public domain",
        "max_pages": None,
        "urls": ["https://www.gutenberg.org/cache/epub/2701/pg2701.txt"],
    },
    {
        "doc_id": "constitution",
        "title": "US Constitution (transcript)",
        "kind": "html",
        "license": "public domain",
        "max_pages": None,
        "urls": ["https://www.archives.gov/founding-docs/constitution-transcript"],
    },
    {
        "doc_id": "scotus",
        "title": "SCOTUS majority opinion (Obergefell v. Hodges)",
        "kind": "html",
        "license": "public domain",
        "max_pages": None,
        "urls": [
            "https://www.law.cornell.edu/supremecourt/text/14-556",
            "https://supreme.justia.com/cases/federal/us/576/644/",
            "https://www.law.cornell.edu/supct/html/14-556.ZO.html",
        ],
    },
    {
        "doc_id": "openstax_phys",
        "title": "OpenStax University Physics vol 1",
        "kind": "pdf",
        "license": "CC-BY 4.0",
        "max_pages": MAX_PAGES_PDF,
        "openstax_slug": "university-physics-volume-1",
        "urls": [
            "https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/UniversityPhysicsVolume1-LR.pdf",
        ],
    },
    {
        "doc_id": "openstax_calc",
        "title": "OpenStax Calculus vol 1",
        "kind": "pdf",
        "license": "CC-BY 4.0",
        "max_pages": MAX_PAGES_PDF,
        "openstax_slug": "calculus-volume-1",
        "urls": [
            "https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/CalculusVolume1-LR.pdf",
        ],
    },
    {
        "doc_id": "d2l",
        "title": "Dive into Deep Learning",
        "kind": "pdf",
        "license": "CC-BY-SA 4.0",
        "max_pages": MAX_PAGES_PDF,
        "urls": ["https://d2l.ai/d2l-en.pdf"],
    },
]

WIKI_TITLES = [
    "Renaissance",
    "Photosynthesis",
    "Quantum mechanics",
    "Plate tectonics",
    "French Revolution",
    "Great Barrier Reef",
]


# ---------------------------------------------------------------- fetch helpers

class _Resp:
    """Minimal response surface pick_url needs; lets tests inject fakes."""

    def __init__(self, content: bytes, status_code: int):
        self.content = content
        self.status_code = status_code


def pick_url(urls: list[str], get) -> str | None:
    """First URL that answers HTTP 200 (network errors and failures are skipped)."""
    for url in urls:
        try:
            r = get(url)
        except Exception:
            continue
        if getattr(r, "status_code", 0) == 200:
            return url
    return None


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _get(client: httpx.Client, url: str) -> _Resp:
    """httpx first; on 403 fall back to curl (some hosts block python TLS fingerprints)."""
    try:
        r = client.get(url, headers=UA)
        if r.status_code != 403:
            return _Resp(r.content, r.status_code)
    except Exception:
        pass
    proc = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "180", "-A", UA["User-Agent"], "-w", "\n%{http_code}", url],
        capture_output=True,
    )
    body, _, code = proc.stdout.rpartition(b"\n")
    return _Resp(body, int(code or 0))


def _openstax_candidates(client: httpx.Client, slug: str) -> list[str]:
    """Scrape the book's details page for its direct PDF hrefs (cloudfront links drift)."""
    try:
        r = client.get(f"https://openstax.org/details/books/{slug}", headers=UA)
        found = [u.replace("\\/", "/") for u in re.findall(r'https:[^"\'\s]+?\.pdf', r.text)]
        if found:
            return found[:3]
    except Exception:
        pass
    return [f"https://openstax.org/api/pages/{slug}?include=pdf"]


def fetch_file(
    client: httpx.Client | None,
    source: dict,
    cache_dir: Path,
    *,
    get=None,
    refresh: bool = False,
) -> tuple[Path, str, str]:
    """Download the first working candidate URL; return (cached_path, sha256, url).

    Reuses an existing cache file (with its .url sidecar) unless refresh=True.
    """
    get = get or (lambda u: _get(client, u))
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in cache_dir.glob(f"{source['doc_id']}.*") if p.suffix != ".url")
    if existing and not refresh:
        path = existing[0]
        sidecar = path.with_suffix(path.suffix + ".url")
        url = sidecar.read_text(encoding="utf-8") if sidecar.exists() else f"cached:{path.name}"
        return path, sha256_bytes(path.read_bytes()), url

    candidates = list(source["urls"])
    if source.get("openstax_slug"):
        candidates = _openstax_candidates(client, source["openstax_slug"]) + candidates
    url = pick_url(candidates, get)
    if url is None:
        raise RuntimeError("no candidate URL answered 200")
    r = get(url)
    if len(r.content) < 1000:
        raise RuntimeError(f"body suspiciously small ({len(r.content)} bytes)")
    suffix = ".pdf" if r.content[:5] == b"%PDF-" else ".txt" if source["kind"].endswith("_txt") else ".html"
    path = cache_dir / f"{source['doc_id']}{suffix}"
    path.write_bytes(r.content)
    path.with_suffix(path.suffix + ".url").write_text(url, encoding="utf-8")
    return path, sha256_bytes(r.content), url


# ---------------------------------------------------------------- extraction

def gutenberg_body(text: str) -> str:
    """Text between the Gutenberg START/END markers, or the whole text if unmarked."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "*** START" in l), None)
    end = next((i for i, l in enumerate(lines) if "*** END" in l), None)
    if start is None or end is None or end <= start:
        return text.strip()
    return "\n".join(lines[start + 1 : end]).strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def parse_wiki_extract(data: dict) -> tuple[str, str]:
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise RuntimeError("wikipedia response had no pages")
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise RuntimeError(f"wikipedia page missing extract: {page.get('title')}")
    return page["title"], page["extract"]


def wiki_slug(title: str) -> str:
    return re.sub(r"\W+", "_", title.lower()).strip("_")


def extract_pages(doc_id: str, path: Path, *, max_pages: int | None = None) -> list[tuple[int, str]]:
    """(page_number, text) pairs; never reads past max_pages."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        if "*** START" in text and "*** END" in text:
            text = gutenberg_body(text)
        parts = [p for p in text.split("\f") if p.strip()]
        if len(parts) <= 1:
            words = text.split()
            parts = [" ".join(words[i : i + PSEUDO_PAGE_WORDS]) for i in range(0, len(words), PSEUDO_PAGE_WORDS)]
        if max_pages:
            parts = parts[:max_pages]
        return list(enumerate((p for p in parts if p.strip()), start=1))
    if suffix == ".html":
        return [(0, html_to_text(path.read_text(encoding="utf-8", errors="replace")))]
    if suffix == ".pdf":
        pdf = pdfium.PdfDocument(str(path))
        count = min(len(pdf), max_pages or len(pdf))
        out: list[tuple[int, str]] = []
        for i in range(count):
            text = pdf[i].get_textpage().get_text_range() or ""
            if text.strip():
                out.append((i + 1, text))
        return out
    raise ValueError(f"unsupported corpus file type: {suffix}")


# ---------------------------------------------------------------- corpus build

def build_corpus(cfg: Config, client: httpx.Client | None = None) -> list:
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(300))
    cache_dir = cfg.data_dir / "cache" / "corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    all_chunks: list = []

    def add(doc_id: str, title: str, license_: str, path: Path, max_pages: int | None, url: str, sha: str) -> None:
        pages = extract_pages(doc_id, path, max_pages=max_pages)
        chunks = chunk_pages(
            pages,
            doc_id,
            license_,
            target_tokens=cfg.chunk_target_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
        )[:MAX_CHUNKS_PER_DOC]
        all_chunks.extend(chunks)
        entries.append(
            {
                "doc_id": doc_id,
                "title": title,
                "license": license_,
                "url": url,
                "sha256": sha,
                "pages": len(pages),
                "chunks": len(chunks),
                "status": "ok",
            }
        )
        print(f"{doc_id}: OK url={url} pages={len(pages)} chunks={len(chunks)}", flush=True)

    for src in SOURCES:
        try:
            path, sha, url = fetch_file(client, src, cache_dir)
            add(src["doc_id"], src["title"], src["license"], path, src["max_pages"], url, sha)
        except Exception as e:  # a failed source never blocks the run; manifest records it
            entries.append({"doc_id": src["doc_id"], "title": src["title"], "license": src["license"], "status": f"failed: {e}"})
            print(f"{src['doc_id']}: FAILED {e}", flush=True)

    for title in WIKI_TITLES:
        doc_id = f"wiki_{wiki_slug(title)}"
        try:
            query = urlencode(
                {"action": "query", "format": "json", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title}
            )
            resp = _get(client, f"{WIKI_API}?{query}")
            data = json.loads(resp.content.decode("utf-8", errors="replace"))
            wtitle, text = parse_wiki_extract(data)
            path = cache_dir / f"{doc_id}.txt"
            path.write_text(text, encoding="utf-8")
            url = f"https://en.wikipedia.org/wiki/{wtitle.replace(' ', '_')}"
            add(doc_id, f"Wikipedia: {wtitle}", "CC-BY-SA 4.0", path, None, url, sha256_bytes(text.encode("utf-8")))
        except Exception as e:
            entries.append({"doc_id": doc_id, "title": f"Wikipedia: {title}", "license": "CC-BY-SA 4.0", "status": f"failed: {e}"})
            print(f"{doc_id}: FAILED {e}", flush=True)

    save_chunks(all_chunks, cfg.data_dir / "chunks" / "chunks.jsonl")
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "chunk_target_tokens": cfg.chunk_target_tokens,
        "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
        "max_pages_pdf": MAX_PAGES_PDF,
        "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
        "sources": entries,
    }
    (cfg.data_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failed = [e["doc_id"] for e in entries if e["status"] != "ok"]
    print(f"corpus: {len(all_chunks)} chunks from {len(entries)} sources; failed: {failed or 'none'}", flush=True)
    return all_chunks


if __name__ == "__main__":
    from rerankbench.config import load_config

    build_corpus(load_config())
