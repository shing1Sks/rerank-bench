"""Nemotron VL rerank arm via OpenRouter, plus page-image rendering for graphical queries.

Transport errors reuse JevError (the status-carrying wire error the shared test
fakes raise); NemotronError marks nemotron-specific failures. API failure never
crashes a run: the arm degrades to candidate order with flags["error"].
"""
from __future__ import annotations

import base64
import io
import time
from typing import Callable

import httpx
import pypdfium2 as pdfium

from rerankbench.rerankers import RerankResult
from rerankbench.rerankers.jev import JevError

RETRY_STATUSES = {429, 529}
ATTEMPTS = 5
# PNG above this size falls down the DPI ladder; OpenRouter image payloads stay small.
MAX_IMAGE_BYTES = 1_800_000
DPI_LADDER = (150, 110, 80)


class NemotronError(JevError):
    pass


def default_openrouter_post(url: str, body: dict, headers: dict) -> dict:
    resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
    if resp.status_code != 200:
        raise JevError(f"http {resp.status_code}: {resp.text[:500]}", status=resp.status_code)
    return resp.json()


class NemotronReranker:
    arm = "nemotron_vl"

    def __init__(
        self,
        cfg,
        post: Callable,
        image_for: Callable | None = None,
        sleep: Callable = time.sleep,
        slug: str | None = None,
    ):
        self._cfg = cfg
        self._post = post
        self._image_for = image_for
        self._sleep = sleep
        # probe_slugs decides the reachable variant; default keeps standalone use working
        self._slug = slug or cfg.nemotron_slug

    def rerank(self, query: str, candidates: list, query_meta: dict) -> RerankResult:
        t0 = time.perf_counter()
        documents = []
        for c in candidates:
            uri = self._image_for(c) if self._image_for else None
            documents.append({"image": uri, "text": c.text} if uri else {"text": c.text})
        body = {
            "model": self._slug,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        headers = {"Authorization": f"Bearer {self._cfg.openrouter_key}"}
        resp = None
        error = None
        calls = 0
        for attempt in range(ATTEMPTS):
            calls += 1
            try:
                resp = self._post(self._cfg.openrouter_rerank_url, body, headers)
                break
            except JevError as e:
                error = str(e)
                if e.status in RETRY_STATUSES and attempt < ATTEMPTS - 1:
                    self._sleep(min(30, 2**attempt))
                    continue
                break
        if resp is None:
            return RerankResult(
                arm=self.arm,
                query_id=query_meta.get("query_id", ""),
                ranked_ids=[c.id for c in candidates],
                scores=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                calls=calls,
                flags={"error": error or "no response"},
            )
        try:
            results = resp["results"]
            ranked_ids = [candidates[r["index"]].id for r in results]
            scores = [float(r["relevance_score"]) for r in results]
        except (KeyError, TypeError, IndexError) as e:
            raise NemotronError(f"unexpected rerank response shape: {e}") from e
        return RerankResult(
            arm=self.arm,
            query_id=query_meta.get("query_id", ""),
            ranked_ids=ranked_ids,
            scores=scores,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            calls=calls,
            flags={},
        )


def probe_slugs(cfg, post: Callable) -> str | None:
    """Return the first slug that answers a 2-document smoke rerank, else None."""
    headers = {"Authorization": f"Bearer {cfg.openrouter_key}"}
    smoke = {"query": "smoke", "documents": [{"text": "a"}, {"text": "b"}], "top_n": 2}
    for slug in (cfg.nemotron_slug, cfg.nemotron_slug_free):
        try:
            post(cfg.openrouter_rerank_url, {**smoke, "model": slug}, headers)
            return slug
        except JevError:
            continue
    return None


def _render_png(pdf_path, page: int, dpi: int) -> bytes:
    pdf = pdfium.PdfDocument(str(pdf_path))
    if page - 1 >= len(pdf):
        return b""
    pil_image = pdf[page - 1].render(scale=dpi / 72).to_pil()
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def page_image_data_uri(cfg, doc_id: str, page: int, dpi: int = 150) -> str | None:
    """Render one PDF page to a base64 PNG, caching on disk; None when impossible.

    Renders at the requested DPI; if the PNG exceeds MAX_IMAGE_BYTES it re-renders
    down DPI_LADDER before giving up (payload stays acceptable to the rerank API).
    """
    cache_png = cfg.data_dir / "cache" / "pages" / doc_id / f"p{page}.png"
    try:
        if cache_png.exists():
            data = cache_png.read_bytes()
            if data:
                return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        pdf_path = cfg.data_dir / "cache" / "corpus" / f"{doc_id}.pdf"
        if not pdf_path.exists():
            return None
        rungs = [d for d in DPI_LADDER if d <= dpi] or [dpi]
        for rung in rungs:
            data = _render_png(pdf_path, page, rung)
            if not data:
                return None
            if len(data) <= MAX_IMAGE_BYTES:
                cache_png.parent.mkdir(parents=True, exist_ok=True)
                cache_png.write_bytes(data)
                return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        print(f"page_image_data_uri: {doc_id} p{page}: png oversized at every DPI rung; sending text-only")
        return None
    except Exception as e:  # noqa: BLE001 - image absence must never fail a run
        print(f"page_image_data_uri: {doc_id} p{page}: {e}")
        return None
