"""Page-aware chunking: word-window chunks that never span pages."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    page_start: int
    page_end: int


def approx_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)) * 4 // 3)


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    source: str,
    *,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    out: list[Chunk] = []
    for page, text in pages:
        words = text.split()
        if not words:
            continue
        win = max(1, int(target_tokens * 3 / 4))  # ~375 words at the 500-token default
        ov = max(1, int(overlap_tokens * 3 / 4))
        step = max(1, win - ov)
        i = 0
        while i < len(words):
            piece = " ".join(words[i : i + win])
            out.append(Chunk(f"{doc_id}_c{len(out):04d}", doc_id, source, piece, page, page))
            if i + win >= len(words):
                break
            i += step
    return out


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
