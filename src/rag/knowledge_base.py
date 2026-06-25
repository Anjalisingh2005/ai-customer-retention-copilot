"""Read + chunk the markdown knowledge base."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.config import settings


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def _split_markdown(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    """Header-aware chunking. Splits on H2 (##), packs into <= max_chars chunks
    with a small character overlap so retrieval finds whole policies."""
    # First, split by H2 headers.
    sections: List[str] = []
    buf: List[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and buf:
            sections.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append("\n".join(buf).strip())

    chunks: List[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
            continue
        # Slide a window with overlap.
        start = 0
        while start < len(section):
            chunks.append(section[start : start + max_chars])
            start += max_chars - overlap
    return [c for c in chunks if c.strip()]


def load_chunks(kb_dir: Path | None = None) -> List[Chunk]:
    kb_dir = kb_dir or settings.knowledge_base_dir
    chunks: List[Chunk] = []
    idx = 0
    for md in sorted(kb_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for piece in _split_markdown(text):
            chunks.append(Chunk(text=piece, source=md.name, chunk_id=idx))
            idx += 1
    return chunks
