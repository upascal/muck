"""Sentence-aware overlapping chunker with provenance.

Ported from ``agent-suite/lore/backend/src/lore/chunker.py`` (boundary detection +
snapping), extended so each chunk records its char offsets into the document text and
the page/locator it falls on — the basis for every citation.
"""

from __future__ import annotations

import re

from ..config import ChunkCfg
from ..schema import Chunk, Page


def _find_sentence_boundaries(text: str) -> list[int]:
    boundaries = [0]
    for m in re.finditer(r"(?<=[.!?])\s+|\n\n+", text):
        boundaries.append(m.end())
    return boundaries


def _snap_to_boundary(target: int, boundaries: list[int]) -> int:
    best = 0
    for b in boundaries:
        if b <= target:
            best = b
        else:
            if abs(b - target) < abs(best - target):
                best = b
            break
    return best


def _page_locator(pages: list[Page] | None, offset: int, doc_locator: str) -> str:
    from ..quality import page_for_offset

    pg = page_for_offset(pages, offset)
    return str(pg.number) if pg is not None else doc_locator


def chunk_document(
    doc_id: str,
    text: str,
    cfg: ChunkCfg,
    pages: list[Page] | None = None,
    doc_locator: str = "",
) -> list[Chunk]:
    if not text.strip():
        return []

    boundaries = _find_sentence_boundaries(text)
    chunk_chars = int(cfg.size_tokens / cfg.token_estimate_factor * 4.5)
    overlap_chars = int(cfg.overlap_tokens / cfg.token_estimate_factor * 4.5)

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            snapped = _snap_to_boundary(end, boundaries)
            if snapped > start:
                end = snapped

        body = text[start:end]
        stripped = body.strip()
        if stripped:
            # Keep offsets aligned with the (stripped) chunk text so citations are exact.
            lead = len(body) - len(body.lstrip())
            real_start = start + lead
            real_end = real_start + len(stripped)
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{idx}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    text=stripped,
                    char_start=real_start,
                    char_end=real_end,
                    locator=_page_locator(pages, real_start, doc_locator),
                    token_count=int(len(stripped.split()) * cfg.token_estimate_factor),
                )
            )

        if end >= len(text):
            break  # this chunk reached end-of-text; no redundant tail chunk
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end
        if next_start >= len(text):
            break
        snapped = _snap_to_boundary(next_start, boundaries)
        start = snapped if (not chunks or snapped > chunks[-1].char_start) else next_start

    return chunks
