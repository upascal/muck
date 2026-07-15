"""Text-quality signals shared across parsing, indexing, and citation.

muck's default parser only reads a PDF's embedded text layer. An image-only scan yields
``""`` per page, which used to be indexed as a normal (empty) document — so a 322-page
scanned corpus looked healthy while retrieving nothing, and the agent read that as "no
evidence" rather than "not readable". These helpers make the difference measurable:

- ``classify_page`` labels a page's *content* (blank / garbled / table / text).
- ``page_coverage`` / ``coverage_verdict`` roll pages up into a per-document verdict
  (native | sparse | image_only) that routes a corpus to OCR or vision transcription.
- ``page_for_offset`` maps a character offset to its page (used for both chunk locators
  and cite-to-pixels).
"""

from __future__ import annotations

import re

from .schema import Page

_ALNUM = re.compile(r"[A-Za-z0-9]")

# A page needs at least this many characters of extractable text to count as "has text".
# Below it, a page is effectively blank (headers/stray marks on an image-only scan).
MIN_PAGE_TEXT_CHARS = 20


def classify_page(text: str) -> str:
    """Label a page by its *content*: empty | bad_ocr | table | text.

    Mirrors the heuristic that previously lived (unused) in the pymupdf parser, plus an
    explicit ``empty`` class so a blank image-only page is not mislabelled ``text``.
    """
    stripped = text.strip()
    if len(stripped) < MIN_PAGE_TEXT_CHARS:
        return "empty"
    alnum = sum(1 for c in text if _ALNUM.match(c))
    if alnum / max(1, len(text)) < 0.5:
        return "bad_ocr"
    if text.count("\t") > 5 or text.count("  ") > text.count(" ") * 0.3:
        return "table"
    return "text"


def page_for_offset(pages: list[Page] | None, offset: int) -> Page | None:
    """The page whose span contains ``offset`` (or the last page if past the end)."""
    if not pages:
        return None
    for pg in pages:
        if pg.char_start <= offset < pg.char_end:
            return pg
    return pages[-1]


def page_coverage(pages: list[Page] | None, text: str) -> dict:
    """Per-document text-coverage stats: pages, pages_with_text, ratio, chars_per_page."""
    if not pages:
        # No page spans (non-PDF, or a whole-file document): treat as one page.
        has_text = len(text.strip()) >= MIN_PAGE_TEXT_CHARS
        return {
            "pages": 1,
            "pages_with_text": 1 if has_text else 0,
            "coverage": 1.0 if has_text else 0.0,
            "chars_per_page": float(len(text)),
        }
    n = len(pages)
    with_text = sum(
        1 for p in pages if (p.char_end - p.char_start) >= MIN_PAGE_TEXT_CHARS
    )
    return {
        "pages": n,
        "pages_with_text": with_text,
        "coverage": with_text / n if n else 0.0,
        "chars_per_page": len(text) / n if n else 0.0,
    }


def coverage_verdict(coverage: float) -> str:
    """Route a document by how much of it muck can actually read.

    native      — a real text layer on most pages; index as-is.
    sparse      — some text, but large gaps (a bad or partial OCR layer); OCR may help.
    image_only  — essentially no extractable text; needs render + OCR/vision to be seen.
    """
    if coverage >= 0.7:
        return "native"
    if coverage >= 0.1:
        return "sparse"
    return "image_only"
