"""Backend-agnostic data structures shared across the pipeline.

A *document* is one logical unit: a whole file (PDF/DOCX/MD/TXT) or a single record
inside a JSON file. Every document carries enough provenance — ``source_path`` +
``locator`` + character offsets — to resolve any span back to its exact origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    """A page span within a parsed document's concatenated text (PDF provenance).

    ``tier`` records how this page's text was derived — ``native`` (a real text layer,
    exact), ``ocr`` (a deterministic engine; a character can be misread but not invented),
    or ``vision`` (an LLM transcribed the pixels; a whole sentence can be fabricated). It
    is what lets a citation state honestly what verification does and does not prove.
    ``page_class`` describes the *content* (blank/table/garbled), ``tier`` the *source*.
    """

    number: int
    char_start: int
    char_end: int
    page_class: str = "text"  # text|table|figure|bad_ocr|empty
    tier: str = "native"  # native|ocr|vision


@dataclass
class ParsedDoc:
    """What a Parser returns for one file: full text + page spans.

    ``text_provenance`` is the document-level roll-up of its pages' tiers
    (native|ocr|vision|mixed) — the default is ``native`` so every existing parser is
    correct without change.
    """

    text: str
    pages: list[Page] = field(default_factory=list)
    title: str | None = None
    text_provenance: str = "native"


@dataclass
class Record:
    """One logical document produced by a Mapper or Parser.

    ``text`` is authoritative for character offsets; chunk/citation offsets index
    into it. ``locator`` is a JSON pointer for a record, or ``""`` for a whole file.
    """

    source_path: str
    doc_type: str  # json|pdf|docx|md|txt
    text: str
    title: str | None = None
    locator: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    pages: list[Page] | None = None
    text_provenance: str = "native"  # native|ocr|vision|mixed


@dataclass
class Chunk:
    """A retrievable text window with provenance into its document."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    locator: str = ""  # page number (PDF) or the document's JSON pointer
    token_count: int = 0


@dataclass
class Citation:
    """A tamper-evident anchor: an exact source span plus a verifiable token."""

    doc_id: str
    source_path: str
    locator: str
    char_start: int
    char_end: int
    quote: str
    token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "locator": self.locator,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "quote": self.quote,
            "token": self.token,
        }


@dataclass
class Hit:
    """A search result with its citation, ready for an agent to quote and verify."""

    chunk_id: str
    doc_id: str
    doc_title: str | None
    source_path: str
    locator: str
    score: float
    match_mode: str
    snippet: str
    citation: Citation
    text_provenance: str = "native"  # native|ocr|vision|mixed — how this text was derived

    def to_dict(self) -> dict[str, Any]:
        d = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_path": self.source_path,
            "locator": self.locator,
            "score": round(self.score, 4),
            "match_mode": self.match_mode,
            "snippet": self.snippet,
            "citation": self.citation.to_dict(),
        }
        if self.text_provenance != "native":  # keep native output byte-identical to before
            d["text_provenance"] = self.text_provenance
        return d
