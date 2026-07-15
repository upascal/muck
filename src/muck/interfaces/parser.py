"""Parser interface: an unstructured file (PDF/DOCX/text) -> one ParsedDoc."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import ParsedDoc
from . import Registry


@runtime_checkable
class Parser(Protocol):
    name: str
    permissive: bool  # True = BSD/Apache (pdfium); False = AGPL (PyMuPDF) — surfaced in traces
    handles: tuple[str, ...]
    # Optional (default via getattr): page_precise=True means citations resolve to a page;
    # tier ∈ native|ocr|vision describes how the text was derived (default native).

    def parse(self, path: str) -> ParsedDoc: ...


PARSERS: Registry[Parser] = Registry("parser")


def register_parser(name, factory):
    PARSERS.register(name, factory)


def get_parser(name: str) -> Parser:
    return PARSERS.get(name)
