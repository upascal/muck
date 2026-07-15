"""Docling parser (upgrade): OCR + table-aware extraction for scanned/complex PDFs.

Converts a PDF to clean Markdown via Docling (OCR fallback, table structure). Use for
scanned outside-source documents where pdfium/pymupdf yield sparse text. Whole document is
one provenance span (citations resolve to the doc; page-precise spans are a refinement).
Needs the ``docling`` extra. Handles ``.pdf`` plus office formats Docling supports.
"""

from __future__ import annotations

from pathlib import Path

from ..interfaces import NotInstalled
from ..interfaces.parser import register_parser
from ..schema import Page, ParsedDoc


class DoclingParser:
    name = "docling"
    permissive = True
    handles = (".pdf", ".docx", ".pptx", ".html")
    page_precise = False  # collapses to one span — citations resolve to the whole document
    tier = "ocr"  # docling may OCR; treat its output as non-native (verify against source)

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise NotInstalled(
                "Docling parsing needs docling; install `uv sync --extra docling`"
            ) from e
        self._converter = None

    def parse(self, path: str) -> ParsedDoc:
        from docling.document_converter import DocumentConverter

        if self._converter is None:
            self._converter = DocumentConverter()
        result = self._converter.convert(path)
        text = result.document.export_to_markdown()
        title = Path(path).stem
        # One whole-document span (not per-page): citations from a docling doc resolve to the
        # document, not a page. For page-precise scanned-PDF provenance, prefer `muck ocr`.
        return ParsedDoc(
            text=text, pages=[Page(1, 0, len(text), tier="ocr")], title=title,
            text_provenance="ocr",
        )


register_parser("docling", DoclingParser)
