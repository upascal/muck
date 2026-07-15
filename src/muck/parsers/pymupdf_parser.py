"""PyMuPDF parser (upgrade): richer extraction than pdfium, incl. light page classes.

Flagged ``permissive=False`` because PyMuPDF is AGPL — an explicit opt-in surfaced in the
manifest (``documents.parser_name``) and traces. Needs the ``pymupdf`` extra.
"""

from __future__ import annotations

from pathlib import Path

from ..interfaces import NotInstalled
from ..interfaces.parser import register_parser
from ..quality import classify_page as _page_class
from ..schema import Page, ParsedDoc

PAGE_SEP = "\n\n"


class PyMuPdfParser:
    name = "pymupdf"
    permissive = False
    handles = (".pdf",)

    def __init__(self) -> None:
        try:
            import fitz  # noqa: F401  (pymupdf)
        except ImportError as e:  # pragma: no cover
            raise NotInstalled(
                "PyMuPDF parsing needs pymupdf; install `uv sync --extra pymupdf`"
            ) from e

    def parse(self, path: str) -> ParsedDoc:
        import fitz

        doc = fitz.open(path)
        try:
            parts: list[str] = []
            pages: list[Page] = []
            pos = 0
            for i in range(doc.page_count):
                content = doc[i].get_text("text") or ""
                if i > 0:
                    parts.append(PAGE_SEP)
                    pos += len(PAGE_SEP)
                start = pos
                parts.append(content)
                pos += len(content)
                pages.append(Page(i + 1, start, pos, _page_class(content)))
            text = "".join(parts)
            title = doc.metadata.get("title") or Path(path).stem
        finally:
            doc.close()
        return ParsedDoc(text=text, pages=pages, title=title)


register_parser("pymupdf", PyMuPdfParser)
