"""Default PDF parser: pypdfium2 (BSD/Apache — no AGPL trap).

Extracts page text in reading order and records each page's [char_start, char_end)
span within the concatenated document text, so every chunk resolves to a page.
"""

from __future__ import annotations

from pathlib import Path

from ..interfaces.parser import register_parser
from ..quality import classify_page
from ..schema import Page, ParsedDoc

PAGE_SEP = "\n\n"


class PdfiumParser:
    name = "pdfium"
    permissive = True
    handles = (".pdf",)

    def parse(self, path: str) -> ParsedDoc:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        try:
            parts: list[str] = []
            pages: list[Page] = []
            pos = 0
            for i in range(len(pdf)):
                page = pdf[i]
                textpage = page.get_textpage()
                content = textpage.get_text_range() or ""
                if i > 0:
                    parts.append(PAGE_SEP)
                    pos += len(PAGE_SEP)
                start = pos
                parts.append(content)
                pos += len(content)
                # Classify honestly: a page with no text layer is "empty", not "text".
                # This is what surfaces an image-only scan instead of hiding it.
                pages.append(Page(i + 1, start, pos, classify_page(content)))
            text = "".join(parts)
        finally:
            pdf.close()
        title = Path(path).stem
        return ParsedDoc(text=text, pages=pages, title=title)


register_parser("pdfium", PdfiumParser)
