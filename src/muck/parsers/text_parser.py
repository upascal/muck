"""Plain-text / Markdown parser. Whole file is one page; provenance = char offsets."""

from __future__ import annotations

from pathlib import Path

from ..interfaces.parser import register_parser
from ..schema import Page, ParsedDoc


class TextParser:
    name = "text"
    permissive = True
    handles = (".txt", ".md", ".markdown", ".text")

    def parse(self, path: str) -> ParsedDoc:
        text = Path(path).read_text(errors="replace")
        title = next((ln.strip().lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), None)
        return ParsedDoc(text=text, pages=[Page(1, 0, len(text))], title=title)


register_parser("text", TextParser)
