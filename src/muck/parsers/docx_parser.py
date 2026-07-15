"""DOCX parser (optional). Requires the ``docx`` extra: ``uv sync --extra docx``."""

from __future__ import annotations

from pathlib import Path

from ..interfaces import NotInstalled
from ..interfaces.parser import register_parser
from ..schema import Page, ParsedDoc


class DocxParser:
    name = "docx"
    permissive = True
    handles = (".docx",)

    def __init__(self) -> None:
        try:
            import docx  # noqa: F401
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise NotInstalled(
                "DOCX parsing needs python-docx; install with `uv sync --extra docx`"
            ) from e

    def parse(self, path: str) -> ParsedDoc:
        import docx

        document = docx.Document(path)
        text = "\n".join(p.text for p in document.paragraphs)
        title = Path(path).stem
        return ParsedDoc(text=text, pages=[Page(1, 0, len(text))], title=title)


register_parser("docx", DocxParser)
