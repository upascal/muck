"""Parsers: unstructured files -> ParsedDoc. Defaults: text, pdfium. Upgrades register lazily."""

from __future__ import annotations

from . import pdfium_parser as _pdfium  # noqa: F401  (registers "pdfium")
from . import text_parser as _text  # noqa: F401  (registers "text")
from . import transcript_parser as _transcript  # noqa: F401  (registers "transcript")
from . import docx_parser as _docx  # noqa: F401  (registers "docx", lazy dep)
from . import pymupdf_parser as _pymupdf  # noqa: F401  (registers "pymupdf", lazy dep)
from . import docling_parser as _docling  # noqa: F401  (registers "docling", lazy dep)
