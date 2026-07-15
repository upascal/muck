"""Transcript parser: re-derive a document from its OCR/vision sidecar (not the PDF).

Registered as a normal parser named ``transcript``. When a file has a sidecar
(``.muck/transcripts/<file_id>.json``), both ``pipeline.extract`` and
``cite.reextract_doc_text`` route to this parser (see ``transcript.parser_name_for``), so:

- ``muck parse --all`` re-derives from the sidecar instead of overwriting OCR text with the
  empty text layer of the scan, and
- ``verify`` re-reads the sidecar deterministically, keeping citation hashes valid without
  re-running OCR or a vision model.

The document's ``source_path`` remains the original PDF (what a human opens); this parser
just knows to read the sidecar beside it.
"""

from __future__ import annotations

from pathlib import Path

from ..interfaces.parser import register_parser
from ..schema import ParsedDoc
from ..transcript import doc_from_sidecar, find_sidecar, read_sidecar


class TranscriptParser:
    name = "transcript"
    permissive = True
    handles = (".pdf", ".docx", ".txt", ".md")

    def parse(self, path: str) -> ParsedDoc:
        sc = find_sidecar(path)
        if sc is None:
            raise FileNotFoundError(f"no transcript sidecar for {path!r}")
        return doc_from_sidecar(read_sidecar(sc), title=Path(path).stem)


register_parser("transcript", TranscriptParser)
