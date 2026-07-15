"""Transcript sidecars — the deterministic source of record for non-native text.

When muck OCRs or an agent vision-transcribes a scanned PDF, the result is written to a
JSON sidecar under ``.muck/transcripts/<file_id>.json`` (never ``.muck/cache/`` — this is
evidence, not a cache). The original PDF stays the document's ``source_path`` (what a human
opens); the rendered page images under ``.muck/pages/<file_id>/`` are the pixel ground
truth; the sidecar is the *deterministic* text a citation re-derives from.

That determinism is the whole point: ``cite.verify`` re-reads the sidecar (a plain file on
disk) and gets byte-identical text, so the tamper-evident hash still holds — without
re-running tesseract or an LLM. The sidecar records the source PDF's hash at transcription
time, so a later edit to the PDF is detectable (``source_stale``).

Routing: once a sidecar exists for a file, both ``parse`` and ``cite`` must use the
``transcript`` parser for it — otherwise ``muck parse --all`` would re-run pdfium over the
scan and overwrite the OCR text with ``""`` (silent data loss). ``parser_name_for`` is the
single rule both call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import MUCK_DIRNAME
from .schema import Page, ParsedDoc

SCHEMA_TAG = "muck.transcript.v1"
PAGE_SEP = "\n\n"  # must match pdfium/pymupdf so re-index offsets line up


# --- paths -------------------------------------------------------------------

def file_id_for(path: str | Path) -> str:
    """The file_id muck assigns to a source file — content-addressed via the shared
    ``pipeline.content_file_id`` (lazy import avoids a module cycle). Kept in lockstep with
    ingest so ``pages/<id>/`` images and ``transcripts/<id>.json`` sidecars resolve."""
    from .pipeline import content_file_id

    return content_file_id(path)


def pages_dir(muck_dir: Path, file_id: str) -> Path:
    return muck_dir / "pages" / file_id


def page_image_path(muck_dir: Path, file_id: str, page_no: int) -> Path:
    return pages_dir(muck_dir, file_id) / f"page_{page_no:04d}.png"


def transcripts_dir(muck_dir: Path) -> Path:
    return muck_dir / "transcripts"


def sidecar_path(muck_dir: Path, file_id: str) -> Path:
    return transcripts_dir(muck_dir) / f"{file_id}.json"


def _muck_dir_for(source_path: str | Path) -> Path | None:
    """Walk up from a source file to its corpus's ``.muck`` directory."""
    cur = Path(source_path).resolve().parent
    for d in [cur, *cur.parents]:
        candidate = d / MUCK_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def find_sidecar(source_path: str | Path) -> Path | None:
    """The sidecar for a source file, if one has been written."""
    muck_dir = _muck_dir_for(source_path)
    if muck_dir is None:
        return None
    p = sidecar_path(muck_dir, file_id_for(source_path))
    return p if p.exists() else None


def has_sidecar(source_path: str | Path) -> bool:
    return find_sidecar(source_path) is not None


# --- read / write ------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def write_sidecar(
    muck_dir: Path,
    file_id: str,
    source_path: str,
    text_provenance: str,
    engine: str,
    pages: list[dict],
    params: dict | None = None,
    created_at: str | None = None,
) -> Path:
    """Write a transcript sidecar. ``pages`` items: {page_no, tier, page_class, text, image?, mean_conf?}."""
    transcripts_dir(muck_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_TAG,
        "file_id": file_id,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path) if Path(source_path).exists() else None,
        "text_provenance": text_provenance,
        "engine": engine,
        "params": params or {},
        "page_sep": PAGE_SEP,
        "pages": pages,
        "created_at": created_at,
    }
    out = sidecar_path(muck_dir, file_id)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out


def read_sidecar(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def doc_from_sidecar(sidecar: dict, title: str | None = None) -> ParsedDoc:
    """Rebuild a ParsedDoc from a sidecar — pure, deterministic concatenation.

    Page spans are recomputed the same way pdfium builds them, so a document re-derived
    from the sidecar is byte-identical every time (what keeps citation hashes valid).
    """
    sep = sidecar.get("page_sep", PAGE_SEP)
    parts: list[str] = []
    page_objs: list[Page] = []
    pos = 0
    for i, pg in enumerate(sidecar.get("pages", [])):
        content = pg.get("text") or ""
        if i > 0:
            parts.append(sep)
            pos += len(sep)
        start = pos
        parts.append(content)
        pos += len(content)
        page_objs.append(
            Page(pg["page_no"], start, pos, pg.get("page_class", "text"), pg.get("tier", "ocr"))
        )
    return ParsedDoc(
        text="".join(parts),
        pages=page_objs,
        title=title,
        text_provenance=sidecar.get("text_provenance", "ocr"),
    )


# --- routing -----------------------------------------------------------------

def commit_transcript(
    conn,
    muck_dir: Path,
    file_id: str,
    source_path: str,
    text_provenance: str,
    engine: str,
    pages_meta: list[dict],
    params: dict | None = None,
    created_at: str | None = None,
):
    """Write a sidecar and replace the document's native (empty) extraction with it.

    Shared by the OCR and vision paths: writes the sidecar, re-derives the document from it
    (via the deterministic rebuild), clears the old chunks, and re-inserts the document as
    ``status='extracted'`` so ``muck index`` re-chunks the transcript. Returns the ParsedDoc.
    """
    from .pipeline import _clear_file, _insert_document, _now
    from .schema import Record

    write_sidecar(
        muck_dir, file_id, source_path, text_provenance, engine, pages_meta, params,
        created_at or _now(),
    )
    pd = doc_from_sidecar(read_sidecar(sidecar_path(muck_dir, file_id)), title=Path(source_path).stem)
    _clear_file(conn, file_id)
    _insert_document(
        conn, file_id, 0,
        Record(source_path=str(source_path), doc_type="pdf", text=pd.text,
               title=pd.title, pages=pd.pages, text_provenance=text_provenance),
        source_name="transcript",
    )
    conn.execute("UPDATE source_files SET status='processed', processed_at=? WHERE file_id=?",
                 (_now(), file_id))
    conn.commit()
    return pd


def parser_name_for(doc_type: str, source_path: str, config_name: str) -> str:
    """Parser to use for a file: the transcript sidecar if present, else the config choice.

    ``config_name`` is the parser the config would otherwise select (e.g. "pdfium"). Both
    ``pipeline.extract`` and ``cite.reextract_doc_text`` consult this so an OCR/vision
    transcript is never clobbered by a re-parse and always re-verifies deterministically.
    """
    if doc_type in ("pdf", "docx", "md", "txt") and has_sidecar(source_path):
        return "transcript"
    return config_name
