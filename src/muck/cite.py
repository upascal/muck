"""Citation tokens + verification — the hallucination guard.

A citation token ``{doc_id}@{start}-{end}#{hash}`` is tamper-evident: the hash binds
the exact source span. ``resolve`` re-reads the span from the **original source file**
(re-running the mapper/parser for that one document) and recomputes the hash; ``verify``
additionally checks a claimed quote actually occurs in that span. A token can only pass
verification if its quote really exists at that source location — so a fabricated or
misattributed citation is caught deterministically, without trusting the model.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from .config import Settings
from .schema import Citation

_SEP = "\x1f"
_TOKEN_RE = re.compile(r"^(?P<doc>.+)@(?P<start>\d+)-(?P<end>\d+)#(?P<hash>[0-9a-f]+)$")


def _hash(char_start: int, char_end: int, span: str) -> str:
    return hashlib.sha256(
        f"{char_start}{_SEP}{char_end}{_SEP}{span}".encode()
    ).hexdigest()[:8]


def make_token(doc_id: str, char_start: int, char_end: int, span: str) -> str:
    return f"{doc_id}@{char_start}-{char_end}#{_hash(char_start, char_end, span)}"


def parse_token(token: str) -> tuple[str, int, int, str]:
    m = _TOKEN_RE.match(token.strip())
    if not m:
        raise ValueError(f"malformed citation token: {token!r}")
    return m["doc"], int(m["start"]), int(m["end"]), m["hash"]


def make_citation(
    doc_id: str, source_path: str, locator: str, char_start: int, char_end: int, span: str
) -> Citation:
    return Citation(
        doc_id=doc_id,
        source_path=source_path,
        locator=locator,
        char_start=char_start,
        char_end=char_end,
        quote=span,
        token=make_token(doc_id, char_start, char_end, span),
    )


def _parser_name(doc_type: str, settings: Settings) -> str:
    if doc_type == "pdf":
        return settings.parser.pdf
    if doc_type == "docx":
        return settings.parser.docx
    return settings.parser.text


def _doc_parser_name(row, settings: Settings) -> str:
    """Which parser re-derives this document — pinned to the one recorded at parse time.

    Using ``documents.source_name`` (not the *current* config) keeps existing citations
    valid after a user switches ``[parser] pdf`` — otherwise re-extraction would follow the
    new parser and every prior token's ``source_confirmed`` would flip false. A transcript
    sidecar always wins (an OCR/vision doc must re-derive from its sidecar, deterministically).
    """
    from .transcript import parser_name_for

    recorded = row["source_name"] if _has(row, "source_name") and row["source_name"] else None
    config_default = recorded or _parser_name(row["doc_type"], settings)
    return parser_name_for(row["doc_type"], row["source_path"], config_default)


def _has(row, key: str) -> bool:
    try:
        return key in row.keys()
    except Exception:
        return False


def reextract_doc_text(conn: sqlite3.Connection, settings: Settings, row) -> tuple[str | None, bool]:
    """Re-derive a document's text from its original source file. (text, source_available)."""
    if not Path(row["source_path"]).exists():
        return None, False
    try:
        if row["doc_type"] in ("json", "xml"):
            from .config import fields_for

            fields = fields_for(settings, row["source_path"])  # per-source map, same as at map time
            if row["doc_type"] == "json":
                from .mappers.json_mapper import map_one_text

                return map_one_text(row["source_path"], row["locator"], fields), True
            from .mappers.xml_mapper import map_one_text as xml_map_one_text

            return xml_map_one_text(row["source_path"], row["locator"], fields), True
        from .interfaces.parser import get_parser

        parser = get_parser(_doc_parser_name(row, settings))
        return parser.parse(row["source_path"]).text, True
    except Exception:
        return None, False


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def _muck_dir(conn: sqlite3.Connection) -> Path | None:
    """The corpus's .muck directory, derived from the open DB file's path."""
    try:
        for r in conn.execute("PRAGMA database_list"):
            if r["name"] == "main" and r["file"]:
                return Path(r["file"]).resolve().parent
    except Exception:
        pass
    return None


def _provenance(conn: sqlite3.Connection, row, start: int) -> dict:
    """Provenance overlay for a span: which tier produced it, and where the pixels are.

    For non-native text, a citation must say *what verification proves* (that our transcript
    says X — not that the document does) and point at the page image so a human can check
    the pixels. This is the honest-citation contribution.
    """
    from .quality import page_for_offset
    from .schema import Page
    from .transcript import find_sidecar, page_image_path, read_sidecar

    text_provenance = (row["text_provenance"] if _has(row, "text_provenance") else None) or "native"
    pages = None
    if _has(row, "pages_json") and row["pages_json"]:
        pages = [Page(*p) for p in json.loads(row["pages_json"])]
    page = page_for_offset(pages, start)
    page_no = page.number if page is not None else None
    page_tier = page.tier if page is not None else text_provenance
    scope = "transcript" if text_provenance in ("ocr", "vision") else "source_text"

    out: dict = {
        "text_provenance": text_provenance,
        "verification_scope": scope,
        "page_no": page_no,
    }
    if scope == "source_text":
        out["guarantee"] = (
            "NATIVE: the quote occurs verbatim in the document's own text layer at this span."
        )
        return out

    # OCR/vision: locate the page image and the transcript engine; state the caveat plainly.
    muck_dir = _muck_dir(conn)
    file_id = row["file_id"] if _has(row, "file_id") else None
    engine = None
    source_stale = None
    if file_id and muck_dir and page_no:
        img = page_image_path(muck_dir, file_id, page_no)
        out["page_image"] = str(img) if img.exists() else None
        if not img.exists():
            out["render_hint"] = f"muck render --file {row['source_path']} --pages {page_no}"
    sc = find_sidecar(row["source_path"])
    if sc is not None:
        data = read_sidecar(sc)
        engine = data.get("engine")
        src_sha = data.get("source_sha256")
        if src_sha and Path(row["source_path"]).exists():
            from .transcript import sha256_file

            source_stale = sha256_file(row["source_path"]) != src_sha
    out["transcript_engine"] = engine
    out["source_stale"] = source_stale
    out["human_review_required"] = page_tier == "vision" or text_provenance == "vision"
    if out["human_review_required"]:
        out["guarantee"] = (
            f"VISION ({engine or 'LLM'}): the quote occurs in an LLM transcription of the page "
            "image — an LLM CAN fabricate text, so this is NOT source-verified. Open page_image "
            "and confirm before relying on it."
        )
    else:
        out["guarantee"] = (
            f"OCR ({engine or 'engine'}): the quote occurs in a deterministic OCR transcript of "
            "the page image. OCR can misread characters but cannot invent sentences — confirm "
            "against page_image before publishing."
        )
    return out


def resolve(conn: sqlite3.Connection, settings: Settings, token: str, context: int = 160) -> dict:
    doc_id, start, end, h = parse_token(token)
    row = conn.execute(
        "SELECT doc_id, file_id, source_path, locator, doc_type, title, text, source_name, "
        "pages_json, text_provenance FROM documents WHERE doc_id=?",
        (doc_id,),
    ).fetchone()
    if row is None:
        return {"token": token, "valid": False, "error": f"unknown doc_id {doc_id!r}"}

    stored = row["text"]
    span = stored[start:end]
    token_valid = _hash(start, end, span) == h

    src_text, src_available = reextract_doc_text(conn, settings, row)
    source_confirmed = None
    if src_available and src_text is not None:
        source_confirmed = src_text[start:end] == span

    return {
        "token": token,
        "doc_id": doc_id,
        "source_path": row["source_path"],
        "locator": row["locator"],
        "doc_title": row["title"],
        "char_start": start,
        "char_end": end,
        "span": span,
        "context_before": stored[max(0, start - context):start],
        "context_after": stored[end:end + context],
        "token_valid": token_valid,
        "source_available": src_available,
        "source_confirmed": source_confirmed,
        "valid": token_valid and (source_confirmed is not False),
        **_provenance(conn, row, start),
    }


def verify(conn: sqlite3.Connection, settings: Settings, token: str, quote: str) -> dict:
    r = resolve(conn, settings, token)
    if not r.get("token_valid"):
        return {**r, "quote": quote, "quote_supported": False, "verified": False}
    quote_supported = _norm(quote) in _norm(r["span"]) if quote else False
    source_ok = r.get("source_confirmed") is not False
    scope = r.get("verification_scope", "source_text")
    if r.get("human_review_required"):
        # Vision transcripts can be fabricated: quote-in-transcript is necessary but NOT
        # sufficient. `verified` stays False until a pixel review confirms the page image.
        return {
            **r, "quote": quote, "quote_supported": quote_supported,
            "verified": False, "needs_pixel_review": True,
        }
    verified = bool(quote_supported and source_ok)
    out = {**r, "quote": quote, "quote_supported": quote_supported, "verified": verified}
    if scope == "transcript" and verified:
        out["caveat"] = "machine_misread_possible"  # deterministic OCR: char errors, not fabrication
    return out
