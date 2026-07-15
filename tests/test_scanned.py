"""End-to-end tests for the scanned/image-only document path.

The rest of the suite fakes PDFs with hand-written strings; these build *real* image-only
PDFs at test time (a tiny raw-PDF writer — no new deps) so the whole chain is exercised:
detection of image-only scans, deterministic OCR/vision transcripts, page-precise citations,
the honest verification tiers, and the `parse --all` data-loss guard.

Offline by default; the one test that shells out to `tesseract` skips when it's absent.
"""

from __future__ import annotations

import shutil
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from muck import cite as citelib
from muck import findings as findingslib
from muck import pipeline
from muck import transcript as tr
from muck.config import Settings
from muck.db import DB_FILENAME, connect, init_schema
from muck.quality import classify_page, coverage_verdict, page_coverage
from muck.schema import Page

# Import adapters for their registration side effects.
from muck import parsers as _parsers  # noqa: F401
from muck import stores as _stores  # noqa: F401
from muck import embedders as _embedders  # noqa: F401


# --- raw PDF builders (no Pillow / no reportlab) -----------------------------

def _pdf(objects: list[bytes]) -> bytes:
    """Assemble numbered PDF objects into a valid document with an xref table."""
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<</Size " + str(len(objects) + 1).encode()
        + b"/Root 1 0 R>>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    )
    return bytes(out)


def _text_pdf(pages_text: list[str]) -> bytes:
    """A normal PDF with a real text layer (used to raster 'scanned' pages)."""
    kids, objs, next_id = [], [], 4 + 2 * len(pages_text)
    page_objs, content_objs = [], []
    oid = 4
    for text in pages_text:
        stream = f"BT /F1 18 Tf 72 720 Td 20 TL ".encode()
        for line in text.split("\n"):
            safe = line.replace("(", r"\(").replace(")", r"\)")
            stream += f"({safe}) Tj T*\n".encode()
        stream += b"ET"
        content = b"<</Length " + str(len(stream)).encode() + b">>\nstream\n" + stream + b"\nendstream"
        content_objs.append(content)
    n = len(pages_text)
    # Object order: 1 catalog, 2 pages, 3 font, then per page: page obj + content obj.
    page_ids = [4 + 2 * i for i in range(n)]
    content_ids = [5 + 2 * i for i in range(n)]
    catalog = b"<</Type/Catalog/Pages 2 0 R>>"
    pages = ("<</Type/Pages/Kids[" + " ".join(f"{pid} 0 R" for pid in page_ids)
             + f"]/Count {n}>>").encode()
    font = b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"
    objs = [catalog, pages, font]
    for i in range(n):
        page = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
                f"/Contents {content_ids[i]} 0 R/Resources<</Font<</F1 3 0 R>>>>>>").encode()
        objs.append(page)
        objs.append(content_objs[i])
    return _pdf(objs)


def _image_only_pdf3(arrays: list[np.ndarray]) -> bytes:
    """Image-only PDF, 3 objects per page: page, content stream, image XObject."""
    n = len(arrays)
    catalog = b"<</Type/Catalog/Pages 2 0 R>>"
    objs: list[bytes] = [catalog, b"", b""]  # 1 catalog; 2 pages (filled below); 3 spare
    page_ids, per = [], []
    for i in range(n):
        base = 4 + 3 * i
        page_id, content_id, img_id = base, base + 1, base + 2
        page_ids.append(page_id)
        h, w = arrays[i].shape
        page = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {w} {h}]"
                f"/Contents {content_id} 0 R/Resources<</XObject<</Im0 {img_id} 0 R>>>>>>").encode()
        draw = f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode()
        content = b"<</Length " + str(len(draw)).encode() + b">>\nstream\n" + draw + b"\nendstream"
        raster = zlib.compress(arrays[i].tobytes(), 6)
        img = (f"<</Type/XObject/Subtype/Image/Width {w}/Height {h}/ColorSpace/DeviceGray"
               f"/BitsPerComponent 8/Filter/FlateDecode/Length {len(raster)}>>").encode()
        img = img + b"\nstream\n" + raster + b"\nendstream"
        per.append((page, content, img))
    objs[1] = ("<</Type/Pages/Kids[" + " ".join(f"{p} 0 R" for p in page_ids)
               + f"]/Count {n}>>").encode()
    for page, content, img in per:
        objs.extend([page, content, img])
    return _pdf(objs)


def _render_text_to_arrays(pages_text: list[str], scale: float = 1.0) -> list[np.ndarray]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(_text_pdf(pages_text))
    arrays = [pdf[i].render(scale=scale, grayscale=True).to_numpy() for i in range(len(pdf))]
    pdf.close()
    return arrays


def make_blank_scan(path: Path, n_pages: int = 2) -> None:
    """An image-only PDF with blank pages — no text layer at all (defect-1 fixture)."""
    arrays = [np.full((200, 150), 255, dtype=np.uint8) for _ in range(n_pages)]
    path.write_bytes(_image_only_pdf3(arrays))


def make_text_scan(path: Path, pages_text: list[str]) -> None:
    """An image-only PDF that *looks* like text but has no text layer (OCR fixture)."""
    arrays = _render_text_to_arrays(pages_text, scale=2.0)
    path.write_bytes(_image_only_pdf3(arrays))


# --- fixtures ----------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / ".muck").mkdir(parents=True)
    conn = connect(root / ".muck" / DB_FILENAME)
    init_schema(conn)
    return root, Settings(embedder={"enabled": False}), conn


def _ingest_parse(conn, settings, root, pdf_path):
    pipeline.ingest(conn, [str(pdf_path)])
    return pipeline.extract(conn, settings, pipeline.PARSER_TYPES, only_new=True)


# --- quality unit tests ------------------------------------------------------

def test_classify_page_labels_blank_and_text():
    assert classify_page("") == "empty"
    assert classify_page("   \n  ") == "empty"
    assert classify_page("The quick brown fox jumped over the lazy dog again.") == "text"


def test_coverage_verdict_thresholds():
    assert coverage_verdict(1.0) == "native"
    assert coverage_verdict(0.3) == "sparse"
    assert coverage_verdict(0.0) == "image_only"


def test_page_coverage_counts_pages_with_text():
    pages = [Page(1, 0, 0, "empty"), Page(2, 2, 400, "text")]
    cov = page_coverage(pages, "x" * 402)
    assert cov["pages"] == 2 and cov["pages_with_text"] == 1
    assert cov["coverage"] == 0.5


# --- defect 1: an image-only scan must NOT look healthy ----------------------

def test_image_only_scan_is_flagged_not_silent(corpus):
    root, settings, conn = corpus
    scan = root / "blank.pdf"
    make_blank_scan(scan, n_pages=3)
    res = _ingest_parse(conn, settings, root, scan)
    # Parsed "successfully" but with a coverage notice — the old silent path is gone.
    assert res["documents"] == 1
    assert "notice" in res and "image-only" in res["notice"]
    doc = conn.execute("SELECT coverage, text, pages_json FROM documents").fetchone()
    assert doc["coverage"] == 0.0
    assert doc["text"].strip() == ""
    # Pages are labelled 'empty', not the old default 'text'.
    import json
    classes = {p[3] for p in json.loads(doc["pages_json"])}
    assert classes == {"empty"}


def test_coverage_report_recommends_ocr(corpus):
    root, settings, conn = corpus
    scan = root / "blank.pdf"
    make_blank_scan(scan, n_pages=2)
    _ingest_parse(conn, settings, root, scan)
    cov = pipeline.coverage(conn, settings)
    assert cov["corpus"]["image_only"] == 1
    assert cov["files"][0]["verdict"] == "image_only"
    assert cov["files"][0]["recommend"] == "ocr"


# --- migration: an old DB gets the new columns -------------------------------

def test_migration_adds_columns_to_legacy_db(tmp_path):
    import sqlite3
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    # A pre-migration documents table (no text_provenance/coverage), with the columns the
    # schema's indexes reference so executescript's CREATE INDEX IF NOT EXISTS succeeds.
    conn.execute(
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, file_id TEXT, "
        "text TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'extracted')"
    )
    conn.execute("CREATE TABLE findings (finding_id TEXT PRIMARY KEY, status TEXT DEFAULT 'unaudited')")
    conn.commit()
    init_schema(conn)  # runs _migrate
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
    assert {"text_provenance", "coverage"} <= cols
    fcols = {r["name"] for r in conn.execute("PRAGMA table_info(findings)")}
    assert "text_provenance" in fcols


# --- sidecar determinism + transcript parser ---------------------------------

def test_sidecar_rebuild_is_deterministic_and_page_precise():
    sidecar = {
        "page_sep": "\n\n",
        "text_provenance": "ocr",
        "pages": [
            {"page_no": 1, "tier": "ocr", "page_class": "text", "text": "First page body."},
            {"page_no": 2, "tier": "ocr", "page_class": "text", "text": "Second page body."},
        ],
    }
    pd1 = tr.doc_from_sidecar(sidecar)
    pd2 = tr.doc_from_sidecar(sidecar)
    assert pd1.text == pd2.text  # deterministic
    assert pd1.text_provenance == "ocr"
    # Page spans bracket the right substrings → page-precise citations.
    p1, p2 = pd1.pages
    assert pd1.text[p1.char_start:p1.char_end] == "First page body."
    assert pd1.text[p2.char_start:p2.char_end] == "Second page body."
    assert p1.tier == "ocr"


# --- vision honesty chain (no tesseract needed) ------------------------------

def _write_vision_transcript(root, settings, conn, scan, pages_text):
    """Simulate the agent: render nothing (fake a page image), write a vision sidecar."""
    muck_dir = root / ".muck"
    file_id = tr.file_id_for(scan)
    # A minimal page image so cite's page_image path resolves.
    (muck_dir / "pages" / file_id).mkdir(parents=True, exist_ok=True)
    from muck.render import _write_png_gray
    for i in range(len(pages_text)):
        _write_png_gray(np.full((20, 20), 255, np.uint8),
                        tr.page_image_path(muck_dir, file_id, i + 1))
    pages_meta = [
        {"page_no": i + 1, "tier": "vision", "page_class": "text",
         "text": t, "image": f"pages/{file_id}/page_{i+1:04d}.png"}
        for i, t in enumerate(pages_text)
    ]
    tr.commit_transcript(conn, muck_dir, file_id, str(scan), "vision",
                         "test-model (agent vision)", pages_meta)
    return file_id


def test_vision_finding_needs_pixel_review_then_confirms(corpus):
    root, settings, conn = corpus
    scan = root / "emails.pdf"
    make_blank_scan(scan, n_pages=2)  # only needs a source_files/doc row
    _ingest_parse(conn, settings, root, scan)
    _write_vision_transcript(root, settings, conn, scan,
                             ["Background Document and Draft Letter", "Second page text here."])
    pipeline.run_index(conn, settings, only_new=False)

    doc = conn.execute("SELECT doc_id, text_provenance FROM documents").fetchone()
    assert doc["text_provenance"] == "vision"
    # Build a citation token for the first-page span.
    chunk = conn.execute(
        "SELECT chunk_id, char_start, char_end, text FROM chunks ORDER BY char_start"
    ).fetchone()
    token = citelib.make_token(doc["doc_id"], chunk["char_start"], chunk["char_end"], chunk["text"])
    quote = "Background Document and Draft Letter"

    v = citelib.verify(conn, settings, token, quote)
    assert v["quote_supported"] is True       # the transcript contains it...
    assert v["verified"] is False              # ...but vision is NOT source-verified
    assert v["needs_pixel_review"] is True
    assert v["verification_scope"] == "transcript"
    assert v["human_review_required"] is True
    assert v["page_image"] and Path(v["page_image"]).exists()

    add = findingslib.add_finding(conn, settings, "claim about screening", quote, token)
    fid = add["finding_id"]
    assert add["status"] == "pending_review"
    assert add["text_provenance"] == "vision"

    a = findingslib.audit(conn, settings)
    assert a["pending_review"] == 1 and a["verified"] == 0
    assert "pixel_review_queue" in a

    r = findingslib.review(conn, settings, fid, "confirmed", reviewer="tester")
    assert r["status"] == "verified"
    a2 = findingslib.audit(conn, settings)
    assert a2["verified"] == 1 and a2["pending_review"] == 0


def test_vision_review_reject_marks_finding_rejected(corpus):
    root, settings, conn = corpus
    scan = root / "emails.pdf"
    make_blank_scan(scan, n_pages=1)
    _ingest_parse(conn, settings, root, scan)
    _write_vision_transcript(root, settings, conn, scan, ["A fabricated-looking sentence."])
    pipeline.run_index(conn, settings, only_new=False)
    doc = conn.execute("SELECT doc_id FROM documents").fetchone()
    ch = conn.execute("SELECT char_start, char_end, text FROM chunks").fetchone()
    token = citelib.make_token(doc["doc_id"], ch["char_start"], ch["char_end"], ch["text"])
    add = findingslib.add_finding(conn, settings, "claim", "A fabricated-looking sentence.", token)
    r = findingslib.review(conn, settings, add["finding_id"], "rejected", reviewer="tester")
    assert r["status"] == "rejected"


# --- parse --all must not clobber a transcript (data-loss guard) --------------

def test_parse_all_preserves_transcript(corpus):
    root, settings, conn = corpus
    scan = root / "scan.pdf"
    make_blank_scan(scan, n_pages=2)
    _ingest_parse(conn, settings, root, scan)
    file_id = _write_vision_transcript(root, settings, conn, scan, ["Real transcript page one.", "Page two."])
    before = conn.execute("SELECT text, text_provenance FROM documents").fetchone()
    assert before["text_provenance"] == "vision"
    # Re-parse everything: the sidecar-aware router must re-derive from the transcript,
    # not re-run pdfium and overwrite it with the scan's empty text layer.
    pipeline.extract(conn, settings, pipeline.PARSER_TYPES, only_new=False)
    after = conn.execute("SELECT text, text_provenance FROM documents").fetchone()
    assert after["text"] == before["text"]
    assert after["text_provenance"] == "vision"


# --- item 0: cite pins to the recorded parser, not current config ------------

def test_cite_pins_parser_to_document(corpus):
    root, settings, conn = corpus
    scan = root / "scan.pdf"
    make_blank_scan(scan, n_pages=1)
    _ingest_parse(conn, settings, root, scan)
    _write_vision_transcript(root, settings, conn, scan, ["Some transcribed text here."])
    row = conn.execute(
        "SELECT source_name, source_path, doc_type FROM documents"
    ).fetchone()
    assert row["source_name"] == "transcript"
    # Even with the PDF config parser set to pdfium, re-extraction follows the sidecar.
    name = citelib._doc_parser_name(row, settings)
    assert name == "transcript"


# --- real OCR round-trip (needs the tesseract binary) ------------------------

@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_ocr_roundtrip_page_precise(corpus):
    from muck import ocr as ocr_mod

    root, settings, conn = corpus
    scan = root / "scanned_text.pdf"
    make_text_scan(scan, ["INVOICE NUMBER 12345", "TOTAL DUE 500 DOLLARS"])
    _ingest_parse(conn, settings, root, scan)
    assert conn.execute("SELECT coverage FROM documents").fetchone()["coverage"] == 0.0

    muck_dir = root / ".muck"
    out = ocr_mod.ocr_file(conn, settings, muck_dir, str(scan), dpi=150)
    assert out["pages"] == 2 and out["chars"] > 0
    doc = conn.execute("SELECT doc_id, text, text_provenance, pages_json FROM documents").fetchone()
    assert doc["text_provenance"] == "ocr"
    assert "INVOICE" in doc["text"].upper()

    # Page-precise spans: the offset of 'TOTAL' (page 2 content) maps to page 2, not page 1.
    import json as _json
    from muck.quality import page_for_offset
    pages = [Page(*p) for p in _json.loads(doc["pages_json"])]
    total_off = doc["text"].upper().index("TOTAL")
    invoice_off = doc["text"].upper().index("INVOICE")
    assert page_for_offset(pages, invoice_off).number == 1
    assert page_for_offset(pages, total_off).number == 2

    pipeline.run_index(conn, settings, only_new=False)
    ch = conn.execute("SELECT char_start, char_end, text, locator FROM chunks ORDER BY char_start").fetchall()
    token = citelib.make_token(doc["doc_id"], ch[0]["char_start"], ch[0]["char_end"], ch[0]["text"])
    r = citelib.resolve(conn, settings, token)
    assert r["token_valid"] and r["source_confirmed"]   # re-derives from the sidecar
    assert r["text_provenance"] == "ocr"
    assert r["verification_scope"] == "transcript"
    assert r["page_image"] and Path(r["page_image"]).exists()
