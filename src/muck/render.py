"""Rasterize PDF pages to PNGs — the pixel ground truth for cite-to-pixels.

Uses pypdfium2 (already a core dependency) to render each page, and writes PNGs with a tiny
stdlib ``zlib`` encoder — so page rendering needs no new dependency (no Pillow, no poppler).
Images land at ``.muck/pages/<file_id>/page_NNNN.png`` and are what a human reviewer (and an
agent's own vision) look at, and what a citation on OCR/vision text resolves to.

Idempotent: a page already rendered is skipped unless ``force``.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from .transcript import file_id_for, page_image_path, pages_dir

DEFAULT_DPI = 200


def _write_png_gray(arr, path: Path) -> None:
    """Write a 2-D uint8 grayscale numpy array as an 8-bit grayscale PNG (stdlib only)."""
    h, w = arr.shape
    # Each scanline is prefixed with a filter byte (0 = none), then raw row bytes.
    raw = bytearray()
    row_bytes = arr.tobytes()
    for y in range(h):
        raw.append(0)
        raw.extend(row_bytes[y * w:(y + 1) * w])

    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit, color type 0 (grayscale)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def render_pdf(
    muck_dir: Path,
    source_path: str,
    dpi: int = DEFAULT_DPI,
    only_pages: list[int] | None = None,
    force: bool = False,
) -> dict:
    """Render a PDF's pages to PNGs under ``.muck/pages/<file_id>/``.

    ``only_pages`` is 1-based page numbers (None = all). Returns the file_id, dpi, and the
    list of pages now present on disk.
    """
    import pypdfium2 as pdfium

    file_id = file_id_for(source_path)
    out_dir = pages_dir(muck_dir, file_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(source_path)
    rendered, skipped = [], []
    try:
        n = len(pdf)
        wanted = only_pages or range(1, n + 1)
        for page_no in wanted:
            if not (1 <= page_no <= n):
                continue
            img_path = page_image_path(muck_dir, file_id, page_no)
            if img_path.exists() and not force:
                skipped.append(page_no)
                continue
            bitmap = pdf[page_no - 1].render(scale=dpi / 72, grayscale=True)
            _write_png_gray(bitmap.to_numpy(), img_path)
            rendered.append(page_no)
    finally:
        pdf.close()

    present = sorted(
        int(p.stem.split("_")[1]) for p in out_dir.glob("page_*.png")
    )
    return {
        "file_id": file_id,
        "source_path": str(source_path),
        "dpi": dpi,
        "rendered": rendered,
        "skipped": skipped,
        "pages_present": present,
    }
