"""Deterministic OCR: render page images, run tesseract, write a transcript sidecar.

tesseract is called as a subprocess (no ``pytesseract`` dependency). Its TSV output gives
per-word confidence for free, so each page carries a ``mean_conf`` — the signal that lets
muck *refuse* to pass off garbage (handwriting, heavy noise) as clean text and route it to
the vision path instead. The result is written to a transcript sidecar (see ``transcript``),
which becomes the deterministic source of record; ``muck index`` then re-chunks from it.

Requires the ``tesseract`` binary on PATH (``NotInstalled`` otherwise) — the usual muck
idiom for an optional capability that isn't a Python package.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import Settings
from .interfaces import NotInstalled
from .quality import classify_page
from .render import DEFAULT_DPI, render_pdf
from .transcript import commit_transcript, file_id_for, page_image_path

# Below this mean word-confidence, an OCR page is treated as unreliable (likely handwriting
# or heavy noise): classified bad_ocr so coverage/verify surface it for the vision route.
LOW_CONF = 45.0


def _engine_id() -> str:
    try:
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        first = (out.stdout or out.stderr).splitlines()[0].strip()
        return first or "tesseract"
    except Exception:
        return "tesseract"


def _require_tesseract() -> None:
    if shutil.which("tesseract") is None:
        raise NotInstalled(
            "OCR needs the `tesseract` binary on PATH "
            "(macOS: `brew install tesseract`; Debian: `apt install tesseract-ocr`)."
        )


def _ocr_image(img_path: Path, lang: str, psm: int) -> tuple[str, float]:
    """Run tesseract on one image → (layout-preserving text, mean word confidence)."""
    proc = subprocess.run(
        ["tesseract", str(img_path), "stdout", "-l", lang, "--psm", str(psm), "tsv"],
        capture_output=True, text=True,
    )
    lines: dict[tuple, list[tuple[int, str]]] = {}
    confs: list[float] = []
    for row in proc.stdout.splitlines()[1:]:  # skip header
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":  # level 5 = word
            continue
        block, par, line, word = cols[2], cols[3], cols[4], int(cols[5])
        conf, text = float(cols[10]), cols[11]
        if not text.strip():
            continue
        lines.setdefault((block, par, line), []).append((word, text))
        if conf >= 0:
            confs.append(conf)
    # Reassemble in reading order; blank line between paragraphs to keep some structure.
    out_lines: list[str] = []
    prev_par = None
    for (block, par, line) in sorted(lines, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        if prev_par is not None and (block, par) != prev_par:
            out_lines.append("")
        prev_par = (block, par)
        words = [w for _, w in sorted(lines[(block, par, line)])]
        out_lines.append(" ".join(words))
    text = "\n".join(out_lines).strip()
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf


def ocr_file(
    conn,
    settings: Settings,
    muck_dir: Path,
    source_path: str,
    dpi: int = DEFAULT_DPI,
    lang: str = "eng",
    psm: int = 3,
    force: bool = False,
) -> dict:
    """OCR one PDF: render → tesseract per page → sidecar → re-extract the document."""
    _require_tesseract()
    file_id = file_id_for(source_path)
    render_pdf(muck_dir, source_path, dpi=dpi, force=force)

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(source_path)
    n = len(pdf)
    pdf.close()

    engine = _engine_id()
    pages_meta: list[dict] = []
    low_conf_pages: list[int] = []
    for page_no in range(1, n + 1):
        img = page_image_path(muck_dir, file_id, page_no)
        text, conf = _ocr_image(img, lang, psm)
        page_class = classify_page(text)
        if conf < LOW_CONF and page_class == "text":
            page_class = "bad_ocr"
            low_conf_pages.append(page_no)
        pages_meta.append({
            "page_no": page_no, "tier": "ocr", "page_class": page_class,
            "text": text, "image": str(img.relative_to(muck_dir)),
            "mean_conf": round(conf, 1),
        })

    pd = commit_transcript(
        conn, muck_dir, file_id, source_path, text_provenance="ocr", engine=engine,
        pages_meta=pages_meta, params={"dpi": dpi, "lang": lang, "psm": psm},
    )

    mean_conf = sum(p["mean_conf"] for p in pages_meta) / max(1, len(pages_meta))
    return {
        "file": str(source_path), "file_id": file_id, "pages": n,
        "engine": engine, "chars": len(pd.text), "mean_conf": round(mean_conf, 1),
        "low_conf_pages": low_conf_pages, "next": "muck index",
    }
