"""Vision transcription protocol — the agent's own multimodal read, ingested honestly.

muck never calls a vision model itself (it stays no-API-key and deterministic). Instead the
*agent* — which already has vision — is the transcriber, and muck hands it a work queue and
ingests what it writes back:

1. ``muck render --file scan.pdf``           → page images under .muck/pages/<file_id>/
2. agent Reads each page_NNNN.png and writes .muck/transcripts/<file_id>/page_NNNN.md
3. ``muck transcribe import --file scan.pdf --model <id>``  → assembles the sidecar

``import`` validates that a rendered image exists for every transcribed page, so a vision
transcript is anchored to pixels the model was actually shown. The resulting text is tier
``vision``: a citation on it is honest that an LLM *can* fabricate, so it is never
source-verified — it needs a pixel review (``muck review``) before a finding stands.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .transcript import (
    commit_transcript,
    file_id_for,
    page_image_path,
    pages_dir,
)

DRAFT_EXTS = (".md", ".txt")


def _draft_dir(muck_dir: Path, file_id: str) -> Path:
    """Where the agent writes per-page transcripts: .muck/transcripts/<file_id>/."""
    return muck_dir / "transcripts" / file_id


def _drafts(muck_dir: Path, file_id: str) -> dict[int, Path]:
    """Map page_no → draft transcript file the agent wrote (page_NNNN.md/.txt)."""
    d = _draft_dir(muck_dir, file_id)
    found: dict[int, Path] = {}
    if d.is_dir():
        for p in d.iterdir():
            if p.suffix.lower() in DRAFT_EXTS and p.stem.startswith("page_"):
                try:
                    found[int(p.stem.split("_")[1])] = p
                except (IndexError, ValueError):
                    continue
    return found


def _rendered_pages(muck_dir: Path, file_id: str) -> set[int]:
    d = pages_dir(muck_dir, file_id)
    if not d.is_dir():
        return set()
    out = set()
    for p in d.glob("page_*.png"):
        try:
            out.add(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out


def transcribe_status(conn, muck_dir: Path, source_paths: list[str]) -> dict:
    """Per-file vision-transcription progress: rendered, drafted, remaining, done."""
    files = []
    for sp in source_paths:
        file_id = file_id_for(sp)
        row = conn.execute(
            "SELECT n_pages, text_provenance FROM documents WHERE file_id=?", (file_id,)
        ).fetchone()
        n_pages = row["n_pages"] if row else 0
        rendered = _rendered_pages(muck_dir, file_id)
        drafts = _drafts(muck_dir, file_id)
        remaining = sorted(set(range(1, n_pages + 1)) - set(drafts)) if n_pages else []
        files.append({
            "source_path": sp,
            "file_id": file_id,
            "pages": n_pages,
            "rendered": len(rendered),
            "transcribed": sorted(drafts),
            "remaining": remaining,
            "text_provenance": row["text_provenance"] if row else None,
            "next": (
                f"muck render --file {sp}" if not rendered else
                f"read .muck/pages/{file_id}/page_NNNN.png and write "
                f".muck/transcripts/{file_id}/page_NNNN.md" if remaining else
                f"muck transcribe import --file {sp} --model <id>"
            ),
        })
    return {"files": files}


def import_transcripts(
    conn, settings: Settings, muck_dir: Path, source_path: str, model: str,
) -> dict:
    """Assemble the agent's per-page vision transcripts into a sidecar and re-extract."""
    from .quality import classify_page

    file_id = file_id_for(source_path)
    row = conn.execute("SELECT n_pages FROM documents WHERE file_id=?", (file_id,)).fetchone()
    n_pages = row["n_pages"] if row else 0
    drafts = _drafts(muck_dir, file_id)
    if not drafts:
        return {"file": str(source_path), "error":
                f"no page transcripts found in .muck/transcripts/{file_id}/ "
                "(write page_NNNN.md files there first)"}

    pages_meta, missing_images = [], []
    for page_no in range(1, (n_pages or max(drafts)) + 1):
        draft = drafts.get(page_no)
        text = draft.read_text().strip() if draft else ""
        img = page_image_path(muck_dir, file_id, page_no)
        if draft and not img.exists():  # a transcript with no pixels the model could have read
            missing_images.append(page_no)
        pages_meta.append({
            "page_no": page_no, "tier": "vision",
            "page_class": classify_page(text),
            "text": text,
            "image": str(img.relative_to(muck_dir)) if img.exists() else None,
        })
    if missing_images:
        return {"file": str(source_path), "error":
                f"transcribed pages have no rendered image: {missing_images}. "
                f"Run `muck render --file {source_path}` so transcripts are pixel-anchored."}

    pd = commit_transcript(
        conn, muck_dir, file_id, source_path, text_provenance="vision",
        engine=f"{model} (agent vision)", pages_meta=pages_meta,
        params={"model": model},
    )
    return {
        "file": str(source_path), "file_id": file_id,
        "pages": len(pages_meta), "chars": len(pd.text),
        "text_provenance": "vision",
        "next": "muck index  (then `muck review` each finding against its page image)",
    }
