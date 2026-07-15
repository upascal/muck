"""muck command-line interface — the deterministic tools an agent drives.

Machine-facing commands (search/grep/cite/verify/status/trace) emit JSON so a skill can
parse them; ingest/map/parse/index print a short summary. Every command appends to the
audit trace.
"""

from __future__ import annotations

import os

# Keep stdout clean for JSON output: silence model-download progress + tokenizer noise.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json as _json
from pathlib import Path

import typer

from . import cite as citelib
from . import config as cfg
from . import findings as findingslib
from . import observations as obslib
from . import peek as peeklib
from . import pipeline
from . import reader as readerlib
from . import trace as tracelog
from .db import DB_FILENAME, connect, init_schema
from .extract import entities as entitieslib
from .search import aggregate as agg_mod
from .search import anomalies as anomalies_mod
from .search import query as query_mod
from .search import recall as recall_mod

# Import adapter packages for their registration side effects.
from . import mappers as _mappers  # noqa: F401
from . import parsers as _parsers  # noqa: F401
from . import embedders as _embedders  # noqa: F401
from . import rerankers as _rerankers  # noqa: F401
from . import stores as _stores  # noqa: F401

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__,
                  pretty_exceptions_enable=False)

SUPPORTED_EXTS = tuple(pipeline.MAPPER_EXTS) + tuple(pipeline.PARSER_EXTS)


def _version_cb(value: bool):
    if value:
        from . import __version__

        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(version: bool = typer.Option(
        None, "--version", callback=_version_cb, is_eager=True,
        help="Show the muck version and exit")):
    """muck — deterministic, citation-verified investigation tooling for large corpora."""


def _resolve():
    muck_dir = cfg.find_muck_dir()
    if muck_dir is None:
        msg = "No .muck index found here. Run `muck init` first."
        # Common slip: run one directory too high. If an immediate child holds the index,
        # point at it instead of a bare "not found" (find_muck_dir only walks *up*).
        try:
            kids = [p.name for p in Path.cwd().iterdir()
                    if p.is_dir() and (p / cfg.MUCK_DIRNAME).is_dir()]
        except OSError:
            kids = []
        if kids:
            msg = ("No .muck index in this directory — found one in a subdirectory. "
                   f"Run muck from there: {' or '.join('cd ' + k for k in kids[:3])}")
        typer.echo(msg, err=True)
        raise typer.Exit(1)
    settings = cfg.load_settings(muck_dir)
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)  # idempotent; forward-compatible if new tables were added since init
    return muck_dir, settings, conn


def _emit(obj) -> None:
    typer.echo(_json.dumps(obj, indent=2, default=str))


def _expand(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in SUPPORTED_EXTS:
                files.extend(str(x) for x in path.rglob(f"*{ext}"))
        elif path.is_file():
            files.append(str(path))
    return sorted(set(files))


@app.command()
def init(directory: str = typer.Argument(".", help="Corpus root to index")):
    """Create .muck/ (config + DB schema) for a corpus directory."""
    root = Path(directory).resolve()
    muck_dir = root / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True, exist_ok=True)
    if not (muck_dir / cfg.CONFIG_FILENAME).exists():
        cfg.write_default_config(muck_dir)
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    typer.echo(f"Initialized muck index at {muck_dir}")
    typer.echo(f"Edit {muck_dir / cfg.CONFIG_FILENAME} to map JSON fields (see references/ADAPTERS.md).")


@app.command()
def ingest(paths: list[str] = typer.Argument(..., help="Files or directories to register")):
    """Register source files in the manifest (dedup by content hash; no parsing yet)."""
    muck_dir, settings, conn = _resolve()
    files = _expand(paths)
    res = pipeline.ingest(conn, files)
    tracelog.log(conn, muck_dir, "ingest", {"n_paths": len(files)}, summary=_json.dumps(res))
    _emit(res)


@app.command("map")
def map_cmd(only_new: bool = typer.Option(True, "--only-new/--all")):
    """Map JSON/JSONL files into logical documents."""
    muck_dir, settings, conn = _resolve()
    res = pipeline.extract(conn, settings, pipeline.MAPPER_TYPES, only_new)
    tracelog.log(conn, muck_dir, "map", {"only_new": only_new}, summary=_json.dumps(res))
    _emit(res)


@app.command()
def parse(only_new: bool = typer.Option(True, "--only-new/--all")):
    """Parse PDF/DOCX/text files into documents (pages carry provenance)."""
    muck_dir, settings, conn = _resolve()
    res = pipeline.extract(conn, settings, pipeline.PARSER_TYPES, only_new)
    tracelog.log(conn, muck_dir, "parse", {"only_new": only_new}, summary=_json.dumps(res))
    _emit(res)


@app.command()
def index(
    only_new: bool = typer.Option(True, "--only-new/--all"),
    embed: bool = typer.Option(True, "--embed/--no-embed",
                               help="--no-embed = keyword-only, skips the embedding long pole"),
    workers: int = typer.Option(0, "--workers",
                                help="Parallel workers (0=auto, 1=serial/deterministic)"),
):
    """Chunk extracted documents, build the FTS5 index, and embed (if enabled)."""
    muck_dir, settings, conn = _resolve()
    if not embed:
        settings.embedder.enabled = False
    res = pipeline.run_index(conn, settings, only_new, workers=workers)
    tracelog.log(conn, muck_dir, "index", {"only_new": only_new, "embed": embed, "workers": workers},
                 summary=_json.dumps(res))
    _emit(res)


@app.command()
def build(
    paths: list[str] = typer.Argument(..., help="Files or directories to ingest and index"),
    only_new: bool = typer.Option(True, "--only-new/--all", help="Skip already-processed work"),
    cluster: bool = typer.Option(False, "--cluster", help="Also build the topical cluster index"),
    embed: bool = typer.Option(True, "--embed/--no-embed",
                               help="--no-embed = keyword-only fast build (skips the long pole)"),
    workers: int = typer.Option(0, "--workers",
                                help="Parallel workers (0=auto, 1=serial/deterministic)"),
):
    """Run the whole pipeline in one command — ingest → map → parse → index [→ cluster] — with
    wall-clock timing per stage. The reproducible, timed rebuild for a fresh corpus.
    """
    muck_dir, settings, conn = _resolve()
    if not embed:
        settings.embedder.enabled = False
    files = _expand(paths)
    out = pipeline.build_all(conn, settings, files, only_new=only_new, cluster=cluster,
                             workers=workers)
    tracelog.log(conn, muck_dir, "build",
                 {"n_paths": len(files), "only_new": only_new, "cluster": cluster,
                  "embed": embed, "workers": workers},
                 summary=_json.dumps(out))
    _emit(out)


@app.command()
def reset(
    hard: bool = typer.Option(False, "--hard", help="Also clear findings/notes + transcripts (destroys evidence)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
):
    """Wipe the derived index for a clean rebuild — PRESERVING config.toml, trace.jsonl,
    transcripts, and the findings/notes working memory. --hard also clears those authored
    tables and transcripts for a true from-scratch reset.
    """
    muck_dir, settings, conn = _resolve()
    if not yes:
        scope = ("the index + findings/notes + transcripts" if hard
                 else "the derived index (config, findings & notes kept)")
        typer.confirm(f"Delete {scope}?", abort=True)
    res = pipeline.reset(conn, muck_dir, hard=hard)
    tracelog.log(conn, muck_dir, "reset", {"hard": hard}, summary=_json.dumps(res))
    _emit(res)


@app.command()
def status():
    """Show what's been ingested/indexed — re-orient on resume."""
    muck_dir, settings, conn = _resolve()
    _emit(pipeline.status(conn, settings))


@app.command()
def fields():
    """What's queryable in this index — aggregatable fields AND searchable record fields.

    Run this before investigating: it reports ground truth from the built index so you never
    guess whether a field is present. A field can be *searchable* (grep/verify) without being
    *aggregatable* — the whole record is flattened into the document text.
    """
    muck_dir, settings, conn = _resolve()
    _emit(pipeline.describe_fields(conn, settings))


@app.command()
def coverage():
    """Per-file text-coverage report + OCR/vision routing verdict (the extraction worklist).

    For each parsed document: pages, pages carrying usable text, coverage ratio, and a
    verdict (native | sparse | image_only) with a recommended next step. Use this the moment
    `parse`/`status` warns about image-only scans.
    """
    muck_dir, settings, conn = _resolve()
    out = pipeline.coverage(conn, settings)
    tracelog.log(conn, muck_dir, "coverage", {}, n_results=len(out["files"]))
    _emit(out)


def _pdf_targets(conn, file: str | None, all_files: bool, image_only: bool) -> list[str]:
    """Resolve which PDF source paths a render/ocr/transcribe command should act on."""
    if file:
        return [str(Path(file).resolve())]
    q = "SELECT source_path FROM documents WHERE doc_type='pdf'"
    if image_only:  # only the un-transcribed image-only scans
        q += " AND text_provenance='native' AND coverage < 0.1"
    rows = conn.execute(q + " ORDER BY source_path").fetchall()
    return [r["source_path"] for r in rows] if (all_files or image_only) else []


def _parse_pages(pages: str | None) -> list[int] | None:
    """Parse a page spec like '1-20,41' into a sorted list of 1-based page numbers."""
    if not pages:
        return None
    out: set[int] = set()
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


@app.command()
def render(
    file: str = typer.Option(None, "--file", help="A specific PDF to render (default: image-only scans)"),
    all_files: bool = typer.Option(False, "--all", help="Render every parsed PDF"),
    dpi: int = typer.Option(200, "--dpi", help="Render resolution"),
    pages: str = typer.Option(None, "--pages", help="Page spec, e.g. 1-20,41 (default: all)"),
    force: bool = typer.Option(False, "--force", help="Re-render pages already on disk"),
):
    """Rasterize PDF pages to PNGs under .muck/pages/ — the pixel ground truth for review.

    The images an agent's vision (or a human) reads, and what a citation on OCR/vision text
    resolves to. Default target is the image-only scans; pass --file or --all to override.
    """
    from . import render as render_mod

    muck_dir, settings, conn = _resolve()
    targets = _pdf_targets(conn, file, all_files, image_only=not (file or all_files))
    only = _parse_pages(pages)
    out = [render_mod.render_pdf(muck_dir, t, dpi=dpi, only_pages=only, force=force) for t in targets]
    tracelog.log(conn, muck_dir, "render", {"file": file, "all": all_files, "dpi": dpi},
                 n_results=len(out))
    _emit({"files": len(out), "results": out})


@app.command()
def ocr(
    file: str = typer.Option(None, "--file", help="A specific PDF to OCR (default: image-only scans)"),
    all_files: bool = typer.Option(False, "--all", help="OCR every image-only scan"),
    dpi: int = typer.Option(200, "--dpi", help="Render resolution for OCR"),
    lang: str = typer.Option("eng", "--lang", help="tesseract language(s)"),
    force: bool = typer.Option(False, "--force", help="Re-render + re-OCR even if done"),
):
    """Deterministic OCR of scanned PDFs (tesseract) → transcript sidecar → re-extract.

    Renders pages, OCRs each, writes a per-page transcript with confidence, and replaces the
    empty native extraction. Run `muck index` afterwards to make the text searchable. Every
    resulting citation is tier `ocr`: verifiable against the transcript, with the page image
    for human confirmation. Low-confidence pages are flagged for the vision route instead.
    """
    from . import ocr as ocr_mod

    muck_dir, settings, conn = _resolve()
    targets = _pdf_targets(conn, file, all_files, image_only=not file)
    if not targets:
        _emit({"error": "no target PDFs (use --file PATH or --all; default = image-only scans)"})
        raise typer.Exit(1)
    results = []
    for t in targets:
        try:
            results.append(ocr_mod.ocr_file(conn, settings, muck_dir, t, dpi=dpi, lang=lang, force=force))
        except Exception as e:  # noqa: BLE001
            results.append({"file": t, "error": f"{type(e).__name__}: {e}"})
    tracelog.log(conn, muck_dir, "ocr", {"file": file, "all": all_files}, n_results=len(results))
    _emit({"files": len(results), "results": results, "next": "muck index"})


def _filters(doc_type, doc_id, path_like) -> dict:
    f = {}
    if doc_type:
        f["doc_type"] = doc_type
    if doc_id:
        f["doc_id"] = doc_id
    if path_like:
        f["path_like"] = path_like
    return f


@app.command()
def search(
    query: str = typer.Argument("", help="Semantic/keyword query (may be empty with --person/--org)"),
    k: int = typer.Option(8, "-k", "--k"),
    mode: str = typer.Option("auto", help="auto|keyword|semantic|hybrid"),
    doc_type: str = typer.Option(None),
    doc_id: str = typer.Option(None),
    path_like: str = typer.Option(None, help="SQL LIKE filter on source path"),
    person: str = typer.Option(None, "--person", help="Only chunks mentioning this person, ranked by cosine"),
    org: str = typer.Option(None, "--org", help="Only chunks mentioning this organization"),
    entity: str = typer.Option(None, "--entity", help="Only chunks mentioning this entity_id"),
    min_score: float = typer.Option(None, "--min-score", help="Min cosine similarity [0..1]"),
):
    """Search the corpus; each hit carries a verifiable citation token.

    With --person/--org/--entity, restricts to chunks mentioning that resolved entity and
    ranks them by cosine similarity to the query (the entity-filter + similarity query).
    """
    muck_dir, settings, conn = _resolve()
    if person or org or entity:
        kwargs = {"k": k, "min_score": min_score}
        if entity:
            kwargs["entity_id"] = entity
        elif person:
            kwargs.update(name=person, etype="person")
        else:
            kwargs.update(name=org, etype="org")
        hits = query_mod.search_in_entity(conn, settings, query, **kwargs)
        logargs = {"query": query, "person": person, "org": org, "entity": entity, "min_score": min_score}
    else:
        hits = query_mod.search(conn, settings, query, mode, k, _filters(doc_type, doc_id, path_like))
        logargs = {"query": query, "mode": mode, "k": k}
    out = [h.to_dict() for h in hits]
    tracelog.log(conn, muck_dir, "search", logargs, n_results=len(out))
    _emit(out)


@app.command()
def grep(
    pattern: str,
    k: int = typer.Option(20, "-k", "--k", help="Max hits to return (0 = all)"),
    all_matches: bool = typer.Option(False, "--all", help="Return every match (same as -k 0)"),
    doc_type: str = typer.Option(None),
    doc_id: str = typer.Option(None),
):
    """Regex search for exact strings (bill numbers, names, IDs).

    Literal spaces match any whitespace (PDF line breaks included). Output reports
    total_matches/truncated so a capped result set is visible; use --all to enumerate.
    """
    muck_dir, settings, conn = _resolve()
    if all_matches:
        k = 0
    hits, total = query_mod.grep(conn, settings, pattern, k, _filters(doc_type, doc_id, None))
    out = {
        "total_matches": total,
        "returned": len(hits),
        "truncated": total > len(hits),
        "hits": [h.to_dict() for h in hits],
    }
    tracelog.log(conn, muck_dir, "grep", {"pattern": pattern, "k": k, "total_matches": total},
                 n_results=len(hits))
    _emit(out)


transcribe_app = typer.Typer(no_args_is_help=True,
    help="Vision transcription: agent-driven multimodal read of scanned pages (no API key in muck).")
app.add_typer(transcribe_app, name="transcribe")


@transcribe_app.command("status")
def transcribe_status_cmd(
    file: str = typer.Option(None, "--file", help="A specific PDF (default: image-only scans)"),
    all_files: bool = typer.Option(False, "--all", help="All parsed PDFs"),
):
    """Show the vision-transcription work queue: rendered, transcribed, remaining pages.

    Workflow: `muck render` the pages, Read each .muck/pages/<file_id>/page_NNNN.png and
    write .muck/transcripts/<file_id>/page_NNNN.md, then `muck transcribe import`.
    """
    from . import vision

    muck_dir, settings, conn = _resolve()
    targets = _pdf_targets(conn, file, all_files, image_only=not (file or all_files))
    out = vision.transcribe_status(conn, muck_dir, targets)
    tracelog.log(conn, muck_dir, "transcribe.status", {"file": file}, n_results=len(out["files"]))
    _emit(out)


@transcribe_app.command("import")
def transcribe_import_cmd(
    file: str = typer.Option(..., "--file", help="The PDF whose page transcripts to ingest"),
    model: str = typer.Option(..., "--model", help="Model id that produced the transcripts (provenance)"),
):
    """Ingest the agent's per-page vision transcripts into a sidecar (tier=vision).

    Validates every transcribed page is pixel-anchored to a rendered image. The text becomes
    searchable after `muck index`; findings on it require `muck review` (pixel confirmation).
    """
    from . import vision

    muck_dir, settings, conn = _resolve()
    out = vision.import_transcripts(conn, settings, muck_dir, str(Path(file).resolve()), model)
    tracelog.log(conn, muck_dir, "transcribe.import", {"file": file, "model": model},
                 summary=out.get("text_provenance", out.get("error", "")))
    _emit(out)


@app.command()
def doc(doc_id: str):
    """Show one document's manifest (source, locator, type, status) + its entities."""
    muck_dir, settings, conn = _resolve()
    row = conn.execute(
        "SELECT doc_id, source_path, locator, doc_type, title, n_pages, source_name, status "
        "FROM documents WHERE doc_id=?",
        (doc_id,),
    ).fetchone()
    if row is None:
        _emit({"error": f"unknown doc_id {doc_id!r}"})
        raise typer.Exit(1)
    out = dict(row)
    out["n_chunks"] = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()[0]
    out["entities"] = [
        dict(r) for r in conn.execute(
            "SELECT DISTINCT e.entity_id, e.canonical_name, e.entity_type "
            "FROM mentions m JOIN entities e ON e.entity_id=m.entity_id WHERE m.doc_id=?",
            (doc_id,),
        )
    ]
    _emit(out)


@app.command()
def peek(
    path: str = typer.Argument(..., help="A raw source file (or dir) to inspect"),
    sample: int = typer.Option(50, "--sample", help="Records to scan for the field inventory"),
    samples: int = typer.Option(2, "--samples", help="Sample records to include"),
):
    """Inspect a raw file's shape (fields/types/samples) to design the field map (the mapper.fields config). No index needed."""
    typer.echo(_json.dumps(peeklib.peek_file(path, sample, samples), indent=2, default=str))


@app.command()
def sample(
    query: str = typer.Argument(None, help="Optional query — sample full docs behind the top hits"),
    n: int = typer.Option(None, "--n", help="Max number of records"),
    tokens: int = typer.Option(8000, "--tokens", help="Total token budget"),
    doc_type: str = typer.Option(None, "--doc-type"),
    entity: str = typer.Option(None, "--entity", help="Sample docs mentioning this entity_id/name"),
    order: str = typer.Option("random", "--order", help="random|first (when no query/entity)"),
    raw: bool = typer.Option(False, "--raw/--no-raw", help="Include the raw source record"),
):
    """Load a budget-bounded sample of full records — to get a feel for the corpus / calibrate queries."""
    muck_dir, settings, conn = _resolve()
    out = readerlib.sample_documents(conn, settings, query, doc_type, entity, n, tokens, order, raw)
    tracelog.log(conn, muck_dir, "sample", {"query": query, "n": n, "tokens": tokens,
                 "doc_type": doc_type, "entity": entity}, n_results=out["returned"])
    _emit(out)


@app.command()
def read(
    doc_ids: list[str] = typer.Argument(..., help="One or more doc_ids to read in full, side by side"),
    tokens: int = typer.Option(16000, "--tokens", help="Total token budget across the batch"),
    raw: bool = typer.Option(True, "--raw/--no-raw", help="Include the raw source record"),
):
    """Read full documents (batch) side by side — for comparison/calibration; token-budgeted.

    Use after search/entities/aggregate have narrowed to relevant doc_ids. Each record carries a
    whole-document citation token. Records beyond the token budget are reported, not dropped silently.
    """
    muck_dir, settings, conn = _resolve()
    out = readerlib.read_documents(conn, settings, doc_ids, tokens, raw)
    tracelog.log(conn, muck_dir, "read", {"n_doc_ids": len(doc_ids), "tokens": tokens},
                 n_results=out["returned"])
    _emit(out)


@app.command()
def cite(token: str):
    """Resolve a citation token to its exact source span + surrounding context."""
    muck_dir, settings, conn = _resolve()
    r = citelib.resolve(conn, settings, token)
    tracelog.log(conn, muck_dir, "cite", {"token": token}, summary="valid" if r.get("valid") else "invalid")
    _emit(r)


@app.command()
def verify(
    token: str = typer.Option(..., "--token"),
    quote: str = typer.Option(..., "--quote"),
):
    """Hallucination guard: confirm a quote actually occurs at the cited source span."""
    muck_dir, settings, conn = _resolve()
    r = citelib.verify(conn, settings, token, quote)
    tracelog.log(conn, muck_dir, "verify", {"token": token}, summary="verified" if r.get("verified") else "FAILED")
    _emit(r)


@app.command()
def entities(
    entity_id: str = typer.Option(None, "--entity", help="Show full detail + cross-references for one entity"),
    type: str = typer.Option(None, "--type", help="org|person|bill"),
    name: str = typer.Option(None, "--name", help="Substring filter on canonical name"),
    min_count: int = typer.Option(1, "--min-count"),
    k: int = typer.Option(50, "-k", "--k"),
):
    """List resolved entities, or cross-reference one entity's mentions and co-occurrences."""
    muck_dir, settings, conn = _resolve()
    if entity_id:
        out = entitieslib.entity_detail(conn, entity_id)
    else:
        out = entitieslib.list_entities(conn, type, name, min_count, k)
    tracelog.log(conn, muck_dir, "entities", {"entity_id": entity_id, "type": type, "name": name},
                 n_results=1 if entity_id else len(out))
    _emit(out)


@app.command()
def entity(
    name: str = typer.Argument(..., help="Person/org name to drill into"),
    examples: int = typer.Option(4, "--examples", help="Number of example citations to include"),
):
    """Cross-source drill-down for ONE entity: where it's named (grouped by source), its resolved
    identity + network, and example citations. A pre-joined dossier for honing in on a person/org —
    a name in both a press source and a contributions source is a say-vs-pay lead to verify.
    """
    muck_dir, settings, conn = _resolve()
    out = entitieslib.dossier(conn, settings, name, examples=examples)
    tracelog.log(conn, muck_dir, "entity", {"name": name}, n_results=out.get("documents_naming_it", 0))
    _emit(out)


finding_app = typer.Typer(no_args_is_help=True, help="Manage the findings ledger.")
app.add_typer(finding_app, name="finding")


@finding_app.command("add")
def finding_add(
    claim: str = typer.Option(..., "--claim"),
    quote: str = typer.Option(..., "--quote"),
    token: str = typer.Option(..., "--token"),
):
    """Record a finding; it is verified against source on entry."""
    muck_dir, settings, conn = _resolve()
    r = findingslib.add_finding(conn, settings, claim, quote, token)
    tracelog.log(conn, muck_dir, "finding.add", {"claim": claim, "token": token}, summary=r["status"])
    _emit(r)


@finding_app.command("list")
def finding_list():
    """List recorded findings and their last audit status."""
    muck_dir, settings, conn = _resolve()
    _emit(findingslib.list_findings(conn))


@app.command()
def aggregate(
    by: str = typer.Option(None, "--by", help="Structured field to GROUP BY (e.g. registrant)"),
    measure: str = typer.Option(None, "--measure", help="Numeric field for sum/avg/min/max"),
    agg: str = typer.Option("count", "--agg", help="count|sum|avg|min|max"),
    resolve: str = typer.Option(None, "--resolve", help="Fold values onto resolved entity: org|person"),
    sql: str = typer.Option(None, "--sql", help="Read-only SELECT/WITH passthrough (DuckDB: muck.* + external read_*)"),
    engine: str = typer.Option("auto", "--engine", help="auto|sqlite|duckdb (auto = DuckDB if installed)"),
    k: int = typer.Option(20, "-k", "--k", help="Row limit"),
):
    """Exact tabular analytics over JSON-record fields (top spenders, counts per period…).

    Runs on DuckDB (over the read-only SQLite source of truth) when available, else SQLite
    json1. With --sql on DuckDB, query the corpus as muck.* and join external files via
    read_json_auto/read_csv_auto/read_parquet.
    """
    muck_dir, settings, conn = _resolve()
    db_path = str(muck_dir / DB_FILENAME)
    eng = agg_mod.resolve_engine(engine)  # surfaces a clear error if duckdb is forced but absent
    if sql:
        out = agg_mod.run_sql(conn, sql, k, engine, db_path)
    elif by:
        # settings: so --resolve folds with the same suffix set the entity index used
        out = agg_mod.aggregate(conn, by, measure, agg, k, resolve, engine, db_path, settings)
    else:
        typer.echo("Provide --by FIELD (with optional --measure/--agg) or --sql.", err=True)
        raise typer.Exit(1)
    tracelog.log(conn, muck_dir, "aggregate",
                 {"by": by, "measure": measure, "agg": agg, "sql": sql, "engine": eng}, n_results=len(out))
    _emit(out)


@app.command()
def anomalies(
    by: str = typer.Option(None, "--by", help="Peer-group field (default registrant.name; client.name in --mode rare)"),
    measure: str = typer.Option("income", "--measure", help="Numeric field tested for outliers"),
    mode: str = typer.Option("peer", "--mode", help="peer (per-record vs its peer baseline) | rare (one-shot whales)"),
    method: str = typer.Option("robust_z", "--method", help="robust_z|iqr (peer mode)"),
    space: str = typer.Option("log", "--space", help="log|linear (peer mode; log suits money — '10x the median')"),
    threshold: float = typer.Option(None, "--threshold", help="flag cutoff (default 3.5 robust_z / 1.5 iqr)"),
    min_peers: int = typer.Option(5, "--min-peers", help="Skip peer groups smaller than this"),
    resolve: str = typer.Option("auto", "--resolve", help="Fold peer key onto a resolved entity: auto|org|person|none"),
    segment: str = typer.Option("filing_type", "--segment", help="Sub-group peers by this field (or 'none' to pool)"),
    per_group_cap: int = typer.Option(3, "--per-group-cap", help="Max flagged records per peer group"),
    sort: str = typer.Option("ratio", "--sort", help="peer rank: ratio (fold-change, the 'Nx' story) | z (raw extremity)"),
    min_ratio: float = typer.Option(1.0, "--min-ratio", help="peer mode: min fold-change from peer median to flag (e.g. 2 = >=2x or <=0.5x)"),
    max_count: int = typer.Option(1, "--max-count", help="rare mode: group-size ceiling (singletons=1)"),
    rarity_pct: float = typer.Option(95.0, "--rarity-pct", help="rare mode: sum-percentile floor"),
    k: int = typer.Option(50, "-k", "--k", help="Max flagged records to return"),
):
    """Surface records that deviate from their peers — leads no one thought to query.

    peer mode flags records whose --measure is far from its --by peer group's baseline
    (log-space robust modified z-score by default; segmented by filing_type; entity-folded).
    rare mode flags groups appearing <= --max-count times yet carrying a top-percentile sum.
    Each flagged record carries a whole-document citation token (verify with `muck cite`/
    `muck verify`) plus the reproducible query + peer stats.
    """
    muck_dir, settings, conn = _resolve()
    try:
        out = anomalies_mod.anomalies(
            conn, by, measure, method=method, space=space, threshold=threshold,
            min_peers=min_peers, mode=mode, resolve=resolve, segment=segment,
            per_group_cap=per_group_cap, min_ratio=min_ratio, sort=sort,
            max_count=max_count, rarity_pct=rarity_pct, k=k, settings=settings)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    tracelog.log(conn, muck_dir, "anomalies",
                 {"by": by, "measure": measure, "mode": mode, "method": method,
                  "threshold": threshold, "min_peers": min_peers, "resolve": resolve,
                  "segment": segment}, n_results=len(out))
    _emit(out)


@app.command()
def cluster(k: str = typer.Option("auto", "--k", help="Number of clusters, or 'auto'")):
    """Build a topical index: KMeans over embeddings + c-TF-IDF labels."""
    muck_dir, settings, conn = _resolve()
    from .cluster import kmeans as cluster_mod

    res = cluster_mod.build_clusters(conn, settings, k)
    tracelog.log(conn, muck_dir, "cluster", {"k": k}, summary=f"{res.get('clusters', 0)} clusters",
                 n_results=res.get("clusters", 0))
    _emit(res)


note_app = typer.Typer(no_args_is_help=True, help="Investigation working memory: notes, decisions, leads, open threads.")
app.add_typer(note_app, name="note")


@note_app.command("add")
def note_add(
    text: str = typer.Option(..., "--text"),
    kind: str = typer.Option("note", "--kind", help="note|decision|lead|hypothesis|question|dead_end"),
    status: str = typer.Option("open", "--status", help="open|confirmed|cold|dropped"),
    entity: list[str] = typer.Option(None, "--entity", help="Related entity_id (repeatable)"),
    doc: list[str] = typer.Option(None, "--doc", help="Related doc_id (repeatable)"),
    token: list[str] = typer.Option(None, "--token", help="Supporting citation token (repeatable)"),
):
    """Record an observation (note / decision / lead / hypothesis / question / dead_end)."""
    muck_dir, settings, conn = _resolve()
    r = obslib.add(conn, text, kind, status, entity or None, doc or None, token or None)
    tracelog.log(conn, muck_dir, "note.add", {"kind": kind, "status": status}, summary=str(r.get("obs_id")))
    _emit(r)


@note_app.command("list")
def note_list(
    status: str = typer.Option(None, "--status"),
    kind: str = typer.Option(None, "--kind"),
    entity: str = typer.Option(None, "--entity"),
    k: int = typer.Option(50, "-k", "--k"),
):
    """List observations (newest first), filtered by status/kind/entity."""
    muck_dir, settings, conn = _resolve()
    _emit(obslib.list(conn, status, kind, entity, k))


@note_app.command("update")
def note_update(
    obs_id: int = typer.Argument(...),
    status: str = typer.Option(None, "--status"),
    text: str = typer.Option(None, "--text"),
):
    """Update an observation's status (e.g. mark a thread cold) or text."""
    muck_dir, settings, conn = _resolve()
    r = obslib.update(conn, obs_id, status, text)
    tracelog.log(conn, muck_dir, "note.update", {"obs_id": obs_id, "status": status})
    _emit(r)


@app.command()
def recall(
    query: str,
    kind: str = typer.Option(None, "--kind", help="Restrict to a memory kind (or 'finding')"),
    status: str = typer.Option(None, "--status"),
    entity: str = typer.Option(None, "--entity", help="entity_id or name"),
    mode: str = typer.Option("auto", "--mode", help="auto|keyword|semantic"),
    k: int = typer.Option(8, "-k", "--k"),
):
    """Search your own accumulated memory (observations + findings). (`search` = the corpus.)"""
    muck_dir, settings, conn = _resolve()
    out = recall_mod.recall(conn, settings, query, kind, status, entity, mode, k)
    tracelog.log(conn, muck_dir, "recall", {"query": query, "mode": mode}, n_results=len(out))
    _emit(out)


@app.command()
def audit():
    """Re-verify every finding's quote against source (green/red for an editor).

    Native/OCR findings verify against text; vision findings (LLM-transcribed pixels) are
    bucketed as pending_review until confirmed with `muck review`.
    """
    muck_dir, settings, conn = _resolve()
    r = findingslib.audit(conn, settings)
    tracelog.log(conn, muck_dir, "audit", {}, summary=f"{r['verified']}/{r['total']} verified",
                 n_results=r["total"])
    _emit(r)


@app.command()
def review(
    finding_id: str = typer.Argument(..., help="The finding to review against its page image"),
    confirm: bool = typer.Option(False, "--confirm", help="The page image supports the claim"),
    reject: bool = typer.Option(False, "--reject", help="The page image does NOT support the claim"),
    note: str = typer.Option(None, "--note", help="Reviewer note"),
):
    """Record a pixel review: confirm/reject a vision-tier finding against its page image.

    This is what lets a finding resting on LLM-transcribed pixels move past pending_review —
    a human (or agent) has looked at the actual scan. Reviewer id comes from $MUCK_REVIEWER.
    """
    if confirm == reject:
        typer.echo("Pass exactly one of --confirm / --reject.", err=True)
        raise typer.Exit(1)
    muck_dir, settings, conn = _resolve()
    reviewer = os.environ.get("MUCK_REVIEWER", "agent")
    r = findingslib.review(conn, settings, finding_id, "confirmed" if confirm else "rejected", reviewer, note)
    tracelog.log(conn, muck_dir, "review", {"finding_id": finding_id}, summary=r.get("status", ""))
    _emit(r)


@app.command()
def reconcile(
    on: str = typer.Option(None, "--on", help="Primary name field to reconcile on (a structured field)"),
    compare: str = typer.Option(None, "--compare", help="Comma-separated corroborating fields (address, zip, …)"),
    type: str = typer.Option("org", "--type", help="Entity type label"),
    threshold: float = typer.Option(0.9, "--threshold", help="Match-probability cutoff (raise for precision)"),
    link: str = typer.Option(None, "--link", help="External JSON dataset to link the corpus against"),
    train: bool = typer.Option(False, "--train", help="Refine u-probabilities by sampling (large corpora)"),
    show: bool = typer.Option(False, "--show", help="Show a reconcile run instead of running one"),
    run: str = typer.Option(None, "--run", help="Run id for --show (default: most recent)"),
):
    """Probabilistic multi-field record reconciliation via Splink (opt-in: --extra splink).

    Dedupes messy records (same firm under drifting names) or links the corpus to an
    external dataset (--link). Name-dominant + fully audited (scores, clusters, threshold).
    """
    muck_dir, settings, conn = _resolve()
    from . import reconcile as rec

    if show:
        out = rec.show_run(conn, run)
    else:
        if not on:
            typer.echo("Provide --on FIELD (or use --show).", err=True)
            raise typer.Exit(1)
        fields = [c.strip() for c in compare.split(",")] if compare else []
        out = rec.reconcile(conn, settings, on=on, compare=fields, etype=type,
                            threshold=threshold, link_path=link, train=train)
    tracelog.log(conn, muck_dir, "reconcile", {"on": on, "threshold": threshold, "link": link},
                 summary=out.get("run_id", ""), n_results=out.get("n_merged_clusters", 0))
    _emit(out)


@app.command("trace")
def trace_cmd(tail: int = typer.Option(20, "--tail")):
    """Print the audit trace (every tool call)."""
    muck_dir, settings, conn = _resolve()
    _emit(tracelog.tail(conn, tail))


def main() -> None:
    """Console entry point. Turns any uncaught library error into a JSON ``{"error": ...}`` on
    stdout (+ exit 1) so an agent parsing output never gets a raw Python traceback. Typer's own
    ``typer.Exit``/``Abort`` control flow (the friendly stderr messages, confirm-abort) raises
    ``SystemExit`` and passes straight through.
    """
    from .interfaces import NotInstalled

    try:
        app()
    except (ValueError, NotInstalled, KeyError, FileNotFoundError) as e:
        typer.echo(_json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
