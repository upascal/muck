"""Pipeline orchestration: ingest -> map/parse (extract) -> index.

Stages are separate and idempotent (dedup by content hash; skip already-done work) so
a large corpus run resumes after interruption — what keeps an investigation organized
across sessions and efficient with the corpus.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .extract.chunk import chunk_document
from .interfaces.mapper import get_mapper
from .interfaces.parser import get_parser
from .schema import Page, Record

MAPPER_EXTS = {".json": "json", ".jsonl": "jsonl", ".xml": "xml"}
PARSER_EXTS = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "md", ".markdown": "md", ".text": "txt"}
MAPPER_TYPES = {"json", "jsonl", "xml"}
PARSER_TYPES = {"pdf", "docx", "md", "txt"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def file_id_from_hash(content_hash: str) -> str:
    """Content-addressed file id. Deriving the id from file *content* (not its path) makes
    doc_ids and citation tokens reproducible across machines/checkouts — an evaluator can
    re-run the pipeline from any directory and the tokens in the findings still resolve."""
    return "f" + content_hash[:10]


def content_file_id(path: str | Path) -> str:
    """The file_id muck assigns to a source file, from its content. Single source of truth
    shared with transcript.file_id_for so page-image/sidecar paths never diverge."""
    return file_id_from_hash(_hash_file(Path(path)))


def _file_type(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in MAPPER_EXTS:
        return MAPPER_EXTS[ext]
    if ext in PARSER_EXTS:
        return PARSER_EXTS[ext]
    return None


def _field_map_hash(settings: Settings) -> str:
    # Hash the whole mapper config (global fields + per-source maps) so editing any
    # [[mapper.sources]] entry triggers the drift warning too.
    return hashlib.sha1(
        json.dumps(settings.mapper.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _config_hash(settings: Settings) -> str:
    # Short hash of the full effective config — recorded in the index so a rebuild's
    # provenance (which settings produced it) is auditable/reproducible.
    return hashlib.sha1(
        json.dumps(settings.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def field_map_drift(conn, settings: Settings) -> str | None:
    """Return a warning if the field map changed since the last `map` (else None).

    The field map is consumed at *map* time and baked into ``documents.structured_json``;
    `index` reads that, not the live config. So a changed field map needs a re-map.
    """
    from .db import meta_get

    stored = meta_get(conn, "field_map_hash")
    if stored is None or stored == _field_map_hash(settings):
        return None
    return "field map changed since last `muck map` — run `muck map --all` then `muck index --all` to apply it"


def _parser_name(file_type: str, settings: Settings) -> str:
    if file_type == "pdf":
        return settings.parser.pdf
    if file_type == "docx":
        return settings.parser.docx
    return settings.parser.text


# --- ingest ------------------------------------------------------------------

def ingest(conn: sqlite3.Connection, paths: list[str]) -> dict:
    added = updated = skipped = unsupported = 0
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        ftype = _file_type(path)
        if ftype is None:
            unsupported += 1
            continue
        chash = _hash_file(path)
        fid = file_id_from_hash(chash)
        existing = conn.execute(
            "SELECT content_hash FROM source_files WHERE file_id=?", (fid,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO source_files(file_id, path, content_hash, file_type, status, ingested_at) "
                "VALUES(?,?,?,?, 'ingested', ?)",
                (fid, str(path.resolve()), chash, ftype, _now()),
            )
            added += 1
        elif existing["content_hash"] != chash:
            _clear_file(conn, fid)
            conn.execute(
                "UPDATE source_files SET content_hash=?, status='ingested', error=NULL, ingested_at=? WHERE file_id=?",
                (chash, _now(), fid),
            )
            updated += 1
        else:
            skipped += 1
    conn.commit()
    return {"added": added, "updated": updated, "skipped": skipped, "unsupported": unsupported}


def _clear_file(conn: sqlite3.Connection, file_id: str) -> None:
    # Set-based deletes keyed on the file's chunks. `chunk_id` is UNINDEXED in the FTS/vec
    # virtual tables, so a per-chunk `WHERE chunk_id=?` full-scans the whole table each time
    # (quadratic when re-mapping a big file). One IN-subquery scans once instead.
    subq = "SELECT chunk_id FROM chunks WHERE doc_id IN (SELECT doc_id FROM documents WHERE file_id=?)"
    conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({subq})", (file_id,))
    try:
        conn.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({subq})", (file_id,))
    except sqlite3.OperationalError:
        pass  # vec table not created (embeddings off)
    conn.execute(
        "DELETE FROM chunks WHERE doc_id IN (SELECT doc_id FROM documents WHERE file_id=?)",
        (file_id,),
    )
    conn.execute("DELETE FROM documents WHERE file_id=?", (file_id,))


def build_all(conn, settings: Settings, files: list[str], *, only_new: bool = True,
              cluster: bool = False, workers: int | None = None) -> dict:
    """Run the whole pipeline (ingest → map → parse → index [→ cluster]) with per-stage
    wall-clock timing. Powers ``muck build`` — the reproducible, timed corpus rebuild.
    """
    import time

    stages: dict = {}

    def _timed(name, fn):
        t = time.monotonic()
        res = fn()
        stages[name] = {"duration_s": round(time.monotonic() - t, 3), **res}
        return res

    t0 = time.monotonic()
    _timed("ingest", lambda: ingest(conn, files))
    _timed("map", lambda: extract(conn, settings, MAPPER_TYPES, only_new))
    _timed("parse", lambda: extract(conn, settings, PARSER_TYPES, only_new))
    idx = _timed("index", lambda: run_index(conn, settings, only_new, workers=workers))
    if cluster:
        from .cluster import kmeans as cluster_mod

        _timed("cluster", lambda: cluster_mod.build_clusters(conn, settings, "auto"))
    return {"stages": stages, "total_s": round(time.monotonic() - t0, 3),
            "documents": idx.get("documents"), "chunks": idx.get("chunks")}


def reset(conn: sqlite3.Connection, muck_dir, *, hard: bool = False) -> dict:
    """Wipe the *derived* index for a clean rebuild, preserving the authored working memory.

    Soft (default) clears the rebuildable tables + ``pages/``/``cache/`` but keeps
    ``config.toml``, ``trace.jsonl``, ``transcripts/``, and the authored
    ``findings``/``observations``/``pixel_reviews`` tables. ``hard`` additionally clears those
    authored tables and removes ``transcripts/`` (destroys evidence) — the audit ``trace_log``
    and ``config.toml`` always survive. This is the safe "wipe indexes, keep config" the
    freeze/rebuild flow needs (config lives inside ``.muck/`` next to ``index.db``).
    """
    derived = [
        "chunks_fts", "chunks_vec", "chunks", "mentions", "entity_edges", "entity_aliases",
        "entities", "chunk_clusters", "clusters", "reconcile_links", "reconcile_records",
        "reconcile_runs", "documents", "source_files",
    ]
    authored = ["findings", "pixel_reviews", "observations"]  # trace_log kept for audit continuity
    cleared: list[str] = []
    for table in derived + (authored if hard else []):
        try:
            conn.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed internal identifiers
            cleared.append(table)
        except sqlite3.OperationalError:
            pass  # e.g. chunks_vec never created (embeddings were off)
    conn.commit()

    removed: list[str] = []
    muck_dir = Path(muck_dir)
    for sub in ["pages", "cache"] + (["transcripts"] if hard else []):
        d = muck_dir / sub
        if d.exists():
            shutil.rmtree(d)
            removed.append(sub + "/")

    preserved = ["config.toml", "trace.jsonl", "trace_log"]
    if not hard:
        preserved += ["transcripts/", "findings", "observations", "pixel_reviews"]
    return {"mode": "hard" if hard else "soft", "tables_cleared": cleared,
            "dirs_removed": removed, "preserved": preserved}


# --- extract (map / parse) ---------------------------------------------------

def _insert_document(conn, file_id, idx, record: Record, source_name: str) -> None:
    from .quality import page_coverage

    doc_id = f"{file_id}.{idx}"
    pages_json = (
        json.dumps([[p.number, p.char_start, p.char_end, p.page_class, p.tier] for p in record.pages])
        if record.pages else None
    )
    cov = page_coverage(record.pages, record.text)["coverage"]
    conn.execute(
        "INSERT OR REPLACE INTO documents(doc_id, file_id, source_path, locator, doc_type, "
        "title, text, n_pages, pages_json, structured_json, raw_json, source_name, "
        "text_provenance, coverage, status, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'extracted', ?)",
        (
            doc_id, file_id, record.source_path, record.locator, record.doc_type,
            record.title, record.text, len(record.pages) if record.pages else 0, pages_json,
            json.dumps(record.structured, default=str) if record.structured else None,
            json.dumps(record.raw, default=str) if record.raw is not None else None,
            source_name, record.text_provenance, cov, _now(),
        ),
    )


def extract(conn, settings: Settings, kinds: set[str], only_new: bool = True) -> dict:
    placeholders = ",".join("?" * len(kinds))
    q = f"SELECT * FROM source_files WHERE file_type IN ({placeholders})"
    if only_new:
        q += " AND status != 'processed'"
    rows = conn.execute(q, list(kinds)).fetchall()
    n_files = n_docs = 0
    errors = 0
    warnings: list[str] = []
    for row in rows:
        fid, path, ftype = row["file_id"], row["path"], row["file_type"]
        try:
            if row["status"] == "processed":
                _clear_file(conn, fid)
            if ftype in MAPPER_TYPES:
                from .config import fields_for

                mapper = get_mapper("xml" if ftype == "xml" else settings.mapper.backend)
                records = mapper.map(path, fields_for(settings, path))
                source_name = mapper.name
            else:
                from .transcript import parser_name_for

                # A transcript sidecar (from `muck ocr`/`transcribe import`) wins over the
                # config parser, so `parse --all` re-derives OCR/vision text instead of
                # overwriting it with the scan's empty text layer.
                pname = parser_name_for(ftype, path, _parser_name(ftype, settings))
                parser = get_parser(pname)
                pd = parser.parse(path)
                if not getattr(parser, "page_precise", True):
                    warnings.append(
                        f"{Path(path).name}: parser {parser.name!r} is not page-precise — "
                        "citations resolve to the whole document, not a page. Use `muck ocr` "
                        "for page-precise scanned-PDF provenance."
                    )
                records = [Record(
                    source_path=path, doc_type=ftype, text=pd.text,
                    title=pd.title, pages=pd.pages, text_provenance=pd.text_provenance,
                )]
                source_name = parser.name
            for i, rec in enumerate(records):
                _insert_document(conn, fid, i, rec, source_name)
            conn.execute(
                "UPDATE source_files SET n_docs=?, status='processed', error=NULL, processed_at=? WHERE file_id=?",
                (len(records), _now(), fid),
            )
            n_files += 1
            n_docs += len(records)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            conn.execute(
                "UPDATE source_files SET status='error', error=? WHERE file_id=?",
                (f"{type(e).__name__}: {e}", fid),
            )
            conn.commit()
            errors += 1
    if kinds & MAPPER_TYPES:  # record the field map used, to detect later config drift
        from .db import meta_set

        meta_set(conn, "field_map_hash", _field_map_hash(settings))
        conn.commit()
    result = {"files": n_files, "documents": n_docs, "errors": errors}
    if warnings:
        result["warnings"] = warnings
    if kinds & PARSER_TYPES:
        notice = _text_coverage_notice(conn)
        if notice:
            result["notice"] = notice
    return result


# --- index -------------------------------------------------------------------

# Flush size for the index loop: how many chunks accumulate before we embed + commit. Chosen
# above model2vec's multiprocessing_threshold (10k) so the default Potion embedder actually
# parallelizes across cores, and large enough that commits (fsync) happen ~once per flush
# instead of once per document. Memory stays bounded (~this many texts + their vectors).
EMBED_FLUSH = 25_000


def resolve_workers(workers: int | None = None) -> int:
    """Effective worker count for the parallel stages. Explicit ``workers > 0`` wins; else the
    ``MUCK_WORKERS`` env var; else auto = ``min(cpu_count, 16)``. ``1`` = serial/deterministic —
    the "unknown judge machine" escape hatch (never requires a GPU or a specific core count)."""
    import os

    if workers and workers > 0:
        return workers
    env = os.environ.get("MUCK_WORKERS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(os.cpu_count() or 1, 16))


def _load_embedder(settings: Settings):
    try:
        from .embedders import build_embedder

        emb = build_embedder(settings)
        if emb is None:
            return None
        _ = emb.dim  # force load now so failures surface before indexing
        return emb
    except Exception:
        return None


def run_index(conn, settings: Settings, only_new: bool = True, batch: int = EMBED_FLUSH,
              workers: int | None = None) -> dict:
    from .interfaces.store import get_store

    store = get_store(settings.store.backend)
    embedder = _load_embedder(settings)
    embeddings_active = embedder is not None and store.supports_vectors(conn)
    if embeddings_active and hasattr(embedder, "set_workers"):
        embedder.set_workers(resolve_workers(workers))

    if not only_new:  # clean re-index: drop existing chunk artifacts first
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM chunks_fts")
        try:
            conn.execute("DELETE FROM chunks_vec")
        except sqlite3.OperationalError:
            pass
        conn.execute("UPDATE documents SET status='extracted'")
        conn.commit()

    docs = conn.execute(
        "SELECT doc_id, title, text, locator, pages_json FROM documents WHERE status='extracted'"
    ).fetchall()

    contextual = settings.embedder.contextual
    n_docs = n_chunks = 0
    pending_ids: list[str] = []
    pending_texts: list[str] = []
    n_since_commit = 0

    def flush():
        # Embed + store any pending vectors, then commit the chunks, status updates AND vectors
        # accumulated since the last flush as one transaction. Committing per-flush (not per-doc)
        # removes an fsync per document; every committed doc still has its vectors, so a crash
        # resumes cleanly (uncommitted docs stay 'extracted' and get reprocessed idempotently).
        nonlocal pending_ids, pending_texts, n_since_commit
        if embeddings_active and pending_ids:
            vecs = embedder.embed(pending_texts)
            store.upsert_vectors(conn, pending_ids, vecs)
        pending_ids, pending_texts = [], []
        n_since_commit = 0
        conn.commit()

    for d in docs:
        pages = None
        if d["pages_json"]:
            pages = [Page(*p) for p in json.loads(d["pages_json"])]
        chunks = chunk_document(d["doc_id"], d["text"], settings.chunk, pages, d["locator"])
        if chunks:
            store.index_chunks(conn, chunks)
            if embeddings_active:
                prefix = f"{d['title']}. " if (contextual and d["title"]) else ""
                for c in chunks:
                    pending_ids.append(c.chunk_id)
                    # Contextual retrieval: embed with doc context; stored chunk + offsets stay raw.
                    pending_texts.append(prefix + c.text if prefix else c.text)
        conn.execute("UPDATE documents SET status='indexed', indexed_at=? WHERE doc_id=?", (_now(), d["doc_id"]))
        n_docs += 1
        n_chunks += len(chunks)
        n_since_commit += len(chunks)
        if n_since_commit >= batch:
            flush()
    flush()
    result = {
        "documents": n_docs,
        "chunks": n_chunks,
        "embeddings": "on" if embeddings_active else "off (keyword-only)",
    }
    if settings.entities.resolve:
        from .extract.entities import build_entities, build_relations

        result["entity_index"] = build_entities(conn, settings)
        result["relations"] = build_relations(conn, settings)  # typed edges over resolved entities
        notice = _entity_inertia_notice(conn)
        if notice:
            result["notice"] = notice
    coverage_notice = _text_coverage_notice(conn)
    if coverage_notice:
        result["coverage_notice"] = coverage_notice
    drift = field_map_drift(conn, settings)
    if drift:
        result["config_drift"] = drift

    # Record index provenance so a rebuild is auditable (which version/model/config made it).
    from . import __version__
    from .db import meta_set

    meta_set(conn, "muck_version", __version__)
    meta_set(conn, "embed_model", settings.embedder.model)
    meta_set(conn, "embed_revision", settings.embedder.revision or "")
    meta_set(conn, "config_hash", _config_hash(settings))
    conn.commit()
    return result


def _text_coverage_notice(conn) -> str | None:
    """Warn when parsed documents have little extractable text (image-only scans).

    Mirrors ``_entity_inertia_notice``: fires from parse/index/status so the agent never
    reads an empty index as "no evidence". This is the extraction-layer counterpart to the
    retrieval-layer completeness fixes — a degraded stage must not look healthy.
    """
    ph = ",".join("?" * len(PARSER_TYPES))
    row = conn.execute(
        f"SELECT COUNT(*) n, SUM(CASE WHEN coverage < 0.1 THEN 1 ELSE 0 END) image_only "
        f"FROM documents WHERE doc_type IN ({ph}) AND text_provenance='native'",
        sorted(PARSER_TYPES),
    ).fetchone()
    total, image_only = row["n"], (row["image_only"] or 0)
    if total and image_only:
        return (
            f"{image_only} of {total} parsed document(s) have almost no extractable text "
            "(image-only scans): they index to ~nothing, so searches return nothing and that "
            "is NOT evidence of absence. Run `muck coverage` to see which, then `muck render` "
            "+ `muck ocr` (deterministic) or the vision route (`muck transcribe`)."
        )
    return None


def coverage(conn, settings: Settings) -> dict:
    """Per-file text-coverage report + routing verdict — the OCR/vision worklist.

    Makes the parse/OCR/vision routing decision (which Hagar's skill makes by hand)
    programmatic: for each parsed file, how many pages carry usable text, and what to do.
    """
    from .quality import coverage_verdict

    ph = ",".join("?" * len(PARSER_TYPES))
    rows = conn.execute(
        f"SELECT doc_id, file_id, source_path, doc_type, n_pages, text, pages_json, "
        f"text_provenance, coverage FROM documents WHERE doc_type IN ({ph}) ORDER BY source_path",
        sorted(PARSER_TYPES),
    ).fetchall()
    files, tally = [], {"native": 0, "sparse": 0, "image_only": 0}
    for r in rows:
        cov = r["coverage"] if r["coverage"] is not None else 1.0
        pages = json.loads(r["pages_json"]) if r["pages_json"] else []
        with_text = sum(1 for p in pages if (p[2] - p[1]) >= 20)
        verdict = coverage_verdict(cov) if r["text_provenance"] == "native" else "native"
        # A doc already OCR'd/vision-transcribed is no longer an image_only problem.
        bucket = verdict if r["text_provenance"] == "native" else "native"
        tally[bucket] = tally.get(bucket, 0) + 1
        recommend = (
            "ocr" if verdict == "image_only" else "review" if verdict == "sparse" else "none"
        )
        files.append({
            "doc_id": r["doc_id"], "source_path": r["source_path"],
            "text_provenance": r["text_provenance"],
            "pages": r["n_pages"], "pages_with_text": with_text,
            "coverage": round(cov, 3), "verdict": verdict, "recommend": recommend,
        })
    return {"corpus": tally, "files": files}


def describe_fields(conn, settings, sample: int = 300) -> dict:
    """Ground truth for 'what can I query, and how' — so an agent checks the index instead of
    inferring from the config. Reports **aggregatable** fields (structured_json keys, for
    ``muck aggregate --by``) AND every top-level **record field**, flagged by whether it is
    searchable (rendered into the document text → findable by ``muck grep``/``search`` and
    citable) and whether it is aggregatable. A field can be searchable without being a structured
    field — the whole record is flattened into text unless ``text_fields`` narrows it.

    Samples ``sample`` documents spread evenly across the corpus (fast indexed point-lookups, so
    all source types are represented even in a huge index).
    """
    from .config import fields_for

    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if total == 0:
        return {"documents": 0, "note": "empty index — run `muck build`/`muck index` first."}

    def _example(v):
        if isinstance(v, list):
            return [(_example(x)) for x in v[:2]]
        s = str(v)
        return s if len(s) <= 60 else s[:60] + "…"

    def _top(name: str) -> str:  # top-level key of a dotted/array-path field spec
        return name.split(".")[0].split("[")[0].strip()

    agg: dict = {}      # structured_json key -> {type, example}
    record: dict = {}   # raw top-level key -> {aggregatable, searchable}
    fm_cache: dict = {}
    step = max(1, total // sample)
    seen = 0
    for rid in range(1, total + 1, step):
        row = conn.execute(
            "SELECT structured_json, raw_json, source_path FROM documents WHERE rowid=?", (rid,)
        ).fetchone()
        if row is None:
            continue
        seen += 1
        sp = row["source_path"]
        fm = fm_cache.get(sp)
        if fm is None:
            fm = fields_for(settings, sp)
            fm_cache[sp] = fm
        text_fields = fm.text_fields or []
        # A field is *searchable* iff the field map renders it into documents.text: either the
        # source has no text_fields (whole record flattened) or the field is a text/entity field.
        renders_all = not text_fields
        rendered_tops = {_top(t) for t in text_fields}
        rendered_tops |= {_top(spec.partition(":")[0]) for spec in (fm.entity_fields or [])}

        sj = json.loads(row["structured_json"]) if row["structured_json"] else {}
        for k, v in sj.items():
            if k.startswith("__"):
                continue
            a = agg.setdefault(k, {"type": "list" if isinstance(v, list) else "scalar", "example": None})
            if a["example"] is None and v not in (None, "", []):
                a["example"] = _example(v)
        rj = json.loads(row["raw_json"]) if row["raw_json"] else {}
        if isinstance(rj, dict):
            for k in rj:
                e = record.setdefault(k, {"aggregatable": False, "searchable": False})
                e["aggregatable"] = e["aggregatable"] or (k in sj)
                e["searchable"] = e["searchable"] or renders_all or (k in rendered_tops)

    return {
        "documents": total,
        "documents_sampled": seen,
        "aggregatable_fields": agg,
        "record_fields": {k: record[k] for k in sorted(record)},
        "note": (
            "aggregatable_fields → `muck aggregate --by <name>`. record_fields lists EVERY field in "
            "the source records: `searchable: true` means it is rendered into the document text, so "
            "`muck grep`/`muck search` find it and `muck verify` can cite it — even when "
            "`aggregatable: false` (e.g. nested arrays like foreign_entities / conviction_disclosures "
            "/ lobbying_activities). Grep the field name or a value. NEVER conclude a field is absent "
            "by reasoning about the config — check here, or grep the field name directly."
        ),
    }


def _entity_inertia_notice(conn) -> str | None:
    """Explain near-empty entity counts on parse-only corpora (no JSON entity_fields)."""
    ph = ",".join("?" * len(PARSER_TYPES))
    parsed = conn.execute(
        f"SELECT COUNT(*) FROM documents WHERE doc_type IN ({ph})", sorted(PARSER_TYPES)
    ).fetchone()[0]
    ph = ",".join("?" * len(MAPPER_TYPES))
    mapped = conn.execute(
        f"SELECT COUNT(*) FROM documents WHERE doc_type IN ({ph})", sorted(MAPPER_TYPES)
    ).fetchone()[0]
    if parsed and not mapped:
        return (
            "entity extraction is minimal on this corpus: mentions come from JSON "
            "entity_fields (none here) plus bill-number regex, so --person/--org "
            "filters and the entity network will be empty. "
            "search/grep/read/aggregate/cluster are unaffected."
        )
    return None


# --- status ------------------------------------------------------------------
# (observations summary is attached in status() below)

def status(conn, settings: Settings) -> dict:
    def count(sql, *p):
        return conn.execute(sql, p).fetchone()[0]

    from .db import meta_get, vec_available

    files_by_status = {
        r["status"]: r["n"]
        for r in conn.execute("SELECT status, COUNT(*) n FROM source_files GROUP BY status")
    }
    docs_by_status = {
        r["status"]: r["n"]
        for r in conn.execute("SELECT status, COUNT(*) n FROM documents GROUP BY status")
    }
    result = {
        "source_files": files_by_status,
        "documents": docs_by_status,
        "chunks": count("SELECT COUNT(*) FROM chunks"),
        "fts_rows": count("SELECT COUNT(*) FROM chunks_fts"),
        "entities": count("SELECT COUNT(*) FROM entities"),
        "mentions": count("SELECT COUNT(*) FROM mentions"),
        "findings": count("SELECT COUNT(*) FROM findings"),
        "embeddings_enabled": settings.embedder.enabled,
        "embedder": settings.embedder.name if settings.embedder.enabled else None,
        "embed_dim": meta_get(conn, "embed_dim"),
        "vector_backend_available": vec_available(conn),
        "index_provenance": {
            "muck_version": meta_get(conn, "muck_version"),
            "embed_model": meta_get(conn, "embed_model"),
            "embed_revision": meta_get(conn, "embed_revision") or None,
            "config_hash": meta_get(conn, "config_hash"),
        },
    }
    # Surface per-file failure reasons (written to source_files.error, otherwise only reachable
    # by opening the DB) so an agent can self-diagnose a failed map/parse.
    errored = [
        {"file": r["path"], "error": r["error"]}
        for r in conn.execute("SELECT path, error FROM source_files WHERE status='error'")
    ]
    if errored:
        result["errored_files"] = errored
    # Next-step hint (matches the ocr/finding next-action convention) so a fresh/empty corpus
    # is not a dead end.
    total_files = sum(files_by_status.values())
    if total_files == 0:
        result["next"] = "no sources yet — run `muck build <files-or-dirs>` (or `muck ingest`)"
    elif result["chunks"] == 0:
        result["next"] = "sources ingested but not indexed — run `muck build --all` or `muck index --all`"

    from . import observations

    result["observations"] = observations.open_summary(conn)
    drift = field_map_drift(conn, settings)
    if drift:
        result["config_drift"] = drift
    notice = _entity_inertia_notice(conn)
    if notice:
        result["notice"] = notice
    coverage_notice = _text_coverage_notice(conn)
    if coverage_notice:
        result["coverage_notice"] = coverage_notice
    return result
