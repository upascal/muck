"""SQLite connection, schema, and small helpers.

One file (``.muck/index.db``) holds the manifest, documents, chunks, the FTS5 keyword
index, the sqlite-vec vector index, index metadata, and the audit trace log.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

DB_FILENAME = "index.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_files (
    file_id      TEXT PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    file_type    TEXT NOT NULL,
    n_docs       INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ingested',   -- ingested|processed|error
    error        TEXT,
    ingested_at  TEXT,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    file_id         TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    locator         TEXT NOT NULL DEFAULT '',
    doc_type        TEXT NOT NULL,
    title           TEXT,
    text            TEXT NOT NULL,
    n_pages         INTEGER NOT NULL DEFAULT 0,
    pages_json      TEXT,
    structured_json TEXT,
    raw_json        TEXT,
    source_name     TEXT,
    status          TEXT NOT NULL DEFAULT 'extracted',  -- extracted|indexed
    created_at      TEXT,
    indexed_at      TEXT
);
CREATE INDEX IF NOT EXISTS ix_documents_file ON documents(file_id);
CREATE INDEX IF NOT EXISTS ix_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    locator     TEXT NOT NULL DEFAULT '',
    token_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, chunk_id UNINDEXED, tokenize='porter unicode61');

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS trace_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT,
    session_id     TEXT,
    command        TEXT,
    args_json      TEXT,
    result_summary TEXT,
    n_results      INTEGER
);

-- Entity index (Phase 1): canonical entities, their aliases, mentions, co-occurrence.
CREATE TABLE IF NOT EXISTS entities (
    entity_id     TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type   TEXT NOT NULL,       -- org|person|bill
    norm_key      TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 0,
    doc_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_entities_type ON entities(entity_type);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id TEXT NOT NULL,
    alias     TEXT NOT NULL,
    source    TEXT,
    PRIMARY KEY (entity_id, alias)
);

CREATE TABLE IF NOT EXISTS mentions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  TEXT NOT NULL,
    chunk_id   TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    raw_text   TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    locator    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_mentions_entity ON mentions(entity_id);
CREATE INDEX IF NOT EXISTS ix_mentions_doc ON mentions(doc_id);

CREATE TABLE IF NOT EXISTS entity_edges (
    src_entity_id TEXT NOT NULL,
    dst_entity_id TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 0,
    edge_type     TEXT NOT NULL DEFAULT 'co_doc',
    PRIMARY KEY (src_entity_id, dst_entity_id, edge_type)
);

-- Typed relations (directed, ground-truth from structured fields — distinct from the symmetric
-- co_doc co-occurrence hints above). Each row derives from a specific record, so it carries a
-- representative doc_id the agent can cite: a lead that feeds the findings/audit assert layer.
CREATE TABLE IF NOT EXISTS entity_relations (
    src_entity_id TEXT NOT NULL,
    dst_entity_id TEXT NOT NULL,
    predicate     TEXT NOT NULL,          -- e.g. lobbied_for, donated_to
    weight        INTEGER NOT NULL DEFAULT 0,   -- # of records establishing this relation
    doc_id        TEXT,                    -- a representative source document (for citation)
    PRIMARY KEY (src_entity_id, dst_entity_id, predicate)
);
CREATE INDEX IF NOT EXISTS ix_entity_relations_src ON entity_relations(src_entity_id);
CREATE INDEX IF NOT EXISTS ix_entity_relations_dst ON entity_relations(dst_entity_id);

-- Topical clustering (Phase 2): KMeans over embeddings + c-tf-idf labels.
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    label      TEXT,
    terms_json TEXT,
    size       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunk_clusters (
    chunk_id   TEXT PRIMARY KEY,
    cluster_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunk_clusters_cluster ON chunk_clusters(cluster_id);

-- Splink record reconciliation (opt-in): probabilistic multi-field linkage, fully audited.
CREATE TABLE IF NOT EXISTS reconcile_runs (
    run_id        TEXT PRIMARY KEY,
    method        TEXT,                 -- e.g. "splink-fellegi-sunter"
    on_field      TEXT,
    compare_fields TEXT,
    threshold     REAL,
    link_dataset  TEXT,                 -- external dataset path, or NULL for self-dedupe
    settings_json TEXT,                 -- full Splink settings (m/u priors, blocking)
    n_records     INTEGER,
    n_clusters    INTEGER,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS reconcile_records (
    run_id      TEXT NOT NULL,
    record_id   TEXT NOT NULL,          -- doc_id (corpus) or "ext:<i>" (linked dataset)
    source      TEXT NOT NULL,          -- 'corpus' or 'link:<file>'
    cluster_id  TEXT NOT NULL,          -- canonical entity within this run
    name        TEXT,
    fields_json TEXT,
    PRIMARY KEY (run_id, record_id)
);
CREATE INDEX IF NOT EXISTS ix_reconcile_cluster ON reconcile_records(run_id, cluster_id);
CREATE TABLE IF NOT EXISTS reconcile_links (
    run_id            TEXT NOT NULL,
    src_record_id     TEXT NOT NULL,
    dst_record_id     TEXT NOT NULL,
    match_probability REAL NOT NULL     -- the Splink score that justified the link
);
CREATE INDEX IF NOT EXISTS ix_reconcile_links_run ON reconcile_links(run_id);

-- Observations: the agent's investigative working memory (notes/leads/threads), distinct
-- from trace_log (mechanical) and findings (verified claims). Keeps work oriented across sessions.
CREATE TABLE IF NOT EXISTS observations (
    obs_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL DEFAULT 'note',   -- note|decision|lead|hypothesis|question|dead_end
    status     TEXT NOT NULL DEFAULT 'open',   -- open|confirmed|cold|dropped
    text       TEXT NOT NULL,
    entity_ids TEXT,                            -- JSON list of related entity_ids
    doc_ids    TEXT,                            -- JSON list of related doc_ids
    tokens     TEXT,                            -- JSON list of citation tokens (evidence)
    session_id TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_observations_status ON observations(status);
CREATE INDEX IF NOT EXISTS ix_observations_kind ON observations(kind);

-- Findings ledger (Phase 1): claim + verifiable citation, auditable.
CREATE TABLE IF NOT EXISTS findings (
    finding_id     TEXT PRIMARY KEY,
    claim          TEXT NOT NULL,
    quote          TEXT NOT NULL,
    citation_token TEXT NOT NULL,
    support_json   TEXT,
    status         TEXT NOT NULL DEFAULT 'unaudited',
    created_at     TEXT,
    audited_at     TEXT
);

-- Pixel reviews: a human (or agent) confirmed a non-native-text finding against the page
-- IMAGE. This is what lets an OCR/vision finding move past 'pending_review' — the transcript
-- guarantee proves only that our transcript says X; a pixel review proves the scan says X.
CREATE TABLE IF NOT EXISTS pixel_reviews (
    finding_id  TEXT PRIMARY KEY,
    verdict     TEXT NOT NULL,          -- confirmed|rejected
    reviewer    TEXT,                   -- $MUCK_REVIEWER or "agent"
    note        TEXT,
    page_image  TEXT,
    reviewed_at TEXT
);
"""

# Columns added after the initial schema shipped. `init_schema` ALTERs them in on any
# existing DB (CREATE TABLE IF NOT EXISTS never adds columns), so a live index migrates on
# first touch — no rebuild. (table, column, DDL type + default.)
MIGRATIONS: list[tuple[str, str, str]] = [
    # How a document's text was derived; the roll-up of its pages' tiers. Drives honest
    # citation scope (native = the file says this; ocr/vision = our transcript says this).
    ("documents", "text_provenance", "TEXT NOT NULL DEFAULT 'native'"),
    # Fraction of pages with usable extractable text [0..1]; 0 = image-only scan.
    ("documents", "coverage", "REAL"),
    # Findings carry their source tier so `audit` can bucket native vs needs-human-review.
    ("findings", "text_provenance", "TEXT NOT NULL DEFAULT 'native'"),
]


@lru_cache(maxsize=256)
def _compiled(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def _regexp(pattern: str, value: str | None) -> int:
    if value is None:
        return 0
    return 1 if _compiled(pattern).search(value) else 0


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.create_function("regexp", 2, _regexp, deterministic=True)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        pass  # vector path degrades gracefully; keyword search still works
    return conn


def vec_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT vec_version()")
        return True
    except sqlite3.OperationalError:
        return False


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema to an existing DB (idempotent)."""
    for table, column, ddl in MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


def meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM index_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO index_meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
