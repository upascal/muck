"""Default store: FTS5 keyword/BM25 + grep + sqlite-vec ANN, all in one DB file."""

from __future__ import annotations

import re
import sqlite3

from ..db import meta_get, meta_set, vec_available
from ..interfaces.store import register_store
from ..schema import Chunk

_TOKEN = re.compile(r"[\w'-]+")


def _fts_safe(query: str) -> str:
    """Quote each term so arbitrary user text is a valid FTS5 MATCH (AND semantics)."""
    terms = _TOKEN.findall(query)
    return " ".join(f'"{t}"' for t in terms)


def _filter_sql(filters: dict | None) -> tuple[str, list]:
    if not filters:
        return "", []
    clauses, params = [], []
    if filters.get("doc_type"):
        clauses.append("d.doc_type = ?")
        params.append(filters["doc_type"])
    if filters.get("doc_id"):
        clauses.append("d.doc_id = ?")
        params.append(filters["doc_id"])
    if filters.get("path_like"):
        clauses.append("d.source_path LIKE ?")
        params.append(filters["path_like"])
    if filters.get("chunk_ids"):
        ids = list(filters["chunk_ids"])
        clauses.append(f"c.chunk_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def _ws_tolerant(pattern: str) -> str:
    """Rewrite literal spaces to ``\\s+`` so phrases match across PDF line breaks.

    Escaped spaces and spaces inside ``[...]`` classes are untouched; a single space
    directly before a quantifier becomes ``\\s`` so the quantifier keeps its meaning.
    """
    out: list[str] = []
    i, in_class = 0, False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            out.append(pattern[i : i + 2])
            i += 2
            continue
        if ch == "[" and not in_class:
            in_class = True
        elif ch == "]" and in_class:
            in_class = False
        elif ch == " " and not in_class:
            j = i
            while j < len(pattern) and pattern[j] == " ":
                j += 1
            nxt = pattern[j] if j < len(pattern) else ""
            out.append(r"\s" if (j - i == 1 and nxt in "?*+{") else r"\s+")
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _snippet(text: str, around: str | None = None, width: int = 220) -> str:
    if around:
        m = re.search(re.escape(around), text, re.IGNORECASE)
        if m:
            lo = max(0, m.start() - width // 2)
            hi = min(len(text), m.end() + width // 2)
            return ("…" if lo else "") + text[lo:hi].strip() + ("…" if hi < len(text) else "")
    return text[:width].strip() + ("…" if len(text) > width else "")


class SqliteStore:
    name = "sqlite-fts5"

    # --- keyword ---------------------------------------------------------
    def index_chunks(self, conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
        conn.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, chunk_index, text, char_start, "
            "char_end, locator, token_count) VALUES(?,?,?,?,?,?,?,?)",
            [
                (c.chunk_id, c.doc_id, c.chunk_index, c.text, c.char_start,
                 c.char_end, c.locator, c.token_count)
                for c in chunks
            ],
        )
        conn.executemany(
            "INSERT INTO chunks_fts(chunk_id, text) VALUES(?, ?)",
            [(c.chunk_id, c.text) for c in chunks],
        )

    def search_bm25(self, conn, query, k, filters=None):
        match = _fts_safe(query)
        if not match:
            return []
        where, params = _filter_sql(filters)
        sql = (
            "SELECT f.chunk_id AS chunk_id, bm25(chunks_fts) AS score, "
            "snippet(chunks_fts, 0, '[', ']', '…', 14) AS snip "
            "FROM chunks_fts f "
            "JOIN chunks c ON c.chunk_id = f.chunk_id "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE chunks_fts MATCH ?" + where + " ORDER BY score LIMIT ?"
        )
        rows = conn.execute(sql, [match, *params, k]).fetchall()
        # bm25() returns lower = better; flip so higher = better.
        return [(r["chunk_id"], -float(r["score"]), r["snip"]) for r in rows]

    def grep(self, conn, pattern, k, filters=None):
        pattern = _ws_tolerant(pattern)
        where, params = _filter_sql(filters)
        sql = (
            "SELECT c.chunk_id AS chunk_id, c.text AS text FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE regexp(?, c.text)" + where + " ORDER BY c.doc_id, c.char_start"
        )
        # Single full scan: exact total alongside the first k hits (k <= 0 keeps all).
        out, total = [], 0
        for r in conn.execute(sql, [pattern, *params]):
            total += 1
            if k <= 0 or len(out) < k:
                m = re.search(pattern, r["text"], re.IGNORECASE)
                out.append((r["chunk_id"], _snippet(r["text"], m.group(0) if m else None)))
        return out, total

    # --- vector ----------------------------------------------------------
    def supports_vectors(self, conn) -> bool:
        return vec_available(conn)

    def _ensure_vec_table(self, conn, dim: int) -> None:
        stored = meta_get(conn, "embed_dim")
        if stored is None:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec "
                f"USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dim}])"
            )
            meta_set(conn, "embed_dim", dim)
        elif int(stored) != dim:
            raise ValueError(
                f"embedding dim {dim} != existing index dim {stored}; re-index after "
                f"changing embedder (delete .muck/index.db or run `muck index --reset`)"
            )

    def upsert_vectors(self, conn, ids, vecs) -> None:
        if not vec_available(conn):
            return
        self._ensure_vec_table(conn, int(vecs.shape[1]))
        for cid, vec in zip(ids, vecs):
            conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (cid,))
            conn.execute(
                "INSERT INTO chunks_vec(chunk_id, embedding) VALUES(?, vec_f32(?))",
                (cid, vec.astype("float32").tobytes()),
            )

    def get_vectors(self, conn, chunk_ids):
        import json

        import numpy as np

        if not chunk_ids or not vec_available(conn) or meta_get(conn, "embed_dim") is None:
            return [], None
        ids = list(chunk_ids)
        out_ids: list[str] = []
        rows_vecs: list[list[float]] = []
        for i in range(0, len(ids), 900):  # stay under SQLite's bound-variable limit
            batch = ids[i:i + 900]
            placeholders = ",".join("?" * len(batch))
            for r in conn.execute(
                f"SELECT chunk_id, vec_to_json(embedding) AS v FROM chunks_vec "
                f"WHERE chunk_id IN ({placeholders})",
                batch,
            ):
                out_ids.append(r["chunk_id"])
                rows_vecs.append(json.loads(r["v"]))
        if not rows_vecs:
            return [], None
        return out_ids, np.asarray(rows_vecs, dtype="float32")

    def search_vector(self, conn, vec, k, filters=None):
        if not vec_available(conn) or meta_get(conn, "embed_dim") is None:
            return []
        blob = vec.astype("float32").tobytes()
        where, params = _filter_sql(filters)
        inner_k = k * 6 if where else k
        sql = (
            "SELECT v.chunk_id AS chunk_id, v.distance AS distance FROM "
            "(SELECT chunk_id, distance FROM chunks_vec "
            " WHERE embedding MATCH ? AND k = ?) v "
            "JOIN chunks c ON c.chunk_id = v.chunk_id "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE 1=1" + where + " ORDER BY v.distance LIMIT ?"
        )
        rows = conn.execute(sql, [blob, inner_k, *params, k]).fetchall()
        return [(r["chunk_id"], float(r["distance"])) for r in rows]


register_store("sqlite-fts5", SqliteStore)
