"""Observations — the agent's investigative working memory (Tier 1).

The third memory layer beside ``trace_log`` (mechanical) and ``findings`` (verified claims):
curated notes, decisions/rationale, leads, hypotheses, open questions, dead ends — each with a
status and optional links to entities / documents / citation tokens. Lives in the same
``.muck/index.db`` so it joins to the rest as plain SQL. This is what keeps a long
investigation oriented across sessions; ``muck status`` surfaces the open threads.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

KINDS = {"note", "decision", "lead", "hypothesis", "question", "dead_end"}
STATUSES = {"open", "confirmed", "cold", "dropped"}
THREAD_KINDS = ("lead", "hypothesis", "question")  # the "what's still open" kinds


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _session() -> str:
    return os.environ.get("MUCK_SESSION") or os.environ.get("CLAUDE_SESSION_ID") or "local"


def _jsonl(values) -> str | None:
    # note: `list` is shadowed by the public list() below, so build with a comprehension
    return json.dumps([v for v in values]) if values else None


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for f in ("entity_ids", "doc_ids", "tokens"):
        d[f] = json.loads(d[f]) if d.get(f) else []
    return d


def add(conn, text: str, kind: str = "note", status: str = "open",
        entity_ids=None, doc_ids=None, tokens=None, session_id: str | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; one of {sorted(KINDS)}")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; one of {sorted(STATUSES)}")
    now = _now()
    cur = conn.execute(
        "INSERT INTO observations(kind, status, text, entity_ids, doc_ids, tokens, "
        "session_id, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (kind, status, text, _jsonl(entity_ids), _jsonl(doc_ids), _jsonl(tokens),
         session_id or _session(), now, now),
    )
    conn.commit()
    return {"obs_id": cur.lastrowid, "kind": kind, "status": status, "text": text}


def list(conn, status: str | None = None, kind: str | None = None,
         entity: str | None = None, limit: int = 50) -> dict:  # noqa: A001 (mirrors CLI verb)
    where, params = [], []
    if status:
        where.append("status = ?")
        params.append(status)
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if entity:
        where.append("entity_ids LIKE ?")
        params.append(f'%"{entity}"%')
    sql = "SELECT * FROM observations"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC, obs_id DESC LIMIT ?"
    return [_row(r) for r in conn.execute(sql, [*params, limit]).fetchall()]


def update(conn, obs_id: int, status: str | None = None, text: str | None = None) -> dict:
    sets, params = [], []
    if status is not None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; one of {sorted(STATUSES)}")
        sets.append("status = ?")
        params.append(status)
    if text is not None:
        sets.append("text = ?")
        params.append(text)
    if not sets:
        return {"error": "nothing to update (provide --status and/or --text)"}
    sets.append("updated_at = ?")
    params.append(_now())
    params.append(obs_id)
    cur = conn.execute(f"UPDATE observations SET {', '.join(sets)} WHERE obs_id = ?", params)
    conn.commit()
    if cur.rowcount == 0:
        return {"error": f"unknown obs_id {obs_id}"}
    return get(conn, obs_id)


def get(conn, obs_id: int) -> dict:
    row = conn.execute("SELECT * FROM observations WHERE obs_id = ?", (obs_id,)).fetchone()
    if row is None:
        return {"error": f"unknown obs_id {obs_id}"}
    d = _row(row)
    if d["entity_ids"]:
        ph = ",".join("?" * len(d["entity_ids"]))
        d["entities"] = [
            {"entity_id": r["entity_id"], "canonical_name": r["canonical_name"]}
            for r in conn.execute(
                f"SELECT entity_id, canonical_name FROM entities WHERE entity_id IN ({ph})",
                d["entity_ids"],
            )
        ]
    return d


def open_summary(conn) -> dict:
    by_status = {
        r["status"]: r["n"]
        for r in conn.execute("SELECT status, COUNT(*) n FROM observations GROUP BY status")
    }
    threads_ph = ",".join("?" * len(THREAD_KINDS))
    open_threads = [
        {"obs_id": r["obs_id"], "kind": r["kind"], "text": r["text"]}
        for r in conn.execute(
            f"SELECT obs_id, kind, text FROM observations "
            f"WHERE status='open' AND kind IN ({threads_ph}) "
            f"ORDER BY updated_at DESC, obs_id DESC LIMIT 10",
            THREAD_KINDS,
        )
    ]
    return {"by_status": by_status, "open_threads": open_threads}
