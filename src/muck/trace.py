"""Append-only audit trace: every CLI call -> DB row + .muck/trace.jsonl line.

Combined with the agent's session transcript, this lets an editor audit *what was asked*
and *what the deterministic tools returned*, without re-running the investigation.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _session_id() -> str:
    return os.environ.get("MUCK_SESSION") or os.environ.get("CLAUDE_SESSION_ID") or "local"


def log(
    conn: sqlite3.Connection,
    muck_dir: Path,
    command: str,
    args: dict,
    summary: str = "",
    n_results: int = 0,
) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sid = _session_id()
    args_json = json.dumps(args, default=str)
    conn.execute(
        "INSERT INTO trace_log(ts, session_id, command, args_json, result_summary, n_results) "
        "VALUES(?,?,?,?,?,?)",
        (ts, sid, command, args_json, summary, n_results),
    )
    conn.commit()
    line = {
        "ts": ts, "session": sid, "command": command,
        "args": args, "summary": summary, "n_results": n_results,
    }
    with (muck_dir / "trace.jsonl").open("a") as fh:
        fh.write(json.dumps(line, default=str) + "\n")


def tail(conn: sqlite3.Connection, n: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, session_id, command, args_json, result_summary, n_results "
        "FROM trace_log ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]
