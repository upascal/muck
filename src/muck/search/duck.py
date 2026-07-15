"""DuckDB analysis engine over the SQLite source of truth (read-only).

DuckDB attaches ``.muck/index.db`` **read-only** — SQLite stays authoritative — and does the
heavy aggregation (columnar/vectorized), with the ability to read external CSV/JSON/Parquet
and join them to the corpus. Used by ``muck aggregate``; the SQLite json1 path is the
zero-dependency fallback when DuckDB (or its ``sqlite`` extension) isn't available.
"""

from __future__ import annotations

from ..interfaces import NotInstalled

_AVAILABLE: bool | None = None  # probe once per process (sqlite ext auto-installs on first use)


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import duckdb

            con = duckdb.connect()
            con.execute("INSTALL sqlite; LOAD sqlite;")
            con.close()
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def connect(db_path: str):
    """Open an in-memory DuckDB with the corpus attached read-only as schema ``muck``."""
    try:
        import duckdb
    except ImportError as e:  # pragma: no cover
        raise NotInstalled("DuckDB analysis needs `uv sync --extra analytics`") from e
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    safe = str(db_path).replace("'", "''")
    con.execute(f"ATTACH '{safe}' AS muck (TYPE sqlite, READ_ONLY)")
    return con
