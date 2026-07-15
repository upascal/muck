"""Exact structured aggregation over JSON-record fields.

Lobbying/disclosure questions ("who spent the most", "filings per quarter") are tabular, not
fuzzy — answered by deterministic SQL over each document's ``structured_json``. The analysis
engine is **DuckDB** when available (columnar/vectorized; reads external files for joins),
attaching the SQLite source of truth read-only; **SQLite json1** is the zero-dependency
fallback. The structured ``--by/--measure`` path is dialect-transparent; results are exact and
reproducible (a finding can cite the query + engine + row count).
"""

from __future__ import annotations

import re
import sqlite3

from ..interfaces import NotInstalled
from . import duck

_FIELD_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_AGGS = {"sum": "SUM", "avg": "AVG", "max": "MAX", "min": "MIN"}


def _path(field: str) -> str:
    if not _FIELD_RE.match(field):
        raise ValueError(f"invalid field name: {field!r}")
    # structured_json stores field-map spec names as FLAT keys (a spec like
    # "registrant.name" is one key, not nesting) — quote so json_extract agrees.
    return f'$."{field}"'


def resolve_engine(engine: str) -> str:
    """Map 'auto' -> duckdb if available else sqlite; validate an explicit choice."""
    if engine == "auto":
        return "duckdb" if duck.available() else "sqlite"
    if engine == "duckdb" and not duck.available():
        raise NotInstalled(
            "DuckDB engine requested but unavailable; `uv sync --extra analytics` "
            "(one-time network for the sqlite extension), or use --engine sqlite"
        )
    return engine


def _is_array_field(conn, by_path: str) -> bool:
    """True if this structured field stores a JSON array (a multi-valued array-path field).

    Uses the 2-arg ``json_type(json, path)`` (navigates within the valid structured_json) —
    ``json_type(json_extract(...))`` would choke on scalar strings, which aren't valid JSON alone.
    """
    row = conn.execute(
        "SELECT json_type(structured_json, ?) AS t FROM documents "
        "WHERE json_extract(structured_json, ?) IS NOT NULL LIMIT 1",
        (by_path, by_path),
    ).fetchone()
    return bool(row) and row["t"] == "array"


def _raw_groups_sqlite(conn, by: str, measure: str | None, agg: str) -> list[dict]:
    by_path = _path(by)
    # An array-path field (e.g. every government entity a filing lobbied) is multi-valued:
    # UNNEST it with json_each so each element is its own group. A filing thus contributes to
    # every element it lists (its measure counted once per element it targets).
    array = _is_array_field(conn, by_path)
    if agg == "count":
        if array:
            sql = ("SELECT je.value AS group_value, COUNT(*) AS value, COUNT(*) AS n "
                   "FROM documents, json_each(json_extract(structured_json, ?)) je "
                   "WHERE json_extract(structured_json, ?) IS NOT NULL GROUP BY group_value")
        else:
            sql = ("SELECT json_extract(structured_json, ?) AS group_value, COUNT(*) AS value, "
                   "COUNT(*) AS n FROM documents "
                   "WHERE json_extract(structured_json, ?) IS NOT NULL GROUP BY group_value")
        params = (by_path, by_path)
    else:
        if agg not in _AGGS or not measure:
            raise ValueError("non-count aggregation requires a numeric --measure and a valid --agg")
        m_path, fn = _path(measure), _AGGS[agg]
        if array:
            sql = (f"SELECT je.value AS group_value, "
                   f"{fn}(CAST(json_extract(structured_json, ?) AS REAL)) AS value, COUNT(*) AS n "
                   f"FROM documents, json_each(json_extract(structured_json, ?)) je "
                   f"WHERE json_extract(structured_json, ?) IS NOT NULL GROUP BY group_value")
            params = (m_path, by_path, by_path)
        else:
            sql = (f"SELECT json_extract(structured_json, ?) AS group_value, "
                   f"{fn}(CAST(json_extract(structured_json, ?) AS REAL)) AS value, COUNT(*) AS n "
                   f"FROM documents WHERE json_extract(structured_json, ?) IS NOT NULL GROUP BY group_value")
            params = (by_path, m_path, by_path)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _raw_groups_duck(db_path: str, by: str, measure: str | None, agg: str) -> list[dict]:
    bexpr = f"json_extract_string(structured_json, '{_path(by)}')"
    if agg == "count":
        select, where = f"{bexpr} AS group_value, COUNT(*) AS value, COUNT(*) AS n", f"{bexpr} IS NOT NULL"
    else:
        if agg not in _AGGS or not measure:
            raise ValueError("non-count aggregation requires a numeric --measure and a valid --agg")
        mexpr = f"TRY_CAST(json_extract_string(structured_json, '{_path(measure)}') AS DOUBLE)"
        select = f"{bexpr} AS group_value, {_AGGS[agg]}({mexpr}) AS value, COUNT(*) AS n"
        where = f"{bexpr} IS NOT NULL"
    con = duck.connect(db_path)
    try:
        cur = con.execute(f"SELECT {select} FROM muck.documents WHERE {where} GROUP BY group_value")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def _engine_for(engine: str, db_path: str | None) -> str:
    # DuckDB needs the DB file path to attach; without it, only SQLite is possible.
    return resolve_engine(engine) if db_path else "sqlite"


def aggregate(conn, by: str, measure: str | None = None, agg: str = "count", limit: int = 20,
              resolve: str | None = None, engine: str = "auto", db_path: str | None = None,
              settings=None) -> list[dict]:
    """GROUP BY ``by``; COUNT or {sum,avg,max,min} of ``measure``.

    With ``resolve`` ('org'|'person'), raw field values are folded onto their canonical
    resolved entity first — so "Acme Strategies LLC" and "Acme Strategies, L.L.C." sum
    together. (Folding runs in Python via the entities table, so it is engine-independent.)
    """
    eng = _engine_for(engine, db_path)
    if eng == "duckdb" and _is_array_field(conn, _path(by)):
        eng = "sqlite"  # array-path unnest is implemented on the sqlite json_each path
    rows = _raw_groups_duck(db_path, by, measure, agg) if eng == "duckdb" \
        else _raw_groups_sqlite(conn, by, measure, agg)

    if not resolve:
        rows.sort(key=lambda r: (r["value"] is not None, r["value"]), reverse=True)
        return [dict(r) for r in rows[:limit]]

    if agg == "avg":
        raise ValueError("avg is not supported with --resolve; use --agg sum or drop --resolve")

    from ..extract.entities import normalize_key

    # Fold with the same suffix set the entity index used, or the two disagree.
    extra = tuple(getattr(getattr(settings, "entities", None), "extra_org_suffixes", ()) or ())
    canon = {
        r["norm_key"]: r["canonical_name"]
        for r in conn.execute("SELECT norm_key, canonical_name FROM entities WHERE entity_type=?", (resolve,))
    }
    fold = {"count": sum, "sum": sum, "min": min, "max": max}[agg]
    merged: dict[str, dict] = {}
    for r in rows:
        gv = r["group_value"]
        key = normalize_key(str(gv), resolve, extra) if gv is not None else ""
        name = canon.get(key, gv)
        m = merged.setdefault(name, {"group_value": name, "_vals": [], "n": 0})
        if r["value"] is not None:
            m["_vals"].append(r["value"])
        m["n"] += r["n"]
    for m in merged.values():
        m["value"] = fold(m["_vals"]) if m["_vals"] else 0
        del m["_vals"]
    return sorted(merged.values(), key=lambda r: r["value"], reverse=True)[:limit]


def run_sql(conn: sqlite3.Connection, sql: str, limit: int = 100,
            engine: str = "auto", db_path: str | None = None) -> list[dict]:
    """Read-only SQL passthrough (SELECT/WITH only).

    On DuckDB the corpus is the ``muck.*`` schema and queries may read external files
    (``read_json_auto('…')`` / ``read_csv_auto`` / ``read_parquet``) for exact joins; the
    read-only attach means the corpus can never be mutated.
    """
    stripped = sql.strip().rstrip(";")
    if not re.match(r"(?is)^(select|with)\b", stripped):
        raise ValueError("only read-only SELECT/WITH queries are allowed")
    if _engine_for(engine, db_path) == "duckdb":
        con = duck.connect(db_path)
        try:
            cur = con.execute(f"SELECT * FROM ({stripped}) LIMIT {int(limit)}")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            con.close()
    rows = conn.execute(f"SELECT * FROM ({stripped}) LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
