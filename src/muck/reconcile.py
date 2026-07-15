"""Probabilistic record reconciliation via Splink (opt-in: ``--extra splink``).

Where the deterministic mention-resolver dedupes on a normalized key, this reconciles
messy *multi-field* records — the same firm/person filed under drifting names/addresses —
and can **link the corpus to an outside dataset** (`link_path`). It uses Fellegi-Sunter
scoring with **name-dominant m/u priors**: name similarity carries the weight, other fields
are weak corroboration that cannot override a name mismatch (so two firms sharing a building
don't get merged). Every merge is audited — pairwise match probabilities, cluster
assignments, and the method/threshold/settings are all persisted (see the ``reconcile_*``
tables) and reversible by re-running at a different threshold.

Defaults work without training (robust on any corpus size); pass ``train=True`` to refine
``u`` by random sampling on a large corpus.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .db import meta_set
from .interfaces import NotInstalled

DEFAULT_THRESHOLD = 0.9
_LAMBDA = 0.1  # prior probability two random records match


def _name_comparison(col: str) -> dict:
    """Dominant signal: near-exact names score high, mismatched names score strongly negative."""
    return {
        "output_column_name": col,
        "comparison_levels": [
            {"sql_condition": f"{col}_l IS NULL OR {col}_r IS NULL", "label_for_charts": "Null", "is_null_level": True},
            {"sql_condition": f"{col}_l = {col}_r", "label_for_charts": "Exact", "m_probability": 0.55, "u_probability": 0.0001},
            {"sql_condition": f"jaro_winkler_similarity({col}_l, {col}_r) >= 0.9", "label_for_charts": "JW>=0.9", "m_probability": 0.30, "u_probability": 0.005},
            {"sql_condition": f"jaro_winkler_similarity({col}_l, {col}_r) >= 0.8", "label_for_charts": "JW>=0.8", "m_probability": 0.12, "u_probability": 0.05},
            {"sql_condition": "ELSE", "label_for_charts": "All other", "m_probability": 0.03, "u_probability": 0.945},
        ],
    }


def _weak_comparison(col: str) -> dict:
    """Corroboration only: low odds so it can't merge records whose names differ."""
    return {
        "output_column_name": col,
        "comparison_levels": [
            {"sql_condition": f"{col}_l IS NULL OR {col}_r IS NULL", "label_for_charts": "Null", "is_null_level": True},
            {"sql_condition": f"{col}_l = {col}_r", "label_for_charts": "Exact", "m_probability": 0.5, "u_probability": 0.2},
            {"sql_condition": f"jaro_winkler_similarity({col}_l, {col}_r) >= 0.8", "label_for_charts": "JW>=0.8", "m_probability": 0.3, "u_probability": 0.3},
            {"sql_condition": "ELSE", "label_for_charts": "All other", "m_probability": 0.2, "u_probability": 0.5},
        ],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _corpus_rows(conn, on: str, compare: list[str]) -> list[dict]:
    rows = []
    for r in conn.execute("SELECT doc_id, structured_json FROM documents WHERE structured_json IS NOT NULL"):
        sj = json.loads(r["structured_json"])
        name = sj.get(on)
        if name is None or str(name).strip() == "":
            continue
        row = {"unique_id": r["doc_id"], "source_dataset": "corpus", on: str(name)}
        for f in compare:
            v = sj.get(f)
            row[f] = None if v is None or str(v).strip() == "" else str(v)
        rows.append(row)
    return rows


def _link_rows(path: str, on: str, compare: list[str]) -> list[dict]:
    data = json.loads(Path(path).read_text())
    records = data if isinstance(data, list) else next(
        (v for v in data.values() if isinstance(v, list)), []
    )
    label = f"link:{Path(path).name}"
    rows = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        name = rec.get(on)
        if name is None or str(name).strip() == "":
            continue
        row = {"unique_id": f"ext:{i}", "source_dataset": label, on: str(name)}
        for f in compare:
            v = rec.get(f)
            row[f] = None if v is None or str(v).strip() == "" else str(v)
        rows.append(row)
    return rows


def reconcile(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    on: str,
    compare: list[str] | None = None,
    etype: str = "org",
    threshold: float = DEFAULT_THRESHOLD,
    link_path: str | None = None,
    train: bool = False,
    min_link_prob: float = 0.5,
) -> dict:
    try:
        import logging

        import pandas as pd
        from splink import DuckDBAPI, Linker, SettingsCreator, block_on

        logging.getLogger("splink").setLevel(logging.ERROR)  # keep stdout clean for JSON
    except ImportError as e:  # pragma: no cover
        raise NotInstalled("reconcile needs Splink; install `uv sync --extra splink`") from e

    compare = [c for c in (compare or []) if c != on]
    rows = _corpus_rows(conn, on, compare)
    link_type = "dedupe_only"
    if link_path:
        rows += _link_rows(link_path, on, compare)
        link_type = "link_and_dedupe"
    if len(rows) < 2:
        return {"error": f"need >=2 records with field {on!r}; found {len(rows)}"}

    df = pd.DataFrame(rows)
    comparisons = [_name_comparison(on)] + [_weak_comparison(c) for c in compare]
    blocking = [block_on(f"substr({on}, 1, 4)")] + [block_on(c) for c in compare]
    settings_obj = SettingsCreator(
        link_type=link_type,
        probability_two_random_records_match=_LAMBDA,
        blocking_rules_to_generate_predictions=blocking,
        comparisons=comparisons,
    )
    linker = Linker(df, settings_obj, DuckDBAPI())
    if train:
        try:
            linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
        except Exception:
            pass  # priors remain in force if sampling can't estimate

    pred = linker.inference.predict()
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        pred, threshold_match_probability=threshold
    )
    cdf = clusters.as_pandas_dataframe()
    pdf = pred.as_pandas_dataframe()

    run_id = "rec_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    n_clusters = int(cdf["cluster_id"].nunique())
    conn.execute(
        "INSERT OR REPLACE INTO reconcile_runs(run_id, method, on_field, compare_fields, "
        "threshold, link_dataset, settings_json, n_records, n_clusters, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (run_id, "splink-fellegi-sunter", on, ",".join(compare), threshold, link_path,
         json.dumps({"comparisons": comparisons, "lambda": _LAMBDA, "trained_u": train}),
         len(rows), n_clusters, _now()),
    )
    for _, r in cdf.iterrows():
        fields = {c: r[c] for c in compare if c in r and r[c] is not None}
        conn.execute(
            "INSERT OR REPLACE INTO reconcile_records(run_id, record_id, source, cluster_id, name, fields_json) "
            "VALUES(?,?,?,?,?,?)",
            (run_id, str(r["unique_id"]), str(r["source_dataset"]), str(r["cluster_id"]),
             str(r[on]), json.dumps(fields)),
        )
    for _, r in pdf.iterrows():
        if float(r["match_probability"]) >= min_link_prob:
            conn.execute(
                "INSERT INTO reconcile_links(run_id, src_record_id, dst_record_id, match_probability) "
                "VALUES(?,?,?,?)",
                (run_id, str(r["unique_id_l"]), str(r["unique_id_r"]), float(r["match_probability"])),
            )
    meta_set(conn, "last_reconcile_run", run_id)
    meta_set(conn, "reconcile_method", "splink-fellegi-sunter")
    meta_set(conn, "reconcile_threshold", str(threshold))
    conn.commit()

    # Surface the merges (clusters with >1 member), and cross-source links if a dataset was linked.
    sizes = cdf.groupby("cluster_id").size()
    merged = [cid for cid, n in sizes.items() if n > 1]
    summary_clusters = []
    for cid in merged:
        members = cdf[cdf["cluster_id"] == cid]
        summary_clusters.append({
            "cluster_id": str(cid),
            "size": int(len(members)),
            "names": sorted(set(members[on].tolist())),
            "sources": sorted(set(members["source_dataset"].tolist())),
            "record_ids": members["unique_id"].tolist(),
        })
    summary_clusters.sort(key=lambda c: c["size"], reverse=True)
    return {
        "run_id": run_id,
        "method": "splink-fellegi-sunter",
        "threshold": threshold,
        "n_records": len(rows),
        "n_clusters": n_clusters,
        "n_merged_clusters": len(merged),
        "link_dataset": link_path,
        "clusters": summary_clusters,
    }


def show_run(conn: sqlite3.Connection, run_id: str | None = None) -> dict:
    if run_id is None:
        row = conn.execute("SELECT value FROM index_meta WHERE key='last_reconcile_run'").fetchone()
        run_id = row[0] if row else None
    if not run_id:
        return {"error": "no reconcile run found; run `muck reconcile` first"}
    run = conn.execute("SELECT * FROM reconcile_runs WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        return {"error": f"unknown run {run_id!r}"}
    clusters = []
    for c in conn.execute(
        "SELECT cluster_id, COUNT(*) n FROM reconcile_records WHERE run_id=? "
        "GROUP BY cluster_id HAVING n>1 ORDER BY n DESC", (run_id,)
    ):
        members = conn.execute(
            "SELECT record_id, source, name FROM reconcile_records WHERE run_id=? AND cluster_id=?",
            (run_id, c["cluster_id"]),
        ).fetchall()
        clusters.append({
            "cluster_id": c["cluster_id"], "size": c["n"],
            "names": sorted({m["name"] for m in members}),
            "sources": sorted({m["source"] for m in members}),
            "members": [{"record_id": m["record_id"], "source": m["source"], "name": m["name"]} for m in members],
        })
    return {"run_id": run_id, "method": run["method"], "threshold": run["threshold"],
            "n_records": run["n_records"], "n_clusters": run["n_clusters"],
            "link_dataset": run["link_dataset"], "merged_clusters": clusters}
