"""Batch full-document read — pull several whole records side by side.

Retrieval returns chunks (fragments); once the agent has narrowed to relevant doc_ids
(via search / entities / aggregate), this loads the **full** records together so it can
compare and spot cross-document patterns, calibrate queries, or read a finding's source in
full. Bounded by a token budget so it can't pull a huge set into context: whole records are
included in order until the budget is hit; the rest are reported as omitted (never silently
truncated). Each record carries a whole-document citation token.
"""

from __future__ import annotations

import json
import sqlite3

from .cite import make_token
from .config import Settings


def read_documents(conn: sqlite3.Connection, settings: Settings, doc_ids: list[str],
                   token_budget: int = 16000, include_raw: bool = True) -> dict:
    factor = settings.chunk.token_estimate_factor
    documents, omitted = [], []
    used = 0
    for did in doc_ids:
        row = conn.execute(
            "SELECT doc_id, source_path, locator, doc_type, title, n_pages, text, "
            "structured_json, raw_json FROM documents WHERE doc_id=?",
            (did,),
        ).fetchone()
        if row is None:
            omitted.append({"doc_id": did, "reason": "unknown"})
            continue
        text = row["text"]
        est = max(1, int(len(text.split()) * factor))
        if documents and used + est > token_budget:  # always include at least the first
            omitted.append({"doc_id": did, "reason": "token_budget"})
            continue
        used += est
        rec = {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "source_path": row["source_path"],
            "locator": row["locator"],
            "doc_type": row["doc_type"],
            "n_pages": row["n_pages"],
            "approx_tokens": est,
            "text": text,
            "citation_token": make_token(row["doc_id"], 0, len(text), text),  # whole-doc citation
        }
        if row["structured_json"]:
            rec["structured"] = json.loads(row["structured_json"])
        if include_raw and row["raw_json"]:
            rec["raw"] = json.loads(row["raw_json"])
        documents.append(rec)
    return {
        "requested": len(doc_ids),
        "returned": len(documents),
        "omitted": omitted,
        "approx_tokens": used,
        "documents": documents,
    }


def sample_documents(conn, settings: Settings, query: str | None = None, doc_type: str | None = None,
                     entity: str | None = None, n: int | None = None, token_budget: int = 8000,
                     order: str = "random", include_raw: bool = False) -> dict:
    """Load a budget-bounded sample of full records to get a feel for the corpus.

    Selection: a ``query`` (full docs behind the top search hits), an ``entity`` (docs that
    mention it), or a plain ``random``/``first`` draw (optionally filtered by ``doc_type``).
    Then reads whole records up to the token budget (via ``read_documents``).
    """
    candidates: list[str] = []
    if query:
        from .search import query as qmod

        filters = {"doc_type": doc_type} if doc_type else None
        seen: set[str] = set()
        for h in qmod.search(conn, settings, query, "auto", (n or 50), filters):
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                candidates.append(h.doc_id)
    elif entity:
        from .search import query as qmod

        eids = qmod._resolve_entity_ids(
            conn, entity_id=entity if entity.startswith("e_") else None,
            name=None if entity.startswith("e_") else entity,
        )
        if eids:
            ph = ",".join("?" * len(eids))
            candidates = [r[0] for r in conn.execute(
                f"SELECT DISTINCT doc_id FROM mentions WHERE entity_id IN ({ph})", eids)]
    else:
        where = " WHERE doc_type=?" if doc_type else ""
        params = [doc_type] if doc_type else []
        ordering = "RANDOM()" if order == "random" else "doc_id"
        rows = conn.execute(
            f"SELECT doc_id FROM documents{where} ORDER BY {ordering} LIMIT ?", [*params, n or 200]
        ).fetchall()
        candidates = [r[0] for r in rows]

    if n:
        candidates = candidates[:n]
    result = read_documents(conn, settings, candidates, token_budget, include_raw)
    result["selection"] = {
        "by": "query" if query else ("entity" if entity else order),
        "query": query, "doc_type": doc_type, "entity": entity, "candidates": len(candidates),
    }
    return result
