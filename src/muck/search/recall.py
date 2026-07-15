"""Recall — search the agent's own accumulated memory (Tier 2).

`muck search` queries the *corpus*; `recall` queries what the agent has *recorded* — its
observations (notes/leads/threads) and findings — ranked by relevance to a query. Memory is
small (hundreds–thousands of items), so it reuses the configured embedder on the fly (no
separate index, no sync), exactly like `search_in_entity`; it falls back to keyword overlap
when embeddings are off or unavailable.
"""

from __future__ import annotations

import json
import re

from ..config import Settings
from ..embedders import build_embedder

_WORD = re.compile(r"[\w']+")


def _candidates(conn, kind, status, entity_id, entity_name) -> list[dict]:
    items: list[dict] = []
    # observations (skip when a finding-only kind filter is set)
    if kind is None or kind != "finding":
        where, params = [], []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT obs_id, kind, status, text, entity_ids, tokens FROM observations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        for r in conn.execute(sql, params):
            eids = json.loads(r["entity_ids"]) if r["entity_ids"] else []
            if entity_id and not (
                entity_id in eids or (entity_name and entity_name.lower() in r["text"].lower())
            ):
                continue
            items.append({
                "type": "observation", "id": r["obs_id"], "kind": r["kind"], "status": r["status"],
                "text": r["text"], "tokens": json.loads(r["tokens"]) if r["tokens"] else [],
                "entity_ids": eids,
            })
    # findings (kind "finding")
    if kind is None or kind == "finding":
        where, params = [], []
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT finding_id, status, claim, quote, citation_token FROM findings"
        if where:
            sql += " WHERE " + " AND ".join(where)
        for r in conn.execute(sql, params):
            text = f'{r["claim"]} — "{r["quote"]}"'
            if entity_id and entity_name and entity_name.lower() not in text.lower():
                continue
            items.append({
                "type": "finding", "id": r["finding_id"], "kind": "finding", "status": r["status"],
                "text": text, "tokens": [r["citation_token"]], "entity_ids": [],
            })
    return items


def recall(conn, settings: Settings, query: str, kind=None, status=None,
           entity=None, mode: str = "auto", k: int = 8) -> list[dict]:
    entity_id = entity_name = None
    if entity:
        row = conn.execute(
            "SELECT entity_id, canonical_name FROM entities WHERE entity_id=? OR canonical_name LIKE ?",
            (entity, f"%{entity}%"),
        ).fetchone()
        entity_id, entity_name = (row["entity_id"], row["canonical_name"]) if row else (entity, entity)

    items = _candidates(conn, kind, status, entity_id, entity_name)
    if not items or not query:
        return items[:k]

    scores = None
    if mode in ("auto", "semantic"):
        emb = build_embedder(settings)
        if emb is not None:
            try:
                import numpy as np

                mat = np.asarray(emb.embed([it["text"] for it in items]), dtype="float32")
                qvec = np.asarray(emb.embed([query])[0], dtype="float32")
                scores = (mat @ qvec).tolist()  # L2-normalized → dot = cosine
            except Exception:
                scores = None  # fall back to keyword if the model can't run

    ranked: list[tuple[dict, float]]
    if scores is not None:
        ranked = sorted(zip(items, scores), key=lambda t: t[1], reverse=True)
    else:  # keyword: query-term overlap, drop zero-overlap
        terms = {t.lower() for t in _WORD.findall(query)}
        scored = [(it, float(len({t.lower() for t in _WORD.findall(it["text"])} & terms))) for it in items]
        ranked = sorted((p for p in scored if p[1] > 0), key=lambda t: t[1], reverse=True)

    out = []
    for it, s in ranked[:k]:
        d = dict(it)
        d["score"] = round(float(s), 4)
        out.append(d)
    return out
