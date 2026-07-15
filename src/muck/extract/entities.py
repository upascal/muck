"""Entity extraction + resolution (Phase 1).

Mentions come from (a) a gazetteer built from the clean structured ``entity_fields`` of
JSON records and (b) bill-number regex over free text — both scanned against chunk text
so every mention carries exact char offsets (a verifiable citation). Resolution groups
surface forms by a deterministic normalized key (corporate-suffix stripping merges
"Acme Strategies LLC" with "Acme Strategies, L.L.C."), the reusable "entity resolver".
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations

from ..cite import make_token

# Only unambiguous legal-form suffixes — never descriptive words like "strategies" or
# "group", which are meaningful parts of a firm's name and would cause false merges.
ORG_SUFFIXES = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "lp", "llp", "pllc", "pc", "ltd", "limited", "plc", "the",
}

BILL_RE = re.compile(
    r"\b(?:H\.?\s?R\.?|S\.?J\.?\s?Res\.?|H\.?J\.?\s?Res\.?|S\.?)\s?\d{1,6}\b",
    re.IGNORECASE,
)


def normalize_key(name: str, etype: str, extra_suffixes: tuple[str, ...] = ()) -> str:
    if etype == "bill":
        return re.sub(r"\s+", "", name.lower().replace(".", ""))
    s = name.lower().replace(".", "")
    s = re.sub(r"[^\w\s&]", " ", s)
    toks = [t for t in s.split() if t]
    if etype == "org":
        suffixes = ORG_SUFFIXES.union(extra_suffixes) if extra_suffixes else ORG_SUFFIXES
        kept = [t for t in toks if t not in suffixes]
        toks = kept or toks  # don't normalize a name entirely away
    return " ".join(toks)


def _entity_id(etype: str, key: str) -> str:
    return "e_" + hashlib.sha1(f"{etype}|{key}".encode()).hexdigest()[:12]


_TOKEN_RE = re.compile(r"\w+")


class NameMatcher:
    """Exact, case-insensitive multi-name matcher that scales to 100K+ names.

    A single alternation regex over tens of thousands of names is quadratic-ish in
    practice (backtracking); instead we index names by their first word and only try
    candidates where that anchor token occurs. Word-boundary semantics match the old
    regex: no \\w immediately before or after the span. Longest match wins per anchor;
    matches don't overlap.
    """

    def __init__(self, gaz: dict[str, str], originals: dict[str, str] | None = None, *,
                 min_len: int = 3, single_word: str = "exact-case") -> None:
        # Single-word org names are often ordinary English words ("Clear", "Advance",
        # "Scale") — matched case-insensitively they hit prose everywhere. Configurable
        # via [entities]: single_word_match = exact-case (default: must match registered
        # casing) | any-case | off; names under min_name_len never free-text match.
        # Candidates are (name, anchor_offset, required_casing): anchor_offset supports
        # names whose first word isn't at position 0 ("&pizza").
        self.by_anchor: dict[str, list[tuple[str, int, str | None]]] = defaultdict(list)
        originals = originals or {}
        for name in gaz:
            if len(name) < max(1, min_len):
                continue
            m = _TOKEN_RE.search(name)
            if not m:
                continue
            single = " " not in name.strip()
            if single and single_word == "off":
                continue
            exact = originals.get(name) if (single and single_word == "exact-case") else None
            self.by_anchor[m.group(0)].append((name, m.start(), exact))
        for cands in self.by_anchor.values():
            cands.sort(key=lambda t: len(t[0]), reverse=True)  # longest-first

    def finditer(self, text: str):
        low = text.lower()
        last_end = 0
        for tok in _TOKEN_RE.finditer(low):
            anchor_pos = tok.start()
            if anchor_pos < last_end:
                continue
            cands = self.by_anchor.get(tok.group(0))
            if not cands:
                continue
            for name, offset, exact in cands:
                start = anchor_pos - offset
                if start < last_end or start < 0:
                    continue
                end = start + len(name)
                if not low.startswith(name, start):
                    continue
                if start > 0 and (low[start - 1].isalnum() or low[start - 1] == "_"):
                    continue  # span begins mid-word
                if end < len(low) and (low[end].isalnum() or low[end] == "_"):
                    continue
                if exact is not None and text[start:end] != exact:
                    continue  # single-word name: require registered casing
                yield start, end
                last_end = end
                break


def _gazetteer(conn, ecfg=None) -> tuple[dict[str, str], "NameMatcher | None"]:
    gaz: dict[str, str] = {}
    originals: dict[str, str] = {}  # lowercase -> as-registered casing
    for row in conn.execute("SELECT structured_json FROM documents WHERE structured_json IS NOT NULL"):
        ents = json.loads(row[0]).get("__entities__") or []
        for e in ents:
            name = str(e.get("name", "")).strip()
            if name:
                gaz.setdefault(name.lower(), e.get("type", "org"))
                originals.setdefault(name.lower(), name)
    if not gaz:
        return gaz, None
    matcher = NameMatcher(
        gaz, originals,
        min_len=getattr(ecfg, "min_name_len", 3),
        single_word=getattr(ecfg, "single_word_match", "exact-case"),
    )
    return gaz, matcher


def build_entities(conn: sqlite3.Connection, settings) -> dict:
    """Full rebuild of the entity index from all indexed chunks (idempotent)."""
    for t in ("entities", "entity_aliases", "mentions", "entity_edges"):
        conn.execute(f"DELETE FROM {t}")

    ecfg = getattr(settings, "entities", None)
    extra_suffixes = tuple(getattr(ecfg, "extra_org_suffixes", ()) or ())
    gaz, gaz_re = _gazetteer(conn, ecfg)

    # (type, key) -> list of (doc_id, chunk_id, raw, start, end, locator); + form counts
    groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    forms: dict[tuple[str, str], Counter] = defaultdict(Counter)
    seen: set[tuple[str, int, int]] = set()  # dedupe overlap-chunk double counts
    n_mentions = 0

    for ch in conn.execute("SELECT chunk_id, doc_id, text, char_start, locator FROM chunks"):
        text, base = ch["text"], ch["char_start"]
        spans: list[tuple[str, str, int, int]] = []  # (raw, type, start, end)
        if gaz_re:
            for s, e in gaz_re.finditer(text):
                raw = text[s:e]
                spans.append((raw, gaz.get(raw.lower(), "org"), s, e))
        for m in BILL_RE.finditer(text):
            spans.append((m.group(0), "bill", m.start(), m.end()))
        for raw, etype, s, e in spans:
            key = normalize_key(raw, etype, extra_suffixes)
            if not key:
                continue
            abs_start, abs_end = base + s, base + e
            if (ch["doc_id"], abs_start, abs_end) in seen:
                continue
            seen.add((ch["doc_id"], abs_start, abs_end))
            gk = (etype, key)
            groups[gk].append((ch["doc_id"], ch["chunk_id"], raw, abs_start, abs_end, ch["locator"]))
            forms[gk][raw] += 1
            n_mentions += 1

    doc_entities: dict[str, set[str]] = defaultdict(set)
    for (etype, key), mlist in groups.items():
        eid = _entity_id(etype, key)
        top = max(forms[(etype, key)].values())
        canonical = min((f for f, c in forms[(etype, key)].items() if c == top), key=len)
        docs = set()
        for doc_id, chunk_id, raw, s, e, loc in mlist:
            conn.execute(
                "INSERT INTO mentions(entity_id, chunk_id, doc_id, raw_text, char_start, char_end, locator) "
                "VALUES(?,?,?,?,?,?,?)",
                (eid, chunk_id, doc_id, raw, s, e, loc),
            )
            docs.add(doc_id)
            doc_entities[doc_id].add(eid)
        conn.execute(
            "INSERT INTO entities(entity_id, canonical_name, entity_type, norm_key, mention_count, doc_count) "
            "VALUES(?,?,?,?,?,?)",
            (eid, canonical, etype, key, len(mlist), len(docs)),
        )
        for alias in {m[2] for m in mlist}:
            conn.execute(
                "INSERT OR IGNORE INTO entity_aliases(entity_id, alias, source) VALUES(?,?, 'surface')",
                (eid, alias),
            )

    edge_w: Counter = Counter()
    for eset in doc_entities.values():
        for a, b in combinations(sorted(eset), 2):
            edge_w[(a, b)] += 1
    for (a, b), w in edge_w.items():
        conn.execute(
            "INSERT INTO entity_edges(src_entity_id, dst_entity_id, weight, edge_type) VALUES(?,?,?, 'co_doc')",
            (a, b, w),
        )

    conn.commit()
    return {"entities": len(groups), "mentions": n_mentions, "edges": len(edge_w)}


# --- query -------------------------------------------------------------------

def list_entities(conn, etype=None, name=None, min_count=1, limit=50) -> list[dict]:
    where, params = ["mention_count >= ?"], [min_count]
    if etype:
        where.append("entity_type = ?")
        params.append(etype)
    if name:
        where.append("(canonical_name LIKE ? OR norm_key LIKE ?)")
        params += [f"%{name}%", f"%{name.lower()}%"]
    rows = conn.execute(
        f"SELECT entity_id, canonical_name, entity_type, mention_count, doc_count, norm_key "
        f"FROM entities WHERE {' AND '.join(where)} ORDER BY mention_count DESC LIMIT ?",
        [*params, limit],
    ).fetchall()
    out = []
    for r in rows:
        n_aliases = conn.execute(
            "SELECT COUNT(*) FROM entity_aliases WHERE entity_id=?", (r["entity_id"],)
        ).fetchone()[0]
        d = dict(r)
        d["n_aliases"] = n_aliases
        out.append(d)
    return out


def entity_detail(conn, entity_id, max_mentions=10) -> dict:
    e = conn.execute("SELECT * FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
    if e is None:
        return {"error": f"unknown entity_id {entity_id!r}"}
    aliases = [r["alias"] for r in conn.execute(
        "SELECT alias FROM entity_aliases WHERE entity_id=? ORDER BY alias", (entity_id,))]
    mentions = []
    for m in conn.execute(
        "SELECT m.doc_id, m.raw_text, m.char_start, m.char_end, m.locator, d.title, d.source_path "
        "FROM mentions m JOIN documents d ON d.doc_id=m.doc_id WHERE m.entity_id=? LIMIT ?",
        (entity_id, max_mentions),
    ):
        mentions.append({
            "doc_id": m["doc_id"],
            "doc_title": m["title"],
            "source_path": m["source_path"],
            "locator": m["locator"],
            "raw_text": m["raw_text"],
            "token": make_token(m["doc_id"], m["char_start"], m["char_end"], m["raw_text"]),
        })
    neighbors = []
    for nb in conn.execute(
        "SELECT CASE WHEN src_entity_id=?1 THEN dst_entity_id ELSE src_entity_id END AS nb, weight "
        "FROM entity_edges WHERE src_entity_id=?1 OR dst_entity_id=?1 ORDER BY weight DESC LIMIT 10",
        (entity_id,),
    ):
        ne = conn.execute(
            "SELECT canonical_name, entity_type FROM entities WHERE entity_id=?", (nb["nb"],)
        ).fetchone()
        if ne:
            neighbors.append({
                "entity_id": nb["nb"], "canonical_name": ne["canonical_name"],
                "entity_type": ne["entity_type"], "co_documents": nb["weight"],
            })
    return {
        "entity_id": entity_id,
        "canonical_name": e["canonical_name"],
        "entity_type": e["entity_type"],
        "mention_count": e["mention_count"],
        "doc_count": e["doc_count"],
        "aliases": aliases,
        "mentions": mentions,
        "co_occurring": neighbors,
    }
