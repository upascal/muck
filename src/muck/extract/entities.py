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
from fnmatch import fnmatch
from itertools import combinations

from ..cite import make_token

# Only unambiguous legal-form suffixes — never descriptive words like "strategies" or
# "group", which are meaningful parts of a firm's name and would cause false merges.
ORG_SUFFIXES = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "lp", "llp", "pllc", "pc", "ltd", "limited", "plc", "the",
}

# Honorifics/titles and generational suffixes to strip from person names so the same
# individual folds to one key across sources — e.g. an LD-203 honoree "The Honorable
# U.S. Senator Angus S. King, Jr." and a press "Angus King" both normalize to "angus king".
PERSON_DROP = {
    "the", "hon", "honorable", "sen", "senator", "rep", "representative", "congressman",
    "congresswoman", "us", "dr", "mr", "mrs", "ms", "jr", "sr", "ii", "iii", "iv",
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
    elif etype == "person":
        # drop titles/suffixes and lone middle initials so titled/formal variants of the
        # same person merge across sources (press member <-> LD-203 honoree)
        kept = [t for t in toks if t not in PERSON_DROP and len(t) > 1]
        toks = kept or toks
    return " ".join(toks)


def _entity_id(etype: str, key: str) -> str:
    return "e_" + hashlib.sha1(f"{etype}|{key}".encode()).hexdigest()[:12]


# --- typed relations (Tier 2) ------------------------------------------------

_REL_SKIP = {"", "n/a", "none", "null", "-", "various"}


def _parse_relation(spec: str) -> dict | None:
    """Parse ``"src_field:src_type -> dst_field:dst_type : predicate"`` into its parts.

    ``src_field``/``dst_field`` name structured_json keys (aliases); types default to ``org``.
    Returns None for a malformed spec (so a bad line is skipped, not fatal).
    """
    lhs, arrow, rhs = spec.partition(" -> ")
    if not arrow:
        return None
    dst_part, sep, predicate = rhs.rpartition(" : ")
    if not sep:
        return None
    sf, _, st = lhs.strip().partition(":")
    df, _, dt = dst_part.strip().partition(":")
    if not (sf.strip() and df.strip() and predicate.strip()):
        return None
    return {
        "predicate": predicate.strip(),
        "src_field": sf.strip(), "src_type": st.strip() or "org",
        "dst_field": df.strip(), "dst_type": dt.strip() or "org",
    }


def _as_list(v) -> list:
    """Scalar leaves of a structured_json value, dropping placeholder non-entities (N/A, None)."""
    if v is None:
        return []
    items = v if isinstance(v, list) else [v]
    return [x for x in items
            if isinstance(x, (str, int, float)) and str(x).strip().lower() not in _REL_SKIP]


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


_RELATIONS_DDL = (
    "CREATE TABLE IF NOT EXISTS entity_relations ("
    " src_entity_id TEXT NOT NULL, dst_entity_id TEXT NOT NULL, predicate TEXT NOT NULL,"
    " weight INTEGER NOT NULL DEFAULT 0, doc_id TEXT,"
    " PRIMARY KEY (src_entity_id, dst_entity_id, predicate));"
    "CREATE INDEX IF NOT EXISTS ix_entity_relations_src ON entity_relations(src_entity_id);"
    "CREATE INDEX IF NOT EXISTS ix_entity_relations_dst ON entity_relations(dst_entity_id);"
)


def build_relations(conn: sqlite3.Connection, settings) -> dict:
    """Typed relations from structured fields → directed, citable ``entity_relations`` rows.

    For each source that declares ``relations`` (``"src_field:type -> dst_field:type : predicate"``),
    read each document's stored ``structured_json`` (no re-parse), resolve both field values to
    entity_ids exactly as the resolver does (``normalize_key`` → ``_entity_id``), and record one
    directed edge per ``(predicate, src, dst)`` with a count and a representative doc. Idempotent;
    runs standalone on a built index and is also called at the end of ``run_index``.
    """
    conn.executescript(_RELATIONS_DDL)  # tolerate an index built before this table existed
    conn.execute("DELETE FROM entity_relations")

    ecfg = getattr(settings, "entities", None)
    extra = tuple(getattr(ecfg, "extra_org_suffixes", ()) or ())

    def _parsed(fm):
        return [r for r in (_parse_relation(s) for s in (getattr(fm, "relations", []) or [])) if r]

    specs: list[tuple[str, list[dict]]] = []
    for sm in getattr(settings.mapper, "sources", []) or []:
        parsed = _parsed(sm)
        if parsed:
            pat = sm.match if sm.match.startswith(("/", "*")) else "*/" + sm.match
            specs.append((pat, parsed))
    gparsed = _parsed(getattr(settings.mapper, "fields", None))
    if not specs and not gparsed:
        return {"predicates": 0, "relations": 0}

    known = {r[0] for r in conn.execute("SELECT entity_id FROM entities")}
    acc: dict[tuple[str, str, str], list] = defaultdict(lambda: [0, None])
    for doc in conn.execute(
        "SELECT doc_id, source_path, structured_json FROM documents "
        "WHERE structured_json IS NOT NULL AND structured_json != ''"
    ):
        rels = gparsed
        for pat, parsed in specs:
            if fnmatch(doc["source_path"], pat):
                rels = parsed
                break
        if not rels:
            continue
        try:
            sj = json.loads(doc["structured_json"])
        except (ValueError, TypeError):
            continue
        for rel in rels:
            svals = _as_list(sj.get(rel["src_field"]))
            dvals = _as_list(sj.get(rel["dst_field"]))
            if not svals or not dvals:
                continue
            src_eids = {_entity_id(rel["src_type"], k)
                        for k in (normalize_key(str(v), rel["src_type"], extra) for v in svals) if k}
            dst_eids = {_entity_id(rel["dst_type"], k)
                        for k in (normalize_key(str(v), rel["dst_type"], extra) for v in dvals) if k}
            for se in src_eids & known:
                for de in dst_eids & known:
                    if de == se:
                        continue
                    slot = acc[(rel["predicate"], se, de)]
                    slot[0] += 1
                    if slot[1] is None:
                        slot[1] = doc["doc_id"]

    conn.executemany(
        "INSERT INTO entity_relations(predicate, src_entity_id, dst_entity_id, weight, doc_id) "
        "VALUES(?,?,?,?,?)",
        [(p, s, d, w, doc) for (p, s, d), (w, doc) in acc.items()],
    )
    conn.commit()
    by_pred: Counter = Counter(p for (p, s, d) in acc)
    return {"predicates": len(by_pred), "relations": len(acc), "by_predicate": dict(by_pred)}


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


def list_relations(conn, predicate=None, name=None, limit=25) -> list[dict]:
    """Strongest typed relations by weight (the directed graph view), optionally filtered."""
    where, params = [], []
    if predicate:
        where.append("r.predicate = ?")
        params.append(predicate)
    if name:
        where.append("(se.canonical_name LIKE ? OR de.canonical_name LIKE ?)")
        params += [f"%{name}%", f"%{name}%"]
    sql = (
        "SELECT r.predicate, r.weight, r.doc_id, "
        "se.canonical_name AS src, se.entity_type AS src_type, "
        "de.canonical_name AS dst, de.entity_type AS dst_type "
        "FROM entity_relations r "
        "JOIN entities se ON se.entity_id=r.src_entity_id "
        "JOIN entities de ON de.entity_id=r.dst_entity_id "
    )
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += "ORDER BY r.weight DESC LIMIT ?"
    try:
        return [dict(r) for r in conn.execute(sql, [*params, limit])]
    except sqlite3.OperationalError:
        return []  # entity_relations absent — run `muck relations --rebuild` or rebuild the index


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
    # typed relations (directed, citable) — grouped by predicate, in both directions
    relations: dict[str, dict] = {"outgoing": {}, "incoming": {}}

    def _rel_row(counter_eid, pred, weight, doc_id, direction):
        ne = conn.execute(
            "SELECT canonical_name, entity_type FROM entities WHERE entity_id=?", (counter_eid,)
        ).fetchone()
        if not ne:
            return
        token = None
        if doc_id:
            m = conn.execute(
                "SELECT char_start, char_end, raw_text FROM mentions WHERE entity_id=? AND doc_id=? LIMIT 1",
                (counter_eid, doc_id),
            ).fetchone()
            if m:
                token = make_token(doc_id, m["char_start"], m["char_end"], m["raw_text"])
        relations[direction].setdefault(pred, []).append({
            "entity_id": counter_eid, "canonical_name": ne["canonical_name"],
            "entity_type": ne["entity_type"], "weight": weight, "doc_id": doc_id, "token": token,
        })

    try:
        for r in conn.execute(
            "SELECT predicate, dst_entity_id, weight, doc_id FROM entity_relations "
            "WHERE src_entity_id=? ORDER BY weight DESC LIMIT 40", (entity_id,)):
            _rel_row(r["dst_entity_id"], r["predicate"], r["weight"], r["doc_id"], "outgoing")
        for r in conn.execute(
            "SELECT predicate, src_entity_id, weight, doc_id FROM entity_relations "
            "WHERE dst_entity_id=? ORDER BY weight DESC LIMIT 40", (entity_id,)):
            _rel_row(r["src_entity_id"], r["predicate"], r["weight"], r["doc_id"], "incoming")
    except sqlite3.OperationalError:
        pass  # index predates entity_relations; run `muck relations` (or rebuild) to populate

    return {
        "entity_id": entity_id,
        "canonical_name": e["canonical_name"],
        "entity_type": e["entity_type"],
        "mention_count": e["mention_count"],
        "doc_count": e["doc_count"],
        "aliases": aliases,
        "mentions": mentions,
        "co_occurring": neighbors,
        "relations": relations,
    }


def _source_label(source_path: str) -> str:
    # Short, corpus-agnostic label for *where* a doc came from (last path segments distinguish
    # e.g. .../congress_press/… from .../filings/… without hardcoding source names).
    parts = [p for p in str(source_path).replace("\\", "/").split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else str(source_path))


def dossier(conn, settings, name: str, examples: int = 4) -> dict:
    """Cross-source drill-down for one entity — the "hone in on a person" view.

    Assembles everything the index knows about a name into one pre-joined dossier: WHERE it's
    named (grouped by source, so press vs filings vs contributions separate out), its resolved
    identity + co-occurrence network, and example citations. Runs at query time over the flattened
    text, so it catches *unresolved* name variants too (e.g. an LD-203 honoree "Sen. Dan Sullivan"
    that never became an entity). This is the context-assembly primitive that lets an agent spot a
    say-vs-pay / follow-the-money juxtaposition without hand-joining sources.
    """
    # 1. resolved identity + network (reuses the deterministic resolver as-is)
    resolved = []
    for etype in ("person", "org"):
        for e in list_entities(conn, etype=etype, name=name, limit=2):
            det = entity_detail(conn, e["entity_id"])
            resolved.append({
                "entity_id": e["entity_id"], "canonical_name": e["canonical_name"],
                "type": etype, "aliases": det["aliases"],
                "mention_count": e["mention_count"], "doc_count": e["doc_count"],
                "co_occurring": det["co_occurring"],
                "relations": det.get("relations", {}),
            })

    # 2. cross-source appearances — FTS phrase over the (flattened) document text, grouped by source
    fts = '"' + re.sub(r"[^\w\s]", " ", name).strip() + '"'
    by_source, total = [], 0
    try:
        for r in conn.execute(
            "SELECT d.source_path sp, COUNT(DISTINCT d.doc_id) n "
            "FROM chunks_fts f JOIN chunks c ON c.chunk_id=f.chunk_id "
            "JOIN documents d ON d.doc_id=c.doc_id WHERE chunks_fts MATCH ? "
            "GROUP BY d.source_path ORDER BY n DESC",
            (fts,),
        ):
            by_source.append({"source": _source_label(r["sp"]), "documents": r["n"]})
            total += r["n"]
    except sqlite3.OperationalError:
        pass  # FTS unavailable or empty phrase

    # 3. example citations (reuse keyword search for snippet + verifiable token)
    from ..search import query as q

    hits = q.search(conn, settings, name, mode="keyword", k=examples)
    ex = [{"doc_id": h.doc_id, "source": _source_label(h.source_path),
           "snippet": h.snippet, "token": h.citation.token} for h in hits]

    return {
        "query": name,
        "documents_naming_it": total,
        "appearances_by_source": by_source,
        "resolved": resolved,
        "examples": ex,
        "note": (
            "Drill-down dossier. `appearances_by_source` shows WHERE this entity is named — a name "
            "present in both a press source and a contributions/filings source is a say-vs-pay / "
            "follow-the-money juxtaposition to chase. `resolved.co_occurring` is its co-occurrence "
            "network; `resolved.relations` are TYPED, directed, cited edges (outgoing/incoming by "
            "predicate, e.g. donated_to / lobbied_for) — each carries a doc_id + token to verify. "
            "Read a source in full with `muck read <doc_id>` and confirm any quote with `muck verify`."
        ),
    }
