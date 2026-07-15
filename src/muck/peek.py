"""Peek at raw source files — schema discovery for calibration (pre-config).

Before you can write a good field map or sensible queries, you need to see the actual data
shape. ``peek`` inspects raw files (no index required):

- **A file** → the detected record array, a **field inventory** (names, types, how often
  present, an example), sample records, and a *heuristic* starter field map to adapt.
- **A directory** → files grouped by schema, with one suggested **``[[mapper.sources]]``**
  entry per distinct schema — copy-paste config for a heterogeneous corpus.

Nested objects are expanded one level into dotted names (``member.name``), which is exactly
what the field map can address (``_get`` walks dotted dict paths) — a bare ``member: object``
row hides the fields you actually need.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .config import FieldMap
from .mappers.json_mapper import _detect_records, _load
from .mappers.xml_mapper import xml_file_to_record_dict

# Org hints are checked FIRST: "name" alone is a weak person signal, so `client.name` must
# classify as org, not person (the leaf "name" would otherwise win).
_ORG_HINTS = ("registrant", "client", "org", "company", "firm", "employer", "affiliated")
_PERSON_HINTS = ("lobbyist", "person", "official", "staff", "member", "contact",
                 "payee", "honoree", "name")
# Fields that *are* an entity by name even though they don't contain "name"
# (flat columns like `registrant`, `lobbyists`).
_ENTITY_FIELDS = _ORG_HINTS + ("lobbyist", "official", "member", "sponsor", "vendor",
                               "payee", "honoree")
# A hint in the path is not enough: `registrant.zip` and `member.party` are attributes of an
# entity, not names of one. Suggesting them would index zip codes as organizations.
_NOT_A_NAME = ("country", "city", "state", "zip", "postal", "address", "phone", "telephone",
               "fax", "date", "url", "uuid", "_id", "id_", "description", "email", "code",
               "type", "status", "effective", "ppb", "updated", "comment", "amount", "income",
               "expense", "period", "year", "chamber", "party", "district", "prefix", "suffix",
               "middle", "display", "select", "entity")
_LONG_TEXT = 60  # a string longer than this is prose, not an entity name


def _classify_entity(name: str, types: list[str], ex) -> str | None:
    """-> 'org' | 'person' | None. Only *names* of entities, never their attributes."""
    if "str" not in types and "list" not in types:
        return None
    if isinstance(ex, str) and len(ex) > _LONG_TEXT:
        return None  # prose, not a name
    low = name.lower()
    if low == "id" or any(tok in low for tok in _NOT_A_NAME):
        return None
    leaf = low.rsplit(".", 1)[-1]
    is_name = "name" in leaf or (
        "." not in low and any(h in low for h in _ENTITY_FIELDS)
    )
    if not is_name:
        return None
    # Org hints win on the full path: `registrant.name` is an org, not a person.
    if any(h in low for h in _ORG_HINTS):
        return "org"
    if any(h in low for h in _PERSON_HINTS):
        return "person"
    return None

SCHEMA_EXTS = (".json", ".jsonl", ".xml")


def _typename(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return "null"


def _note_field(name, v, types, present, example) -> None:
    types[name].add(_typename(v))
    present[name] += 1
    if name not in example and isinstance(v, (str, int, float, bool)):
        example[name] = v


def _scan_fields(recs: list, sample: int) -> list[dict]:
    types: dict[str, set] = defaultdict(set)
    present: Counter = Counter()
    example: dict[str, object] = {}
    seen = 0
    for r in recs[:sample]:
        if not isinstance(r, dict):
            continue
        seen += 1
        for k, v in r.items():
            _note_field(k, v, types, present, example)
            if isinstance(v, dict):  # one level of nesting — what a dotted field map can reach
                for k2, v2 in v.items():
                    _note_field(f"{k}.{k2}", v2, types, present, example)
    return [
        {"name": k, "types": sorted(types[k]),
         "present_pct": round(100 * present[k] / seen) if seen else 0,
         "example": example.get(k)}
        for k in types
    ]


def _suggest_map(fields: list[dict]) -> dict:
    record_id = None
    text_fields, structured_fields, entity_fields = [], [], []
    for f in fields:
        name, types, ex = f["name"], f["types"], f.get("example")
        low = name.lower()
        if record_id is None and ("id" in low or "uuid" in low) and "str" in types:
            record_id = name
        is_prose = "str" in types and isinstance(ex, str) and len(ex) > _LONG_TEXT
        if "str" in types:
            (text_fields if is_prose else structured_fields).append(name)
        elif "list" not in types and "object" not in types:
            structured_fields.append(name)
        etype = _classify_entity(name, types, ex)
        if etype:
            entity_fields.append(f"{name}:{etype}")
    return {
        "record_id": record_id,
        "text_fields": text_fields,
        "structured_fields": structured_fields,
        "entity_fields": entity_fields,
        "_note": "heuristic starting point — adjust to the corpus manual",
    }


def _peek_json(p: Path, sample: int, n_samples: int) -> dict:
    kind, data = _load(str(p))
    if kind == "jsonl":
        recs = data
        record_path = "(jsonl lines)"
    else:
        pairs = _detect_records(data, FieldMap())
        recs = [r for _, r in pairs]
        first = pairs[0][0] if pairs else ""
        record_path = first.rsplit("/", 1)[0] if first.count("/") > 1 else (first or "(root)")
    fields = _scan_fields(recs, sample)
    return {
        "path": str(p), "format": kind, "record_path": record_path, "n_records": len(recs),
        "fields": fields, "samples": recs[:n_samples], "suggested_field_map": _suggest_map(fields),
    }


def _peek_xml(p: Path, sample: int, n_samples: int) -> dict:
    rec = xml_file_to_record_dict(str(p))
    fields = _scan_fields([rec], sample)
    return {
        "path": str(p), "format": "xml",
        "record_path": None,  # one record per file; set record_path for bulk-export XML
        "n_records": 1, "fields": fields, "samples": [rec][:n_samples],
        "suggested_field_map": _suggest_map(fields),
    }


def _peek_one(p: Path, sample: int, n_samples: int) -> dict:
    ext = p.suffix.lower()
    if ext in (".json", ".jsonl"):
        return _peek_json(p, sample, n_samples)
    if ext == ".xml":
        return _peek_xml(p, sample, n_samples)
    if ext in (".txt", ".md", ".markdown", ".text"):
        return {"path": str(p), "format": ext.lstrip("."),
                "preview": p.read_text(errors="replace")[:1500]}
    return {"path": str(p), "format": ext.lstrip("."),
            "note": "binary/unsupported for peek; ingest + parse it, then `muck read <doc_id>`"}


def _glob_for(rel: Path) -> str:
    """Directory -> a [[mapper.sources]] glob; pure-digit segments (years) generalize to `*`."""
    parts = ["*" if seg.isdigit() else seg for seg in rel.parts if seg not in (".", "")]
    return "*/" + "/".join(parts) + "/*" if parts else "*"


def _peek_dir(d: Path, sample: int, max_sources: int = 8, max_files: int = 20000) -> dict:
    files: list[Path] = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in SCHEMA_EXTS and not p.name.startswith("."):
            files.append(p)
            if len(files) >= max_files:
                break
    if not files:
        return {"error": f"no {'/'.join(SCHEMA_EXTS)} files under {str(d)!r}"}

    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        by_parent[f.parent].append(f)

    groups: dict[tuple, dict] = {}  # schema signature -> group
    for parent, fs in sorted(by_parent.items()):
        try:
            info = _peek_one(fs[0], sample, 0)
        except Exception:
            continue  # a malformed file shouldn't sink the whole directory peek
        if not info.get("fields"):
            continue
        sig = tuple(sorted(f["name"] for f in info["fields"]))
        glob = _glob_for(parent.relative_to(d))
        g = groups.get(sig)
        if g is None:
            groups[sig] = {"match": glob, "n_files": len(fs), "rep": fs[0], "info": info}
        else:
            g["n_files"] += len(fs)
            # `*` crosses `/` in fnmatch, so the shallower glob already subsumes deeper ones.
            if glob.count("/") < g["match"].count("/"):
                g["match"] = glob

    sources = []
    for g in sorted(groups.values(), key=lambda g: g["n_files"], reverse=True)[:max_sources]:
        info = g["info"]
        sm = {k: v for k, v in info["suggested_field_map"].items() if k != "_note"}
        sources.append({
            "match": g["match"], "n_files": g["n_files"], "example_file": str(g["rep"]),
            "format": info.get("format"), "record_path": info.get("record_path"),
            "n_records": info.get("n_records"), "fields": info["fields"],
            "suggested_source_map": {"match": g["match"], **sm},
        })
    return {
        "path": str(d), "format": "directory", "n_files": len(files),
        "schemas": len(groups), "sources": sources,
        "_note": "copy each suggested_source_map into a [[mapper.sources]] block; "
                 "peek example_file to see sample records",
    }


def peek_file(path: str, sample: int = 50, n_samples: int = 2) -> dict:
    p = Path(path)
    if p.is_dir():
        return _peek_dir(p, sample)
    if not p.is_file():
        return {"error": f"not a file: {path!r}"}
    return _peek_one(p, sample, n_samples)
