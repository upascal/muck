"""Default mapper: a JSON / JSONL file -> many logical documents (Records).

Config-driven with auto-detection (see ``FieldMap``). The text *rendering* is a pure,
deterministic function so that citations can re-derive the exact same text from source
later (``map_one_text``), which is what makes the hallucination guard sound.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import FieldMap
from ..interfaces.mapper import register_mapper
from ..schema import Record


# --- JSON pointer (RFC 6901) -------------------------------------------------

def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _navigate(data: Any, pointer: str) -> Any:
    if pointer == "":
        return data
    cur = data
    for raw in pointer.split("/")[1:]:
        token = _unescape(raw)
        if isinstance(cur, list):
            cur = cur[int(token)]
        else:
            cur = cur[token]
    return cur


# --- loading -----------------------------------------------------------------

def _load(path: str) -> tuple[str, Any]:
    """Return (kind, data). kind is 'jsonl' (data=list of lines) or 'json'."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        lines = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        return "jsonl", lines
    return "json", json.loads(p.read_text())


def _detect_records(data: Any, fields: FieldMap) -> list[tuple[str, Any]]:
    """Return [(json_pointer, record)] for a parsed JSON document."""
    if fields.record_path:
        container = _navigate(data, fields.record_path)
        base = fields.record_path
    else:
        container = data
        base = ""
    if isinstance(container, list):
        return [(f"{base}/{i}", item) for i, item in enumerate(container)]
    if isinstance(container, dict):
        # Auto-detect: the first value that is a non-empty list of dicts is the corpus.
        if not fields.record_path:
            for key, val in container.items():
                if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
                    return [(f"/{_escape(key)}/{i}", item) for i, item in enumerate(val)]
        # Otherwise the dict itself is one record.
        return [(base, container)]
    return [(base, container)]


# --- field access + rendering ------------------------------------------------

def _get(record: Any, key: str) -> Any:
    """Get a (possibly dotted) field from a record dict; None if absent."""
    if not isinstance(record, dict):
        return None
    cur: Any = record
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _parse_field_spec(spec: str) -> tuple[str, str | None]:
    """Split a ``structured_fields`` entry into (path, alias). ``path as alias`` -> alias set.

    A path may contain ``[]`` after a key to traverse that array (e.g.
    ``lobbying_activities[].government_entities[].name``). Array-paths should carry an alias
    (the raw path is an unwieldy structured_json key), but one is not required.
    """
    path, sep, alias = spec.partition(" as ")
    return path.strip(), (alias.strip() or None) if sep else None


def _get_multi(record: Any, path: str) -> list:
    """Traverse a path that may cross arrays (``key[]``), collecting scalar leaves.

    Returns a flat, de-duplicated, order-preserved list — one value per distinct leaf across
    the whole record (so a filing that lists the same agency in three activities yields it
    once). General: works for any nesting depth and any JSON shape, not just this corpus.
    """
    def walk(node: Any, parts: list[str]) -> list:
        if not parts:
            return [node]
        head, rest = parts[0], parts[1:]
        is_arr = head.endswith("[]")
        key = head[:-2] if is_arr else head
        if not isinstance(node, dict) or key not in node:
            return []
        val = node[key]
        if is_arr:
            items = val if isinstance(val, list) else [val]
            out: list = []
            for item in items:
                out.extend(walk(item, rest))
            return out
        return walk(val, rest)

    seen: set = set()
    result: list = []
    for leaf in walk(record, path.split(".")):
        if isinstance(leaf, (str, int, float, bool)) and leaf not in seen:
            seen.add(leaf)
            result.append(leaf)
    return result


def _scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _flatten(obj: Any, prefix: str, out: list[str]) -> None:
    """Append deterministic 'dotted.key: value' lines for scalar leaves, in doc order."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        scalars = [x for x in obj if isinstance(x, (str, int, float, bool))]
        if scalars and len(scalars) == len(obj):
            out.append(f"{prefix}: {'; '.join(_scalar(x) for x in scalars)}")
        else:
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}[{i}]", out)
    elif obj is None:
        return
    else:
        out.append(f"{prefix}: {_scalar(obj)}")


def _render_field(record: Any, name: str, lines: list[str]) -> None:
    val = _get(record, name)
    if val is None:
        return
    if isinstance(val, (list, dict)):
        _flatten(val, name, lines)
    else:
        lines.append(f"{name}: {_scalar(val)}")


def render_text(record: Any, fields: FieldMap) -> str:
    """Deterministically render a record to searchable text. Shared by map + cite.

    Declared ``entity_fields`` are always rendered, even when ``text_fields`` doesn't list
    them: entity mentions are found by scanning this text, so a name that never appears here
    would yield no entity and no citable span — declaring a field would silently do nothing.
    (With no ``text_fields``, everything is flattened, so they're already covered.)
    """
    if not isinstance(record, dict):
        return _scalar(record)
    lines: list[str] = []
    if fields.text_fields:
        for name in fields.text_fields:
            _render_field(record, name, lines)
        listed = set(fields.text_fields)
        for spec in fields.entity_fields:
            name = spec.partition(":")[0].strip()
            if name and name not in listed:
                listed.add(name)
                _render_field(record, name, lines)
    else:
        _flatten(record, "", lines)
    return "\n".join(lines)


def _extract_entity_values(record: dict, entity_fields: list[str]) -> list[dict]:
    """Parse 'field:type' declarations into typed entity mentions (type defaults to org)."""
    out: list[dict] = []
    for spec in entity_fields:
        field_name, _, etype = spec.partition(":")
        etype = etype or "org"
        val = _get(record, field_name.strip())
        if val is None:
            continue
        values = val if isinstance(val, list) else [val]
        for v in values:
            if isinstance(v, (str, int, float)) and str(v).strip():
                out.append({"type": etype, "name": str(v).strip()})
    return out


def _title(record: Any, pointer: str, text: str, fields: FieldMap) -> str:
    if fields.title_template and isinstance(record, dict):
        flat = {k: _scalar(v) for k, v in record.items() if not isinstance(v, (list, dict))}

        class _Default(dict):
            def __missing__(self, key):  # tolerate missing template keys
                return ""

        try:
            return fields.title_template.format_map(_Default(flat))
        except Exception:
            pass
    if fields.record_id:
        rid = _get(record, fields.record_id)
        if rid is not None:
            return _scalar(rid)
    first = text.strip().splitlines()[0] if text.strip() else pointer
    return first[:120] or pointer


class JsonMapper:
    name = "json"
    handles = (".json", ".jsonl")

    def map(self, path: str, fields: FieldMap) -> list[Record]:
        kind, data = _load(path)
        records: list[Record] = []
        if kind == "jsonl":
            pairs = [(f"/{i}", rec) for i, rec in enumerate(data)]
        else:
            pairs = _detect_records(data, fields)
        for pointer, rec in pairs:
            text = render_text(rec, fields)
            structured = {}
            if fields.structured_fields and isinstance(rec, dict):
                for spec in fields.structured_fields:
                    fpath, alias = _parse_field_spec(spec)  # not `path` — that's the source file
                    if "[]" in fpath:  # array-path -> multi-valued column
                        structured[alias or fpath] = _get_multi(rec, fpath)
                    else:
                        structured[alias or fpath] = _get(rec, fpath)
            if fields.entity_fields and isinstance(rec, dict):
                structured["__entities__"] = _extract_entity_values(rec, fields.entity_fields)
            records.append(
                Record(
                    source_path=str(path),
                    doc_type="json",
                    text=text,
                    title=_title(rec, pointer, text, fields),
                    locator=pointer,
                    structured=structured,
                    raw=rec,
                )
            )
        return records


def map_one_text(path: str, locator: str, fields: FieldMap) -> str:
    """Re-derive one record's exact text from source — used by cite/verify/audit."""
    kind, data = _load(path)
    if kind == "jsonl":
        idx = int(locator.split("/")[1]) if locator else 0
        rec = data[idx]
    else:
        rec = _navigate(data, locator)
    return render_text(rec, fields)


register_mapper("json", JsonMapper)
