"""XML mapper: an XML file -> one Record, or many when ``record_path`` is set.

Flattens the XML tree into the same dotted-key dict shape the JSON mapper produces, then
reuses its deterministic rendering + field-map machinery (``render_text``,
``_extract_entity_values``) — so ``[mapper.fields]`` specs like ``organizationName:org`` or
``lobbyists.lobbyist`` work unchanged, and citations re-derive from source the same way.

One record per file by default (House LDA ships one filing per XML). For bulk-export XML that
packs many records into one file, set ``record_path`` to an ElementTree path (e.g. ``"filing"``
or ``".//filing"``); each matched element becomes a Record with ``locator = str(index)``.

Repeated sibling elements become lists; leaf text is stripped (House files are full of
whitespace-only elements). Attributes are ignored (LDA filings don't use them for data).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..config import FieldMap
from ..interfaces.mapper import register_mapper
from ..schema import Record
from .json_mapper import _extract_entity_values, _get, render_text


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def element_to_obj(el: ET.Element) -> Any:
    """Element -> str (leaf) | dict (children; repeated tags -> lists) | None (empty)."""
    children = list(el)
    if not children:
        text = (el.text or "").strip()
        return text or None
    out: dict[str, Any] = {}
    for child in children:
        key = _strip_ns(child.tag)
        val = element_to_obj(child)
        if val is None:
            continue
        if key in out:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(val)
        else:
            out[key] = val
    return out or None


def _to_dict(el: ET.Element) -> dict:
    obj = element_to_obj(el)
    rec = obj if isinstance(obj, dict) else {"text": obj}
    rec["_root"] = _strip_ns(el.tag)  # e.g. LOBBYINGDISCLOSURE1 vs LOBBYINGDISCLOSURE2
    return rec


def xml_file_to_record_dict(path: str) -> dict:
    """Whole file as one record dict (no ``record_path``)."""
    return _to_dict(ET.parse(path).getroot())


def _record_dicts(path: str, fields: FieldMap) -> list[dict]:
    """[(record dict)] — one per ``record_path`` element, else the whole file."""
    root = ET.parse(path).getroot()
    if not fields.record_path:
        return [_to_dict(root)]
    return [_to_dict(el) for el in root.findall(fields.record_path)]


def _title_for(rec: dict, fields: FieldMap, path: str, idx: int, multi: bool) -> str:
    if fields.title_template:
        flat = {k: str(v) for k, v in rec.items() if not isinstance(v, (list, dict))}

        class _Default(dict):
            def __missing__(self, key):
                return ""

        try:
            title = fields.title_template.format_map(_Default(flat)).strip()
            if title:
                return title
        except Exception:
            pass
    if fields.record_id:
        rid = _get(rec, fields.record_id)
        if rid is not None and str(rid).strip():
            return str(rid).strip()
    stem = Path(path).stem
    return f"{stem}#{idx}" if multi else stem


class XmlMapper:
    name = "xml"
    handles = (".xml",)

    def map(self, path: str, fields: FieldMap) -> list[Record]:
        recs = _record_dicts(path, fields)
        multi = bool(fields.record_path)
        out: list[Record] = []
        for i, rec in enumerate(recs):
            structured: dict[str, Any] = {}
            if fields.structured_fields:
                structured = {name: _get(rec, name) for name in fields.structured_fields}
            if fields.entity_fields:
                structured["__entities__"] = _extract_entity_values(rec, fields.entity_fields)
            out.append(Record(
                source_path=str(path), doc_type="xml", text=render_text(rec, fields),
                title=_title_for(rec, fields, path, i, multi),
                locator=str(i) if multi else "",
                structured=structured, raw=rec,
            ))
        return out


def map_one_text(path: str, locator: str, fields: FieldMap) -> str:
    """Re-derive one record's exact text from the source XML — for cite/verify/audit."""
    recs = _record_dicts(path, fields)
    idx = int(locator) if (fields.record_path and str(locator).strip()) else 0
    return render_text(recs[idx], fields)


register_mapper("xml", XmlMapper)
