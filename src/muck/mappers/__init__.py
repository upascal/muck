"""Mappers: structured-data files -> logical documents. Default: JSON records."""

from __future__ import annotations

from . import json_mapper as _json_mapper  # noqa: F401  (registers "json")
from . import xml_mapper as _xml_mapper  # noqa: F401  (registers "xml")
