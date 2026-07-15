"""Stores. Default (and source of truth): sqlite_store (FTS5 + sqlite-vec, one DB file)."""

from __future__ import annotations

from . import sqlite_store as _sqlite  # noqa: F401  (registers "sqlite-fts5")
from . import turbovec_store as _turbovec  # noqa: F401  (registers "turbovec", lazy dep)
