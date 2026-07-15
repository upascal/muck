"""Rerankers (optional). Cross-encoder registered lazily (needs the ``sbert`` extra)."""

from __future__ import annotations

from . import cross_encoder as _cross_encoder  # noqa: F401  (registers "cross-encoder")
