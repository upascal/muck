"""Store interface: keyword (FTS5/BM25) + vector (ANN) indexing and search.

Adapters are stateless and take the open DB connection per call, so the zero-arg
factory registry pattern holds. The default ``sqlite_store`` implements both keyword
and vector against one ``.muck/index.db``; an upgrade (e.g. a turbovec-backed store) can
implement the vector side and delegate keyword search to SQLite.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..schema import Chunk
from . import Registry

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Store(Protocol):
    name: str

    def index_chunks(self, conn: sqlite3.Connection, chunks: list[Chunk]) -> None: ...

    def search_bm25(
        self, conn: sqlite3.Connection, query: str, k: int, filters: dict | None = None
    ) -> list[tuple[str, float, str]]:  # (chunk_id, score, snippet)
        ...

    def grep(
        self, conn: sqlite3.Connection, pattern: str, k: int, filters: dict | None = None
    ) -> tuple[list[tuple[str, str]], int]:  # ([(chunk_id, snippet)], total_matches)
        ...

    def supports_vectors(self, conn: sqlite3.Connection) -> bool: ...

    def upsert_vectors(
        self, conn: sqlite3.Connection, ids: list[str], vecs: "np.ndarray"
    ) -> None: ...

    def search_vector(
        self, conn: sqlite3.Connection, vec: "np.ndarray", k: int, filters: dict | None = None
    ) -> list[tuple[str, float]]:  # (chunk_id, distance)
        ...

    def get_vectors(
        self, conn: sqlite3.Connection, chunk_ids: list[str]
    ) -> tuple[list[str], "np.ndarray | None"]:
        """Return the stored vectors for the given chunk_ids (those present), aligned.

        Lets callers reuse vectors computed at index time instead of re-embedding —
        e.g. entity-filtered cosine search. Returns ([], None) if no vectors are stored.
        """
        ...


STORES: Registry[Store] = Registry("store")


def register_store(name, factory):
    STORES.register(name, factory)


def get_store(name: str) -> Store:
    return STORES.get(name)
