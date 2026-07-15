"""Reranker interface (optional stage): re-score (query, passage) pairs for precision."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from . import Registry


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, passages: list[str]) -> list[float]: ...  # higher = better


RERANKERS: Registry[Reranker] = Registry("reranker")


def register_reranker(name, factory):
    RERANKERS.register(name, factory)


def get_reranker(name: str) -> Reranker:
    return RERANKERS.get(name)
