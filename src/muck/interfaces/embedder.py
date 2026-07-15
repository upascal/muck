"""Embedder interface: texts -> dense vectors (N, dim) float32."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from . import Registry

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Embedder(Protocol):
    name: str
    requires_api_key: bool

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str], batch_size: int = 256) -> "np.ndarray": ...


EMBEDDERS: Registry[Embedder] = Registry("embedder")


def register_embedder(name, factory):
    EMBEDDERS.register(name, factory)


def get_embedder(name: str) -> Embedder:
    return EMBEDDERS.get(name)
