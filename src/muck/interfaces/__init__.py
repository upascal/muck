"""Pluggable stage interfaces: a ``Protocol`` + name->factory registry per stage.

Pattern ported from ``document-parsing/src/pdfoutline/backends/base.py``: adapters
register a zero-arg factory by name; optional-dependency adapters raise
``NotInstalled`` *from the factory* (not at import), so the package imports cleanly
with only the default adapters present. Adding an adapter = drop one file that calls
``register_*`` and name it in ``config.toml``.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class NotInstalled(RuntimeError):
    """Raised when an adapter's optional dependency is not installed."""


def pick_torch_device() -> str | None:
    """Best available torch device for the opt-in sbert/cross-encoder path: ``cuda`` → ``mps``
    (Apple Silicon) → ``cpu``. Returns ``None`` on any failure so the caller falls back to the
    library default. sentence-transformers auto-picks CUDA but *not* MPS, so without this an
    Apple-Silicon machine silently runs on CPU. Never required — GPU is a bonus, CPU always works.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return None


class Registry(Generic[T]):
    """A name -> zero-arg factory map for one pipeline stage."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[[], T]] = {}

    def register(self, name: str, factory: Callable[[], T]) -> None:
        self._factories[name] = factory

    def get(self, name: str) -> T:
        if name not in self._factories:
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered: {sorted(self._factories)}"
            )
        return self._factories[name]()

    def available(self) -> list[str]:
        return sorted(self._factories)
