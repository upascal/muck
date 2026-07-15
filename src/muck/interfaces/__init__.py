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
