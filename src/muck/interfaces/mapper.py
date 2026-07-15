"""Mapper interface: a structured-data file -> many logical documents (Records)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import FieldMap
from ..schema import Record
from . import Registry


@runtime_checkable
class Mapper(Protocol):
    name: str
    handles: tuple[str, ...]  # file extensions, e.g. (".json", ".jsonl")

    def map(self, path: str, fields: FieldMap) -> list[Record]: ...


MAPPERS: Registry[Mapper] = Registry("mapper")


def register_mapper(name, factory):
    MAPPERS.register(name, factory)


def get_mapper(name: str) -> Mapper:
    return MAPPERS.get(name)
