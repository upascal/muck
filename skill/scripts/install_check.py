#!/usr/bin/env python3
"""Report which muck capabilities are active. Run with the project's venv python.

Confirms the no-API/no-GPU default path works, and shows which optional upgrades are
installed. Exits non-zero only if the core (keyword) path is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


def _fts5() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False


def _sqlite_vec() -> bool:
    if not _have("sqlite_vec"):
        return False
    try:
        import sqlite_vec

        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.execute("SELECT vec_version()")
        return True
    except Exception:
        return False


def main() -> int:
    core = {
        "muck importable": _have("muck"),
        "SQLite FTS5 (keyword/BM25)": _fts5(),
        "pypdfium2 (PDF)": _have("pypdfium2"),
    }
    semantic = {
        "model2vec / Potion (default embeddings)": _have("model2vec"),
        "sqlite-vec (vector index)": _sqlite_vec(),
    }
    upgrades = {
        "analytics: duckdb + scikit-learn (aggregate/cluster)": _have("duckdb") and _have("sklearn"),
        "docx (python-docx)": _have("docx"),
        "pymupdf (PyMuPDF parser)": _have("fitz"),
        "sbert (sentence-transformers / reranker)": _have("sentence_transformers"),
        "api embedders (httpx)": _have("httpx"),
        "splink (reconcile: probabilistic record linkage)": _have("splink"),
        "turbovec (in-memory ANN accelerator, large corpora)": _have("turbovec"),
    }
    semantic_on = all(semantic.values())
    path = (
        "keyword + Potion semantic search (no API key, no GPU)"
        if semantic_on else "keyword-only (BM25 + entities); enable embeddings with `uv sync`"
    )
    report = {"active_path": path, "core": core, "semantic": semantic, "upgrades": upgrades}
    print(json.dumps(report, indent=2))
    if not all(core.values()):
        print("\nCore path unavailable — run `uv sync --no-editable`.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
