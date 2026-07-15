"""turbovec vector backend (opt-in: ``--extra turbovec``) — in-memory ANN at scale.

sqlite-vec remains the **source of truth** (exact vectors persist in ``.muck/index.db``).
turbovec is a *derived accelerator*: a quantized ANN index built from those stored vectors
and cached at ``.muck/cache/turbovec.tv``, keyed to a fingerprint of the index state so it
auto-rebuilds when stale — no second store to hand-sync. Keyword/grep/``get_vectors`` are
inherited from ``SqliteStore`` (so entity-filtered search still uses exact vectors); turbovec
only accelerates ``search_vector`` (general semantic search), with native ``allowlist``
filtering. Requires the embedding dim be a multiple of 8 (Potion's 256 qualifies).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..db import meta_get, meta_set
from ..interfaces import NotInstalled
from ..interfaces.store import register_store
from .sqlite_store import SqliteStore, _filter_sql

BIT_WIDTH = 4  # 4-bit quantization: best recall/speed balance per turbovec benchmarks


def _turbovec():
    try:
        import turbovec
    except ImportError as e:  # pragma: no cover
        raise NotInstalled("the turbovec store needs `uv sync --extra turbovec`") from e
    return turbovec


class TurbovecStore(SqliteStore):
    name = "turbovec"

    def __init__(self) -> None:
        self._index = None
        self._fp: str | None = None

    # --- cache management ------------------------------------------------
    def _cache_path(self, conn) -> Path:
        row = next((r for r in conn.execute("PRAGMA database_list") if r["name"] == "main"), None)
        base = Path(row["file"]).parent if row and row["file"] else Path(".muck")
        d = base / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d / "turbovec.tv"

    def _fingerprint(self, conn) -> str:
        n = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        mx = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM chunks").fetchone()[0]
        return f"{meta_get(conn, 'embed_dim')}:{n}:{mx}"

    def upsert_vectors(self, conn, ids, vecs) -> None:
        super().upsert_vectors(conn, ids, vecs)  # sqlite-vec is the source of truth
        self._index, self._fp = None, None  # invalidate; rebuilt lazily on next search
        cache = self._cache_path(conn)
        if cache.exists():
            cache.unlink()
        meta_set(conn, "turbovec_fp", "")
        conn.commit()

    def _build(self, conn, dim: int):
        import numpy as np

        rows = conn.execute(
            "SELECT c.rowid AS rid, vec_to_json(cv.embedding) AS v "
            "FROM chunks_vec cv JOIN chunks c ON c.chunk_id = cv.chunk_id"
        ).fetchall()
        if not rows:
            return None
        rids = np.asarray([r["rid"] for r in rows], dtype="uint64")
        mat = np.asarray([json.loads(r["v"]) for r in rows], dtype="float32")
        idx = _turbovec().IdMapIndex(dim=dim, bit_width=BIT_WIDTH)
        idx.add_with_ids(mat, rids)
        idx.prepare()
        return idx

    def _ensure(self, conn):
        dim_s = meta_get(conn, "embed_dim")
        if dim_s is None:
            return None
        dim = int(dim_s)
        if dim % 8 != 0:
            raise NotInstalled(
                f"turbovec needs an embedding dim that is a multiple of 8; got {dim}. "
                f"Use Potion (256) or keep the default sqlite-fts5 store."
            )
        fp = self._fingerprint(conn)
        if self._index is not None and self._fp == fp:
            return self._index
        cache = self._cache_path(conn)
        if cache.exists() and meta_get(conn, "turbovec_fp") == fp:
            self._index, self._fp = _turbovec().IdMapIndex.load(str(cache)), fp
            return self._index
        idx = self._build(conn, dim)
        if idx is None:
            return None
        idx.write(str(cache))
        meta_set(conn, "turbovec_fp", fp)
        conn.commit()
        self._index, self._fp = idx, fp
        return idx

    def _allowlist(self, conn, filters):
        import numpy as np

        if not filters:
            return None
        rids = None
        if filters.get("chunk_ids"):
            cids = list(filters["chunk_ids"])
            ph = ",".join("?" * len(cids))
            rids = [r[0] for r in conn.execute(f"SELECT rowid FROM chunks WHERE chunk_id IN ({ph})", cids)]
        doc_filters = {k: v for k, v in filters.items() if k != "chunk_ids"}
        if doc_filters:
            where, params = _filter_sql(doc_filters)
            doc_rids = [r[0] for r in conn.execute(
                f"SELECT c.rowid FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE 1=1{where}", params
            )]
            rids = doc_rids if rids is None else [r for r in rids if r in set(doc_rids)]
        return None if rids is None else np.asarray(rids, dtype="uint64")

    def search_vector(self, conn, vec, k, filters=None):
        import numpy as np

        idx = self._ensure(conn)
        if idx is None:
            return []
        allow = self._allowlist(conn, filters)
        kwargs = {"k": k}
        if allow is not None:
            if len(allow) == 0:
                return []
            kwargs["allowlist"] = allow
        scores, ids = idx.search(np.asarray(vec, dtype="float32").reshape(1, -1), **kwargs)
        rids = [int(x) for x in ids[0]]
        sims = [float(s) for s in scores[0]]
        if not rids:
            return []
        ph = ",".join("?" * len(rids))
        rid2cid = {r["rowid"]: r["chunk_id"]
                   for r in conn.execute(f"SELECT rowid, chunk_id FROM chunks WHERE rowid IN ({ph})", rids)}
        # turbovec returns similarity (higher=better); expose as distance (lower=better) for the contract.
        return [(rid2cid[rid], 1.0 - sim) for rid, sim in zip(rids, sims) if rid in rid2cid]

    def supports_vectors(self, conn) -> bool:
        try:
            _turbovec()
            return True
        except NotInstalled:
            return False


register_store("turbovec", lambda: TurbovecStore())
