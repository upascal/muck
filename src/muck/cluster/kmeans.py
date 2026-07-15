"""KMeans clustering over chunk embeddings + c-TF-IDF topic labels.

Builds a topical index: each chunk is assigned a cluster, and each cluster gets a label
from the terms that most distinguish it (class-based TF-IDF). Reuses the Potion vectors
already stored in sqlite-vec, falling back to re-embedding if vec readout is unavailable.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from ..interfaces import NotInstalled


def _load_vectors(conn, settings):
    import numpy as np

    try:
        rows = conn.execute(
            "SELECT cv.chunk_id AS cid, vec_to_json(cv.embedding) AS emb, c.text AS text "
            "FROM chunks_vec cv JOIN chunks c ON c.chunk_id = cv.chunk_id"
        ).fetchall()
        if rows:
            ids = [r["cid"] for r in rows]
            texts = [r["text"] for r in rows]
            X = np.asarray([json.loads(r["emb"]) for r in rows], dtype="float32")
            return ids, texts, X
    except sqlite3.OperationalError:
        pass
    # Fallback: re-embed chunk texts.
    rows = conn.execute("SELECT chunk_id, text FROM chunks").fetchall()
    if not rows:
        return [], [], None
    from ..interfaces.embedder import get_embedder

    ids = [r["chunk_id"] for r in rows]
    texts = [r["text"] for r in rows]
    X = get_embedder(settings.embedder.name).embed(texts)
    return ids, texts, X


def _auto_k(n: int) -> int:
    return max(2, min(40, round((n / 2) ** 0.5)))


def build_clusters(conn: sqlite3.Connection, settings, k="auto") -> dict:
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:  # pragma: no cover
        raise NotInstalled(
            "clustering needs scikit-learn; install with `uv sync --extra analytics`"
        ) from e
    import numpy as np

    ids, texts, X = _load_vectors(conn, settings)
    n = len(ids)
    if n == 0:
        return {"clusters": 0, "chunks": 0, "note": "no embeddings — run `muck index` with embeddings on"}

    kk = _auto_k(n) if isinstance(k, str) else int(k)
    kk = max(1, min(kk, n))
    labels = [0] * n if kk == 1 else KMeans(n_clusters=kk, n_init=10, random_state=0).fit_predict(X).tolist()

    cluster_texts: dict[int, list[str]] = defaultdict(list)
    for cid, text in zip(labels, texts):
        cluster_texts[int(cid)].append(text)
    cluster_ids = sorted(cluster_texts)

    terms_map: dict[int, list[str]] = {c: [] for c in cluster_ids}
    docs = [" ".join(cluster_texts[c]) for c in cluster_ids]
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
        tfidf = vec.fit_transform(docs)
        vocab = np.asarray(vec.get_feature_names_out())
        for i, c in enumerate(cluster_ids):
            row = tfidf[i].toarray().ravel()
            top = row.argsort()[::-1][:5]
            terms_map[c] = [vocab[j] for j in top if row[j] > 0]
    except ValueError:
        pass  # too few terms to vectorize; leave labels empty

    conn.execute("DELETE FROM clusters")
    conn.execute("DELETE FROM chunk_clusters")
    summary = []
    for c in cluster_ids:
        label = ", ".join(terms_map[c]) or f"cluster {c}"
        size = len(cluster_texts[c])
        conn.execute(
            "INSERT INTO clusters(cluster_id, label, terms_json, size) VALUES(?,?,?,?)",
            (c, label, json.dumps(terms_map[c]), size),
        )
        summary.append({"cluster_id": c, "label": label, "size": size})
    for cid, chunk_id in zip(labels, ids):
        conn.execute(
            "INSERT OR REPLACE INTO chunk_clusters(chunk_id, cluster_id) VALUES(?, ?)",
            (chunk_id, int(cid)),
        )
    conn.commit()
    return {"clusters": len(cluster_ids), "chunks": n, "labels": summary}


def list_clusters(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT cluster_id, label, size FROM clusters ORDER BY size DESC"
    )]
