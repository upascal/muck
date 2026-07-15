"""Progressive search: keyword (BM25) / semantic (vectors) / hybrid (RRF).

Mirrors the graceful multi-level fallback of ``deep-reader``'s search: ``auto`` degrades
to keyword when embeddings aren't available, so retrieval never fails for lack of a model
or API key. Every result is returned as a ``Hit`` carrying a verifiable ``Citation``.
"""

from __future__ import annotations

import math
import re
import sqlite3

from ..cite import make_citation
from ..config import Settings
from ..db import meta_get
from ..embedders import build_embedder
from ..interfaces.store import get_store
from ..schema import Hit


def _vectors_ready(conn, settings: Settings, store) -> bool:
    return (
        settings.embedder.enabled
        and store.supports_vectors(conn)
        and meta_get(conn, "embed_dim") is not None
    )


def _rrf(rank_lists: list[list[str]], k0: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k0 + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


_WORD = re.compile(r"[\w'-]{3,}")
_WS = re.compile(r"\s+")

# PDF text arrives hard-wrapped: a lone newline mid-sentence is a line wrap, not a break.
_WRAP = re.compile(r"(?<![.!?•])\r?\n(?![•\r\n])")
# Don't end a sentence on a known abbreviation or a single initial ("Dr. Tarek", "J. Smith").
_ABBR = "".join(
    rf"(?<!\b{a}\.)" for a in ("Dr", "Mr", "Ms", "Mrs", "Prof", "No", "Inc", "Ltd", "St", "vs")
) + r"(?<!\b[A-Z]\.)"
_SENT_BREAK = re.compile(_ABBR + r"(?<=[.!?])\s+|\s*[•·▪]\s*|(?:\r?\n){2,}")
_MIN_SEG = 25  # chars; shorter spans are headings/fragments, not evidence
_TOP_WORDS = 3  # a sentence is scored on its few most salient words, not its average


def _segments(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Un-wrap PDF line breaks, then split into sentence-ish spans of the normalized text."""
    norm = _WRAP.sub(" ", text)
    spans, start = [], 0
    for m in _SENT_BREAK.finditer(norm):
        if norm[start : m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if norm[start:].strip():
        spans.append((start, len(norm)))
    return norm, [(s, e) for s, e in spans if len(norm[s:e].strip()) >= _MIN_SEG]


class _Salience:
    """Scores a word by how query-related *and* corpus-specific it is: ``cos(word, query) x idf``.

    Relevance alone picks boilerplate: in a corpus about ANNA membership, every sentence in a
    chunk matches a query about ANNA membership, and the blandest one usually wins. Rarity alone
    picks noise (phone numbers, names). Their product is what the answer sentence looks like — it
    says something both on-topic and unusual ("suspension ... Nairobi"), which is exactly what
    boilerplate cannot. idf comes from the FTS index; word vectors from the already-loaded
    embedder. Both caches are shared across every hit in one search.
    """

    def __init__(self, conn, query: str, emb=None, qvec=None):
        self.conn = conn
        self.emb = emb
        self.n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 1
        self.terms = set(_WORD.findall(query.lower()))
        self._df: dict[str, int] = {}
        self._wv: dict[str, object] = {}
        self.qvec = None
        if emb is not None and qvec is not None:
            import numpy as np

            qv = np.asarray(qvec, dtype="float32")
            self.qvec = qv / (float(np.linalg.norm(qv)) + 1e-9)

    def idf(self, word: str) -> float:
        """BM25 smoothed idf: always positive, and stays meaningful on tiny corpora.

        The naive ``log(n / (1 + df))`` collapses to 0 (or negative) once df approaches n, which
        on a small index zeroes out every score.
        """
        if word not in self._df:
            try:
                self._df[word] = self.conn.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{word}"',)
                ).fetchone()[0]
            except sqlite3.Error:
                self._df[word] = self.n  # unindexable term -> idf ~0, i.e. no opinion
        df = self._df[word]
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _warm(self, words: list[str]) -> None:
        """Embed every not-yet-cached word in one batch (per-word calls dominate otherwise)."""
        import numpy as np

        todo = [w for w in dict.fromkeys(words) if w not in self._wv]
        if not todo:
            return
        vecs = np.asarray(self.emb.embed(todo), dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        for w, v in zip(todo, vecs / norms):
            self._wv[w] = v

    def score(self, seg: str) -> float:
        """A sentence is worth as much as its few most salient words."""
        words = set(_WORD.findall(seg.lower()))
        if not words:
            return 0.0
        if self.qvec is None:  # no embedder: idf-weighted overlap with the query's own terms
            vals = sorted((self.idf(w) for w in words & self.terms), reverse=True)
        else:
            self._warm(list(words))
            vals = sorted((self.word_score(w) for w in words), reverse=True)
        return sum(vals[:_TOP_WORDS]) / _TOP_WORDS

    def word_score(self, word: str) -> float:
        if self.qvec is None:
            return self.idf(word) if word in self.terms else -1.0
        return max(0.0, float(self._wv[word] @ self.qvec)) * self.idf(word)

    def anchor(self, norm: str, s: int, e: int) -> int:
        """Offset to centre a window on: the segment's single most salient word."""
        words = [(m.start(), m.group().lower()) for m in _WORD.finditer(norm[s:e])]
        if not words:
            return s
        if self.qvec is not None:
            self._warm([w for _, w in words])
        return s + max(words, key=lambda ow: self.word_score(ow[1]))[0]


def _norm01(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    return [0.5] * len(vals) if hi - lo < 1e-9 else [(v - lo) / (hi - lo) for v in vals]


def _best_window(text: str, query: str, salience=None, width: int = 260) -> str:
    """Query-relevant snippet: a window around the sentence most likely to hold the evidence.

    Vector-leg hits have no FTS5 snippet and the chunk head is uncorrelated with why the vector
    leg matched, so the evidence gets buried (see docs/ field report). Sentences are ranked by
    ``_Salience``; the window centres on the winning sentence's most salient word, so a long
    sentence still shows the part that matters. Falls back to the chunk head when there is no
    query, no sentence structure, or nothing scores.
    """
    head = _WS.sub(" ", text[:width]).strip() + ("…" if len(text) > width else "")
    if not query or salience is None:
        return head
    norm, spans = _segments(text)
    if len(spans) < 2:
        return head

    scores = [salience.score(_WS.sub(" ", norm[s:e])) for s, e in spans]
    if max(scores) <= 0.0:
        return head

    s, e = spans[max(range(len(scores)), key=scores.__getitem__)]
    if e - s > width:
        lo = max(s, min(salience.anchor(norm, s, e) - width // 2, e - width))
        hi = lo + width
    else:
        lo = max(0, s - (width - (e - s)) // 2)
        hi = min(len(norm), lo + width)
        lo = max(0, hi - width)
    return ("…" if lo else "") + _WS.sub(" ", norm[lo:hi]).strip() + ("…" if hi < len(norm) else "")


def _build_hits(conn, ordered_ids, snippets, mode, query="", qvec=None, emb=None) -> list[Hit]:
    if not ordered_ids:
        return []
    placeholders = ",".join("?" * len(ordered_ids))
    rows = conn.execute(
        f"SELECT c.chunk_id, c.doc_id, c.text, c.char_start, c.char_end, c.locator, "
        f"d.source_path, d.title, d.text_provenance FROM chunks c JOIN documents d ON d.doc_id=c.doc_id "
        f"WHERE c.chunk_id IN ({placeholders})",
        ordered_ids,
    ).fetchall()
    by_id = {r["chunk_id"]: r for r in rows}
    # One salience cache per search: idf and word-vector lookups are shared across every snippet.
    salience = _Salience(conn, query, emb, qvec) if query else None
    hits = []
    for rank, cid in enumerate(ordered_ids):
        r = by_id.get(cid)
        if r is None:
            continue
        citation = make_citation(
            r["doc_id"], r["source_path"], r["locator"],
            r["char_start"], r["char_end"], r["text"],
        )
        snippet = snippets.get(cid) or _best_window(r["text"], query, salience)
        hits.append(
            Hit(
                chunk_id=cid,
                doc_id=r["doc_id"],
                doc_title=r["title"],
                source_path=r["source_path"],
                locator=r["locator"],
                score=1.0 / (1 + rank),
                match_mode=mode,
                snippet=snippet,
                citation=citation,
                text_provenance=(r["text_provenance"] if _row_has(r, "text_provenance") else None) or "native",
            )
        )
    return hits


def _row_has(row, key: str) -> bool:
    try:
        return key in row.keys()
    except Exception:
        return False


def search(conn, settings: Settings, query: str, mode: str = "auto", k: int = 8, filters=None) -> list[Hit]:
    store = get_store(settings.store.backend)
    has_vec = _vectors_ready(conn, settings, store)
    if mode == "auto":
        mode = "hybrid" if has_vec else "keyword"

    rerank_on = settings.reranker.enabled
    pool = max(k * 5, 20) if rerank_on else k  # rerank a wider candidate set, then trim

    bm_ids: list[str] = []
    vec_ids: list[str] = []
    snippets: dict[str, str] = {}
    emb = qvec = None

    if mode in ("keyword", "hybrid"):
        bm_res = store.search_bm25(conn, query, pool, filters)
        bm_ids = [cid for cid, _, _ in bm_res]
        snippets = {cid: snip for cid, _, snip in bm_res}

    if mode in ("semantic", "hybrid") and has_vec:
        emb = build_embedder(settings)
        qvec = emb.embed([query])[0]
        vec_ids = [cid for cid, _ in store.search_vector(conn, qvec, pool, filters)]

    if mode == "keyword":
        ordered = bm_ids
    elif mode == "semantic":
        ordered = vec_ids
    else:
        ordered = _rrf([bm_ids, vec_ids])

    hits = _build_hits(conn, ordered[:pool], snippets, mode, query, qvec, emb)

    if rerank_on and hits:
        from ..interfaces.reranker import get_reranker

        reranker = get_reranker(settings.reranker.name)
        scores = reranker.rerank(query, [h.citation.quote for h in hits])
        for h, s in zip(hits, scores):
            h.score = float(s)
            h.match_mode = f"{mode}+rerank"
        hits.sort(key=lambda h: h.score, reverse=True)

    return hits[:k]


def _resolve_entity_ids(conn, entity_id=None, name=None, etype=None) -> list[str]:
    if entity_id:
        return [entity_id]
    q, params = "SELECT entity_id FROM entities WHERE 1=1", []
    if etype:
        q += " AND entity_type = ?"
        params.append(etype)
    if name:
        q += (" AND (canonical_name LIKE ? OR norm_key LIKE ? OR entity_id IN "
              "(SELECT entity_id FROM entity_aliases WHERE alias LIKE ?))")
        params += [f"%{name}%", f"%{name.lower()}%", f"%{name}%"]
    return [r[0] for r in conn.execute(q, params).fetchall()]


def _entity_chunk_ids(conn, entity_ids: list[str]) -> list[str]:
    if not entity_ids:
        return []
    ph = ",".join("?" * len(entity_ids))
    return [r[0] for r in conn.execute(
        f"SELECT DISTINCT chunk_id FROM mentions WHERE entity_id IN ({ph})", entity_ids
    ).fetchall()]


def search_in_entity(conn, settings: Settings, query: str, *, entity_id=None, name=None,
                     etype=None, k: int = 8, min_score: float | None = None) -> list[Hit]:
    """Chunks that mention an entity (e.g. person = X), ranked by cosine to ``query``.

    Resolves the entity (merged aliases included) -> its chunk_ids, then ranks that
    candidate set by true cosine similarity to the query embedding. Falls back to keyword
    ranking within the same set if embeddings are disabled. ``min_score`` filters by cosine.
    """
    import numpy as np

    candidates = _entity_chunk_ids(conn, _resolve_entity_ids(conn, entity_id, name, etype))
    label = f"{etype or 'entity'}+cosine"
    if not candidates:
        return []
    if not query:
        return _build_hits(conn, candidates[:k], {}, f"{etype or 'entity'}-only")

    emb = build_embedder(settings)
    if emb is not None:
        store = get_store(settings.store.backend)
        ids, mat = store.get_vectors(conn, candidates)  # reuse vectors stored at index time
        if not ids:
            # Keyword-only index (no stored vectors): recompute as a fallback.
            ph = ",".join("?" * len(candidates))
            rows = conn.execute(
                f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({ph})", candidates
            ).fetchall()
            ids = [r["chunk_id"] for r in rows]
            mat = np.asarray(emb.embed([r["text"] for r in rows]), dtype="float32")
        qvec = np.asarray(emb.embed([query])[0], dtype="float32")
        cos = (mat @ qvec).tolist()  # vectors are L2-normalized → dot product = cosine
        scored = sorted(zip(ids, cos), key=lambda t: t[1], reverse=True)
        if min_score is not None:
            scored = [(i, s) for i, s in scored if s >= min_score]
        scored = scored[:k]
        hits = _build_hits(conn, [i for i, _ in scored], {}, label, query, qvec, emb)
        score_map = dict(scored)
        for h in hits:
            h.score = round(float(score_map[h.chunk_id]), 4)
        return hits

    # No embeddings: rank the entity's chunks by BM25 within the candidate set.
    store = get_store(settings.store.backend)
    res = store.search_bm25(conn, query, k, {"chunk_ids": candidates})
    return _build_hits(conn, [c for c, _, _ in res], {c: s for c, _, s in res}, f"{etype or 'entity'}+keyword")


def grep(conn, settings: Settings, pattern: str, k: int = 20, filters=None) -> tuple[list[Hit], int]:
    """Regex hits (first ``k``; all when ``k <= 0``) plus the exact total match count."""
    store = get_store(settings.store.backend)
    res, total = store.grep(conn, pattern, k, filters)
    ids = [cid for cid, _ in res]
    snippets = {cid: snip for cid, snip in res}
    return _build_hits(conn, ids, snippets, "grep"), total
