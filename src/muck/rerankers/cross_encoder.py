"""Cross-encoder reranker (upgrade). Ported from ``flashvec/rerank/simple.py``.

Needs the ``sbert`` extra. Re-scores retrieved passages against the query for a precision
boost; enable via ``[reranker] enabled = true`` in config.
"""

from __future__ import annotations

from ..interfaces import NotInstalled
from ..interfaces.reranker import register_reranker


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:  # pragma: no cover
                raise NotInstalled(
                    "Reranking needs sentence-transformers; install `uv sync --extra sbert`"
                ) from e
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._ensure()
        scores = model.predict([(query, p) for p in passages])
        return [float(s) for s in scores]


register_reranker("cross-encoder", lambda: CrossEncoderReranker())
