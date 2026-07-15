"""SBERT embedder (upgrade): sentence-transformers, better quality than Potion.

Ported from ``agent-suite/lore``'s embedder. Needs the ``sbert`` extra. CPU works;
GPU is used automatically if available.
"""

from __future__ import annotations

from ..interfaces import NotInstalled
from ..interfaces.embedder import register_embedder


class SbertEmbedder:
    name = "sbert"
    requires_api_key = False

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model = None
        self._dim: int | None = None

    def configure(self, cfg) -> None:
        if self._model is None and cfg.model and cfg.model != "minishlab/potion-base-8M":
            self._model_name = cfg.model

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise NotInstalled(
                    "SBERT embeddings need sentence-transformers; install `uv sync --extra sbert`"
                ) from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[-1])
        return self._dim

    def embed(self, texts: list[str], batch_size: int = 64):
        import numpy as np

        model = self._ensure()
        vecs = np.asarray(
            model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True),
            dtype="float32",
        )
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self._dim = vecs.shape[-1]
        return vecs


register_embedder("sbert", lambda: SbertEmbedder())
