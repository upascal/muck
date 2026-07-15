"""SBERT embedder (upgrade): sentence-transformers, better quality than Potion.

Ported from ``agent-suite/lore``'s embedder. Needs the ``sbert`` extra. CPU works;
GPU is used automatically if available.
"""

from __future__ import annotations

from ..interfaces import NotInstalled, pick_torch_device
from ..interfaces.embedder import register_embedder


class SbertEmbedder:
    name = "sbert"
    requires_api_key = False

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model = None
        self._dim: int | None = None
        self._device: str | None = None

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
            # Use the GPU if there is one (incl. Apple-Silicon MPS, which ST won't auto-pick);
            # fall back to CPU / the library default on any failure. GPU is never required.
            self._device = pick_torch_device()
            try:
                self._model = SentenceTransformer(self._model_name, device=self._device)
            except Exception:
                self._device = None
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
        # Larger batches keep a GPU busy; CPU stays at the conservative default.
        if self._device in ("cuda", "mps"):
            batch_size = max(batch_size, 256)
        vecs = np.asarray(
            model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True),
            dtype="float32",
        )
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self._dim = vecs.shape[-1]
        return vecs


register_embedder("sbert", lambda: SbertEmbedder())
