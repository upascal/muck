"""Default embedder: Potion static embeddings via model2vec.

CPU-only, ~30MB, no torch at inference, no API key — the default-on semantic path.
Vectors are L2-normalized so a vector store using L2 distance ranks like cosine.
"""

from __future__ import annotations

from ..interfaces import NotInstalled
from ..interfaces.embedder import register_embedder


class PotionEmbedder:
    name = "potion"
    requires_api_key = False

    def __init__(self, model: str = "minishlab/potion-base-8M") -> None:
        self._model_name = model
        self._revision: str | None = None
        self._model = None
        self._dim: int | None = None

    def configure(self, cfg) -> None:
        if self._model is None and cfg.model:
            self._model_name = cfg.model
        if self._model is None:
            self._revision = getattr(cfg, "revision", None)

    def _ensure(self):
        if self._model is None:
            try:
                from model2vec import StaticModel
            except ImportError as e:  # pragma: no cover
                raise NotInstalled(
                    "Potion embeddings need model2vec (in core deps); run `uv sync`"
                ) from e
            source = self._model_name
            if self._revision:
                # Pin the exact weights: model2vec's from_pretrained doesn't forward `revision`,
                # so resolve the revision to a local snapshot and load that (reproducible index).
                from huggingface_hub import snapshot_download

                source = snapshot_download(self._model_name, revision=self._revision)
            self._model = StaticModel.from_pretrained(source)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[-1])
        return self._dim

    def embed(self, texts: list[str], batch_size: int = 256):
        import numpy as np

        model = self._ensure()
        vecs = np.asarray(model.encode(list(texts)), dtype="float32")
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._dim = vecs.shape[-1]
        return vecs / norms


register_embedder("potion", lambda: PotionEmbedder())
