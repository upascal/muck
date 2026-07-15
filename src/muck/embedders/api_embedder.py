"""API embedders (upgrade): OpenAI-compatible /embeddings for several providers.

Ported in spirit from ``deep-reader``'s batch+retry embedding service. Needs the ``api``
extra (httpx) and a provider API key in the environment. Select via config, e.g.::

    [embedder]
    name = "voyage"           # or openai | deepinfra | gemini
    model = "voyage-3"
    api_key_env = "VOYAGE_API_KEY"
"""

from __future__ import annotations

import os

from ..interfaces import NotInstalled
from ..interfaces.embedder import register_embedder

# provider -> (base_url, default_key_env, default_model)
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "text-embedding-3-small"),
    "voyage": ("https://api.voyageai.com/v1", "VOYAGE_API_KEY", "voyage-3"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY",
                  "Qwen/Qwen3-Embedding-4B"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
               "text-embedding-004"),
}


class ApiEmbedder:
    requires_api_key = True

    def __init__(self, provider: str) -> None:
        base, key_env, model = PROVIDERS[provider]
        self.name = provider
        self._base = base
        self._key_env = key_env
        self._model = model
        self._dim: int | None = None

    def configure(self, cfg) -> None:
        if cfg.model:
            self._model = cfg.model
        if cfg.api_key_env:
            self._key_env = cfg.api_key_env

    def _client(self):
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise NotInstalled("API embedders need httpx; install `uv sync --extra api`") from e
        key = os.environ.get(self._key_env)
        if not key:
            raise NotInstalled(
                f"set ${self._key_env} to use the {self.name!r} embedder (or switch to potion)"
            )
        return httpx, key

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[-1])
        return self._dim

    def embed(self, texts: list[str], batch_size: int = 128):
        import numpy as np

        httpx, key = self._client()
        out: list[list[float]] = []
        with httpx.Client(timeout=60) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = client.post(
                    f"{self._base}/embeddings",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                out.extend(item["embedding"] for item in resp.json()["data"])
        vecs = np.asarray(out, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._dim = vecs.shape[-1]
        return vecs / norms


for _provider in PROVIDERS:
    register_embedder(_provider, lambda p=_provider: ApiEmbedder(p))
register_embedder("api", lambda: ApiEmbedder("openai"))
