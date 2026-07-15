"""Embedders. Default: Potion (model2vec, CPU). Upgrades: sbert, api (registered lazily).

``build_embedder(settings)`` is the single construction path used by the pipeline and
search: it resolves the configured adapter, applies config overrides (model / api key
env), and caches the instance per process so the model loads once.
"""

from __future__ import annotations

from ..interfaces.embedder import get_embedder
from . import api_embedder as _api  # noqa: F401  (registers openai/voyage/deepinfra/gemini/api)
from . import potion_embedder as _potion  # noqa: F401  (registers "potion")
from . import sbert_embedder as _sbert  # noqa: F401  (registers "sbert")

_CACHE: dict = {}


def build_embedder(settings):
    if not settings.embedder.enabled:
        return None
    cfg = settings.embedder
    key = (cfg.name, cfg.model, cfg.api_key_env)
    if key not in _CACHE:
        emb = get_embedder(cfg.name)
        if hasattr(emb, "configure"):
            emb.configure(cfg)
        _CACHE[key] = emb
    return _CACHE[key]
