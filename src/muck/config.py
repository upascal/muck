"""Configuration: ``.muck/config.toml`` -> typed pydantic ``Settings``.

The config selects which adapter implements each pipeline stage (mapper, parser,
embedder, store) plus their parameters. Defaults give the no-API, CPU-only path:
config-driven JSON mapping + Potion embeddings + SQLite (FTS5 + sqlite-vec).
"""

from __future__ import annotations

import tomllib
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

MUCK_DIRNAME = ".muck"
CONFIG_FILENAME = "config.toml"


class FieldMap(BaseModel):
    """How a JSON record maps to a document. All optional — empty = auto-detect."""

    record_path: str | None = None  # JSON pointer to the list of records; None = auto
    record_id: str | None = None  # field used as the record id; None = use json pointer
    title_template: str | None = None  # e.g. "{registrant} — {client} ({filed_date})"
    text_fields: list[str] = Field(default_factory=list)  # [] = flatten all string fields
    structured_fields: list[str] = Field(default_factory=list)  # promoted for SQL/aggregation
    entity_fields: list[str] = Field(default_factory=list)  # names fed to the resolver


class SourceMap(FieldMap):
    """A FieldMap scoped to source files matching a glob (heterogeneous corpora)."""

    match: str  # fnmatch-style; `*` crosses `/`. Relative patterns are prefixed `*/`.


class MapperCfg(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    backend: str = Field("json", alias="json")  # TOML key "json"; avoids shadowing BaseModel.json
    fields: FieldMap = Field(default_factory=FieldMap)  # global default
    sources: list[SourceMap] = Field(default_factory=list)  # first matching glob wins


class ParserCfg(BaseModel):
    pdf: str = "pdfium"  # -> pymupdf | docling
    docx: str = "docx"
    text: str = "text"


class EmbedderCfg(BaseModel):
    enabled: bool = True  # default-on; set false for keyword-only (zero model download)
    name: str = "potion"  # -> sbert | voyage | openai | deepinfra | gemini
    model: str = "minishlab/potion-base-8M"
    revision: str | None = None  # HF commit/tag to pin the weights (reproducibility); None = latest
    contextual: bool = False  # upgrade seam: prepend generated context before embedding
    api_key_env: str | None = None


class StoreCfg(BaseModel):
    backend: str = "sqlite-fts5"  # keyword (FTS5) + vectors (sqlite-vec) in one .muck/index.db


class RerankerCfg(BaseModel):
    enabled: bool = False
    name: str = "cross-encoder"


class ChunkCfg(BaseModel):
    size_tokens: int = 320
    overlap_tokens: int = 64
    token_estimate_factor: float = 0.75


class ClusterCfg(BaseModel):
    enabled: bool = True
    k: int | str = "auto"


class EntitiesCfg(BaseModel):
    resolve: bool = True
    # Free-text gazetteer matching knobs (see ADAPTERS.md for the rationale):
    min_name_len: int = 3  # names shorter than this never free-text match
    single_word_match: str = "exact-case"  # exact-case | any-case | off
    extra_org_suffixes: list[str] = Field(default_factory=list)  # e.g. ["gmbh", "sa", "ab"]


class Settings(BaseModel):
    mapper: MapperCfg = Field(default_factory=MapperCfg)
    parser: ParserCfg = Field(default_factory=ParserCfg)
    embedder: EmbedderCfg = Field(default_factory=EmbedderCfg)
    store: StoreCfg = Field(default_factory=StoreCfg)
    chunk: ChunkCfg = Field(default_factory=ChunkCfg)
    cluster: ClusterCfg = Field(default_factory=ClusterCfg)
    entities: EntitiesCfg = Field(default_factory=EntitiesCfg)
    reranker: RerankerCfg = Field(default_factory=RerankerCfg)


DEFAULT_CONFIG_TOML = """\
# muck configuration — selects the adapter for each pipeline stage.
# Defaults need no API key and no GPU. See references/ADAPTERS.md to swap components.

[mapper]
json = "json"                 # JSON record mapper (the corpus path)

# Global field map (fallback). Leave empty to auto-detect + flatten all text.
# `muck peek <file-or-dir>` suggests these from the actual data.
[mapper.fields]
# record_path = "/results"          # JSON pointer to the list of records (XML: an ElementTree path)
# record_id = "filing_uuid"
# title_template = "{registrant} - {client} ({filing_period})"
# text_fields = ["description", "issues"]
# structured_fields = ["registrant", "client", "amount", "filing_period"]
# entity_fields = ["registrant:org", "client:org", "lobbyists:person"]

# Heterogeneous corpora: per-source field maps (first matching glob wins; `*` crosses `/`).
# [[mapper.sources]]
# match = "*/congress_press/*"
# record_id = "url"
# title_template = "{title}"
# text_fields = ["title", "text"]
# entity_fields = ["member.name:person"]
#
# [[mapper.sources]]
# match = "*/senate/*/filings*"
# record_id = "filing_uuid"
# entity_fields = ["registrant.name:org", "client.name:org"]

[parser]
pdf = "pdfium"                # -> pymupdf (richer, AGPL) | docling (OCR/tables)
docx = "docx"
text = "text"

[embedder]
enabled = true               # default-on Potion embeddings (CPU, no API)
name = "potion"              # -> sbert | voyage | openai | deepinfra | gemini
model = "minishlab/potion-base-8M"
revision = "bf8b056651a2c21b8d2565580b8569da283cab23"  # pinned weights → reproducible index
contextual = false           # upgrade seam: prepend generated context before embedding

[store]
backend = "sqlite-fts5"      # keyword (FTS5) + vectors (sqlite-vec), all in .muck/index.db

[chunk]
size_tokens = 320
overlap_tokens = 64
token_estimate_factor = 0.75

[cluster]
enabled = true
k = "auto"

[entities]
resolve = true

[reranker]
enabled = false              # -> true with name = "cross-encoder" (needs --extra sbert)
name = "cross-encoder"
"""


def fields_for(settings: Settings, path: str | Path) -> FieldMap:
    """Resolve the field map for a source file: first matching [[mapper.sources]] glob wins.

    Globs are fnmatch-style (``*`` crosses ``/``); relative patterns are treated as
    path suffixes (auto-prefixed ``*/``). Falls back to the global ``[mapper.fields]``.
    """
    p = Path(path).resolve().as_posix()
    for src in settings.mapper.sources:
        pat = src.match if src.match.startswith(("/", "*")) else "*/" + src.match
        if fnmatch(p, pat):
            return src
    return settings.mapper.fields


def find_muck_dir(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (or cwd) looking for a ``.muck`` directory."""
    cur = (start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        candidate = d / MUCK_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def load_settings(muck_dir: Path) -> Settings:
    cfg_path = muck_dir / CONFIG_FILENAME
    if not cfg_path.exists():
        return Settings()
    with cfg_path.open("rb") as fh:
        data = tomllib.load(fh)
    return Settings.model_validate(data)


def write_default_config(muck_dir: Path) -> Path:
    cfg_path = muck_dir / CONFIG_FILENAME
    cfg_path.write_text(DEFAULT_CONFIG_TOML)
    return cfg_path
