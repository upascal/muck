# Swapping components

Every pipeline stage sits behind a `Protocol` + name→factory registry
(`src/muck/interfaces/`). You change a stage by editing `.muck/config.toml`; you add a new
implementation by dropping one file that calls `register_*(...)` and naming it in the config.
Optional-dependency adapters raise a friendly `NotInstalled` (never a bare `ImportError`),
so the default install stays clean.

## The config (`.muck/config.toml`)

```toml
[mapper]
json = "json"
[mapper.fields]                 # fill these from the corpus manual — the key setup step
record_path = "/results"        # JSON pointer to the list of records ("" / omit = auto-detect)
record_id   = "filing_uuid"
title_template = "{registrant} - {client} ({filing_period})"
structured_fields = ["registrant","client","amount","filing_period","filed_date"]  # for aggregation
entity_fields     = ["registrant:org","client:org","lobbyists:person"]              # field:type

[parser]   pdf = "pdfium"       # -> pymupdf (richer, AGPL) | docling (OCR/tables)
[embedder] name = "potion"      # -> sbert | openai | voyage | deepinfra | gemini ; enabled=false for keyword-only
           contextual = false   # prepend doc context before embedding (citations stay raw)
[store]    backend = "sqlite-fts5"  # keyword (FTS5) + vectors (sqlite-vec) in one DB; -> lancedb
[reranker] enabled = false  name = "cross-encoder"   # precision boost (needs --extra sbert)
[entities] resolve = true
```

`entity_fields` are **always rendered into the document text**, even if `text_fields` omits
them — mentions are found by scanning that text, so a name that never appears there would
yield no entity and no citable span.

## Heterogeneous corpora: per-source field maps

One field map can't serve two schemas (`record_id`/`title_template` fit only one). Add a
`[[mapper.sources]]` block per schema — **first matching glob wins**, falling back to the
global `[mapper.fields]`. `muck peek <dir>` prints these blocks ready to paste.

```toml
[[mapper.sources]]
match = "*/congress_press/*"          # fnmatch; `*` crosses `/`, so this matches any depth.
record_id = "url"                     # A pattern without a leading `/` or `*` is treated as
title_template = "{title}"            # a path suffix (auto-prefixed `*/`).
text_fields = ["title", "text"]
entity_fields = ["member.name:person"]   # dotted paths address nested objects

[[mapper.sources]]
match = "*/senate/*/filings*"         # `*` also stands in for the year directory
record_id = "filing_uuid"
entity_fields = ["registrant.name:org", "client.name:org"]
```

Editing any source map changes the field-map hash, so `muck status`/`index` will tell you to
re-run `muck map --all` (values are baked in at *map* time).

## Nested arrays (multi-valued fields) — for the aggregation/anomaly playbooks

Dotted paths reach into nested *objects* (`registrant.name`). For nested **arrays** — a record
holding a *list* of sub-records — use **`field[].sub as alias`**: `[]` iterates the array and
`as alias` names the resulting (multi-valued) column. Aggregations **UNNEST** it, so each element
counts once per record. `muck peek` shows scalar leaves but does **not** suggest these array-paths —
add them by hand from the corpus manual.

```toml
structured_fields = [
    "income", "registrant.name",
    "lobbying_activities[].general_issue_code_display as issue_code",         # one array level
    "lobbying_activities[].government_entities[].name as government_entity",   # arrays inside arrays
    "lobbying_activities[].lobbyists[].covered_position as covered_position",
    "lobbying_activities[].description as activity_desc",
]
```

Then `muck aggregate --by government_entity --agg count` counts every body each filing lobbied, and
`muck aggregate --by activity_desc --agg count` surfaces recycled boilerplate. Values are de-duped
per record (a filing listing the same agency three times counts once), so use array-paths for
*categorical* fields; for a sum that must not de-dup (e.g. many line-item amounts), aggregate the
scalar or drop to `--sql`. **The `PLAYBOOKS.md` archetypes assume these fields are mapped** — without
them, the who's-lobbied / boilerplate / revolving-door playbooks have nothing to query.

**Array-paths also work in `entity_fields`** — declare a nested actor and it resolves as an entity
(and renders into the searchable text) just like a flat field:

```toml
entity_fields = ["contribution_items[].honoree_name:person"]   # each honoree -> a person entity
```

This is what makes an actor resolvable *across sources*: an LD-203 honoree and a press member fold
to one entity, so `muck entity "<name>"` / `muck entities --entity <id>` show that person's words,
money, and lobbying together. Person names are normalized to merge titled/formal variants (`The
Honorable U.S. Senator Angus S. King, Jr.` → `angus king`), so cross-source linking survives the
honorifics and middle initials that filings add.

### Typed relations (the directed graph — who → who)

Beyond co-occurrence, declare **directed relations between two resolved entities** with
`relations`, read from structured fields: `"src_field:type -> dst_field:type : predicate"`. Both
endpoints must also be `entity_fields` (so they resolve); `src_field`/`dst_field` name the
`structured_json` keys (aliases).

```toml
# filings: the registrant (firm) lobbies for the client
relations = ["registrant.name:org -> client.name:org : lobbied_for"]
# contributions: the registrant's PAC donated to the honoree (member) — the say-vs-pay edge
relations = ["registrant.name:org -> honoree:person : donated_to"]
```

`muck build` turns these into a directed, **cited** `entity_relations` table (each edge derives from
a specific record, so it carries a representative `doc_id` + citation token). Then `muck relations`
lists the strongest edges (`--predicate donated_to`), and `muck entity "<name>"` shows one entity's
`relations.outgoing` / `relations.incoming` grouped by predicate — each a lead you *verify* by
citing its record (the token feeds straight into the findings/`audit` layer). Unlike the symmetric
`co_doc` co-occurrence hints, these are ground truth from the data, not inference. `muck relations
--rebuild` re-derives them on an existing index without re-embedding.

### Citable vs aggregatable — and how to get both (no pre-flattening)

A field in `text_fields` is rendered into the document's searchable text, so it is **citable**
(`verify` can quote it); a field in `structured_fields` lands in `structured_json`, so it is
**aggregatable** but has no text span to cite. Crucially, **`text_fields` flattens nested content**:
a *plain* nested key (no `[]`) renders **every scalar leaf beneath it — arrays included** — into
citable text. So to make nested story-content (covered positions, foreign entities, activity
descriptions) **both** citable and aggregatable, list the plain parent key in `text_fields` and the
`field[].sub as alias` paths in `structured_fields`:

```toml
text_fields       = ["lobbying_activities", "foreign_entities"]                # citable (auto-flattened)
structured_fields = ["lobbying_activities[].description as activity_desc",     # aggregatable
                     "foreign_entities[].name as foreign_entity"]
```

The `[]` array-path syntax works **only** in `structured_fields`; in `text_fields` (and
`entity_fields`) it renders nothing — use the **plain key** there. **Do not pre-flatten or rewrite
the source data** to make nested fields citable — muck already flattens them, and rewriting the
corpus breaks reproducibility (the evaluator builds from the original files).

## Entity matching knobs (`[entities]`)

Gazetteer matching defaults are tuned for precision on English org names; change them per corpus:

```toml
[entities]
resolve = true
min_name_len = 3               # names shorter than this never free-text match ("IB", "EU")
single_word_match = "exact-case"  # exact-case | any-case | off
extra_org_suffixes = ["gmbh", "sa", "ab"]   # folded like LLC/Inc when resolving names
```

`single_word_match` exists because single-word org names are often ordinary words — a client
literally named "Clear" or "Scale" matches prose everywhere if matched case-insensitively.
`exact-case` (default) requires the casing the name was registered with; `any-case` maximizes
recall; `off` disables free-text matching of single-word names entirely. Multi-word names are
always case-insensitive. `extra_org_suffixes` extends the built-in US/English legal forms
(LLC, Inc, Ltd…) — set it and `--resolve` folding and the entity index stay consistent.

## XML sources

`.xml` files route to the `xml` mapper automatically. One record per file by default (House LDA
ships one filing per file). For bulk-export XML that packs many records into one file, set
`record_path` to an ElementTree path — each match becomes its own document with its own citation:

```toml
[[mapper.sources]]
match = "*/filings/*"
record_path = "filing"        # or ".//filing"; each element -> one document
record_id = "filingId"
entity_fields = ["organizationName:org", "clientName:org"]
```

## Built-in adapters and their extras

| Stage | Default (core) | Upgrades (extra) |
|---|---|---|
| mapper | `json`, `xml` | — |
| parser | `pdfium`, `text`, `transcript` | `docx` (`--extra docx`), `pymupdf` (`--extra pymupdf`), `docling` (`--extra docling`, whole-doc OCR — not page-precise) |
| embedder | `potion` (model2vec) | `sbert` (`--extra sbert`); `openai`/`voyage`/`deepinfra`/`gemini` (`--extra api` + API key) |
| store | `sqlite-fts5` (FTS5 + sqlite-vec — the single source of truth) | turbovec in-memory ANN accelerator for large corpora (`--extra turbovec`) |
| reranker | off | `cross-encoder` (`--extra sbert`) |

API embedders read the key from the env var named by `api_key_env` (provider defaults:
`OPENAI_API_KEY`, `VOYAGE_API_KEY`, `DEEPINFRA_API_KEY`, `GEMINI_API_KEY`).

**Record reconciliation** is a separate opt-in capability (not a stage adapter):
`uv sync --extra splink` enables `muck reconcile` — probabilistic multi-field linkage over
structured records (dedupe or `--link` an external dataset). The deterministic mention
resolver (normalized-key) stays the always-on default; reconcile is for messy multi-field /
cross-dataset cases and writes its own audited tables. See `QUERIES.md` §6.

## Scanned / image-only documents (a separate capability, not a parser swap)

A PDF with no text layer indexes to nothing on the default parser. Rather than swap in a
lossy whole-document OCR parser, muck adds a page-precise pipeline that keeps provenance:

- `muck coverage` — per-file text coverage + verdict (native | sparse | image_only).
- `muck render` — rasterize pages to `.muck/pages/<file_id>/page_NNNN.png` (pypdfium2, a
  core dep — no Pillow/poppler). These are the pixel ground truth for review.
- `muck ocr` — deterministic OCR (needs the **`tesseract`** binary on PATH, not a Python
  package: `brew install tesseract` / `apt install tesseract-ocr`). Writes a transcript
  sidecar and re-extracts; citations become tier `ocr`.
- `muck transcribe` — the vision route: muck renders, *the agent* reads the page images and
  writes per-page transcripts, `muck transcribe import` ingests them as tier `vision`. muck
  never calls a vision model itself, so the no-API-key/default path is unchanged.

Transcripts live in `.muck/transcripts/<file_id>.json` (evidence, not cache) and are re-read
by the `transcript` parser, so `verify` stays deterministic and `muck parse --all` never
clobbers an OCR/vision result. `docling` still exists but is not page-precise (citations
resolve to the whole document); prefer `muck ocr` for scanned PDFs. See `CITATIONS.md` for
what verification proves at each tier.

## Adding an adapter (example: a new embedder)

```python
# src/muck/embedders/my_embedder.py
from ..interfaces.embedder import register_embedder

class MyEmbedder:
    name = "mine"
    requires_api_key = False
    def configure(self, cfg): ...                 # optional: read model / api_key_env
    @property
    def dim(self) -> int: return 768
    def embed(self, texts, batch_size=128): ...   # -> np.ndarray (N, dim), L2-normalized

register_embedder("mine", lambda: MyEmbedder())
```

Import it in `src/muck/embedders/__init__.py`, then set `[embedder] name = "mine"`.
The same shape applies to `register_parser`, `register_store`, `register_reranker`,
`register_mapper`.

## Storage model & the vector seam

**SQLite (`.muck/index.db`) is the single source of truth** — relational tables + FTS5
keyword + sqlite-vec vectors, all in one file. There is no second persistent store.

For very large corpora, **turbovec** is an opt-in ANN accelerator — enable with
`uv sync --extra turbovec` and `[store] backend = "turbovec"`. sqlite-vec stays the source of
truth; turbovec builds a quantized index *from* the stored vectors and caches it at
`.muck/cache/turbovec.tv`, keyed to a fingerprint of the index so it **auto-rebuilds when
stale** — no second store to hand-sync. It accelerates general semantic search (with native
`allowlist` filtering); keyword and entity-filtered search stay exact via SQLite. Needs a
prebuilt wheel (arm64-mac / manylinux) and an embedding dim that's a multiple of 8 (Potion's
256 qualifies).

To add a different vector engine, subclass `SqliteStore` (keeping FTS5/BM25/grep) and override
the vector methods (`supports_vectors`, `upsert_vectors`, `search_vector`, `get_vectors`), then
`register_store("name", lambda: MyStore())` and set `[store] backend = "name"`.
