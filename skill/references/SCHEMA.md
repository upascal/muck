# Database schema (`.muck/index.db`)

One embedded SQLite file holds the whole investigation state. It persists across sessions
(in Claude Code), so `muck status` re-orients the agent on resume. All provenance joins are
first-class SQL, which is what makes findings auditable.

| Table | Purpose | Key columns |
|---|---|---|
| `source_files` | Manifest of ingested files; dedup by hash; resumable | `file_id`, `path`, `content_hash`, `file_type`, `status` |
| `documents` | One logical doc (a JSON record or a whole file) | `doc_id`, `source_path`, `locator` (JSON pointer/page), `doc_type`, `text`, `structured_json`, `raw_json`, `pages_json`, `status` |
| `chunks` | Retrievable windows with provenance | `chunk_id`, `doc_id`, `char_start`, `char_end`, `locator`, `text` |
| `chunks_fts` | FTS5 keyword/BM25 index over chunk text | (virtual) `text`, `chunk_id` |
| `chunks_vec` | sqlite-vec ANN index (created when embeddings on) | `chunk_id`, `embedding float[D]` |
| `entities` | Canonical resolved entities | `entity_id`, `canonical_name`, `entity_type`, `norm_key`, `mention_count`, `doc_count` |
| `entity_aliases` | Surface forms merged into each entity | `entity_id`, `alias`, `source` |
| `mentions` | Each occurrence, with exact offsets (→ citation) | `entity_id`, `doc_id`, `chunk_id`, `raw_text`, `char_start`, `char_end` |
| `entity_edges` | Co-occurrence graph (same document) | `src_entity_id`, `dst_entity_id`, `weight`, `edge_type` |
| `clusters` / `chunk_clusters` | Topical index (KMeans + c-TF-IDF labels) | `cluster_id`, `label`, `terms_json`, `size` |
| `findings` | Claim + verifiable citation, audit status | `finding_id`, `claim`, `quote`, `citation_token`, `status`, `audited_at` |
| `observations` | Investigative working memory (notes/decisions/leads/threads) | `obs_id`, `kind`, `status`, `text`, `entity_ids`, `doc_ids`, `tokens` |
| `trace_log` | Append-only audit trail of every CLI call | `ts`, `session_id`, `command`, `args_json`, `n_results` |
| `index_meta` | Index version, embedder name/dims, config hash | `key`, `value` |

## The three memory layers
muck separates investigative memory so each is queryable on its own terms:
- **`trace_log`** — mechanical: every CLI call. *What the tools did.*
- **`findings`** — verified claims + citation + audit status. *What you can stand behind.*
- **`observations`** — curated working memory: notes, decisions/rationale, leads, hypotheses,
  open threads, dead ends, each with a `status` and links to entities/docs/tokens. *Where the
  investigation is and what's next.* `muck status` surfaces open threads; `muck recall` searches
  observations + findings (your memory), as distinct from `muck search` (the corpus).

## The citation primitive
A citation resolves through `documents` (and the original file) via `(doc_id, char_start,
char_end)`. Because `documents.text` is rendered deterministically (JSON) or re-parsed
deterministically (PDF), `muck cite`/`verify` can re-derive any span from source — see
`CITATIONS.md`.

## Auditing by hand
```sql
-- everything the agent searched for, in order
SELECT ts, command, args_json, n_results FROM trace_log ORDER BY id;
-- every claim and whether it last verified against source
SELECT status, claim, citation_token FROM findings;
-- which surface forms were merged into one entity
SELECT canonical_name, alias FROM entities JOIN entity_aliases USING(entity_id)
  WHERE entity_type='org' ORDER BY canonical_name;
```
Open the DB read-only: `sqlite3 <corpus>/.muck/index.db`.
