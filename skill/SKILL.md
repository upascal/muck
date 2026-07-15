---
name: muck
description: >-
  Investigate a large corpus of documents (JSON records like lobbying filings or press
  releases, plus PDF/DOCX/TXT) without reading every file. muck builds a persistent,
  citation-verified index so you retrieve only the relevant page/field-level snippets,
  resolve and cross-reference entities, run exact aggregations, and tie every claim to a
  verifiable source. Use when pointed at a folder of many documents and asked to find,
  cross-reference, verify, or quantify entities, relationships, money, or timing across
  them — and whenever a finding must be auditable.
allowed-tools: Bash, Read, Grep, Glob
---

# muck — investigate many documents without reading every token

You drive the deterministic `muck` CLI, which parses, indexes, resolves entities, and
searches **locally**. The CLI does the heavy lifting; you read only retrieved snippets and
**cite every claim to a verifiable source**. State persists in `<corpus>/.muck/index.db`,
so an investigation stays organized across sessions.

Run commands as `muck …` (installed console script) or, if not on PATH,
`python -m muck …`. Every command prints JSON (except `init`). Run `muck COMMAND --help`
for options.

## When to use
A folder of many documents + a question that requires finding, cross-referencing,
verifying, or quantifying facts across them (who-paid-whom, undisclosed ties, timing
anomalies, spending patterns, revolving-door moves). **Do not read the documents
directly** — let muck retrieve and aggregate.

## Setup (once per corpus)
1. Install: `uv sync --no-editable` (add `--extra analytics` for `aggregate`/`cluster`).
   Then `python skill/scripts/install_check.py` reports which path is active (no API key
   or GPU is required — Potion embeddings and SQLite run locally).
2. `muck init <corpus_dir>`
3. **Calibrate, then configure.** Don't guess the schema — look at it. `muck peek <a file>`
   reports the record array, field names/types (nested objects expanded to dotted names like
   `member.name`), sample records, and a *suggested* field map. **`muck peek <the corpus dir>`**
   groups files by schema and prints one ready-to-paste `[[mapper.sources]]` block per schema —
   use that for a **multi-source corpus** (press releases + filings have different shapes, and one
   global map can't serve both). Adapt into `<corpus_dir>/.muck/config.toml`; see
   `references/ADAPTERS.md`. This makes citations field-precise and entity resolution accurate.
   For records with **nested arrays** (e.g. a filing holding a list of `lobbying_activities`),
   map the sub-fields you'll aggregate on with the array-path syntax `field[].sub as alias`
   (see ADAPTERS.md) — `muck peek` won't suggest these, but the `references/PLAYBOOKS.md`
   aggregation/anomaly archetypes depend on them.
   After indexing, `muck sample` (optionally `--query "<topic>"`) loads a token-budgeted batch of
   full records so you can get a feel for the data and form hypotheses before running targeted
   queries.
4. `muck ingest <corpus_dir>` → `muck map` (JSON) and/or `muck parse` (PDF/DOCX/text) →
   `muck index` → (optional) `muck cluster`. These are idempotent — re-run anytime to pick
   up new files; interrupted runs resume.

If you edit `[mapper.fields]` after the first run, re-run `muck map --all` then
`muck index --all` to apply it (field values are produced at *map* time). `muck status`/`index`
surface a `config_drift` message when the field map changed but hasn't been re-mapped.

## Scanned / image-only documents (PDFs with no text layer)
muck's default parser reads a PDF's **text layer**. A scanned or photographed document has
none, so it would index to nothing. muck now refuses to hide this: `parse`/`index`/`status`
emit a `coverage_notice` when parsed docs have little extractable text. **When you see it,
do not treat empty search results as evidence of absence** — the corpus hasn't been read yet.
1. **Diagnose:** `muck coverage` reports, per file, `coverage` (fraction of pages with text)
   and a `verdict` (native | sparse | image_only) with a `recommend`.
2. **See the pixels:** `muck render` rasterizes pages to `.muck/pages/<file_id>/` — the
   ground truth a human (or you) checks a claim against.
3. **Deterministic OCR** (typed/printed scans): `muck ocr` (needs the `tesseract` binary)
   renders + OCRs each page into a transcript sidecar, then `muck index`. Citations become
   tier `ocr`: verifiable against the transcript, with the page image attached. Low-confidence
   pages are flagged — route those to vision instead.
4. **Vision transcription** (handwriting, hard scans): muck does **not** call a vision model;
   *you* transcribe. `muck render --file <pdf>`, then read each `.muck/pages/<file_id>/
   page_NNNN.png` with your **Read** tool and write `.muck/transcripts/<file_id>/page_NNNN.md`,
   then `muck transcribe import --file <pdf> --model <your-model-id>` (`muck transcribe status`
   shows the queue). Text becomes tier `vision`.
5. **Structure it (optional):** to turn a transcript into typed records, write the extracted
   JSON and `muck ingest` it as a JSON source (calibrate with `muck peek` → `[mapper.fields]`),
   exactly like any structured corpus — don't hand-roll a parallel path.

## Investigation loop
- **Resume / orient:** `muck status` shows what's indexed *and your open threads* (open
  leads/hypotheses/questions). On a new session, start here, then `muck recall "<topic>"` or
  `muck note list --status open` to pick up where you left off — don't re-derive state.
- **Record as you work (memory):** `muck note add --text "…" --kind lead|decision|hypothesis|
  question [--entity <id>] [--token <cite>]` — capture leads, the *why* behind a choice
  (`decision`), and threads. Mark threads `--status cold` when they dead-end, `confirmed` when
  they pan out (a confirmed lead becomes a `finding`). `muck recall "<query>"` searches *your
  memory* (notes + findings); `muck search` searches *the corpus* — keep them distinct.
- **Generate leads (what to look for):** when you don't already know what to hunt for, start from
  `references/PLAYBOOKS.md` — reusable story archetypes (revolving door, red-flag disclosures,
  foreign influence, peer anomalies, one-shot whales, say-vs-pay, undisclosed relationships,
  timing, concentration, follow-the-money), each with a paste-ready prompt and the deterministic
  `muck` hunt behind it. Run several early; treat every hit as a lead to verify, not a finding.
- **Find:** `muck search "<question>"` (auto = hybrid keyword+semantic; degrades to keyword
  if embeddings are off). Use `muck grep "<regex>"` for exact strings (bill numbers, IDs);
  literal spaces in the pattern match across line breaks in PDF text. grep returns
  `{total_matches, returned, truncated, hits}` — when `truncated` is true you have NOT
  seen every match. Snippets are previews, not evidence: before dismissing a hit, read its
  `citation.quote` (the full chunk).
- **Enumerate:** for "list all X / which countries…" questions, never bound the answer
  with ranked top-k results — run `muck grep "<term>" --all`, group hits by document, and
  read every one. Absence from a top-k list is not evidence of absence. And check the
  corpus's *own* vocabulary before concluding an event never happened (a corpus may say
  "suspension" where you'd say "termination" or "expulsion").
- **Read in full:** once you've narrowed to relevant `doc_id`s, `muck read <id> <id> …` returns
  the **whole records side by side** (token-budgeted), each with a whole-document citation token.
  Use this — not repeated chunk reads — when you need full context to compare records, spot
  patterns, or read a finding's source. (Searching/aggregating still does the heavy lifting at
  scale; `read` is for the handful you've zeroed in on.)
- **Entities:** `muck entities --type org --name "Acme"` to find a resolved entity, then
  `muck entities --entity <id>` to see every mention (each with a citation token) and the
  entities it co-occurs with — your cross-reference / network view.
- **Entity + similarity:** `muck search "<query>" --person "X"` (or `--org`, `--entity <id>`)
  restricts to chunks mentioning that resolved entity and ranks them by cosine to the query;
  add `--min-score` to threshold. This is the "person = X AND semantically about Y" query.
- **Quantify:** `muck aggregate --by registrant --measure amount --agg sum --resolve org`
  (exact SQL totals, folded onto resolved entities), or `muck aggregate --sql "SELECT …"`.
  Runs on **DuckDB** over the read-only SQLite store when available (else SQLite json1).
  With `--sql` on DuckDB, the corpus is `muck.*` and you can join **external files**:
  `--sql "SELECT … FROM read_json_auto('fec.json') f JOIN muck.documents d ON …"`.
- **Reconcile / link (opt-in):** for messy multi-field records or linking to an outside
  dataset, `muck reconcile --on registrant --compare address,zip [--link other.json]`
  (Splink, `--extra splink`). Name-dominant + fully audited (scores, clusters, threshold);
  cite the match probability for any linkage claim. See `references/QUERIES.md` §6.
- **Topics:** `muck cluster` then read cluster labels to navigate themes.

## Citation discipline — REQUIRED
Every search hit and entity mention carries a **citation token** (`docid@start-end#hash`).
Before you assert any finding:
1. Quote the supporting text **verbatim** from `citation.quote` (the full chunk) — not
   from the `snippet`, whose `[`/`]` match markers and `…` ellipses break verification.
2. Run `muck verify --token <token> --quote "<verbatim quote>"`. State the finding only if
   `verified: true`. If `quote_supported` is false, you misread it — fix the quote or drop
   the claim. Never assert a claim you cannot verify; say "unsupported" instead.
3. Record it: `muck finding add --claim "…" --quote "…" --token <token>`.
4. Before presenting results, run `muck audit` — it re-verifies every finding against
   source and reports green/red. Drop or fix anything red.

For aggregate claims (totals, counts), cite the exact `muck aggregate` command and its row
count rather than a snippet — the query is the reproducible evidence.

**Text provenance — OCR/vision claims are not source-verified.** A citation on scanned text
carries `text_provenance` (native | ocr | vision), a `verification_scope`, and a `page_image`:
- `native` — `verify` proves the *document* says this; treat as before.
- `ocr` — `verify` proves the *deterministic OCR transcript* says this. OCR can misread
  characters (not fabricate sentences); confirm anything load-bearing against `page_image`.
- `vision` — the text was transcribed from pixels by an LLM, which **can fabricate**. `verify`
  returns `verified: false` / `needs_pixel_review: true` even when the quote matches, and the
  finding is `pending_review`. You MUST open the `page_image` (your Read tool) and confirm the
  claim, then `muck review <finding_id> --confirm|--reject`. `muck audit` buckets these under
  `pending_review` until reviewed. Never present a vision-tier finding as verified without it.

The audit trail (`muck trace`) records every tool call; keep it as part of the submission's
interaction traces.

## Extending muck
Swap the parser, embedder, vector store, or reranker, or enable API models — one config
edit + (sometimes) one adapter file. See `references/ADAPTERS.md`.

## Reusable investigation patterns
`references/QUERIES.md`: entity-resolution sweep, money-flow cross-reference, timing-anomaly
scan, and a revolving-door (firm↔government) detector. `references/CITATIONS.md` explains how
the verification guarantee works; `references/SCHEMA.md` documents the database for auditors.
