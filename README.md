# muck

A modular, citation-verified pipeline for investigating large document corpora with an AI
agent — packaged as an **Agent Skill**. Point it at a folder of JSON records (e.g. lobbying
filings, press releases) or PDFs; it parses, indexes, resolves entities, and searches
**locally**, so the agent reads only relevant, source-tied snippets instead of every token.

The default path needs **no API key and no GPU**: local Potion embeddings (model2vec) +
SQLite (FTS5 keyword + sqlite-vec vectors). Swap in API embedders (OpenAI/Voyage/DeepInfra/
Gemini), heavier parsers (PyMuPDF, Docling OCR), or a cross-encoder reranker via one config
edit — see `skill/references/ADAPTERS.md`. Vectors persist in SQLite; for very large corpora
turbovec is the opt-in in-memory ANN accelerator.

## Install (for evaluators)

muck is **two things**: the `muck` **CLI** (a command) and the Claude **skill** (a directory the
agent loads). One script installs both:

```bash
./install.sh
```

…or manually, from this folder (the unpacked submission — **no GitHub access needed**):

```bash
# 1) the CLI  → puts `muck` on PATH (clean, non-editable — no dev setup, no `--no-editable`)
uv tool install .                                         # installs from these local files
# 2) the skill → Claude Code discovers it by location
cp -R skill ~/.claude/skills/muck
```

> If the repo is public, `uv tool install git+https://github.com/upascal/muck@v1.0.0` also works.

Verify: `muck --version` (→ `1.0.0`) and `python skill/scripts/install_check.py`.

## Reproduce the index (one command, ~90s)

Evaluators supply the corpus. Point muck at it with the checked-in field map, then build:

```bash
mkdir -p run/corpus/.muck && cp configs/corpus_q1.toml run/corpus/.muck/config.toml
cp -R <the challenge corpus> run/corpus/data          # senate/… + congress_press/…
cd run/corpus && muck build data                      # 35,987 docs / 110,359 chunks in ~90s
```

`muck build` runs the whole pipeline (ingest → map → parse → index) and reports per-stage timing;
the index is reproducible (content-addressed doc_ids/citation tokens, a pinned embedding model, and
`muck status` records the version/model/config hash). Then investigate — start from
`skill/references/PLAYBOOKS.md`:

```bash
muck anomalies                                    # records that deviate from their peers
muck aggregate --by activity_desc --agg count     # recycled boilerplate / filing mills
muck search "who lobbied on H.R. 1234"
muck verify --token <token> --quote "<verbatim quote>"   # the hallucination guard
```

> Dev note: to edit the source in place, use `uv sync --no-editable --reinstall-package muck`
> (a uv/`_virtualenv.pth` editable-install quirk). Evaluators don't need this — the clean install
> above just works from any directory.

## Submission map

- **Agent Skill** — `skill/` (`SKILL.md` + `scripts/` + `references/`). The primary
  artifact: the reusable investigation workflow. Drop `skill/` into `~/.claude/skills/muck/`
  (or a project `.claude/skills/`) to use it elsewhere.
  - `references/ADAPTERS.md` — swap parser/embedder/store/reranker; add an adapter.
  - `references/QUERIES.md` — entity-resolution sweep, money-flow cross-ref, timing-anomaly
    scan, revolving-door detector.
  - `references/CITATIONS.md` — how citation tokens + `verify`/`audit` make claims auditable.
  - `references/SCHEMA.md` — the database, for human auditors.
- **Code** — `src/muck/` (CLI + pluggable stages). `pyproject.toml` defines the tiny core
  and optional extras.
- **Tests** — `tests/` (`uv run --no-sync pytest -q`): end-to-end pipeline, the citation
  guard (real vs fabricated vs tampered), entity resolution + cross-source linking, resolved
  aggregation, clustering, and adapter-seam degradation.
- **Findings report / interaction traces** — produced by running the skill on the corpus;
  traces come from `muck trace` (per-tool audit log) keyed to the session.

## How it scores against the rubric
- **Organized across sessions:** persistent `<corpus>/.muck/index.db`; `muck status` resumes.
- **Efficient with the corpus:** deterministic parse/index/aggregate/cluster do the heavy
  lifting; the agent reads only retrieved snippets. Analysis runs on **DuckDB** (over the
  read-only SQLite source of truth) when available — fast columnar aggregation that can also
  join external CSV/JSON/Parquet — with a SQLite json1 fallback.
- **Human-verifiable:** every hit/mention carries a tamper-evident citation token;
  `muck verify`/`audit` re-derive evidence from source. See `references/CITATIONS.md`.
- **Extends the agent:** a deterministic entity resolver, cross-reference/network traversal,
  exact resolved aggregation, topical clustering, optional **Splink record reconciliation**
  (messy multi-field dedupe + cross-dataset linkage, fully audited), entity-filtered semantic
  search, and an enforced citation-audit framework — all reusable on other investigations.
