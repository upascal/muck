# muck backlog (post-v1.0.1)

Improvements surfaced during the GAIN challenge build + blind runs. None are in the frozen
`v1.0.1` submission tag; they land on `master` for a future `v1.0.2`.

## UX / clarity
- **Version in the `--help` header.** `--version` works but the header doesn't show it. One line:
  `@app.callback(help=f"muck {__version__} — …")` in `src/muck/cli.py` (import `__version__` at top).
- **Document `text_fields` flattening prominently.** Agents misread this 3× — concluding nested
  fields (`foreign_entities`, `conviction_disclosures`, `covered_position`) "aren't in the index"
  and either skipping findings or pre-flattening the corpus. The truth: with **no `text_fields`**
  (or a *plain* nested key in `text_fields`), muck flattens the whole record into searchable +
  citable text; the `[]` array-path syntax works **only** in `structured_fields`. Add a bold callout
  in `skill/SKILL.md` (configure step) and near the top of `skill/references/ADAPTERS.md`.
- **`muck peek` should suggest array-path fields.** `peek.py` `suggested_field_map` only emits
  scalar leaves; for arrays-of-objects it should propose `field[].sub as alias` so an agent
  designing a field map for a nested corpus gets the aggregation fields handed to it.
- **A `--dir <corpus>` flag** so muck isn't cwd-dependent (complements the v1.0.1 child-dir hint in
  `_resolve`). Lets an agent anchor without relying on the shell's cwd surviving interrupts.

## Performance (the 715k-doc corpus built in ~34 min → ~20 GB index)
- **Embeddings are the long pole.** Batch larger + multiprocess across cores in
  `potion_embedder.embed` / `pipeline.run_index` `flush_vectors`.
- **Drop `raw_json` bloat.** Storing the full original record per doc is a large share of the 20 GB;
  make it optional (`--no-raw` / config) or store only when needed for `muck read --raw`.
- **Cap / incrementally build entity edges.** Co-occurrence edges scale fast (36k docs → 765k edges;
  715k docs → 4.5M edges). Add a per-node degree cap or a `--no-edges` option for huge corpora
  (`extract/entities.py` `build_entities`).
- **A keyword-only fast build.** A `muck build --no-embed` flag (currently needs a config edit) so a
  reproducer can index a large corpus in minutes; the deterministic playbooks (anomalies/aggregate/
  boilerplate/grep) don't need embeddings.
- **Wire turbovec** (already an opt-in store seam) for ANN search at multi-million-chunk scale.

## Capability
- **Make structured figures span-citable.** Option to also render a chosen structured field (e.g.
  `income`) into `documents.text` so the *number* passes the citation guard, not just its record.
- **Cross-source entity join + Tier-3 relationship graph.** The deeper ceiling: wire the currently-
  orphaned Splink `reconcile` clusters into the query path, and give `entity_edges.edge_type` real
  typed predicates (only `'co_doc'` today) — enabling say-vs-pay, money-flow A→B→C, donor-is-client
  triangles that no single record contains.
