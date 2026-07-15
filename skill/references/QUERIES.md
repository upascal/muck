# Reusable investigation patterns

Concrete recipes that generalize past any single corpus. Each ends in verifiable citations
or reproducible SQL. Run `muck COMMAND --help` for full options.

## 1. Entity-resolution sweep
Find who actually matters, with messy-name variants already merged.
```
muck entities --type org --min-count 2          # most-mentioned organizations
muck entities --type person --min-count 2       # most-mentioned people
muck entities --entity <id>                      # aliases + every cited mention + co-occurring entities
```
The `aliases` list shows what was merged (e.g. "Acme Strategies LLC" ≡ "Acme Strategies,
L.L.C."); `mention_count` vs `doc_count` distinguishes prolific filers from broadly-named ones.

## 2. Money-flow cross-reference
Who spends the most, with spending folded onto the resolved entity (not raw name variants).
```
muck aggregate --by registrant --measure amount --agg sum --resolve org -k 25
muck aggregate --by client     --measure amount --agg sum --resolve org -k 25
```
Then drill into a top entity's relationships: `muck entities --entity <id>` shows the clients
/ bills / people it co-occurs with. Verify any single filing with `muck cite <token>`.

`aggregate` runs on **DuckDB** (over the read-only SQLite store) when the `analytics` extra is
installed, else SQLite json1 — same results, faster at scale. With `--sql` on DuckDB you can
**join an outside dataset** (exact-SQL, complementing the probabilistic `reconcile --link`):
```
muck aggregate --engine duckdb --sql "SELECT d_reg, SUM(f.amount) FROM \
  (SELECT json_extract_string(structured_json,'\$.registrant') d_reg FROM muck.documents) d \
  JOIN read_csv_auto('fec_contributions.csv') f ON lower(f.org)=lower(d.d_reg) GROUP BY d_reg"
```

## 3. Timing-anomaly scan
Spot spikes and suspicious sequencing.
```
muck aggregate --by filing_period --agg count            # volume per period
muck aggregate --sql "SELECT json_extract(structured_json,'$.filing_period') AS period, \
  SUM(CAST(json_extract(structured_json,'$.amount') AS REAL)) AS spend \
  FROM documents GROUP BY period ORDER BY period"
```
Cross with events from press releases: `muck search "<bill or event>"`, read dates, compare
to the spend timeline. Cite the filing and the press release for any timing claim.

## 3a. Peer-outlier / rarity scan (leads you wouldn't think to query)
Surface records that *deviate from their peers* — the tool proposes the hypothesis, you don't.
```
muck anomalies                                          # income far from each firm's OWN median
muck anomalies --by client.name --measure income        # a client paying one firm unlike its others
muck anomalies --method iqr    --threshold 1.5          # cross-check with an IQR fence
muck anomalies --space linear                           # raw-dollar (default is log: "10x the median")
muck anomalies --mode rare -k 20                         # one-shot whales: a client seen once for a huge sum
```
Default peer mode scores each filing with a **log-space robust modified z-score** against its
own `registrant.name × filing_type` baseline (entity-folded via `--resolve`, so name variants
share one baseline; `--min-peers 5`). Log space makes *low-side* outliers visible too — a firm
of six-figure engagements filing one tiny report. Every flagged record carries a resolvable
whole-document `citation_token` **and** a `query` block (params + peer stats). Watch the built-in
context: `filing_type` exposes the Q1-vs-Q1Y period trap; `client.name` (in `context`) is often
the actual lead. Run `muck cite <token>` / `muck verify` before asserting.

Note on evidence: the *record* is hash-verified by its token; the outlier *figure and score* are
**query-cited** (reproducible), because a `structured_json` field has no char span — promote it
into the field map's `text_fields` (a re-index) if you need the number itself span-citable.

## 4. Revolving-door detector (firm ↔ government)
Surface people connected to both private firms and government offices.
```
muck entities --type person --min-count 2                # candidates
muck entities --entity <person_id>                       # which orgs they co-occur with
muck grep "(formerly|previously|chief of staff|served as|now at|joined)" --all   # role-change language, every match
```
A person whose `co_occurring` orgs include both a registrant/firm and a government office,
or whose mentions span a firm filing and a government press release, is a lead. Confirm each
leg with `muck verify` before asserting a move.

## 5. Entity-filtered semantic search (person/org = X AND cosine to Y)
Find chunks that **mention a specific resolved entity** and rank them by similarity to a
query — the structured filter + vector query in one step.
```
muck search "prescription drug pricing" --person "Jane Doe"        # her chunks, cosine-ranked
muck search "defense procurement"        --org "Acme" --min-score 0.3   # threshold on cosine
muck search "" --entity <entity_id>                                # all chunks for an entity, no query
```
Restricts to the entity's chunks (merged aliases + cross-source mentions included), then
ranks by true cosine to the query; `--min-score` filters by similarity. Each hit keeps its
citation token. Falls back to keyword ranking within the entity's chunks if embeddings are off.

## 6. Reconcile messy records / link to outside data (Splink)
When the same firm files under drifting names/addresses, or you bring in an outside dataset,
use probabilistic multi-field linkage (opt-in: `uv sync --extra splink`).
```
muck reconcile --on registrant --compare address,zip                 # dedupe the corpus
muck reconcile --on registrant --compare address,zip --link fec.json # link to an outside dataset
muck reconcile --show                                                # the last run's clusters
```
Name-dominant by design (records that share an address but not a name do **not** merge), and
fully audited: every cluster, the pairwise match probabilities, and the method + threshold are
stored (`reconcile_runs` / `reconcile_records` / `reconcile_links`). Raise `--threshold` for
precision; a linkage finding should cite the match probability. Use this for cleaner spend
totals (aggregate by reconciled cluster) and for cross-dataset "who is also in X" leads.

## 7. Undisclosed-relationship cross-reference
Two entities that co-occur heavily but where the relationship isn't stated.
```
muck entities --entity <org_id>                          # read top co_occurring weights
muck search "<orgA> <orgB>"                              # passages mentioning both
```

## Always
- Quote verbatim + `muck verify --token … --quote "…"` before asserting.
- `muck finding add …` for each claim, then `muck audit` before presenting.
- Aggregate findings cite the command + row count as evidence (reproducible).
- `anomalies` findings: cite the record's whole-doc `citation_token` (`muck finding add` verifies
  it) plus the `query` block; the deviation is reproducible, the figure itself is query-cited.
