# Investigation playbooks — hypothesis generators

Retrieval and verification are the easy part; the hard part is deciding *what to look for*.
These are reusable **story archetypes** — the kinds of findings a reporter chases (undisclosed
relationships, revolving doors, timing anomalies, hypocrisy, follow-the-money). Each is a
paste-ready prompt plus the deterministic `muck` hunt behind it. Run several at the start of an
investigation to surface leads neither you nor the analyst pre-specified, then verify every
survivor with a citation (see `CITATIONS.md`).

**How to use:** pick the archetypes that fit the corpus, run the hunt, and treat each hit as a
*lead*, not a finding — confirm it with `muck read`/`muck verify` and record it with
`muck note add` / `muck finding add`. Every hunt below ends in a citation token or reproducible
query. The field names (`covered_position`, `government_entity`, `activity_desc`, `issue_code`, …)
come from the field map; run `muck peek <file>` first and substitute the corpus's own names.
**Several archetypes read fields nested inside arrays — map those first** with the array-path
syntax `field[].sub as alias` (see `ADAPTERS.md`); `muck peek` shows the raw structure but won't
suggest them, so a shallow field map leaves these playbooks with nothing to query.

---

## 1. Revolving door — an insider now working their former employer
**Story:** a lobbyist/official whose `covered_position` names a specific agency, committee, or
member is now lobbying that same body on behalf of a paying client.
**Prompt:** *"Find people whose prior-government-role field names an office they now lobby or act
before. For each, name the former role, the current client, and the body being lobbied."*
**Hunt:**
```
muck aggregate --by covered_position --agg count -k 40          # the pool of ex-gov roles
muck grep "(Chief of Staff|Counsel|Staff Director|Legislative Director) to (Sen|Rep|Committee)" --all
muck search "former <committee/agency> <role>" --org "<firm>"   # tie a person to a current client
```
Cross the `covered_position` text against the `government_entity` values on the same filing — a
match ("Senate Commerce counsel" → lobbies "SENATE") is the lead.
**Verify:** `muck read <doc_id>` the filing; quote the covered-position + the client + the target.
**Generalizes to:** any corpus with a prior-affiliation field (board members, auditors, ex-regulators).

## 2. Red-flag disclosures — actors carrying a legal/ethics flag
**Story:** a registrant/lobbyist with a **conviction disclosure** (or any self-reported red flag)
is actively lobbying — often for notable clients.
**Prompt:** *"List every record that carries a conviction/sanction/debarment disclosure, with the
actor, the client, and what the disclosure says."*
**Hunt:**
```
muck grep "conviction|convicted|pleaded|sentenced|debarred|sanctioned" --all
muck search "conviction disclosure"                              # semantic net for the field
muck read <doc_id>                                               # read the disclosure verbatim
```
**Verify:** quote the disclosure text + the actor name; `muck verify` the token.
**Generalizes to:** any disclosure/attestation field (bankruptcies, litigation, prior debarment).

## 3. Foreign influence — a foreign principal behind a domestic actor
**Story:** a domestic-looking client is funded by / acting for a foreign parent or government.
**Prompt:** *"Find records where a foreign entity is the ultimate principal, and name the domestic
front, the foreign owner, its country, and the issues lobbied."*
**Hunt:**
```
muck grep "foreign_entities|foreign entity" --all
muck search "foreign parent company owner government"
muck aggregate --by government_entity --agg count               # what a foreign-backed client targets
```
Filter to records whose `foreign_entities` (or a non-US country field) is populated.
**Verify:** quote the foreign entity + country + the domestic filer.
**Generalizes to:** any ownership/beneficial-owner or country-of-origin field.

## 4. Peer-relative anomaly — a record unlike its peers
**Story:** one record's number is wildly off its own group's baseline (10× its firm's median, or
suspiciously tiny) — a possible windfall engagement or under-reporting.
**Prompt:** *"Surface records whose <amount> deviates most from their <peer group>'s own norm, and
say which counterparty the outlier is for."*
**Hunt:**
```
muck anomalies                                                  # income vs each firm's own median
muck anomalies --by client.name --measure income                # a client paying one firm unlike others
muck anomalies --space linear --method iqr                      # cross-check
```
**Verify:** each flagged row carries a whole-doc `citation_token` + the reproducible query; the
`context` (e.g. `client.name`) is usually the lead.
**Generalizes to:** any numeric field with a natural peer key (price per vendor, grant per agency).

## 5. One-shot whale — a singleton with outsized magnitude
**Story:** a counterparty that appears exactly once, yet for a top-percentile sum — a one-time
blitz (an M&A fight, a single regulatory battle) rather than a routine relationship.
**Prompt:** *"Find counterparties that appear only once but carry a top-percentile amount."*
**Hunt:**
```
muck anomalies --mode rare -k 20                                # singletons in the top spend percentile
```
**Verify:** read the single record; confirm the amount + that it's a lone appearance.
**Generalizes to:** any actor×amount corpus (one-time mega-donors, single huge contracts).

## 6. Say-vs-pay — public words vs. private money
**Story:** an actor's public statements contradict their financial/lobbying behavior (rails
against an industry while taking its money, or lobbies against a bill they publicly champion).
**Prompt:** *"For a public figure, compare their press-release topics to the lobbying aimed at
their chamber/committee and the money they received, around the same period. Flag contradictions."*
**Hunt:**
```
muck search "<member> <issue>" --person "<member>"             # what they SAY (press releases)
muck aggregate --by government_entity --agg count               # what's lobbied at their body
muck aggregate --by <recipient> --measure amount --agg sum --resolve org   # who PAYS them
```
Line the two up by entity + quarter; a mismatch is the story.
**Verify:** cite the public statement AND the contradicting record; both tokens must verify.
**Generalizes to:** any corpus pairing public statements with financial disclosures.

## 7. Undisclosed relationship — entities linked but never stated
**Story:** two entities co-occur far more than chance, across sources, but no record states the tie.
**Prompt:** *"Find entity pairs that co-occur heavily across documents where the relationship is
never explicitly disclosed."*
**Hunt:**
```
muck entities --entity <id>                                     # read top co_occurring weights
muck search "<orgA> <orgB>"                                     # passages naming both
muck reconcile --on registrant --compare address,zip           # same actor under drifting names
```
**Verify:** show the co-occurrence + the absence of a stated relationship; cite both mentions.
**Generalizes to:** any multi-source corpus with resolvable entities.

## 8. Timing anomaly — activity clustered around an event
**Story:** registrations, terminations, or spend spike right before/after a vote, deadline, or news
event — a sign of coordinated or reactive influence.
**Prompt:** *"Find dates where new relationships, terminations, or spend spike, and line them up
against known events."*
**Hunt:**
```
muck aggregate --by filing_period --agg count                  # volume per period
muck aggregate --sql "SELECT json_extract(structured_json,'$.\"dt_posted\"') d, COUNT(*) n FROM documents GROUP BY d ORDER BY n DESC"
muck search "<bill or event>"                                  # date the event, compare
```
**Verify:** cite the clustered records + the event; a timing claim needs both.
**Generalizes to:** any dated corpus (contracts, filings, contributions).

## 9. Concentration / targeting — who is disproportionately aimed at one body
**Story:** a client or firm is targeting a single agency/committee far more than peers — a focused
campaign worth naming.
**Prompt:** *"Rank who most concentrates their activity on a specific target body or issue."*
**Hunt:**
```
muck aggregate --by government_entity --measure income --agg sum   # $ aimed at each body
muck aggregate --by issue_code --agg count                        # issue concentration
muck anomalies --by government_entity --measure income             # a client unusually focused on one body
```
**Verify:** cite the concentrated records; report the share vs. the baseline.
**Generalizes to:** any actor→target corpus (donations by recipient, contracts by agency).

## 10. Follow-the-money — A pays B who acts on C
**Story:** a chain — a donor who is also a client; money flowing donor→PAC→member whose committee
is being lobbied by the donor's firm. No single record contains the whole chain.
**Prompt:** *"Trace money/influence chains across sources: link the same entity as donor, client,
and lobbying target, and describe the triangle."*
**Hunt:**
```
muck entities --entity <id>                                     # every role an entity plays
muck aggregate --sql "SELECT … FROM read_json_auto('contributions.json') c JOIN muck.documents d ON …"  # join sources
muck reconcile --on registrant --link fec.json                 # bridge to an outside dataset
```
**Verify:** cite each leg of the chain separately; the triangle is only as strong as its weakest cited leg.
**Generalizes to:** any multi-party financial corpus — this is the entity-graph investigation.

## 11. Boilerplate & filing mills — the copy-paste tell
**Story:** the free-text field that's supposed to reveal *what* an actor is doing (a lobbying
"description", a grant "purpose", a permit "justification") is often recycled boilerplate. When
**one** filer stamps the *same* vague sentence across dozens of unrelated clients, that's a
template mill — a disclosure engineered to disclose nothing. When **many** independent filers
reach for the same empty phrase, it's an industry vagueness norm. Either way the "transparency"
field is hiding the real activity, and exact-duplicate text makes it provable.
**Prompt:** *"Find free-text purpose/description fields that repeat verbatim across unrelated
records. For the most-recycled phrases, determine whether one filer/firm is responsible (a
template mill) or many are (herd boilerplate), and flag disclosures too vague to reveal the ask."*
**Hunt:**
```
muck aggregate --by activity_desc --agg count -k 30            # the most-recycled description sentences
muck grep "<a top repeated phrase from the list above>" --all  # how many documents share it verbatim?
muck read <doc_id> <doc_id> …                                  # confirm: same firm? many unrelated clients?
```
A phrase shared by many clients but **concentrated in one registrant** is a filing mill; the same
phrase **spread across many registrants** is topic-du-jour vagueness. Expect a large share of
descriptions to repeat verbatim, and the most-recycled phrases to trace back to just one or two firms.
**Verify:** quote the identical description from ≥2 records and name the shared registrant; cite both.
**Generalizes to:** any corpus with a free-text purpose/narrative field — grant applications,
permit filings, contract justifications, suspicious-activity narratives (template-farm detection).

## 12. The slip — over-disclosure in high-volume filers
**Principle:** an organization that files the *same* routine disclosure hundreds of times will
eventually break its own template — writing a candid, specific description; naming the official
it's working through; filling in a field it usually leaves blank; or stating a fact that
contradicts its other filings. The **deviation from the filer's own norm** is the thing they
didn't mean to disclose. (The law of large numbers, applied to mistakes.) This is the inverse of
Playbook 11: #11 finds the boilerplate; this finds the one record that *broke* it.
**Prompt:** *"For each high-volume filer, learn its template (its usual terse/boilerplate
pattern), then surface every filing that breaks it — an unusually detailed free-text field, a
named official/agency, a normally-blank field filled in once, or a value that contradicts the
filer's other records. The break is the lead; read it in full."*
**Hunt:**
```
# a) Over-disclosure tells — routine filings rarely name their channel:
muck grep "(the office of|in coordination with|on behalf of).{0,40}(Congress|Senator|Representative|Sen\.|Rep\.)" --all
# b) Break-the-template — read a prolific filer's records side by side; find the one that says more:
muck aggregate --by registrant.name --agg count -k 30       # who files a LOT (the pattern-repeaters)
muck read <their doc_ids…>                                   # spot the description far longer/specific than the rest
# c) Rare-populated field — a field this filer leaves blank in ~all filings but fills in once:
muck grep "foreign_entities|conviction|covered_position" --all   # then group by registrant, find the singleton
# d) Amendment diff — what they went back to change:
muck grep "Amendment" --all                                 # read the original vs the amended filing (same filing_uuid)
# e) Paste-error leak — an entity named in a filing where it doesn't belong (copied from another client's template):
muck entities --entity <id>                                 # a name surfacing in filings that aren't its own
```
**Verify:** quote the over-disclosure verbatim and contrast it with the filer's boilerplate norm
(e.g. "this filer's usual description is a few words; this one runs several paragraphs / names a
specific member's office"); cite the token.
**Generalizes to:** any high-volume routine-filing corpus — SEC/FARA filings, permits,
procurement justifications. The exception to a filer's *own* template is where the mistake lives.

---

**Discipline for every playbook:** a hit is a *lead*. Read the source in full, quote verbatim,
`muck verify` the token, and only then `muck finding add`. Note what you ruled out
(`muck note add --kind decision`) so the thread stays auditable across sessions. If a strong lead
lives in a field that isn't indexed, add it to the field map and re-map — index everything as
queryable, keep verification honest at citation time.
