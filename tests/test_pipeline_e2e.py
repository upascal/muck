"""End-to-end: ingest -> map/parse -> index -> search -> cite/verify.

The keyword path needs no network (no model download), so this runs offline. A separate
test exercises the embedding path and skips if the model can't be loaded.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import muck.embedders  # noqa: F401  (register adapters)
import muck.mappers  # noqa: F401
import muck.parsers  # noqa: F401
import muck.stores  # noqa: F401
from muck import cite as citelib
from muck import config as cfg
from muck import pipeline
from muck.db import DB_FILENAME, connect, init_schema
from muck.search import query as q

FIX = Path(__file__).parent / "fixtures"
FILES = ["lobbying_sample.json", "press_release.txt"]


def build(tmp_path: Path, embeddings: bool):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for f in FILES:
        shutil.copy(FIX / f, corpus / f)
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = embeddings
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(corpus / f) for f in FILES])
    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES)
    pipeline.extract(conn, settings, pipeline.PARSER_TYPES)
    result = pipeline.run_index(conn, settings)
    return corpus, muck_dir, settings, conn, result


def test_pipeline_and_citation_guard(tmp_path):
    corpus, muck_dir, settings, conn, _ = build(tmp_path, embeddings=False)

    st = pipeline.status(conn, settings)
    assert st["documents"].get("indexed") == 4  # 3 filings + 1 press release
    assert st["chunks"] >= 4
    assert st["fts_rows"] == st["chunks"]

    hits = q.search(conn, settings, "prescription drug pricing Medicare", "keyword", 5)
    assert hits, "keyword search returned nothing"
    top = hits[0]
    assert top.citation.token and "@" in top.citation.token

    # A real substring of the cited span verifies.
    real_quote = "prescription drug pricing"
    assert real_quote.lower() in top.citation.quote.lower()
    v = citelib.verify(conn, settings, top.citation.token, real_quote)
    assert v["token_valid"] and v["quote_supported"] and v["verified"]
    assert v["source_confirmed"] is True  # re-derived from the original file

    # A fabricated quote does NOT verify.
    bad = citelib.verify(conn, settings, top.citation.token, "Acme paid undisclosed bribes zzqqxx")
    assert bad["token_valid"] and not bad["quote_supported"] and not bad["verified"]

    # A tampered token is rejected.
    tampered = top.citation.token.rsplit("#", 1)[0] + "#deadbeef"
    bad_tok = citelib.verify(conn, settings, tampered, real_quote)
    assert not bad_tok["token_valid"] and not bad_tok["verified"]


def test_grep_exact_string(tmp_path):
    _, _, settings, conn, _ = build(tmp_path, embeddings=False)
    hits, total = q.grep(conn, settings, r"H\.R\. 1234", 10)
    assert len(hits) >= 2  # appears in filings + the press release
    assert total == len(hits)  # k=10 is not a binding cap here
    assert all(h.citation.token for h in hits)


def test_grep_total_and_all(tmp_path):
    _, _, settings, conn, _ = build(tmp_path, embeddings=False)
    all_hits, total = q.grep(conn, settings, r"\w", 0)  # matches every chunk; k=0 = no cap
    assert total > 1 and len(all_hits) == total
    capped, capped_total = q.grep(conn, settings, r"\w", 1)
    assert len(capped) == 1 and capped_total == total  # exact total survives the cap
    keys = [(h.doc_id, h.citation.char_start) for h in all_hits]
    assert keys == sorted(keys)  # deterministic, grouped by document


def build_texts(tmp_path: Path, files: dict, embeddings: bool = False):
    """Corpus of parser-only files (txt) for grep/snippet/notice tests."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, content in files.items():
        p = corpus / name
        p.write_bytes(content) if isinstance(content, bytes) else p.write_text(content)
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = embeddings
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(corpus / n) for n in files])
    pipeline.extract(conn, settings, pipeline.PARSER_TYPES)
    result = pipeline.run_index(conn, settings)
    return settings, conn, result


def test_grep_phrase_across_linebreak(tmp_path):
    """Literal spaces match PDF-style line breaks, consistent with verify's normalization."""
    settings, conn, _ = build_texts(tmp_path, {
        "minutes.txt": (
            b"The General Meeting resolved the suspension of the Libyan \r\nStock Market "
            b"from the association effective immediately."
        ),
    })
    hits, total = q.grep(conn, settings, "Libyan Stock Market", 10)
    assert total == 1 and "Libyan" in hits[0].snippet


def test_ws_tolerant_translation():
    from muck.stores.sqlite_store import _ws_tolerant

    assert _ws_tolerant("Libyan Stock Market") == r"Libyan\s+Stock\s+Market"
    assert _ws_tolerant("a  b") == r"a\s+b"  # runs collapse to one \s+
    assert _ws_tolerant(r"[a ]") == r"[a ]"  # classes untouched
    assert _ws_tolerant(r"a\ b") == r"a\ b"  # escaped space untouched
    assert _ws_tolerant("a ?b") == r"a\s?b"  # quantifier keeps its meaning


_FILLER = "The committee discussed procedural matters and vote counting rules at length. "
_TARGET = "The Nairobi Stock Exchange was suspended from membership by resolution. "


def test_best_window_lexical_fallback(tmp_path):
    """No embedder: sentences are scored by idf-weighted overlap with the query's terms."""
    from muck.search.query import _Salience, _best_window

    text = _FILLER * 8 + _TARGET + _FILLER * 8
    _, conn, _ = build_texts(tmp_path, {"report.txt": text})
    sal = lambda q: _Salience(conn, q)  # noqa: E731

    q = "which members were suspended from membership"
    assert "Nairobi" in _best_window(text, q, sal(q))
    # No query / no salience / no overlapping term -> chunk head, the old behavior.
    assert _best_window(text, q) == _best_window(text, "")
    dud = "quantum blockchain synergy"
    assert _best_window(text, dud, sal(dud)) == _best_window(text, "")
    assert _best_window(text, "").startswith(_FILLER.strip()[:40])


def test_snippet_query_relevant_for_semantic_hits(tmp_path):
    """Vector-leg hits (no FTS5 snippet) window around the query-relevant sentence."""
    from muck.embedders import build_embedder

    try:
        _ = build_embedder(cfg.Settings()).dim
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {e}")
    settings, conn, _ = build_texts(
        tmp_path, {"report.txt": _FILLER * 8 + _TARGET + _FILLER * 8}, embeddings=True
    )
    hits = q.search(conn, settings, "member removed lost membership", "semantic", 3)
    assert hits, "semantic search returned nothing"
    assert any("Nairobi" in h.snippet for h in hits)


# The real failure mode (ANNA corpus): the boilerplate is saturated with the query's own common
# words, so relevance alone always prefers it over the one sentence naming names. Rendered the way
# a PDF parser emits it — hard line wraps mid-sentence, bullets, an abbreviation.
_MINUTES = (
    "•Resolution 5: The ANNA members approve an amendment to the ANNA Membership\r\n"
    "Guidelines to include ISO 18774 alongside the existing assignment responsibilities.\r\n"
    "The ANNA members approve the adoption of the ANNA Membership Guidelines as presented.\r\n"
    "•Resolution 6: The ANNA members approve the membership guidelines amendment that\r\n"
    "clarifies the position on the absence of members during a General Meeting.\r\n"
    "Suspension of Partnership/Membership\r\n"
    "•Resolution 4: Due to the inability to resolve the outstanding issues raised, the ANNA\r\n"
    "Membership authorises the ANNA Board to proceed with the suspension of their\r\n"
    "partnership/membership and to communicate this to the Nairobi \r\nStock Exchange & "
    "Libyan Stock Market and their local market Regulator.\r\n"
    "The ANNA Treasurer, Dr. Tarek confirmed the audited figures were accepted by the members.\r\n"
)
_COMMON = (
    "•Resolution 1: The ANNA members approve the adoption of the ANNA Membership Guidelines.\r\n"
    "The ANNA members approve the amendment to the membership guidelines as presented.\r\n"
)


def test_segments_handle_pdf_wraps_bullets_and_abbreviations():
    """PDF text is hard-wrapped mid-sentence; segments must re-join it before splitting."""
    import re

    from muck.search.query import _segments

    norm, spans = _segments(_MINUTES)
    segs = [re.sub(r"\s+", " ", norm[s:e]).strip() for s, e in spans]
    # "Nairobi \r\nStock Exchange" is a line wrap, not a sentence end: the phrase stays whole.
    assert any("to the Nairobi Stock Exchange & Libyan Stock Market" in s for s in segs)
    assert any(s.startswith("Resolution 4:") for s in segs)  # a bullet starts a new segment
    assert not any(s.startswith("Tarek") for s in segs)  # "Dr." is not a sentence end
    assert any("Dr. Tarek" in s for s in segs)


def test_snippet_prefers_salient_evidence_over_boilerplate(tmp_path):
    """The bug from the field report: boilerplate matches the query, only one sentence answers it."""
    from muck.embedders import build_embedder

    try:
        emb = build_embedder(cfg.Settings())
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {e}")
    from muck.search.query import _Salience, _best_window

    # _COMMON is in every doc -> its words are corpus-common; the suspension sentence is unique.
    files = {f"filler{i}.txt": _COMMON * 4 for i in range(8)}
    files["minutes.txt"] = _MINUTES
    _, conn, _ = build_texts(tmp_path, files)

    # Note the vocabulary gap: the query says "removed/lost", the corpus says "suspension".
    query = "which members were removed and lost their membership"
    sal = _Salience(conn, query, emb, emb.embed([query])[0])
    snip = _best_window(_MINUTES, query, sal)
    assert "Nairobi" in snip, f"evidence buried; snippet was: {snip!r}"
    assert "\n" not in snip and "\r" not in snip  # snippets are single-line previews

    # ...and the scorer stays query-directed: a different question gets a different sentence.
    other = "who is the ANNA treasurer"
    sal2 = _Salience(conn, other, emb, emb.embed([other])[0])
    assert "Dr. Tarek" in _best_window(_MINUTES, other, sal2)


def test_index_notice_pure_parsed_corpus(tmp_path):
    settings, conn, result = build_texts(
        tmp_path, {"doc.txt": "Plain parsed text about the annual meeting. " * 20}
    )
    assert "entity_fields" in result.get("notice", "")
    assert "notice" in pipeline.status(conn, settings)


def test_index_no_notice_with_json(tmp_path):
    _, _, settings, conn, result = build(tmp_path, embeddings=False)
    assert "notice" not in result
    assert "notice" not in pipeline.status(conn, settings)


def test_idempotent_resume(tmp_path):
    corpus, muck_dir, settings, conn, _ = build(tmp_path, embeddings=False)
    # Re-running extract/index with only_new should be a no-op (resume-safe).
    again_map = pipeline.extract(conn, settings, pipeline.MAPPER_TYPES, only_new=True)
    again_idx = pipeline.run_index(conn, settings, only_new=True)
    assert again_map["documents"] == 0
    assert again_idx["documents"] == 0


def build_with_entities(tmp_path: Path, embeddings: bool = False):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for f in FILES:
        shutil.copy(FIX / f, corpus / f)
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = embeddings  # offline by default; entities don't need embeddings
    settings.mapper.fields = cfg.FieldMap(
        record_path="/results",
        record_id="filing_uuid",
        structured_fields=["registrant", "client", "amount", "filing_period"],
        entity_fields=["registrant:org", "client:org", "lobbyists:person"],
    )
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(corpus / f) for f in FILES])
    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES)
    pipeline.extract(conn, settings, pipeline.PARSER_TYPES)
    pipeline.run_index(conn, settings)
    return settings, conn


def test_entity_resolution_and_crossref(tmp_path):
    import muck.extract.entities as ent

    settings, conn = build_with_entities(tmp_path)
    orgs = ent.list_entities(conn, etype="org", name="Acme")
    assert orgs, "Acme not resolved"
    acme = ent.entity_detail(conn, orgs[0]["entity_id"])
    # The two surface variants merge into one entity...
    assert set(acme["aliases"]) == {"Acme Strategies LLC", "Acme Strategies, L.L.C."}
    # ...and the press release links to the same entity (cross-source).
    assert acme["doc_count"] == 3
    assert any(m["source_path"].endswith("press_release.txt") for m in acme["mentions"])
    # Distinct firm "Capitol Advisors Group" must NOT merge into Acme.
    assert any(e["canonical_name"] == "Capitol Advisors Group" for e in ent.list_entities(conn, etype="org"))
    # Bill numbers are extracted as entities.
    assert ent.list_entities(conn, etype="bill")
    # Every mention carries a citation token.
    assert all("@" in m["token"] for m in acme["mentions"])


def test_findings_ledger_and_audit(tmp_path):
    import muck.extract.entities as ent
    from muck import findings as fl

    settings, conn = build_with_entities(tmp_path)
    acme = ent.entity_detail(conn, ent.list_entities(conn, etype="org", name="Acme")[0]["entity_id"])
    m = acme["mentions"][0]

    good = fl.add_finding(conn, settings, "Acme lobbied for Global Pharma", m["raw_text"], m["token"])
    bad = fl.add_finding(conn, settings, "Acme bribed a senator", "Acme bribed a senator", m["token"])
    assert good["status"] == "verified"
    assert bad["status"] == "unsupported"

    report = fl.audit(conn, settings)
    assert report["total"] == 2 and report["verified"] == 1 and report["failed"] == 1


def test_aggregate_raw_and_resolved(tmp_path):
    from muck.search import aggregate as agg

    settings, conn = build_with_entities(tmp_path)
    raw = agg.aggregate(conn, "registrant", "amount", "sum")
    names = [r["group_value"] for r in raw]
    assert "Acme Strategies LLC" in names and "Acme Strategies, L.L.C." in names

    resolved = agg.aggregate(conn, "registrant", "amount", "sum", resolve="org")
    acme = [r for r in resolved if "Acme" in str(r["group_value"])]
    assert len(acme) == 1 and acme[0]["value"] == 400000.0 and acme[0]["n"] == 2

    counts = agg.aggregate(conn, "filing_period", agg="count")
    assert sum(r["value"] for r in counts) == 3  # three filings


def _build_anomaly_corpus(tmp_path: Path, records: list[dict], field_map, index: bool = False):
    """Map (and optionally index) an inline JSON corpus — mirrors build_firms for anomaly tests."""
    import json as _json

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "records.json").write_text(_json.dumps(records))
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = False
    settings.mapper.fields = field_map
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(corpus / "records.json")])
    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES)
    if index:
        pipeline.run_index(conn, settings)
    return settings, conn


def test_anomalies_peer_outlier(tmp_path):
    from muck.search import anomalies as an

    recs = [{"id": str(i), "firm": "A", "amount": a}
            for i, a in enumerate([98, 99, 100, 101, 102, 105, 5000])]
    fmap = cfg.FieldMap(record_id="id", structured_fields=["firm", "amount"])
    settings, conn = _build_anomaly_corpus(tmp_path, recs, fmap)

    out = an.anomalies(conn, "firm", "amount", resolve="none", segment="none", min_peers=5)
    assert len(out) == 1  # only the $5000 filing deviates from the firm's own baseline
    top = out[0]
    assert top["measure_value"] == 5000 and top["direction"] == "high" and top["group_n"] == 7
    assert top["ratio_to_median"] > 40  # ~50x the peer median
    # The whole-doc citation token resolves and passes the guard.
    assert citelib.resolve(conn, settings, top["citation_token"])["valid"] is True


def test_anomalies_min_peers_skip(tmp_path):
    from muck.search import anomalies as an

    recs = [{"id": str(i), "firm": "C", "amount": a} for i, a in enumerate([10, 11, 12, 1000])]
    fmap = cfg.FieldMap(record_id="id", structured_fields=["firm", "amount"])
    settings, conn = _build_anomaly_corpus(tmp_path, recs, fmap)

    assert an.anomalies(conn, "firm", "amount", resolve="none", segment="none", min_peers=5) == []
    small = an.anomalies(conn, "firm", "amount", resolve="none", segment="none", min_peers=2)
    assert len(small) == 1 and small[0]["measure_value"] == 1000


def test_anomalies_rarity(tmp_path):
    from muck.search import anomalies as an

    recs = []
    for c in range(10):  # ten small recurring clients (3 filings each, modest sums)
        recs += [{"id": f"s{c}_{j}", "client": f"c{c}", "amount": 100} for j in range(3)]
    recs += [{"id": f"big{j}", "client": "bigrecurring", "amount": 100000} for j in range(4)]
    recs.append({"id": "whale", "client": "OneShot Whale", "amount": 500000})  # singleton whale
    fmap = cfg.FieldMap(record_id="id", structured_fields=["client", "amount"])
    settings, conn = _build_anomaly_corpus(tmp_path, recs, fmap)

    out = an.anomalies(conn, "client", "amount", mode="rare", resolve="none")
    assert len(out) == 1  # the singleton, not the bigger recurring client
    assert out[0]["group_key"] == "OneShot Whale"
    assert out[0]["group_count"] == 1 and out[0]["measure_value"] == 500000
    assert "bigrecurring" not in {r["group_key"] for r in out}


def test_anomalies_resolve_fold(tmp_path):
    from muck.search import anomalies as an

    recs = ([{"id": f"a{i}", "registrant": "Acme Strategies LLC", "amount": a}
             for i, a in enumerate([100, 110, 105])]
            + [{"id": f"b{i}", "registrant": "Acme Strategies, L.L.C.", "amount": a}
               for i, a in enumerate([95, 102, 9000])])
    fmap = cfg.FieldMap(record_id="id", structured_fields=["registrant", "amount"],
                        entity_fields=["registrant:org"])
    settings, conn = _build_anomaly_corpus(tmp_path, recs, fmap, index=True)

    # Raw: each spelling is its own 3-record group (< min_peers) -> nothing flagged.
    assert an.anomalies(conn, "registrant", "amount", resolve="none", segment="none",
                        min_peers=5) == []
    # Folded: both variants share one 6-record baseline -> the $9000 filing flags.
    folded = an.anomalies(conn, "registrant", "amount", resolve="org", segment="none",
                          min_peers=5)
    assert len(folded) == 1 and folded[0]["measure_value"] == 9000
    assert folded[0]["group_n"] == 6 and "Acme" in str(folded[0]["group_key"])


def test_array_path_field_map_is_general(tmp_path):
    """Array-path (`key[].sub`) extraction + aggregate unnest — proven on a NON-lobbying corpus."""
    from muck.search import aggregate as agg

    # Purchase orders: line_items is a nested array; tags is an array-of-scalars two levels deep.
    orders = [
        {"order_id": "1", "vendor": "Acme",
         "line_items": [{"sku": "A", "dept": "eng", "tags": ["x", "y"]},
                        {"sku": "B", "dept": "eng", "tags": ["y"]}]},
        {"order_id": "2", "vendor": "Globex",
         "line_items": [{"sku": "A", "dept": "ops", "tags": ["z"]}]},
    ]
    fmap = cfg.FieldMap(
        record_id="order_id",
        structured_fields=["vendor",
                           "line_items[].sku as sku",
                           "line_items[].dept as dept",
                           "line_items[].tags[] as tag"],
    )
    settings, conn = _build_anomaly_corpus(tmp_path, orders, fmap)

    # Multi-valued extraction, de-duped per document (order 1's two 'eng' depts collapse to one).
    import json as _json
    s1 = _json.loads(conn.execute(
        "SELECT structured_json FROM documents WHERE title='1'").fetchone()["structured_json"])
    assert s1["sku"] == ["A", "B"] and s1["dept"] == ["eng"] and s1["tag"] == ["x", "y"]

    # aggregate UNNESTs the array field: SKU 'A' appears in both orders, 'B' in one.
    sku = {r["group_value"]: r["value"] for r in agg.aggregate(conn, "sku", agg="count")}
    assert sku == {"A": 2, "B": 1}
    dept = {r["group_value"]: r["value"] for r in agg.aggregate(conn, "dept", agg="count")}
    assert dept == {"eng": 1, "ops": 1}  # per-document distinct: order1=eng once, order2=ops
    tag = {r["group_value"]: r["value"] for r in agg.aggregate(conn, "tag", agg="count")}
    assert tag == {"x": 1, "y": 1, "z": 1}  # two-level array-of-scalars traversal works
    # A scalar field beside array fields still aggregates the old way.
    assert {r["group_value"] for r in agg.aggregate(conn, "vendor", agg="count")} == {"Acme", "Globex"}


def test_describe_fields_separates_aggregatable_from_searchable(tmp_path):
    """`muck fields` must show that a record field can be *searchable* without being *aggregatable*
    — the fix for agents concluding 'that field isn't in the index'."""
    orders = [{"order_id": "1", "vendor": "Acme", "line_items": [{"sku": "A", "dept": "eng"}]}]
    fmap = cfg.FieldMap(record_id="order_id", structured_fields=["vendor", "line_items[].sku as sku"])
    settings, conn = _build_anomaly_corpus(tmp_path, orders, fmap)
    out = pipeline.describe_fields(conn, settings)
    assert "vendor" in out["aggregatable_fields"] and "sku" in out["aggregatable_fields"]
    rf = out["record_fields"]
    # line_items is in the record + flattened into searchable text, but is NOT a structured column
    assert rf["line_items"]["searchable"] and not rf["line_items"]["aggregatable"]
    assert "NEVER conclude a field is absent" in out["note"]


def test_entity_dossier_cross_source(tmp_path):
    """The `muck entity` drill-down assembles one entity's cross-source picture: appearances by
    source + resolved identity/network + verifiable example citations."""
    from muck.extract import entities as ent

    settings, conn = build_with_entities(tmp_path)  # filings (Acme Strategies) + a press release naming Acme
    out = ent.dossier(conn, settings, "Acme")
    assert out["documents_naming_it"] >= 1
    assert out["appearances_by_source"]  # grouped by source (Acme spans filings + press)
    assert any(r["type"] == "org" and "Acme" in r["canonical_name"] for r in out["resolved"])
    assert out["examples"] and out["examples"][0]["token"]


def test_content_addressed_ids_reproduce_across_paths(tmp_path):
    """The reproducibility fix: identical content at different paths → identical doc_ids/tokens,
    so a citation minted in one index resolves against an index built elsewhere."""
    recs = [{"id": "1", "firm": "A", "amount": 100}, {"id": "2", "firm": "A", "amount": 5000}]
    fmap = cfg.FieldMap(record_id="id", structured_fields=["firm", "amount"])
    (tmp_path / "loc1").mkdir()
    (tmp_path / "loc2").mkdir()
    s1, c1 = _build_anomaly_corpus(tmp_path / "loc1", recs, fmap)
    s2, c2 = _build_anomaly_corpus(tmp_path / "loc2", recs, fmap)
    ids1 = [r["doc_id"] for r in c1.execute("SELECT doc_id FROM documents ORDER BY doc_id")]
    ids2 = [r["doc_id"] for r in c2.execute("SELECT doc_id FROM documents ORDER BY doc_id")]
    assert ids1 and ids1 == ids2  # content-addressed, not path-addressed
    row = c1.execute("SELECT doc_id, text FROM documents LIMIT 1").fetchone()
    tok = citelib.make_token(row["doc_id"], 0, len(row["text"]), row["text"])
    assert citelib.resolve(c2, s2, tok)["valid"] is True  # token from index 1 resolves in index 2


def test_build_all_reports_timing_and_provenance(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "recs.json").write_text(json.dumps([{"id": "1", "firm": "A", "amount": 100}]))
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = False
    settings.mapper.fields = cfg.FieldMap(record_id="id", structured_fields=["firm", "amount"])
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    out = pipeline.build_all(conn, settings, [str(corpus / "recs.json")])
    assert set(out["stages"]) == {"ingest", "map", "parse", "index"}
    assert all("duration_s" in s for s in out["stages"].values())
    assert out["documents"] == 1 and out["total_s"] >= 0
    from muck.db import meta_get
    assert meta_get(conn, "muck_version") and meta_get(conn, "config_hash")  # provenance recorded


def test_reset_soft_preserves_config_and_authored(tmp_path):
    settings, conn = build_with_entities(tmp_path)  # full index + entities
    muck_dir = tmp_path / "corpus" / cfg.MUCK_DIRNAME
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] > 0
    r = pipeline.reset(conn, muck_dir, hard=False)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert (muck_dir / cfg.CONFIG_FILENAME).exists()  # config survives the wipe
    assert r["mode"] == "soft" and "findings" not in r["tables_cleared"]
    r2 = pipeline.reset(conn, muck_dir, hard=True)
    assert {"findings", "observations", "pixel_reviews"} <= set(r2["tables_cleared"])


def test_main_wraps_errors_as_json(monkeypatch, capsys):
    """A raised library error becomes {"error": ...} on stdout + exit 1 (not a traceback)."""
    from muck import cli

    def boom():
        raise ValueError("boom detail")

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert json.loads(capsys.readouterr().out) == {"error": "boom detail"}
    assert ei.value.code == 1


def test_duckdb_engine_parity_and_external(tmp_path):
    """DuckDB engine matches SQLite, reads external files, and stays read-only."""
    from muck.search import aggregate as agg
    from muck.search import duck

    if not duck.available():
        pytest.skip("duckdb / sqlite extension unavailable")
    settings, conn = build_firms(tmp_path)
    db_path = str(tmp_path / "corpus" / cfg.MUCK_DIRNAME / DB_FILENAME)

    def norm(rows):
        return sorted((r["group_value"], round(r["value"], 2)) for r in rows)

    sq = agg.aggregate(conn, "registrant", "amount", "sum", engine="sqlite")
    dk = agg.aggregate(conn, "registrant", "amount", "sum", engine="duckdb", db_path=db_path)
    assert norm(sq) == norm(dk)  # identical results across engines

    ext = str(FIX / "external_orgs.json")
    n = agg.run_sql(conn, f"SELECT COUNT(*) AS n FROM read_json_auto('{ext}')", engine="duckdb", db_path=db_path)
    assert n[0]["n"] == 2  # external-file read works
    with pytest.raises(ValueError):
        agg.run_sql(conn, "DELETE FROM muck.documents", engine="duckdb", db_path=db_path)  # read-only guard


def test_duckdb_resolve_folds_entities(tmp_path):
    """--resolve folding (Python over the entities table) works on the DuckDB engine too."""
    from muck.search import aggregate as agg
    from muck.search import duck

    if not duck.available():
        pytest.skip("duckdb unavailable")
    settings, conn = build_with_entities(tmp_path)
    db_path = str(tmp_path / "corpus" / cfg.MUCK_DIRNAME / DB_FILENAME)
    res = agg.aggregate(conn, "registrant", "amount", "sum", resolve="org", engine="duckdb", db_path=db_path)
    acme = [r for r in res if "Acme" in str(r["group_value"])]
    assert len(acme) == 1 and acme[0]["value"] == 400000.0  # two variants folded + summed


def test_sql_passthrough_is_readonly(tmp_path):
    from muck.search import aggregate as agg

    _, conn = build_with_entities(tmp_path)
    rows = agg.run_sql(conn, "SELECT entity_type, COUNT(*) AS n FROM entities GROUP BY entity_type")
    assert rows
    with pytest.raises(ValueError):
        agg.run_sql(conn, "DELETE FROM entities")


def test_clustering_optional(tmp_path):
    settings, conn = build_with_entities(tmp_path)
    from muck.cluster import kmeans

    try:
        res = kmeans.build_clusters(conn, settings, "auto")
    except Exception as e:  # model unavailable offline
        pytest.skip(f"embedding model unavailable: {e}")
    assert res["clusters"] >= 1
    assert kmeans.list_clusters(conn)


def test_adapter_seams_register_and_degrade():
    """Upgrade adapters are discoverable and fail with a friendly NotInstalled, not ImportError."""
    import muck.embedders  # noqa: F401
    import muck.parsers  # noqa: F401
    import muck.rerankers  # noqa: F401
    import muck.stores  # noqa: F401
    from muck.interfaces import NotInstalled
    from muck.interfaces.embedder import EMBEDDERS, get_embedder
    from muck.interfaces.parser import PARSERS, get_parser
    from muck.interfaces.reranker import RERANKERS, get_reranker
    from muck.interfaces.store import STORES

    assert {"potion", "sbert", "openai", "voyage", "deepinfra", "gemini"} <= set(EMBEDDERS.available())
    assert {"pdfium", "text", "docx", "pymupdf", "docling"} <= set(PARSERS.available())
    assert {"sqlite-fts5", "turbovec"} <= set(STORES.available())
    assert "cross-encoder" in RERANKERS.available()

    # Optional-dependency adapters raise NotInstalled (clean message), never bare ImportError.
    with pytest.raises(NotInstalled):
        get_parser("pymupdf")  # pymupdf not in the test env
    with pytest.raises(NotInstalled):
        get_parser("docling")  # docling not in the test env
    with pytest.raises(NotInstalled):
        get_embedder("sbert").dim  # sentence-transformers not in the test env
    with pytest.raises(NotInstalled):
        get_embedder("voyage").embed(["x"])  # httpx / API key absent
    with pytest.raises(NotInstalled):
        get_reranker("cross-encoder").rerank("q", ["a", "b"])


def test_turbovec_accelerator(tmp_path):
    """turbovec backend: ANN over vectors built from sqlite-vec, with allowlist filtering."""
    try:
        import turbovec  # noqa: F401
    except ImportError:
        pytest.skip("turbovec not installed")
    import numpy as np

    import muck.stores  # noqa: F401
    from muck.db import DB_FILENAME, connect, init_schema
    from muck.interfaces.store import get_store
    from muck.schema import Chunk

    conn = connect(tmp_path / DB_FILENAME)
    init_schema(conn)
    store = get_store("turbovec")
    chunks = [Chunk(f"c{i}", "d.0", i, f"t{i}", 0, 2) for i in range(8)]
    store.index_chunks(conn, chunks)
    store.upsert_vectors(conn, [c.chunk_id for c in chunks], np.eye(8, dtype="float32"))
    conn.commit()

    q = np.zeros(8, dtype="float32")
    q[2] = 1.0
    res = store.search_vector(conn, q, k=2)
    assert res and res[0][0] == "c2"  # nearest neighbour is the matching unit vector

    # allowlist: never returns chunks outside the set, best-in-set first.
    res2 = store.search_vector(conn, q, k=3, filters={"chunk_ids": ["c5", "c6", "c2"]})
    assert {cid for cid, _ in res2} <= {"c5", "c6", "c2"} and res2[0][0] == "c2"

    assert (tmp_path / "cache" / "turbovec.tv").exists()  # derived index is cached on disk


def test_get_vectors_roundtrip(tmp_path):
    """Stored chunk vectors can be read back by chunk_id (so search needn't re-embed)."""
    import numpy as np

    import muck.stores  # noqa: F401
    from muck.db import DB_FILENAME, connect, init_schema
    from muck.interfaces.store import get_store
    from muck.schema import Chunk

    conn = connect(tmp_path / DB_FILENAME)
    init_schema(conn)
    store = get_store("sqlite-fts5")
    chunks = [Chunk(f"c{i}", "d.0", i, f"text {i}", 0, 6) for i in range(4)]
    store.index_chunks(conn, chunks)
    store.upsert_vectors(conn, [c.chunk_id for c in chunks], np.eye(4, dtype="float32"))
    conn.commit()

    ids, mat = store.get_vectors(conn, ["c2", "c0", "missing"])
    assert set(ids) == {"c0", "c2"} and mat.shape == (2, 4)  # missing dropped, rest aligned
    assert int(mat[ids.index("c2")].argmax()) == 2  # values round-trip


def test_entity_search_uses_stored_vectors(tmp_path):
    """End-to-end --person cosine search reads persisted vectors instead of re-embedding."""
    from muck.search import query as q

    try:
        settings, conn = build_with_entities(tmp_path, embeddings=True)
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {e}")
    from muck.db import meta_get
    if meta_get(conn, "embed_dim") is None:
        pytest.skip("vectors not stored")

    jane = q._resolve_entity_ids(conn, name="Jane Doe", etype="person")
    cands = q._entity_chunk_ids(conn, jane)
    from muck.interfaces.store import get_store
    ids, mat = get_store(settings.store.backend).get_vectors(conn, cands)
    assert ids and mat is not None and mat.shape[0] == len(ids)  # vectors are already stored

    hits = q.search_in_entity(conn, settings, "prescription drug pricing", name="Jane Doe", etype="person")
    assert hits and all(h.chunk_id in set(cands) for h in hits)
    assert hits[0].score >= hits[-1].score


def test_entity_filtered_cosine_search(tmp_path):
    """NER.person = X AND cosine similarity: restrict to a person's chunks, rank by query."""
    from muck.search import query as q

    settings, conn = build_with_entities(tmp_path)  # indexed without vectors; re-embeds candidates

    # Keyword fallback (embeddings disabled): still restricted to the person's chunks.
    kw = q.search_in_entity(conn, settings, "drug pricing", name="Jane Doe", etype="person")
    jane_ids = set(q._entity_chunk_ids(conn, q._resolve_entity_ids(conn, name="Jane Doe", etype="person")))
    assert jane_ids and all(h.chunk_id in jane_ids for h in kw)

    # Cosine path needs an embedder.
    settings.embedder.enabled = True
    try:
        hot = q.search_in_entity(conn, settings, "prescription drug pricing", name="Jane Doe", etype="person")
        cold = q.search_in_entity(conn, settings, "defense procurement NDAA", name="Jane Doe", etype="person")
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {e}")

    assert all(h.chunk_id in jane_ids for h in hot)        # never leaves the person's chunks
    # The top chunk changes with the query -> the cosine ranking is real.
    assert hot[0].chunk_id != cold[0].chunk_id
    assert hot[0].score >= hot[-1].score                   # sorted by cosine, descending


def build_firms(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(FIX / "firms_filings.json", corpus / "firms_filings.json")
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = False
    settings.entities.resolve = False
    settings.mapper.fields = cfg.FieldMap(
        record_id="filing_id",
        structured_fields=["registrant", "address", "zip", "amount"],
    )
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(corpus / "firms_filings.json")])
    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES)
    return settings, conn


def test_field_map_drift_detected(tmp_path):
    """Changing the field map without re-mapping is surfaced, not silent."""
    settings, conn = build_with_entities(tmp_path)  # maps + indexes with a field map
    assert pipeline.field_map_drift(conn, settings) is None  # fresh: no drift
    assert "config_drift" not in pipeline.status(conn, settings)

    settings.mapper.fields = cfg.FieldMap(record_path="/results", entity_fields=["registrant:org"])
    assert pipeline.field_map_drift(conn, settings) is not None  # changed map -> drift
    st = pipeline.status(conn, settings)
    assert "config_drift" in st and "muck map --all" in st["config_drift"]

    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES, only_new=False)  # re-map clears it
    assert pipeline.field_map_drift(conn, settings) is None


def test_xml_mapper_house_filing(tmp_path):
    """House LDA XML -> Record via the xml mapper; citations re-derive from source."""
    from muck import cite as citelib
    from muck import config as cfg
    from muck import pipeline
    from muck.db import DB_FILENAME, connect, init_schema
    from muck.mappers.xml_mapper import xml_file_to_record_dict

    rec = xml_file_to_record_dict(str(FIX / "house_ld2_sample.xml"))
    assert rec["_root"] == "LOBBYINGDISCLOSURE2"
    assert rec["organizationName"] == "Acme Strategies LLC"
    assert rec["alis"]["ali_Code"] == ["HCR", "TAX"]  # repeated tags -> list; whitespace-only dropped
    assert isinstance(rec["lobbyists"]["lobbyist"], list) and len(rec["lobbyists"]["lobbyist"]) == 2
    assert "expenses" not in rec  # whitespace-only element dropped

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(FIX / "house_ld2_sample.xml", corpus / "301642857.xml")
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = False
    settings.mapper.fields = cfg.FieldMap(
        structured_fields=["organizationName", "clientName", "income", "reportYear", "senateID"],
        entity_fields=["organizationName:org", "clientName:org"],
    )
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(corpus / "301642857.xml")])
    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES)
    res = pipeline.run_index(conn, settings)
    assert res["documents"] == 1 and res["chunks"] >= 1
    assert res["entity_index"]["entities"] >= 2  # Acme + Global Pharma from entity_fields

    # A citation over the XML-derived text verifies against the source file.
    row = conn.execute("SELECT doc_id, text FROM documents").fetchone()
    from muck.cite import make_token
    tok = make_token(row["doc_id"], 0, len(row["text"]), row["text"])
    v = citelib.verify(conn, settings, tok, "prescription drug pricing")
    assert v["verified"] and v["source_confirmed"] is True


def test_peek_schema_discovery():
    """`peek` reports a raw file's record array, field inventory, and a starter field map."""
    from muck import peek

    out = peek.peek_file(str(FIX / "lobbying_sample.json"))
    assert out["format"] == "json" and out["record_path"] == "/results" and out["n_records"] == 3
    names = {f["name"] for f in out["fields"]}
    assert {"registrant", "client", "issues", "lobbyists", "amount"} <= names
    assert len(out["samples"]) == 2
    sug = out["suggested_field_map"]["entity_fields"]
    assert "registrant:org" in sug and "lobbyists:person" in sug  # heuristic starter map


def test_sample_random_and_query(tmp_path):
    """`sample` loads budget-bounded full records — random, and behind a query."""
    from muck import reader

    settings, conn = build_with_entities(tmp_path)  # indexed (keyword), embeddings off

    rand = reader.sample_documents(conn, settings, n=2, token_budget=8000, order="first")
    assert rand["returned"] <= 2 and rand["returned"] >= 1
    assert all(d["text"] for d in rand["documents"])
    assert rand["selection"]["by"] == "first"

    q = reader.sample_documents(conn, settings, query="prescription drug pricing", token_budget=8000)
    assert q["returned"] >= 1
    assert any("drug pricing" in d["text"].lower() for d in q["documents"])  # full docs behind hits


def test_batch_read_full_docs(tmp_path):
    """`read` pulls multiple full records side by side, budget-bounded, with whole-doc citations."""
    from muck import cite as citelib
    from muck import reader

    settings, conn = build_firms(tmp_path)
    ids = [r[0] for r in conn.execute("SELECT doc_id FROM documents ORDER BY doc_id LIMIT 3")]

    out = reader.read_documents(conn, settings, ids, token_budget=16000)
    assert out["returned"] == 3 and not out["omitted"]
    first = out["documents"][0]
    assert first["text"] and first["structured"]  # full text + structured fields present
    # the whole-document citation token verifies against source
    quote = first["text"].split("\n")[0][:20]
    assert citelib.verify(conn, settings, first["citation_token"], quote)["verified"]

    # token budget never silently truncates: tiny budget -> first doc only, rest reported
    tight = reader.read_documents(conn, settings, ids, token_budget=1)
    assert tight["returned"] == 1
    assert {o["reason"] for o in tight["omitted"]} == {"token_budget"} and len(tight["omitted"]) == 2

    # unknown ids are reported, not crashed
    miss = reader.read_documents(conn, settings, ["nope"], token_budget=16000)
    assert miss["returned"] == 0 and miss["omitted"][0]["reason"] == "unknown"


def test_observations_diary(tmp_path):
    """Tier 1: notes/decisions/leads with status; open threads surface in `status`."""
    from muck import config as cfg
    from muck import observations as obs
    from muck import pipeline
    from muck.db import DB_FILENAME, connect, init_schema

    conn = connect(tmp_path / DB_FILENAME)
    init_schema(conn)
    obs.add(conn, "pursuing Acme Q4 timing — spend spiked", kind="decision")
    lead = obs.add(conn, "check revolving door for Jane Doe", kind="lead",
                   entity_ids=["e_x"], tokens=["d.0@0-5#abc"])
    obs.add(conn, "general scratch note", kind="note")

    assert len(obs.list(conn, status="open")) == 3
    assert {o["kind"] for o in obs.list(conn, kind="lead")} == {"lead"}
    assert [o["obs_id"] for o in obs.list(conn, entity="e_x")] == [lead["obs_id"]]  # entity filter

    obs.update(conn, lead["obs_id"], status="cold")  # thread goes cold
    assert all(o["obs_id"] != lead["obs_id"] for o in obs.list(conn, status="open"))
    summ = obs.open_summary(conn)
    assert summ["by_status"]["open"] == 2 and summ["by_status"]["cold"] == 1
    assert summ["open_threads"] == []  # the only thread-kind item is now cold

    st = pipeline.status(conn, cfg.Settings())
    assert st["observations"]["by_status"]["cold"] == 1  # surfaced on resume


def test_recall_over_memory(tmp_path):
    """Tier 2: recall searches observations + findings (keyword), with filters."""
    from muck import config as cfg
    from muck import observations as obs
    from muck.db import DB_FILENAME, connect, init_schema
    from muck.search import recall as rec

    conn = connect(tmp_path / DB_FILENAME)
    init_schema(conn)
    obs.add(conn, "Acme lobbied on prescription drug pricing", kind="lead")
    obs.add(conn, "weather is nice today", kind="note")
    conn.execute(
        "INSERT INTO findings(finding_id, claim, quote, citation_token, status, created_at, audited_at) "
        "VALUES('f1','drug pricing finding','prescription drug pricing','tok','verified','x','x')"
    )
    conn.commit()

    settings = cfg.Settings()
    settings.embedder.enabled = False  # keyword path, offline
    res = rec.recall(conn, settings, "prescription drug pricing", mode="keyword")
    assert res and "weather" not in res[0]["text"]  # relevant first
    assert {r["type"] for r in res} == {"observation", "finding"}  # both layers searched
    only_findings = rec.recall(conn, settings, "drug", kind="finding", mode="keyword")
    assert only_findings and all(r["type"] == "finding" for r in only_findings)


def test_recall_semantic_optional(tmp_path):
    """Tier 2 semantic: recall finds memory by meaning, not just shared words."""
    from muck import config as cfg
    from muck import observations as obs
    from muck.db import DB_FILENAME, connect, init_schema
    from muck.embedders import build_embedder
    from muck.search import recall as rec

    conn = connect(tmp_path / DB_FILENAME)
    init_schema(conn)
    obs.add(conn, "Acme lobbied on prescription drug pricing", kind="lead")
    obs.add(conn, "office snacks restock schedule", kind="note")
    settings = cfg.Settings()
    try:
        _ = build_embedder(settings).dim
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {e}")
    res = rec.recall(conn, settings, "lowering medication costs", mode="semantic")
    assert res and "drug pricing" in res[0]["text"].lower()  # matched by meaning, no shared terms


def test_reconcile_dedupe_and_link(tmp_path):
    """Splink multi-field reconciliation: merge messy variants, avoid false merges, link datasets."""
    try:
        import splink  # noqa: F401
    except ImportError:
        pytest.skip("splink not installed")
    from muck import reconcile as rec

    settings, conn = build_firms(tmp_path)

    out = rec.reconcile(conn, settings, on="registrant", compare=["address", "zip"], threshold=0.9)
    by_name = {tuple(sorted(c["names"])): c for c in out["clusters"]}
    acme = next(c for c in out["clusters"] if any("Acme" in n for n in c["names"]))
    assert acme["size"] == 3  # all three Acme variants (incl. the typo) merge
    # The different firm at the SAME address must NOT merge into Acme.
    assert not any("Beacon" in n for n in acme["names"])
    assert any(any("Global" in n for n in c["names"]) and c["size"] == 2 for c in out["clusters"])

    # Audit trail persisted.
    assert conn.execute("SELECT COUNT(*) FROM reconcile_runs").fetchone()[0] >= 1
    assert conn.execute("SELECT COUNT(*) FROM reconcile_records").fetchone()[0] >= 7
    from muck.db import meta_get
    assert meta_get(conn, "reconcile_method") == "splink-fellegi-sunter"

    # Cross-dataset linkage.
    linked = rec.reconcile(conn, settings, on="registrant", compare=["address", "zip"],
                           threshold=0.9, link_path=str(FIX / "external_orgs.json"))
    acme2 = next(c for c in linked["clusters"] if any("Acme" in n for n in c["names"]))
    assert len(acme2["sources"]) == 2  # spans corpus + external dataset
    assert any(s.startswith("link:") for s in acme2["sources"])


def test_embedding_path_optional(tmp_path):
    try:
        _, _, settings, conn, result = build(tmp_path, embeddings=True)
    except Exception as e:  # model download unavailable
        pytest.skip(f"embedding model unavailable: {e}")
    if result["embeddings"] != "on":
        pytest.skip("vector backend or model not active")
    hits = q.search(conn, settings, "lowering medication costs for families", "semantic", 3)
    assert hits  # semantic recall finds the drug-pricing docs without exact keywords


# --- generalization hardening: per-source maps, entity knobs, XML records, peek -----------

def _build_multisource(tmp_path: Path, sources: list[cfg.SourceMap]):
    """Two sources in different dirs, each with its own [[mapper.sources]] field map."""
    corpus = tmp_path / "corpus"
    (corpus / "press").mkdir(parents=True)
    (corpus / "filings").mkdir(parents=True)
    shutil.copy(FIX / "lobbying_sample.json", corpus / "filings" / "lobbying_sample.json")
    (corpus / "press" / "releases.jsonl").write_text(
        '{"url": "https://x.gov/a", "title": "Senator Acts on Drug Pricing", '
        '"member": {"name": "Jane Public", "party": "D"}, '
        '"text": "The senator announced action on prescription drug pricing today."}\n'
    )
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = False
    settings.mapper.sources = sources
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(p) for p in corpus.rglob("*") if p.is_file() and ".muck" not in p.parts])
    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES)
    pipeline.run_index(conn, settings)
    return settings, conn


PRESS_SRC = cfg.SourceMap(
    match="*/press/*", record_id="url", title_template="{title}",
    text_fields=["title", "text"], entity_fields=["member.name:person"],
)
FILING_SRC = cfg.SourceMap(
    match="*/filings/*", record_path="/results", record_id="filing_uuid",
    title_template="{registrant} - {client}",
    structured_fields=["registrant", "client", "amount"],
    entity_fields=["registrant:org", "client:org"],
)


def test_per_source_field_maps(tmp_path):
    """Each source gets its own map — and citations re-derive per-source (fields_for threading)."""
    settings, conn = _build_multisource(tmp_path, [PRESS_SRC, FILING_SRC])

    titles = {r["source_path"].split("/")[-2]: r["title"]
              for r in conn.execute("SELECT source_path, title FROM documents")}
    assert titles["press"] == "Senator Acts on Drug Pricing"      # press title_template
    assert " - " in titles["filings"]                              # filing title_template

    # Per-source entity_fields: a person from press, orgs from filings.
    kinds = {r["entity_type"] for r in conn.execute("SELECT entity_type FROM entities")}
    assert {"org", "person"} <= kinds
    people = [r["canonical_name"] for r in conn.execute(
        "SELECT canonical_name FROM entities WHERE entity_type='person'")]
    assert "Jane Public" in people  # only reachable via the press source's member.name spec

    # cite/verify must re-derive with the SAME per-source map for BOTH sources.
    for like, quote in [("%/press/%", "prescription drug pricing"),
                        ("%/filings/%", "Acme Strategies LLC")]:
        row = conn.execute(
            "SELECT doc_id, text FROM documents WHERE source_path LIKE ? AND text LIKE ?",
            (like, f"%{quote}%"),
        ).fetchone()
        assert row is not None, f"no doc for {like}"
        tok = citelib.make_token(row["doc_id"], 0, len(row["text"]), row["text"])
        v = citelib.verify(conn, settings, tok, quote)
        assert v["verified"] and v["source_confirmed"] is True, f"{like} failed to re-derive"


def test_source_map_edit_triggers_drift(tmp_path):
    settings, conn = _build_multisource(tmp_path, [PRESS_SRC, FILING_SRC])
    assert pipeline.field_map_drift(conn, settings) is None
    settings.mapper.sources[0].entity_fields = ["member.name:org"]  # edit a source map
    assert "muck map --all" in (pipeline.field_map_drift(conn, settings) or "")


def test_entity_matching_knobs(tmp_path):
    """single_word_match / min_name_len / extra_org_suffixes are config, not hard-coded."""
    from muck.extract.entities import NameMatcher, normalize_key

    gaz = {"clear": "org", "acme strategies": "org", "&pizza": "org", "ib": "org"}
    orig = {"clear": "CLEAR", "acme strategies": "Acme Strategies", "&pizza": "&pizza", "ib": "IB"}
    text = "We met clear and CLEAR, plus Acme Strategies, IB and &pizza today."

    def spans(**kw):
        return {text[s:e] for s, e in NameMatcher(gaz, orig, **kw).finditer(text)}

    # exact-case (default): only the registered casing of a single-word name matches.
    ec = spans()
    assert "CLEAR" in ec and "clear" not in ec
    assert "Acme Strategies" in ec          # multi-word: case-insensitive
    assert "&pizza" in ec                   # anchor-offset fix (first token isn't at index 0)
    assert "IB" not in ec and "ib" not in ec  # under min_name_len

    assert {"clear", "CLEAR"} <= spans(single_word="any-case")
    off = spans(single_word="off")
    assert "CLEAR" not in off and "Acme Strategies" in off  # off kills single-word only
    assert "IB" in spans(min_len=2, single_word="any-case")

    # extra_org_suffixes makes a non-US legal form fold like LLC does.
    assert normalize_key("Acme GmbH", "org") != normalize_key("Acme", "org")
    assert normalize_key("Acme GmbH", "org", ("gmbh",)) == normalize_key("Acme", "org", ("gmbh",))


def test_person_name_folds_titles_suffixes_initials():
    """A titled/formal person variant folds to the same key as a bare name (cross-source link)."""
    from muck.extract.entities import normalize_key

    bare = normalize_key("Angus King", "person")
    assert bare == "angus king"
    # honorific + U.S. Senator + middle initial + generational suffix all strip away
    assert normalize_key("The Honorable U.S. Senator Angus S. King, Jr.", "person") == bare
    assert normalize_key("Sen. Angus King", "person") == bare
    # distinct people must NOT merge
    assert normalize_key("Angus King", "person") != normalize_key("Amy King", "person")
    # a name that is ALL titles doesn't normalize away to empty
    assert normalize_key("Senator", "person") != ""


def test_array_path_entity_fields_resolve():
    """`field[].sub:type` declarations resolve nested actors as entities (not just flat fields)."""
    from muck.mappers.json_mapper import _extract_entity_values, render_text

    rec = {
        "registrant": {"name": "PAC Co"},
        "contribution_items": [
            {"honoree_name": "Sen. Angus King", "amount": 5000},
            {"honoree_name": "Rep. Jane Public", "amount": 2500},
        ],
    }
    ents = _extract_entity_values(rec, ["contribution_items[].honoree_name:person"])
    names = {e["name"] for e in ents}
    assert names == {"Sen. Angus King", "Rep. Jane Public"}
    assert all(e["type"] == "person" for e in ents)
    # and the array-path entity field renders into searchable text even when text_fields is set
    # (exercises _render_field's `[]` branch, not the whole-record flatten fallback), so those
    # honoree mentions are findable by the resolver
    fields = cfg.FieldMap(
        text_fields=["registrant.name"],
        entity_fields=["contribution_items[].honoree_name:person"],
    )
    text = render_text(rec, fields)
    assert "Sen. Angus King" in text and "Rep. Jane Public" in text


def test_xml_record_path_multi(tmp_path):
    """Bulk-export XML: record_path splits one file into many docs, each independently citable."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(FIX / "filings_multi.xml", corpus / "filings_multi.xml")
    muck_dir = corpus / cfg.MUCK_DIRNAME
    (muck_dir / "cache").mkdir(parents=True)
    cfg.write_default_config(muck_dir)
    settings = cfg.load_settings(muck_dir)
    settings.embedder.enabled = False
    settings.mapper.fields = cfg.FieldMap(
        record_path="filing", record_id="filingId",
        title_template="{organizationName} - {clientName}",
        structured_fields=["organizationName", "clientName", "income"],
        entity_fields=["organizationName:org", "clientName:org"],
    )
    conn = connect(muck_dir / DB_FILENAME)
    init_schema(conn)
    pipeline.ingest(conn, [str(corpus / "filings_multi.xml")])
    pipeline.extract(conn, settings, pipeline.MAPPER_TYPES)
    pipeline.run_index(conn, settings)

    rows = conn.execute("SELECT doc_id, locator, title, text FROM documents ORDER BY locator").fetchall()
    assert len(rows) == 3  # one file -> three records
    assert [r["locator"] for r in rows] == ["0", "1", "2"]
    assert rows[0]["title"] == "Acme Strategies LLC - Global Pharma Inc."

    # Each element re-derives independently from source (map_one_text honors the locator).
    r = rows[1]
    tok = citelib.make_token(r["doc_id"], 0, len(r["text"]), r["text"])
    v = citelib.verify(conn, settings, tok, "defense procurement")
    assert v["verified"] and v["source_confirmed"] is True
    # ...and the *wrong* record's text must not verify against this one's span.
    assert not citelib.verify(conn, settings, tok, "prescription drug pricing")["verified"]


def test_peek_nested_and_directory_sources(tmp_path):
    """peek expands nested objects (dotted) and suggests [[mapper.sources]] for mixed dirs."""
    from muck import peek

    corpus = tmp_path / "corpus"
    (corpus / "press").mkdir(parents=True)
    (corpus / "filings").mkdir(parents=True)
    shutil.copy(FIX / "lobbying_sample.json", corpus / "filings" / "lobbying_sample.json")
    (corpus / "press" / "releases.jsonl").write_text(
        '{"url": "https://x.gov/a", "title": "T", '
        '"member": {"name": "Jane Public", "party": "D"}, "text": "body"}\n'
    )

    one = peek.peek_file(str(corpus / "press" / "releases.jsonl"))
    names = {f["name"] for f in one["fields"]}
    assert {"member.name", "member.party"} <= names  # nested expanded one level
    ents = one["suggested_field_map"]["entity_fields"]
    assert "member.name:person" in ents
    # An entity's *attributes* are never entities: suggesting these would index
    # party labels and URLs as people.
    assert not any(e.startswith(("member.party", "member.state", "url", "title")) for e in ents)

    # Hint ordering: `client` (an org word) must win over the weak `name` person hint.
    filings = peek.peek_file(str(corpus / "filings" / "lobbying_sample.json"))
    ents = filings["suggested_field_map"]["entity_fields"]
    assert "registrant:org" in ents and "lobbyists:person" in ents

    d = peek.peek_file(str(corpus))
    assert d["format"] == "directory" and d["schemas"] == 2
    matches = {s["match"] for s in d["sources"]}
    assert matches == {"*/press/*", "*/filings/*"}
    for s in d["sources"]:  # each is a copy-pasteable [[mapper.sources]] block
        assert s["suggested_source_map"]["match"] == s["match"]
