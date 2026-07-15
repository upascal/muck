"""Peer-relative anomaly surfacing over structured JSON-record fields.

Retrieval and citation are solved; the missing rung is *hypothesis generation* — surfacing
records worth looking at without a human pre-deciding what to look for. "Interesting" ≈
"deviates from a baseline", and ``aggregate`` computes group *totals*, never per-record
deviation. This module does.

Two modes:

* **peer** — for a numeric ``measure`` (e.g. income) and a categorical ``by`` peer key (e.g.
  registrant.name), flag records whose measure is far from *their own peer group's* baseline.
  The score is a **log-space robust modified z-score** (Iglewicz–Hoaglin: median + MAD): money
  is multiplicative, so "10× the firm's median" is the native unit, low-side outliers (a
  six-figure firm filing one tiny report) become visible, and the threshold auto-adapts to each
  group's own tightness. Peers are segmented by ``segment`` (default ``filing_type``) so
  quarterly (Q1) and year-to-date (Q1Y) figures aren't pooled into fake outliers, and the peer
  key is folded onto its resolved entity (``resolve``) so "Acme LLC" / "Acme, L.L.C." share one
  baseline — reusing the exact fold from :mod:`muck.search.aggregate`.

* **rare** — the "one-shot whale": group by ``by`` (default client.name), flag groups whose
  record count is ``<= max_count`` (default 1) yet whose measure-sum lands in the top
  ``rarity_pct`` (default 95th) percentile of all group sums. The count cap excludes the
  expected big *recurring* players; the surprise is the singleton carrying outsized money.

Stats run in pure Python over one row-level SQLite pull (stdlib :mod:`statistics`, no numpy /
no analytics extra). The pull carries each record's ``doc_id``, so every flagged row is
citation-backed: a whole-document token (verifiable via ``muck cite`` / ``muck verify``, the
same vehicle :mod:`muck.reader` mints) plus the reproducible query params + peer stats. The
*record* is hash-verified; the structured *figure* itself is query-cited (a ``structured_json``
field has no char span) unless it is promoted into the field map's ``text_fields``.
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
from collections import defaultdict

from ..cite import make_token
from ..extract.entities import normalize_key
from .aggregate import _path

# MAD / mean-abs-dev -> consistent σ estimators under normality (Iglewicz–Hoaglin).
_MAD_TO_SIGMA = 1.4826  # 1 / 0.6745
_MEANAD_TO_SIGMA = 1.2533  # sqrt(pi / 2), the MAD==0 fallback scale
_LOW_CONF_N = 10  # 5 <= n < 10 groups: MAD still noisy -> flag low_confidence


def _resolve_label(resolve: str, by: str) -> str | None:
    """Turn ``--resolve auto|org|person|none`` into an entity type (or None)."""
    if resolve in (None, "none"):
        return None
    if resolve in ("org", "person"):
        return resolve
    if resolve == "auto":
        low = by.lower()
        if any(w in low for w in ("member", "lobbyist", "person")):
            return "person"
        if any(w in low for w in ("registrant", "client", "org", "name")):
            return "org"
        return None
    raise ValueError(f"invalid --resolve {resolve!r} (expected auto|org|person|none)")


def _fold_change(ratio: float) -> float:
    """Symmetric multiplicative deviation from the peer median (a 10x jump and a 1/10th dip
    both score 10). Used both to gate trivial deviations and to rank leads by magnitude."""
    if ratio is None or ratio <= 0:
        return float("inf")
    return max(ratio, 1.0 / ratio)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def _canon_map(conn, resolve: str, settings) -> dict[str, str]:
    """norm_key -> canonical_name for one entity type (empty if not yet indexed)."""
    return {
        r["norm_key"]: r["canonical_name"]
        for r in conn.execute(
            "SELECT norm_key, canonical_name FROM entities WHERE entity_type=?", (resolve,)
        )
    }


def _fold(gv, resolve, canon, extra):
    """Fold a raw group value onto its canonical entity (or return it unchanged)."""
    if not resolve:
        return gv
    return canon.get(normalize_key(str(gv), resolve, extra), gv)


def anomalies(
    conn,
    by: str | None = None,
    measure: str = "income",
    *,
    method: str = "robust_z",
    space: str = "log",
    threshold: float | None = None,
    min_peers: int = 5,
    mode: str = "peer",
    resolve: str = "auto",
    segment: str | None = "filing_type",
    per_group_cap: int = 3,
    min_ratio: float = 1.0,
    sort: str = "ratio",
    max_count: int = 1,
    rarity_pct: float = 95.0,
    k: int = 50,
    settings=None,
) -> list[dict]:
    """Surface peer-relative outliers (``mode='peer'``) or one-shot whales (``mode='rare'``).

    Returns a JSON-ready ``list[dict]``, one row per flagged record. Peer rows are ranked by
    ``sort`` — ``ratio`` (default: biggest fold-change from the peer median, the "10x the firm's
    median" headline) or ``z`` (raw statistical extremity, which over-rewards near-constant
    groups). A record must clear both the significance gate (|z| >= threshold) and ``min_ratio``
    (fold-change floor) to flag. Rare rows are ranked by group-sum. Each row carries a
    ``citation_token`` (a whole-document token that resolves/verifies to the source record)
    and a ``query`` block (the reproducible params + peer stats).
    """
    if sort not in ("ratio", "z"):
        raise ValueError(f"invalid --sort {sort!r} (expected ratio|z)")
    if mode not in ("peer", "rare"):
        raise ValueError(f"invalid --mode {mode!r} (expected peer|rare)")
    if method not in ("robust_z", "iqr"):
        raise ValueError(f"invalid --method {method!r} (expected robust_z|iqr)")
    if space not in ("log", "linear"):
        raise ValueError(f"invalid --space {space!r} (expected log|linear)")
    if by is None:
        by = "client.name" if mode == "rare" else "registrant.name"
    if threshold is None:
        threshold = 1.5 if method == "iqr" else 3.5

    resolve_label = _resolve_label(resolve, by)
    extra = tuple(getattr(getattr(settings, "entities", None), "extra_org_suffixes", ()) or ())
    canon = _canon_map(conn, resolve_label, settings) if resolve_label else {}

    seg_p = _path(segment) if (mode == "peer" and segment and segment != "none") else None
    by_p, m_p = _path(by), _path(measure)

    select = ["doc_id", "json_extract(structured_json, ?) AS gv",
              "json_extract(structured_json, ?) AS mv"]
    params: list = [by_p, m_p]
    if seg_p:
        select.append("json_extract(structured_json, ?) AS seg")
        params.append(seg_p)
    params.append(by_p)  # WHERE bind
    rows = conn.execute(
        f"SELECT {', '.join(select)} FROM documents "
        f"WHERE json_extract(structured_json, ?) IS NOT NULL",
        params,
    ).fetchall()

    query = {
        "command": "anomalies", "by": by, "measure": measure, "mode": mode, "method": method,
        "space": space, "threshold": threshold, "min_peers": min_peers,
        "segment": (segment if seg_p else None), "resolve": resolve_label,
    }
    if mode == "rare":
        query.update(max_count=max_count, rarity_pct=rarity_pct)
        flagged = _rare(rows, by, measure, resolve_label, canon, extra, max_count, rarity_pct)
    else:
        query.update(sort=sort, min_ratio=min_ratio)
        flagged = _peer(rows, by, measure, seg_p is not None, segment, resolve_label, canon,
                        extra, method, space, threshold, min_peers, min_ratio)

    # Rank, cap so one pathological group can't monopolize, top-k.
    if mode == "peer":
        if sort == "z":  # raw statistical extremity
            flagged.sort(key=lambda f: abs(f["score"]), reverse=True)
        else:            # fold-change magnitude: the "Nx the firm's median" story leads
            flagged.sort(key=lambda f: _fold_change(f["ratio_to_median"]), reverse=True)
        if per_group_cap:
            seen: dict = defaultdict(int)
            kept = []
            for f in flagged:
                gk = f["_gk"]
                if seen[gk] >= per_group_cap:
                    continue
                seen[gk] += 1
                kept.append(f)
            flagged = kept
    else:
        flagged.sort(key=lambda f: f["group_sum"], reverse=True)
    flagged = flagged[:k]
    for f in flagged:
        f.pop("_gk", None)

    return _attach(conn, flagged, by, measure, query)


def _peer(rows, by, measure, has_seg, segment, resolve_label, canon, extra,
          method, space, threshold, min_peers, min_ratio) -> list[dict]:
    # Bucket (folded peer key, segment) -> [(doc_id, value)].
    groups: dict = defaultdict(list)
    n_seen = n_bad = 0
    for r in rows:
        gv = r["gv"]
        if gv is None or str(gv).strip() == "":
            continue
        try:
            val = float(r["mv"])
        except (TypeError, ValueError):
            n_bad += 1
            continue
        n_seen += 1
        seg = r["seg"] if has_seg else None
        groups[(_fold(gv, resolve_label, canon, extra), seg)].append((r["doc_id"], val))
    if n_seen == 0:
        if n_bad:
            raise ValueError(f"--measure {measure!r} has no numeric values")
        return []

    out: list[dict] = []
    for (name, seg), recs in groups.items():
        if space == "log":
            recs = [(d, v) for d, v in recs if v > 0]  # ln needs positive values
        n = len(recs)
        if n < min_peers:
            continue
        vals = [v for _, v in recs]
        ys = [math.log(v) for v in vals] if space == "log" else list(vals)
        median_raw = statistics.median(vals)
        low_conf = n < _LOW_CONF_N

        if method == "iqr":
            q1, _, q3 = statistics.quantiles(ys, n=4, method="inclusive")
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            for (doc_id, v), y in zip(recs, ys):
                if y > hi:
                    score = (y - hi) / iqr
                elif y < lo:
                    score = (y - lo) / iqr
                else:
                    continue
                if median_raw and _fold_change(v / median_raw) < min_ratio:
                    continue  # statistically extreme but not materially different
                out.append(_peer_row(doc_id, by, measure, name, seg, segment, v, n,
                                     median_raw, score, "iqr", low_conf))
            continue

        med = statistics.median(ys)
        devs = [abs(y - med) for y in ys]
        mad = statistics.median(devs)
        if mad > 0:
            scale, tag = mad * _MAD_TO_SIGMA, "robust_z"
        else:
            mean_ad = statistics.fmean(devs)
            if mean_ad == 0:
                continue  # zero spread -> no outliers possible
            scale, tag = mean_ad * _MEANAD_TO_SIGMA, "meanad_fallback"
        for (doc_id, v), y in zip(recs, ys):
            score = (y - med) / scale
            if abs(score) >= threshold:
                if median_raw and _fold_change(v / median_raw) < min_ratio:
                    continue  # statistically extreme but not materially different
                out.append(_peer_row(doc_id, by, measure, name, seg, segment, v, n,
                                     median_raw, score, tag, low_conf))
    return out


def _peer_row(doc_id, by, measure, name, seg, segment, value, n, median_raw, score, tag,
              low_conf) -> dict:
    ratio = value / median_raw if median_raw else float("inf")
    direction = "high" if score > 0 else "low"
    seg_part = f", {segment}={seg}" if seg is not None else ""
    why = (f"{measure} {value:,.0f} is {ratio:.1f}x its {by} peer median {median_raw:,.0f} "
           f"(score={score:+.1f}, n={n}{seg_part})")
    return {
        "_gk": (name, seg),
        "mode": "peer", "doc_id": doc_id, "by_field": by, "group_key": name,
        "segment": seg, "measure_field": measure, "measure_value": value,
        "group_n": n, "group_median": round(median_raw, 2), "score": round(score, 3),
        "direction": direction, "ratio_to_median": round(ratio, 2), "method": tag,
        "low_confidence": low_conf, "why": why,
    }


def _rare(rows, by, measure, resolve_label, canon, extra, max_count, rarity_pct) -> list[dict]:
    groups: dict = defaultdict(list)
    for r in rows:
        gv = r["gv"]
        if gv is None or str(gv).strip() == "":
            continue
        try:
            val = float(r["mv"])
        except (TypeError, ValueError):
            continue
        groups[_fold(gv, resolve_label, canon, extra)].append((r["doc_id"], val))
    if not groups:
        return []

    sums = {name: sum(v for _, v in recs) for name, recs in groups.items()}
    ordered = sorted(sums.values())
    thresh = _percentile(ordered, rarity_pct)

    out: list[dict] = []
    for name, recs in groups.items():
        cnt = len(recs)
        total = sums[name]
        if cnt > max_count or total < thresh:
            continue
        rank_pct = 100.0 * bisect.bisect_left(ordered, total) / len(ordered)  # % of groups it exceeds
        for doc_id, v in recs:
            why = (f"{by} {name!r} appears {cnt}x with total {measure} {total:,.0f} — "
                   f"top {max(1, round(100 - rank_pct))}% of {by} totals")
            out.append({
                "mode": "rare", "doc_id": doc_id, "by_field": by, "group_key": name,
                "measure_field": measure, "measure_value": v, "group_count": cnt,
                "group_sum": total, "sum_percentile_rank": round(rank_pct, 1), "why": why,
            })
    return out


def _attach(conn, flagged, by, measure, query) -> list[dict]:
    """Fetch text/structured for the flagged doc_ids; mint whole-doc citation tokens + context."""
    if not flagged:
        return flagged
    ids = [f["doc_id"] for f in flagged]
    placeholders = ",".join("?" * len(ids))
    meta = {
        row["doc_id"]: row
        for row in conn.execute(
            f"SELECT doc_id, text, structured_json FROM documents WHERE doc_id IN ({placeholders})",
            ids,
        )
    }
    for f in flagged:
        row = meta.get(f["doc_id"])
        text = row["text"] if row and row["text"] is not None else ""
        structured = json.loads(row["structured_json"]) if row and row["structured_json"] else {}
        f["citation_token"] = make_token(f["doc_id"], 0, len(text), text)  # whole-doc citation
        fu = structured.get("filing_uuid")
        if fu is not None:
            f["filing_uuid"] = fu
        f["context"] = {kk: vv for kk, vv in structured.items()
                        if kk not in (by, measure) and not kk.startswith("__")}
        f["query"] = query
    return flagged
