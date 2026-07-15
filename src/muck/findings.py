"""Findings ledger (Phase 1): claim + verifiable citation, batch-auditable.

A finding is only as good as its citation. ``add`` verifies on entry; ``audit`` re-verifies
every stored finding's quote against source — the green/red report an editor reads instead
of re-doing the work. Status: verified | unsupported (quote not at span) | invalid_token.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from .cite import verify
from .config import Settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status(v: dict, pixel_verdict: str | None = None) -> str:
    """Finding status from its verify result and any pixel review.

    Vision-tier findings rest on LLM-transcribed pixels, so quote-in-transcript is not
    enough — they sit at ``pending_review`` until a human/agent confirms the page image
    (``muck review``), then ``verified``/``rejected``. Native and OCR tiers verify as before.
    """
    if not v.get("token_valid"):
        return "invalid_token"
    if v.get("needs_pixel_review"):
        if pixel_verdict == "confirmed":
            return "verified" if v.get("quote_supported") else "unsupported"
        if pixel_verdict == "rejected":
            return "rejected"
        return "pending_review"
    return "verified" if v.get("verified") else "unsupported"


def _pixel_verdict(conn: sqlite3.Connection, finding_id: str) -> str | None:
    row = conn.execute(
        "SELECT verdict FROM pixel_reviews WHERE finding_id=?", (finding_id,)
    ).fetchone()
    return row["verdict"] if row else None


def add_finding(conn: sqlite3.Connection, settings: Settings, claim: str, quote: str, token: str) -> dict:
    v = verify(conn, settings, token, quote)
    fid = "find_" + hashlib.sha1(f"{claim}|{token}|{quote}".encode()).hexdigest()[:10]
    status = _status(v, _pixel_verdict(conn, fid))
    provenance = v.get("text_provenance", "native")
    conn.execute(
        "INSERT OR REPLACE INTO findings(finding_id, claim, quote, citation_token, support_json, "
        "status, text_provenance, created_at, audited_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (fid, claim, quote, token, json.dumps(v, default=str), status, provenance, _now(), _now()),
    )
    conn.commit()
    out = {"finding_id": fid, "status": status, "verified": v.get("verified", False),
           "text_provenance": provenance}
    if status == "pending_review":
        out["needs"] = f"muck review {fid} --confirm|--reject  (check {v.get('page_image')})"
    return out


def list_findings(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT finding_id, claim, quote, citation_token, status, audited_at FROM findings ORDER BY created_at"
    )]


def audit(conn: sqlite3.Connection, settings: Settings) -> dict:
    rows = conn.execute("SELECT finding_id, claim, quote, citation_token FROM findings").fetchall()
    results, verified, failed, pending = [], 0, 0, 0
    by_provenance: dict[str, dict] = {}
    review_queue = []
    for r in rows:
        v = verify(conn, settings, r["citation_token"], r["quote"])
        status = _status(v, _pixel_verdict(conn, r["finding_id"]))
        provenance = v.get("text_provenance", "native")
        conn.execute(
            "UPDATE findings SET status=?, support_json=?, text_provenance=?, audited_at=? WHERE finding_id=?",
            (status, json.dumps(v, default=str), provenance, _now(), r["finding_id"]),
        )
        bucket = by_provenance.setdefault(provenance, {"total": 0, "verified": 0})
        bucket["total"] += 1
        if status == "verified":
            verified += 1
            bucket["verified"] += 1
        elif status == "pending_review":
            pending += 1
            review_queue.append({
                "finding_id": r["finding_id"], "claim": r["claim"],
                "page_no": v.get("page_no"), "page_image": v.get("page_image"),
            })
        else:
            failed += 1
        results.append({
            "finding_id": r["finding_id"], "claim": r["claim"], "status": status,
            "text_provenance": provenance, "citation_token": r["citation_token"],
        })
    conn.commit()
    out = {
        "total": len(rows), "verified": verified, "failed": failed,
        "pending_review": pending, "by_provenance": by_provenance, "findings": results,
    }
    if review_queue:
        out["pixel_review_queue"] = review_queue
        out["notice"] = (
            f"{len(review_queue)} finding(s) rest on LLM-transcribed pixels and are NOT "
            "source-verified. Open each page_image and run `muck review <id> --confirm|--reject`."
        )
    return out


def review(conn: sqlite3.Connection, settings: Settings, finding_id: str, verdict: str,
           reviewer: str | None = None, note: str | None = None) -> dict:
    """Record a pixel review of a non-native-text finding (confirm/reject vs the page image)."""
    row = conn.execute(
        "SELECT support_json, quote, citation_token FROM findings WHERE finding_id=?", (finding_id,)
    ).fetchone()
    if row is None:
        return {"error": f"unknown finding {finding_id!r}"}
    support = json.loads(row["support_json"]) if row["support_json"] else {}
    conn.execute(
        "INSERT OR REPLACE INTO pixel_reviews(finding_id, verdict, reviewer, note, page_image, reviewed_at) "
        "VALUES(?,?,?,?,?,?)",
        (finding_id, verdict, reviewer, note, support.get("page_image"), _now()),
    )
    # Re-derive the finding's status now that a review exists.
    v = verify(conn, settings, row["citation_token"], row["quote"])
    status = _status(v, verdict)
    conn.execute("UPDATE findings SET status=?, audited_at=? WHERE finding_id=?", (status, _now(), finding_id))
    conn.commit()
    return {"finding_id": finding_id, "verdict": verdict, "status": status,
            "page_image": support.get("page_image")}
