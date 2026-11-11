"""Read-only Scale integration and an append-only sidecar assessment ledger.

No covariance is inferred from a TLE. Callers supply the normalized ONP study
and explicit current-state lineage hashes; binding is provenance, not physics parity.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sim.pro_conjunction.assessment import assess
from sim.pro_conjunction.contracts import digest, validate_problem
from sim.pro_conjunction.evidence import inspect_bundle, write_bundle
from sim.pro_features import FEATURE_SCALE_ANALYTICS, require_pro_feature


def snapshot(store, candidate_id):
    require_pro_feature(FEATURE_SCALE_ANALYTICS)
    path = Path(store).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        metadata = [dict(row) for row in conn.execute("SELECT * FROM store_metadata")]
        candidate = conn.execute(
            "SELECT * FROM conjunction_candidates WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        if candidate is None:
            raise ValueError("unknown Scale conjunction candidate")
        candidate = dict(candidate)
        newer = conn.execute(
            "SELECT candidate_id FROM conjunction_candidates WHERE supersedes_candidate_id=?", (candidate_id,)
        ).fetchall()
        if candidate["promotion_status"] == "superseded" or newer:
            raise ValueError("Scale candidate has been superseded")
        states = [dict(row) for row in conn.execute("SELECT * FROM current_object_state ORDER BY object_id")]
        state_products = []
        for state in states:
            row = conn.execute(
                "SELECT * FROM od_solutions WHERE od_solution_id=?", (state["od_solution_id"],)
            ).fetchone()
            if row is None:
                raise ValueError("current state missing its selected OD/TLE product")
            state_products.append(dict(row))
        invalidations = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM invalidation_queue WHERE status IN ('open','planned','claimed') ORDER BY invalidation_id"
            )
        ]
        # Include actual referenced products, not only their IDs in the candidate.
        window = conn.execute(
            "SELECT * FROM screening_windows WHERE screening_window_id=?", (candidate["screening_window_id"],)
        ).fetchone()
        if window is None:
            raise ValueError("candidate missing screening window")
        products = []
        for product_id in json.loads(window["source_product_ids_json"]):
            row = conn.execute(
                "SELECT * FROM propagation_products WHERE propagation_product_id=?", (product_id,)
            ).fetchone()
            if row is None:
                raise ValueError("candidate missing source propagation product")
            if row["status"] != "completed":
                raise ValueError("candidate source propagation product is not completed/current")
            products.append(dict(row))
    value = {
        "metadata": metadata,
        "candidate": candidate,
        "screening_window": dict(window),
        "source_products": products,
        "current_states": states,
        "state_products": state_products,
        "open_invalidations": invalidations,
    }
    return {
        "snapshot_sha256": digest(value),
        "snapshot": value,
        "state_bindings": {
            row["object_id"]: digest({"selection": row, "product": product})
            for row, product in zip(states, state_products, strict=True)
        },
    }


def assess_catalog(problem, store, candidate_id, state_bindings, ledger):
    return _catalog_workflow(problem, store, candidate_id, state_bindings, ledger)


def optimize_catalog(problem, trade, store, candidate_id, state_bindings, ledger):
    """Constrained trade bound to a complete current-state Scale snapshot."""
    return _catalog_workflow(problem, store, candidate_id, state_bindings, ledger, trade=trade)


def _catalog_workflow(problem, store, candidate_id, state_bindings, ledger, *, trade=None):
    p = validate_problem(problem)
    before = snapshot(store, candidate_id)
    ids = {o["object_id"] for o in p["objects"]}
    if ids != set(before["state_bindings"]) or state_bindings != before["state_bindings"]:
        raise ValueError("complete current-state inventory and exact explicit state-lineage hashes required")
    c = before["snapshot"]["candidate"]
    if {c["deputy_id"], c["chief_id"]} - ids or p["primary_id"] not in {c["deputy_id"], c["chief_id"]}:
        raise ValueError("candidate pair does not match study primary/inventory")
    if not p["epoch_jd_utc"] <= c["tca_jd_utc"] <= p["epoch_jd_utc"] + p["duration_s"] / 86400:
        raise ValueError("candidate TCA outside declared assessment horizon")
    if trade is None:
        result = assess(p)
        operation = "assess"
    else:
        from sim.pro_conjunction.trades import optimize

        result = optimize(p, trade)
        operation = "optimize"
    after = snapshot(store, candidate_id)
    if after["snapshot_sha256"] != before["snapshot_sha256"]:
        raise ValueError("Scale inputs changed during assessment; result cannot be promoted")
    inputs = {"problem": p, "catalog": before, "candidate_id": candidate_id}
    if trade is not None:
        inputs["trade"] = trade
    destination = Path(ledger) / digest(inputs)
    if destination.exists():
        existing = inspect_bundle(destination)
        if existing["inputs.json"] != inputs or existing["result.json"] != result:
            raise ValueError("catalog receipt collision or runtime drift; retain prior evidence")
    else:
        write_bundle(destination, operation, inputs, result)
    return {
        "status": "assessment_retained" if trade is None else "trade_retained",
        "bundle": str(destination),
        "snapshot_sha256": before["snapshot_sha256"],
        "scope": "complete current-state inventory in this Scale store; supplied ONP state lineage is analyst-declared",
    }


def catalog_status(bundle, store):
    verified = inspect_bundle(bundle)
    inputs = verified["inputs.json"]
    try:
        current = snapshot(store, inputs["candidate_id"])
        stale = current["snapshot_sha256"] != inputs["catalog"]["snapshot_sha256"]
        reason = "catalog_inputs_changed" if stale else None
    except ValueError as exc:
        stale, reason = True, str(exc)
    return {
        "status": "stale" if stale else "current",
        "reason": reason,
        "retained_snapshot_sha256": inputs["catalog"]["snapshot_sha256"],
        "risk_disposition": "analyst_review_required",
    }
