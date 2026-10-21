"""Preserve OEL-owned promotion decisions across normalized interchange."""

from __future__ import annotations

from .common import read_text, source_receipt, strict_json
from .contracts import DataIOError


def seed_decision(trajectory):
    review = trajectory.metadata.get("review_binding")
    if review is not None:
        from sim.review import ReviewWorkspace

        with ReviewWorkspace.open(review["completed_run"]) as workspace:
            if workspace.evidence_identity()["sha256"] != review["sha256"]:
                raise DataIOError("Source review changed after object import.")
    binding = trajectory.metadata.get("oel_product_binding")
    if binding is None:
        od = trajectory.metadata.get("tracking_od_binding")
        if od is not None:
            from .objects.od import verify_tracking_od_binding

            return verify_tracking_od_binding(od)
        return {"valid": True, "promotable": True, "blockers": []}
    source = binding["source"]
    if source_receipt(source["path"]) != source:
        raise DataIOError("OEL source product changed after import.")
    product = strict_json(read_text(source["path"]))
    if product != binding["product"]:
        raise DataIOError("Normalized OEL product binding does not match its source.")
    from sim.interchange.materialization import inspect_state_materialization

    return inspect_state_materialization(product, source_path=source["path"])


def require_seed_eligible(trajectory):
    decision = seed_decision(trajectory)
    if not decision["promotable"]:
        codes = [str(row.get("code", row.get("message", "blocked"))) for row in decision["blockers"]]
        raise DataIOError("Object materialization blocked by OEL promotion/compatibility gates: " + ", ".join(codes))
    return decision
