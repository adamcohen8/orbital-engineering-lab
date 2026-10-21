"""Carry OEL quality/context through hash-bound external-format sidecars."""

from pathlib import Path

from .common import read_text, source_receipt, strict_json
from .contracts import DataIOError

CONTEXT_KEYS = {
    "oel_product_binding",
    "tracking_od_binding",
    "review_binding",
    "quality",
    "freshness",
    "data_markings",
    "covariance_disposition",
    "object_specs",
    "model_assumptions",
    "seed_simulator",
    "planetary_config",
    "state_semantics",
}


def inherited_annotations(path):
    sidecar = Path(str(path) + ".receipt.json")
    if not sidecar.is_file():
        return {}
    receipt = strict_json(read_text(sidecar))
    if receipt.get("schema") != "oel.data-io-receipt.v1":
        raise DataIOError("An adjacent IO receipt uses an unknown schema; its context cannot be trusted.")
    actual = source_receipt(path)
    if receipt.get("output", {}).get("sha256") != actual["sha256"]:
        raise DataIOError("Input content no longer matches its IO receipt.")
    context = {}
    for field in ("input_metadata", "source_annotations"):
        value = receipt.get(field)
        if isinstance(value, dict):
            context.update({key: item for key, item in value.items() if key in CONTEXT_KEYS})
    return context
