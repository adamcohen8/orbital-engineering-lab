from __future__ import annotations

import hashlib
import json

ONP_REFINEMENT_SCHEMA_VERSION = 1
ONP_INITIAL_STATE_FRAME = "eci"
ONP_INITIAL_STATE_FRAME_TRANSFORM = "teme_to_eci_iau80"


def onp_materialization_signature(*, refinement_step_s: float) -> str:
    """Return the identity of config inputs that shape a generated ONP case."""

    payload = {
        "schema_version": ONP_REFINEMENT_SCHEMA_VERSION,
        "refinement_step_s": float(refinement_step_s),
        "initial_state_frame": ONP_INITIAL_STATE_FRAME,
        "frame_transform": ONP_INITIAL_STATE_FRAME_TRANSFORM,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
