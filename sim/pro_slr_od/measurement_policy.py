"""Bind physical row corrections to a fit and require the same forecast policy."""

from __future__ import annotations

import json


def row_measurement_policy(rows):
    policies = []
    for row in rows:
        station = row.get("slr_station_displacement")
        reflector = row.get("slr_center_of_mass")
        policy = {}
        if station is not None:
            policy["station_displacement"] = {
                key: station[key] for key in ("model", "tide_system", "evaluation", "source_sha256")
            }
        if reflector is not None:
            policy["center_of_mass"] = {key: reflector[key] for key in ("satellite_ilrs_id", "sha256", "source_url")}
        policies.append(policy)
    distinct = {json.dumps(policy, sort_keys=True) for policy in policies}
    if len(distinct) > 1:
        raise ValueError("SLR rows mix physical measurement policies.")
    return policies[0] if policies else {}


def require_measurement_policy(rows, solution):
    if row_measurement_policy(rows) != solution.get("measurement_model_policy", {}):
        raise ValueError("Forecast physical measurement policy differs from fitted solution.")


def require_complete_measurement_models(rows):
    """Freeze admission check, independent of the numerical residual gates."""
    row_measurement_policy(rows)
    for row in rows:
        if row.get("slr_center_of_mass", {}).get("status") == "unresolved_development_only":
            raise ValueError("Unresolved center-of-mass coverage blocks candidate qualification.")
