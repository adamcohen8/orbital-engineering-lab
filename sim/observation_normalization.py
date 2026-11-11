# ruff: noqa: E402,F401,F403,F405,I001
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.ingestion import MissionInputPacket, ingest_state_vector

OBSERVATION_PACKET_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 1
_JD_TO_S = 86400.0
_PARTITIONS = {"fit", "holdout", "excluded", "unassigned"}

from sim.observation_contracts import *  # noqa: F403

def _normalize_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    object_id: str,
    source_frame: str,
    position_units: str,
    velocity_units: str,
    time_units: str,
    default_position_sigma_km: float | None,
    default_velocity_sigma_km_s: float | None,
) -> tuple[list[dict[str, Any]], bool]:
    if len(observations) < 2:
        raise ValueError("at least two observations are required for preliminary OD.")
    raw_rows = [dict(raw or {}) for raw in observations]
    jd_presence = [row.get("jd_utc") is not None for row in raw_rows]
    if any(jd_presence) and not all(jd_presence):
        raise ValueError("observation time system is mixed; provide jd_utc for every row or for no rows.")
    pos_scale = _position_scale(position_units)
    vel_scale = _velocity_scale(velocity_units)
    time_scale = _time_scale(time_units)
    rows: list[dict[str, Any]] = []
    has_velocity = False
    first_jd: float | None = None
    previous_time: float | None = None
    seen_ids: set[str] = set()
    time_anchor_s: float | None = None
    jd_anchor: float | None = None
    for idx, row in enumerate(raw_rows):
        jd = _optional_float(row.get("jd_utc"), field_name=f"observation {idx} jd_utc")
        raw_time = _optional_float(row.get("time_s"), field_name=f"observation {idx} time_s")
        if raw_time is None and jd is None:
            raise ValueError(f"observation {idx} must include time_s or jd_utc.")
        if first_jd is None and jd is not None:
            first_jd = jd
        time_s = raw_time * time_scale if raw_time is not None else float((jd - first_jd) * _JD_TO_S)
        expected_time_system = "utc_julian_date" if jd is not None else "relative_seconds"
        declared_time_system = str(row.get("time_system", expected_time_system) or "").strip().lower()
        if declared_time_system != expected_time_system:
            raise ValueError(
                f"observation {idx} time_system must be {expected_time_system!r} for its declared epoch fields."
            )
        if row.get("normalization") and row.get("frame") is None:
            raise ValueError(f"normalized observation {idx} must explicitly declare frame='ECI'.")
        if row.get("frame") is not None:
            row_frame = _inertial_frame_key(str(row["frame"]))
            expected_row_frame = "eci" if row.get("normalization") else source_frame
            if row_frame != expected_row_frame:
                if row.get("normalization"):
                    raise ValueError(f"normalized observation {idx} frame must be 'eci'.")
                raise ValueError(f"observation {idx} frame does not match the packet source frame {source_frame!r}.")
        if jd is not None and raw_time is not None:
            if jd_anchor is None:
                jd_anchor = jd
                time_anchor_s = time_s
            expected_time_s = float(time_anchor_s) + (jd - float(jd_anchor)) * _JD_TO_S
            if abs(time_s - expected_time_s) > 1.0e-4:
                raise ValueError(f"observation {idx} time_s and jd_utc are inconsistent with the packet time anchor.")
        if previous_time is not None and time_s < previous_time:
            raise ValueError("observation times must be nondecreasing.")
        if previous_time is not None and time_s == previous_time and row.get("observation_id") is None:
            raise ValueError("repeated observation epochs require explicit, unique observation_id values.")
        previous_time = time_s
        observation_id = str(row.get("observation_id", f"{object_id}:observation:{idx:06d}") or "").strip()
        if not observation_id:
            raise ValueError(f"observation {idx} observation_id must be non-empty.")
        if observation_id in seen_ids:
            raise ValueError(f"duplicate observation_id: {observation_id!r}.")
        seen_ids.add(observation_id)
        partition = str(row.get("partition", "unassigned") or "unassigned").strip().lower()
        if partition not in _PARTITIONS:
            raise ValueError(f"observation {idx} partition must be one of {sorted(_PARTITIONS)}.")
        position = row.get("position_eci_km", row.get("position_eci"))
        if position is None:
            raise ValueError(f"observation {idx} must include position_eci_km.")
        item: dict[str, Any] = {
            "observation_id": observation_id,
            "time_s": float(time_s),
            "jd_utc": jd,
            "time_system": expected_time_system,
            "frame": "ECI",
            "partition": partition,
            "position_eci_km": _scaled_vector(
                position,
                length=3,
                scale=pos_scale,
                field_name=f"observation {idx} position",
            ),
            "source_record": deepcopy(
                dict(row.get("source_record", {}) or {})
                or {
                    "record_id": observation_id,
                    "values": _observation_jsonable(row),
                }
            ),
            "normalization": deepcopy(
                dict(row.get("normalization", {}) or {})
                or {
                    "position_scale_to_km": float(pos_scale),
                    "velocity_scale_to_km_s": float(vel_scale),
                    "time_scale_to_s": float(time_scale),
                    "source_frame": source_frame.upper(),
                    "frame_transform": "identity" if source_frame == "eci" else "declared_packet_frame_treated_as_eci",
                }
            ),
        }
        sigma = _optional_float(row.get("position_sigma_km"), field_name=f"observation {idx} position_sigma_km")
        if sigma is None:
            sigma = default_position_sigma_km
        if sigma is not None:
            if sigma <= 0.0:
                raise ValueError(f"observation {idx} position sigma must be positive.")
            item["position_sigma_km"] = float(sigma)
        velocity = row.get("velocity_eci_km_s", row.get("velocity_eci"))
        if velocity is not None:
            has_velocity = True
            item["velocity_eci_km_s"] = _scaled_vector(
                velocity,
                length=3,
                scale=vel_scale,
                field_name=f"observation {idx} velocity",
            )
            velocity_sigma = _optional_float(
                row.get("velocity_sigma_km_s"),
                field_name=f"observation {idx} velocity_sigma_km_s",
            )
            if velocity_sigma is None:
                velocity_sigma = default_velocity_sigma_km_s
            if velocity_sigma is not None:
                if velocity_sigma <= 0.0:
                    raise ValueError(f"observation {idx} velocity sigma must be positive.")
                item["velocity_sigma_km_s"] = float(velocity_sigma)
        item["measurement_type"] = "eci_position_velocity" if velocity is not None else "eci_position"
        item["uncertainty"] = _normalize_observation_uncertainty(
            row,
            observation_index=idx,
            has_velocity=velocity is not None,
            position_sigma_km=item.get("position_sigma_km"),
            velocity_sigma_km_s=item.get("velocity_sigma_km_s"),
            # Covariance transforms with the declared unit conversion just as
            # the state vector does.  Caller-authored provenance metadata must
            # never select the numerical transform.
            position_covariance_scale=1.0 / pos_scale,
            velocity_covariance_scale=1.0 / vel_scale,
        )
        attitude = row.get("attitude_quat_bn", row.get("quaternion_bn"))
        if attitude is not None:
            q = np.array(
                _scaled_vector(
                    attitude,
                    length=4,
                    scale=1.0,
                    field_name=f"observation {idx} attitude_quat_bn",
                ),
                dtype=float,
            )
            q_norm = float(np.linalg.norm(q))
            if q_norm <= 0.0 or not np.isfinite(q_norm):
                raise ValueError(f"observation {idx} attitude_quat_bn must be finite and nonzero.")
            item["attitude_quat_bn"] = (q / q_norm).tolist()
        angular_rate = row.get("angular_rate_body_rad_s", row.get("omega_body_rad_s"))
        if angular_rate is not None:
            item["angular_rate_body_rad_s"] = _scaled_vector(
                angular_rate,
                length=3,
                scale=1.0,
                field_name=f"observation {idx} angular_rate_body_rad_s",
            )
        rows.append(item)
    return rows, has_velocity


def _normalize_observation_uncertainty(
    row: Mapping[str, Any],
    *,
    observation_index: int,
    has_velocity: bool,
    position_sigma_km: float | None,
    velocity_sigma_km_s: float | None,
    position_covariance_scale: float,
    velocity_covariance_scale: float,
) -> dict[str, Any]:
    components = ["x_km", "y_km", "z_km"]
    if has_velocity:
        components.extend(["vx_km_s", "vy_km_s", "vz_km_s"])
    raw_uncertainty = dict(row.get("uncertainty", {}) or {})
    raw_covariance = row.get(
        "covariance",
        row.get(
            "covariance_matrix",
            raw_uncertainty.get("matrix") if raw_uncertainty.get("representation") == "covariance" else None,
        ),
    )
    if raw_covariance is not None:
        covariance = np.asarray(raw_covariance, dtype=float)
        expected = len(components)
        if covariance.shape != (expected, expected):
            raise ValueError(
                f"observation {observation_index} covariance must be {expected}x{expected} for its measurement dimension."
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"observation {observation_index} covariance must be finite.")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError(f"observation {observation_index} covariance must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        if np.any(eigenvalues < -1.0e-14):
            raise ValueError(f"observation {observation_index} covariance must be positive semidefinite.")
        component_scales = np.asarray(
            [position_covariance_scale] * 3
            + ([velocity_covariance_scale] * 3 if has_velocity else []),
            dtype=float,
        )
        covariance = covariance * np.outer(component_scales, component_scales)
        return {
            "representation": "covariance",
            "components": components,
            "matrix": covariance.tolist(),
            "source": str(raw_uncertainty.get("source", "provided_covariance")),
        }
    sigmas: list[float] = []
    if position_sigma_km is not None:
        sigmas.extend([float(position_sigma_km)] * 3)
    if has_velocity and velocity_sigma_km_s is not None:
        sigmas.extend([float(velocity_sigma_km_s)] * 3)
    if len(sigmas) == len(components):
        covariance = np.diag(np.square(np.asarray(sigmas, dtype=float)))
        return {
            "representation": "covariance",
            "components": components,
            "matrix": covariance.tolist(),
            "source": "scalar_sigmas",
        }
    return {
        "representation": "unspecified",
        "components": components,
        "matrix": None,
        "source": "unspecified",
    }


def _observation_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _observation_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_observation_jsonable(item) for item in value]
    return value


def _observation_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_frame: str,
) -> dict[str, Any]:
    time_systems = sorted({str(row.get("time_system", "")) for row in rows})
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "time_system": time_systems[0] if len(time_systems) == 1 else "mixed_invalid",
        "time_origin": "first_observation",
        "frame": "ECI",
        "source_frame": source_frame.upper(),
        "normalized_units": {"position": "km", "velocity": "km/s", "time": "s"},
        "ordering": "nondecreasing_time_s",
        "duplicate_epoch_policy": "allow_with_explicit_unique_observation_id",
        "partition_values": sorted(_PARTITIONS),
        "uncertainty_representation": "component_ordered_covariance",
        "source_value_policy": "preserved_per_observation",
        "normalization_transform_policy": "recorded_per_observation",
    }


def _observation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    velocity_count = sum(1 for row in rows if "velocity_eci_km_s" in row)
    attitude_count = sum(1 for row in rows if "attitude_quat_bn" in row)
    partition_counts = {
        name: sum(1 for row in rows if str(row.get("partition", "unassigned")) == name) for name in sorted(_PARTITIONS)
    }
    return {
        "observation_count": len(rows),
        "first_time_s": float(rows[0]["time_s"]),
        "last_time_s": float(rows[-1]["time_s"]),
        "arc_length_s": float(rows[-1]["time_s"] - rows[0]["time_s"]),
        "has_velocity": bool(rows) and velocity_count == len(rows),
        "has_any_velocity": velocity_count > 0,
        "velocity_observation_count": velocity_count,
        "has_attitude": bool(rows) and attitude_count == len(rows),
        "has_any_attitude": attitude_count > 0,
        "attitude_observation_count": attitude_count,
        "has_position_sigma": any(row.get("position_sigma_km") is not None for row in rows),
        "has_velocity_sigma": any(row.get("velocity_sigma_km_s") is not None for row in rows),
        "partition_counts": partition_counts,
        "uncertainty_complete": all(
            dict(row.get("uncertainty", {}) or {}).get("representation") == "covariance" for row in rows
        ),
    }


def _estimate_quality_flags(
    *,
    observation_count: int,
    arc_length_s: float,
    rms_position_residual_km: float,
    position_sigmas_km: np.ndarray,
) -> list[str]:
    flags: list[str] = []
    if int(observation_count) < 3:
        flags.append("low_observation_count")
    if float(arc_length_s) < 30.0:
        flags.append("short_arc")
    median_sigma = float(np.median(position_sigmas_km)) if position_sigmas_km.size else 1.0
    if rms_position_residual_km > max(3.0 * median_sigma, 1.0e-9):
        flags.append("high_position_residual")
    return flags


def _kalman_quality_flags(
    *,
    observation_count: int,
    mean_nis: float,
    max_nis: float,
    final_covariance_trace: float,
) -> list[str]:
    flags: list[str] = []
    if int(observation_count) < 3:
        flags.append("low_observation_count")
    if float(mean_nis) > 9.0 or float(max_nis) > 25.0:
        flags.append("large_innovation")
    if not math.isfinite(float(final_covariance_trace)) or float(final_covariance_trace) <= 0.0:
        flags.append("invalid_covariance")
    return flags


def _residual_samples(
    observations: Sequence[Mapping[str, Any]],
    position_residuals: np.ndarray,
    velocity_residuals: np.ndarray | None,
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = min(len(observations), int(limit))
    for idx in range(count):
        row: dict[str, Any] = {
            "time_s": float(observations[idx]["time_s"]),
            "position_residual_km": position_residuals[idx].tolist(),
            "position_residual_norm_km": float(np.linalg.norm(position_residuals[idx])),
        }
        if velocity_residuals is not None:
            row["velocity_residual_km_s"] = velocity_residuals[idx].tolist()
            row["velocity_residual_norm_km_s"] = float(np.linalg.norm(velocity_residuals[idx]))
        rows.append(row)
    return rows


def _state_estimate_packet(
    *,
    object_id: str,
    role: str,
    state: np.ndarray,
    covariance: np.ndarray,
    epoch_jd_utc: float | None,
    method: str,
    source_packet: ObservationPacket,
    diagnostics: Mapping[str, Any],
    warnings: Iterable[str],
) -> MissionInputPacket:
    packet = ingest_state_vector(
        object_id=object_id,
        role=role,
        position=state[:3].tolist(),
        velocity=state[3:].tolist(),
        epoch_jd_utc=epoch_jd_utc,
    ).to_dict()
    obj = packet["objects"][object_id]
    obj["state_type"] = "estimated_state_vector"
    obj["state_estimate"] = {
        "method": method,
        "covariance_6x6": np.array(covariance, dtype=float).tolist(),
        "diagnostics": deepcopy(dict(diagnostics)),
        "source_observation_summary": deepcopy(dict(source_packet.to_dict().get("summary", {}) or {})),
    }
    packet["source"] = {
        "type": "state_estimate",
        "method": method,
        "observation_source": deepcopy(dict(source_packet.to_dict().get("source", {}) or {})),
    }
    warning_list = list(packet.get("warnings", []) or [])
    warning_list.extend(str(item) for item in warnings)
    packet["warnings"] = _unique(warning_list)
    packet["validation"]["status"] = "ready_with_warnings"
    return MissionInputPacket(packet)


def _position_observation_errors(observations: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(observations) < 2:
        errors.append("at least two position observations are required.")
    previous_time: float | None = None
    for idx, row in enumerate(observations):
        if "position_eci_km" not in row:
            errors.append(f"observation {idx} is missing position_eci_km.")
        else:
            try:
                pos = np.array(row["position_eci_km"], dtype=float).reshape(3)
                if not np.all(np.isfinite(pos)):
                    errors.append(f"observation {idx} position contains non-finite values.")
            except Exception:
                errors.append(f"observation {idx} position must be length-3 numeric.")
        try:
            time_s = float(row["time_s"])
        except Exception:
            errors.append(f"observation {idx} is missing numeric time_s.")
            continue
        if previous_time is not None and time_s < previous_time:
            errors.append("observation times must be nondecreasing.")
            break
        previous_time = time_s
    return errors


def _estimate_epoch_s(observations: list[dict[str, Any]], epoch: str) -> float:
    key = str(epoch or "first").strip().lower()
    times = [float(row["time_s"]) for row in observations]
    if key == "first":
        return times[0]
    if key == "last":
        return times[-1]
    if key == "mean":
        return float(np.mean(times))
    raise ValueError("epoch must be first, last, or mean.")


def _epoch_jd_from_time(observations: list[dict[str, Any]], epoch_time_s: float) -> float | None:
    anchors = [(float(row["time_s"]), row.get("jd_utc")) for row in observations if row.get("jd_utc") is not None]
    if not anchors:
        return None
    t0, jd0 = anchors[0]
    return float(jd0) + (float(epoch_time_s) - t0) / _JD_TO_S


def _weighted_linear_fit(design: np.ndarray, values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w_sqrt = np.sqrt(weights).reshape(-1, 1)
    a = design * w_sqrt
    b = values * w_sqrt.reshape(-1)
    beta, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual = values - design @ beta
    dof = max(values.size - design.shape[1], 1)
    variance = float(np.sum(weights * residual**2) / dof)
    normal = design.T @ (weights.reshape(-1, 1) * design)
    # The supplied weights are inverse measurement variances, not merely
    # relative weights.  Reduced chi-square may inflate a mismodeled fit, but
    # allowing it below one would claim arbitrarily precise state knowledge for
    # a coincidentally low-residual arc.  This matches the conservative policy
    # used by the nonlinear batch estimator.
    cov = np.linalg.pinv(normal) * max(variance, 1.0)
    return beta, cov


def _constant_velocity_matrices(dt_s: float, accel_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    dt = float(dt_s)
    f = np.eye(6)
    f[:3, 3:] = np.eye(3) * dt
    q1 = float(accel_sigma) ** 2
    q = np.zeros((6, 6), dtype=float)
    q[:3, :3] = np.eye(3) * (dt**4 / 4.0) * q1
    q[:3, 3:] = np.eye(3) * (dt**3 / 2.0) * q1
    q[3:, :3] = q[:3, 3:].T
    q[3:, 3:] = np.eye(3) * (dt**2) * q1
    return f, q


def _position_scale(units: str) -> float:
    key = str(units or "km").strip().lower().replace(" ", "")
    if key in {"km", "kilometer", "kilometers"}:
        return 1.0
    if key in {"m", "meter", "meters"}:
        return 1000.0
    raise ValueError("position units must be 'km' or 'm'.")


def _velocity_scale(units: str) -> float:
    key = str(units or "km/s").strip().lower().replace(" ", "")
    if key in {"km/s", "km_s", "kmps", "kilometer/second", "kilometers/second"}:
        return 1.0
    if key in {"m/s", "m_s", "mps", "meter/second", "meters/second"}:
        return 1000.0
    raise ValueError("velocity units must be 'km/s' or 'm/s'.")


def _time_scale(units: str) -> float:
    key = str(units or "s").strip().lower()
    if key in {"s", "sec", "second", "seconds"}:
        return 1.0
    if key in {"min", "minute", "minutes"}:
        return 60.0
    if key in {"h", "hr", "hour", "hours"}:
        return 3600.0
    raise ValueError("time units must be seconds, minutes, or hours.")


def _inertial_frame_key(frame: str) -> str:
    key = str(frame or "eci").strip().lower()
    if key == "eci":
        return "eci"
    if key in {"gcrf", "eme2000", "j2000", "icrf"}:
        return "gcrf"
    raise ValueError("frame must be 'eci' or a supported inertial frame alias.")


def _packet_inertial_frame_key(frame: str) -> str:
    key = str(frame or "eci").strip().lower()
    if key.startswith("gcrf") and "eci" in key:
        return "gcrf"
    return _inertial_frame_key(key)


def _scaled_vector(value: Iterable[Any], *, length: int, scale: float, field_name: str) -> list[float]:
    out = [float(item) / float(scale) for item in value]
    if len(out) != int(length):
        raise ValueError(f"{field_name} must contain exactly {length} numeric values.")
    if any(not math.isfinite(item) for item in out):
        raise ValueError(f"{field_name} must contain only finite numeric values.")
    return out


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _clean_id(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    if not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{field_name} must contain only letters, numbers, underscores, or hyphens.")
    return text


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _print_observation_inspection(summary: Mapping[str, Any]) -> None:
    print(f"packet_version: {summary.get('packet_version')}")
    source = dict(summary.get("source", {}) or {})
    print(f"source_type: {source.get('type')}")
    print(f"object_id: {summary.get('object_id')}")
    print(f"measurement_type: {summary.get('measurement_type')}")
    print(f"observation_count: {summary.get('observation_count')}")
    print("estimators:")
    for name, status in dict(summary.get("estimators", {}) or {}).items():
        row = dict(status or {})
        print(f"- {name}: {'ready' if row.get('ready') else 'not_ready'}")
        for error in list(row.get("errors", []) or []):
            print(f"  - {error}")

def observation_packet_from_dict(data: Mapping[str, Any]) -> ObservationPacket:
    packet = deepcopy(dict(data))
    if int(packet.get("packet_version", 0) or 0) != OBSERVATION_PACKET_VERSION:
        raise ValueError(f"unsupported observation packet version: {packet.get('packet_version')!r}")
    if str(packet.get("kind", "") or "") != "oel.observation_packet":
        raise ValueError("observation packet kind must be 'oel.observation_packet'.")
    schema_version = int(packet.get("observation_schema_version", OBSERVATION_SCHEMA_VERSION) or 0)
    if schema_version != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported observation schema version: {schema_version!r}")
    observations = packet.get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("observation packet observations must be a list.")
    object_id = _clean_id(str(packet.get("object_id", "") or ""), "object_id")
    normalized_units = dict(packet.get("normalized_units", {}) or {})
    expected_units = {"position": "km", "velocity": "km/s", "time": "s"}
    for name, expected in expected_units.items():
        actual = str(normalized_units.get(name, expected) or expected).strip().lower()
        if actual != expected:
            raise ValueError(
                f"normalized observation packet unit {name!r} must be {expected!r}, not {actual!r}."
            )
    existing_contract = dict(packet.get("observation_contract", {}) or {})
    source_frame = _packet_inertial_frame_key(
        str(existing_contract.get("source_frame", packet.get("frame", "eci")) or "eci")
    )
    rows, _ = _normalize_observations(
        observations,
        object_id=object_id,
        source_frame=source_frame,
        position_units="km",
        velocity_units="km/s",
        time_units="s",
        default_position_sigma_km=None,
        default_velocity_sigma_km_s=None,
    )
    packet["observations"] = rows
    packet["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION
    packet["frame"] = "ECI"
    packet["observation_contract"] = _observation_contract(rows, source_frame=source_frame)
    packet["summary"] = _observation_summary(rows)
    return ObservationPacket(packet)


def load_observation_packet(path: str | Path) -> ObservationPacket:
    return observation_packet_from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))

__all__ = [name for name in globals() if not name.startswith("__")]
