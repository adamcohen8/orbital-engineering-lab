from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.public_api.config import (
    MetricCallback,
    SimulationConfig,
)
from sim.public_api.snapshots import SimulationSnapshot
from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue

_STATE_COLUMNS = [
    "x_eci_km",
    "y_eci_km",
    "z_eci_km",
    "vx_eci_km_s",
    "vy_eci_km_s",
    "vz_eci_km_s",
    "q0",
    "q1",
    "q2",
    "q3",
    "wx_body_rad_s",
    "wy_body_rad_s",
    "wz_body_rad_s",
    "mass_kg",
]
_RIC_STATE_COLUMNS = [
    "r_km",
    "i_km",
    "c_km",
    "rdot_km_s",
    "idot_km_s",
    "cdot_km_s",
]
_ECI_REL_STATE_COLUMNS = [
    "dx_eci_km",
    "dy_eci_km",
    "dz_eci_km",
    "dvx_eci_km_s",
    "dvy_eci_km_s",
    "dvz_eci_km_s",
]


def _closest_approach_metric(payload: dict[str, Any]) -> float:
    from sim.execution.metrics import closest_approach_from_run_payload

    return closest_approach_from_run_payload(payload)


def _as_array_map(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for key, arr in value.items():
        try:
            out[str(key)] = np.array(arr, dtype=float)
        except (TypeError, ValueError):
            continue
    return out


def _as_nested_array_map(value: Any) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, inner in value.items():
        if not isinstance(inner, dict):
            continue
        out[str(key)] = _as_array_map(inner)
    return out


def _as_2d_state_history(value: Any, *, width: int = 6) -> np.ndarray:
    arr = np.array(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, width), dtype=float)
    if arr.ndim == 1:
        if arr.size < width:
            raise ValueError(f"State history must have at least {width} columns.")
        return arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < width:
        raise ValueError(f"State history must be a 2D array with at least {width} columns.")
    return arr


def _json_safe_metric_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric_value(v) for v in value]
    return value


def _metric_callback_name(callback: MetricCallback, index: int) -> str:
    name = str(getattr(callback, "__name__", "") or "").strip()
    if name and name != "<lambda>":
        return name
    return f"metric_{index}"


def _evaluate_metric_callbacks(
    result: SimulationResult,
    metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...] | None,
) -> dict[str, Any]:
    if metrics is None:
        return {}
    out: dict[str, Any] = {}
    if isinstance(metrics, Mapping):
        iterable = list(metrics.items())
    else:
        iterable = [(_metric_callback_name(callback, idx), callback) for idx, callback in enumerate(metrics)]
    for name, callback in iterable:
        value = callback(result)
        if isinstance(value, Mapping):
            for key, item in value.items():
                out[str(key)] = _json_safe_metric_value(item)
        else:
            out[str(name)] = _json_safe_metric_value(value)
    return out


def _aggregate_custom_metrics(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({str(key) for row in run_metrics for key in row.keys()})
    aggregate: dict[str, Any] = {"run_count": int(len(run_metrics))}
    for key in keys:
        values = [row.get(key) for row in run_metrics if key in row]
        bool_values = [bool(v) for v in values if isinstance(v, (bool, np.bool_))]
        if bool_values and len(bool_values) == len(values):
            true_count = int(sum(1 for v in bool_values if v))
            aggregate[key] = {
                "true_count": true_count,
                "false_count": int(len(bool_values) - true_count),
                "probability_true": float(true_count / max(len(bool_values), 1)),
            }
            continue
        numeric: list[float] = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x):
                numeric.append(x)
        if numeric:
            arr = np.array(numeric, dtype=float)
            aggregate[key] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p05": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }
    return aggregate


def _range_scale(units: str) -> float:
    key = str(units or "km").strip().lower()
    if key in {"km", "kilometer", "kilometers"}:
        return 1.0
    if key in {"m", "meter", "meters"}:
        return 1000.0
    raise ValueError("range units must be 'km' or 'm'.")


def _validate_event_radius_km(radius_km: float) -> float:
    radius = float(radius_km)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_km must be finite and nonnegative.")
    return radius


def _segment_minimum_range(
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, float]:
    delta = end - start
    denom = float(np.dot(delta, delta))
    fraction = 0.0 if denom <= 0.0 else float(np.clip(-np.dot(start, delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(start + fraction * delta)), fraction


def _segment_sphere_entry_fraction(start: np.ndarray, end: np.ndarray, radius_km: float) -> float | None:
    if float(np.linalg.norm(start)) <= radius_km:
        return 0.0
    delta = end - start
    a = float(np.dot(delta, delta))
    if a <= 0.0:
        return None
    b = 2.0 * float(np.dot(start, delta))
    c = float(np.dot(start, start)) - radius_km * radius_km
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_discriminant = float(np.sqrt(max(discriminant, 0.0)))
    roots = sorted(((-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)))
    for fraction in roots:
        if 0.0 <= fraction <= 1.0:
            return float(fraction)
    return None


def _records_dataframe(records: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return records
    return pd.DataFrame.from_records(records)


def _numeric_metric_value(row: Mapping[str, Any], metric: str) -> float:
    try:
        return float(row.get(metric))
    except (TypeError, ValueError):
        return float("nan")


def _artifact_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _artifact_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_artifact_paths(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MetricStudyResult(dict):
    """Dictionary result with convenience methods for custom metric studies."""

    def metrics_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for run in list(self.get("runs", []) or []):
            row = {
                "iteration": int(dict(run).get("iteration", len(records))),
                "seed": dict(run).get("seed"),
            }
            sampled = dict(dict(run).get("sampled_parameters", {}) or {})
            row.update({f"sampled.{key}": value for key, value in sampled.items()})
            row.update(dict(dict(run).get("metrics", {}) or {}))
            records.append(row)
        return records

    def metrics_dataframe(self) -> Any:
        return _records_dataframe(self.metrics_records())

    def failures(self, metric: str) -> list[dict[str, Any]]:
        return [dict(run) for run in list(self.get("runs", []) or []) if bool(dict(run).get("metrics", {}).get(metric))]

    def top(self, metric: str, *, n: int = 10, smallest: bool = True) -> list[dict[str, Any]]:
        runs = [dict(run) for run in list(self.get("runs", []) or [])]
        runs = [run for run in runs if np.isfinite(_numeric_metric_value(dict(run).get("metrics", {}) or {}, metric))]
        return sorted(
            runs,
            key=lambda run: _numeric_metric_value(dict(run).get("metrics", {}) or {}, metric),
            reverse=not bool(smallest),
        )[: int(max(n, 0))]


@dataclass
class SimulationResult:
    config: SimulationConfig
    payload: dict[str, Any]

    def _snapshot_time_s(self) -> np.ndarray:
        return np.array(self.payload.get("time_s", []), dtype=float).reshape(-1)

    def _snapshot_history(self, payload_key: str) -> Mapping[str, Any]:
        """Return a live payload mapping without copying full histories.

        ``SimulationResult.payload`` is public and mutable. Snapshot access
        therefore converts only the requested row so caller edits are visible
        to subsequent reads.
        """

        value = self.payload.get(payload_key, {})
        return value if isinstance(value, Mapping) else {}

    def _snapshot_time_count(self) -> int:
        value = self.payload.get("time_s", [])
        try:
            if len(value) and np.asarray(value[0]).ndim != 0:
                return int(np.asarray(value, dtype=float).size)
            return int(len(value))
        except TypeError:
            return int(np.array(value, dtype=float).reshape(-1).size)

    def _snapshot_time_at(self, step_index: int) -> float:
        value = self.payload.get("time_s", [])
        try:
            item = value[step_index]
            if np.asarray(item).ndim != 0:
                item = np.asarray(value, dtype=float).reshape(-1)[step_index]
        except (IndexError, KeyError, TypeError):
            item = np.array(value, dtype=float).reshape(-1)[step_index]
        return float(np.asarray(item, dtype=float))

    def _snapshot_rows(self, payload_key: str, step_index: int) -> dict[str, np.ndarray]:
        rows: dict[str, np.ndarray] = {}
        for object_id, history in self._snapshot_history(payload_key).items():
            try:
                if len(history) <= step_index:
                    continue
                row = history[step_index]
            except (IndexError, KeyError, TypeError):
                continue
            try:
                rows[str(object_id)] = np.array(row, dtype=float)
            except (TypeError, ValueError):
                continue
        return rows

    @property
    def analysis(self) -> dict[str, Any]:
        return dict(self.payload.get("analysis", {}) or {})

    @property
    def analysis_study_type(self) -> str:
        analysis = self.analysis
        if bool(analysis.get("enabled", False)):
            return str(analysis.get("study_type", "unknown"))
        if self.is_monte_carlo:
            return "monte_carlo"
        return "single_run"

    @property
    def is_batch_analysis(self) -> bool:
        return self.analysis_study_type in {"monte_carlo", "sensitivity", "covariance"}

    @property
    def is_monte_carlo(self) -> bool:
        return bool(dict(self.payload.get("monte_carlo", {}) or {}).get("enabled", False))

    @property
    def summary(self) -> dict[str, Any]:
        if isinstance(self.payload.get("summary"), dict):
            return dict(self.payload["summary"])
        if isinstance(self.payload.get("run"), dict):
            return dict(self.payload["run"])
        return {}

    @property
    def time_s(self) -> np.ndarray:
        return np.array(self.payload.get("time_s", []), dtype=float).reshape(-1)

    @property
    def truth(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("truth_by_object", {}))

    @property
    def target_reference_orbit(self) -> np.ndarray:
        arr = np.array(self.payload.get("target_reference_orbit_truth", []), dtype=float)
        if arr.size == 0:
            return np.empty((0, 6), dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1, 6)
        return arr

    @property
    def belief(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("belief_by_object", {}))

    @property
    def applied_thrust(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("applied_thrust_by_object", {}))

    @property
    def applied_torque(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("applied_torque_by_object", {}))

    @property
    def knowledge(self) -> dict[str, dict[str, np.ndarray]]:
        return _as_nested_array_map(self.payload.get("knowledge_by_observer", {}))

    @property
    def ground_station_access(self) -> dict[str, Any]:
        return dict(self.payload.get("ground_station_access", {}) or {})

    @property
    def ground_station_measurements(self) -> dict[str, Any]:
        return dict(self.payload.get("ground_station_measurements", {}) or {})

    @property
    def artifacts(self) -> dict[str, Any]:
        if self.is_batch_analysis:
            return dict(self.payload.get("artifacts", {}) or {})
        summary = self.summary
        artifacts: dict[str, Any] = {
            "plots": dict(summary.get("plot_outputs", {}) or {}),
            "animations": dict(summary.get("animation_outputs", {}) or {}),
        }
        orbital_analysis = dict(self.payload.get("orbital_analysis", {}) or {})
        if orbital_analysis:
            artifacts["orbital_analysis"] = {
                "coverage": [dict(item.get("artifacts", {}) or {}) for item in orbital_analysis.get("coverage", [])],
                "directed_links": [
                    dict(item.get("artifacts", {}) or {}) for item in orbital_analysis.get("directed_links", [])
                ],
            }
        if summary.get("history_binary_outputs"):
            artifacts["history_npz"] = dict(summary.get("history_binary_outputs", {}) or {})
        return artifacts

    @property
    def output_dir(self) -> Path:
        return Path(self.config.scenario.outputs.output_dir)

    def review(self) -> Any:
        from sim.review import ReviewWorkspace

        return ReviewWorkspace.open(self.output_dir)

    def evidence_manifest(self) -> dict[str, Any]:
        from sim.plotting.style import get_oel_version
        from sim.review.evidence_capsule import evidence_file_exists, evidence_file_sha256

        output_dir = self.output_dir
        review_db = output_dir / "review" / "run.sqlite"
        review_schema = output_dir / "review" / "schema.json"
        review_saved_views = output_dir / "review" / "saved_views.json"
        review_compressed = output_dir / "review" / "run.sqlite.gz"
        review_capsule = output_dir / "review" / "evidence_capsule.json"
        review_evidence_exists = evidence_file_exists(review_db)
        return {
            "schema_version": 1,
            "scenario_name": self.config.scenario_name,
            "scenario_description": self.config.scenario.scenario_description,
            "study_type": self.analysis_study_type,
            "oel_version": get_oel_version(),
            "config": {
                "source_path": str(self.config.source_path) if self.config.source_path is not None else None,
                "output_dir": str(output_dir),
                "duration_s": float(self.config.scenario.simulator.duration_s),
                "dt_s": float(self.config.scenario.simulator.dt_s),
            },
            "summary": dict(self.summary),
            "metrics": dict(self.metrics),
            "artifacts": _artifact_paths(self.artifacts),
            "review": {
                "enabled": bool(self.config.scenario.outputs.review.enabled),
                "detail": str(self.config.scenario.outputs.review.detail),
                "db_path": str(review_db),
                "db_exists": review_db.is_file(),
                "evidence_exists": review_evidence_exists,
                "storage": (
                    "sqlite"
                    if review_db.is_file()
                    else "sqlite_gzip_capsule"
                    if review_evidence_exists
                    else "missing"
                ),
                "logical_sha256": evidence_file_sha256(review_db) if review_evidence_exists else None,
                "compressed_path": str(review_compressed),
                "compressed_exists": review_compressed.is_file(),
                "capsule_manifest_path": str(review_capsule),
                "capsule_manifest_exists": review_capsule.is_file(),
                "schema_path": str(review_schema),
                "schema_exists": review_schema.is_file(),
                "saved_views_path": str(review_saved_views),
                "saved_views_exists": review_saved_views.is_file(),
            },
        }

    @property
    def metrics(self) -> dict[str, Any]:
        if self.is_monte_carlo:
            return dict(self.payload.get("aggregate_stats", {}) or {})
        if self.analysis_study_type == "sensitivity":
            return {
                "parameter_count": int(self.analysis.get("parameter_count", 0)),
                "run_count": int(self.analysis.get("run_count", 0)),
                "metrics": list(self.analysis.get("metrics", []) or []),
            }
        out = dict(self.summary)
        closest_approach_km = _closest_approach_metric(self.payload)
        if np.isfinite(closest_approach_km):
            out["closest_approach_km"] = float(closest_approach_km)
        return out

    @property
    def num_steps(self) -> int:
        return int(self.time_s.size)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.truth.keys()))

    @property
    def reference_object_id(self) -> str | None:
        value = self.summary.get("reference_object_id")
        text = str(value or "").strip()
        return text or None

    @property
    def primary_pair(self) -> tuple[str, str] | None:
        raw = self.summary.get("primary_object_pair")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            a = str(raw[0])
            b = str(raw[1])
            if a and b and a != b:
                return a, b
        if "chaser" in self.truth and "target" in self.truth:
            return "chaser", "target"
        pairs = self.pairs()
        return pairs[0] if pairs else None

    def pairs(self) -> list[tuple[str, str]]:
        ids = list(self.object_ids)
        return [(a, b) for idx, a in enumerate(ids) for b in ids[idx + 1 :]]

    def snapshot(self, step_index: int) -> SimulationSnapshot:
        if self.is_batch_analysis:
            raise RuntimeError("Snapshots are only available for single-run results.")
        sample_count = self._snapshot_time_count()
        if step_index < 0 or step_index >= sample_count:
            raise IndexError(f"step_index {step_index} is out of range for {sample_count} samples.")

        time_s = self._snapshot_time_at(step_index)
        truth = self._snapshot_rows("truth_by_object", step_index)
        belief = self._snapshot_rows("belief_by_object", step_index)
        thrust = self._snapshot_rows("applied_thrust_by_object", step_index)
        torque = self._snapshot_rows("applied_torque_by_object", step_index)
        from sim.spacecraft_resources.artifacts import resource_snapshot

        return SimulationSnapshot(
            step_index=int(step_index),
            time_s=time_s,
            truth=truth,
            belief=belief,
            applied_thrust=thrust,
            applied_torque=torque,
            ground_segment=deepcopy(
                next(
                    (
                        row
                        for row in self.payload.get("ground_segment", [])
                        if abs(row["time_s"] - time_s) < 1e-9
                    ),
                    {},
                )
            ),
            spacecraft_resources=resource_snapshot(self.payload.get("spacecraft_resources", {}), time_s),
            state_metadata={key: self.summary[key] for key in ("central_body", "origin_center_id", "inertial_axes") if key in self.summary},
        )

    def state_history(self, object_id: str) -> np.ndarray:
        oid = str(object_id)
        if oid == "target_reference":
            return self.target_reference_orbit
        histories = self.truth
        if oid not in histories:
            raise KeyError(f"Unknown object_id {oid!r}. Available objects: {sorted(histories.keys())}")
        return _as_2d_state_history(histories[oid], width=6)

    def _state_history_for_query(self, object_id: str) -> np.ndarray:
        """Return a validated live history without converting every object."""

        oid = str(object_id)
        if oid == "target_reference":
            return _as_2d_state_history(self.target_reference_orbit, width=6)
        histories = self._snapshot_history("truth_by_object")
        if oid not in histories:
            raise KeyError(f"Unknown object_id {oid!r}. Available objects: {sorted(histories.keys())}")
        return _as_2d_state_history(histories[oid], width=6)

    def time_window_mask(self, start_s: float | None = None, end_s: float | None = None) -> np.ndarray:
        t_s = self._snapshot_time_s()
        mask = np.ones(t_s.shape, dtype=bool)
        if start_s is not None:
            mask &= t_s >= float(start_s)
        if end_s is not None:
            mask &= t_s <= float(end_s)
        return mask

    def relative_state(
        self,
        deputy: str,
        chief: str,
        *,
        frame: str = "ric_rect",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> np.ndarray:
        dep = self._state_history_for_query(deputy)
        ref = self._state_history_for_query(chief)
        t_s = self._snapshot_time_s()
        n = int(min(dep.shape[0], ref.shape[0], t_s.size))
        if n <= 0:
            return np.empty((0, 6), dtype=float)
        dep = dep[:n, :6]
        ref = ref[:n, :6]
        mask = self.time_window_mask(start_s=start_s, end_s=end_s)[:n]
        indices = np.flatnonzero(mask)
        frame_key = str(frame or "ric_rect").strip().lower()
        planetary = self.payload.get("summary", {}).get("central_body", "earth") != "earth"
        if planetary and frame_key != "inertial":
            raise ValueError("Planetary relative states require frame='inertial'; Earth ECI/RIC recipes are unsupported")
        if frame_key in {"eci", "inertial"}:
            rel = dep[indices] - ref[indices]
        elif frame_key in {"ric", "ric_rect", "rect", "rectangular"}:
            from sim.utils.frames import eci_relative_to_ric_rect

            rel = (
                np.vstack([eci_relative_to_ric_rect(dep[k, :6], ref[k, :6]) for k in indices])
                if indices.size
                else np.empty((0, 6), dtype=float)
            )
        elif frame_key in {"ric_curv", "curv", "curvilinear"}:
            from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_to_curv

            rows = []
            for k in indices:
                rect = eci_relative_to_ric_rect(dep[k, :6], ref[k, :6])
                rows.append(ric_rect_to_curv(rect, r0_km=float(np.linalg.norm(ref[k, :3]))))
            rel = np.vstack(rows) if rows else np.empty((0, 6), dtype=float)
        else:
            raise ValueError("frame must be one of 'eci', 'ric_rect', or 'ric_curv'.")
        return np.array(rel, dtype=float)

    def range_between(
        self,
        a: str,
        b: str,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
        units: str = "km",
    ) -> np.ndarray:
        rel = self.relative_state(a, b, frame="inertial" if self.summary.get("central_body", "earth") != "earth" else "eci", start_s=start_s, end_s=end_s)
        if rel.size == 0:
            return np.array([], dtype=float)
        return np.linalg.norm(rel[:, :3], axis=1) * _range_scale(units)

    def min_range(
        self,
        a: str,
        b: str,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
        units: str = "km",
    ) -> float:
        ranges = self.range_between(a, b, start_s=start_s, end_s=end_s, units=units)
        finite = ranges[np.isfinite(ranges)]
        return float(np.min(finite)) if finite.size else float("nan")

    def time_of_min_range(
        self,
        a: str,
        b: str,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> float:
        ranges = self.range_between(a, b, start_s=start_s, end_s=end_s)
        times = self.time_s[self.time_window_mask(start_s=start_s, end_s=end_s)]
        n = int(min(ranges.size, times.size))
        if n <= 0:
            return float("nan")
        finite_idx = np.where(np.isfinite(ranges[:n]))[0]
        if finite_idx.size == 0:
            return float("nan")
        local = int(finite_idx[int(np.argmin(ranges[finite_idx]))])
        return float(times[local])

    def collision_event(
        self,
        a: str,
        b: str,
        *,
        radius_km: float,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, Any]:
        radius = _validate_event_radius_km(radius_km)
        rel = self.relative_state(a, b, frame="inertial" if self.summary.get("central_body", "earth") != "earth" else "eci", start_s=start_s, end_s=end_s)
        times = self.time_s[self.time_window_mask(start_s=start_s, end_s=end_s)]
        n = int(min(rel.shape[0], times.size))
        min_range_km = float("nan")
        event_time_s: float | None = None
        if n > 0:
            for idx in range(n):
                pos = np.asarray(rel[idx, :3], dtype=float)
                if not bool(np.all(np.isfinite(pos))):
                    continue
                candidate = float(np.linalg.norm(pos))
                if not np.isfinite(min_range_km) or candidate < min_range_km:
                    min_range_km = candidate
                    event_time_s = float(times[idx])
            for idx in range(n - 1):
                start = np.asarray(rel[idx, :3], dtype=float)
                end = np.asarray(rel[idx + 1, :3], dtype=float)
                if not bool(np.all(np.isfinite(start))) or not bool(np.all(np.isfinite(end))):
                    continue
                candidate, fraction = _segment_minimum_range(start, end)
                if not np.isfinite(min_range_km) or candidate < min_range_km:
                    min_range_km = candidate
                    event_time_s = float(times[idx] + fraction * (times[idx + 1] - times[idx]))
        hit = bool(np.isfinite(min_range_km) and min_range_km <= radius)
        return {
            "event": hit,
            "threshold_km": radius,
            "min_range_km": min_range_km,
            "time_s": event_time_s,
        }

    def keepout_violations(
        self,
        a: str,
        b: str,
        *,
        radius_km: float,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[dict[str, float]]:
        radius = _validate_event_radius_km(radius_km)
        rel = self.relative_state(a, b, frame="inertial" if self.summary.get("central_body", "earth") != "earth" else "eci", start_s=start_s, end_s=end_s)
        times = self.time_s[self.time_window_mask(start_s=start_s, end_s=end_s)]
        n = int(min(rel.shape[0], times.size))
        violations: list[dict[str, float]] = []
        for idx in range(n):
            pos = np.asarray(rel[idx, :3], dtype=float)
            if bool(np.all(np.isfinite(pos))):
                sample_range = float(np.linalg.norm(pos))
                if sample_range <= radius:
                    violations.append(
                        {"time_s": float(times[idx]), "range_km": sample_range, "threshold_km": radius}
                    )
        for idx in range(n - 1):
            start = np.asarray(rel[idx, :3], dtype=float)
            end = np.asarray(rel[idx + 1, :3], dtype=float)
            if not bool(np.all(np.isfinite(start))) or not bool(np.all(np.isfinite(end))):
                continue
            fraction = _segment_sphere_entry_fraction(start, end, radius)
            if fraction is None or fraction <= 0.0:
                continue
            crossing_time = float(times[idx] + fraction * (times[idx + 1] - times[idx]))
            if any(np.isclose(crossing_time, row["time_s"], rtol=0.0, atol=1.0e-12) for row in violations):
                continue
            violations.append({"time_s": crossing_time, "range_km": radius, "threshold_km": radius})
        return sorted(violations, key=lambda row: row["time_s"])

    def first_crossing(
        self,
        values: np.ndarray,
        *,
        threshold: float,
        direction: str = "below",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, Any]:
        arr = np.array(values, dtype=float).reshape(-1)
        time_s = self.time_s
        window_mask = self.time_window_mask(start_s=start_s, end_s=end_s)
        times = time_s[window_mask]
        if arr.size == time_s.size:
            arr = arr[window_mask]
        elif arr.size != times.size:
            n_total = int(min(arr.size, time_s.size))
            arr = arr[:n_total][window_mask[:n_total]]
            times = time_s[:n_total][window_mask[:n_total]]
        n = int(min(arr.size, times.size))
        key = str(direction or "below").strip().lower()
        if key in {"below", "<", "<="}:
            mask = arr[:n] <= float(threshold)
        elif key in {"above", ">", ">="}:
            mask = arr[:n] >= float(threshold)
        else:
            raise ValueError("direction must be 'below' or 'above'.")
        hits = np.where(mask & np.isfinite(arr[:n]))[0]
        if hits.size == 0:
            return {"event": False, "time_s": None, "value": None, "threshold": float(threshold)}
        idx = int(hits[0])
        return {"event": True, "time_s": float(times[idx]), "value": float(arr[idx]), "threshold": float(threshold)}

    def to_records(self, kind: str = "truth", *, object_id: str) -> list[dict[str, Any]]:
        key = str(kind or "truth").strip().lower()
        source_map = {
            "truth": self.truth,
            "belief": self.belief,
            "applied_thrust": self.applied_thrust,
            "thrust": self.applied_thrust,
            "applied_torque": self.applied_torque,
            "torque": self.applied_torque,
        }
        if key not in source_map:
            raise ValueError("kind must be one of 'truth', 'belief', 'applied_thrust', or 'applied_torque'.")
        data = source_map[key]
        oid = str(object_id)
        if oid not in data:
            raise KeyError(f"Unknown object_id {oid!r}. Available objects: {sorted(data.keys())}")
        arr = np.array(data[oid], dtype=float)
        if arr.ndim != 2:
            return []
        if key in {"applied_thrust", "thrust"}:
            columns = ["ax_eci_km_s2", "ay_eci_km_s2", "az_eci_km_s2"]
        elif key in {"applied_torque", "torque"}:
            columns = ["tx_body_nm", "ty_body_nm", "tz_body_nm"]
        else:
            columns = _STATE_COLUMNS[: arr.shape[1]]
        planetary = self.payload.get("summary", {}).get("central_body", "earth") != "earth"
        if planetary:
            columns = [name.replace("_eci_", "_inertial_") for name in columns]
        n = int(min(self.time_s.size, arr.shape[0]))
        records = []
        for idx in range(n):
            row = {"time_s": float(self.time_s[idx]), "object_id": oid}
            if planetary:
                row.update(origin_center_id=self.payload["summary"]["origin_center_id"], inertial_axes="J2000")
            row.update({columns[j]: float(arr[idx, j]) for j in range(min(len(columns), arr.shape[1]))})
            records.append(row)
        return records

    def to_dataframe(self, kind: str = "truth", *, object_id: str) -> Any:
        return _records_dataframe(self.to_records(kind, object_id=object_id))

    def relative_records(
        self,
        deputy: str,
        chief: str,
        *,
        frame: str = "ric_rect",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[dict[str, Any]]:
        rel = self.relative_state(deputy, chief, frame=frame, start_s=start_s, end_s=end_s)
        times = self.time_s[self.time_window_mask(start_s=start_s, end_s=end_s)]
        n = int(min(rel.shape[0], times.size))
        frame_key = str(frame or "ric_rect").strip().lower()
        columns = _ECI_REL_STATE_COLUMNS if frame_key in {"eci", "inertial"} else _RIC_STATE_COLUMNS
        if self.summary.get("central_body", "earth") != "earth":
            columns = [name.replace("_eci", "_inertial") for name in columns]
        records = []
        for idx in range(n):
            row = {"time_s": float(times[idx]), "deputy": str(deputy), "chief": str(chief), "frame": frame_key}
            row.update({columns[j]: float(rel[idx, j]) for j in range(min(len(columns), rel.shape[1]))})
            if self.summary.get("central_body", "earth") != "earth":
                row.update(origin_center_id=self.summary["origin_center_id"], inertial_axes="J2000")
            records.append(row)
        return records

    def relative_dataframe(
        self,
        deputy: str,
        chief: str,
        *,
        frame: str = "ric_rect",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Any:
        return _records_dataframe(
            self.relative_records(deputy, chief, frame=frame, start_s=start_s, end_s=end_s)
        )

    def evaluate_metrics(
        self,
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...],
    ) -> dict[str, Any]:
        return _evaluate_metric_callbacks(self, metrics)
