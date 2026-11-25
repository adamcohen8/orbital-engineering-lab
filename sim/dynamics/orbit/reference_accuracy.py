from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.ogp import (
    ogp_propagate_teme,
    ogp_propagator_name_for_elements,
    ogp_regime_for_elements,
)
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.utils.io import write_json


def _as_vector(row: dict[str, Any], key: str) -> np.ndarray:
    value = row.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"OGP reference sample requires {key} as a length-3 array.")
    vector = np.array(value, dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"OGP reference sample {key} contains non-finite values.")
    return vector


def _load_reference(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"OGP reference file must contain a JSON object: {path}")
    tle = raw.get("tle")
    if not isinstance(tle, dict):
        raise ValueError("OGP reference file requires a tle object.")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("OGP reference file requires at least one sample.")
    if str(raw.get("reference_frame", "") or "").strip().lower() != "teme":
        raise ValueError("OGP reference file must explicitly declare reference_frame='teme'.")
    return raw


def _runtime_status(rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("status", "")) for row in rows}
    if statuses == {"matched"}:
        return "matched"
    if statuses == {"unsupported"}:
        return "unsupported"
    if "error" in statuses:
        return "error"
    return "mixed"


def run_ogp_reference_vector_validation(
    *,
    reference_path: str | Path,
    output_dir: str | Path,
    backend: str | None = None,
) -> dict[str, Any]:
    """Compare OGP output against checked-in native-TEME reference rows.

    Deep-space OGP-SDP4 fixtures gate the native runtime on position/velocity
    parity while preserving explicit error accounting for unsupported or invalid
    rows if future fixtures exercise those boundaries.
    """

    ref_path = Path(reference_path).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    reference = _load_reference(ref_path)
    tle = dict(reference["tle"])
    elements = parse_tle_lines(str(tle.get("line1", "")), str(tle.get("line2", "")))
    ogp_regime = ogp_regime_for_elements(elements)
    propagator_name = ogp_propagator_name_for_elements(elements)

    rows: list[dict[str, Any]] = []
    pos_err_m: list[float] = []
    vel_err_mm_s: list[float] = []
    dispatch = {} if backend is None else {"backend": backend}
    for sample in reference["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("OGP reference samples must be objects.")
        tsince_min = float(sample.get("tsince_min"))
        if not math.isfinite(tsince_min):
            raise ValueError("OGP reference sample tsince_min must be finite.")
        expected_pos = _as_vector(sample, "position_teme_km")
        expected_vel = _as_vector(sample, "velocity_teme_km_s")
        actual = ogp_propagate_teme(elements, tsince_min, **dispatch)
        if actual.error:
            status = "unsupported" if "OGP-SDP4" in str(actual.error) and "not implemented" in str(actual.error) else "error"
            rows.append(
                {
                    "tsince_min": tsince_min,
                    "status": status,
                    "error": str(actual.error),
                    "pos_err_m": "",
                    "vel_err_mm_s": "",
                    "pos_err_x_m": "",
                    "pos_err_y_m": "",
                    "pos_err_z_m": "",
                    "vel_err_x_mm_s": "",
                    "vel_err_y_mm_s": "",
                    "vel_err_z_mm_s": "",
                }
            )
            continue

        actual_pos = np.array(actual.position_teme_km, dtype=float)
        actual_vel = np.array(actual.velocity_teme_km_s, dtype=float)
        pos_delta_m = (actual_pos - expected_pos) * 1000.0
        vel_delta_mm_s = (actual_vel - expected_vel) * 1.0e6
        pos_norm_m = float(np.linalg.norm(pos_delta_m))
        vel_norm_mm_s = float(np.linalg.norm(vel_delta_mm_s))
        if not math.isfinite(pos_norm_m) or not math.isfinite(vel_norm_mm_s):
            raise ValueError(f"Non-finite propagation difference at {tsince_min} min")
        pos_err_m.append(pos_norm_m)
        vel_err_mm_s.append(vel_norm_mm_s)
        rows.append(
            {
                "tsince_min": tsince_min,
                "status": "matched",
                "error": "",
                "pos_err_m": pos_norm_m,
                "vel_err_mm_s": vel_norm_mm_s,
                "pos_err_x_m": float(pos_delta_m[0]),
                "pos_err_y_m": float(pos_delta_m[1]),
                "pos_err_z_m": float(pos_delta_m[2]),
                "vel_err_x_mm_s": float(vel_delta_mm_s[0]),
                "vel_err_y_mm_s": float(vel_delta_mm_s[1]),
                "vel_err_z_mm_s": float(vel_delta_mm_s[2]),
            }
        )

    csv_path = outdir / "ogp_reference_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matched_count = sum(1 for row in rows if row["status"] == "matched")
    unsupported_count = sum(1 for row in rows if row["status"] == "unsupported")
    error_count = sum(1 for row in rows if row["status"] == "error")
    result: dict[str, Any] = {
        "reference_name": str(reference.get("name", ref_path.stem)),
        "description": str(reference.get("description", "")),
        "regime": str(reference.get("regime", "")),
        "ogp_regime": ogp_regime,
        "expected_propagator_name": propagator_name,
        "reference_path": str(ref_path),
        "reference_frame": str(reference["reference_frame"]),
        "source": dict(reference.get("source", {})) if isinstance(reference.get("source"), dict) else {},
        "sample_count": int(len(rows)),
        "matched_sample_count": int(matched_count),
        "unsupported_sample_count": int(unsupported_count),
        "error_sample_count": int(error_count),
        "runtime_status": _runtime_status(rows),
        "errors_csv": str(csv_path),
        "samples": rows,
    }
    if backend is not None:
        result["backend"] = backend
    if pos_err_m:
        pos_arr = np.array(pos_err_m, dtype=float)
        vel_arr = np.array(vel_err_mm_s, dtype=float)
        result.update(
            {
                "pos_err_rms_m": float(np.sqrt(np.mean(pos_arr * pos_arr))),
                "pos_err_max_m": float(np.max(pos_arr)),
                "vel_err_rms_mm_s": float(np.sqrt(np.mean(vel_arr * vel_arr))),
                "vel_err_max_mm_s": float(np.max(vel_arr)),
            }
        )
    write_json(str(outdir / "ogp_reference_validation_report.json"), result)
    return result


def compare_backends(reference_path: Path, output_dir: Path, *,
                     position_limit_m: float, velocity_limit_mm_s: float) -> dict:
    """Score both dispatch paths independently against the same reference."""
    from inspect import signature

    for limit in (position_limit_m, velocity_limit_mm_s):
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("Accuracy limits must be finite and nonnegative.")
    summary = {
        "reference": str(reference_path), "default_backend": signature(ogp_propagate_teme).parameters["backend"].default,
        "position_limit_m": position_limit_m, "velocity_limit_mm_s": velocity_limit_mm_s, "backends": {},
    }
    for backend in ("python", "rust"):
        report = run_ogp_reference_vector_validation(reference_path=reference_path,
                                                      output_dir=output_dir / backend, backend=backend)
        position = report.get("pos_err_max_m")
        velocity = report.get("vel_err_max_mm_s")
        complete = report["matched_sample_count"] == report["sample_count"]
        summary["backends"][backend] = {
            "model": report["expected_propagator_name"], "epochs": report["sample_count"],
            "compared_epochs": report["matched_sample_count"], "runtime_status": report["runtime_status"],
            "max_position_error_m": position, "max_velocity_error_mm_s": velocity,
            "passed": complete and position <= position_limit_m and velocity <= velocity_limit_mm_s,
            "errors_csv": report["errors_csv"],
        }
    write_json(str(output_dir / "backend_accuracy.json"), summary)
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Measure OGP backend accuracy against independent TEME state vectors.")
    parser.add_argument("--reference", type=Path, help="User-supplied TEME reference JSON; defaults to both archived civil cases.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ogp_reference_accuracy"))
    parser.add_argument("--position-limit-m", type=float, default=1.0)
    parser.add_argument("--velocity-limit-mm-s", type=float, default=1.0)
    args = parser.parse_args()
    data = Path(__file__).with_name("data")
    references = [args.reference] if args.reference else [data / f"{case}_teme_reference.json" for case in ("vanguard1", "cbers2")]
    summaries = [compare_backends(path, args.output_dir / path.stem, position_limit_m=args.position_limit_m,
                                  velocity_limit_mm_s=args.velocity_limit_mm_s) for path in references]
    print(json.dumps(summaries, indent=2))
    return 0 if all(row["passed"] for s in summaries for row in s["backends"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
