from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.ogp import (
    ogp_propagate_teme_batch_accelerated,
    ogp_regime_for_elements,
)
from sim.dynamics.orbit.sgp4 import transform_teme_to_output_frame
from sim.scale._output import staged_sqlite_store
from sim.scale.planner import (
    ScalePlannerConfig,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.store import (
    CurrentTLEState,
    PropagationProductRecord,
    current_ogp_elements,
    load_current_tle_states,
    write_propagation_products,
)
from sim.scale.windows import build_propagation_windows, sample_count_for_window
from sim.security.output_writes import OutputWriter

_OGP_IMPLEMENTATION_FILES = (
    "dynamics/orbit/frames.py",
    "dynamics/orbit/ogp.py",
    "dynamics/orbit/sdp4.py",
    "dynamics/orbit/_sdp4_equations.py",
    "dynamics/orbit/sgp4.py",
    "scale/propagation.py",
)


@enforce_scale_planner_limits("OGP propagation")
def run_ogp_propagation(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("OGP propagation requires scale_catalog.mode: run.")
    if cfg.propagation_model != "ogp":
        raise ValueError("OGP propagation requires scale_catalog.propagation.model: ogp.")
    if cfg.propagation_backend not in {"scalar", "rust"}:
        raise ValueError("Scale OGP propagation supports propagation.backend: scalar or rust.")
    with staged_sqlite_store(cfg.store_path, writer=writer) as store_path:
        states = load_current_tle_states(store_path, object_ids=resolve_selected_object_ids(cfg))
        products, telemetry = build_ogp_products_with_telemetry(cfg, states=states)
        write_summary = write_propagation_products(store_path, products=products)
    sample_errors = int(dict(telemetry or {}).get("sample_error_count", 0) or 0)
    status = "completed" if states and products and sample_errors == 0 else "failed"
    return {
        "schema_version": 1,
        "status": status,
        "errors": (
            (["no current TLE states matched the selected catalog scope"] if not states else [])
            + (["propagation produced no products"] if states and not products else [])
            + ([f"propagation produced {sample_errors} sample error(s)"] if sample_errors else [])
        ),
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "selected_object_count": len(states),
        "chunk_count_per_object": len(_chunk_windows(cfg)),
        "ogp": telemetry,
        "products_seen": write_summary["products_seen"],
        "products_inserted": write_summary["products_inserted"],
        "samples_inserted": write_summary["samples_inserted"],
        "invalidations_completed": write_summary["invalidations_completed"],
        "propagator_family": "OGP",
        "propagator_name": "OGP",
        "non_claims": [
            "Passive OGP propagation products were written for catalog screening.",
            "No conjunction candidates were generated.",
            "No ONP scenarios were generated or validated.",
        ],
    }


def run_sgp4_propagation(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the older Scale SGP4 propagation command."""

    return run_ogp_propagation(config_path, writer=writer)


def build_sgp4_products(cfg: ScalePlannerConfig, *, states: list[CurrentTLEState]) -> list[PropagationProductRecord]:
    products, _telemetry = build_ogp_products_with_telemetry(cfg, states=states)
    return products


def build_ogp_products_with_telemetry(
    cfg: ScalePlannerConfig, *, states: list[CurrentTLEState]
) -> tuple[list[PropagationProductRecord], dict[str, Any]]:
    products, telemetry = build_sgp4_products_for_windows(
        cfg,
        states=states,
        windows=tuple((window.start_jd_utc, window.end_jd_utc, window.step_s) for window in _chunk_windows(cfg)),
    )
    telemetry["source_state_evidence"] = [
        {
            "object_id": state.object_id,
            "od_solution_id": state.od_solution_id,
            "source_product_id": state.source_product_id,
            "reference_epoch_jd_utc": state.tle_epoch_jd_utc,
            "age_at_screening_start_days": float(cfg.start_jd_utc - state.tle_epoch_jd_utc),
            "age_at_screening_end_days": float(
                cfg.start_jd_utc + cfg.duration_s / 86400.0 - state.tle_epoch_jd_utc
            ),
            "source_provenance": state.source_provenance,
        }
        for state in states
    ]
    return products, telemetry


def build_sgp4_products_for_windows(
    cfg: ScalePlannerConfig,
    *,
    states: list[CurrentTLEState],
    windows: tuple[tuple[float, float, float], ...],
) -> tuple[list[PropagationProductRecord], dict[str, Any]]:
    products: list[PropagationProductRecord] = []
    elements_by_state = [current_ogp_elements(state) for state in states]
    regime_by_state = [ogp_regime_for_elements(elements) for elements in elements_by_state]
    backend_labels: set[str] = set()
    batch_execution: list[dict[str, Any]] = []
    error_count = 0
    frame_transform = _frame_transform(cfg.output_frame)
    if not states:
        return products, _ogp_telemetry(
            cfg,
            runtime_backends=[],
            sgp4_count=0,
            sdp4_count=0,
            total_count=0,
            sample_error_count=0,
        )

    for start_jd, end_jd, step_s in windows:
        step = float(step_s)
        sample_count = sample_count_for_window(start_jd_utc=start_jd, end_jd_utc=end_jd, step_s=step)
        sample_times_s = np.array(
            [round((float(start_jd) - float(cfg.start_jd_utc)) * 86400.0, 9) + index * step for index in range(sample_count)],
            dtype=float,
        )
        jd_grid = np.array([float(cfg.start_jd_utc) + float(t_s) / 86400.0 for t_s in sample_times_s], dtype=float)
        time_grid = np.zeros((len(states), sample_count), dtype=float)
        for object_index, elements in enumerate(elements_by_state):
            time_grid[object_index, :] = (jd_grid - float(elements.epoch_jd_utc)) * 1440.0
        batch = ogp_propagate_teme_batch_accelerated(
            elements_by_state, time_grid,
            backend="rust" if cfg.propagation_backend == "rust" else "auto",
            **({"workers": cfg.propagation_workers} if cfg.propagation_backend == "rust" else {}),
        )
        backend_labels.add(str(batch.backend))
        if batch.execution_metadata:
            batch_execution.append({
                "start_jd_utc": float(start_jd), "end_jd_utc": float(end_jd),
                "object_count": len(states), "samples_per_object": sample_count,
                **batch.execution_metadata,
            })

        for object_index, state in enumerate(states):
            samples: list[dict[str, Any]] = []
            for sample_index, jd_utc in enumerate(jd_grid):
                t_s = float(sample_times_s[sample_index])
                error = str(batch.errors[object_index, sample_index] or "")
                if error:
                    error_count += 1
                    samples.append(
                        {"sample_index": sample_index, "jd_utc": float(jd_utc), "t_s": t_s, "error": error}
                    )
                    continue
                pos, vel = transform_teme_to_output_frame(
                    batch.position_teme_km[object_index, sample_index, :],
                    batch.velocity_teme_km_s[object_index, sample_index, :],
                    jd_utc=float(jd_utc),
                    output_frame=cfg.output_frame,
                    frame_transform=frame_transform,
                )
                samples.append(
                    {
                        "sample_index": sample_index,
                        "jd_utc": float(jd_utc),
                        "t_s": t_s,
                        "pos_x_km": float(pos[0]),
                        "pos_y_km": float(pos[1]),
                        "pos_z_km": float(pos[2]),
                        "vel_x_km_s": float(vel[0]),
                        "vel_y_km_s": float(vel[1]),
                        "vel_z_km_s": float(vel[2]),
                        "error": "",
                    }
                )
            status = "error" if any(sample.get("error") for sample in samples) else "completed"
            error = "; ".join(sorted({str(sample.get("error")) for sample in samples if sample.get("error")}))
            regime = regime_by_state[object_index]
            runtime_backend = _runtime_backend_for_regime(regime=regime, runtime_backends=[str(batch.backend)])
            digest = _product_digest(
                {
                    "object_id": state.object_id,
                    "od_solution_id": state.od_solution_id,
                    "model": cfg.propagation_model,
                    "backend": cfg.propagation_backend,
                    "ogp_regime": regime,
                    "runtime_backend": runtime_backend,
                    "output_frame": cfg.output_frame,
                    "frame_transform": frame_transform,
                    "start_jd_utc": start_jd,
                    "end_jd_utc": end_jd,
                    "step_s": step,
                    "tle_digest": state.input_digest,
                    "implementation_digest": _ogp_implementation_digest(cfg.propagation_backend),
                }
            )
            products.append(
                PropagationProductRecord(
                    propagation_product_id=f"prop:{digest}",
                    object_id=state.object_id,
                    source_od_solution_id=state.od_solution_id,
                    model=cfg.propagation_model,
                    backend=cfg.propagation_backend,
                    precision="float64",
                    native_frame="teme",
                    output_frame=cfg.output_frame,
                    frame_transform=frame_transform,
                    start_jd_utc=float(start_jd),
                    end_jd_utc=float(end_jd),
                    step_s=step,
                    sample_count=len(samples),
                    input_digest=digest,
                    samples=samples,
                    status=status,
                    error=error,
                    ogp_regime=regime,
                    runtime_backend=runtime_backend,
                    metadata={
                        "propagator_family": "OGP",
                        "propagator_name": "OGP-SDP4" if regime == "sdp4" else "OGP-SGP4",
                        "configured_model": cfg.propagation_model,
                        "configured_backend": cfg.propagation_backend,
                        "runtime_backend": runtime_backend,
                        **({"batch_execution": dict(batch.execution_metadata)} if batch.execution_metadata else {}),
                    },
                )
            )

    backend_list = sorted(backend_labels)
    telemetry = _ogp_telemetry(
        cfg,
        runtime_backends=backend_list,
        sgp4_count=regime_by_state.count("sgp4"),
        sdp4_count=regime_by_state.count("sdp4"),
        total_count=len(states),
        sample_error_count=error_count,
    )
    if batch_execution:
        telemetry["batch_execution"] = batch_execution
    return products, telemetry


def _ogp_telemetry(
    cfg: ScalePlannerConfig,
    *,
    runtime_backends: list[str],
    sgp4_count: int,
    sdp4_count: int,
    total_count: int,
    sample_error_count: int,
) -> dict[str, Any]:
    return {
        "propagator_family": "OGP",
        "model": cfg.propagation_model,
        "configured_backend": cfg.propagation_backend,
        "runtime_backends": list(runtime_backends),
        "sgp4_backend": _sgp4_backend_label(runtime_backends, sgp4_count),
        "sdp4_backend": ("rust" if cfg.propagation_backend == "rust" else "scalar_context") if sdp4_count else "",
        "object_counts": {
            "total": total_count,
            "sgp4": sgp4_count,
            "sdp4": sdp4_count,
        },
        "sample_error_count": sample_error_count,
    }


@lru_cache(maxsize=2)
def _ogp_implementation_digest(backend: str = "scalar") -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for relative in _OGP_IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    if backend == "rust":
        from sim.rust_orbit_backend import native_extension_path

        adapter = root / "rust_ogp_backend.py"
        digest.update(adapter.read_bytes())
        digest.update(native_extension_path().read_bytes())
    return digest.hexdigest()


def _chunk_windows(cfg: ScalePlannerConfig):
    return build_propagation_windows(
        start_jd_utc=cfg.start_jd_utc,
        duration_s=cfg.duration_s,
        step_s=cfg.step_s,
        chunk_duration_s=cfg.chunk_duration_s,
    )


def _frame_transform(output_frame: str) -> str:
    return "native" if str(output_frame or "teme").strip().lower() == "teme" else "teme_as_eci"


def _sgp4_backend_label(runtime_backends: list[str], object_count: int) -> str:
    if object_count <= 0:
        return ""
    if "ogp_rust" in runtime_backends:
        return "rust"
    if any("sgp4_numba_cpu" in item for item in runtime_backends):
        return "numba_cpu"
    if any("sgp4_scalar_reference" in item for item in runtime_backends):
        return "scalar_reference"
    return "scalar"


def _runtime_backend_for_regime(*, regime: str, runtime_backends: list[str]) -> str:
    if "ogp_rust" in runtime_backends:
        return "rust"
    if regime == "sdp4":
        return "scalar_context"
    if regime == "sgp4":
        return _sgp4_backend_label(runtime_backends, 1)
    return "unknown"


def _product_digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
