"""Single-run payload slicing, assembly, and artifact dispatch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from sim.acceleration.settings import acceleration_context_from_config
from sim.dynamics.attitude.rigid_body import get_attitude_guardrail_stats
from sim.reporting.single_run_artifacts import SingleRunArtifactContext, write_single_run_artifacts
from sim.reporting.single_run_payload import SingleRunPayloadContext, build_single_run_payload

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine


@dataclass(frozen=True)
class _SingleRunPayloadParts:
    n_used: int
    t_s: np.ndarray
    truth_hist: dict[str, np.ndarray]
    target_reference_orbit_truth: np.ndarray | None
    belief_hist: dict[str, np.ndarray]
    thrust_hist: dict[str, np.ndarray]
    torque_hist: dict[str, np.ndarray]
    desired_attitude_hist: dict[str, np.ndarray]
    knowledge_hist: dict[str, dict[str, np.ndarray]]
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]]
    rocket_metrics: dict[str, np.ndarray]
    reentry_metrics: dict[str, dict[str, np.ndarray]]
    thrust_stats: dict[str, dict[str, Any]]


class SingleRunPayloadAssembler:
    """Build reporting views without owning engine lifecycle or physics."""

    def __init__(
        self,
        engine: _SingleRunEngine,
        *,
        initialization_metadata_builder: Callable[..., dict[str, dict[str, Any]]],
    ) -> None:
        self.engine = engine
        self.initialization_metadata_builder = initialization_metadata_builder

    def _events_through_termination(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        engine = self.engine
        if not bool(engine.terminated_early) or engine.termination_time_s is None:
            return rows
        cutoff = float(engine.termination_time_s) + 1.0e-12
        retained: list[dict[str, Any]] = []
        for row in rows:
            try:
                row_time_s = float(dict(row).get("t_s"))
            except (TypeError, ValueError):
                retained.append(row)
                continue
            if row_time_s <= cutoff:
                clipped = dict(row)
                for key in ("interval_end_t_s", "end_t_s"):
                    try:
                        if key in clipped and float(clipped[key]) > cutoff:
                            clipped[key] = float(engine.termination_time_s)
                    except (TypeError, ValueError):
                        pass
                retained.append(clipped)
        return retained

    def build_parts(self) -> _SingleRunPayloadParts:
        engine = self.engine
        n_used = engine.current_index + 1
        t_out = engine.t_s[:n_used]
        truth_out = {key: value[:n_used, :] for key, value in engine.truth_hist.items()}
        target_reference_orbit_out = (
            None
            if engine.target_reference_orbit_hist is None
            else engine.target_reference_orbit_hist[:n_used, :]
        )
        belief_out = {key: value[:n_used, :] for key, value in engine.belief_hist.items()}
        thrust_out = {key: value[:n_used, :] for key, value in engine.thrust_hist.items()}
        torque_out = {key: value[:n_used, :] for key, value in engine.torque_hist.items()}
        desired_attitude_out = {
            key: value[:n_used, :] for key, value in engine.desired_attitude_hist.items()
        }
        knowledge_out = {
            observer: {target: arr[:n_used, :] for target, arr in by_target.items()}
            for observer, by_target in engine.knowledge_hist.items()
        }
        knowledge_measurements_out = {
            observer: {target: arr[:n_used, :] for target, arr in by_target.items()}
            for observer, by_target in getattr(engine, "knowledge_measurement_hist", {}).items()
        }
        rocket_metrics_out: dict[str, np.ndarray] = {}
        if engine.rocket is not None:
            rocket_object_id = str(getattr(engine.rocket, "object_id", "rocket") or "rocket")
            if engine.rocket_stage_hist is not None:
                rocket_metrics_out["stage_index"] = engine.rocket_stage_hist[:n_used]
            if engine.rocket_q_dyn_hist is not None:
                rocket_metrics_out["q_dyn_pa"] = engine.rocket_q_dyn_hist[:n_used]
            if engine.rocket_mach_hist is not None:
                rocket_metrics_out["mach"] = engine.rocket_mach_hist[:n_used]
            if rocket_object_id in engine.throttle_hist:
                rocket_metrics_out["throttle_cmd"] = engine.throttle_hist[rocket_object_id][:n_used]
            for metric_key, metric_hist in getattr(engine, "rocket_metric_hists", {}).items():
                rocket_metrics_out[metric_key] = metric_hist[:n_used]
        reentry_metrics_out = {
            object_id: {key: hist[:n_used] for key, hist in metrics.items()}
            for object_id, metrics in getattr(engine, "reentry_metric_hists", {}).items()
        }
        termination_time_s = getattr(engine, "termination_time_s", None)
        if (
            bool(getattr(engine, "terminated_early", False))
            and termination_time_s is not None
            and n_used >= 2
            and float(t_out[-2]) < float(termination_time_s) < float(t_out[-1])
        ):
            interval_s = float(t_out[-1] - t_out[-2])
            fraction = float((float(termination_time_s) - float(t_out[-2])) / interval_s)
            t_out = np.array(t_out, copy=True)
            t_out[-1] = float(termination_time_s)

            def interpolate_last(hist: np.ndarray, *, unit_slice: slice | None = None) -> np.ndarray:
                out = np.array(hist, copy=True)
                out[-1] = out[-2] + fraction * (out[-1] - out[-2])
                if unit_slice is not None:
                    unit_values = np.asarray(out[-1][unit_slice], dtype=float)
                    norm = float(np.linalg.norm(unit_values))
                    if np.isfinite(norm) and norm > 0.0:
                        out[-1][unit_slice] = unit_values / norm
                return out

            truth_out = {key: interpolate_last(value, unit_slice=slice(6, 10)) for key, value in truth_out.items()}
            target_reference_orbit_out = (
                None
                if target_reference_orbit_out is None
                else interpolate_last(target_reference_orbit_out)
            )
            belief_out = {key: interpolate_last(value) for key, value in belief_out.items()}
            desired_attitude_out = {
                key: interpolate_last(value, unit_slice=slice(0, 4))
                for key, value in desired_attitude_out.items()
            }
            knowledge_out = {
                observer: {target: interpolate_last(arr) for target, arr in by_target.items()}
                for observer, by_target in knowledge_out.items()
            }
            knowledge_measurements_out = {
                observer: {
                    target: np.vstack((arr[:-1], np.full_like(arr[-1:], np.nan)))
                    for target, arr in by_target.items()
                }
                for observer, by_target in knowledge_measurements_out.items()
            }
            rocket_metrics_out = {key: interpolate_last(value) for key, value in rocket_metrics_out.items()}
            reentry_metrics_out = {
                object_id: {key: interpolate_last(hist) for key, hist in metrics.items()}
                for object_id, metrics in reentry_metrics_out.items()
            }
        if not bool(getattr(engine, "terminated_early", False)):
            thrust_stats = {
                object_id: {
                    "burn_samples": int(engine.burn_samples_by_object.get(object_id, 0)),
                    "max_accel_km_s2": float(engine.max_accel_km_s2_by_object.get(object_id, 0.0)),
                    "total_dv_m_s": float(engine.total_dv_m_s_by_object.get(object_id, 0.0)),
                }
                for object_id in thrust_out
            }
        else:
            # Runtime accumulators include the entire final integration step. Rebuild
            # these values from the clipped output timeline after early termination so
            # no post-termination impulse leaks into the run summary.
            thrust_stats = {}
            sample_dt_s = np.diff(
                np.asarray(t_out, dtype=float),
                prepend=float(t_out[0]) if t_out.size else 0.0,
            )
            for object_id, history in thrust_out.items():
                accel = np.asarray(history, dtype=float)
                magnitudes = np.linalg.norm(np.nan_to_num(accel, nan=0.0), axis=1)
                n = min(sample_dt_s.size, magnitudes.size)
                active = magnitudes[:n] > 0.0
                thrust_stats[object_id] = {
                    "burn_samples": int(np.count_nonzero(active & (sample_dt_s[:n] > 0.0))),
                    "max_accel_km_s2": float(np.max(magnitudes[:n])) if n else 0.0,
                    "total_dv_m_s": float(
                        np.sum(magnitudes[:n] * np.clip(sample_dt_s[:n], 0.0, None)) * 1000.0
                    ),
                }
        return _SingleRunPayloadParts(
            n_used=n_used,
            t_s=t_out,
            truth_hist=truth_out,
            target_reference_orbit_truth=target_reference_orbit_out,
            belief_hist=belief_out,
            thrust_hist=thrust_out,
            torque_hist=torque_out,
            desired_attitude_hist=desired_attitude_out,
            knowledge_hist=knowledge_out,
            knowledge_measurement_hist=knowledge_measurements_out,
            rocket_metrics=rocket_metrics_out,
            reentry_metrics=reentry_metrics_out,
            thrust_stats=thrust_stats,
        )

    def payload_from_parts(self, parts: _SingleRunPayloadParts) -> dict[str, Any]:
        engine = self.engine
        runtime_profile = engine.runtime_profiler.payload(
            completed_steps=int(max(parts.n_used - 1, 0)),
            object_count=len(engine.agents),
        )
        runtime_profile["executor"] = {
            "object_step_backend": str(getattr(engine.object_step_executor, "backend_name", "unknown")),
            "object_step_workers": int(getattr(engine.object_step_executor, "max_workers", 1) or 1),
            "planner": dict(getattr(engine, "object_execution_plan", {}) or {}),
        }
        object_propagation = {
            object_id: asdict(provider.metadata())
            for object_id, provider in engine.general_propagation.items()
        }
        for object_id, agent in engine.agents.items():
            if object_id in object_propagation:
                continue
            propagator = getattr(getattr(agent, "dynamics", None), "orbit_propagator", None)
            if getattr(propagator, "state_frame", "eci") != "eci" or engine.cfg.simulator.additional_gravity is not None:
                object_propagation[object_id] = propagator.propagation_metadata()

        payload = build_single_run_payload(
            SingleRunPayloadContext(
                cfg=engine.cfg,
                planetary_environment=getattr(engine, "planetary_environment", None),
                planetary_force_plans={oid: agent.dynamics.orbit_propagator.plan for oid, agent in engine.agents.items()} if getattr(engine, "planetary_environment", None) is not None else None,
                object_ids=list(engine.agents.keys()),
                dt_s=engine.dt,
                t_s=parts.t_s,
                truth_hist=parts.truth_hist,
                target_reference_orbit_truth=parts.target_reference_orbit_truth,
                belief_hist=parts.belief_hist,
                thrust_hist=parts.thrust_hist,
                torque_hist=parts.torque_hist,
                desired_attitude_hist=parts.desired_attitude_hist,
                knowledge_hist=parts.knowledge_hist,
                knowledge_measurement_hist=parts.knowledge_measurement_hist,
                bridge_hist={key: self._events_through_termination(rows) for key, rows in engine.bridge_hist.items()},
                controller_debug_hist={
                    key: self._events_through_termination(rows) for key, rows in engine.controller_debug_hist.items()
                },
                command_decision_hist={
                    key: self._events_through_termination(rows) for key, rows in engine.command_decision_hist.items()
                },
                flight_software_evidence={
                    object_id: runtime.review_evidence()
                    for object_id, agent in engine.agents.items()
                    if (runtime := getattr(agent, "flight_software_runtime", None)) is not None
                },
                rocket_throttle_cmd=np.asarray(parts.rocket_metrics.get("throttle_cmd", np.array([]))),
                rocket_metrics=parts.rocket_metrics,
                reentry_metrics=parts.reentry_metrics,
                thrust_stats=parts.thrust_stats,
                runtime_profile=runtime_profile,
                object_initialization=self.initialization_metadata_builder(engine.cfg, engine.object_configs),
                object_propagation=object_propagation,
                attitude_guardrail_stats=get_attitude_guardrail_stats(engine.attitude_guardrail_stats),
                knowledge_detection_by_observer={
                    object_id: agent.knowledge_base.detection_summary()
                    for object_id, agent in engine.agents.items()
                    if agent.knowledge_base is not None
                }
                if engine.worker_knowledge_detection_by_observer is None
                else dict(engine.worker_knowledge_detection_by_observer),
                knowledge_consistency_by_observer={
                    object_id: agent.knowledge_base.consistency_summary()
                    for object_id, agent in engine.agents.items()
                    if agent.knowledge_base is not None
                }
                if engine.worker_knowledge_consistency_by_observer is None
                else dict(engine.worker_knowledge_consistency_by_observer),
                terminated_early=engine.terminated_early,
                termination_reason=engine.termination_reason,
                termination_time_s=engine.termination_time_s,
                termination_object_id=engine.termination_object_id,
                rocket_inserted=engine.rocket_inserted,
                rocket_insertion_time_s=engine.rocket_insertion_time_s,
            )
        )

        if getattr(engine, "planetary_domain_exit", None) is not None:
            payload["summary"]["gravity_domain_exit"] = dict(engine.planetary_domain_exit)
        if engine.cfg.simulator.central_body == "earth" and engine.cfg.simulator.additional_gravity is not None:
            from sim.reporting.additional_gravity import attach_earth_evidence
            attach_earth_evidence(payload, engine, parts)
        resources = {
            aid: [row for row in rows if row["time_s"] <= float(parts.t_s[-1]) + 1e-9]
            for aid, rows in getattr(engine, "resource_hist", {}).items()
        }
        if getattr(engine, "ground_segment", None) is not None:
            from copy import deepcopy
            payload["ground_segment"] = deepcopy([r for r in engine.ground_segment.ground.history if r["time_s"] <= float(parts.t_s[-1]) + 1e-9])
        if resources:
            payload["spacecraft_resources"] = resources
        payload["impulsive_maneuvers"] = list(engine.impulsive_maneuvers)
        payload["collision_events"] = list(engine.collision_events)
        if engine.collision_events:
            payload["summary"]["collisions"] = {"count": len(engine.collision_events)}
        if engine.impulsive_maneuvers:
            payload["summary"]["impulsive_maneuvers"] = {
                "count": len(engine.impulsive_maneuvers),
                "total_delta_v_m_s_by_object": {
                    object_id: float(sum(
                        np.linalg.norm(row["delta_v_eci_m_s"])
                        for row in engine.impulsive_maneuvers if row["object_id"] == object_id
                    ))
                    for object_id in sorted({row["object_id"] for row in engine.impulsive_maneuvers})
                },
            }
        return payload

    def primary_rocket_throttle_history(self) -> np.ndarray:
        engine = self.engine
        if not engine.throttle_hist or engine.rocket is None:
            return np.array([])
        rocket_object_id = str(getattr(engine.rocket, "object_id", "rocket") or "rocket")
        return engine.throttle_hist.get(rocket_object_id, np.array([]))

    def ensure_full_history_allowed(self) -> None:
        if self.engine.history_mode == "dynamic":
            raise RuntimeError("Dynamic history mode is only supported for step-driven game sessions.")

    def build_run_payload(self) -> dict[str, Any]:
        self.ensure_full_history_allowed()
        return self.payload_from_parts(self.build_parts())

    def write_artifacts(self, payload: dict[str, Any], parts: _SingleRunPayloadParts) -> dict[str, Any]:
        engine = self.engine
        return write_single_run_artifacts(
            payload,
            SingleRunArtifactContext(
                cfg=engine.cfg,
                outdir=engine.outdir,
                t_s=parts.t_s,
                truth_hist=parts.truth_hist,
                target_reference_orbit_truth=parts.target_reference_orbit_truth,
                belief_hist=parts.belief_hist,
                thrust_hist=parts.thrust_hist,
                torque_hist=parts.torque_hist,
                desired_attitude_hist=parts.desired_attitude_hist,
                knowledge_hist=parts.knowledge_hist,
                knowledge_measurement_hist=parts.knowledge_measurement_hist,
                rocket_metrics=parts.rocket_metrics,
                reentry_metrics=parts.reentry_metrics,
                bridge_hist={
                    key: self._events_through_termination(rows) for key, rows in engine.bridge_hist.items()
                },
                object_state_frames=dict(payload.get("object_state_frames", {}) or {}),
                object_propagation=dict(payload.get("object_propagation", {}) or {}),
                extra_artifacts=dict(payload.get("extra_artifacts", {}) or {}),
            ),
        )

    def build_payload(self) -> dict[str, Any]:
        engine = self.engine
        self.ensure_full_history_allowed()
        if not bool(getattr(engine, "_acceleration_context_active", False)):
            with acceleration_context_from_config(engine.cfg):
                engine._acceleration_context_active = True
                try:
                    return self.build_payload()
                finally:
                    engine._acceleration_context_active = False
        parts = self.build_parts()
        payload = self.payload_from_parts(parts)
        return self.write_artifacts(payload, parts)
