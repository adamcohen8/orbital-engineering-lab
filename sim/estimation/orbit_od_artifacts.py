
import io

from sim.numeric_backend import normalize_numeric_backend
from sim.security.output_writes import staged_output_file, write_output_text
# ruff: noqa: F401,F403,F405,I001
from .orbit_od_common import *

def _coerce_observation_packet(packet: ObservationPacket | Mapping[str, Any]) -> ObservationPacket:
    if isinstance(packet, ObservationPacket):
        return packet
    if hasattr(packet, "to_dict"):
        return observation_packet_from_dict(packet.to_dict())
    return observation_packet_from_dict(packet)


def _candidate_packet(
    *,
    object_id: str,
    role: str,
    state: np.ndarray,
    epoch_jd_utc: float | None,
    source_packet: ObservationPacket,
    base_specs: Mapping[str, Any] | None,
    parameter_values: Mapping[str, float],
    attitude_history_rows: Sequence[Mapping[str, Any]] | None = None,
) -> MissionInputPacket:
    packet = ingest_state_vector(
        object_id=object_id,
        role=role,
        position=state[:3].tolist(),
        velocity=state[3:].tolist(),
        epoch_jd_utc=epoch_jd_utc,
        source_label="dynamics_orbit_determination",
    ).to_dict()
    obj = packet["objects"][object_id]
    specs = _candidate_specs(base_specs, parameter_values)
    if specs:
        obj["specs"] = specs
    attitude_rows = list(attitude_history_rows or [])
    if attitude_rows:
        first_att = dict(attitude_rows[0])
        obj.setdefault("initial_state", {})["attitude_quat_bn"] = list(first_att["attitude_quat_bn"])
        if first_att.get("angular_rate_body_rad_s") is not None:
            obj.setdefault("initial_state", {})["angular_rate_body_rad_s"] = list(first_att["angular_rate_body_rad_s"])
    obj["state_type"] = "dynamics_od_estimated_state_vector"
    obj["state_estimate"] = {
        "method": "dynamics_orbit_least_squares",
        "estimated_parameter_values": dict(parameter_values),
        "source_observation_summary": dict(source_packet.to_dict().get("summary", {}) or {}),
    }
    packet["source"] = {
        "type": "dynamics_orbit_determination",
        "observation_source": dict(source_packet.to_dict().get("source", {}) or {}),
    }
    warnings = list(packet.get("warnings", []) or [])
    warnings.append("State estimate comes from OEL-dynamics least-squares OD over structured ECI observations.")
    packet["warnings"] = _unique(warnings)
    packet["validation"]["status"] = "ready_with_warnings"
    return MissionInputPacket(packet)


def _local_initial_state(observations: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, str]:
    if len(observations) < 2:
        raise ValueError("at least two observations are required for an initial state guess.")
    first = dict(observations[0])
    second = dict(observations[1])
    r0 = np.array(first["position_eci_km"], dtype=float).reshape(3)
    if "velocity_eci_km_s" in first:
        v0 = np.array(first["velocity_eci_km_s"], dtype=float).reshape(3)
        method = "first_observation_velocity"
    else:
        dt = float(second["time_s"]) - float(first["time_s"])
        if dt <= 0.0:
            raise ValueError("observation times must be strictly increasing.")
        r1 = np.array(second["position_eci_km"], dtype=float).reshape(3)
        v0 = None
        method = "first_observation_chord"
        if len(observations) >= 3:
            third = dict(observations[2])
            dt2 = float(third["time_s"]) - float(second["time_s"])
            r2 = np.array(third["position_eci_km"], dtype=float).reshape(3)
            gibbs_v0 = _gibbs_velocity_at_first(r0, r1, r2)
            if gibbs_v0 is not None:
                v0 = gibbs_v0
                method = "gibbs_three_position"
            elif abs(dt2 - dt) <= max(1.0e-9, 1.0e-6 * abs(dt)):
                v0 = (-3.0 * r0 + 4.0 * r1 - r2) / (2.0 * dt)
                method = "second_order_forward_difference"
        if v0 is None:
            v0 = (r1 - r0) / dt
    return np.hstack((r0, v0)), method


def _gibbs_velocity_at_first(r0: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray | None:
    r0_norm = float(np.linalg.norm(r0))
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    if min(r0_norm, r1_norm, r2_norm) <= 0.0:
        return None
    angle01 = _angle_between(r0, r1)
    angle12 = _angle_between(r1, r2)
    if min(angle01, angle12) < np.deg2rad(5.0):
        return None
    z01 = np.cross(r0, r1)
    z12 = np.cross(r1, r2)
    z20 = np.cross(r2, r0)
    d_vec = z01 + z12 + z20
    n_vec = r0_norm * z12 + r1_norm * z20 + r2_norm * z01
    s_vec = r0 * (r1_norm - r2_norm) + r1 * (r2_norm - r0_norm) + r2 * (r0_norm - r1_norm)
    d_norm = float(np.linalg.norm(d_vec))
    n_norm = float(np.linalg.norm(n_vec))
    if d_norm <= 1.0e-12 or n_norm <= 1.0e-12:
        return None
    coplanar_scale = abs(float(np.dot(r0 / r0_norm, z12 / max(float(np.linalg.norm(z12)), 1.0e-12))))
    if coplanar_scale > 1.0e-2:
        return None
    b0 = np.cross(d_vec, r0)
    return float(np.sqrt(EARTH_MU_KM3_S2 / (n_norm * d_norm))) * (b0 / r0_norm + s_vec)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.arccos(np.clip(float(np.dot(a, b)) / denom, -1.0, 1.0)))


def _candidate_artifact(
    packet: MissionInputPacket,
    *,
    object_id: str,
    scenario_name: str,
    output_dir: Path,
    duration_s: float,
    dt_s: float,
    dynamics_model: str,
    j2: bool,
    drag: bool,
    srp: bool,
    atmosphere_model: str | None,
    orbit_force_model: Mapping[str, Any] | None = None,
    attitude_source: str = "none",
    attitude_mode: str = "sun_track",
    attitude_body_axis: np.ndarray | None = None,
    attitude_controller: str = "surrogate_snap",
    attitude_history_rows: Sequence[Mapping[str, Any]] | None = None,
    maneuver: Mapping[str, Any] | None = None,
) -> ScenarioArtifact:
    artifact = build_basic_propagation_scenario(
        packet,
        scenario_name=scenario_name,
        output_dir=output_dir,
        duration_s=max(float(duration_s), float(dt_s)),
        dt_s=float(dt_s),
        initial_jd_utc=_first_epoch_jd(packet),
        dynamics_model=dynamics_model,
        j2=j2,
        review_detail="standard",
    )
    raw = artifact.to_dict()
    orbit = raw.setdefault("simulator", {}).setdefault("dynamics", {}).setdefault("orbit", {})
    orbit["drag"] = bool(drag)
    orbit["srp"] = bool(srp)
    if atmosphere_model:
        orbit["atmosphere_model"] = str(atmosphere_model).strip().lower()
        raw.setdefault("simulator", {}).setdefault("environment", {})["atmosphere_model"] = (
            str(atmosphere_model).strip().lower()
        )
    force_model = _normalize_orbit_force_model(
        orbit_force_model,
        dynamics_model=dynamics_model,
        j2=j2,
        drag=drag,
        srp=srp,
        atmosphere_model=atmosphere_model,
    )
    environment_overrides = dict(force_model.pop("environment", {}) or {})
    orbit.update(_deepcopy_jsonable(force_model))
    if force_model.get("atmosphere_model"):
        raw.setdefault("simulator", {}).setdefault("environment", {})["atmosphere_model"] = str(
            force_model["atmosphere_model"]
        )
    if environment_overrides:
        environment = raw.setdefault("simulator", {}).setdefault("environment", {})
        environment.update(_deepcopy_jsonable(environment_overrides))
    _apply_attitude_source_to_candidate_artifact(
        raw,
        object_id=object_id,
        attitude_source=attitude_source,
        attitude_mode=attitude_mode,
        attitude_body_axis=attitude_body_axis,
        attitude_controller=attitude_controller,
        attitude_history_rows=attitude_history_rows,
        dt_s=dt_s,
    )
    if maneuver:
        _apply_scheduled_maneuver_to_candidate_artifact(raw, object_id=object_id, maneuver=maneuver, dt_s=dt_s)
    outputs = raw.setdefault("outputs", {})
    outputs["mode"] = "save"
    outputs["plots"] = {"enabled": False, "figure_ids": []}
    outputs["animations"] = {"enabled": False, "types": []}
    outputs["stats"] = {"print_summary": False, "save_json": False, "save_csv": False, "save_full_log": False}
    outputs["review"] = {"enabled": False}
    return ScenarioArtifact.from_dict(raw)


def _normalize_orbit_force_model(
    force_model: Mapping[str, Any] | None,
    *,
    dynamics_model: str,
    j2: bool,
    drag: bool,
    srp: bool,
    atmosphere_model: str | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "model": str(dynamics_model or "two_body"),
        "j2": bool(j2),
        "drag": bool(drag),
        "srp": bool(srp),
    }
    if atmosphere_model:
        merged["atmosphere_model"] = str(atmosphere_model).strip().lower()
    if force_model:
        merged = _deep_merge_dicts(merged, dict(force_model))
    if merged.get("atmosphere_model") is not None:
        merged["atmosphere_model"] = str(merged["atmosphere_model"]).strip().lower()
    if merged.get("model") is not None:
        merged["model"] = str(merged["model"] or "two_body")
    return _deepcopy_jsonable(merged)


def _apply_scheduled_maneuver_to_candidate_artifact(
    raw: dict[str, Any],
    *,
    object_id: str,
    maneuver: Mapping[str, Any],
    dt_s: float,
) -> None:
    obj = raw.setdefault("objects", {}).setdefault(object_id, {})
    burn_duration_s = float(maneuver.get("burn_duration_s", dt_s) or dt_s)
    burn = {
        "start_time_s": float(maneuver.get("time_s", 0.0)),
        "duration_s": max(burn_duration_s, float(dt_s), 1.0e-12),
        "frame": str(maneuver.get("frame", "ric") or "ric"),
        "delta_v_m_s": [
            float(x) for x in np.array(maneuver.get("delta_v_m_s", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        ],
    }
    existing = dict(obj.get("flight_software", {}) or {})
    existing_params = dict(existing.get("params", {}) or {})
    if existing.get("stack") == "fsw.attitude_reference":
        mode = existing_params.pop("reference_mode", "quaternion")
        existing_params = {
            "attitude_reference_mode": mode,
            "attitude_quaternion_bn": existing_params.pop("quaternion_bn", [1.0, 0.0, 0.0, 0.0]),
            "attitude_boresight_body": existing_params.pop("boresight_body", [1.0, 0.0, 0.0]),
            "attitude_replay_times_s": existing_params.pop("replay_times_s", []),
            "attitude_replay_quaternions_bn": existing_params.pop("replay_quaternions_bn", []),
            **existing_params,
        }
    burns = list(existing_params.get("scheduled_burns", []) or [])
    burns.append(burn)
    existing_params["scheduled_burns"] = burns
    required_acceleration_m_s2 = float(np.linalg.norm(np.asarray(burn["delta_v_m_s"], dtype=float))) / float(
        burn["duration_s"]
    )
    existing_params["max_acceleration_m_s2"] = max(
        float(existing_params.get("max_acceleration_m_s2", 0.0) or 0.0),
        required_acceleration_m_s2,
    )
    obj["flight_software"] = {
        "stack": "fsw.orbit_reference",
        "hardware_profile": "hardware.ideal_wrench.v1",
        "task_period_s": float(dt_s),
        "params": existing_params,
    }


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = {str(k): _deepcopy_jsonable(v) for k, v in dict(base or {}).items()}
    for raw_key, raw_value in dict(override or {}).items():
        key = str(raw_key)
        if isinstance(raw_value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge_dicts(out[key], raw_value)
        else:
            out[key] = _deepcopy_jsonable(raw_value)
    return out


def _deepcopy_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _deepcopy_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_deepcopy_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_attitude_source(value: str | None) -> str:
    key = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "observed": "observed_history",
        "observed_history": "observed_history",
        "history": "observed_history",
        "modeled_inline": "modeled_inline",
        "inline": "modeled_inline",
        "modeled": "modeled_inline",
        "modeled_replay": "modeled_replay",
        "replay": "modeled_replay",
    }
    if key not in aliases:
        raise ValueError("attitude_source must be none, observed_history, modeled_inline, or modeled_replay.")
    return aliases[key]


def _attitude_body_axis(value: Sequence[float] | None) -> np.ndarray:
    axis = np.array([0.0, 0.0, 1.0] if value is None else list(value), dtype=float).reshape(-1)
    if axis.size != 3:
        raise ValueError("attitude_body_axis must contain three values.")
    n = float(np.linalg.norm(axis))
    if n <= 0.0 or not np.isfinite(n):
        raise ValueError("attitude_body_axis must be finite and nonzero.")
    return axis / n


def _coerce_attitude_history(value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    rows_raw: Sequence[Mapping[str, Any]]
    if isinstance(value, Mapping):
        if "samples" in value:
            rows_raw = list(value.get("samples", []) or [])
        elif "times_s" in value and "attitude_quat_bn" in value:
            times = list(value.get("times_s", []) or [])
            quats = list(value.get("attitude_quat_bn", []) or [])
            rates = value.get("angular_rate_body_rad_s")
            rate_rows = list(rates or []) if rates is not None else [None] * len(times)
            if len(times) != len(quats) or len(times) != len(rate_rows):
                raise ValueError(
                    "attitude history times, quaternions, and angular rates must have equal lengths."
                )
            rows_raw = [
                {"time_s": t, "attitude_quat_bn": q, "angular_rate_body_rad_s": w}
                for t, q, w in zip(times, quats, rate_rows, strict=True)
            ]
        else:
            rows_raw = []
    else:
        rows_raw = list(value or [])
    rows: list[dict[str, Any]] = []
    previous_time: float | None = None
    for idx, raw in enumerate(rows_raw):
        item = dict(raw or {})
        if "attitude_quat_bn" not in item:
            continue
        if item.get("time_s") is None:
            raise ValueError(f"attitude history sample {idx} is missing time_s.")
        time_s = float(item["time_s"])
        if previous_time is not None and time_s <= previous_time:
            raise ValueError("attitude history times must be strictly increasing.")
        q = np.array(item["attitude_quat_bn"], dtype=float).reshape(-1)
        if q.size != 4:
            raise ValueError(f"attitude history sample {idx} attitude_quat_bn must be length-4.")
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0.0 or not np.isfinite(q_norm):
            raise ValueError(f"attitude history sample {idx} attitude_quat_bn must be finite and nonzero.")
        row: dict[str, Any] = {"time_s": time_s, "attitude_quat_bn": (q / q_norm).tolist()}
        if item.get("angular_rate_body_rad_s") is not None:
            w = np.array(item["angular_rate_body_rad_s"], dtype=float).reshape(-1)
            if w.size != 3 or not np.all(np.isfinite(w)):
                raise ValueError(f"attitude history sample {idx} angular_rate_body_rad_s must be length-3 finite.")
            row["angular_rate_body_rad_s"] = w.tolist()
        rows.append(row)
        previous_time = time_s
    return rows


def _apply_attitude_source_to_candidate_artifact(
    raw: dict[str, Any],
    *,
    object_id: str,
    attitude_source: str,
    attitude_mode: str,
    attitude_body_axis: np.ndarray | None,
    attitude_controller: str,
    attitude_history_rows: Sequence[Mapping[str, Any]] | None,
    dt_s: float,
) -> None:
    source = _normalize_attitude_source(attitude_source)
    if source == "none":
        return
    raw.setdefault("simulator", {}).setdefault("dynamics", {}).setdefault("attitude", {})["enabled"] = True
    raw["simulator"]["dynamics"]["attitude"]["attitude_substep_s"] = float(dt_s)
    objects = raw.setdefault("objects", {})
    if object_id not in objects:
        raise ValueError(f"candidate artifact does not contain object_id '{object_id}'.")
    obj = objects[object_id]
    axis = _attitude_body_axis(attitude_body_axis)
    if source == "modeled_inline":
        mode = _v2_attitude_mode(attitude_mode)
        obj["flight_software"] = {
            "stack": "fsw.attitude_reference",
            "hardware_profile": "hardware.ideal_wrench.v1",
            "task_period_s": float(dt_s),
            "params": {
                "reference_mode": mode,
                "boresight_body": axis.tolist(),
                "max_torque_n_m": 0.1,
            },
        }
        return

    rows = _coerce_attitude_history(attitude_history_rows)
    if not rows:
        raise ValueError(f"attitude_source='{source}' requires attitude history rows.")
    replay_t0 = float(rows[0]["time_s"])
    replay_times_s = [float(row["time_s"]) - replay_t0 for row in rows]
    obj["flight_software"] = {
        "stack": "fsw.attitude_reference",
        "hardware_profile": "hardware.ideal_wrench.v1",
        "task_period_s": float(dt_s),
        "params": {
            "reference_mode": "replay",
            "replay_times_s": replay_times_s,
            "replay_quaternions_bn": [list(row["attitude_quat_bn"]) for row in rows],
            "max_torque_n_m": 0.1,
        },
    }


def _v2_attitude_mode(value: str) -> str:
    key = str(value or "sun_track").strip().lower().replace("-", "_")
    aliases = {
        "sun_track": "sun",
        "sun": "sun",
        "hold_eci": "quaternion",
        "quaternion": "quaternion",
        "nadir": "nadir",
        "nadir_track": "nadir",
        "velocity": "velocity",
        "velocity_track": "velocity",
        "ric": "ric",
        "hold_ric": "ric",
    }
    if key not in aliases:
        raise ValueError(f"attitude_mode {value!r} is not supported by the GNC v2 reference stack")
    return aliases[key]


def _build_modeled_attitude_history(
    *,
    object_id: str,
    role: str,
    state: np.ndarray,
    epoch_jd_utc: float | None,
    source_packet: ObservationPacket,
    base_specs: Mapping[str, Any] | None,
    scenario_name: str,
    output_root: Path,
    duration_s: float,
    dt_s: float,
    dynamics_model: str,
    j2: bool,
    drag: bool,
    srp: bool,
    atmosphere_model: str | None,
    orbit_force_model: Mapping[str, Any] | None,
    attitude_mode: str,
    attitude_body_axis: np.ndarray,
    attitude_controller: str,
) -> list[dict[str, Any]]:
    candidate_packet = _candidate_packet(
        object_id=object_id,
        role=role,
        state=state,
        epoch_jd_utc=epoch_jd_utc,
        source_packet=source_packet,
        base_specs=base_specs,
        parameter_values={},
    )
    artifact = _candidate_artifact(
        candidate_packet,
        object_id=object_id,
        scenario_name=f"{scenario_name}_modeled_attitude_replay_source",
        output_dir=output_root / "_od_eval_scratch" / "modeled_attitude_replay_source",
        duration_s=duration_s,
        dt_s=dt_s,
        dynamics_model=dynamics_model,
        j2=j2,
        drag=drag,
        srp=srp,
        atmosphere_model=atmosphere_model,
        orbit_force_model=orbit_force_model,
        attitude_source="modeled_inline",
        attitude_mode=attitude_mode,
        attitude_body_axis=attitude_body_axis,
        attitude_controller=attitude_controller,
    )
    from sim.api import SimulationWorkspace

    payload = SimulationWorkspace().run_payload(artifact)
    sim_t = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = dict(payload.get("truth_by_object", {}) or {})
    if object_id not in truth:
        raise ValueError(f"modeled attitude payload did not include object_id '{object_id}'.")
    hist = np.array(truth[object_id], dtype=float)
    if hist.ndim != 2 or hist.shape[0] != sim_t.size or hist.shape[1] < 13:
        raise ValueError("modeled attitude payload does not contain quaternion/body-rate history.")
    return [
        {
            "time_s": float(t),
            "attitude_quat_bn": hist[idx, 6:10].tolist(),
            "angular_rate_body_rad_s": hist[idx, 10:13].tolist(),
        }
        for idx, t in enumerate(sim_t)
    ]


def _state_from_parameters(initial_state: np.ndarray, values: Mapping[str, float]) -> np.ndarray:
    state = np.array(initial_state, dtype=float).reshape(6).copy()
    state[:3] += np.array([values.get("dx_m", 0.0), values.get("dy_m", 0.0), values.get("dz_m", 0.0)]) / 1000.0
    state[3:] += (
        np.array([values.get("dvx_mm_s", 0.0), values.get("dvy_mm_s", 0.0), values.get("dvz_mm_s", 0.0)]) / 1.0e6
    )
    return state


def _base_spec_cd(specs: Mapping[str, Any] | None) -> float:
    raw = dict(specs or {})
    for key in ("cd", "drag_cd"):
        if raw.get(key) is not None:
            return float(raw[key])
    aero = raw.get("aero")
    if isinstance(aero, Mapping):
        for key in ("cd", "drag_cd"):
            if aero.get(key) is not None:
                return float(aero[key])
    return 2.2


def _set_spec_cd(specs: dict[str, Any], value: float) -> None:
    cd = float(value)
    if isinstance(specs.get("aero"), Mapping) and "cd" not in specs and "drag_cd" not in specs:
        aero = dict(specs.get("aero") or {})
        aero["cd"] = cd
        specs["aero"] = aero
    else:
        specs["cd"] = cd


def _derived_estimated_parameters(
    estimated_parameters: Sequence[Mapping[str, Any]],
    *,
    base_specs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values = {str(item.get("name")): float(item.get("value", 0.0)) for item in estimated_parameters}
    out: dict[str, Any] = {}
    if "cd_scale" in values:
        base_cd = _base_spec_cd(base_specs)
        out["base_cd"] = float(base_cd)
        out["estimated_cd"] = float(base_cd * values["cd_scale"])
    return out


def _spec_has_geometry_profile(specs: Mapping[str, Any] | None) -> bool:
    raw = dict(specs or {})
    if any(raw.get(key) for key in ("geometry_profile_path", "area_profile_path", "attitude_area_profile_path")):
        return True
    geometry = raw.get("geometry")
    if isinstance(geometry, Mapping) and any(
        geometry.get(key) for key in ("profile_path", "area_profile_path", "attitude_area_profile_path")
    ):
        return True
    aero = raw.get("aero")
    return bool(
        isinstance(aero, Mapping) and any(aero.get(key) for key in ("geometry_profile_path", "area_profile_path"))
    )


def _projected_attitude_variation_warning(attitude_rows: Sequence[Mapping[str, Any]]) -> bool:
    quats = [
        np.array(row["attitude_quat_bn"], dtype=float).reshape(4)
        for row in attitude_rows
        if row.get("attitude_quat_bn") is not None
    ]
    if len(quats) < 2:
        return True
    q0 = quats[0] / max(float(np.linalg.norm(quats[0])), 1.0e-12)
    max_angle = 0.0
    for q in quats[1:]:
        qn = q / max(float(np.linalg.norm(q)), 1.0e-12)
        dot = float(abs(np.dot(q0, qn)))
        max_angle = max(max_angle, float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))
    return bool(max_angle < np.deg2rad(1.0))


def _state_history_from_payload(payload: Mapping[str, Any], *, object_id: str) -> tuple[np.ndarray, np.ndarray]:
    sim_t = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = dict(payload.get("truth_by_object", {}) or {})
    if object_id not in truth:
        raise ValueError(f"OEL payload did not include object_id '{object_id}'.")
    sim_x = np.array(truth[object_id], dtype=float)[:, :6]
    if sim_t.size == 0 or sim_x.shape[0] != sim_t.size:
        raise ValueError("OEL payload time/state history has an unexpected shape.")
    return sim_t, sim_x


def _states_at_epochs(t_src_s: np.ndarray, x_src: np.ndarray, t_query_s: np.ndarray) -> np.ndarray:
    t_src = np.array(t_src_s, dtype=float).reshape(-1)
    x = np.array(x_src, dtype=float)
    t_query = np.array(t_query_s, dtype=float).reshape(-1)
    if t_query.size == 0:
        return np.empty((0, x.shape[1]), dtype=float)
    indices = np.searchsorted(t_src, t_query)
    if np.any(indices >= t_src.size) or not np.allclose(
        t_src[indices],
        t_query,
        rtol=1.0e-12,
        atol=1.0e-10,
    ):
        raise ValueError("OD propagation did not produce an exact state for every observation epoch.")
    return x[indices]


def _whiten_state_observation_residuals(
    simulated_states: np.ndarray,
    *,
    observations: Sequence[Mapping[str, Any]],
    reference_positions: np.ndarray,
    reference_velocities: np.ndarray | None,
    position_sigmas: np.ndarray,
    velocity_sigmas: np.ndarray | None,
    covariance_factors: Sequence[np.ndarray] | None = None,
    numeric_backend: str = "rust",
) -> np.ndarray:
    residual_blocks: list[np.ndarray] = []
    factor_blocks: list[np.ndarray] = []
    for index, observation in enumerate(observations):
        residual = np.asarray(simulated_states[index, :3] - reference_positions[index], dtype=float)
        sigmas = [float(position_sigmas[index])] * 3
        if reference_velocities is not None and velocity_sigmas is not None:
            residual = np.concatenate(
                (residual, np.asarray(simulated_states[index, 3:] - reference_velocities[index], dtype=float))
            )
            sigmas.extend([float(velocity_sigmas[index])] * 3)
        if covariance_factors is None:
            covariance, _source = observation_covariance(
                observation,
                sigmas=sigmas,
                dimension=residual.size,
            )
            factor = prepare_covariance_whitener(
                covariance,
                dimension=residual.size,
                field_name=f"observation {observation.get('observation_id', index)!r} covariance",
            )
        else:
            factor = np.asarray(covariance_factors[index], dtype=float)
        residual_blocks.append(residual)
        factor_blocks.append(factor)
    if not residual_blocks:
        return np.zeros(0, dtype=float)
    if normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_estimation_backend import measurement_whiten_variable_rows

        return measurement_whiten_variable_rows(residual_blocks, factor_blocks)
    return np.concatenate(
        [
            whiten_residual_with_factor(
                residual,
                factor,
                field_name=f"observation {observations[index].get('observation_id', index)!r} covariance",
            )
            for index, (residual, factor) in enumerate(zip(residual_blocks, factor_blocks, strict=True))
        ]
    )


def _prepare_state_observation_factors(
    *,
    observations: Sequence[Mapping[str, Any]],
    position_sigmas: np.ndarray,
    velocity_sigmas: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    factors: list[np.ndarray] = []
    for index, observation in enumerate(observations):
        sigmas = [float(position_sigmas[index])] * 3
        if velocity_sigmas is not None:
            sigmas.extend([float(velocity_sigmas[index])] * 3)
        covariance, _source = observation_covariance(
            observation,
            sigmas=sigmas,
            dimension=len(sigmas),
        )
        factors.append(
            prepare_covariance_whitener(
                covariance,
                dimension=len(sigmas),
                field_name=f"observation {observation.get('observation_id', index)!r} covariance",
            )
        )
    return tuple(factors)


def _label_state_residual_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    observations: Sequence[Mapping[str, Any]],
    include_velocity: bool,
) -> list[dict[str, Any]]:
    components = ["x_km", "y_km", "z_km"]
    if include_velocity:
        components.extend(["vx_km_s", "vy_km_s", "vz_km_s"])
    width = len(components)
    labeled: list[dict[str, Any]] = []
    for decision in decisions:
        row = dict(decision)
        residual_index = int(row["residual_index"])
        observation_index, component_index = divmod(residual_index, width)
        if observation_index < len(observations):
            observation = observations[observation_index]
            row.update(
                {
                    "observation_id": str(observation.get("observation_id", observation_index)),
                    "measurement_type": str(observation.get("measurement_type", "eci_position")),
                    "time_s": float(observation["time_s"]),
                    "component": components[component_index],
                    "partition": "fit",
                    "residual_space": "cholesky_whitened_component",
                }
            )
        labeled.append(row)
    return labeled


def _state_error_metrics(
    sim_x: np.ndarray, ref_position_km: np.ndarray, ref_velocity_km_s: np.ndarray | None
) -> dict[str, Any]:
    if np.asarray(sim_x).shape[0] == 0:
        out: dict[str, Any] = {
            "sample_count": 0,
            "position_rms_m": float("nan"),
            "position_max_m": float("nan"),
            "final_position_error_m": float("nan"),
            "position_axis_rms_m": {"x": float("nan"), "y": float("nan"), "z": float("nan")},
        }
        if ref_velocity_km_s is not None:
            out.update(
                {
                    "velocity_rms_mm_s": float("nan"),
                    "velocity_max_mm_s": float("nan"),
                    "final_velocity_error_mm_s": float("nan"),
                }
            )
        return out
    pos_err_m = (np.array(sim_x[:, :3], dtype=float) - np.array(ref_position_km, dtype=float)) * 1000.0
    pos_norm = np.linalg.norm(pos_err_m, axis=1)
    out: dict[str, Any] = {
        "sample_count": int(np.asarray(sim_x).shape[0]),
        "position_rms_m": float(np.sqrt(np.mean(pos_norm**2))),
        "position_max_m": float(np.max(pos_norm)),
        "final_position_error_m": float(pos_norm[-1]),
        "position_axis_rms_m": {
            "x": float(np.sqrt(np.mean(pos_err_m[:, 0] ** 2))),
            "y": float(np.sqrt(np.mean(pos_err_m[:, 1] ** 2))),
            "z": float(np.sqrt(np.mean(pos_err_m[:, 2] ** 2))),
        },
    }
    if ref_velocity_km_s is not None:
        vel_err_mm_s = (np.array(sim_x[:, 3:], dtype=float) - np.array(ref_velocity_km_s, dtype=float)) * 1.0e6
        vel_norm = np.linalg.norm(vel_err_mm_s, axis=1)
        out.update(
            {
                "velocity_rms_mm_s": float(np.sqrt(np.mean(vel_norm**2))),
                "velocity_max_mm_s": float(np.max(vel_norm)),
                "final_velocity_error_mm_s": float(vel_norm[-1]),
            }
        )
    return out


def _write_error_csv(
    path: Path,
    t_s: np.ndarray,
    sim_x: np.ndarray,
    ref_position_km: np.ndarray,
    ref_velocity_km_s: np.ndarray | None,
    *,
    observations: Sequence[Mapping[str, Any]] | None = None,
    partition: str = "unassigned",
) -> None:
    rows: list[dict[str, Any]] = []
    pos_err_m = (sim_x[:, :3] - ref_position_km) * 1000.0
    vel_err_mm_s = (sim_x[:, 3:] - ref_velocity_km_s) * 1.0e6 if ref_velocity_km_s is not None else None
    for idx, tt in enumerate(t_s):
        observation = dict(observations[idx]) if observations is not None else {}
        row = {
            "observation_id": str(observation.get("observation_id", f"observation:{idx:06d}")),
            "partition": str(partition),
            "residual_kind": "postfit" if partition == "fit" else "prediction",
            "frame": "ECI",
            "epoch_evaluation_method": "simulation_session_variable_step_exact",
            "measurement_type": str(observation.get("measurement_type", "eci_position")),
            "time_s": float(tt),
            "observed_x_km": float(ref_position_km[idx, 0]),
            "observed_y_km": float(ref_position_km[idx, 1]),
            "observed_z_km": float(ref_position_km[idx, 2]),
            "predicted_x_km": float(sim_x[idx, 0]),
            "predicted_y_km": float(sim_x[idx, 1]),
            "predicted_z_km": float(sim_x[idx, 2]),
            "dx_m": float(pos_err_m[idx, 0]),
            "dy_m": float(pos_err_m[idx, 1]),
            "dz_m": float(pos_err_m[idx, 2]),
            "position_error_m": float(np.linalg.norm(pos_err_m[idx])),
        }
        position_sigma_km = observation.get("position_sigma_km")
        if position_sigma_km is not None:
            sigma = max(float(position_sigma_km), 1.0e-12)
            row.update(
                {
                    "position_sigma_km": sigma,
                    "normalized_dx": float((sim_x[idx, 0] - ref_position_km[idx, 0]) / sigma),
                    "normalized_dy": float((sim_x[idx, 1] - ref_position_km[idx, 1]) / sigma),
                    "normalized_dz": float((sim_x[idx, 2] - ref_position_km[idx, 2]) / sigma),
                }
            )
        if vel_err_mm_s is not None:
            row.update(
                {
                    "observed_vx_km_s": float(ref_velocity_km_s[idx, 0]),
                    "observed_vy_km_s": float(ref_velocity_km_s[idx, 1]),
                    "observed_vz_km_s": float(ref_velocity_km_s[idx, 2]),
                    "predicted_vx_km_s": float(sim_x[idx, 3]),
                    "predicted_vy_km_s": float(sim_x[idx, 4]),
                    "predicted_vz_km_s": float(sim_x[idx, 5]),
                    "dvx_mm_s": float(vel_err_mm_s[idx, 0]),
                    "dvy_mm_s": float(vel_err_mm_s[idx, 1]),
                    "dvz_mm_s": float(vel_err_mm_s[idx, 2]),
                    "velocity_error_mm_s": float(np.linalg.norm(vel_err_mm_s[idx])),
                }
            )
            velocity_sigma_km_s = observation.get("velocity_sigma_km_s")
            if velocity_sigma_km_s is not None:
                sigma = max(float(velocity_sigma_km_s), 1.0e-12)
                row.update(
                    {
                        "velocity_sigma_km_s": sigma,
                        "normalized_dvx": float((sim_x[idx, 3] - ref_velocity_km_s[idx, 0]) / sigma),
                        "normalized_dvy": float((sim_x[idx, 4] - ref_velocity_km_s[idx, 1]) / sigma),
                        "normalized_dvz": float((sim_x[idx, 5] - ref_velocity_km_s[idx, 2]) / sigma),
                    }
                )
        rows.append(row)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_output_text(path, stream.getvalue())


def _write_error_plot(path: Path, t_s: np.ndarray, position_error_km: np.ndarray, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err_m = np.array(position_error_km, dtype=float) * 1000.0
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(t_s, err_m[:, 0], label="x", linewidth=1.3)
    ax.plot(t_s, err_m[:, 1], label="y", linewidth=1.3)
    ax.plot(t_s, err_m[:, 2], label="z", linewidth=1.3)
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    with staged_output_file(path) as physical_path:
        fig.savefig(physical_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_report_md(path: Path, result: Mapping[str, Any]) -> None:
    gates = dict(result.get("quality_gates", {}) or {})
    verdict = dict(result.get("verdict", {}) or {})
    warnings = list(gates.get("warnings", []) or [])
    lines = [
        "# Dynamics Orbit Determination",
        "",
        f"- Method: `{result.get('method')}`",
        f"- Object: `{result.get('object_id')}`",
        f"- Dynamics model: `{result.get('dynamics_model')}`",
        f"- J2: `{result.get('j2')}`",
        f"- Drag/SRP: `{result.get('drag')}` / `{result.get('srp')}`",
        f"- Atmosphere model: `{result.get('atmosphere_model')}`",
        f"- Estimate spec: `{result.get('estimate_spec')}`",
        f"- Attitude source: `{result.get('attitude_source', 'none')}`",
        f"- Attitude mode: `{result.get('attitude_mode', '')}`",
        f"- Fit duration: {float(result.get('fit_duration_s', 0.0)):.3f} s",
        f"- Holdout duration: {float(result.get('holdout_duration_s', 0.0)):.3f} s",
        f"- Observations: {int(result.get('observation_count', 0))}",
        "",
        "## Verdict",
        "",
        f"- Evidence status: `{verdict.get('evidence_status', gates.get('evidence_status', 'unknown'))}`",
        f"- Summary: {verdict.get('summary', 'Review OD metrics and warnings before use.')}",
        f"- Analyst action: `{verdict.get('analyst_action', 'review_warnings_before_use')}`",
        f"- Holdout acceptable: `{verdict.get('holdout_acceptable', gates.get('holdout_acceptable', False))}`",
        "",
        "## Evidence Status",
        "",
        f"- Status: `{gates.get('evidence_status', 'unknown')}`",
        f"- Solver success: `{gates.get('solver_success', False)}`",
        f"- Fit improved initial guess: `{gates.get('fit_improved_prefit_rms', False)}`",
        f"- Covariance valid: `{gates.get('covariance_valid', False)}`",
        f"- Holdout degradation ratio: {_fmt_optional(gates.get('holdout_degradation_ratio'))}",
        f"- Parameters at bounds: {len(list(gates.get('parameter_bounds_hit', []) or []))}",
        "",
        "### Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None.")
    lines.extend(["", "### Non-Claims", ""])
    lines.extend(f"- {item}" for item in list(gates.get("non_claims", []) or []))
    lines.extend(["", "## Estimated Parameters", ""])
    for item in list(result.get("estimated_parameters", []) or []):
        unit = str(item.get("unit") or "")
        suffix = "" if unit in {"", "1"} else f" {unit}"
        lines.append(f"- {item.get('name')}: {float(item.get('value', 0.0)):.9g}{suffix}")
    derived = dict(result.get("derived_parameters", {}) or {})
    if derived:
        lines.extend(["", "## Derived Parameters", ""])
        if derived.get("estimated_cd") is not None:
            lines.append(
                f"- Estimated Cd: {float(derived.get('estimated_cd', 0.0)):.9g} "
                f"(base Cd {float(derived.get('base_cd', 0.0)):.9g})"
            )
    fit = dict(result.get("fit_metrics", {}) or {})
    hold = dict(result.get("holdout_metrics", {}) or {})
    lines.extend(
        [
            "",
            "## Fit Metrics",
            "",
            f"- Position RMS: {float(fit.get('position_rms_m', 0.0)):.6f} m",
            f"- Position Max: {float(fit.get('position_max_m', 0.0)):.6f} m",
            "",
            "## Holdout Metrics",
            "",
            f"- Position RMS: {float(hold.get('position_rms_m', 0.0)):.6f} m",
            f"- Position Max: {float(hold.get('position_max_m', 0.0)):.6f} m",
            f"- Final Position Error: {float(hold.get('final_position_error_m', 0.0)):.6f} m",
        ]
    )
    maneuver = dict(result.get("maneuver_detection", {}) or {})
    if maneuver:
        best = dict(maneuver.get("best_candidate", {}) or {})
        lines.extend(
            [
                "",
                "## Maneuver Detection",
                "",
                f"- Status: `{maneuver.get('status', 'unknown')}`",
                f"- Candidates evaluated: {int(maneuver.get('candidate_count', 0))}",
                f"- Supported candidates: {int(maneuver.get('supported_candidate_count', 0))}",
            ]
        )
        if best:
            lines.extend(
                [
                    f"- Best candidate time: {float(best.get('time_s', 0.0)):.6f} s",
                    f"- Best candidate delta-v norm: {float(best.get('delta_v_norm_m_s', 0.0)):.6f} m/s",
                    f"- Best candidate improvement ratio: {_fmt_optional(best.get('improvement_ratio'))}",
                ]
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for key in (
        "fitted_mission_input_packet_path",
        "estimated_parameters_path",
        "fit_residuals_csv",
        "holdout_errors_csv",
        "fit_plot_path",
        "holdout_plot_path",
        "materialized_fit_config_path",
        "materialized_prediction_config_path",
        "report_json_path",
    ):
        lines.append(f"- {key}: `{result.get(key, '')}`")
    if maneuver:
        for key, value in dict(maneuver.get("artifacts", {}) or {}).items():
            lines.append(f"- maneuver_{key}: `{value}`")
    write_output_text(path, "\n".join(lines) + "\n")


def _first_epoch_jd(packet: MissionInputPacket) -> float | None:
    for obj in packet.objects.values():
        initial = dict(obj.get("initial_state", {}) or {})
        if initial.get("epoch_jd_utc") is not None:
            return float(initial["epoch_jd_utc"])
    return None


def _default_dt_from_times(times: np.ndarray) -> float:
    diffs = np.diff(np.array(times, dtype=float).reshape(-1))
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return 10.0
    return float(max(min(float(np.min(positive)), 60.0), 1.0))


def _ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(float(numerator)) or not np.isfinite(float(denominator)) or abs(float(denominator)) <= 1.0e-12:
        return None
    return float(numerator) / float(denominator)


def _is_covariance_valid(covariance: np.ndarray | None) -> bool:
    if covariance is None:
        return False
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.size == 0:
        return False
    if not np.all(np.isfinite(cov)):
        return False
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-12):
        return False
    eigvals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    if np.any(eigvals < -1.0e-12):
        return False
    cond = np.linalg.cond(cov)
    return bool(np.isfinite(cond) and cond < 1.0e16)


def _parameter_bound_hits(parameter_metadata: list[dict[str, Any]], *, rel_tol: float = 1.0e-6) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in parameter_metadata:
        name = str(item.get("name", ""))
        value = float(item.get("value", np.nan))
        lower = float(item.get("lower", -np.inf))
        upper = float(item.get("upper", np.inf))
        scale = max(
            abs(value) if np.isfinite(value) else 0.0,
            abs(lower) if np.isfinite(lower) else 0.0,
            abs(upper) if np.isfinite(upper) else 0.0,
            1.0,
        )
        tol = float(rel_tol) * scale
        if not np.isfinite(value):
            hits.append({"name": name, "side": "nonfinite", "value": value})
        elif np.isfinite(lower) and value <= lower + tol:
            hits.append({"name": name, "side": "lower", "value": value, "bound": lower})
        elif np.isfinite(upper) and value >= upper - tol:
            hits.append({"name": name, "side": "upper", "value": value, "bound": upper})
    return hits


def _fmt_optional(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return "`not_available`"


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out



def _candidate_specs(base_specs, parameter_values):
    """Apply OD parameter policy once for both artifacts and native contexts."""
    specs = dict(base_specs or {})
    if "drag_scale" in parameter_values:
        base = float(specs.get("drag_area_m2", specs.get("area_m2", 1.0)))
        specs["drag_area_m2"] = base * float(parameter_values["drag_scale"])
    if "cd_scale" in parameter_values:
        _set_spec_cd(specs, _base_spec_cd(specs) * float(parameter_values["cd_scale"]))
    if "srp_scale" in parameter_values:
        key = "srp_area_m2" if "srp_area_m2" in specs else "solar_area_m2"
        base = float(specs.get(key, specs.get("area_m2", specs.get("drag_area_m2", 1.0))))
        specs[key] = base * float(parameter_values["srp_scale"])
    return specs

__all__ = [name for name in globals() if not name.startswith("__")]
