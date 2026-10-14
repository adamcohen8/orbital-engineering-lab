"""Explicit opt-in extended third-body gravity configuration."""

from __future__ import annotations

from copy import deepcopy

from .primitives import _as_dict, _reject_unknown_fields


def normalize_additional_gravity(raw, origin):
    from sim.dynamics.planetary.additional_gravity import external_body

    from .planetary import normalize_planetary

    path = "simulator.additional_gravity"
    out = deepcopy(_as_dict(raw, path))
    _reject_unknown_fields(out, path, {"ephemeris", "sources"})
    ephem = _as_dict(out.get("ephemeris"), f"{path}.ephemeris")
    _reject_unknown_fields(ephem, f"{path}.ephemeris", {"model", "data_path", "sha256"})
    ephem.setdefault("model", "table")
    if ephem["model"] != "table" or not ephem.get("data_path"):
        raise ValueError("Additional gravity requires a geometric J2000 ephemeris table")
    out["ephemeris"] = ephem
    sources = out.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 16:
        raise ValueError("Additional gravity requires 1..16 sources")
    seen = set()
    for i, source in enumerate(sources):
        p = f"{path}.sources[{i}]"
        source = _as_dict(source, p)
        _reject_unknown_fields(source, p, {"body_id", "gravity", "gravity_frame"})
        body_id = source.get("body_id")
        external_body(body_id)
        if body_id in seen or body_id in {origin, "sun"}:
            raise ValueError(
                "Additional gravity sources must be unique external solid-body fields; solar harmonics unsupported"
            )
        seen.add(body_id)
        gravity = _as_dict(source.get("gravity"), f"{p}.gravity")
        frame = _as_dict(source.get("gravity_frame"), f"{p}.gravity_frame")
        if body_id == "earth" and gravity.get("model") == "earth_static":
            _reject_unknown_fields(
                gravity, f"{p}.gravity", {"model", "coeff_path", "frame_id", "sha256", "max_degree", "max_order"}
            )
            for key in ("coeff_path", "frame_id"):
                if not isinstance(gravity.get(key), str) or not gravity[key]:
                    raise ValueError(f"{p}.gravity.{key} is required")
            gravity.setdefault("max_degree", 20)
            gravity.setdefault("max_order", gravity["max_degree"])
            n, m = gravity["max_degree"], gravity["max_order"]
            if type(n) is not int or type(m) is not int or not 2 <= n <= 20 or not 0 <= m <= n:
                raise ValueError("Additional harmonic gravity qualifies degree 2..20 and order <= degree")
        else:
            normalized = normalize_planetary({"gravity": gravity}, "moon" if body_id == "earth" else body_id)
            gravity = normalized["gravity"]
        if gravity["model"] == "point_mass":
            raise ValueError("Use the existing point-mass third-body selection; additional_gravity requires harmonics")
        if body_id == "earth" and frame.get("model") == "earth":
            _reject_unknown_fields(frame, f"{p}.gravity_frame", {"model", "frame_id", "frame_model", "eop_path"})
            frame.setdefault("frame_model", "iau76_80_eop")
            if frame["frame_model"] != "iau76_80_eop" or not frame.get("eop_path") or not frame.get("frame_id"):
                raise ValueError(
                    "Additional Earth gravity requires an explicit iau76_80_eop frame, frame_id and EOP file"
                )
        else:
            if body_id == "earth" and frame.get("model") != "resource":
                raise ValueError("Additional Earth gravity requires model=earth or an explicit rotation resource")
            normalized = normalize_planetary({"gravity_frame": frame}, "moon" if body_id == "earth" else body_id)
            frame = normalized["gravity_frame"]
            if gravity["model"] == "gmm2b_static" and not source.get("gravity_frame"):
                frame = {"model": "iau1991"}
        sources[i] = {"body_id": body_id, "gravity": gravity, "gravity_frame": frame}
    return out


def validate_additional_gravity(cfg):
    s = cfg.simulator
    orbit = s.dynamics.get("orbit", {})
    if s.initial_jd_utc is None or s.duration_s <= 0:
        raise ValueError("Additional gravity requires an absolute UTC epoch and positive duration")
    if (
        orbit.get("model", "two_body") != "two_body"
        or orbit.get("propagation_method", "special") != "special"
        or orbit.get("numeric_backend", "rust") != "rust"
        or orbit.get("integrator", "rk4") not in {"rk4", "rkf78", "adaptive", "dopri5"}
    ):
        raise ValueError("Additional gravity requires Rust special ONP two_body with RK4, RKF78 or DOPRI5")
    from sim.dynamics.planetary.additional_gravity import external_body

    targets = {external_body(v["body_id"]).center_id for v in s.additional_gravity["sources"]}
    execution = s.execution.get("object_parallelism", {})
    if execution.get("enabled", False) or execution.get("backend", "serial") != "serial":
        raise ValueError("Additional gravity initially requires serial execution")
    if s.central_body == "earth":
        if 301 in targets and orbit.get("third_body_moon", False):
            raise ValueError("Moon gravity selected twice: disable third_body_moon when adding Moon harmonics")
        if s.system_force_models or s.collisions.get("enabled", False):
            raise ValueError("Additional gravity with system forces/collisions is not qualified")
        if s.dynamics.get("rocket", {}).get("enabled", False) or s.dynamics.get("reentry", {}).get("enabled", False):
            raise ValueError("Additional gravity with rockets/reentry is not qualified")
        for obj in cfg.objects.values():
            if obj.enabled and (
                obj.kind != "satellite"
                or obj.propagation_method not in {"", "special"}
                or obj.general
                or obj.force_models
            ):
                raise ValueError(
                    "Additional gravity requires ONP satellites without TLE/general or custom force models"
                )
            if obj.enabled and (
                obj.runtime_profile != "trajectory_only"
                or any(
                    getattr(obj, field, None)
                    for field in (
                        "flight_software",
                        "knowledge",
                        "orbit_control",
                        "attitude_control",
                        "mission_strategy",
                        "mission_execution",
                        "bridge",
                        "reference_orbit",
                        "guidance",
                        "base_guidance",
                        "guidance_modifiers",
                        "mission_objectives",
                    )
                )
            ):
                raise ValueError("Additional gravity initially supports passive trajectory_only satellites")
        if s.dynamics.get("attitude", {}).get("enabled", False):
            raise ValueError("Additional gravity attitude coupling is not qualified")
        if cfg.analysis.enabled or cfg.monte_carlo.enabled:
            raise ValueError("Additional gravity covariance/campaign workflows are not qualified")
    elif targets & {v["target_center_id"] for v in s.planetary["third_bodies"]}:
        raise ValueError("Additional gravity duplicates a planetary point-mass third body")
    from sim.dynamics.planetary.additional_gravity import prepare_additional_gravity

    additional = prepare_additional_gravity(cfg)
    if s.central_body == "earth":
        import numpy as np

        from sim.runtime.state_initialization import _default_truth_from_agent

        for obj in cfg.objects.values():
            if obj.enabled:
                truth = _default_truth_from_agent(obj, target_jd_utc=s.initial_jd_utc, numeric_backend="rust")
                additional.native.stage(0.0, np.r_[truth.position_eci_km, truth.velocity_eci_km_s].tolist())
