# ruff: noqa: F401,F403,F405,I001
import hashlib
import json

from .bench_common import *


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _should_use_tqdm() -> bool:
    return not _is_truthy_env("OEL_PLAIN_PROGRESS")


def _make_plain_progress_reporter(label: str):
    def _report(index: int, total: int, detail: str) -> None:
        print(f"{label}: {index}/{total} - {detail}")

    return _report


def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object.")
    return dict(value)


def _deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    tokens = [tok for tok in str(path).split(".") if tok]
    if not tokens:
        raise ValueError("Sweep path must be non-empty.")
    cur: Any = root
    for token in tokens[:-1]:
        if isinstance(cur, dict):
            nxt = cur.get(token)
            if nxt is None:
                nxt = {}
                cur[token] = nxt
        elif isinstance(cur, list) and token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(cur):
                raise ValueError(f"Path '{path}' list index {idx} is out of range.")
            nxt = cur[idx]
        else:
            raise ValueError(f"Path '{path}' cannot descend through token '{token}'.")
        cur = nxt
    final = tokens[-1]
    if isinstance(cur, dict):
        cur[final] = deepcopy(value)
    elif isinstance(cur, list) and final.isdigit():
        idx = int(final)
        if idx < 0 or idx >= len(cur):
            raise ValueError(f"Path '{path}' list index {idx} is out of range.")
        cur[idx] = deepcopy(value)
    else:
        raise ValueError(f"Path '{path}' cannot set token '{final}'.")


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_sweep_value(v) for v in value) + "]"
    if value is None:
        return "none"
    return str(value)


def _slug_token(text: str) -> str:
    signed = str(text or "").strip().lower().replace("-", "m").replace("+", "p")
    out = re.sub(r"[^A-Za-z0-9]+", "_", signed).strip("_")
    return out or "value"


def _sweep_labels(paths: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    used: set[str] = set()
    token_lists = {path: [tok for tok in str(path).split(".") if tok] for path in paths}
    for path in paths:
        tokens = token_lists[path]
        candidate_parts = [tokens[-1]] if tokens else ["value"]
        idx = 2
        label = _slug_token("_".join(candidate_parts))
        while label in used:
            if idx <= len(tokens):
                candidate_parts.insert(0, tokens[-idx])
                idx += 1
            else:
                candidate_parts = tokens
            label = _slug_token("_".join(candidate_parts))
        used.add(label)
        labels[path] = label
    return labels


def _parse_variant_pointer(raw: Any, *, context: str) -> ControllerVariantPointer:
    d = _as_dict(raw, context)
    object_id = str(d.get("object_id", "") or "").strip()
    slot = str(d.get("slot", "") or "").strip()
    pointer = _as_dict(d.get("controller", d.get("pointer")), f"{context}.controller")
    if not object_id:
        raise ValueError(f"{context}.object_id must be non-empty.")
    if slot not in {"flight_software", "base_guidance"}:
        raise ValueError(f"{context}.slot must be one of: flight_software, base_guidance.")
    return ControllerVariantPointer(object_id=object_id, slot=slot, pointer=pointer)


def _parse_variant_overrides(raw: Any, *, context: str) -> tuple[ControllerVariantOverride, ...]:
    if raw is None:
        return ()
    out: list[ControllerVariantOverride] = []
    if isinstance(raw, dict):
        for path, value in raw.items():
            path_txt = str(path or "").strip()
            if not path_txt:
                raise ValueError(f"{context} keys must be non-empty dotted paths.")
            out.append(ControllerVariantOverride(path=path_txt, value=value))
        return tuple(out)
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            d = _as_dict(item, f"{context}[{idx}]")
            path_txt = str(d.get("path", "") or "").strip()
            if not path_txt:
                raise ValueError(f"{context}[{idx}].path must be non-empty.")
            if "value" not in d:
                raise ValueError(f"{context}[{idx}].value is required.")
            out.append(ControllerVariantOverride(path=path_txt, value=d.get("value")))
        return tuple(out)
    raise ValueError(f"{context} must be a mapping or list of path/value entries.")


def _variant_with(
    *,
    name: str,
    pointer: dict[str, Any],
    description: str,
    pointers: tuple[ControllerVariantPointer, ...],
    overrides: tuple[ControllerVariantOverride, ...],
) -> ControllerVariant:
    if not pointer and not pointers and not overrides:
        raise ValueError("variants[*] must define controller, pointers, or overrides.")
    return ControllerVariant(
        name=name,
        pointer=pointer,
        description=description,
        pointers=pointers,
        overrides=overrides,
    )


def _expand_variant_entry(raw_variant: dict[str, Any]) -> list[ControllerVariant]:
    d = _as_dict(raw_variant, "variants[*]")
    name = str(d.get("name", "") or "").strip()
    pointer = dict(_as_dict(d.get("controller"), "variants[*].controller")) if d.get("controller") is not None else {}
    if not name:
        raise ValueError("variants[*].name must be non-empty.")
    description = str(d.get("description", "") or "")
    pointers = tuple(
        _parse_variant_pointer(item, context=f"variants[*].pointers[{idx}]")
        for idx, item in enumerate(list(d.get("pointers", []) or []))
    )
    overrides = _parse_variant_overrides(
        d.get("overrides", d.get("scenario_overrides")),
        context="variants[*].overrides",
    )

    raw_sweep = d.get("sweep")
    if raw_sweep is None:
        return [
            _variant_with(
                name=name,
                pointer=pointer,
                description=description,
                pointers=pointers,
                overrides=overrides,
            )
        ]
    sweep = _as_dict(raw_sweep, "variants[*].sweep")
    if not sweep:
        return [
            _variant_with(
                name=name,
                pointer=pointer,
                description=description,
                pointers=pointers,
                overrides=overrides,
            )
        ]

    sweep_paths = [str(path or "").strip() for path in sweep.keys()]
    if any(not path for path in sweep_paths):
        raise ValueError("variants[*].sweep keys must be non-empty dotted paths.")
    labels = _sweep_labels(sweep_paths)
    value_lists: list[list[Any]] = []
    for path in sweep_paths:
        values = sweep.get(path)
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"variants[*].sweep['{path}'] must be a non-empty list.")
        value_lists.append(list(values))

    expanded: list[ControllerVariant] = []
    for combo in product(*value_lists):
        pointer_copy = deepcopy(pointer)
        pointers_copy = deepcopy(pointers)
        overrides_copy = list(deepcopy(overrides))
        sweep_parts: list[str] = []
        sweep_desc_parts: list[str] = []
        for path, value in zip(sweep_paths, combo):
            root_path = path
            if root_path.startswith("controller."):
                root_path = root_path[len("controller.") :]
                _deep_set(pointer_copy, root_path, value)
            elif root_path.startswith("overrides."):
                overrides_copy.append(
                    ControllerVariantOverride(path=root_path[len("overrides.") :], value=deepcopy(value))
                )
            else:
                _deep_set(pointer_copy, root_path, value)
            label = labels[path]
            pretty = _format_sweep_value(value)
            sweep_parts.append(f"{label}_{_slug_token(pretty)}")
            sweep_desc_parts.append(f"{label}={pretty}")
        variant_name = f"{name}__{'__'.join(sweep_parts)}"
        variant_desc = description.strip()
        sweep_desc = ", ".join(sweep_desc_parts)
        if variant_desc:
            variant_desc = f"{variant_desc} [{sweep_desc}]"
        else:
            variant_desc = f"Sweep: {sweep_desc}"
        expanded.append(
            _variant_with(
                name=variant_name,
                pointer=pointer_copy,
                description=variant_desc,
                pointers=tuple(pointers_copy),
                overrides=tuple(overrides_copy),
            )
        )
    return expanded


def _parse_metric(raw: Any) -> ControllerBenchMetric:
    d = _as_dict(raw, "metric")
    desired = d.get("desired_quat_bn")
    desired_q = None
    if desired is not None:
        vals = tuple(float(x) for x in list(desired))
        if len(vals) != 4:
            raise ValueError("metric.desired_quat_bn must be length-4.")
        desired_q = vals
    target_relative = d.get("target_relative_state_ric_m")
    target_relative_q = None
    if target_relative is not None:
        target_relative_q = tuple(float(x) for x in list(target_relative))
        if len(target_relative_q) != 6:
            raise ValueError("metric.target_relative_state_ric_m must be length-6.")
    waypoints: list[tuple[float, float, float, float, float, float]] = []
    for waypoint in list(d.get("waypoints_ric_m", []) or []):
        values = tuple(float(x) for x in list(waypoint))
        if len(values) != 6:
            raise ValueError("metric.waypoints_ric_m entries must be length-6.")
        waypoints.append(values)
    return ControllerBenchMetric(
        name=str(d.get("name", "") or "").strip(),
        source_path=str(d.get("source_path", "") or "").strip(),
        kind=str(d.get("kind", "") or "").strip(),
        object_id=str(d.get("object_id", "") or "").strip(),
        reference_object_id=str(d.get("reference_object_id", "") or "").strip(),
        desired_quat_bn=desired_q,
        keepout_radius_km=float(d.get("keepout_radius_km")) if d.get("keepout_radius_km") is not None else None,
        settle_error_deg=float(d.get("settle_error_deg")) if d.get("settle_error_deg") is not None else None,
        settle_rate_norm_rad_s=(
            float(d.get("settle_rate_norm_rad_s")) if d.get("settle_rate_norm_rad_s") is not None else None
        ),
        settle_hold_s=float(d.get("settle_hold_s")) if d.get("settle_hold_s") is not None else None,
        target_relative_state_ric_m=target_relative_q,
        target_value=float(d.get("target_value")) if d.get("target_value") is not None else None,
        expected_value=str(d.get("expected_value", "") or "").strip(),
        diagnostic_field=str(d.get("diagnostic_field", "") or "").strip(),
        waypoints_ric_m=tuple(waypoints),
        waypoint_tolerance_m=(
            float(d.get("waypoint_tolerance_m")) if d.get("waypoint_tolerance_m") is not None else None
        ),
        waypoint_velocity_tolerance_m_s=(
            float(d.get("waypoint_velocity_tolerance_m_s"))
            if d.get("waypoint_velocity_tolerance_m_s") is not None
            else None
        ),
    )


def _parse_pass_criterion(raw: Any) -> ControllerBenchPassCriterion:
    d = _as_dict(raw, "pass_criterion")
    metric = str(d.get("metric", "") or "").strip()
    op = str(d.get("op", "") or "").strip()
    if not metric or not op:
        raise ValueError("pass criteria require non-empty 'metric' and 'op'.")
    return ControllerBenchPassCriterion(metric=metric, op=op, value=d.get("value"))


def _numeric_bound_shape(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_numeric_bound_shape(item) for item in value)
    return float(value)


def _parse_optimization(raw: Any) -> ControllerBenchOptimizationConfig:
    if raw is None:
        return ControllerBenchOptimizationConfig()
    d = _as_dict(raw, "optimization")
    enabled = bool(d.get("enabled", False))
    params_raw = list(d.get("parameters", []) or [])
    parameters: list[ControllerBenchOptimizationParameter] = []
    for idx, item in enumerate(params_raw):
        p = _as_dict(item, f"optimization.parameters[{idx}]")
        name = str(p.get("name", "") or "").strip()
        path = str(p.get("path", "") or "").strip()
        if not name:
            name = _slug_token(path)
        if not path:
            raise ValueError("optimization.parameters[*].path must be non-empty.")
        lower = _numeric_bound_shape(p.get("lower"))
        upper = _numeric_bound_shape(p.get("upper"))
        lower_flat = np.array(_flatten_numeric_tree(lower), dtype=float)
        upper_flat = np.array(_flatten_numeric_tree(upper), dtype=float)
        if lower_flat.size == 0 or lower_flat.size != upper_flat.size:
            raise ValueError(f"optimization parameter '{name}' lower/upper bounds must have matching shape.")
        if np.any(upper_flat <= lower_flat):
            raise ValueError(f"optimization parameter '{name}' upper bound must be greater than lower bound.")
        parameters.append(ControllerBenchOptimizationParameter(name=name, path=path, lower=lower, upper=upper))

    cost_raw = d.get("cost")
    cost = None
    if cost_raw is not None:
        c = _as_dict(cost_raw, "optimization.cost")
        object_id = str(c.get("object_id", "") or "").strip()
        reference_object_id = str(c.get("reference_object_id", "") or "").strip()
        if not object_id or not reference_object_id:
            raise ValueError("optimization.cost requires object_id and reference_object_id.")
        cost = ControllerBenchOptimizationCost(
            object_id=object_id,
            reference_object_id=reference_object_id,
            terminal_state_weight=float(c.get("terminal_state_weight", 1.0)),
            terminal_position_scale_km=float(c.get("terminal_position_scale_km", 0.01)),
            terminal_speed_scale_km_s=float(c.get("terminal_speed_scale_km_s", 0.001)),
            fuel_weight=float(c.get("fuel_weight", 1.0)),
            fuel_scale_kg=float(c.get("fuel_scale_kg", 1.0)),
            time_weight=float(c.get("time_weight", 1.0)),
            rendezvous_radius_km=float(c.get("rendezvous_radius_km", 0.01)),
            time_scale_s=float(c.get("time_scale_s", 600.0)),
            miss_time_penalty_s=float(c.get("miss_time_penalty_s", 3600.0)),
        )
    if enabled:
        if str(d.get("algorithm", "pso") or "pso").strip().lower() != "pso":
            raise ValueError("controller bench optimization currently supports only algorithm: pso.")
        if not parameters:
            raise ValueError("optimization.enabled requires at least one parameter.")
        particles = int(d.get("particles", 8))
        iterations = int(d.get("iterations", 10))
        if particles < 1 or iterations < 1:
            raise ValueError("optimization particles and iterations must be positive integers.")
        coefficients = {
            "inertia": float(d.get("inertia", 0.72)),
            "cognitive": float(d.get("cognitive", 1.49)),
            "social": float(d.get("social", 1.49)),
        }
        if any(not np.isfinite(value) or value < 0.0 for value in coefficients.values()):
            raise ValueError("optimization PSO coefficients must be finite and nonnegative.")
        if cost is not None:
            positive_scales = (
                cost.terminal_position_scale_km,
                cost.terminal_speed_scale_km_s,
                cost.fuel_scale_kg,
                cost.time_scale_s,
                cost.rendezvous_radius_km,
            )
            weights = (cost.terminal_state_weight, cost.fuel_weight, cost.time_weight)
            if any(not np.isfinite(value) or value <= 0.0 for value in positive_scales):
                raise ValueError("optimization cost scales/radius must be finite and positive.")
            if any(not np.isfinite(value) or value < 0.0 for value in weights):
                raise ValueError("optimization cost weights must be finite and nonnegative.")
    return ControllerBenchOptimizationConfig(
        enabled=enabled,
        algorithm=str(d.get("algorithm", "pso") or "pso").strip().lower(),
        base_variant=str(d.get("base_variant", "") or "").strip(),
        particles=int(d.get("particles", 8)),
        iterations=int(d.get("iterations", 10)),
        inertia=float(d.get("inertia", 0.72)),
        cognitive=float(d.get("cognitive", 1.49)),
        social=float(d.get("social", 1.49)),
        seed=int(d.get("seed", 0)),
        parameters=tuple(parameters),
        cost=cost,
    )


def _parse_envelope(raw: Any) -> ControllerBenchEnvelopeConfig:
    if raw is None:
        return ControllerBenchEnvelopeConfig()
    d = _as_dict(raw, "envelope")
    variations: list[ControllerBenchEnvelopeVariation] = []
    for idx, item in enumerate(list(d.get("variations", []) or [])):
        v = _as_dict(item, f"envelope.variations[{idx}]")
        path = str(v.get("path", "") or "").strip()
        if not path:
            raise ValueError("envelope.variations[*].path must be non-empty.")
        name = str(v.get("name", "") or "").strip() or _slug_token(path)
        values = v.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"envelope variation '{name}' must define a non-empty values list.")
        variations.append(ControllerBenchEnvelopeVariation(name=name, path=path, values=tuple(values)))
    base_cases = tuple(str(name).strip() for name in list(d.get("base_cases", []) or []) if str(name).strip())
    enabled = bool(d.get("enabled", False))
    if enabled and not variations:
        raise ValueError("envelope.enabled requires at least one variation.")
    return ControllerBenchEnvelopeConfig(
        enabled=enabled,
        seed=int(d.get("seed", 0)),
        max_cases=int(d.get("max_cases", 0) or 0),
        base_cases=base_cases,
        variations=tuple(variations),
    )


def _parse_objective(raw: Any) -> ControllerBenchObjective:
    d = _as_dict(raw, "objective")
    desired = d.get("desired_quat_bn")
    desired_q = None
    if desired is not None:
        vals = tuple(float(x) for x in list(desired))
        if len(vals) != 4:
            raise ValueError("objective.desired_quat_bn must be length-4.")
        desired_q = vals
    kind = str(d.get("kind", "") or "").strip().lower()
    if not kind:
        raise ValueError("objective.kind must be non-empty.")
    supported_kinds = {
        "attitude_hold",
        "attitude_slew_settle",
        "relative_rendezvous",
        "terminal_rendezvous_box",
        "rocket_orbit_insertion",
        "flight_software_stack_conformance",
        "nonzero_ric_hold",
        "waypoint_sequence",
        "safe_rpo_approach",
        "passive_retreat",
        "scheduled_burn_delivery",
        "orbital_element_tracking",
        "navigation_estimation",
        "multirate_consistency",
        "actuator_realization",
        "executive_preemption_recovery",
    }
    if kind not in supported_kinds:
        raise ValueError(
            f"Unsupported controller bench objective kind {kind!r}; choose one of: "
            + ", ".join(sorted(supported_kinds))
            + "."
        )

    def opt_float(name: str) -> float | None:
        return float(d.get(name)) if d.get(name) is not None else None

    target_relative = d.get("target_relative_state_ric_m")
    target_relative_q = None
    if target_relative is not None:
        target_relative_q = tuple(float(x) for x in list(target_relative))
        if len(target_relative_q) != 6:
            raise ValueError("objective.target_relative_state_ric_m must be length-6.")
    waypoints: list[tuple[float, float, float, float, float, float]] = []
    for waypoint in list(d.get("waypoints_ric_m", []) or []):
        values = tuple(float(x) for x in list(waypoint))
        if len(values) != 6:
            raise ValueError("objective.waypoints_ric_m entries must be length-6.")
        waypoints.append(values)

    return ControllerBenchObjective(
        kind=kind,
        name=str(d.get("name", "") or "").strip(),
        object_id=str(d.get("object_id", "") or "").strip(),
        reference_object_id=str(d.get("reference_object_id", "") or "").strip(),
        desired_quat_bn=desired_q,
        keepout_radius_km=opt_float("keepout_radius_km"),
        settle_error_deg=opt_float("settle_error_deg"),
        settle_rate_norm_rad_s=opt_float("settle_rate_norm_rad_s"),
        settle_hold_s=opt_float("settle_hold_s"),
        max_final_attitude_error_deg=opt_float("max_final_attitude_error_deg"),
        max_rms_attitude_error_deg=opt_float("max_rms_attitude_error_deg"),
        max_peak_attitude_error_deg=opt_float("max_peak_attitude_error_deg"),
        max_final_body_rate_norm_rad_s=opt_float("max_final_body_rate_norm_rad_s"),
        max_body_rate_norm_rad_s=opt_float("max_body_rate_norm_rad_s"),
        max_settle_time_s=opt_float("max_settle_time_s"),
        max_total_torque_norm_n_m_s=opt_float("max_total_torque_norm_n_m_s"),
        max_final_relative_distance_km=opt_float("max_final_relative_distance_km"),
        max_final_range_km=opt_float("max_final_range_km"),
        max_rms_relative_distance_km=opt_float("max_rms_relative_distance_km"),
        max_final_relative_speed_km_s=opt_float("max_final_relative_speed_km_s"),
        max_final_speed_km_s=opt_float("max_final_speed_km_s"),
        max_radial_error_km=opt_float("max_radial_error_km"),
        max_intrack_error_km=opt_float("max_intrack_error_km"),
        max_crosstrack_error_km=opt_float("max_crosstrack_error_km"),
        max_radial_rate_km_s=opt_float("max_radial_rate_km_s"),
        max_intrack_rate_km_s=opt_float("max_intrack_rate_km_s"),
        max_crosstrack_rate_km_s=opt_float("max_crosstrack_rate_km_s"),
        max_time_inside_keepout_s=opt_float("max_time_inside_keepout_s"),
        max_total_dv_m_s=opt_float("max_total_dv_m_s"),
        max_fuel_used_kg=opt_float("max_fuel_used_kg"),
        target_relative_state_ric_m=target_relative_q,
        max_final_position_error_m=opt_float("max_final_position_error_m"),
        max_final_velocity_error_m_s=opt_float("max_final_velocity_error_m_s"),
        waypoints_ric_m=tuple(waypoints),
        waypoint_tolerance_m=opt_float("waypoint_tolerance_m"),
        waypoint_velocity_tolerance_m_s=opt_float("waypoint_velocity_tolerance_m_s"),
        min_final_range_km=opt_float("min_final_range_km"),
        min_range_km=opt_float("min_range_km"),
        max_range_km=opt_float("max_range_km"),
        target_delta_v_m_s=opt_float("target_delta_v_m_s"),
        max_delta_v_error_m_s=opt_float("max_delta_v_error_m_s"),
        target_semimajor_axis_km=opt_float("target_semimajor_axis_km"),
        max_semimajor_axis_error_km=opt_float("max_semimajor_axis_error_km"),
        target_eccentricity=opt_float("target_eccentricity"),
        max_eccentricity_error=opt_float("max_eccentricity_error"),
        max_navigation_position_error_m=opt_float("max_navigation_position_error_m"),
        max_navigation_velocity_error_m_s=opt_float("max_navigation_velocity_error_m_s"),
        max_actuator_saturation_fraction=opt_float("max_actuator_saturation_fraction"),
        max_realization_force_error_n=opt_float("max_realization_force_error_n"),
        max_realization_torque_error_n_m=opt_float("max_realization_torque_error_n_m"),
        min_accepted_command_fraction=opt_float("min_accepted_command_fraction"),
        min_physical_realization_fraction=opt_float("min_physical_realization_fraction"),
        min_audit_completeness_fraction=opt_float("min_audit_completeness_fraction"),
        max_command_count=(int(d.get("max_command_count")) if d.get("max_command_count") is not None else None),
        max_missed_task_releases=(
            int(d.get("max_missed_task_releases")) if d.get("max_missed_task_releases") is not None else None
        ),
        max_missed_sensor_releases=(
            int(d.get("max_missed_sensor_releases")) if d.get("max_missed_sensor_releases") is not None else None
        ),
        expected_translation_mode=str(d.get("expected_translation_mode", "") or "").strip(),
        expected_control_law=str(d.get("expected_control_law", "") or "").strip(),
        min_invocation_count=(
            int(d.get("min_invocation_count")) if d.get("min_invocation_count") is not None else None
        ),
        max_cadence_jitter_s=opt_float("max_cadence_jitter_s"),
        target_altitude_km=opt_float("target_altitude_km"),
        max_final_apoapsis_error_km=opt_float("max_final_apoapsis_error_km"),
        max_final_periapsis_error_km=opt_float("max_final_periapsis_error_km"),
        max_final_eccentricity=opt_float("max_final_eccentricity"),
        max_insertion_time_s=opt_float("max_insertion_time_s"),
        max_dynamic_pressure_pa=opt_float("max_dynamic_pressure_pa"),
        max_mach=opt_float("max_mach"),
        max_abs_alpha_deg=opt_float("max_abs_alpha_deg"),
        max_tvc_gimbal_deg=opt_float("max_tvc_gimbal_deg"),
        max_aero_force_n=opt_float("max_aero_force_n"),
        min_propellant_remaining_fraction=opt_float("min_propellant_remaining_fraction"),
        require_insertion=bool(d.get("require_insertion", False)),
        require_not_terminated_early=bool(d.get("require_not_terminated_early", False)),
        require_preemption=bool(d.get("require_preemption", False)),
        require_recovery=bool(d.get("require_recovery", False)),
        require_primary_resume=bool(d.get("require_primary_resume", False)),
    )


def _parse_engineering(raw: Any) -> ControllerBenchEngineeringConfig:
    if raw is None:
        return ControllerBenchEngineeringConfig()
    d = _as_dict(raw, "engineering")
    max_rows = int(d.get("timeline_max_rows", 20000))
    if max_rows < 1:
        raise ValueError("engineering.timeline_max_rows must be positive.")
    return ControllerBenchEngineeringConfig(
        enabled=bool(d.get("enabled", True)),
        semantic_lint=bool(d.get("semantic_lint", True)),
        stack_conformance=bool(d.get("stack_conformance", True)),
        frame_sign_probe=bool(d.get("frame_sign_probe", True)),
        cadence_analysis=bool(d.get("cadence_analysis", True)),
        aligned_timeline=bool(d.get("aligned_timeline", True)),
        authority_analysis=bool(d.get("authority_analysis", True)),
        stability_analysis=bool(d.get("stability_analysis", True)),
        failure_reduction=bool(d.get("failure_reduction", True)),
        evidence_manifest=bool(d.get("evidence_manifest", True)),
        timeline_max_rows=max_rows,
    )


def load_controller_bench_config(
    path: str | Path,
    *,
    generated_cases_root: str | Path | None = None,
) -> ControllerBenchConfig:
    cfg_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Controller bench YAML root must be a mapping/object.")

    target_raw = _as_dict(raw.get("controller_target"), "controller_target")
    target = ControllerBenchTarget(
        object_id=str(target_raw.get("object_id", "target") or "target"),
        slot=str(target_raw.get("slot", "flight_software") or "flight_software"),
    )

    variants: list[ControllerVariant] = []
    for entry in list(raw.get("variants", []) or []):
        variants.extend(_expand_variant_entry(_as_dict(entry, "variants[*]")))
    variant_names = [variant.name for variant in variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("Controller bench variants must have unique names after sweep expansion.")

    output_dir = Path(str(raw.get("output_dir", f"outputs/controller_bench/{raw.get('suite_name', 'suite')}")))
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    cases: list[ControllerBenchCase] = []
    for entry in list(raw.get("cases", []) or []):
        cases.append(_case_from_raw_entry(entry, cfg_path))

    metrics = tuple(_parse_metric(m) for m in list(raw.get("metrics", []) or []))
    pass_criteria = tuple(_parse_pass_criterion(p) for p in list(raw.get("pass_criteria", []) or []))
    objectives = tuple(_parse_objective(o) for o in list(raw.get("objectives", []) or []))
    envelope = _parse_envelope(raw.get("envelope"))
    envelope_output_dir = output_dir if generated_cases_root is None else Path(generated_cases_root).resolve()
    cases, envelope_manifest = _expand_envelope_cases(
        cases=cases,
        envelope=envelope,
        output_dir=envelope_output_dir,
    )
    plot_mode = str(raw.get("plot_mode", "save") or "save").strip().lower()
    if plot_mode not in {"interactive", "save", "both"}:
        raise ValueError("controller bench plot_mode must be one of: interactive, save, both.")

    parallel_backend = str(raw.get("parallel_backend", "process_pool") or "process_pool").strip().lower()
    if parallel_backend not in {"process_pool", "thread_pool"}:
        raise ValueError("controller bench parallel_backend must be 'process_pool' or 'thread_pool'.")

    return ControllerBenchConfig(
        suite_name=str(raw.get("suite_name", cfg_path.stem) or cfg_path.stem),
        description=str(raw.get("description", "") or ""),
        output_dir=output_dir,
        plot_mode=plot_mode,
        controller_target=target,
        variants=tuple(variants),
        cases=tuple(cases),
        metrics=metrics,
        pass_criteria=pass_criteria,
        objectives=objectives,
        save_run_payloads=bool(raw.get("save_run_payloads", True)),
        disable_plots=bool(raw.get("disable_plots", True)),
        disable_animations=bool(raw.get("disable_animations", True)),
        print_individual_run_summaries=bool(raw.get("print_individual_run_summaries", False)),
        parallel_enabled=bool(raw.get("parallel_enabled", False)),
        parallel_workers=int(raw.get("parallel_workers", 0)),
        parallel_backend=parallel_backend,
        optimization=_parse_optimization(raw.get("optimization")),
        envelope=envelope,
        envelope_manifest=envelope_manifest,
        engineering=_parse_engineering(raw.get("engineering")),
    )


def _set_envelope_path_value(root: dict[str, Any], path: str, value: Any) -> None:
    paths = object_synced_parameter_paths(root, path)
    parts = str(path).split(".")
    if len(parts) >= 3 and parts[0] == "objects" and parts[1] in {"rocket", "chaser", "target"}:
        alias = ".".join([parts[1], *parts[2:]])
        if isinstance(root.get(parts[1]), dict) and alias not in paths:
            paths.append(alias)
    for synced_path in paths:
        set_parameter_path_value(root, synced_path, value)


def _case_from_raw_entry(entry: Any, cfg_path: Path) -> ControllerBenchCase:
    d = _as_dict(entry, "cases[*]")
    name = str(d.get("name", "") or "").strip()
    cfg_ref = str(d.get("config_path", "") or "").strip()
    if not name or not cfg_ref:
        raise ValueError("cases[*] require non-empty 'name' and 'config_path'.")
    case_metrics = tuple(_parse_metric(m) for m in list(d.get("metrics", []) or []))
    pass_criteria = tuple(_parse_pass_criterion(p) for p in list(d.get("pass_criteria", []) or []))
    objectives = tuple(_parse_objective(o) for o in list(d.get("objectives", []) or []))
    return ControllerBenchCase(
        name=name,
        config_path=(cfg_path.parent / cfg_ref).resolve(),
        description=str(d.get("description", "") or ""),
        metrics=case_metrics,
        pass_criteria=pass_criteria,
        objectives=objectives,
    )


def _expand_envelope_cases(
    *,
    cases: list[ControllerBenchCase],
    envelope: ControllerBenchEnvelopeConfig,
    output_dir: Path,
) -> tuple[list[ControllerBenchCase], dict[str, Any]]:
    manifest: dict[str, Any] = {
        "enabled": bool(envelope.enabled),
        "seed": int(envelope.seed),
        "max_cases": int(envelope.max_cases),
        "base_cases": list(envelope.base_cases),
        "variations": [
            {"name": variation.name, "path": variation.path, "values": list(variation.values)}
            for variation in envelope.variations
        ],
        "generated_case_count": 0,
        "generated_cases": [],
    }
    if not envelope.enabled:
        return cases, manifest

    by_name = {case.name: case for case in cases}
    if envelope.base_cases:
        missing = [name for name in envelope.base_cases if name not in by_name]
        if missing:
            raise ValueError(f"envelope.base_cases references unknown case(s): {', '.join(sorted(missing))}.")
        base_cases = [by_name[name] for name in envelope.base_cases]
    else:
        base_cases = list(cases)
    if not base_cases:
        raise ValueError("envelope.enabled requires at least one base case.")

    generated_dir = output_dir / "generated_cases" / "envelope"
    generated_dir.mkdir(parents=True, exist_ok=True)
    variation_names = [variation.name for variation in envelope.variations]
    variation_value_sets = [list(variation.values) for variation in envelope.variations]
    max_cases = int(envelope.max_cases)
    generated: list[ControllerBenchCase] = []
    generated_total = 0
    for base_case in base_cases:
        base_raw = yaml.safe_load(base_case.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(base_raw, dict):
            raise ValueError(f"Envelope base case '{base_case.name}' YAML root must be a mapping/object.")
        for combo_index, combo in enumerate(product(*variation_value_sets)):
            if max_cases > 0 and generated_total >= max_cases:
                break
            root = deepcopy(base_raw)
            case_parts = [base_case.name, "env"]
            sampled: dict[str, Any] = {}
            for variation, value in zip(envelope.variations, combo):
                _set_envelope_path_value(root, variation.path, value)
                sampled[variation.path] = deepcopy(value)
                case_parts.append(f"{_slug_token(variation.name)}_{_slug_token(_format_sweep_value(value))}")
            case_name = "__".join(case_parts)
            case_identity = hashlib.sha256(
                json.dumps(sampled, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()[:12]
            case_name = f"{case_name}__{case_identity}"
            root.setdefault("metadata", {})
            if isinstance(root["metadata"], dict):
                root["metadata"]["controller_bench_envelope"] = {
                    "base_case": base_case.name,
                    "case_name": case_name,
                    "combo_index": int(combo_index),
                    "sampled_parameters": sampled,
                }
            generated_path = generated_dir / f"{case_name}.yaml"
            generated_path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
            generated_case = ControllerBenchCase(
                name=case_name,
                config_path=generated_path.resolve(),
                description=f"{base_case.description} Envelope: "
                + ", ".join(
                    f"{variation.name}={_format_sweep_value(value)}"
                    for variation, value in zip(envelope.variations, combo)
                ),
                metrics=base_case.metrics,
                pass_criteria=base_case.pass_criteria,
                objectives=base_case.objectives,
            )
            generated.append(generated_case)
            manifest["generated_cases"].append(
                {
                    "name": generated_case.name,
                    "base_case": base_case.name,
                    "config_path": str(generated_case.config_path),
                    "sampled_parameters": sampled,
                    "variation_names": list(variation_names),
                }
            )
            generated_total += 1
        if max_cases > 0 and generated_total >= max_cases:
            break
    manifest["generated_case_count"] = len(generated)
    return [*cases, *generated], manifest


def _flatten_numeric_tree(value: Any) -> list[float]:
    if isinstance(value, tuple):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric_tree(item))
        return out
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric_tree(item))
        return out
    return [float(value)]


__all__ = [name for name in globals() if not name.startswith("__")]
