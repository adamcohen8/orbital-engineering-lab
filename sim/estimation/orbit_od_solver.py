
from sim.numeric_backend import normalize_numeric_backend
from sim.security.output_writes import mkdir_output
# ruff: noqa: F401,F403,F405,I001
from collections import OrderedDict

from .orbit_od_common import *
from .orbit_od_artifacts import *
from .orbit_od_parameters import *
from .orbit_od_quality import *
from .orbit_od_maneuvers import *

from sim.interchange.adapters.batch_od import emit_batch_od_state_product

def solve_dynamics_orbit_determination(
    packet: ObservationPacket | Mapping[str, Any],
    *,
    output_dir: str | Path,
    object_id: str | None = None,
    role: str | None = None,
    scenario_name: str = "dynamics_orbit_determination",
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    dt_s: float | None = None,
    estimate: str = "state",
    dynamics_model: str = "two_body",
    j2: bool = False,
    drag: bool = False,
    srp: bool = False,
    atmosphere_model: str | None = None,
    orbit_force_model: Mapping[str, Any] | None = None,
    base_specs: Mapping[str, Any] | None = None,
    attitude_source: str = "none",
    attitude_mode: str = "sun_track",
    attitude_body_axis: Sequence[float] | None = None,
    attitude_controller: str = "surrogate_snap",
    attitude_history: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    max_nfev: int = 24,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: Sequence[float] | None = None,
    prior_covariance_native: Sequence[Sequence[float]] | None = None,
    detect_maneuvers: bool = False,
    maneuver_frame: str = "ric",
    max_maneuver_dv_m_s: float = 25.0,
    maneuver_min_delta_v_m_s: float = 0.05,
    maneuver_min_improvement_ratio: float = 0.35,
    maneuver_burn_duration_s: float | None = None,
    maneuver_guard_observations: int = 3,
    maneuver_max_candidates: int = 12,
    maneuver_max_nfev: int | None = None,
    numeric_backend: str = "rust",
    covariance_scale_policy: str = "conservative",
    forecast_uncertainty: bool = False,
    forecast_process_noise: Mapping[str, Any] | None = None,
    fit_observation_covariance: Sequence[Sequence[float]] | None = None,
    forecast_consider: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if forecast_consider and (not forecast_uncertainty or covariance_scale_policy != "absolute"):
        raise ValueError("forecast_consider requires forecast_uncertainty and absolute measurement covariance")
    numeric_backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
    packet_obj = _coerce_observation_packet(packet)
    observations = packet_obj.observations
    if len(observations) < 2:
        raise ValueError("dynamics OD requires at least two observations.")
    oid = str(object_id or packet_obj.to_dict().get("object_id") or "target")
    out_role = str(role or oid)
    output_root = Path(output_dir).expanduser()
    mkdir_output(output_root)
    orbit_force_model_cfg = _normalize_orbit_force_model(
        orbit_force_model,
        dynamics_model=dynamics_model,
        j2=j2,
        drag=drag,
        srp=srp,
        atmosphere_model=atmosphere_model,
    )
    # A Rust OD request also selects the Rust orbit integrator unless the
    # caller explicitly pinned the force model to another backend.
    force_backend_explicit = "numeric_backend" in orbit_force_model_cfg
    if numeric_backend == "rust" and not force_backend_explicit:
        orbit_force_model_cfg["numeric_backend"] = "rust"

    times = np.array([float(row["time_s"]) for row in observations], dtype=float)
    times = times - float(times[0])
    positions = np.array([row["position_eci_km"] for row in observations], dtype=float)
    position_sigmas = np.array([float(row.get("position_sigma_km") or 1.0) for row in observations], dtype=float)
    has_velocity = all("velocity_eci_km_s" in row for row in observations)
    velocities = np.array([row["velocity_eci_km_s"] for row in observations], dtype=float) if has_velocity else None
    velocity_sigmas = (
        np.array([float(row.get("velocity_sigma_km_s") or 1.0e-3) for row in observations], dtype=float)
        if has_velocity
        else None
    )

    partition = partition_time_arc(
        times,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        allow_repeated_epochs=True,
    )
    fit_duration = partition.fit_duration_s
    holdout_duration = partition.holdout_duration_s
    total_duration = max(fit_duration + holdout_duration, fit_duration)
    fit_mask = partition.fit_mask
    holdout_mask = partition.holdout_mask
    if dt_s is None:
        dt = _default_dt_from_times(times)
    else:
        dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError("dt_s must be positive.")

    preliminary_observations = [
        dict(observations[index]) for index in np.flatnonzero(fit_mask)
    ]
    preliminary_packet_data = packet_obj.to_dict()
    preliminary_packet_data["observations"] = preliminary_observations
    preliminary_packet = observation_packet_from_dict(preliminary_packet_data)
    preliminary_guess = fit_state_from_position_observations(
        preliminary_packet,
        object_id=oid,
        role=out_role,
        epoch="first",
    )
    preliminary_fit_observations = _fit_partition_observations(preliminary_observations)
    x0, initial_guess_method = _local_initial_state(preliminary_fit_observations)
    epoch_jd_utc = preliminary_guess.get("epoch_jd_utc")
    selected = selected_orbit_od_parameters(estimate)
    parameters = build_orbit_od_parameter_set(selected)
    attitude_source_key = _normalize_attitude_source(attitude_source)
    attitude_history_rows = _coerce_attitude_history(
        attitude_history if attitude_history is not None else packet_obj.observations
    )
    if attitude_source_key == "observed_history" and attitude_history_rows:
        if not np.isclose(
            float(attitude_history_rows[0]["time_s"]), 0.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "observed attitude history must begin at the OD observation time origin (time_s=0)."
            )
    if attitude_source_key == "observed_history" and not attitude_history_rows:
        raise ValueError("attitude_source='observed_history' requires attitude_quat_bn samples.")
    if (
        attitude_source_key in {"modeled_inline", "modeled_replay"}
        and str(attitude_mode or "").strip().lower() != "sun_track"
    ):
        raise ValueError("Only attitude_mode='sun_track' is currently supported for modeled attitude OD.")
    attitude_body_axis_vec = _attitude_body_axis(attitude_body_axis)
    if attitude_source_key == "modeled_replay":
        attitude_history_rows = _build_modeled_attitude_history(
            object_id=oid,
            role=out_role,
            state=x0,
            epoch_jd_utc=None if epoch_jd_utc is None else float(epoch_jd_utc),
            source_packet=packet_obj,
            base_specs=base_specs,
            scenario_name=scenario_name,
            output_root=output_root,
            duration_s=total_duration,
            dt_s=dt,
            dynamics_model=dynamics_model,
            j2=j2,
            drag=drag,
            srp=srp,
            atmosphere_model=atmosphere_model,
            orbit_force_model=orbit_force_model_cfg,
            attitude_mode=attitude_mode,
            attitude_body_axis=attitude_body_axis_vec,
            attitude_controller=attitude_controller,
        )
    eval_counter = {"count": 0}
    fit_indices = np.flatnonzero(fit_mask)
    holdout_indices = np.flatnonzero(holdout_mask)
    fit_observations = [observations[index] for index in fit_indices]
    holdout_observations = [observations[index] for index in holdout_indices]
    fit_covariance_factors = _prepare_state_observation_factors(
        observations=fit_observations,
        position_sigmas=position_sigmas[fit_mask],
        velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
    )
    holdout_covariance_factors = _prepare_state_observation_factors(
        observations=holdout_observations,
        position_sigmas=position_sigmas[holdout_mask],
        velocity_sigmas=velocity_sigmas[holdout_mask] if velocity_sigmas is not None else None,
    )

    # The fused two-body/J2 context and batched built-in force-plan context
    # are opt-in accelerators. Other force configurations retain the exact
    # session evaluator with the requested orbit backend. Python keeps fit
    # policy, priors, clipping, and artifact semantics authoritative.
    native_fit_context = None
    native_all_context = None
    native_evaluation = None
    native_step_s = None
    if numeric_backend == "rust":
        native_evaluation = {
            "requested": "rust",
            "selected": "python_reference_fallback",
            "fallback_reason": None,
        }
        support_reason = (
            "orbit_force_model.numeric_backend explicitly selects Python propagation"
            if force_backend_explicit and str(orbit_force_model_cfg.get("numeric_backend")).strip().lower() == "python"
            else _onp_native_context_support_reason(
                selected_parameters=selected,
                dynamics_model=dynamics_model,
                orbit_force_model=orbit_force_model_cfg,
                base_specs=base_specs,
                attitude_source=attitude_source_key,
                attitude_history_rows=attitude_history_rows,
            )
        )
        if support_reason is None:
            try:
                from sim.rust_onp_od_backend import RustONPObservationContext

                native_step_s = min(float(orbit_force_model_cfg.get("orbit_substep_s") or dt), dt)
                if not np.isfinite(native_step_s) or native_step_s <= 0.0:
                    raise ValueError("orbit_substep_s must be finite and positive")
                # Build the same validated candidate configuration that the
                # maintained evaluator uses.  Its runtime scheduler is the
                # source of truth for flight-software boundaries; a hand-built
                # cadence would silently turn a complete-stack candidate into
                # a trajectory-only candidate.
                schedule_packet = _candidate_packet(
                    object_id=oid,
                    role=out_role,
                    state=x0,
                    epoch_jd_utc=None if epoch_jd_utc is None else float(epoch_jd_utc),
                    source_packet=packet_obj,
                    base_specs=base_specs,
                    parameter_values={},
                    attitude_history_rows=attitude_history_rows,
                )
                schedule_artifact = _candidate_artifact(
                    schedule_packet,
                    object_id=oid,
                    scenario_name=f"{scenario_name}_rust_schedule",
                    output_dir=output_root / "_od_eval_scratch" / "rust_schedule",
                    duration_s=total_duration,
                    dt_s=dt,
                    dynamics_model=dynamics_model,
                    j2=j2,
                    drag=drag,
                    srp=srp,
                    atmosphere_model=atmosphere_model,
                    orbit_force_model=orbit_force_model_cfg,
                    attitude_source=attitude_source_key,
                    attitude_mode=attitude_mode,
                    attitude_body_axis=attitude_body_axis_vec,
                    attitude_controller=attitude_controller,
                    attitude_history_rows=attitude_history_rows,
                )
                dimensions = [6 if velocities is not None else 3] * len(fit_observations)
                active_mask = fit_mask | holdout_mask
                active_indices = np.flatnonzero(active_mask)
                active_observations = [observations[index] for index in active_indices]
                factors_by_index = dict(zip(fit_indices, fit_covariance_factors, strict=True))
                factors_by_index.update(zip(holdout_indices, holdout_covariance_factors, strict=True))
                active_factors = [factors_by_index[int(index)] for index in active_indices]
                active_widths, active_end_indices = _onp_capture_native_step_schedule(
                    schedule_artifact,
                    object_id=oid,
                    epochs_s=times[active_mask],
                    fallback_step_s=native_step_s,
                )
                # Fit epochs are a subset of the active arc. The validated
                # scheduler's one exact prefix supplies the same integration
                # widths to both immutable contexts without a second replay.
                fit_active_indices = np.searchsorted(active_indices, fit_indices)
                fit_end_indices = [active_end_indices[int(index)] for index in fit_active_indices]
                fit_widths = active_widths[:fit_end_indices[-1]]
                native_fit_context = RustONPObservationContext(
                    times[fit_mask],
                    positions[fit_mask],
                    velocities[fit_mask] if velocities is not None else None,
                    fit_covariance_factors,
                    dimensions,
                    fit_widths,
                    fit_end_indices,
                    mu_km3_s2=EARTH_MU_KM3_S2,
                    include_j2=bool(orbit_force_model_cfg.get("j2", j2)),
                )
                native_all_context = RustONPObservationContext(
                    times[active_mask],
                    positions[active_mask],
                    velocities[active_mask] if velocities is not None else None,
                    active_factors,
                    [6 if velocities is not None else 3] * len(active_observations),
                    active_widths,
                    active_end_indices,
                    mu_km3_s2=EARTH_MU_KM3_S2,
                    include_j2=bool(orbit_force_model_cfg.get("j2", j2)),
                )
                native_evaluation["selected"] = RustONPObservationContext.backend_name
                native_evaluation["schedule_source"] = "validated_candidate_simulation_session"
            except (RuntimeError, ValueError) as exc:
                native_fit_context = None
                native_all_context = None
                reason = f"{type(exc).__name__}: {exc}"
                if str(orbit_force_model_cfg.get("numeric_backend", "")).strip().lower() == "rust":
                    native_evaluation["selected"] = "rust_orbit_step_python_session"
                    native_evaluation["batch_exclusion_reason"] = reason
                else:
                    native_evaluation["fallback_reason"] = reason
        else:
            native_evaluation["fallback_reason"] = support_reason
            if str(orbit_force_model_cfg.get("numeric_backend", "")).strip().lower() == "rust":
                native_evaluation["selected"] = "rust_orbit_step_python_session"
                native_evaluation["batch_exclusion_reason"] = support_reason
                native_evaluation["fallback_reason"] = None
                if set(selected) <= {"state", "drag_scale", "cd_scale", "srp_scale"} and attitude_source_key in {"", "none", "off"} and not attitude_history_rows:
                    try:
                        from sim.rust_onp_od_backend import RustONPForcePlanObservationContext

                        native_step_s = min(float(orbit_force_model_cfg.get("orbit_substep_s") or dt), dt)
                        if not np.isfinite(native_step_s) or native_step_s <= 0.0:
                            raise ValueError("orbit_substep_s must be finite and positive")
                        schedule_packet = _candidate_packet(
                            object_id=oid, role=out_role, state=x0,
                            epoch_jd_utc=None if epoch_jd_utc is None else float(epoch_jd_utc),
                            source_packet=packet_obj, base_specs=base_specs,
                            parameter_values={}, attitude_history_rows=attitude_history_rows,
                        )
                        schedule_artifact = _candidate_artifact(
                            schedule_packet, object_id=oid,
                            scenario_name=f"{scenario_name}_rust_force_schedule",
                            output_dir=output_root / "_od_eval_scratch" / "rust_force_schedule",
                            duration_s=total_duration, dt_s=dt,
                            dynamics_model=dynamics_model, j2=j2, drag=drag, srp=srp,
                            atmosphere_model=atmosphere_model,
                            orbit_force_model=orbit_force_model_cfg,
                            attitude_source=attitude_source_key, attitude_mode=attitude_mode,
                            attitude_body_axis=attitude_body_axis_vec,
                            attitude_controller=attitude_controller,
                            attitude_history_rows=attitude_history_rows,
                        )
                        active_mask = fit_mask | holdout_mask
                        active_indices = np.flatnonzero(active_mask)
                        active_factors_by_index = dict(zip(fit_indices, fit_covariance_factors, strict=True))
                        active_factors_by_index.update(zip(holdout_indices, holdout_covariance_factors, strict=True))
                        active_factors = [active_factors_by_index[int(index)] for index in active_indices]
                        active_widths, active_ends = _onp_capture_native_step_schedule(
                            schedule_artifact, object_id=oid, epochs_s=times[active_mask],
                            fallback_step_s=native_step_s,
                            metadata=native_evaluation, allow_passive_calendar=True,
                        )
                        fit_active_indices = np.searchsorted(active_indices, fit_indices)
                        fit_ends = [active_ends[int(index)] for index in fit_active_indices]
                        native_fit_context = RustONPForcePlanObservationContext(
                            schedule_artifact, object_id=oid,
                            times_s=times[fit_mask], reference_positions_km=positions[fit_mask],
                            reference_velocities_km_s=velocities[fit_mask] if velocities is not None else None,
                            factor_blocks=fit_covariance_factors,
                            step_widths_s=active_widths[:fit_ends[-1]], sample_end_indices=fit_ends,
                            guard_physical_steps=True,
                        )
                        native_all_context = RustONPForcePlanObservationContext(
                            schedule_artifact, object_id=oid,
                            times_s=times[active_mask], reference_positions_km=positions[active_mask],
                            reference_velocities_km_s=velocities[active_mask] if velocities is not None else None,
                            factor_blocks=active_factors,
                            step_widths_s=active_widths, sample_end_indices=active_ends,
                            guard_physical_steps=True,
                        )
                        # Verify the full configured arc before fitting. A
                        # callback, scheduler effect, or wheel mismatch must
                        # leave the maintained session evaluator in charge.
                        parity_epochs = np.unique(times[active_mask])
                        if native_evaluation.get("calendar_only"):
                            parity_epochs = parity_epochs[:2]
                        reference_t, reference_x = evaluate_artifact_at_epochs(
                            schedule_artifact, object_id=oid, epochs_s=parity_epochs,
                        )
                        native_x, _ = native_all_context.evaluate(x0)
                        native_on_parity = _states_at_epochs(times[active_mask], native_x, parity_epochs)
                        if not np.allclose(native_on_parity, reference_x, rtol=0.0, atol=2.0e-9):
                            raise ValueError("native sampled force history differs from configured session evaluation")
                        native_evaluation["selected"] = RustONPForcePlanObservationContext.backend_name
                        native_evaluation["fallback_reason"] = None
                        native_evaluation.pop("batch_exclusion_reason", None)
                        native_evaluation["schedule_source"] = "validated_candidate_simulation_session"
                        native_evaluation["session_parity_epoch_count"] = len(parity_epochs)
                        native_evaluation["physical_step_guard"] = True
                    except (RuntimeError, ValueError, AttributeError) as exc:
                        native_fit_context = None
                        native_all_context = None
                        native_evaluation["batch_exclusion_reason"] = f"{type(exc).__name__}: {exc}"

    from sim.rust_onp_od_backend import RustONPForcePlanObservationContext

    propagation_cache: OrderedDict[
        tuple[bytes, float, tuple[str, ...]], tuple[np.ndarray, np.ndarray]
    ] = OrderedDict()

    def evaluate(
        native_values: np.ndarray,
        *,
        duration_s: float,
        scratch_name: str,
        parameters_override: ParameterSet | None = None,
        maneuver: Mapping[str, Any] | None = None,
        allow_native: bool = True,
        consider: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, MissionInputPacket, ScenarioArtifact]:
        eval_counter["count"] += 1
        active_parameters = parameters if parameters_override is None else parameters_override
        parameter_values = active_parameters.mapping(native_values)
        state = _state_from_parameters(x0, parameter_values)
        candidate_base_specs = dict(base_specs or {})
        for name, field in (("mass_scale", "mass_kg"), ("density_scale", "cd"),
                            ("srp_area_scale", "solar_area_m2")):
            if consider and name in consider:
                if field not in candidate_base_specs:
                    raise ValueError(f"consider {name} requires explicit base_specs.{field}")
                candidate_base_specs[field] *= float(consider[name])
        candidate_packet = _candidate_packet(
            object_id=oid,
            role=out_role,
            state=state,
            epoch_jd_utc=None if epoch_jd_utc is None else float(epoch_jd_utc),
            source_packet=packet_obj,
            base_specs=candidate_base_specs,
            parameter_values=parameter_values,
            attitude_history_rows=attitude_history_rows,
        )
        artifact = _candidate_artifact(
            candidate_packet,
            object_id=oid,
            scenario_name=f"{scenario_name}_{scratch_name}",
            output_dir=output_root / "_od_eval_scratch" / f"{scratch_name}_{eval_counter['count']:04d}",
            duration_s=duration_s,
            dt_s=dt,
            dynamics_model=dynamics_model,
            j2=j2,
            drag=drag,
            srp=srp,
            atmosphere_model=atmosphere_model,
            orbit_force_model=orbit_force_model_cfg,
            attitude_source=attitude_source_key,
            attitude_mode=attitude_mode,
            attitude_body_axis=attitude_body_axis_vec,
            attitude_controller=attitude_controller,
            attitude_history_rows=attitude_history_rows,
            maneuver=maneuver,
        )
        requested_epochs = np.unique(times[times <= float(duration_s) + 1.0e-9])
        cache_key = (np.asarray(native_values, dtype=float).tobytes(), float(duration_s),
                     tuple(active_parameters.names), tuple(sorted((consider or {}).items())))
        cached = None if maneuver is not None else propagation_cache.get(cache_key)
        if cached is None:
            native_context = None
            if maneuver is None and allow_native:
                native_context = (
                    native_fit_context
                    if float(duration_s) <= float(fit_duration) + 1.0e-9
                    else native_all_context
                )
            if native_context is not None:
                try:
                    sim_t = native_context.times_s.copy()
                    sim_x, _native_residual = (native_context.evaluate(state, specs=_candidate_specs(candidate_base_specs, parameter_values))
                        if isinstance(native_context, RustONPForcePlanObservationContext)
                        else native_context.evaluate(state))
                except (RuntimeError, ValueError):
                    # Preserve the maintained evaluator's termination and
                    # failure policy for candidates the fused kernel cannot
                    # represent (for example an impact before the final row).
                    native_context = None
            if native_context is None:
                sim_t, sim_x = evaluate_artifact_at_epochs(
                    artifact,
                    object_id=oid,
                    epochs_s=requested_epochs,
                )
            elif _native_states_require_python_fallback(sim_x, initial_state=state):
                # The native kernel has no event-localization contract.  If a
                # requested sample is at or below Earth, delegate the entire
                # candidate to SimulationSession so its exact termination and
                # error semantics remain authoritative.
                sim_t, sim_x = evaluate_artifact_at_epochs(
                    artifact,
                    object_id=oid,
                    epochs_s=requested_epochs,
                )
            if maneuver is None:
                propagation_cache[cache_key] = (sim_t, sim_x)
                propagation_cache.move_to_end(cache_key)
                if len(propagation_cache) > 8:
                    propagation_cache.popitem(last=False)
        else:
            propagation_cache.move_to_end(cache_key)
            sim_t, sim_x = cached
        return sim_t, sim_x, candidate_packet, artifact

    def independent_residual(native_values: np.ndarray) -> np.ndarray:
        if native_fit_context is not None:
            parameter_values = parameters.mapping(native_values)
            state = _state_from_parameters(x0, parameter_values)
            try:
                _simulated_states, native_residual = (
                    native_fit_context.evaluate(state, specs=_candidate_specs(base_specs, parameter_values))
                    if isinstance(native_fit_context, RustONPForcePlanObservationContext)
                    else native_fit_context.evaluate(state)
                )
            except (RuntimeError, ValueError):
                _simulated_states = None
                native_residual = None
            if (
                _simulated_states is not None
                and not _native_states_require_python_fallback(_simulated_states, initial_state=state)
            ):
                return native_residual
            # Re-enter through the maintained artifact evaluator.  This path
            # retains its existing exception, early-termination, and residual
            # policy for invalid candidate trajectories.
            sim_t, sim_x, _packet, _artifact = evaluate(
                native_values,
                duration_s=fit_duration,
                scratch_name="fit",
                allow_native=False,
            )
            sim_on_obs = _states_at_epochs(sim_t, sim_x, times[fit_mask])
            return _whiten_state_observation_residuals(
                sim_on_obs,
                observations=fit_observations,
                reference_positions=positions[fit_mask],
                reference_velocities=velocities[fit_mask] if velocities is not None else None,
                position_sigmas=position_sigmas[fit_mask],
                velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
                covariance_factors=fit_covariance_factors,
                numeric_backend=numeric_backend,
            )
        sim_t, sim_x, _packet, _artifact = evaluate(native_values, duration_s=fit_duration, scratch_name="fit")
        sim_on_obs = _states_at_epochs(sim_t, sim_x, times[fit_mask])
        return _whiten_state_observation_residuals(
            sim_on_obs,
            observations=fit_observations,
            reference_positions=positions[fit_mask],
            reference_velocities=velocities[fit_mask] if velocities is not None else None,
            position_sigmas=position_sigmas[fit_mask],
            velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
            covariance_factors=fit_covariance_factors,
            numeric_backend=numeric_backend,
        )

    # Optional complete fit-arc covariance includes temporal and cross-component
    # dependence. Its order is observation-major Cartesian position then velocity.
    if fit_observation_covariance is not None:
        from scipy.linalg import block_diag, solve_triangular
        from sim.estimation.weighting import prepare_covariance_whitener

        fit_dimension = len(fit_observations) * (6 if velocities is not None else 3)
        full_factor = prepare_covariance_whitener(np.asarray(fit_observation_covariance),
            dimension=fit_dimension, field_name="fit observation covariance")
        row_factors = block_diag(*fit_covariance_factors)
        def residual(values):
            return solve_triangular(full_factor, row_factors @ independent_residual(values), lower=True)
    else:
        residual = independent_residual

    prefit_t, prefit_x, _prefit_packet, _prefit_artifact = evaluate(
        parameters.initial_native(),
        duration_s=fit_duration,
        scratch_name="prefit_metrics",
    )
    prefit_on_obs = _states_at_epochs(prefit_t, prefit_x, times[fit_mask])
    prefit_metrics = _state_error_metrics(
        prefit_on_obs,
        positions[fit_mask],
        velocities[fit_mask] if velocities is not None else None,
    )
    # The zero-valued Cartesian correction makes SciPy's default relative
    # finite differences microscopic. A fixed perturbation stays above the
    # long-arc integrator's roundoff while remaining small relative to the
    # metre-scale observation noise. Keep this policy scoped to the parity-
    # checked native force-plan fits. Scale perturbations use the same fixed
    # native-unit policy, avoiding roundoff-sized force changes in joint fits.
    jacobian_step_native = (
        np.array([1.0 if name in {"dx_m", "dy_m", "dz_m"}
                  else 0.1 if name in {"dvx_mm_s", "dvy_mm_s", "dvz_mm_s"}
                  else 1.0e-4 for name in parameters.names], dtype=float)
        if native_evaluation is not None
        and native_evaluation.get("selected") == "rust_onp_force_plan_observation_context"
        and "state" in selected
        else None
    )
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=max_nfev,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=None if prior_mean_native is None else np.asarray(prior_mean_native, dtype=float),
        prior_covariance_native=None
        if prior_covariance_native is None
        else np.asarray(prior_covariance_native, dtype=float),
        jacobian_step_native=jacobian_step_native,
        covariance_scale_policy=covariance_scale_policy,
    )
    fit_t, fit_x, fitted_packet, fitted_artifact = evaluate(
        solve.x_native, duration_s=fit_duration, scratch_name="postfit"
    )
    fit_on_obs = _states_at_epochs(fit_t, fit_x, times[fit_mask])
    fit_metrics = _state_error_metrics(
        fit_on_obs, positions[fit_mask], velocities[fit_mask] if velocities is not None else None
    )

    pred_t, pred_x, prediction_packet, prediction_artifact = evaluate(
        solve.x_native,
        duration_s=total_duration,
        scratch_name="prediction",
    )
    holdout_t_obs = times[holdout_mask]
    holdout_ref_pos = positions[holdout_mask]
    holdout_ref_vel = velocities[holdout_mask] if velocities is not None else None
    pred_on_holdout = _states_at_epochs(pred_t, pred_x, holdout_t_obs)
    holdout_metrics = _state_error_metrics(pred_on_holdout, holdout_ref_pos, holdout_ref_vel)
    covariance_forecast = None
    if forecast_uncertainty:
        from sim.estimation.orbit_od_covariance import consider_parameter_covariance, propagate_joint_covariance, transport_process_covariance
        from sim.estimation.sequential_od import SequentialProcessNoise

        if solve.covariance_native is None:
            covariance_forecast = {"status": "unavailable_rank_deficient", "records": []}
        elif solve.covariance_evidence["status"] != "formal_local_gaussian":
            covariance_forecast = {"status": "requires_non_gaussian_calibration", "records": [],
                                   "reason": solve.covariance_evidence["status"]}
        else:
            all_times = times[fit_mask | holdout_mask]
            consider_spec = dict(forecast_consider or {})
            consider_names = list(consider_spec.get("names", []))
            allowed_consider = {"mass_scale", "density_scale", "srp_area_scale", "position_bias_x_km",
                                "position_bias_y_km", "position_bias_z_km"}
            if set(consider_names) - allowed_consider:
                raise ValueError("unsupported consider parameter; use mass_scale, density_scale, srp_area_scale or position_bias_[xyz]_km")
            consider_nominal = np.array([1. if name.endswith("scale") else 0. for name in consider_names])
            consider_steps = np.asarray(consider_spec.get("steps", [1e-3] * len(consider_names)), dtype=float)
            if consider_steps.shape != consider_nominal.shape or np.any(~np.isfinite(consider_steps)) or np.any(consider_steps <= 0):
                raise ValueError("consider steps must be finite positive values matching names")
            def forecast_evaluator(values, considers):
                mapping = dict(zip(consider_names, considers, strict=True))
                t, x, _, _ = evaluate(values, duration_s=total_duration, scratch_name="covariance_prediction",
                                     consider=mapping, allow_native=not any(n in mapping for n in ("mass_scale", "density_scale", "srp_area_scale")) or isinstance(native_all_context, RustONPForcePlanObservationContext))
                return _states_at_epochs(t, x, all_times)
            noise = SequentialProcessNoise(**dict(forecast_process_noise or {"spectral_density": (0, 0, 0)}))
            # Forecast process noise is additional uncertainty after the last
            # assimilated epoch, never re-counted over the fitted arc.
            q = None  # Process increments are transported after engine sensitivities are available.
            steps = jacobian_step_native if jacobian_step_native is not None else np.array([
                1.0 if name in {"dx_m", "dy_m", "dz_m"} else
                0.1 if name in {"dvx_mm_s", "dvy_mm_s", "dvz_mm_s"} else 1e-4 for name in parameters.names])
            fitted_covariance = solve.covariance_native.copy()
            consider_kwargs = {}
            if consider_names:
                from scipy.linalg import solve_triangular
                def consider_residual(z):
                    predicted = forecast_evaluator(solve.x_native, z)[:len(fit_observations)].copy()
                    for name, value in zip(consider_names, z, strict=True):
                        if name.startswith("position_bias_"):
                            predicted[:, "xyz".index(name[-4])] += value
                    raw_white = _whiten_state_observation_residuals(predicted,
                        observations=fit_observations, reference_positions=positions[fit_mask],
                        reference_velocities=velocities[fit_mask] if velocities is not None else None,
                        position_sigmas=position_sigmas[fit_mask], velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
                        covariance_factors=fit_covariance_factors, numeric_backend=numeric_backend)
                    if fit_observation_covariance is not None:
                        raw_white = solve_triangular(full_factor, row_factors @ raw_white, lower=True)
                    return raw_white
                b = []
                for i, step in enumerate(consider_steps):
                    plus, minus = consider_nominal.copy(), consider_nominal.copy()
                    plus[i] += step
                    minus[i] -= step
                    b.append((consider_residual(plus) - consider_residual(minus)) / (2 * step))
                b = np.asarray(b).T
                a = np.asarray(solve.jacobian) / parameters.scales()
                if len(a) > len(b):
                    b = np.vstack((b, np.zeros((len(a) - len(b), b.shape[1]))))
                cc = np.asarray(consider_spec["covariance"], dtype=float)
                extra, cross = consider_parameter_covariance(a, b, cc)
                fitted_covariance += extra
                consider_kwargs = dict(consider_values=consider_nominal, consider_names=consider_names,
                    consider_steps=consider_steps, consider_covariance=cc,
                    parameter_consider_cross_covariance=cross)
            covariance_forecast = propagate_joint_covariance(forecast_evaluator,
                parameter_values=solve.x_native, parameter_covariance=fitted_covariance,
                parameter_steps=steps, times_s=all_times, parameter_names=parameters.names,
                process_noise_covariances=q, **consider_kwargs)
            covariance_forecast["formal_covariance_policy"] = solve.covariance_evidence
            if any(value > 0 for value in noise.spectral_density):
                state_names = ["dx_m", "dy_m", "dz_m", "dvx_mm_s", "dvy_mm_s", "dvz_mm_s"]
                if not set(state_names) <= set(parameters.names):
                    raise ValueError("forecast process-noise transport requires state sensitivities; include state in estimate")
                state_indices = [parameters.names.index(name) for name in state_names]
                sensitivity = np.asarray([row["sensitivity_native"] for row in covariance_forecast["records"]])
                transitions = sensitivity[:, :, state_indices] * np.array([1000.] * 3 + [1e6] * 3)
                anchor = float(times[fit_mask][-1])
                transported = transport_process_covariance(all_times, transitions, start_time_s=anchor,
                    local_covariance=noise.covariance)
                for row, process_covariance in zip(covariance_forecast["records"], transported, strict=True):
                    row["covariance_eci_km_km_s"] = (np.asarray(row["covariance_eci_km_km_s"]) + process_covariance).tolist()
                covariance_forecast["process_noise_supplied"] = True
                covariance_forecast["process_noise_policy"] = {"model": noise.model,
                    "spectral_density": list(noise.spectral_density), "start_time_s": anchor,
                    "transport": "ONP_finite_difference_state_transition",
                    "injection": "discrete_covariance_at_each_forecast_epoch",
                    "continuous_noise_integral_exact": False}

    fit_whitened_residual = _whiten_state_observation_residuals(
        fit_on_obs,
        observations=fit_observations,
        reference_positions=positions[fit_mask],
        reference_velocities=velocities[fit_mask] if velocities is not None else None,
        position_sigmas=position_sigmas[fit_mask],
        velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
        covariance_factors=fit_covariance_factors,
        numeric_backend=numeric_backend,
    )
    holdout_whitened_residual = _whiten_state_observation_residuals(
        pred_on_holdout,
        observations=holdout_observations,
        reference_positions=holdout_ref_pos,
        reference_velocities=holdout_ref_vel,
        position_sigmas=position_sigmas[holdout_mask],
        velocity_sigmas=velocity_sigmas[holdout_mask] if velocity_sigmas is not None else None,
        covariance_factors=holdout_covariance_factors,
        numeric_backend=numeric_backend,
    )

    estimated_parameters = parameters.metadata(solve.x_native)
    solver_summary = {
        "success": bool(solve.success),
        "message": solve.message,
        "nfev": int(solve.nfev),
        "initial_cost": float(solve.initial_cost),
        "final_cost": float(solve.cost),
        "rms_weighted_residual": float(solve.rms_residual),
        "diagnostics": solve.diagnostics,
        "decision_records": _label_state_residual_decisions(
            solve.decision_records,
            observations=[observations[index] for index in np.flatnonzero(fit_mask)],
            include_velocity=velocities is not None,
        ),
    }
    state_component_count = 6 if velocities is not None else 3
    residual_audit = build_residual_audit(
        [
            *residual_records_from_vectors(
                fit_whitened_residual,
                partition="fit",
                measurement_type="eci_position_velocity" if velocities is not None else "eci_position",
                component_count=state_component_count,
                arc_id=scenario_name,
            ),
            *residual_records_from_vectors(
                holdout_whitened_residual,
                partition="holdout",
                measurement_type="eci_position_velocity" if velocities is not None else "eci_position",
                component_count=state_component_count,
                arc_id=scenario_name,
            ),
        ],
        decision_records=solver_summary["decision_records"],
    )
    quality_gates = build_dynamics_od_quality_gates(
        solver=solver_summary,
        parameter_metadata=estimated_parameters,
        covariance=solve.covariance_native,
        prefit_position_rms_m=float(prefit_metrics["position_rms_m"]),
        fit_metrics=fit_metrics,
        holdout_metrics=holdout_metrics,
        selected_parameters=selected,
        attitude_source=attitude_source_key,
        attitude_history_rows=attitude_history_rows,
        base_specs=base_specs,
    )

    fitted_packet_path = output_root / "fitted_mission_input_packet.json"
    estimated_parameters_path = output_root / "estimated_parameters.json"
    materialized_fit_config_path = output_root / "materialized_fit_config.yaml"
    materialized_prediction_config_path = output_root / "materialized_prediction_config.yaml"
    fit_residuals_csv = output_root / "fit_residuals.csv"
    holdout_errors_csv = output_root / "holdout_errors.csv"
    report_json_path = output_root / "od_fit_report.json"
    report_md_path = output_root / "od_fit_report.md"
    fit_plot_path = output_root / "fit_position_residuals.png"
    holdout_plot_path = output_root / "holdout_position_errors.png"

    write_json(str(fitted_packet_path), fitted_packet.to_dict())
    write_json(
        str(estimated_parameters_path),
        {
            "parameters": estimated_parameters,
            "covariance": None if solve.covariance_native is None else solve.covariance_native.tolist(),
            "data_only_covariance": None if solve.data_covariance_native is None else solve.data_covariance_native.tolist(),
            "covariance_evidence": solve.covariance_evidence,
        },
    )
    fitted_artifact.write(materialized_fit_config_path)
    prediction_artifact.write(materialized_prediction_config_path)
    _write_error_csv(
        fit_residuals_csv,
        times[fit_mask],
        fit_on_obs,
        positions[fit_mask],
        velocities[fit_mask] if velocities is not None else None,
        observations=[observations[idx] for idx in np.flatnonzero(fit_mask)],
        partition="fit",
    )
    _write_error_csv(
        holdout_errors_csv,
        holdout_t_obs,
        pred_on_holdout,
        holdout_ref_pos,
        holdout_ref_vel,
        observations=[observations[idx] for idx in np.flatnonzero(holdout_mask)],
        partition="holdout",
    )
    _write_error_plot(
        fit_plot_path,
        times[fit_mask],
        fit_on_obs[:, :3] - positions[fit_mask],
        title="Dynamics OD Fit Position Residuals",
    )
    _write_error_plot(
        holdout_plot_path,
        holdout_t_obs - fit_duration,
        pred_on_holdout[:, :3] - holdout_ref_pos,
        title="Dynamics OD Holdout Position Errors",
    )

    estimation_policy = {
        "robust_loss": str(robust_loss),
        "robust_f_scale": float(robust_f_scale),
        "sigma_clip_threshold": sigma_clip_threshold,
        "prior_enabled": prior_mean_native is not None,
        "covariance_scale_policy": covariance_scale_policy,
        "fit_temporal_covariance_supplied": fit_observation_covariance is not None,
    }
    if numeric_backend == "rust":
        estimation_policy["numeric_backend"] = numeric_backend
        if native_evaluation is not None:
            estimation_policy["native_evaluation"] = dict(native_evaluation)
        if jacobian_step_native is not None:
            estimation_policy["jacobian_step_native"] = jacobian_step_native.tolist()
    epoch_evaluation = exact_epoch_provenance(times[fit_mask | holdout_mask])
    if (
        native_evaluation is not None
        and native_evaluation.get("selected") in {
            "rust_onp_observation_context", "rust_onp_force_plan_observation_context",
        }
    ):
        epoch_evaluation.update(
            {
                "method": (
                    "rust_onp_observation_context_configured_substep_exact"
                    if native_evaluation["selected"] == "rust_onp_observation_context"
                    else "rust_onp_force_plan_sampled_history_exact"
                ),
                "evaluator": (
                    "oel_rust_orbit.ONPObservationContext"
                    if native_evaluation["selected"] == "rust_onp_observation_context"
                    else ("oel_rust_orbit.ONPForceContext.sampled_adaptive_history"
                          if str(orbit_force_model_cfg.get("integrator", "rk4")) in {"adaptive", "rkf78", "dopri5"}
                          else "oel_rust_orbit.ONPForceContext.sampled_history")
                ),
                "integration_substep_s": float(native_step_s),
            }
        )
    result = {
        "method": "dynamics_orbit_least_squares",
        "observation_packet_source": dict(packet_obj.to_dict().get("source", {}) or {}),
        "object_id": oid,
        "scenario_name": scenario_name,
        "dynamics_model": dynamics_model,
        "j2": bool(j2),
        "drag": bool(drag),
        "srp": bool(srp),
        "atmosphere_model": atmosphere_model or "",
        "orbit_force_model": orbit_force_model_cfg,
        "attitude_source": attitude_source_key,
        "attitude_mode": str(attitude_mode or ""),
        "attitude_body_axis": attitude_body_axis_vec.tolist(),
        "attitude_history_sample_count": len(attitude_history_rows),
        "fit_duration_s": fit_duration,
        "holdout_duration_s": holdout_duration,
        "dt_s": dt,
        "observation_partition": partition.summary,
        "epoch_evaluation": epoch_evaluation,
        "estimate_spec": estimate,
        "estimation_policy": estimation_policy,
        "selected_parameters": selected,
        "observation_count": len(observations),
        "fit_observation_count": int(np.count_nonzero(fit_mask)),
        "holdout_observation_count": int(np.count_nonzero(holdout_mask)),
        "initial_guess": {
            "method": initial_guess_method,
            "state_eci_km_s": x0.tolist(),
            "preliminary_batch_position_fit": {
                "state_eci_km_s": preliminary_guess.get("state_eci_km_s"),
                "diagnostics": dict(preliminary_guess.get("diagnostics", {}) or {}),
            },
        },
        "estimated_parameters": estimated_parameters,
        "forecast_uncertainty": covariance_forecast,
        "derived_parameters": _derived_estimated_parameters(estimated_parameters, base_specs=base_specs),
        "solver": solver_summary,
        "quality_gates": quality_gates,
        "verdict": _dynamics_od_verdict(quality_gates),
        "prefit_metrics": prefit_metrics,
        "fit_metrics": fit_metrics,
        "holdout_metrics": holdout_metrics,
        "residual_audit": residual_audit,
        "fitted_mission_input_packet_path": str(fitted_packet_path),
        "estimated_parameters_path": str(estimated_parameters_path),
        "fit_residuals_csv": str(fit_residuals_csv),
        "holdout_errors_csv": str(holdout_errors_csv),
        "fit_plot_path": str(fit_plot_path),
        "holdout_plot_path": str(holdout_plot_path),
        "materialized_fit_config_path": str(materialized_fit_config_path),
        "materialized_prediction_config_path": str(materialized_prediction_config_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
    }
    if detect_maneuvers:
        maneuver_report = _detect_single_maneuver(
            output_root=output_root,
            object_id=oid,
            selected_parameters=selected,
            base_parameters=parameters,
            baseline_native_values=solve.x_native,
            baseline_fit_metrics=fit_metrics,
            baseline_holdout_metrics=holdout_metrics,
            baseline_covariance=solve.covariance_native,
            evaluate=evaluate,
            times=times,
            fit_mask=fit_mask,
            holdout_mask=holdout_mask,
            observations=observations,
            positions=positions,
            velocities=velocities,
            position_sigmas=position_sigmas,
            velocity_sigmas=velocity_sigmas,
            fit_duration_s=fit_duration,
            total_duration_s=total_duration,
            dt_s=dt,
            frame=maneuver_frame,
            max_delta_v_m_s=max_maneuver_dv_m_s,
            min_delta_v_m_s=maneuver_min_delta_v_m_s,
            min_improvement_ratio=maneuver_min_improvement_ratio,
            burn_duration_s=maneuver_burn_duration_s,
            guard_observations=maneuver_guard_observations,
            max_candidates=maneuver_max_candidates,
            max_nfev=max_nfev if maneuver_max_nfev is None else int(maneuver_max_nfev),
            robust_loss=robust_loss,
            robust_f_scale=robust_f_scale,
            sigma_clip_threshold=sigma_clip_threshold,
        )
        result["maneuver_detection"] = maneuver_report
        for key, value in dict(maneuver_report.get("artifacts", {}) or {}).items():
            result[f"maneuver_{key}"] = value
    write_json(str(report_json_path), result)
    _write_report_md(report_md_path, result)
    emission_record_path = output_root / "state_estimate_product_emission.json"
    try:
        emission = emit_batch_od_state_product(output_root)
    except Exception as exc:
        emission = {
            "status": "not_emitted",
            "product_path": str(output_root / "state_estimate_product.json"),
            "reason": str(exc),
            "error_type": type(exc).__name__,
        }
    write_json(str(emission_record_path), emission)
    result["state_estimate_product_emission"] = emission
    result["state_estimate_product_emission_path"] = str(emission_record_path)
    return result


def _onp_native_step_schedule(
    times_s: Sequence[float] | np.ndarray, step_s: float
) -> tuple[np.ndarray, list[int]]:
    """Build the nested numerical substep schedule for trajectory-only ONP.

    Complete-stack candidates use :func:`_onp_capture_native_step_schedule`,
    which records the maintained flight-software scheduler directly.  This
    helper remains useful for the trajectory-only path and for validating the
    nested outer/inner stepping contract without constructing a session.
    """

    from sim.dynamics.model import OrbitalAttitudeDynamics

    times = np.asarray(times_s, dtype=float).reshape(-1)
    if not np.all(np.isfinite(times)):
        raise ValueError("ONP observation epochs must be finite")
    if np.any(np.diff(times) < -1.0e-12):
        raise ValueError("ONP observation epochs must be non-decreasing")
    if not np.isfinite(float(step_s)) or float(step_s) <= 0.0:
        raise ValueError("ONP integration step must be finite and positive")
    outer_step_s = max(float(step_s), 1.0e-9)
    widths: list[float] = []
    end_indices: list[int] = []
    previous_time = 0.0
    for target_time in times:
        target = float(target_time)
        gap = target - previous_time
        if gap < -1.0e-12:
            raise ValueError("ONP observation epochs must be non-decreasing")
        current = previous_time
        while current < target:
            # Mirror _step_trajectory_only's outer loop.  In particular, do
            # not call _substep_sequence over the whole observation gap: the
            # maintained engine applies the configured outer step repeatedly,
            # then lets dynamics split each outer interval independently.
            h = min(outer_step_s, target - current)
            if h <= 0.0:
                raise ValueError("ONP schedule produced a nonpositive interval")
            inner_step = OrbitalAttitudeDynamics._effective_substep(outer_step_s, h)
            widths.extend(OrbitalAttitudeDynamics._substep_sequence(h, inner_step))
            current += h
        end_indices.append(len(widths))
        previous_time = target
    return np.asarray(widths, dtype=np.float64), end_indices


def _onp_capture_native_step_schedule(
    artifact: ScenarioArtifact,
    *,
    object_id: str,
    epochs_s: Sequence[float] | np.ndarray,
    fallback_step_s: float,
    metadata: dict[str, Any] | None = None,
    allow_passive_calendar: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Capture exact target integration widths from a validated OEL session.

    The OD candidate builder defaults to the complete ``fsw.passive`` stack.
    That stack can introduce hard task, sensor, input, or actuator boundaries
    inside the configured orbit step.  Recording the target runtime's
    ``next_hard_boundary_ns`` calls keeps the native context aligned with the
    maintained scheduler without reproducing its policy in Rust.

    A trajectory-only artifact has no runtime boundary owner; in that narrow
    case the nested numerical helper above is the maintained equivalent.
    """

    epochs = np.asarray(epochs_s, dtype=float).reshape(-1)
    if not np.all(np.isfinite(epochs)):
        raise ValueError("ONP observation epochs must be finite")
    if np.any(epochs < -1.0e-12) or np.any(np.diff(epochs) < -1.0e-12):
        raise ValueError("ONP observation epochs must be non-negative and non-decreasing")
    if epochs.size == 0:
        return np.zeros(0, dtype=np.float64), []

    from sim.api import SimulationSession

    session = SimulationSession.from_config(artifact, history_mode="dynamic")
    initial = session.reset()
    if initial is None:
        raise ValueError("native ONP schedule capture requires a single-run artifact")
    engine = session._engine
    agent = engine.agents.get(str(object_id))
    if agent is None or not bool(getattr(agent, "active", True)):
        raise ValueError(f"native ONP schedule capture could not find active object {object_id!r}")

    runtime = getattr(agent, "flight_software_runtime", None)
    widths: list[float] = []
    from sim.dynamics.model import OrbitalAttitudeDynamics

    dynamics = getattr(agent, "dynamics", None)
    orbit_substep_s = max(
        float(getattr(dynamics, "orbit_substep_s", fallback_step_s) or fallback_step_s),
        1.0e-9,
    )

    def expand_dynamics_interval(width_s: float) -> list[float]:
        if isinstance(dynamics, OrbitalAttitudeDynamics):
            inner_step = OrbitalAttitudeDynamics._effective_substep(orbit_substep_s, width_s)
            return OrbitalAttitudeDynamics._substep_sequence(width_s, inner_step)
        return [float(width_s)]

    # Only the runtime owner can certify a state-independent passive calendar.
    # Preserve its exact one-second (or configured task/sensor) boundaries; do
    # not coarsen numerical integration to the observation cadence.
    if (allow_passive_calendar and runtime is not None and len(engine.agents) == 1 and agent.bridge is None
            and engine.collision_stepper is None and engine.system_force_stepper is None
            and type(dynamics) is OrbitalAttitudeDynamics
            and dynamics.resource_model is None and not engine.attitude_enabled):
        boundaries = runtime.passive_prediction_boundaries_ns(start_time_ns=0,
            end_time_ns=int(round(float(epochs[-1]) * 1e9)))
        if boundaries is not None:
            current_ns = 0
            cursor = 0
            end_indices = []
            for epoch in epochs:
                target_ns = int(round(float(epoch) * 1e9))
                while cursor < len(boundaries) and boundaries[cursor] <= target_ns:
                    end_ns = boundaries[cursor]
                    if end_ns > current_ns:
                        widths.extend(expand_dynamics_interval((end_ns - current_ns) / 1e9))
                    current_ns = end_ns
                    cursor += 1
                if current_ns < target_ns:
                    widths.extend(expand_dynamics_interval((target_ns - current_ns) / 1e9))
                    current_ns = target_ns
                end_indices.append(len(widths))
            if metadata is not None:
                metadata.update(calendar_only=True, calendar_owner="SatelliteFlightSoftwareRuntime",
                                fsw_telemetry_executed=False)
            return np.asarray(widths, dtype=np.float64), end_indices

    original_boundary = None
    original_step = None
    if runtime is not None:
        original_boundary = runtime.next_hard_boundary_ns

        def record_boundary(*args: Any, **kwargs: Any) -> int:
            after_ns = int(kwargs.get("after_time_ns", args[0] if args else 0))
            end_ns = int(original_boundary(*args, **kwargs))
            width_s = (end_ns - after_ns) / 1.0e9
            if width_s <= 0.0 or not np.isfinite(width_s):
                raise ValueError("validated OEL scheduler produced a nonpositive interval")
            widths.extend(expand_dynamics_interval(float(width_s)))
            return end_ns

        runtime.next_hard_boundary_ns = record_boundary  # type: ignore[method-assign]
    else:
        # This branch is primarily a contract check for trajectory-only test
        # artifacts.  PassiveHistoryCache may bypass dynamics.step entirely,
        # in which case the deterministic nested helper below is authoritative.
        propagator = getattr(dynamics, "orbit_propagator", None)
        if propagator is None:
            raise ValueError("native ONP schedule capture requires target dynamics")
        original_step = propagator.propagate

        def record_step(*args: Any, **kwargs: Any) -> Any:
            dt_value = kwargs.get("dt_s")
            if dt_value is None and len(args) >= 2:
                dt_value = args[1]
            width_s = float(dt_value)
            if width_s <= 0.0 or not np.isfinite(width_s):
                raise ValueError("validated OEL dynamics produced a nonpositive interval")
            widths.append(width_s)
            return original_step(*args, **kwargs)

        propagator.propagate = record_step

    end_indices: list[int] = []
    current_time = float(initial.time_s)
    try:
        for epoch in epochs:
            target = float(epoch)
            if target < current_time - 1.0e-9:
                raise ValueError("ONP observation epochs precede the active simulation epoch")
            if target > current_time + 1.0e-12:
                snapshot = session.step(target - current_time)
                current_time = float(snapshot.time_s)
                if abs(current_time - target) > max(1.0e-10, 1.0e-12 * max(abs(target), 1.0)):
                    raise ValueError(
                        "validated OEL scheduler ended before requested observation epoch: "
                        f"requested={target}, reached={current_time}"
                    )
            end_indices.append(len(widths))
    finally:
        if runtime is not None and original_boundary is not None:
            runtime.next_hard_boundary_ns = original_boundary  # type: ignore[method-assign]
        elif original_step is not None:
            propagator.propagate = original_step

    if runtime is None and not widths and np.any(epochs > 1.0e-12):
        return _onp_native_step_schedule(epochs, fallback_step_s)
    return np.asarray(widths, dtype=np.float64), end_indices


def _native_states_require_python_fallback(
    states: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
) -> bool:
    """Return whether native sampled states require maintained OEL evaluation."""

    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or not np.all(np.isfinite(values)):
        return True
    radii = np.linalg.norm(values[:, :3], axis=1)
    if np.any(~np.isfinite(radii)) or np.any(radii <= float(EARTH_RADIUS_KM)):
        return True
    if initial_state is not None:
        initial = np.asarray(initial_state, dtype=float).reshape(-1)
        if initial.size < 3 or not np.all(np.isfinite(initial[:3])):
            return True
        if float(np.linalg.norm(initial[:3])) <= float(EARTH_RADIUS_KM):
            return True
    return False


def _onp_native_context_support_reason(
    *,
    selected_parameters: Sequence[str],
    dynamics_model: str,
    orbit_force_model: Mapping[str, Any],
    base_specs: Mapping[str, Any] | None = None,
    attitude_source: str,
    attitude_history_rows: Sequence[Mapping[str, Any]],
) -> str | None:
    """Explain why the fused native ONP OD envelope is unavailable."""

    if list(selected_parameters) != ["state"]:
        return "fused native ONP OD currently estimates the Cartesian state only"
    if str(dynamics_model or "two_body").strip().lower() != "two_body":
        return "fused native ONP OD currently supports the two_body model only"
    specs = dict(base_specs or {})
    unsupported_specs = sorted(set(specs) - {"mass_kg"})
    if unsupported_specs:
        return "base spacecraft specifications require the Python path: " + ", ".join(unsupported_specs)
    force_model = dict(orbit_force_model or {})
    allowed_keys = {
        "model", "j2", "drag", "srp", "orbit_substep_s", "integrator",
        "numeric_backend", "propagation_method",
    }
    unknown_keys = sorted(set(force_model) - allowed_keys)
    if unknown_keys:
        if any(key in force_model for key in ("plugins", "custom_forces", "force_plugins")):
            return "custom force plugins require the Python callback path"
        return "force model fields require the Python path: " + ", ".join(unknown_keys)
    if str(force_model.get("model", "two_body") or "two_body").strip().lower() != "two_body":
        return "fused native ONP OD requires a two_body force model"
    if str(force_model.get("propagation_method", "special") or "special").strip().lower() != "special":
        return "fused native ONP OD requires special ECI propagation"
    backend = normalize_numeric_backend(force_model.get("numeric_backend", "rust"))
    if backend not in {"python", "rust"}:
        return "force model numeric_backend must be python or rust"
    for flag in ("j2", "drag", "srp"):
        if flag in force_model and not isinstance(force_model[flag], bool):
            return f"force model {flag} must be boolean"
    if any(bool(force_model.get(key, False)) for key in (
        "j3", "j4", "drag", "srp", "third_body_sun", "third_body_moon",
        "third_body_planets", "spherical_harmonics", "relativity", "tidal",
    )):
        return "fused native ONP OD supports only two_body or J2 forces"
    if force_model.get("atmosphere_model") not in (None, ""):
        return "atmosphere configuration requires the Python force path"
    if str(force_model.get("integrator", "rk4") or "rk4").strip().lower() != "rk4":
        return "fused native ONP OD currently supports RK4 exact-epoch stepping only"
    if force_model.get("plugins") or force_model.get("custom_forces") or force_model.get("force_plugins"):
        return "custom force plugins require the Python callback path"
    if str(attitude_source or "none").strip().lower() not in {"", "none", "off"}:
        return "attitude-coupled OD requires the Python path"
    if attitude_history_rows:
        return "attitude history requires the Python path"
    return None
