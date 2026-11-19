"""Focused construction owner for v2 satellite flight-software runtimes."""

from __future__ import annotations

import importlib
from math import pi
from typing import Any

from sim.dynamics.orbit.cr3bp import cr3bp_moon_state_km_s
from sim.flight_software import (
    AttitudeReferenceFlightSoftwareStack,
    AttitudeReferenceStackConfig,
    ClockScale,
    ClockTag,
    ConstraintDefinition,
    ConstraintKind,
    FrameId,
    GameAerodynamicEffectorBinding,
    GamePilotInputProfile,
    GamePilotMode,
    GamePilotReferenceFlightSoftwareStack,
    GamePilotReferenceStackConfig,
    GoalDefinition,
    GoalMode,
    LowThrustReferenceFlightSoftwareStack,
    LowThrustReferenceStackConfig,
    OnboardMissionConfigurationLoad,
    OrbitReferenceFlightSoftwareStack,
    OrbitReferenceStackConfig,
    PassiveFlightSoftwareStack,
    PassiveStackConfig,
    RpoReferenceFlightSoftwareStack,
    RpoReferenceStackConfig,
    TelemetryField,
    from_primitive,
)
from sim.gnc.attitude_v2 import (
    AttitudeAllocatorConfig,
    AttitudeAllocatorKind,
    AttitudeReferenceConfig,
    AttitudeReferenceMode,
    QuaternionTorqueController,
)
from sim.gnc.executive_v2 import ActionDefinition, ActionKind, ReferenceExecutiveConfig
from sim.gnc.navigation_v2 import LoadedOwnState, NavigationInitializationMode, OrbitFilterKind
from sim.gnc.operations_v2 import (
    AdcsModeConfig,
    ConjunctionConfig,
    HcwManeuverConfig,
    HealthManagerConfig,
    MomentumUnloadConfig,
    ResourceLimits,
)
from sim.gnc.orbit_v2 import (
    RcsThrusterBelief,
    ScheduledBurn,
    TranslationAllocatorConfig,
    TranslationAllocatorKind,
    TranslationControlConfig,
    TranslationControlLaw,
    TranslationMode,
)
from sim.knowledge.object_tracking import KnowledgeConditionConfig
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs
from sim.runtime.satellites.flight_software_runtime import (
    SatelliteFlightSoftwareRuntime,
    aerodynamic_effector_device,
    cmg_device,
    continuous_engine_device,
    docking_mechanism_device,
    ideal_wrench_device,
    magnetorquer_device,
    rcs_thruster_device,
    reaction_wheel_device,
)


def build_satellite_flight_software_runtime(
    *,
    object_id: str,
    agent_cfg: Any,
    scenario_cfg: Any,
    mass_kg: float,
    specific_impulse_s: float | None = None,
    dry_mass_kg: float = 0.0,
) -> SatelliteFlightSoftwareRuntime | None:
    """Build one complete stack without exposing simulator truth to plugins."""

    if str(getattr(agent_cfg, "runtime_profile", "flight_software") or "flight_software") == "trajectory_only":
        return None
    section = getattr(agent_cfg, "flight_software", None)
    if section is None:
        specs = dict(getattr(agent_cfg, "specs", {}) or {})
        if float(specs.get("fuel_mass_kg", 0.0) or 0.0) > 0.0 and specific_impulse_s is None:
            raise ValueError(f"Satellite {object_id!r} declares game-mode propellant without a specific impulse.")
        return build_game_flight_software_runtime(
            object_id=object_id,
            agent_cfg=agent_cfg,
            scenario_cfg=scenario_cfg,
            mass_kg=mass_kg,
            specific_impulse_s=specific_impulse_s,
            dry_mass_kg=dry_mass_kg,
        )
    stack_id = str(section.stack or "")
    profile_id = str(getattr(section, "profile", "") or "") or None
    task_period_s = _stack_task_period_s(stack_id, section.task_period_s, scenario_cfg)
    task_period_ns = max(1, int(round(task_period_s * 1.0e9)))
    inertial = FrameId("OEL/ECI/J2000", "frames-v1")
    body = FrameId(f"OEL/BODY/{object_id}", "frames-v1")
    params = dict(section.params or {})
    initial_checkpoint = None if section.checkpoint is None else dict(section.checkpoint)
    knowledge = dict(getattr(agent_cfg, "knowledge", {}) or {})
    sensor_period_s = float(knowledge.get("refresh_rate_s", task_period_s) or task_period_s)
    sensor_period_ns = max(1, int(round(sensor_period_s * 1.0e9)))
    sensor_error = dict(knowledge.get("sensor_error", {}) or {})
    sensor_conditions = (
        KnowledgeConditionConfig.from_knowledge(knowledge, default_period_s=sensor_period_s)
        if knowledge.get("conditions")
        else None
    )
    navigation_mode = NavigationInitializationMode(
        str(params.get("navigation_initialization", "cold" if knowledge else "ideal"))
    )
    ideal_navigation = navigation_mode is NavigationInitializationMode.IDEAL
    sensor_seed = int(sensor_error.get("seed", 0) or 0)
    specs = dict(getattr(agent_cfg, "specs", {}) or {})
    if (
        float(specs.get("fuel_mass_kg", 0.0) or 0.0) > 0.0
        and specific_impulse_s is None
        and str(section.hardware_profile or "")
        in {"hardware.ideal_wrench.v1", "hardware.continuous_engine.v1", "hardware.rcs.v1"}
    ):
        raise ValueError(
            f"Satellite {object_id!r} declares propellant but its v2 force hardware has no specific impulse. "
            "Set specs.isp_s, specs.thruster_isp_s, a supported thruster preset, or hardware-specific isp_s."
        )
    initial_mission_load = (
        None if section.mission_load is None else from_primitive(OnboardMissionConfigurationLoad, section.mission_load)
    )
    reference_object_id = str(params.get("reference_object_id", "") or _reference_object_id(agent_cfg))

    if stack_id == "fsw.passive":
        stack = PassiveFlightSoftwareStack(
            PassiveStackConfig(
                satellite_id=object_id,
                emit_diagnostics=bool(params.get("emit_diagnostics", True)),
                ideal_navigation=ideal_navigation,
                body_frame=body,
                inertial_frame=inertial,
                measurement_stale_after_s=float(params.get("measurement_stale_after_s", 30.0)),
                expected_sensor_frames=(("ideal_own_state", inertial),),
            ),
            numeric_backend=params.get("numeric_backend", "rust"),
        )
        return SatelliteFlightSoftwareRuntime(
            satellite_id=object_id,
            stack=stack,
            devices=(),
            hardware={},
            inertial_frame=inertial,
            body_frame=body,
            task_period_ns=task_period_ns,
            sensor_period_ns=sensor_period_ns,
            tick_period_ns=1,
            profile_id=profile_id,
            profile_params=params,
            initial_mission_load=initial_mission_load,
            dry_mass_kg=dry_mass_kg,
            ideal_navigation=ideal_navigation,
            sensor_error=sensor_error,
            sensor_conditions=sensor_conditions,
            sensor_seed=sensor_seed,
            initial_checkpoint=initial_checkpoint,
        )

    if stack_id == "fsw.attitude_reference":
        actuator_frame = FrameId(f"OEL/ACTUATOR/{object_id}/attitude", "frames-v1")
        hardware_profile = str(section.hardware_profile or "hardware.ideal_wrench.v1")
        max_torque = float(
            params.get(
                "max_torque_n_m",
                5.0e-4 if hardware_profile == "hardware.magnetorquer.v1" else 0.08,
            )
        )
        axes_body = tuple(
            tuple(float(component) for component in axis)
            for axis in params.get("wheel_axes_body", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        )
        coordinate_limits: tuple[float, ...]
        cmg_momentum = tuple(float(value) for value in params.get("cmg_momentum_n_m_s", (1.0, 1.0, 1.0)))
        magnetic_field = tuple(float(value) for value in params.get("magnetic_field_body_t", (0.0, 0.0, 3.0e-5)))
        momentum_unload = None
        extra_devices = []
        extra_hardware = {}
        if hardware_profile in {"hardware.reaction_wheels.v1", "hardware.reaction_wheels_magnetorquer.v1"}:
            allocator_kind = AttitudeAllocatorKind.REACTION_WHEEL
            coordinate_limits = tuple(float(value) for value in params.get("wheel_max_torque_n_m", (max_torque,)))
            wheel_momentum_limits = tuple(float(value) for value in params.get("wheel_max_momentum_n_m_s", (1.0,)))
            expanded_momentum_limits = (
                wheel_momentum_limits * len(axes_body) if len(wheel_momentum_limits) == 1 else wheel_momentum_limits
            )
            initial_wheel_momentum_raw = tuple(
                float(value) for value in params.get("wheel_initial_momentum_n_m_s", (0.0,) * len(axes_body))
            )
            initial_wheel_momentum = (
                initial_wheel_momentum_raw * len(axes_body)
                if len(initial_wheel_momentum_raw) == 1
                else initial_wheel_momentum_raw
            )
            device, model = reaction_wheel_device(
                object_id,
                "attitude",
                actuator_frame,
                axes_body=axes_body,
                max_torque_n_m=coordinate_limits,
                max_momentum_n_m_s=wheel_momentum_limits,
                initial_momentum_n_m_s=initial_wheel_momentum,
                inertia_kg_m2=params.get("wheel_inertia_kg_m2"),
                max_speed_rad_s=params.get("wheel_max_speed_rad_s"),
            )
            if hardware_profile == "hardware.reaction_wheels_magnetorquer.v1":
                torquer_frame = FrameId(f"OEL/ACTUATOR/{object_id}/momentum_dump", "frames-v1")
                max_dipole = float(params.get("momentum_dump_max_dipole_a_m2", 20.0))
                torquer_device, torquer_model = magnetorquer_device(
                    object_id,
                    "momentum_dump",
                    torquer_frame,
                    max_dipole_a_m2=(max_dipole,),
                    magnetic_field_body_t=magnetic_field,  # type: ignore[arg-type]
                )
                extra_devices.append(torquer_device)
                extra_hardware["momentum_dump"] = torquer_model
                momentum_unload = MomentumUnloadConfig(
                    start_fraction=float(params.get("momentum_dump_start_fraction", 0.8)),
                    stop_fraction=float(params.get("momentum_dump_stop_fraction", 0.55)),
                    wheel_max_momentum_n_m_s=expanded_momentum_limits,
                    wheel_axes_body=axes_body,
                    gain=float(params.get("momentum_dump_gain", 0.25)),
                    max_dipole_a_m2=max_dipole,
                    command_validity_ticks=task_period_ns,
                )
        elif hardware_profile == "hardware.magnetorquer.v1":
            allocator_kind = AttitudeAllocatorKind.MAGNETORQUER
            coordinate_limits = tuple(float(value) for value in params.get("max_dipole_a_m2", (20.0,)))
            device, model = magnetorquer_device(
                object_id,
                "attitude",
                actuator_frame,
                max_dipole_a_m2=coordinate_limits,
                magnetic_field_body_t=magnetic_field,  # type: ignore[arg-type]
            )
        elif hardware_profile == "hardware.cmg.v1":
            allocator_kind = AttitudeAllocatorKind.CMG
            coordinate_limits = tuple(float(value) for value in params.get("max_gimbal_rate_rad_s", (0.08,)))
            device, model = cmg_device(
                object_id,
                "attitude",
                actuator_frame,
                momentum_n_m_s=cmg_momentum,
                max_gimbal_rate_rad_s=coordinate_limits,
            )
        else:
            allocator_kind = AttitudeAllocatorKind.IDEAL_WRENCH
            coordinate_limits = (max_torque, max_torque, max_torque)
            device, model = ideal_wrench_device(
                object_id,
                "attitude",
                actuator_frame,
                max_force_n=0.0,
                max_torque_n_m=max_torque,
            )
        reference_mode = AttitudeReferenceMode(str(params.get("reference_mode", "quaternion")))
        reference = AttitudeReferenceConfig(
            mode=reference_mode,
            quaternion_bn=tuple(float(value) for value in params.get("quaternion_bn", (1.0, 0.0, 0.0, 0.0))),
            ric_axis=str(params.get("ric_axis", "radial_out")),
            boresight_body=tuple(float(value) for value in params.get("boresight_body", (1.0, 0.0, 0.0))),
            target_position_eci_m=(
                None
                if params.get("target_position_eci_m") is None
                else tuple(float(value) for value in params["target_position_eci_m"])
            ),
            thrust_direction_eci=(
                None
                if params.get("thrust_direction_eci") is None
                else tuple(float(value) for value in params["thrust_direction_eci"])
            ),
            replay_times_ns=tuple(int(round(float(value) * 1.0e9)) for value in params.get("replay_times_s", ())),
            replay_quaternions_bn=tuple(
                tuple(float(component) for component in value) for value in params.get("replay_quaternions_bn", ())
            ),
            validity_ticks=task_period_ns,
        )
        stack = AttitudeReferenceFlightSoftwareStack(
            AttitudeReferenceStackConfig(
                object_id,
                body,
                inertial,
                AttitudeAllocatorConfig(
                    object_id,
                    allocator_kind,
                    "attitude",
                    actuator_frame,
                    axes_body=axes_body,
                    limits=coordinate_limits,
                    cmg_momentum_n_m_s=cmg_momentum,
                ),
                reference=reference,
                controller=QuaternionTorqueController(
                    kp=_gain3(params.get("kp", 0.25)),
                    kd=_gain3(params.get("kd", 1.0)),
                    max_torque_n_m=max_torque,
                ),
                measurement_stale_after_s=float(params.get("measurement_stale_after_s", 30.0)),
                health=_health_config(
                    params,
                    default_fallbacks=(
                        (("attitude", "momentum_dump"),)
                        if hardware_profile == "hardware.reaction_wheels_magnetorquer.v1"
                        else ()
                    ),
                    isolate_on_saturation=False,
                ),
                momentum_unload=momentum_unload,
                mode_config=AdcsModeConfig(
                    detumble_entry_rate_rad_s=float(params.get("detumble_entry_rate_rad_s", 0.5)),
                    detumble_exit_rate_rad_s=float(params.get("detumble_exit_rate_rad_s", 0.02)),
                ),
            ),
            numeric_backend=params.get("numeric_backend", "rust"),
        )
        return SatelliteFlightSoftwareRuntime(
            satellite_id=object_id,
            stack=stack,
            devices=(device, *extra_devices),
            hardware={"attitude": model, **extra_hardware},
            inertial_frame=inertial,
            body_frame=body,
            task_period_ns=task_period_ns,
            sensor_period_ns=sensor_period_ns,
            tick_period_ns=1,
            profile_id=profile_id,
            profile_params=params,
            initial_mission_load=initial_mission_load,
            # Coarse-Sun recovery must remain available when fine attitude
            # navigation is unavailable, regardless of the primary mode.
            ideal_sun_sensor=True,
            ideal_magnetic_field_body_t=(
                magnetic_field
                if hardware_profile in {"hardware.magnetorquer.v1", "hardware.reaction_wheels_magnetorquer.v1"}
                else None
            ),
            initial_jd_utc=scenario_cfg.simulator.initial_jd_utc,
            dry_mass_kg=dry_mass_kg,
            ideal_navigation=ideal_navigation,
            sensor_error=sensor_error,
            sensor_conditions=sensor_conditions,
            sensor_seed=sensor_seed,
            initial_checkpoint=initial_checkpoint,
        )

    if stack_id in {"fsw.orbit_reference", "fsw.rpo_reference", "fsw.low_thrust_reference"}:
        return _build_translation_runtime(
            object_id=object_id,
            stack_id=stack_id,
            profile_id=profile_id,
            params=params,
            agent_cfg=agent_cfg,
            mass_kg=mass_kg,
            task_period_ns=task_period_ns,
            sensor_period_ns=sensor_period_ns,
            inertial=inertial,
            body=body,
            reference_object_id=reference_object_id,
            hardware_profile=str(section.hardware_profile or "hardware.ideal_wrench.v1"),
            initial_mission_load=initial_mission_load,
            initial_jd_utc=scenario_cfg.simulator.initial_jd_utc,
            specific_impulse_s=specific_impulse_s,
            dry_mass_kg=dry_mass_kg,
            navigation_initialization=navigation_mode,
            sensor_error=sensor_error,
            sensor_conditions=sensor_conditions,
            sensor_seed=sensor_seed,
            initial_checkpoint=initial_checkpoint,
            live_game_fast_path=bool(dict(getattr(scenario_cfg, "metadata", {}).get("game", {}) or {})),
            game_numeric_backend=str(dict(getattr(scenario_cfg, "metadata", {}).get("game", {}) or {}).get("backend", "python")),
        )
    if stack_id == "fsw.game_pilot_reference":
        return build_game_flight_software_runtime(
            object_id=object_id,
            agent_cfg=agent_cfg,
            scenario_cfg=scenario_cfg,
            mass_kg=mass_kg,
            specific_impulse_s=specific_impulse_s,
            dry_mass_kg=dry_mass_kg,
            settings={
                **dict(getattr(scenario_cfg, "metadata", {}).get("game", {}) or {}),
                **params,
                "controlled_object_id": object_id,
                "flight_software_stack": stack_id,
            },
            initial_checkpoint=initial_checkpoint,
        )
    if section.module and section.class_name:
        try:
            module = importlib.import_module(str(section.module))
            stack_type = getattr(module, str(section.class_name))
            stack = stack_type(**params)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to construct requested plugin flight-software stack "
                f"{section.module}.{section.class_name}: {exc}"
            ) from exc
        devices = ()
        hardware: dict[str, Any] = {}
        if section.hardware_profile == "hardware.ideal_wrench.v1":
            physical = dict(specs.get("flight_software_hardware", {}) or {})
            actuator_id = str(physical.get("actuator_id", params.get("actuator_id", "wrench")))
            actuator_frame = FrameId(f"OEL/ACTUATOR/{object_id}/{actuator_id}", "frames-v1")
            device, model = ideal_wrench_device(
                object_id,
                actuator_id,
                actuator_frame,
                max_force_n=float(physical.get("max_force_n", 1.0e9)),
                max_torque_n_m=float(physical.get("max_torque_n_m", 1.0e9)),
                specific_impulse_s=specific_impulse_s,
            )
            devices = (device,)
            hardware[actuator_id] = model
        if getattr(scenario_cfg.simulator, "docking", {}).get("commanded"):
            device, model = docking_mechanism_device(object_id)
            devices = (*devices, device)
            hardware[device.actuator_id] = model
        return SatelliteFlightSoftwareRuntime(
            satellite_id=object_id,
            stack=stack,
            devices=devices,
            hardware=hardware,
            inertial_frame=inertial,
            body_frame=body,
            task_period_ns=task_period_ns,
            sensor_period_ns=sensor_period_ns,
            tick_period_ns=1,
            reference_object_id=reference_object_id or None,
            initial_mission_load=initial_mission_load,
            dry_mass_kg=dry_mass_kg,
            ideal_navigation=ideal_navigation,
            sensor_error=sensor_error,
            sensor_conditions=sensor_conditions,
            sensor_seed=sensor_seed,
            initial_checkpoint=initial_checkpoint,
        )
    raise ValueError(f"Unsupported flight-software selection for satellite {object_id!r}.")


def _build_translation_runtime(
    *,
    object_id: str,
    stack_id: str,
    profile_id: str | None,
    params: dict[str, Any],
    agent_cfg: Any,
    mass_kg: float,
    task_period_ns: int,
    inertial: FrameId,
    body: FrameId,
    reference_object_id: str,
    hardware_profile: str,
    initial_mission_load: OnboardMissionConfigurationLoad | None,
    initial_jd_utc: float | None,
    specific_impulse_s: float | None,
    dry_mass_kg: float,
    navigation_initialization: NavigationInitializationMode,
    sensor_error: dict[str, Any],
    sensor_conditions: KnowledgeConditionConfig | None,
    sensor_seed: int,
    sensor_period_ns: int,
    initial_checkpoint: dict[str, object] | None,
    live_game_fast_path: bool = False,
    game_numeric_backend: str = "rust",
) -> SatelliteFlightSoftwareRuntime:
    defaults = {
        "fsw.orbit_reference": TranslationMode.STATIONKEEPING,
        "fsw.rpo_reference": TranslationMode.RIC_HOLD,
        "fsw.low_thrust_reference": TranslationMode.LOW_THRUST_PHASING,
    }
    mode = TranslationMode(str(params.get("translation_mode", defaults[stack_id].value)))
    max_accel = float(params.get("max_acceleration_m_s2", 0.02))
    max_force = float(params.get("max_force_n", max(max_accel * mass_kg, 1.0e-9)))
    if mode in {TranslationMode.INTERCEPT_COAST, TranslationMode.PREDICTIVE_EVASION}:
        if hardware_profile != "hardware.ideal_wrench.v1":
            raise ValueError(
                "predictive intercept/evasion currently requires hardware.ideal_wrench.v1 so the searched "
                "actuation envelope matches the authoritative allocator"
            )
        max_accel = min(max_accel, max_force / mass_kg)
    actuator_frame = FrameId(f"OEL/ACTUATOR/{object_id}/translation", "frames-v1")
    relative = FrameId(f"OEL/RIC/{reference_object_id}", "frames-v1")
    target_relative = tuple(
        float(value) for value in params.get("target_relative_state_ric_m", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    )
    target_coes_raw = {str(key): float(value) for key, value in dict(params.get("target_coes", {}) or {}).items()}
    target_semi_major_axis_m = params.get("target_semi_major_axis_m")
    if target_semi_major_axis_m is None:
        target_a_km = target_coes_raw.get("a_km", target_coes_raw.get("semi_major_axis_km"))
        if target_a_km is not None:
            target_semi_major_axis_m = target_a_km * 1.0e3
    target_eccentricity = params.get("target_eccentricity")
    if target_eccentricity is None:
        target_eccentricity = target_coes_raw.get("ecc", target_coes_raw.get("e", 0.0))
    scheduled_burns = tuple(
        ScheduledBurn(
            int(round(float(item.get("start_time_s", 0.0)) * 1.0e9)),
            int(round(float(item["duration_s"]) * 1.0e9)),
            tuple(float(value) / float(item["duration_s"]) for value in item["delta_v_m_s"]),
            str(item.get("frame", "eci") or "eci").strip().lower(),
        )
        for item in list(params.get("scheduled_burns", []) or [])
    )
    if scheduled_burns and "translation_mode" not in params:
        if stack_id != "fsw.orbit_reference":
            raise ValueError("scheduled_burns are supported only by fsw.orbit_reference")
        mode = TranslationMode.SCHEDULED_BURN
    elif scheduled_burns and stack_id != "fsw.orbit_reference":
        raise ValueError("scheduled_burns are supported only by fsw.orbit_reference")
    control = TranslationControlConfig(
        mode,
        float(params.get("assumed_mass_kg", mass_kg)),
        max_accel,
        target_state_eci=(
            None
            if params.get("target_state_eci_m_m_s") is None
            else tuple(float(value) for value in params["target_state_eci_m_m_s"])
        ),
        target_semi_major_axis_m=(None if target_semi_major_axis_m is None else float(target_semi_major_axis_m)),
        target_eccentricity=float(target_eccentricity),
        eccentricity_tolerance=float(params.get("eccentricity_tolerance", 1.0e-4)),
        target_relative_state_ric=target_relative,
        waypoints_ric=tuple(
            tuple(float(component) for component in waypoint)
            for waypoint in list(params.get("waypoints_ric", []) or [])
        ),
        control_axis_mask=tuple(float(value) for value in params.get("control_axis_mask", (1.0, 1.0, 1.0))),
        kp_position_s2=float(params.get("kp_position_s2", 4.0e-6)),
        kd_velocity_s_inv=float(params.get("kd_velocity_s_inv", 4.0e-3)),
        mean_motion_rad_s=float(params.get("mean_motion_rad_s", 0.0)),
        approach_speed_m_s=float(params.get("approach_speed_m_s", 0.1)),
        slowdown_distance_m=float(params.get("slowdown_distance_m", 250.0)),
        terminal_box_m=float(params.get("terminal_box_m", 100.0)),
        terminal_max_closing_speed_m_s=float(params.get("terminal_max_closing_speed_m_s", 0.05)),
        retreat_speed_m_s=float(params.get("retreat_speed_m_s", 0.2)),
        retreat_coast_range_m=float(params.get("retreat_coast_range_m", 1000.0)),
        position_tolerance_m=float(params.get("position_tolerance_m", 20.0)),
        velocity_tolerance_m_s=float(params.get("velocity_tolerance_m_s", 0.02)),
        target_id=reference_object_id or None,
        validity_ticks=task_period_ns,
        scheduled_burns=scheduled_burns,
        atmospheric_raise_start_ns=int(round(float(params.get("raise_start_s", 0.0)) * 1.0e9)),
        atmospheric_raise_end_ns=int(round(float(params.get("raise_end_s", 0.0)) * 1.0e9)),
        atmospheric_prograde_acceleration_m_s2=float(params.get("prograde_acceleration_m_s2", 0.0)),
        atmospheric_min_raise_altitude_m=float(params.get("min_raise_altitude_m", 0.0)),
        atmospheric_pass_entry_altitude_m=(
            None if params.get("pass_entry_altitude_m") is None else float(params["pass_entry_altitude_m"])
        ),
        atmospheric_pass_exit_altitude_m=(
            None if params.get("pass_exit_altitude_m") is None else float(params["pass_exit_altitude_m"])
        ),
        atmospheric_recovery_delta_v_m_s=float(params.get("recovery_delta_v_m_s", 0.0)),
        orbital_element_control_law=str(params.get("orbital_element_control_law", "energy_eccentricity")),
        target_coes=tuple(target_coes_raw.items()),
        controlled_elements=tuple(
            str(value) for value in params.get("controlled_elements", ("a", "ecc", "inc", "raan", "argp"))
        ),
        energy_gain_per_s=float(params.get("energy_gain_per_s", 1.0e-3)),
        eccentricity_gain_per_s=float(params.get("eccentricity_gain_per_s", 5.0e-4)),
        plane_gain_per_s=float(params.get("plane_gain_per_s", 5.0e-4)),
        control_law=TranslationControlLaw(str(params.get("control_law", "reference_pd"))),
        control_design_dt_s=float(params.get("control_design_dt_s", task_period_ns / 1.0e9)),
        lqr_q_weights=tuple(
            float(value)
            for value in params.get(
                "lqr_q_weights",
                (8660.0, 8660.0, 8660.0, 1330.0, 1330.0, 1330.0),
            )
        ),
        lqr_r_weights=tuple(float(value) for value in params.get("lqr_r_weights", (1.94e13, 1.94e13, 1.94e13))),
        rmoe_target_radial_center_m=float(params.get("rmoe_target_radial_center_m", 0.0)),
        rmoe_target_in_track_center_m=float(params.get("rmoe_target_in_track_center_m", 0.0)),
        rmoe_target_in_track_drift_rate_m_s=float(params.get("rmoe_target_in_track_drift_rate_m_s", 0.0)),
        rmoe_target_cross_track_amplitude_m=float(params.get("rmoe_target_cross_track_amplitude_m", 0.0)),
        rmoe_max_drift_rate_m_s=float(params.get("rmoe_max_drift_rate_m_s", 0.02)),
        rmoe_close_zone_m=float(params.get("rmoe_close_zone_m", 50.0)),
        rmoe_cross_track_burn_gate_m=float(params.get("rmoe_cross_track_burn_gate_m", 50.0)),
        transfer_time_s=float(params.get("transfer_time_s", 4_800.0)),
        burn_time_constant_s=float(params.get("burn_time_constant_s", 45.0)),
        correction_interval_s=float(params.get("correction_interval_s", 300.0)),
        velocity_deadband_m_s=float(params.get("velocity_deadband_m_s", 0.015)),
        final_brake_start_s=float(params.get("final_brake_start_s", 180.0)),
        terminal_start_s=float(params.get("terminal_start_s", 750.0)),
        terminal_range_m=float(params.get("terminal_range_m", 200.0)),
        thrust_window_period_s=float(params.get("thrust_window_period_s", 0.0)),
        thrust_window_duration_s=float(params.get("thrust_window_duration_s", 0.0)),
        thrust_window_phase_s=float(params.get("thrust_window_phase_s", 0.0)),
        thrust_command_deadband_m_s2=float(params.get("thrust_command_deadband_m_s2", 0.0)),
        element_averaging_window_s=float(params.get("element_averaging_window_s", 0.0)),
        prediction_horizon_s=float(params.get("prediction_horizon_s", 1_800.0)),
        prediction_step_s=float(params.get("prediction_step_s", 30.0)),
        prediction_decision_interval_s=float(params.get("prediction_decision_interval_s", 120.0)),
        prediction_pulse_duration_s=float(params.get("prediction_pulse_duration_s", 60.0)),
        capture_radius_m=float(params.get("capture_radius_m", 100.0)),
        capture_margin_m=float(params.get("capture_margin_m", 20.0)),
        opponent_max_acceleration_m_s2=float(params.get("opponent_max_acceleration_m_s2", 0.02)),
        prediction_acceleration_fractions=tuple(
            float(value) for value in params.get("prediction_acceleration_fractions", (0.5, 1.0))
        ),
    )
    use_continuous_engine = hardware_profile == "hardware.continuous_engine.v1"
    use_rcs = hardware_profile == "hardware.rcs.v1"
    rcs_rows = list(params.get("rcs_thrusters", []) or [])
    if use_rcs and not rcs_rows:
        rcs_rows = [
            {"thruster_id": thruster_id, "direction_body": direction, "max_thrust_n": max_force}
            for thruster_id, direction in (
                ("rcs_x_plus", (1.0, 0.0, 0.0)),
                ("rcs_x_minus", (-1.0, 0.0, 0.0)),
                ("rcs_y_plus", (0.0, 1.0, 0.0)),
                ("rcs_y_minus", (0.0, -1.0, 0.0)),
                ("rcs_z_plus", (0.0, 0.0, 1.0)),
                ("rcs_z_minus", (0.0, 0.0, -1.0)),
            )
        ]
    rcs_beliefs = tuple(
        RcsThrusterBelief(
            str(item["thruster_id"]),
            tuple(float(value) for value in item["direction_body"]),
            float(item.get("max_thrust_n", max_force)),
        )
        for item in rcs_rows
    )
    allocator = TranslationAllocatorConfig(
        object_id,
        TranslationAllocatorKind.CONTINUOUS_ENGINE
        if use_continuous_engine
        else TranslationAllocatorKind.RCS_PULSE
        if use_rcs
        else TranslationAllocatorKind.IDEAL_WRENCH,
        "translation",
        actuator_frame,
        max_force,
        rcs_thrusters=rcs_beliefs,
        pulse_window_s=float(params.get("rcs_pulse_window_s", task_period_ns / 1.0e9)),
        gimbal_limit_rad=float(params.get("gimbal_limit_rad", pi / 2.0)),
    )
    attitude_reference = None
    attitude_allocator = None
    attitude_device = None
    attitude_hardware = None
    attitude_mode_raw = params.get("attitude_reference_mode")
    if attitude_mode_raw not in (None, "", "none"):
        attitude_mode = AttitudeReferenceMode(str(attitude_mode_raw))
        attitude_reference = AttitudeReferenceConfig(
            mode=attitude_mode,
            quaternion_bn=tuple(float(value) for value in params.get("attitude_quaternion_bn", (1.0, 0.0, 0.0, 0.0))),
            ric_axis=str(params.get("attitude_ric_axis", "radial_out")),
            boresight_body=tuple(float(value) for value in params.get("attitude_boresight_body", (1.0, 0.0, 0.0))),
            target_position_eci_m=(
                None
                if params.get("attitude_target_position_eci_m") is None
                else tuple(float(value) for value in params["attitude_target_position_eci_m"])
            ),
            thrust_direction_eci=(
                None
                if params.get("attitude_thrust_direction_eci") is None
                else tuple(float(value) for value in params["attitude_thrust_direction_eci"])
            ),
            replay_times_ns=tuple(
                int(round(float(value) * 1.0e9)) for value in params.get("attitude_replay_times_s", ())
            ),
            replay_quaternions_bn=tuple(
                tuple(float(component) for component in value)
                for value in params.get("attitude_replay_quaternions_bn", ())
            ),
            validity_ticks=task_period_ns,
        )
        max_attitude_torque = float(params.get("max_attitude_torque_n_m", 0.1))
        attitude_frame = FrameId(f"OEL/ACTUATOR/{object_id}/attitude", "frames-v1")
        attitude_allocator = AttitudeAllocatorConfig(
            object_id,
            AttitudeAllocatorKind.IDEAL_WRENCH,
            "attitude",
            attitude_frame,
            limits=(max_attitude_torque, max_attitude_torque, max_attitude_torque),
        )
        attitude_device, attitude_hardware = ideal_wrench_device(
            object_id,
            "attitude",
            attitude_frame,
            max_force_n=0.0,
            max_torque_n_m=max_attitude_torque,
        )
        if params.get("attitude_hardware_profile") == "hardware.reaction_wheels.v1":
            wheel_axes = tuple(tuple(float(v) for v in axis) for axis in params["wheel_axes_body"])
            wheel_torque = tuple(float(v) for v in params["wheel_max_torque_n_m"])
            attitude_allocator = AttitudeAllocatorConfig(
                object_id, AttitudeAllocatorKind.REACTION_WHEEL, "attitude", attitude_frame,
                limits=wheel_torque, axes_body=wheel_axes,
            )
            attitude_device, attitude_hardware = reaction_wheel_device(
                object_id, "attitude", attitude_frame, axes_body=wheel_axes,
                max_torque_n_m=wheel_torque,
                max_momentum_n_m_s=tuple(float(v) for v in params["wheel_max_momentum_n_m_s"]),
                inertia_kg_m2=params.get("wheel_inertia_kg_m2"),
                max_speed_rad_s=params.get("wheel_max_speed_rad_s"),
            )
    goal = GoalDefinition(
        str(params.get("goal_id", "primary")),
        str(params.get("goal_type", mode.value)),
        GoalMode(str(params.get("goal_mode", "terminal"))),
        target_frame=relative if stack_id == "fsw.rpo_reference" else inertial,
        dwell_s=float(params.get("goal_dwell_s", 0.0)),
    )
    executive = ReferenceExecutiveConfig(
        goal,
        mode.value,
        constraints=tuple(
            ConstraintDefinition(
                str(item["constraint_id"]),
                ConstraintKind(str(item.get("kind", "performance_requirement"))),
                str(item["evaluator_id"]),
                tuple(
                    TelemetryField(str(name), value) for name, value in dict(item.get("parameters", {}) or {}).items()
                ),
                tuple(str(value) for value in item.get("applies_to_goal_ids", ())),
                bool(item.get("enabled", True)),
            )
            for item in list(params.get("constraints", []) or [])
        ),
        actions=tuple(
            ActionDefinition(
                str(item["action_id"]),
                str(item["mode"]),
                ActionKind(str(item["kind"])),
                timeout_s=None if item.get("timeout_s") is None else float(item["timeout_s"]),
                duration_s=None if item.get("duration_s") is None else float(item["duration_s"]),
                pulse_count=int(item.get("pulse_count", 1)),
                condition_id=None if item.get("condition_id") is None else str(item["condition_id"]),
            )
            for item in list(params.get("actions", []) or [])
        ),
        recovery_mode=str(params.get("recovery_mode", "passive_retreat")),
        recover_on_fault=bool(params.get("recover_on_fault", True)),
        recovery_clear_dwell_s=float(params.get("recovery_clear_dwell_s", 1.0)),
        recover_on_action_timeout=bool(params.get("recover_on_action_timeout", True)),
        recovery_constraint_kinds=tuple(
            ConstraintKind(str(value)) for value in params.get("recovery_constraint_kinds", ())
        ),
    )
    config_type, stack_type = {
        "fsw.orbit_reference": (OrbitReferenceStackConfig, OrbitReferenceFlightSoftwareStack),
        "fsw.rpo_reference": (RpoReferenceStackConfig, RpoReferenceFlightSoftwareStack),
        "fsw.low_thrust_reference": (LowThrustReferenceStackConfig, LowThrustReferenceFlightSoftwareStack),
    }[stack_id]
    if game_numeric_backend != "python":
        from sim.flight_software.rust_game_stacks import game_stack_type

        stack_type = game_stack_type(stack_type, game_numeric_backend)
    stack = stack_type(
        config_type(
            object_id,
            body,
            inertial,
            relative,
            navigation_initialization,
            control,
            allocator,
            executive,
            loaded_own_state=_loaded_own_state(
                params,
                object_id=object_id,
                tick_period_ns=1,
            ),
            attitude_allocator=attitude_allocator,
            attitude_controller=QuaternionTorqueController(
                kp=tuple(float(value) for value in params.get("attitude_kp", (0.25, 0.25, 0.25))),
                kd=tuple(float(value) for value in params.get("attitude_kd", (1.0, 1.0, 1.0))),
                max_torque_n_m=float(params.get("max_attitude_torque_n_m", 0.1)),
            ),
            attitude_reference=attitude_reference,
            pointing_tolerance_rad=float(params.get("pointing_tolerance_rad", 5.0 * pi / 180.0)),
            require_pointing_for_translation=bool(
                params.get("require_pointing_for_translation", use_continuous_engine or use_rcs)
            ),
            dry_mass_kg=dry_mass_kg,
            navigation_filter=OrbitFilterKind(str(params.get("navigation_filter", "sample_hold"))),
            navigation_alpha=float(params.get("navigation_alpha", 0.85)),
            navigation_beta=float(params.get("navigation_beta", 0.05)),
            navigation_ekf_step_s=float(params.get("navigation_ekf_step_s", task_period_ns * 1.0e-9)),
            navigation_process_noise_diag_si=tuple(
                float(value)
                for value in params.get(
                    "navigation_process_noise_diag_si", (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8, 1.0e-8, 1.0e-8)
                )
            ),
            navigation_measurement_noise_diag_si=tuple(
                float(value)
                for value in params.get("navigation_measurement_noise_diag_si", (25.0, 25.0, 25.0, 0.01, 0.01, 0.01))
            ),
            navigation_initial_covariance_diag_si=tuple(
                float(value)
                for value in params.get(
                    "navigation_initial_covariance_diag_si", (1.0e4, 1.0e4, 1.0e4, 100.0, 100.0, 100.0)
                )
            ),
            navigation_relative_mean_motion_rad_s=float(
                params.get("navigation_relative_mean_motion_rad_s", control.mean_motion_rad_s or 0.0011)
            ),
            navigation_nis_limit=float(params.get("navigation_nis_limit", 30.0)),
            measurement_stale_after_s=float(params.get("measurement_stale_after_s", 30.0)),
            health=_health_config(params),
            resources=ResourceLimits(
                minimum_battery_soc=float(params.get("minimum_battery_soc", 0.15)),
                minimum_available_power_w=float(params.get("minimum_available_power_w", 0.0)),
                maximum_temperature_k=float(params.get("maximum_temperature_k", 333.15)),
                maximum_storage_fraction=float(params.get("maximum_storage_fraction", 0.95)),
                minimum_propellant_kg=float(params.get("minimum_propellant_kg", 0.0)),
            ),
            conjunction=ConjunctionConfig(
                enabled=bool(params.get("conjunction_avoidance_enabled", False)),
                keep_out_radius_m=float(params.get("conjunction_keep_out_radius_m", 100.0)),
                prediction_horizon_s=float(params.get("conjunction_prediction_horizon_s", 600.0)),
                avoidance_delta_v_m_s=float(params.get("conjunction_avoidance_delta_v_m_s", 0.1)),
                maneuver_lead_time_s=float(params.get("conjunction_maneuver_lead_time_s", 30.0)),
            ),
            autonomous_maneuver=HcwManeuverConfig(
                enabled=bool(params.get("autonomous_maneuver_enabled", False)),
                transfer_time_s=float(params.get("maneuver_transfer_time_s", 300.0)),
                target_position_ric_m=tuple(
                    float(value) for value in params.get("maneuver_target_position_ric_m", (0.0, 0.0, 0.0))
                ),
                maximum_delta_v_m_s=float(params.get("maneuver_maximum_delta_v_m_s", 5.0)),
            ),
        ),
        _live_navigation_fast_path=live_game_fast_path,
        numeric_backend=params.get("numeric_backend", "rust"),
    )
    if use_continuous_engine:
        device, model = continuous_engine_device(
            object_id,
            "translation",
            actuator_frame,
            max_thrust_n=max_force,
            specific_impulse_s=specific_impulse_s,
        )
        devices = [device]
        hardware = {"translation": model}
    elif use_rcs:
        devices = []
        hardware = {}
        for thruster, source_row in zip(rcs_beliefs, rcs_rows, strict=True):
            device, model = rcs_thruster_device(
                object_id,
                thruster.thruster_id,
                actuator_frame,
                direction_body=thruster.force_direction_body,
                max_thrust_n=thruster.max_thrust_n,
                position_body_m=tuple(float(value) for value in source_row.get("position_body_m", (0.0, 0.0, 0.0))),
                specific_impulse_s=source_row.get("isp_s", specific_impulse_s),
            )
            devices.append(device)
            hardware[thruster.thruster_id] = model
    else:
        device, model = ideal_wrench_device(
            object_id,
            "translation",
            actuator_frame,
            max_force_n=max_force,
            max_torque_n_m=0.0,
            specific_impulse_s=specific_impulse_s,
        )
        devices = [device]
        hardware = {"translation": model}
    if attitude_device is not None and attitude_hardware is not None:
        devices.append(attitude_device)
        hardware["attitude"] = attitude_hardware
    device_ids = {device.actuator_id for device in devices}
    for primary, backup in stack.config.health.actuator_fallbacks:
        missing = [actuator_id for actuator_id in (primary, backup) if actuator_id not in device_ids]
        if missing:
            raise ValueError(
                "translation actuator_fallbacks must reference configured actuator devices; "
                f"missing: {', '.join(sorted(set(missing)))}"
            )
    return SatelliteFlightSoftwareRuntime(
        satellite_id=object_id,
        stack=stack,
        devices=tuple(devices),
        hardware=hardware,
        inertial_frame=inertial,
        body_frame=body,
        task_period_ns=task_period_ns,
        sensor_period_ns=sensor_period_ns,
        tick_period_ns=1,
        profile_id=profile_id,
        profile_params=params,
        reference_object_id=(
            reference_object_id or None
            if stack_id in {"fsw.rpo_reference", "fsw.low_thrust_reference"}
            or (
                attitude_reference is not None
                and attitude_reference.mode in {AttitudeReferenceMode.TARGET, AttitudeReferenceMode.RIC}
            )
            else None
        ),
        initial_mission_load=initial_mission_load,
        ideal_sun_sensor=attitude_reference is not None and attitude_reference.mode is AttitudeReferenceMode.SUN,
        initial_jd_utc=initial_jd_utc,
        dry_mass_kg=dry_mass_kg,
        ideal_navigation=navigation_initialization is NavigationInitializationMode.IDEAL,
        sensor_error=sensor_error,
        sensor_conditions=sensor_conditions,
        sensor_seed=sensor_seed,
        initial_checkpoint=initial_checkpoint,
    )


def _gain3(value: object) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        values = tuple(float(item) for item in value)
        if len(values) != 3:
            raise ValueError("attitude gain vectors must contain exactly three values")
        return values
    scalar = float(value)
    return (scalar, scalar, scalar)


def _loaded_own_state(
    params: dict[str, Any],
    *,
    object_id: str,
    tick_period_ns: int,
) -> LoadedOwnState | None:
    if str(params.get("navigation_initialization", "ideal")) != NavigationInitializationMode.LOADED.value:
        return None
    position = params.get("loaded_position_eci_m")
    velocity = params.get("loaded_velocity_eci_m_s")
    if position is None or velocity is None:
        raise ValueError("loaded navigation initialization requires loaded_position_eci_m and loaded_velocity_eci_m_s")
    epoch_ns = int(round(float(params.get("loaded_epoch_s", 0.0)) * 1.0e9))
    return LoadedOwnState(
        tuple(float(value) for value in position),
        tuple(float(value) for value in velocity),
        ClockTag(f"{object_id}/onboard", epoch_ns // tick_period_ns, tick_period_ns, ClockScale.ONBOARD),
    )


def build_game_flight_software_runtime(
    *,
    object_id: str,
    agent_cfg: Any,
    scenario_cfg: Any,
    mass_kg: float,
    specific_impulse_s: float | None = None,
    dry_mass_kg: float = 0.0,
    settings: dict[str, Any] | None = None,
    initial_checkpoint: dict[str, object] | None = None,
) -> SatelliteFlightSoftwareRuntime | None:
    game = dict(settings or getattr(scenario_cfg, "metadata", {}).get("game", {}) or {})
    if not game or str(game.get("controlled_object_id", "chaser")) != object_id:
        return None
    stack_id = str(game.get("flight_software_stack", "") or "")
    if stack_id != "fsw.game_pilot_reference":
        return None
    control_mode = str(game.get("control_mode", "") or "").strip().lower()
    profile_id = str(game.get("input_profile", "") or "")
    mode = _game_mode(control_mode, profile_id)
    inertial = FrameId("OEL/ECI/J2000", "frames-v1")
    body = FrameId(f"OEL/BODY/{object_id}", "frames-v1")
    reference_object_id = str(game.get("ric_reference_object_id", "") or _reference_object_id(agent_cfg))
    relative = FrameId(
        f"OEL/RIC/{reference_object_id}",
        "frames-v1",
    )
    translation_frame = FrameId(f"OEL/ACTUATOR/{object_id}/translation", "frames-v1")
    attitude_frame = FrameId(f"OEL/ACTUATOR/{object_id}/attitude", "frames-v1")
    specs = dict(getattr(agent_cfg, "specs", {}) or {})
    max_acceleration_m_s2 = _max_acceleration_m_s2(game, specs, mass_kg)
    operator_impulse_duration_s = max(float(game.get("operator_impulse_duration_s", 1.0e-3)), 1.0e-9)
    operator_max_delta_v_m_s = max(float(game.get("operator_max_burn_delta_v_m_s", 5.0)), 0.0)
    max_force_n = max(
        max_acceleration_m_s2 * mass_kg,
        operator_max_delta_v_m_s / operator_impulse_duration_s * mass_kg,
        1.0e-9,
    )
    task_period_s = _task_period_s(mode, scenario_cfg)
    task_period_ns = max(1, int(round(task_period_s * 1.0e9)))
    knowledge = dict(getattr(agent_cfg, "knowledge", {}) or {})
    sensor_period_s = float(knowledge.get("refresh_rate_s", task_period_s) or task_period_s)
    sensor_period_ns = max(1, int(round(sensor_period_s * 1.0e9)))
    sensor_error = dict(knowledge.get("sensor_error", {}) or {})
    sensor_conditions = (
        KnowledgeConditionConfig.from_knowledge(knowledge, default_period_s=sensor_period_s)
        if knowledge.get("conditions")
        else None
    )
    sensor_seed = int(sensor_error.get("seed", 0) or 0)
    translation_allocator = TranslationAllocatorConfig(
        object_id,
        TranslationAllocatorKind.IDEAL_WRENCH,
        "translation",
        translation_frame,
        max_force_n,
    )
    attitude_allocator = None
    devices = []
    hardware = {}
    translation_device, translation_hardware = ideal_wrench_device(
        object_id,
        "translation",
        translation_frame,
        max_force_n=max_force_n,
        max_torque_n_m=0.0,
        specific_impulse_s=specific_impulse_s,
    )
    devices.append(translation_device)
    hardware["translation"] = translation_hardware
    if mode is GamePilotMode.ATTITUDE_THRUST:
        attitude_allocator = AttitudeAllocatorConfig(
            object_id,
            AttitudeAllocatorKind.IDEAL_WRENCH,
            "attitude",
            attitude_frame,
            limits=(100.0, 100.0, 100.0),
        )
        attitude_device, attitude_hardware = ideal_wrench_device(
            object_id,
            "attitude",
            attitude_frame,
            max_force_n=0.0,
            max_torque_n_m=100.0,
        )
        devices.append(attitude_device)
        hardware["attitude"] = attitude_hardware
    effectors: list[GameAerodynamicEffectorBinding] = []
    if mode is GamePilotMode.AERODYNAMIC:
        aero = dict(game.get("aerodynamic_control", {}) or {})
        deployment_frame = FrameId(f"OEL/ACTUATOR/{object_id}/deployment", "frames-v1")
        bank_frame = FrameId(f"OEL/ACTUATOR/{object_id}/bank", "frames-v1")
        effectors.extend(
            (
                GameAerodynamicEffectorBinding(
                    "deployment", "deployment", "deployment", deployment_frame, "fraction", 0.0, 1.0, 0.5
                ),
                GameAerodynamicEffectorBinding("bank", "bank", "bank", bank_frame, "rad", -pi, pi, 0.0),
            )
        )
        bc_min = float(aero.get("ballistic_coefficient_min_kg_m2", 40.0))
        bc_max = float(aero.get("ballistic_coefficient_max_kg_m2", 200.0))
        bc_rate = float(aero.get("ballistic_coefficient_rate_kg_m2_s", 8.0))
        deployment_rate = bc_rate / max(bc_max - bc_min, 1.0e-12)
        bank_rate = float(aero.get("lift_bank_rate_deg_s", 18.0)) * pi / 180.0
        for binding, rate in zip(effectors, (deployment_rate, bank_rate), strict=True):
            device, model = aerodynamic_effector_device(
                object_id,
                binding.actuator_id,
                binding.coordinate_id,
                binding.actuator_frame,
                unit=binding.unit,
                minimum=binding.minimum,
                maximum=binding.maximum,
                neutral=binding.neutral,
                rate_limit_per_s=rate,
            )
            devices.append(device)
            hardware[binding.actuator_id] = model
    profile = GamePilotInputProfile(profile_id, mode)
    frame_key = str(game.get("relative_frame", "") or "").strip().lower()
    translation_origin_state = (
        tuple(float(value) * 1.0e3 for value in cr3bp_moon_state_km_s())
        if frame_key == "moon_ric" or "moon_ric" in control_mode
        else None
    )
    stack_type = GamePilotReferenceFlightSoftwareStack
    backend = str(game.get("backend", "python"))
    if backend != "python":
        from sim.flight_software.rust_game_stacks import game_stack_type

        stack_type = game_stack_type(stack_type, backend)
    stack = stack_type(
        GamePilotReferenceStackConfig(
            object_id,
            body,
            inertial,
            relative,
            profile,
            translation_allocator,
            mass_kg,
            max_acceleration_m_s2,
            attitude_allocator=attitude_allocator,
            maximum_attitude_rate_rad_s=float(game.get("attitude_rate_deg_s", 8.0)) * pi / 180.0,
            effectors=tuple(effectors),
            validity_ticks=task_period_ns,
            translation_reference_origin_state_eci_m_m_s=translation_origin_state,
            operator_impulse_duration_s=operator_impulse_duration_s,
            measurement_stale_after_s=float(game.get("measurement_stale_after_s", 30.0)),
        )
    )
    runtime = SatelliteFlightSoftwareRuntime(
        satellite_id=object_id,
        stack=stack,
        devices=tuple(devices),
        hardware=hardware,
        inertial_frame=inertial,
        body_frame=body,
        task_period_ns=task_period_ns,
        sensor_period_ns=sensor_period_ns,
        tick_period_ns=1,
        reference_object_id=reference_object_id,
        dry_mass_kg=dry_mass_kg,
        sensor_error=sensor_error,
        sensor_conditions=sensor_conditions,
        sensor_seed=sensor_seed,
        initial_checkpoint=initial_checkpoint,
    )
    if mode is GamePilotMode.AERODYNAMIC:
        aero = dict(game.get("aerodynamic_control", {}) or {})
        runtime.aerodynamic_config = {
            "ballistic_coefficient_min_kg_m2": float(aero.get("ballistic_coefficient_min_kg_m2", 40.0)),
            "ballistic_coefficient_max_kg_m2": float(aero.get("ballistic_coefficient_max_kg_m2", 200.0)),
            "drag_coefficient": float(aero.get("drag_coefficient", 2.2)),
            "lift_coefficient": float(aero.get("lift_coefficient", 0.45)),
            "lift_area_m2": float(aero.get("lift_area_m2", 20.0)),
        }
    return runtime


def _game_mode(control_mode: str, profile_id: str) -> GamePilotMode:
    if control_mode == "direct_eci" or "direct_eci" in profile_id:
        return GamePilotMode.DIRECT_ECI
    if "aero" in control_mode or "aerodynamic" in profile_id:
        return GamePilotMode.AERODYNAMIC
    if control_mode in {"attitude", "attitude_thrust", "thrust"} or "attitude" in profile_id:
        return GamePilotMode.ATTITUDE_THRUST
    return GamePilotMode.TRANSLATION


def _reference_object_id(agent_cfg: Any) -> str:
    knowledge = dict(getattr(agent_cfg, "knowledge", {}) or {})
    targets = tuple(knowledge.get("targets", ()) or ())
    return str(targets[0]) if targets else "target"


def _max_acceleration_m_s2(game: dict[str, Any], specs: dict[str, Any], mass_kg: float) -> float:
    if game.get("max_acceleration_m_s2") is not None:
        return max(float(game["max_acceleration_m_s2"]), 0.0)
    if game.get("player_max_accel_km_s2") is not None:
        return max(float(game["player_max_accel_km_s2"]) * 1.0e3, 0.0)
    maximum_thrust = resolve_thruster_max_thrust_n_from_specs(specs)
    if maximum_thrust is not None and mass_kg > 0.0:
        return max(float(maximum_thrust) / mass_kg, 0.0)
    return 0.02


def _task_period_s(mode: GamePilotMode, scenario_cfg: Any) -> float:
    simulator = scenario_cfg.simulator
    dynamics = dict(getattr(simulator, "dynamics", {}) or {})
    orbit = dict(dynamics.get("orbit", {}) or {})
    attitude = dict(dynamics.get("attitude", {}) or {})
    if mode is GamePilotMode.ATTITUDE_THRUST:
        value = attitude.get("attitude_substep_s", simulator.dt_s)
    else:
        value = orbit.get("orbit_substep_s", simulator.dt_s)
    return max(float(value or simulator.dt_s), 1.0e-9)


def _stack_task_period_s(stack_id: str, configured_period_s: float | None, _scenario_cfg: Any) -> float:
    """Resolve an onboard cadence without coupling it to physics integration steps."""

    if configured_period_s is not None:
        return max(float(configured_period_s), 1.0e-9)
    defaults_s = {
        "fsw.passive": 1.0,
        "fsw.attitude_reference": 0.1,
        "fsw.orbit_reference": 1.0,
        "fsw.rpo_reference": 1.0,
        "fsw.low_thrust_reference": 1.0,
    }
    return defaults_s.get(stack_id, 1.0)


def _health_config(
    params: dict[str, Any],
    *,
    default_fallbacks: tuple[tuple[str, str], ...] = (),
    isolate_on_saturation: bool = True,
) -> HealthManagerConfig:
    configured = tuple(
        (str(primary), str(backup)) for primary, backup in dict(params.get("actuator_fallbacks", {}) or {}).items()
    )
    fallbacks = configured or default_fallbacks
    return HealthManagerConfig(
        rejection_limit=int(params.get("fdir_rejection_limit", 3)),
        saturation_limit=int(params.get("fdir_saturation_limit", 5)),
        clear_dwell_s=float(params.get("fdir_clear_dwell_s", 2.0)),
        isolate_on_saturation=bool(params.get("fdir_isolate_on_saturation", isolate_on_saturation)),
        actuator_fallbacks=fallbacks,
    )
