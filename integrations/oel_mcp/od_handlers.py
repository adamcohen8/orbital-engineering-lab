"""Transport-neutral handlers for the typed Pro orbit-determination tools.

The handlers are intentionally thin adapters.  They validate closed JSON
packets, enforce MCP path/resource policy, and call the existing deterministic
estimators with explicit named arguments.  They do not import candidate code,
accept arbitrary solver kwargs, pass truth payloads to solvers, or expose them
through inspection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from integrations.oel_mcp.contracts import MAX_MANIFEST_BYTES, ToolContract, legacy_local_tool_id
from integrations.oel_mcp.execution import (
    MCPExecutionCancelled,
    cancellation_callback,
    complete_manifest,
    manifest_base,
    write_execution_manifest,
)
from integrations.oel_mcp.od_registry import (
    OD_TOOL_IDS,
)
from integrations.oel_mcp.public_handlers import inspect_od_json_metadata, is_sensitive_od_key
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.estimation.ground_station_od import solve_ground_station_measurement_od
from sim.estimation.ogp_od import fit_ogp_mean_elements_to_observations
from sim.estimation.relative_od import make_relative_od_arc, solve_batch_relative_od
from sim.estimation.sequential_od import (
    SequentialODConfig,
    SequentialProcessNoise,
    run_sequential_orbit_od,
)
from sim.observations import observation_packet_from_dict
from sim.security.output_writes import bind_output_writer

MAX_OD_INPUT_BYTES = MAX_MANIFEST_BYTES
MAX_OD_OBSERVATIONS = 2_000
MAX_SEQUENTIAL_OBSERVATIONS = 500
MAX_RELATIVE_OBSERVATIONS = 500
MAX_OD_ARC_S = 604_800.0
MAX_INTEGRATION_STEPS = 20_000
MAX_SOLVER_WORK_UNITS = 2_000_000
_GROUND_ROW_KEYS = frozenset(
    {
        "measurement_id",
        "object_id",
        "station_id",
        "time_jd_utc",
        "jd_utc",
        "measurement_type",
        "components",
        "vector",
        "sigma",
        "station_metadata",
        "station",
        "azimuth_deg",
        "elevation_deg",
        "range_km",
        "range_rate_km_s",
        "angle_sigma_deg",
        "range_sigma_km",
        "range_rate_sigma_km_s",
        "covariance",
        "covariance_matrix",
        "uncertainty",
        "arc_id",
    }
)
_STATION_KEYS = frozenset({"id", "lat_deg", "lon_deg", "alt_km"})
_OGP_PACKET_KEYS = frozenset(
    {"schema_version", "kind", "measurement_contract", "template_tle", "observations", "fit_options"}
)
_OGP_TLE_KEYS = frozenset({"line1", "line2"})
_OGP_ROW_KEYS = frozenset({"jd_utc", "position_teme_km", "velocity_teme_km_s"})
_OGP_OPTION_KEYS = frozenset(
    {
        "position_sigma_km",
        "velocity_sigma_km_s",
        "covariance_blocks",
        "fit_until_jd_utc",
        "parameterization",
        "estimate_bstar",
        "bstar_prior_sigma",
        "estimate_epoch_offset",
        "epoch_offset_prior_sigma_s",
        "max_nfev",
        "robust_loss",
        "robust_f_scale",
        "sigma_clip_threshold",
        "propagation_backend",
    }
)
_RELATIVE_PACKET_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "times_s",
        "measurements_ric",
        "measurement_model",
        "noise_diag",
        "covariance_blocks",
        "measurement_origin",
        "fit_duration_s",
        "holdout_duration_s",
        "mean_motion_rad_s",
        "chief_initial_state_eci_km_s",
        "integration_substep_s",
        "transition_evaluation_method",
        "numeric_backend",
    }
)


class ODOperations:
    """Explicit dispatch and solver adapters for OD MCP contracts."""

    def run_ground_sensor_od(
        self,
        handler: Any,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Any | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source, document, source_bytes = _authorized_json_document(
                handler, arguments["measurements_path"]
            )
            rows = _ground_measurements(document)
            if len(rows) < 2 or len(rows) > MAX_OD_OBSERVATIONS:
                raise ValueError("Ground-sensor OD requires 2 to 2,000 bounded measurements.")
            _validate_ground_resource(rows, arguments)
            output = _new_output(handler, arguments["output_dir"])
            data_label = _data_label(arguments)
            resource_plan = _resource_plan(
                operation="ground_sensor_od",
                observation_count=len(rows),
                arc_duration_s=_ground_arc_duration(rows),
                max_nfev=_bounded_int(arguments.get("max_nfev", 24), 1, 24, "max_nfev"),
                integration_substep_s=_ground_step_s(rows, arguments),
            )
            handler._enforce_approval_resource_ceiling(resource_plan)
            with _od_output_scope(handler, output) as writer:
                manifest, manifest_path = _start_manifest(
                    contract,
                    arguments,
                    output=output,
                    source_bytes=source_bytes,
                    resource_plan=resource_plan,
                    data_label=data_label,
                    writer=writer,
                )
                check = cancellation_callback(cancel_event)
                try:
                    check()
                    _progress(progress, 1, 2, "Loaded and bounded ground-sensor measurements.")
                    dynamics = str(arguments.get("dynamics_model", "two_body"))
                    raw = solve_ground_station_measurement_od(
                        rows,
                        object_id=str(arguments["object_id"]),
                        output_dir=output,
                        initial_state_eci_km_s=_optional_state(arguments.get("initial_state_eci_km_s"), "initial_state_eci_km_s"),
                        initial_state_source=_optional_text(arguments.get("initial_state_source")),
                        fit_duration_s=_optional_nonnegative(arguments.get("fit_duration_s"), "fit_duration_s"),
                        holdout_duration_s=_optional_nonnegative(arguments.get("holdout_duration_s"), "holdout_duration_s"),
                        dt_s=_optional_positive(arguments.get("dt_s"), "dt_s"),
                        dynamics_model="two_body",
                        j2=dynamics == "j2",
                        max_nfev=resource_plan["max_nfev"],
                        robust_loss=str(arguments.get("robust_loss", "linear")),
                        robust_f_scale=_positive_finite(arguments.get("robust_f_scale", 1.0), "robust_f_scale"),
                        sigma_clip_threshold=_optional_nonnegative(
                            arguments.get("sigma_clip_threshold"), "sigma_clip_threshold"
                        ),
                        exclude_station_ids=tuple(str(item) for item in arguments.get("exclude_station_ids", []) or []),
                        holdout_station_ids=tuple(str(item) for item in arguments.get("holdout_station_ids", []) or []),
                        scenario_name=str(arguments.get("scenario_name", "ground_station_sensor_od")),
                        numeric_backend=str(arguments.get("numeric_backend", "python")),
                    )
                    check()
                    return _finish_execution(
                        handler,
                        contract,
                        arguments,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        raw=raw,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id=str(arguments["object_id"]),
                        check=check,
                        progress=progress,
                        source_kind="ground-sensor",
                        deferred_capabilities=(),
                    )
                except MCPExecutionCancelled:
                    return _cancel_execution(
                        contract=contract,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id=str(arguments["object_id"]),
                        method="ground_station_sensor_dynamics_least_squares",
                    )
                except Exception as exc:
                    complete_manifest(manifest, status="failed", error_type=type(exc).__name__)
                    write_execution_manifest(output, manifest, writer=writer)
                    raise

        return _envelope(
            handler,
            contract,
            arguments,
            operation,
            outcome_status=_od_outcome_status,
        )

    def run_ogp_od(
        self,
        handler: Any,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Any | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source, document, source_bytes = _authorized_json_document(
                handler, arguments["observations_path"]
            )
            payload = _ogp_packet(document)
            rows = payload["observations"]
            if len(rows) < 2 or len(rows) > MAX_OD_OBSERVATIONS:
                raise ValueError("OGP OD requires 2 to 2,000 bounded TEME PV observations.")
            times = np.asarray([row["jd_utc"] for row in rows], dtype=float)
            arc_duration = float((times[-1] - times[0]) * 86400.0)
            _validate_arc_duration(arc_duration, "OGP")
            options = dict(payload.get("fit_options", {}) or {})
            argument_options = _ogp_argument_overrides(arguments)
            for key, value in argument_options.items():
                if key in options and not _same_json_value(options[key], value):
                    raise ValueError(f"OGP fit_options {key} conflicts with the MCP argument.")
            options.update(argument_options)
            options = _validate_ogp_options(options)
            tle = dict(payload["template_tle"])
            template = parse_tle_lines(str(tle["line1"]), str(tle["line2"]))
            _validate_ogp_template_epoch(times, float(template.epoch_jd_utc))
            output = _new_output(handler, arguments["output_dir"])
            data_label = _data_label(arguments)
            resource_plan = _resource_plan(
                operation="ogp_od",
                observation_count=len(rows),
                arc_duration_s=arc_duration,
                max_nfev=_bounded_int(options.get("max_nfev", 24), 1, 24, "max_nfev"),
                solver_multiplier=8,
            )
            handler._enforce_approval_resource_ceiling(resource_plan)
            with _od_output_scope(handler, output) as writer:
                manifest, manifest_path = _start_manifest(
                    contract,
                    arguments,
                    output=output,
                    source_bytes=source_bytes,
                    resource_plan=resource_plan,
                    data_label=data_label,
                    writer=writer,
                )
                check = cancellation_callback(cancel_event)
                try:
                    check()
                    _progress(progress, 1, 2, "Loaded and bounded native TEME PV observations.")
                    raw = fit_ogp_mean_elements_to_observations(
                        times_jd_utc=times,
                        positions_teme_km=np.asarray([row["position_teme_km"] for row in rows], dtype=float),
                        velocities_teme_km_s=np.asarray([row["velocity_teme_km_s"] for row in rows], dtype=float),
                        template_elements=template,
                        output_dir=output,
                        position_sigma_km=options["position_sigma_km"],
                        velocity_sigma_km_s=options["velocity_sigma_km_s"],
                        covariance_blocks=options.get("covariance_blocks"),
                        fit_until_jd_utc=options.get("fit_until_jd_utc"),
                        parameterization=options["parameterization"],
                        estimate_bstar=options["estimate_bstar"],
                        bstar_prior_sigma=options.get("bstar_prior_sigma"),
                        estimate_epoch_offset=options["estimate_epoch_offset"],
                        epoch_offset_prior_sigma_s=options.get("epoch_offset_prior_sigma_s"),
                        max_nfev=resource_plan["max_nfev"],
                        robust_loss=options["robust_loss"],
                        robust_f_scale=options["robust_f_scale"],
                        sigma_clip_threshold=options.get("sigma_clip_threshold"),
                        validation_reference=None,
                        propagation_backend=options["propagation_backend"],
                    )
                    check()
                    return _finish_execution(
                        handler,
                        contract,
                        arguments,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        raw=raw,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id="",
                        check=check,
                        progress=progress,
                        source_kind="ogp-teme-pv",
                        deferred_capabilities=(
                            "OGP OD support is bounded local analysis and is not operationally qualified.",
                            "This adapter does not publish an operational TLE or precise-orbit product.",
                        ),
                    )
                except MCPExecutionCancelled:
                    return _cancel_execution(
                        contract=contract,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id="",
                        method="ogp_mean_elements_least_squares",
                    )
                except Exception as exc:
                    complete_manifest(manifest, status="failed", error_type=type(exc).__name__)
                    write_execution_manifest(output, manifest, writer=writer)
                    raise

        return _envelope(
            handler,
            contract,
            arguments,
            operation,
            outcome_status=_od_outcome_status,
        )

    def run_sequential_od(
        self,
        handler: Any,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Any | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source, document, source_bytes = _authorized_json_document(
                handler, arguments["observation_packet_path"]
            )
            _reject_truth_keys(document)
            _validate_observation_packet_version(document)
            packet = observation_packet_from_dict(document)
            rows = list(packet.observations)
            if len(rows) < 3 or len(rows) > MAX_SEQUENTIAL_OBSERVATIONS:
                raise ValueError("Sequential OD requires 3 to 500 bounded observations.")
            times = _strict_times([row.get("time_s") for row in rows], "sequential OD")
            arc_duration = float(times[-1] - times[0])
            _validate_arc_duration(arc_duration, "Sequential OD")
            initial = _finite_state(arguments["initial_state_eci_km_s"], "initial_state_eci_km_s")
            covariance = _finite_covariance(arguments["initial_covariance"], "initial_covariance")
            estimator = str(arguments.get("estimator", "ekf"))
            process_model = str(arguments.get("process_noise_model", "white_acceleration"))
            spectral = arguments.get("process_noise_spectral_density")
            if spectral is None:
                spectral = (1.0e-14, 1.0e-14, 1.0e-14) if process_model == "white_acceleration" else (0.0,) * 6
            spectral_values = _finite_vector(spectral, "process_noise_spectral_density")
            expected_noise = 3 if process_model == "white_acceleration" else 6
            if spectral_values.size != expected_noise or np.any(spectral_values < 0.0):
                raise ValueError(f"{process_model} process noise requires {expected_noise} finite nonnegative values.")
            integration_substep = _positive_finite(
                arguments.get("integration_substep_s", 10.0), "integration_substep_s"
            )
            _validate_integration_budget(arc_duration, integration_substep, "sequential OD")
            cfg = SequentialODConfig(
                estimator=estimator,
                process_noise=SequentialProcessNoise(
                    model=process_model,
                    spectral_density=tuple(float(item) for item in spectral_values),
                ),
                innovation_gate_probability=_probability(
                    arguments.get("innovation_gate_probability", 0.999), "innovation_gate_probability"
                ),
                integration_substep_s=integration_substep,
                checkpoint_interval=_bounded_int(
                    arguments.get("checkpoint_interval", 25), 1, 1000, "checkpoint_interval"
                ),
            )
            output = _new_output(handler, arguments["output_dir"])
            data_label = _data_label(arguments)
            resource_plan = _resource_plan(
                operation="sequential_od",
                observation_count=len(rows),
                arc_duration_s=arc_duration,
                max_nfev=1,
                    max_observations=MAX_SEQUENTIAL_OBSERVATIONS,
                integration_substep_s=integration_substep,
                integration_steps=int(math.ceil(arc_duration / integration_substep)),
                solver_multiplier=8,
            )
            handler._enforce_approval_resource_ceiling(resource_plan)
            with _od_output_scope(handler, output) as writer:
                manifest, manifest_path = _start_manifest(
                    contract,
                    arguments,
                    output=output,
                    source_bytes=source_bytes,
                    resource_plan=resource_plan,
                    data_label=data_label,
                    writer=writer,
                )
                check = cancellation_callback(cancel_event)
                try:
                    check()
                    _progress(progress, 1, 2, "Loaded and bounded the ECI observation packet.")
                    raw = run_sequential_orbit_od(
                        packet,
                        initial_state_eci_km_s=initial,
                        initial_covariance=covariance,
                        config=cfg,
                        truth_states_eci_km_s=None,
                        output_dir=output,
                        checkpoint_path=None,
                        resume=False,
                        compare_batch=False,
                    )
                    check()
                    return _finish_execution(
                        handler,
                        contract,
                        arguments,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        raw=raw,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id=str(packet.to_dict().get("object_id", "")),
                        check=check,
                        progress=progress,
                        source_kind="sequential-eci",
                        deferred_capabilities=(),
                        method="sequential_orbit_determination",
                    )
                except MCPExecutionCancelled:
                    return _cancel_execution(
                        contract=contract,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id=str(packet.to_dict().get("object_id", "")),
                        method="sequential_orbit_determination",
                    )
                except Exception as exc:
                    complete_manifest(manifest, status="failed", error_type=type(exc).__name__)
                    write_execution_manifest(output, manifest, writer=writer)
                    raise

        return _envelope(
            handler,
            contract,
            arguments,
            operation,
            outcome_status=_od_outcome_status,
        )

    def run_relative_od(
        self,
        handler: Any,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Any | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source, document, source_bytes = _authorized_json_document(
                handler, arguments["observations_path"]
            )
            payload = _relative_packet(document)
            times = _strict_times(payload["times_s"], "relative OD")
            if len(times) < 2 or len(times) > MAX_RELATIVE_OBSERVATIONS:
                raise ValueError("Relative OD requires 2 to 500 bounded observations.")
            arc_duration = float(times[-1] - times[0])
            _validate_arc_duration(arc_duration, "Relative OD")
            model = str(arguments["dynamics_model"])
            measurement_model = str(arguments["measurement_model"])
            measurement_origin = str(
                _relative_option(arguments, payload, document, "measurement_origin", default="chief")
            )
            noise_diag = _finite_vector(payload["noise_diag"], "noise_diag")
            measurements = _finite_matrix(payload["measurements_ric"], "measurements_ric")
            if measurements.shape[0] != len(times):
                raise ValueError("relative measurements_ric must have one row per time.")
            covariance_blocks = payload.get("covariance_blocks")
            if covariance_blocks is not None:
                covariance_blocks = _finite_array(covariance_blocks, "covariance_blocks")
            arc = make_relative_od_arc(
                times_s=times,
                measurements=measurements,
                measurement_model=measurement_model,
                noise_diag=noise_diag,
                covariance_blocks=covariance_blocks,
                measurement_origin=measurement_origin,
            )
            chief = _relative_option(arguments, payload, document, "chief_initial_state_eci_km_s")
            chief_state = None if chief is None else _finite_state(chief, "chief_initial_state_eci_km_s")
            mean_motion = _relative_option(arguments, payload, document, "mean_motion_rad_s")
            mean_motion_value = None if mean_motion is None else _positive_finite(mean_motion, "mean_motion_rad_s")
            if model in {"hcw", "ss_j2"} and mean_motion_value is None:
                raise ValueError("mean_motion_rad_s is required for HCW and SS-J2 relative OD.")
            if model in {"ss_j2", "th", "ya"} and chief_state is None:
                raise ValueError("chief_initial_state_eci_km_s is required for this relative OD model.")
            if "measurement_model" in document and str(document["measurement_model"]) != measurement_model:
                raise ValueError("relative packet measurement_model conflicts with the MCP argument.")
            if "measurement_origin" in document and str(document["measurement_origin"]) != measurement_origin:
                raise ValueError("relative packet measurement_origin conflicts with the MCP argument.")
            # ``_relative_packet`` keeps the optional field so that the
            # normalized packet is self-describing.  Its value is ``None``
            # when omitted, however, and ``dict.get(..., default)`` does not
            # apply the default in that case.  Resolve the three sources
            # explicitly so the documented 10 s bound is actually used.
            requested_substep = _relative_option(
                arguments,
                payload,
                document,
                "integration_substep_s",
                default=10.0,
            )
            integration_substep = _positive_finite(requested_substep, "integration_substep_s")
            transition_method = str(
                _relative_option(
                    arguments,
                    payload,
                    document,
                    "transition_evaluation_method",
                    default="direct",
                )
            )
            if transition_method not in {"direct", "sequential_composition"}:
                raise ValueError("transition_evaluation_method must be direct or sequential_composition.")
            relative_step_count = _relative_integration_steps(
                times,
                integration_substep,
                model=model,
                transition_method=transition_method,
            )
            output = _new_output(handler, arguments["output_dir"])
            data_label = _data_label(arguments)
            resource_plan = _resource_plan(
                operation="relative_od",
                observation_count=len(times),
                arc_duration_s=arc_duration,
                max_nfev=_bounded_int(arguments.get("max_nfev", 24), 1, 24, "max_nfev"),
                max_observations=MAX_RELATIVE_OBSERVATIONS,
                integration_substep_s=integration_substep,
                integration_steps=relative_step_count,
                solver_multiplier=8,
            )
            handler._enforce_approval_resource_ceiling(resource_plan)
            with _od_output_scope(handler, output) as writer:
                manifest, manifest_path = _start_manifest(
                    contract,
                    arguments,
                    output=output,
                    source_bytes=source_bytes,
                    resource_plan=resource_plan,
                    data_label=data_label,
                    writer=writer,
                )
                check = cancellation_callback(cancel_event)
                try:
                    check()
                    _progress(progress, 1, 2, "Loaded and bounded relative RIC observations.")
                    raw = solve_batch_relative_od(
                        arc,
                        dynamics_model=model,
                        initial_guess_ric=_finite_state(arguments["initial_guess_ric"], "initial_guess_ric"),
                        output_dir=output,
                        fit_duration_s=_optional_nonnegative(
                            _relative_option(arguments, payload, document, "fit_duration_s"), "fit_duration_s"
                        ),
                        holdout_duration_s=_optional_nonnegative(
                            _relative_option(arguments, payload, document, "holdout_duration_s"), "holdout_duration_s"
                        ),
                        mean_motion_rad_s=mean_motion_value,
                        chief_initial_state_eci_km_s=chief_state,
                        truth_initial_relative_state_ric=None,
                        max_nfev=resource_plan["max_nfev"],
                        robust_loss=str(arguments.get("robust_loss", "linear")),
                        robust_f_scale=_positive_finite(arguments.get("robust_f_scale", 1.0), "robust_f_scale"),
                        sigma_clip_threshold=_optional_nonnegative(
                            arguments.get("sigma_clip_threshold"), "sigma_clip_threshold"
                        ),
                        integration_substep_s=integration_substep,
                        transition_evaluation_method=transition_method,
                        numeric_backend=str(
                            _relative_option(arguments, payload, document, "numeric_backend", default="python")
                        ),
                    )
                    check()
                    return _finish_execution(
                        handler,
                        contract,
                        arguments,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        raw=raw,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id="",
                        check=check,
                        progress=progress,
                        source_kind="relative-ric-batch",
                        deferred_capabilities=(
                            "Integrated relative OD remains deferred because its current engine requires a truth initial state and emits truth-derived metrics.",
                        ),
                    )
                except MCPExecutionCancelled:
                    return _cancel_execution(
                        contract=contract,
                        output=output,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        writer=writer,
                        resource_plan=resource_plan,
                        data_label=data_label,
                        object_id="",
                        method="batch_relative_orbit_determination",
                    )
                except Exception as exc:
                    complete_manifest(manifest, status="failed", error_type=type(exc).__name__)
                    write_execution_manifest(output, manifest, writer=writer)
                    raise

        return _envelope(
            handler,
            contract,
            arguments,
            operation,
            outcome_status=_od_outcome_status,
        )

    def inspect_od_json(
        self,
        handler: Any,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Any | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        del cancel_event, progress

        def operation() -> dict[str, Any]:
            maximum = _bounded_int(
                arguments.get("max_file_bytes", MAX_OD_INPUT_BYTES), 1, MAX_OD_INPUT_BYTES, "max_file_bytes",
            )
            _, raw = handler.path_policy.read_authorized_bytes(arguments["path"], maximum=maximum)
            return {
                **inspect_od_json_metadata(raw),
                "data_label": _data_label(arguments),
                "entitlement": _entitlement_payload(contract),
            }

        return _envelope(handler, contract, arguments, operation)


OD_HANDLERS = ODOperations()


def call_od_tool(
    handler: Any,
    contract: ToolContract,
    arguments: dict[str, Any],
    *,
    cancel_event: Any | None = None,
    progress: Any | None = None,
) -> dict[str, Any] | None:
    """Dispatch one registered OD contract, returning ``None`` for others."""

    legacy_id = legacy_local_tool_id(contract.tool_id)
    if legacy_id not in OD_TOOL_IDS:
        return None
    method = {
        OD_TOOL_IDS[0]: OD_HANDLERS.run_ground_sensor_od,
        OD_TOOL_IDS[1]: OD_HANDLERS.run_ogp_od,
        OD_TOOL_IDS[2]: OD_HANDLERS.run_sequential_od,
        OD_TOOL_IDS[3]: OD_HANDLERS.run_relative_od,
        OD_TOOL_IDS[4]: OD_HANDLERS.inspect_od_json,
    }[legacy_id]
    return method(
        handler,
        contract,
        arguments,
        cancel_event=cancel_event,
        progress=progress,
    )


def _envelope(
    handler: Any,
    contract: ToolContract,
    arguments: dict[str, Any],
    operation: Any,
    *,
    outcome_status: Any | None = None,
) -> dict[str, Any]:
    return handler._envelope(
        contract=contract,
        arguments=arguments,
        operation=operation,
        outcome_status=outcome_status,
        evidence=_od_evidence,
    )


def _od_outcome_status(result: dict[str, Any]) -> str:
    status = str(result.get("status", "completed"))
    # The frozen MCP envelope has no cancelled status; cancellation remains
    # explicit in the typed result and terminal execution manifest.
    return "partial" if status == "cancelled" else status if status in {"completed", "partial"} else "partial"


def _od_evidence(result: Mapping[str, Any]) -> dict[str, bool]:
    """Separate solver completion from evidence that passed quality gates."""

    complete = str(result.get("status", "")) == "completed"
    quality = result.get("quality_gates")
    if complete and isinstance(quality, Mapping):
        # A completed numerical operation may still need review.  Keep the
        # operation status useful to callers while refusing to mark its
        # evidence complete until the estimator's own gates say it is ready.
        evidence_status = str(quality.get("evidence_status", "")).strip().lower()
        if evidence_status in {"review", "review_required", "failed", "not_ready"}:
            complete = False
    return {"complete": complete, "empty": False, "truncated": False}


def _authorized_json_document(handler: Any, value: str | Path) -> tuple[Path, Any, bytes]:
    """Read an OD JSON input through the retained, descriptor-bound root.

    Returning the logical path separately keeps provenance and operation
    messages stable while ensuring the parser consumes exactly the bytes that
    passed path authorization.  Reopening the visible pathname after
    ``resolve_read`` would permit a concurrent replacement or symlink swap to
    change the input between authorization and use.
    """

    path, raw = handler.path_policy.read_authorized_bytes(
        value,
        maximum=MAX_OD_INPUT_BYTES,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OD input requires a UTF-8 JSON document.") from exc
    if not isinstance(value, (dict, list)):
        raise ValueError("OD input must be a JSON object or array.")
    _assert_json_finite(value)
    return path, value, raw


def _new_output(handler: Any, value: str | Path) -> Path:
    return handler.path_policy.resolve_write(value)


@contextmanager
def _od_output_scope(handler: Any, output: Path):
    writer = handler.path_policy.open_secure_writer(
        output,
        create=True,
        require_empty=True,
        active_opid=handler.active_operation_id,
    )
    try:
        with bind_output_writer(writer):
            yield writer
    finally:
        writer.close()


def _start_manifest(
    contract: ToolContract,
    arguments: Mapping[str, Any],
    *,
    output: Path,
    source_bytes: bytes,
    resource_plan: dict[str, Any],
    data_label: str,
    writer: Any,
) -> tuple[dict[str, Any], Path]:
    approval = dict(arguments.get("approval", {}) or {})
    manifest = manifest_base(
        tool_id=contract.tool_id,
        approval_id=str(approval.get("approval_id", "")),
    )
    manifest.update(
        {
            "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "solver_arguments_sha256": _solver_arguments_sha256(arguments),
            "solver_arguments_sha256_semantics": "typed_solver_arguments_without_approval_or_handling_values",
            "resource_plan": dict(resource_plan),
            "execution_authorized_outputs": False,
            "data_label": data_label,
            "hidden_truth_included": False,
        }
    )
    target = write_execution_manifest(output, manifest, writer=writer)
    return manifest, target


def _solver_arguments_sha256(arguments: Mapping[str, Any]) -> str:
    retained = {
        str(key): _json_safe(value)
        for key, value in arguments.items()
        if str(key) not in {"approval", "handling"}
    }
    payload = json.dumps(retained, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finish_execution(
    handler: Any,
    contract: ToolContract,
    arguments: dict[str, Any],
    *,
    output: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    raw: Mapping[str, Any],
    resource_plan: dict[str, Any],
    data_label: str,
    object_id: str,
    check: Any,
    progress: Any | None,
    source_kind: str,
    deferred_capabilities: Sequence[str],
    writer: Any,
    method: str | None = None,
) -> dict[str, Any]:
    check()
    artifacts = _artifact_paths(output)
    status = _raw_status(raw)
    complete_manifest(manifest, status=status, artifacts=artifacts)
    write_execution_manifest(output, manifest, writer=writer)
    _progress(progress, 2, 2, f"Completed {source_kind} orbit determination.")
    return _project_result(
        raw,
        status=status,
        output=output,
        artifacts=artifacts,
        contract=contract,
        manifest_path=manifest_path,
        resource_plan=resource_plan,
        data_label=data_label,
        object_id=object_id,
        deferred_capabilities=deferred_capabilities,
        method=method,
    )


def _cancel_execution(
    *,
    contract: ToolContract,
    output: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    resource_plan: dict[str, Any],
    data_label: str,
    object_id: str,
    method: str,
    writer: Any,
) -> dict[str, Any]:
    artifacts = _artifact_paths(output)
    complete_manifest(
        manifest,
        status="cancelled",
        cancelled=True,
        artifacts=artifacts,
        error_type="MCPExecutionCancelled",
    )
    write_execution_manifest(output, manifest, writer=writer)
    return _project_result(
        {},
        status="cancelled",
        output=output,
        artifacts=artifacts,
        contract=contract,
        manifest_path=manifest_path,
        resource_plan=resource_plan,
        data_label=data_label,
        object_id=object_id,
        deferred_capabilities=(),
        method=method,
    )


def _project_result(
    raw: Mapping[str, Any],
    *,
    status: str,
    output: Path,
    artifacts: Sequence[str],
    contract: ToolContract,
    manifest_path: Path,
    resource_plan: Mapping[str, Any],
    data_label: str,
    object_id: str,
    deferred_capabilities: Sequence[str],
    method: str | None = None,
) -> dict[str, Any]:
    report_path = _report_path(raw, output)
    solver = dict(raw.get("solver", {}) or {})
    solver_summary = {
        key: _json_safe(solver.get(key))
        for key in ("success", "message", "nfev", "initial_cost", "final_cost", "rms_weighted_residual")
        if key in solver
    }
    if "success" in solver_summary:
        solver_summary["success"] = bool(solver_summary["success"])
    quality = _json_safe(dict(raw.get("quality_gates", {}) or {}))
    verdict = _json_safe(dict(raw.get("verdict", {}) or {}))
    non_claims = [str(item) for item in list(raw.get("non_claims", ()) or ())]
    non_claims.extend(
        [
            "This local OD adapter does not authorize spacecraft actions or operational orbit certification.",
            "Covariance and quality gates remain estimator evidence and require separate review before downstream use.",
        ]
    )
    return {
        "status": status,
        "method": str(method or raw.get("method", "orbit_determination")),
        "object_id": str(object_id or raw.get("object_id", "")),
        "observation_count": _nonnegative_int(raw.get("observation_count", raw.get("fit_sample_count", 0))),
        "fit_observation_count": _nonnegative_int(raw.get("fit_observation_count", raw.get("fit_sample_count", 0))),
        "holdout_observation_count": _nonnegative_int(raw.get("holdout_observation_count", raw.get("holdout_sample_count", 0))),
        "quality_gates": quality,
        "solver": solver_summary,
        "verdict": verdict,
        "output_dir": str(output),
        "report_path": str(report_path),
        "artifacts": [str(path) for path in artifacts],
        "resource_plan": _json_safe(dict(resource_plan)),
        "operation_manifest_path": str(manifest_path),
        "execution_authorized_outputs": False,
        "hidden_truth_visible": False,
        "local_only": True,
        "data_label": data_label,
        "entitlement": _entitlement_payload(contract),
        "deferred_capabilities": [str(item) for item in deferred_capabilities],
        "non_claims": list(dict.fromkeys(non_claims)),
    }


def _report_path(raw: Mapping[str, Any], output: Path) -> Path:
    direct = raw.get("report_json_path")
    if direct:
        return Path(str(direct))
    artifacts = raw.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("report_json", "report_json_path"):
            if artifacts.get(key):
                return Path(str(artifacts[key]))
    for name in (
        "ground_station_od_report.json",
        "ground_sensor_od_report.json",
        "ogp_od_report.json",
        "sequential_od_report.json",
        "relative_od_fit_report.json",
    ):
        candidate = output / name
        if candidate.is_file():
            return candidate
    return output / "od_report.json"


def _artifact_paths(output: Path) -> list[str]:
    return sorted(str(path) for path in output.iterdir() if path.is_file() and path.name != "mcp_execution_manifest.json")


def _raw_status(raw: Mapping[str, Any]) -> str:
    solver = raw.get("solver")
    if isinstance(solver, Mapping) and "success" in solver and not bool(solver.get("success")):
        return "partial"
    quality = raw.get("quality_gates")
    if isinstance(quality, Mapping):
        evidence_status = str(quality.get("evidence_status", ""))
        if evidence_status in {"review_required", "failed", "not_ready"}:
            return "partial"
    return "completed"


def _resource_plan(
    *,
    operation: str,
    observation_count: int,
    arc_duration_s: float,
    max_nfev: int,
    max_observations: int = MAX_OD_OBSERVATIONS,
    integration_substep_s: float | None = None,
    integration_steps: int | None = None,
    solver_multiplier: int = 8,
) -> dict[str, Any]:
    plan = {
        "resource_profile": "laptop-safe",
        "operation": operation,
        "observation_count": int(observation_count),
        "max_observations": int(max_observations),
        "arc_duration_s": float(arc_duration_s),
        "max_arc_duration_s": MAX_OD_ARC_S,
        "max_nfev": int(max_nfev),
        "action": "proceed",
        "estimate_only": True,
        "qualification": "planning_estimate_not_measured",
    }
    if integration_substep_s is not None:
        plan["integration_substep_s"] = float(integration_substep_s)
        plan["integration_step_count_upper_bound"] = int(
            integration_steps
            if integration_steps is not None
            else math.ceil(max(arc_duration_s, 0.0) / integration_substep_s)
        )
    step_count = int(plan.get("integration_step_count_upper_bound", 0))
    solver_units = max(int(max_nfev), 1) * max(int(observation_count), 1) * max(int(solver_multiplier), 1)
    propagation_units = max(int(max_nfev), 1) * step_count * max(int(solver_multiplier), 1)
    plan["estimated_solver_work_units"] = int(solver_units)
    plan["estimated_propagation_work_units"] = int(propagation_units)
    plan["estimated_work_units"] = int(solver_units + propagation_units)
    plan["work_budget"] = int(MAX_SOLVER_WORK_UNITS)
    if plan["estimated_work_units"] > MAX_SOLVER_WORK_UNITS:
        raise ValueError(
            f"{operation} resource estimate exceeds the bounded OD work budget of {MAX_SOLVER_WORK_UNITS:,} units."
        )
    return plan


def _progress(progress: Any | None, value: float, total: float, message: str) -> None:
    if progress is not None:
        progress(value, total, message)


def _data_label(arguments: Mapping[str, Any]) -> str:
    handling = dict(arguments.get("handling", {}) or {})
    value = str(handling.get("data_label", "oel_pro_internal") or "oel_pro_internal")
    if value not in {"oel_pro_internal", "public_source_local_analysis"}:
        raise ValueError("handling.data_label is not supported for OD tools.")
    return value


def _entitlement_payload(contract: ToolContract) -> dict[str, Any]:
    return {
        "feature": contract.required_entitlement,
        "required": bool(contract.required_entitlement),
        "execution_authorized": False,
    }


def _ground_measurements(document: Any) -> list[dict[str, Any]]:
    _reject_truth_keys(document)
    if isinstance(document, list):
        rows = document
    elif isinstance(document, Mapping):
        allowed = {"measurements"}
        _reject_unknown_keys(document, allowed, "ground measurement document")
        rows = document.get("measurements", [])
    else:
        raise ValueError("Ground measurements must be a JSON list or an object with measurements.")
    if not isinstance(rows, list):
        raise ValueError("Ground measurement document measurements must be a list.")
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Ground measurement row {index} must be an object.")
        _reject_unknown_keys(raw, _GROUND_ROW_KEYS, f"ground measurement row {index}")
        for key in ("station_metadata", "station"):
            station = raw.get(key)
            if station is not None:
                if not isinstance(station, Mapping):
                    raise ValueError(f"ground measurement row {index} {key} must be an object.")
                _reject_unknown_keys(station, _STATION_KEYS, f"ground measurement row {index} {key}")
        _assert_json_finite(raw)
        out.append(dict(raw))
    return out


def _ground_arc_duration(rows: Sequence[Mapping[str, Any]]) -> float:
    epochs = []
    for index, row in enumerate(rows):
        value = row.get("time_jd_utc", row.get("jd_utc"))
        if value is None or not _is_finite_number(value):
            raise ValueError(f"ground measurement row {index} epoch must be finite.")
        epochs.append(float(value))
    if any(later < earlier for earlier, later in zip(epochs, epochs[1:])):
        # The deterministic ground solver sorts rows, but MCP resource checks
        # should use the same interval regardless of caller ordering.
        epochs = sorted(epochs)
    return float((max(epochs) - min(epochs)) * 86400.0)


def _validate_ground_resource(rows: Sequence[Mapping[str, Any]], arguments: Mapping[str, Any]) -> None:
    arc_duration = _ground_arc_duration(rows)
    _validate_arc_duration(arc_duration, "Ground-sensor OD")
    fit = _optional_nonnegative(arguments.get("fit_duration_s"), "fit_duration_s") or 0.0
    holdout = _optional_nonnegative(arguments.get("holdout_duration_s"), "holdout_duration_s") or 0.0
    if fit + holdout > MAX_OD_ARC_S:
        raise ValueError("Ground-sensor OD fit plus holdout duration is limited to seven days.")
    dt = _ground_step_s(rows, arguments)
    _validate_integration_budget(max(arc_duration, fit + holdout), dt, "Ground-sensor OD")


def _ground_step_s(rows: Sequence[Mapping[str, Any]], arguments: Mapping[str, Any]) -> float:
    requested = _optional_positive(arguments.get("dt_s"), "dt_s")
    if requested is not None:
        return requested
    epochs = sorted(
        float(row.get("time_jd_utc", row.get("jd_utc")))
        for row in rows
    )
    differences = [later - earlier for earlier, later in zip(epochs, epochs[1:]) if later > earlier]
    minimum = min(differences) * 86400.0 if differences else 10.0
    return float(max(min(minimum, 60.0), 1.0))


def _ogp_packet(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError("OGP observations must use a JSON object packet.")
    _reject_truth_keys(document)
    _reject_unknown_keys(document, _OGP_PACKET_KEYS, "OGP observation packet")
    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ValueError("OGP PV packet must use schema_version=1.")
    if str(document.get("kind", "")) != "oel.ogp_pv_observation_packet":
        raise ValueError("OGP PV packet kind is not supported.")
    if str(document.get("measurement_contract", "")) != "oel_ogp_teme_pv_observations_v1":
        raise ValueError("OGP PV packet requires the native TEME PV measurement contract.")
    tle = document.get("template_tle")
    if not isinstance(tle, Mapping):
        raise ValueError("OGP PV packet requires template_tle line1 and line2.")
    _reject_unknown_keys(tle, _OGP_TLE_KEYS, "OGP template TLE")
    if not str(tle.get("line1", "")).strip() or not str(tle.get("line2", "")).strip():
        raise ValueError("OGP template TLE requires non-empty line1 and line2.")
    observations = document.get("observations")
    if not isinstance(observations, list):
        raise ValueError("OGP PV packet observations must be a list.")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(observations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"OGP observation {index} must be an object.")
        _reject_unknown_keys(raw, _OGP_ROW_KEYS, f"OGP observation {index}")
        _assert_json_finite(raw)
        jd = raw.get("jd_utc")
        pos = _finite_vector(raw.get("position_teme_km"), f"OGP observation {index} position_teme_km")
        vel = _finite_vector(raw.get("velocity_teme_km_s"), f"OGP observation {index} velocity_teme_km_s")
        if pos.size != 3 or vel.size != 3 or not _is_finite_number(jd):
            raise ValueError(f"OGP observation {index} requires finite TEME PV vectors of length three.")
        rows.append({"jd_utc": float(jd), "position_teme_km": pos.tolist(), "velocity_teme_km_s": vel.tolist()})
    if len(rows) < 2:
        raise ValueError("OGP PV packet requires at least two observations.")
    times = _strict_times([row["jd_utc"] for row in rows], "OGP OD")
    rows = [dict(row) for row in rows]
    options = document.get("fit_options", {})
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError("OGP fit_options must be an object.")
    _reject_unknown_keys(options, _OGP_OPTION_KEYS, "OGP fit_options")
    return {
        "template_tle": {"line1": str(tle["line1"]), "line2": str(tle["line2"])},
        "observations": rows,
        "fit_options": dict(options),
        "times_jd_utc": times,
    }


def _ogp_argument_overrides(arguments: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "position_sigma_km",
        "velocity_sigma_km_s",
        "covariance_blocks",
        "fit_until_jd_utc",
        "parameterization",
        "estimate_bstar",
        "bstar_prior_sigma",
        "estimate_epoch_offset",
        "epoch_offset_prior_sigma_s",
        "max_nfev",
        "robust_loss",
        "robust_f_scale",
        "sigma_clip_threshold",
        "propagation_backend",
    )
    return {key: arguments[key] for key in keys if key in arguments}


def _validate_ogp_options(options: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "position_sigma_km": 0.001,
        "velocity_sigma_km_s": 1.0e-6,
        "covariance_blocks": None,
        "fit_until_jd_utc": None,
        "parameterization": "auto",
        "estimate_bstar": False,
        "bstar_prior_sigma": None,
        "estimate_epoch_offset": False,
        "epoch_offset_prior_sigma_s": None,
        "max_nfev": 24,
        "robust_loss": "linear",
        "robust_f_scale": 1.0,
        "sigma_clip_threshold": None,
        "propagation_backend": "scalar_reference",
    }
    out.update(dict(options))
    out["position_sigma_km"] = _positive_finite(out["position_sigma_km"], "position_sigma_km")
    out["velocity_sigma_km_s"] = _positive_finite(out["velocity_sigma_km_s"], "velocity_sigma_km_s")
    out["max_nfev"] = _bounded_int(out["max_nfev"], 1, 24, "max_nfev")
    out["robust_f_scale"] = _positive_finite(out["robust_f_scale"], "robust_f_scale")
    out["fit_until_jd_utc"] = _optional_finite(out["fit_until_jd_utc"], "fit_until_jd_utc")
    out["sigma_clip_threshold"] = _optional_nonnegative(out["sigma_clip_threshold"], "sigma_clip_threshold")
    out["parameterization"] = str(out["parameterization"])
    if out["parameterization"] not in {"auto", "classical_delta", "equinoctial_nonsingular"}:
        raise ValueError("OGP parameterization is not supported.")
    out["robust_loss"] = str(out["robust_loss"])
    if out["robust_loss"] not in {"linear", "soft_l1", "huber"}:
        raise ValueError("OGP robust_loss is not supported.")
    out["propagation_backend"] = str(out["propagation_backend"])
    if out["propagation_backend"] not in {"scalar_reference", "accelerated"}:
        raise ValueError("OGP propagation_backend is not supported.")
    for key in ("estimate_bstar", "estimate_epoch_offset"):
        if not isinstance(out[key], bool):
            raise ValueError(f"OGP {key} must be boolean.")
    if out["bstar_prior_sigma"] is not None:
        out["bstar_prior_sigma"] = _positive_finite(out["bstar_prior_sigma"], "bstar_prior_sigma")
    if out["epoch_offset_prior_sigma_s"] is not None:
        out["epoch_offset_prior_sigma_s"] = _positive_finite(
            out["epoch_offset_prior_sigma_s"], "epoch_offset_prior_sigma_s"
        )
    covariance = out.get("covariance_blocks")
    if covariance is not None:
        covariance_array = _finite_array(covariance, "covariance_blocks")
        if covariance_array.shape not in {(6, 6)} and not (
            covariance_array.ndim == 3 and covariance_array.shape[1:] == (6, 6)
        ):
            raise ValueError("OGP covariance_blocks must have shape (6, 6) or (N, 6, 6).")
        out["covariance_blocks"] = covariance_array.tolist()
    return out


def _relative_packet(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError("Relative observations must use a JSON object packet.")
    _reject_truth_keys(document)
    _reject_unknown_keys(document, _RELATIVE_PACKET_KEYS, "relative observation packet")
    schema_version = document.get("schema_version")
    if "schema_version" in document and (
        isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1
    ):
        raise ValueError("Relative observation packet schema_version must be 1.")
    if "kind" in document and str(document.get("kind")) not in {
        "oel.relative_od_observation_packet",
        "oel.relative_observation_packet",
    }:
        raise ValueError("Relative observation packet kind is not supported.")
    times = _finite_vector(document.get("times_s"), "times_s")
    measurements = _finite_matrix(document.get("measurements_ric"), "measurements_ric")
    noise = _finite_vector(document.get("noise_diag"), "noise_diag")
    covariance = document.get("covariance_blocks")
    covariance_array = None if covariance is None else _finite_array(covariance, "covariance_blocks")
    return {
        "times_s": times,
        "measurements_ric": measurements,
        "noise_diag": noise,
        "covariance_blocks": covariance_array,
        "measurement_model": str(document.get("measurement_model", "relative_state")),
        "measurement_origin": str(document.get("measurement_origin", "chief")),
        "fit_duration_s": document.get("fit_duration_s"),
        "holdout_duration_s": document.get("holdout_duration_s"),
        "mean_motion_rad_s": document.get("mean_motion_rad_s"),
        "chief_initial_state_eci_km_s": document.get("chief_initial_state_eci_km_s"),
        "integration_substep_s": document.get("integration_substep_s"),
        "transition_evaluation_method": str(document.get("transition_evaluation_method", "direct")),
        "numeric_backend": str(document.get("numeric_backend", "python")),
    }


def _relative_option(
    arguments: Mapping[str, Any],
    payload: Mapping[str, Any],
    document: Mapping[str, Any],
    key: str,
    *,
    default: Any = None,
) -> Any:
    """Resolve an optional packet/MCP duplicate without silent reinterpretation."""

    argument_value = arguments.get(key)
    packet_present = key in document
    packet_value = payload.get(key) if packet_present else None
    if argument_value is not None and packet_present and not _same_json_value(argument_value, packet_value):
        raise ValueError(f"relative packet {key} conflicts with the MCP argument.")
    if argument_value is not None:
        return argument_value
    if packet_present:
        return packet_value
    return default


def _validate_observation_packet_version(document: Any) -> None:
    if not isinstance(document, Mapping):
        raise ValueError("Sequential observations must use a JSON object packet.")
    packet_version = document.get("packet_version")
    schema_version = document.get("observation_schema_version", 1)
    if isinstance(packet_version, bool) or not isinstance(packet_version, int) or packet_version != 1:
        raise ValueError("Sequential observation packet must use packet_version=1.")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ValueError("Sequential observation packet must use observation_schema_version=1.")


def _same_json_value(left: Any, right: Any) -> bool:
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except (TypeError, ValueError):
        return left == right


def _finite_state(value: Any, field: str) -> np.ndarray:
    array = _finite_vector(value, field)
    if array.size != 6:
        raise ValueError(f"{field} must contain exactly six finite values.")
    return array


def _finite_covariance(value: Any, field: str) -> np.ndarray:
    array = _finite_array(value, field)
    if array.shape != (6, 6):
        raise ValueError(f"{field} must have shape (6, 6).")
    return array


def _finite_vector(value: Any, field: str) -> np.ndarray:
    array = _finite_array(value, field)
    if array.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional numeric array.")
    return array


def _finite_matrix(value: Any, field: str) -> np.ndarray:
    array = _finite_array(value, field)
    if array.ndim != 2:
        raise ValueError(f"{field} must be a two-dimensional numeric array.")
    return array


def _finite_array(value: Any, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only numeric values.") from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain finite numeric values.")
    return array


def _strict_times(values: Any, field: str) -> np.ndarray:
    times = _finite_vector(values, f"{field} epochs")
    if times.size < 1 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{field} epochs must be finite and strictly increasing.")
    return times


def _validate_arc_duration(duration_s: float, label: str) -> None:
    if not np.isfinite(duration_s) or duration_s < 0.0 or duration_s > MAX_OD_ARC_S:
        raise ValueError(f"{label} observation arcs are limited to seven days.")


def _validate_ogp_template_epoch(times_jd_utc: Sequence[float] | np.ndarray, template_epoch_jd_utc: float) -> None:
    offsets_s = np.abs((np.asarray(times_jd_utc, dtype=float) - float(template_epoch_jd_utc)) * 86400.0)
    if not np.all(np.isfinite(offsets_s)) or float(np.max(offsets_s, initial=0.0)) > MAX_OD_ARC_S:
        raise ValueError("OGP observation epochs must be within seven days of the template TLE epoch.")


def _validate_integration_budget(duration_s: float, substep_s: float, label: str) -> None:
    if not np.isfinite(substep_s) or substep_s <= 0.0:
        raise ValueError(f"{label} integration_substep_s must be finite and positive.")
    count = int(math.ceil(max(float(duration_s), 0.0) / float(substep_s)))
    if count > MAX_INTEGRATION_STEPS:
        raise ValueError(f"{label} integration step budget is limited to {MAX_INTEGRATION_STEPS:,} steps.")


def _relative_integration_steps(
    times: Sequence[float] | np.ndarray,
    substep_s: float,
    *,
    model: str,
    transition_method: str,
) -> int:
    epochs = np.asarray(times, dtype=float).reshape(-1)
    if epochs.size == 0:
        return 0
    if model in {"th", "ya"}:
        # The TH/YA cache can integrate every requested epoch from the
        # initial epoch.  ``sequential_composition`` falls back to this direct
        # path outside its narrow validated envelope, so use the conservative
        # bound for both method requests rather than letting a fallback evade
        # the work limit.
        count = int(sum(math.ceil(max(float(epoch - epochs[0]), 0.0) / substep_s) for epoch in epochs))
    else:
        count = int(math.ceil(max(float(epochs[-1] - epochs[0]), 0.0) / substep_s))
    if count > MAX_INTEGRATION_STEPS:
        raise ValueError(
            f"Relative OD integration step budget is limited to {MAX_INTEGRATION_STEPS:,} steps."
        )
    return count


def _optional_state(value: Any, field: str) -> np.ndarray | None:
    return None if value is None else _finite_state(value, field)


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_finite(value: Any, field: str) -> float | None:
    return None if value is None else _positive_or_zero_finite(value, field)


def _optional_nonnegative(value: Any, field: str) -> float | None:
    return None if value is None else _positive_or_zero_finite(value, field)


def _optional_positive(value: Any, field: str) -> float | None:
    return None if value is None else _positive_finite(value, field)


def _positive_or_zero_finite(value: Any, field: str) -> float:
    if not _is_finite_number(value) or float(value) < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative.")
    return float(value)


def _positive_finite(value: Any, field: str) -> float:
    if not _is_finite_number(value) or float(value) <= 0.0:
        raise ValueError(f"{field} must be finite and positive.")
    return float(value)


def _probability(value: Any, field: str) -> float:
    number = _positive_finite(value, field)
    if number >= 1.0:
        raise ValueError(f"{field} must be less than one.")
    return number


def _bounded_int(value: Any, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer.")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return result


def _nonnegative_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(result, 0)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))


def _assert_json_finite(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_json_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numeric values must be finite.")


def _reject_truth_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_od_key(key_text):
                raise ValueError(f"Truth-like input field is not accepted at {path}.{key_text}.")
            _reject_truth_keys(item, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_truth_keys(item, f"{path}[{index}]")


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str] | frozenset[str], context: str) -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{context} contains unsupported fields: {', '.join(unknown)}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "MAX_OD_ARC_S",
    "MAX_OD_INPUT_BYTES",
    "OD_HANDLERS",
    "ODOperations",
    "call_od_tool",
]
