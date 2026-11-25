from __future__ import annotations

import math
import re
from typing import Any

from sim.config.scenario.models import (
    OutputsSection,
)
from sim.config.scenario.primitives import (
    _OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES,
    _OUTPUT_PLOTS_UNSUPPORTED_ALIASES,
    _OUTPUTS_UNSUPPORTED_ALIASES,
    _as_dict,
    _reject_unknown_fields,
    _reject_unsupported_aliases,
)
from sim.numeric_backend import normalize_numeric_backend
from sim.security import ConfigPathPolicy

__all__ = [
    '_parse_outputs_section',
]


_SAFE_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _validate_artifact_id(value: Any, path: str) -> str:
    identifier = str(value or "").strip()
    if identifier in {".", ".."} or _SAFE_ARTIFACT_ID.fullmatch(identifier) is None:
        raise ValueError(
            f"{path} must be a path-safe identifier containing only letters, numbers, '.', '_', or '-', "
            "and must start with a letter or number."
        )
    return identifier


def _validate_orbital_analysis_section(section: dict[str, Any]) -> None:
    coverage_analysis_ids: set[str] = set()
    link_analysis_ids: set[str] = set()
    link_ids: set[str] = set()
    coverage_allowed = {
        "analysis_id", "source_object_id", "sensor_id", "order", "half_angle_deg",
        "quat_body_from_sensor", "max_range_km", "chunk_size", "max_working_memory_bytes",
        "max_cell_time_comparisons", "transition_time_tolerance_s", "transition_max_iterations",
        "max_transition_refinement_evaluations",
        "include_cell_csv", "include_fraction_plot",
        "numeric_backend",
    }
    for index, raw in enumerate(list(section.get("coverage", []) or [])):
        path = f"outputs.orbital_analysis.coverage[{index}]"
        _reject_unknown_fields(raw, path, coverage_allowed)
        if "numeric_backend" in raw:
            backend = normalize_numeric_backend(raw["numeric_backend"], field_name=f"{path}.numeric_backend")
            if backend not in {"python", "rust"}:
                raise ValueError(f"{path}.numeric_backend must be python or rust.")
            raw["numeric_backend"] = backend
        for field_name in ("analysis_id", "source_object_id", "sensor_id", "order", "half_angle_deg"):
            if raw.get(field_name) in (None, ""):
                raise ValueError(f"{path}.{field_name} is required.")
        analysis_id = _validate_artifact_id(raw["analysis_id"], f"{path}.analysis_id")
        if analysis_id in coverage_analysis_ids:
            raise ValueError(f"{path}.analysis_id duplicates coverage analysis {analysis_id!r}.")
        coverage_analysis_ids.add(analysis_id)
        if "include_cell_csv" in raw and not isinstance(raw["include_cell_csv"], bool):
            raise ValueError(f"{path}.include_cell_csv must be a boolean true/false value.")
        if "include_fraction_plot" in raw and not isinstance(raw["include_fraction_plot"], bool):
            raise ValueError(f"{path}.include_fraction_plot must be a boolean true/false value.")
        order = raw["order"]
        if isinstance(order, bool) or not isinstance(order, int) or order not in range(5, 9):
            raise ValueError(f"{path}.order must be an integer from 5 through 8.")
        half_angle = float(raw["half_angle_deg"])
        if not math.isfinite(half_angle) or not 0.0 < half_angle < 90.0:
            raise ValueError(f"{path}.half_angle_deg must be finite and within (0, 90).")
        if raw.get("max_range_km") is not None:
            maximum_range = float(raw["max_range_km"])
            if not math.isfinite(maximum_range) or maximum_range <= 0.0:
                raise ValueError(f"{path}.max_range_km must be positive and finite.")
        quaternion = list(raw.get("quat_body_from_sensor", [1.0, 0.0, 0.0, 0.0]) or [])
        if len(quaternion) != 4 or any(not math.isfinite(float(value)) for value in quaternion):
            raise ValueError(f"{path}.quat_body_from_sensor must contain four finite values.")
        if abs(math.sqrt(sum(float(value) ** 2 for value in quaternion)) - 1.0) > 1.0e-10:
            raise ValueError(f"{path}.quat_body_from_sensor must be normalized within 1e-10.")
        for field_name in (
            "chunk_size",
            "max_working_memory_bytes",
            "max_cell_time_comparisons",
            "max_transition_refinement_evaluations",
        ):
            if field_name in raw:
                value = raw[field_name]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{path}.{field_name} must be a positive integer.")
        refinement = ("transition_time_tolerance_s" in raw, "transition_max_iterations" in raw)
        if refinement[0] != refinement[1]:
            raise ValueError(f"{path} must declare transition refinement tolerance and iterations together.")
        if refinement[0]:
            tolerance = float(raw["transition_time_tolerance_s"])
            iterations = raw["transition_max_iterations"]
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(f"{path}.transition_time_tolerance_s must be positive and finite.")
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
                raise ValueError(f"{path}.transition_max_iterations must be a positive integer.")

    link_allowed = {
        "analysis_id", "link_id", "tx_object_id", "rx_object_id", "tx_ground_station_id",
        "rx_ground_station_id", "tx_terminal", "rx_terminal",
        "carrier_frequency_hz", "tx_power_w", "data_rate_bps", "system_noise_temperature_k",
        "required_eb_n0_db", "tx_line_loss_db", "rx_line_loss_db", "misc_loss_db", "max_range_km",
        "min_fixed_site_elevation_deg", "transition_time_tolerance_s", "transition_max_iterations",
        "include_margin_plot",
        "numeric_backend",
    }
    for index, raw in enumerate(list(section.get("directed_links", []) or [])):
        path = f"outputs.orbital_analysis.directed_links[{index}]"
        _reject_unknown_fields(raw, path, link_allowed)
        if "numeric_backend" in raw:
            backend = normalize_numeric_backend(raw["numeric_backend"], field_name=f"{path}.numeric_backend")
            if backend not in {"python", "rust"}:
                raise ValueError(f"{path}.numeric_backend must be python or rust.")
            raw["numeric_backend"] = backend
        required = (
            "analysis_id", "link_id", "tx_terminal", "rx_terminal",
            "carrier_frequency_hz", "tx_power_w", "data_rate_bps", "system_noise_temperature_k", "required_eb_n0_db",
        )
        for field_name in required:
            if raw.get(field_name) in (None, ""):
                raise ValueError(f"{path}.{field_name} is required.")
        analysis_id = _validate_artifact_id(raw["analysis_id"], f"{path}.analysis_id")
        link_id = _validate_artifact_id(raw["link_id"], f"{path}.link_id")
        if analysis_id in link_analysis_ids:
            raise ValueError(f"{path}.analysis_id duplicates directed-link analysis {analysis_id!r}.")
        if link_id in link_ids:
            raise ValueError(f"{path}.link_id duplicates directed link {link_id!r}.")
        link_analysis_ids.add(analysis_id)
        link_ids.add(link_id)
        if "include_margin_plot" in raw and not isinstance(raw["include_margin_plot"], bool):
            raise ValueError(f"{path}.include_margin_plot must be a boolean true/false value.")
        endpoints: dict[str, tuple[str, str]] = {}
        for endpoint in ("tx", "rx"):
            selectors = [
                ("object", str(raw.get(f"{endpoint}_object_id") or "").strip()),
                ("ground_station", str(raw.get(f"{endpoint}_ground_station_id") or "").strip()),
            ]
            selected = [(kind, identifier) for kind, identifier in selectors if identifier]
            if len(selected) != 1:
                raise ValueError(
                    f"{path} must declare exactly one of {endpoint}_object_id or "
                    f"{endpoint}_ground_station_id."
                )
            endpoints[endpoint] = selected[0]
        if endpoints["tx"] == endpoints["rx"]:
            raise ValueError(f"{path} endpoints must name different assets.")
        if endpoints["tx"][0] == endpoints["rx"][0] == "ground_station":
            raise ValueError(f"{path} does not support ground-station to ground-station links.")
        for field_name in ("carrier_frequency_hz", "tx_power_w", "data_rate_bps", "system_noise_temperature_k"):
            value = float(raw[field_name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{path}.{field_name} must be positive and finite.")
        if not math.isfinite(float(raw["required_eb_n0_db"])):
            raise ValueError(f"{path}.required_eb_n0_db must be finite.")
        for field_name in ("tx_line_loss_db", "rx_line_loss_db", "misc_loss_db"):
            if field_name in raw:
                value = float(raw[field_name])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{path}.{field_name} must be nonnegative and finite.")
        if raw.get("max_range_km") is not None:
            maximum_range = float(raw["max_range_km"])
            if not math.isfinite(maximum_range) or maximum_range <= 0.0:
                raise ValueError(f"{path}.max_range_km must be positive and finite.")
        if raw.get("min_fixed_site_elevation_deg") is not None:
            minimum_elevation = float(raw["min_fixed_site_elevation_deg"])
            if not math.isfinite(minimum_elevation) or not -90.0 <= minimum_elevation <= 90.0:
                raise ValueError(f"{path}.min_fixed_site_elevation_deg must be within [-90, 90].")
        for terminal_name in ("tx_terminal", "rx_terminal"):
            terminal_path = f"{path}.{terminal_name}"
            terminal = _as_dict(raw.get(terminal_name), terminal_path)
            _reject_unknown_fields(
                terminal,
                terminal_path,
                {"terminal_id", "quat_parent_from_terminal", "quat_body_from_terminal", "pattern"},
            )
            if not str(terminal.get("terminal_id") or "").strip():
                raise ValueError(f"{terminal_path}.terminal_id is required.")
            if "quat_parent_from_terminal" in terminal and "quat_body_from_terminal" in terminal:
                raise ValueError(
                    f"{terminal_path} must not declare both quat_parent_from_terminal and "
                    "quat_body_from_terminal."
                )
            quaternion = list(
                terminal.get(
                    "quat_parent_from_terminal",
                    terminal.get("quat_body_from_terminal", [1.0, 0.0, 0.0, 0.0]),
                )
                or []
            )
            if len(quaternion) != 4 or any(not math.isfinite(float(value)) for value in quaternion):
                raise ValueError(f"{terminal_path} terminal quaternion must contain four finite values.")
            if abs(math.sqrt(sum(float(value) ** 2 for value in quaternion)) - 1.0) > 1.0e-10:
                raise ValueError(f"{terminal_path} terminal quaternion must be normalized within 1e-10.")
            pattern_path = f"{terminal_path}.pattern"
            pattern = _as_dict(terminal.get("pattern"), pattern_path)
            _reject_unknown_fields(pattern, pattern_path, {"kind", "gain_dbi", "half_angle_deg"})
            kind = str(pattern.get("kind", "constant") or "constant").lower()
            if kind not in {"constant", "axisymmetric_hard_cone"}:
                raise ValueError(f"{pattern_path}.kind must be constant or axisymmetric_hard_cone.")
            if pattern.get("gain_dbi") is None or not math.isfinite(float(pattern["gain_dbi"])):
                raise ValueError(f"{pattern_path}.gain_dbi must be finite.")
            if kind == "axisymmetric_hard_cone":
                if pattern.get("half_angle_deg") is None:
                    raise ValueError(f"{pattern_path}.half_angle_deg is required for a directional pattern.")
                angle = float(pattern.get("half_angle_deg"))
                if not math.isfinite(angle) or not 0.0 < angle <= 180.0:
                    raise ValueError(f"{pattern_path}.half_angle_deg must be within (0, 180].")
            elif pattern.get("half_angle_deg") is not None:
                raise ValueError(f"{pattern_path}.half_angle_deg is not valid for a constant pattern.")
        refinement = ("transition_time_tolerance_s" in raw, "transition_max_iterations" in raw)
        if refinement[0] != refinement[1]:
            raise ValueError(f"{path} must declare transition refinement tolerance and iterations together.")
        if refinement[0]:
            tolerance = float(raw["transition_time_tolerance_s"])
            iterations = raw["transition_max_iterations"]
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(f"{path}.transition_time_tolerance_s must be positive and finite.")
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
                raise ValueError(f"{path}.transition_max_iterations must be a positive integer.")

def _parse_outputs_section(value: Any, path_policy: ConfigPathPolicy | None = None) -> OutputsSection:
    d = _as_dict(value, "outputs")
    _reject_unsupported_aliases(d, "outputs", _OUTPUTS_UNSUPPORTED_ALIASES)
    _reject_unknown_fields(
        d,
        "outputs",
        {
            "output_dir",
            "mode",
            "stats",
            "plots",
            "animations",
            "monte_carlo",
            "campaign_retention",
            "ai_report",
            "ai_config",
            "review",
            "orbital_analysis",
            "resource_limits",
        },
    )
    plots = _as_dict(d.get("plots"), "outputs.plots")
    animations = _as_dict(d.get("animations"), "outputs.animations")
    stats = _as_dict(d.get("stats"), "outputs.stats")
    monte_carlo_outputs = _as_dict(d.get("monte_carlo"), "outputs.monte_carlo")
    campaign_retention = _as_dict(d.get("campaign_retention"), "outputs.campaign_retention")
    if campaign_retention:
        from sim.execution.campaign_retention import validate_campaign_retention

        validate_campaign_retention(campaign_retention)
    ai_report = _as_dict(d.get("ai_report"), "outputs.ai_report")
    ai_config = _as_dict(d.get("ai_config"), "outputs.ai_config")
    review = _as_dict(d.get("review"), "outputs.review")
    orbital_analysis = _as_dict(d.get("orbital_analysis"), "outputs.orbital_analysis")
    resource_limits = _as_dict(d.get("resource_limits"), "outputs.resource_limits")
    _reject_unsupported_aliases(plots, "outputs.plots", _OUTPUT_PLOTS_UNSUPPORTED_ALIASES)
    _reject_unsupported_aliases(animations, "outputs.animations", _OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES)
    _reject_unknown_fields(
        stats,
        "outputs.stats",
        {"enabled", "print_summary", "save_json", "save_csv", "save_full_log", "save_history_npz", "controller_debug"},
    )
    _reject_unknown_fields(
        plots,
        "outputs.plots",
        {
            "enabled", "figure_ids", "dpi", "style", "preset", "reference_object_id",
            "reference_object_label", "orbital_elements_object_id", "keepout_radius_km",
            "burn_marker_object_ids", "ric_2d_planes", "draw_earth_map", "thrust_direction_body",
        },
    )
    _reject_unknown_fields(
        animations,
        "outputs.animations",
        {
            "enabled", "types", "fps", "style", "frame_stride", "speed_multiple", "draw_earth_map",
            "battlespace_dashboard_attitude_dims_m", "battlespace_dashboard_chaser_object_id",
            "battlespace_dashboard_show_trajectory", "battlespace_dashboard_target_object_id",
            "battlespace_dashboard_thruster_active_threshold_km_s2",
            "attitude_ric_thruster_active_threshold_km_s2", "attitude_ric_thruster_dims_m",
            "attitude_ric_thruster_object_ids", "ric_curv_prism_dims_m", "ric_curv_prism_object_ids",
            "ric_side_by_side_dims_m", "ric_side_by_side_left_object_id", "ric_side_by_side_right_object_id",
            "target_object_id", "target_reference_ric_curv_object_ids",
            "target_reference_ric_curv_3d_show_trajectory", "target_reference_ric_curv_2d_show_trajectory",
            "target_reference_ric_curv_2d_ri_show_trajectory", "target_reference_ric_curv_2d_ic_show_trajectory",
            "target_reference_ric_curv_2d_rc_show_trajectory", "target_reference_ric_curv_2d_planes",
        },
    )
    _reject_unknown_fields(
        monte_carlo_outputs,
        "outputs.monte_carlo",
        {
            "baseline_summary_json", "catastrophic_failure_reasons", "checkpoint_enabled",
            "display_histograms", "display_ops_dashboard", "gates", "require_rocket_insertion",
            "save_aggregate_summary", "save_histograms", "save_iteration_summaries", "save_ops_dashboard",
            "save_raw_runs", "success_termination_reasons",
        },
    )
    _reject_unknown_fields(
        ai_report,
        "outputs.ai_report",
        {
            "enabled", "provider", "model", "endpoint", "api_key_env", "timeout_s", "options", "dry_run",
            "fail_on_error", "data_scope", "include_figure_data", "include_json_appendix", "max_examples",
            "prompt_profile", "prompt_file", "report_mode", "user_questions", "user_questions_file",
            "include_full_config", "fail_on_quality", "pricing", "estimated_output_tokens",
            "output_token_estimate", "max_output_tokens", "max_tokens", "max_prompt_chars", "chars_per_token",
            "max_failure_examples", "generation_config", "generationConfig", "anthropic_version",
        },
    )
    _reject_unknown_fields(
        ai_config,
        "outputs.ai_config",
        {
            "enabled", "provider", "model", "endpoint", "api_key_env", "timeout_s", "validate_timeout_s",
            "options", "dry_run", "repair_attempts",
        },
    )
    if "max_prompt_chars" in ai_report:
        max_prompt_chars = ai_report["max_prompt_chars"]
        if isinstance(max_prompt_chars, bool) or not isinstance(max_prompt_chars, int) or max_prompt_chars <= 0:
            raise ValueError("outputs.ai_report.max_prompt_chars must be a positive integer.")
    _reject_unknown_fields(review, "outputs.review", {"enabled", "detail", "strict"})
    _reject_unknown_fields(
        orbital_analysis,
        "outputs.orbital_analysis",
        {"enabled", "coverage", "directed_links", "access_chains"},
    )
    for field_name in ("coverage", "directed_links", "access_chains"):
        value = orbital_analysis.get(field_name, [])
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError(f"outputs.orbital_analysis.{field_name} must be a list of mappings.")
    _validate_orbital_analysis_section(orbital_analysis)
    if orbital_analysis.get("access_chains"):
        from sim.analysis.access_chains import Link, parse_chain

        catalog = {row["analysis_id"]: Link.from_scenario(row) for row in orbital_analysis.get("directed_links", [])}
        coverage = {row["analysis_id"]: (row["source_object_id"], row["order"])
                    for row in orbital_analysis.get("coverage", [])}
        chain_ids = [parse_chain(row, catalog, coverage).analysis_id for row in orbital_analysis["access_chains"]]
        if len(set(chain_ids)) != len(chain_ids):
            raise ValueError("Access chain analysis IDs must be unique.")
        if orbital_analysis.get("enabled") is not True:
            raise ValueError("Access chains require outputs.orbital_analysis.enabled: true.")
    if stats.get("enabled") is False:
        for field_name in (
            "print_summary", "save_json", "save_csv", "save_full_log", "save_history_npz", "controller_debug"
        ):
            stats[field_name] = False
    _reject_unknown_fields(
        resource_limits,
        "outputs.resource_limits",
        {
            "max_history_memory_mb", "checkpoint_enabled", "hard_min_available_memory_mb", "max_load_per_cpu",
            "min_available_memory_mb", "resource_max_wait_s", "resource_pause_seconds", "resource_profile",
            "throttle_enabled",
        },
    )
    out = OutputsSection(
        output_dir=str(d.get("output_dir", "outputs")),
        mode=str(d.get("mode", "interactive")),
        stats=stats,
        plots=plots,
        animations=animations,
        monte_carlo=monte_carlo_outputs,
        campaign_retention=campaign_retention,
        ai_report=ai_report,
        ai_config=ai_config,
        review=review,
        orbital_analysis=orbital_analysis,
        resource_limits=resource_limits,
    )
    if out.mode not in ("interactive", "save", "both"):
        raise ValueError("outputs.mode must be one of: interactive, save, both.")
    if not out.output_dir.strip():
        raise ValueError("outputs.output_dir must be non-empty.")
    max_history_memory_mb = out.resource_limits.max_history_memory_mb
    if max_history_memory_mb is not None and max_history_memory_mb <= 0:
        raise ValueError("outputs.resource_limits.max_history_memory_mb must be positive when set.")
    if out.review.detail not in {"compact", "standard", "full"}:
        raise ValueError("outputs.review.detail must be one of: compact, standard, full.")
    if path_policy is not None:
        path_policy.resolve_output_dir(out.output_dir, purpose="outputs.output_dir")
    return out
