# ruff: noqa: F401,F403,F405,I001
from .bench_common import *
from .config import *
from .variants import *
from .cases import *
from .criteria import *
from .diagnostics import *
from .engineering import write_minimal_reproducer, write_run_engineering_artifacts

def _dedupe_metrics(metrics: list[ControllerBenchMetric]) -> tuple[ControllerBenchMetric, ...]:
    seen: set[str] = set()
    out: list[ControllerBenchMetric] = []
    for metric in metrics:
        if metric.name in seen:
            continue
        seen.add(metric.name)
        out.append(metric)
    return tuple(out)


def _objective_specifications(
    objective: ControllerBenchObjective,
) -> tuple[tuple[ControllerBenchMetric, ...], tuple[ControllerBenchPassCriterion, ...]]:
    prefix = _slug(objective.name or objective.kind)
    metrics: list[ControllerBenchMetric] = []
    criteria: list[ControllerBenchPassCriterion] = []

    if objective.kind == "attitude_hold":
        if not objective.object_id:
            raise ValueError("attitude_hold objective requires object_id.")
        if objective.desired_quat_bn is not None:
            final_name = _metric_name(prefix, "final_attitude_error_deg")
            rms_name = _metric_name(prefix, "rms_attitude_error_deg")
            metrics.append(
                ControllerBenchMetric(
                    name=final_name,
                    kind="final_attitude_error_deg",
                    object_id=objective.object_id,
                    desired_quat_bn=objective.desired_quat_bn,
                )
            )
            metrics.append(
                ControllerBenchMetric(
                    name=rms_name,
                    kind="rms_attitude_error_deg",
                    object_id=objective.object_id,
                    desired_quat_bn=objective.desired_quat_bn,
                )
            )
            if objective.max_final_attitude_error_deg is not None:
                criteria.append(
                    ControllerBenchPassCriterion(
                        metric=final_name,
                        op="<=",
                        value=objective.max_final_attitude_error_deg,
                    )
                )
            if objective.max_rms_attitude_error_deg is not None:
                criteria.append(
                    ControllerBenchPassCriterion(
                        metric=rms_name,
                        op="<=",
                        value=objective.max_rms_attitude_error_deg,
                    )
                )

        rate_name = _metric_name(prefix, "final_body_rate_norm_rad_s")
        metrics.append(
            ControllerBenchMetric(
                name=rate_name,
                kind="final_body_rate_norm_rad_s",
                object_id=objective.object_id,
            )
        )
        if objective.max_final_body_rate_norm_rad_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=rate_name,
                    op="<=",
                    value=objective.max_final_body_rate_norm_rad_s,
                )
            )

    elif objective.kind == "attitude_slew_settle":
        if not objective.object_id:
            raise ValueError("attitude_slew_settle objective requires object_id.")
        if objective.desired_quat_bn is None:
            raise ValueError("attitude_slew_settle objective requires desired_quat_bn.")
        final_name = _metric_name(prefix, "final_attitude_error_deg")
        rms_name = _metric_name(prefix, "rms_attitude_error_deg")
        peak_name = _metric_name(prefix, "max_attitude_error_deg")
        final_rate_name = _metric_name(prefix, "final_body_rate_norm_rad_s")
        max_rate_name = _metric_name(prefix, "max_body_rate_norm_rad_s")
        settle_name = _metric_name(prefix, "settle_time_s")
        hold_name = _metric_name(prefix, "settled_hold_duration_s")
        torque_name = _metric_name(prefix, "total_torque_norm_n_m_s")
        attitude_metric_kwargs = {
            "object_id": objective.object_id,
            "desired_quat_bn": objective.desired_quat_bn,
        }
        settle_metric_kwargs = {
            **attitude_metric_kwargs,
            "settle_error_deg": objective.settle_error_deg,
            "settle_rate_norm_rad_s": objective.settle_rate_norm_rad_s,
            "settle_hold_s": objective.settle_hold_s,
        }
        metrics.extend(
            [
                ControllerBenchMetric(name=final_name, kind="final_attitude_error_deg", **attitude_metric_kwargs),
                ControllerBenchMetric(name=rms_name, kind="rms_attitude_error_deg", **attitude_metric_kwargs),
                ControllerBenchMetric(name=peak_name, kind="max_attitude_error_deg", **attitude_metric_kwargs),
                ControllerBenchMetric(
                    name=final_rate_name,
                    kind="final_body_rate_norm_rad_s",
                    object_id=objective.object_id,
                ),
                ControllerBenchMetric(
                    name=max_rate_name,
                    kind="max_body_rate_norm_rad_s",
                    object_id=objective.object_id,
                ),
                ControllerBenchMetric(name=settle_name, kind="settle_time_s", **settle_metric_kwargs),
                ControllerBenchMetric(name=hold_name, kind="settled_hold_duration_s", **settle_metric_kwargs),
                ControllerBenchMetric(
                    name=torque_name,
                    kind="total_torque_norm_n_m_s",
                    object_id=objective.object_id,
                ),
            ]
        )
        if objective.max_final_attitude_error_deg is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=final_name,
                    op="<=",
                    value=objective.max_final_attitude_error_deg,
                )
            )
        if objective.max_rms_attitude_error_deg is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=rms_name, op="<=", value=objective.max_rms_attitude_error_deg)
            )
        if objective.max_peak_attitude_error_deg is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=peak_name, op="<=", value=objective.max_peak_attitude_error_deg)
            )
        if objective.max_final_body_rate_norm_rad_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=final_rate_name,
                    op="<=",
                    value=objective.max_final_body_rate_norm_rad_s,
                )
            )
        if objective.max_body_rate_norm_rad_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=max_rate_name, op="<=", value=objective.max_body_rate_norm_rad_s)
            )
        if objective.max_settle_time_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=settle_name, op="<=", value=objective.max_settle_time_s)
            )
        if objective.settle_hold_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=hold_name, op=">=", value=objective.settle_hold_s))
        if objective.max_total_torque_norm_n_m_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=torque_name,
                    op="<=",
                    value=objective.max_total_torque_norm_n_m_s,
                )
            )

    elif objective.kind == "relative_rendezvous":
        if not objective.object_id or not objective.reference_object_id:
            raise ValueError("relative_rendezvous objective requires object_id and reference_object_id.")
        final_range_name = _metric_name(prefix, "final_relative_distance_km")
        rms_range_name = _metric_name(prefix, "rms_relative_distance_km")
        closest_name = _metric_name(prefix, "closest_approach_km")
        final_speed_name = _metric_name(prefix, "final_relative_speed_km_s")
        dv_name = _metric_name(prefix, "total_dv_m_s")
        fuel_name = _metric_name(prefix, "fuel_used_kg")
        metrics.extend(
            [
                ControllerBenchMetric(
                    name=final_range_name,
                    kind="final_relative_distance_km",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=rms_range_name,
                    kind="rms_relative_distance_km",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=closest_name,
                    kind="closest_approach_km",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=final_speed_name,
                    kind="final_relative_speed_km_s",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=dv_name,
                    kind="total_dv_m_s",
                    object_id=objective.object_id,
                ),
                ControllerBenchMetric(
                    name=fuel_name,
                    kind="fuel_used_kg",
                    object_id=objective.object_id,
                ),
            ]
        )
        if objective.keepout_radius_km is not None:
            keepout_name = _metric_name(prefix, "time_inside_keepout_s")
            metrics.append(
                ControllerBenchMetric(
                    name=keepout_name,
                    kind="time_inside_keepout_s",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                    keepout_radius_km=objective.keepout_radius_km,
                )
            )
            if objective.max_time_inside_keepout_s is not None:
                criteria.append(
                    ControllerBenchPassCriterion(
                        metric=keepout_name,
                        op="<=",
                        value=objective.max_time_inside_keepout_s,
                    )
                )
        if objective.max_final_relative_distance_km is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=final_range_name,
                    op="<=",
                    value=objective.max_final_relative_distance_km,
                )
            )
        if objective.max_rms_relative_distance_km is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=rms_range_name,
                    op="<=",
                    value=objective.max_rms_relative_distance_km,
                )
            )
        if objective.max_final_relative_speed_km_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=final_speed_name,
                    op="<=",
                    value=objective.max_final_relative_speed_km_s,
                )
            )
        if objective.max_total_dv_m_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=dv_name, op="<=", value=objective.max_total_dv_m_s))
        if objective.max_fuel_used_kg is not None:
            criteria.append(ControllerBenchPassCriterion(metric=fuel_name, op="<=", value=objective.max_fuel_used_kg))
    elif objective.kind == "terminal_rendezvous_box":
        if not objective.object_id or not objective.reference_object_id:
            raise ValueError("terminal_rendezvous_box objective requires object_id and reference_object_id.")
        final_range_name = _metric_name(prefix, "final_relative_distance_km")
        final_speed_name = _metric_name(prefix, "final_relative_speed_km_s")
        radial_name = _metric_name(prefix, "final_relative_radial_error_km")
        intrack_name = _metric_name(prefix, "final_relative_intrack_error_km")
        crosstrack_name = _metric_name(prefix, "final_relative_crosstrack_error_km")
        radial_rate_name = _metric_name(prefix, "final_relative_radial_rate_km_s")
        intrack_rate_name = _metric_name(prefix, "final_relative_intrack_rate_km_s")
        crosstrack_rate_name = _metric_name(prefix, "final_relative_crosstrack_rate_km_s")
        dv_name = _metric_name(prefix, "total_dv_m_s")
        fuel_name = _metric_name(prefix, "fuel_used_kg")
        rel_metric_kwargs = {
            "object_id": objective.object_id,
            "reference_object_id": objective.reference_object_id,
        }
        metrics.extend(
            [
                ControllerBenchMetric(name=final_range_name, kind="final_relative_distance_km", **rel_metric_kwargs),
                ControllerBenchMetric(name=final_speed_name, kind="final_relative_speed_km_s", **rel_metric_kwargs),
                ControllerBenchMetric(name=radial_name, kind="final_relative_radial_error_km", **rel_metric_kwargs),
                ControllerBenchMetric(name=intrack_name, kind="final_relative_intrack_error_km", **rel_metric_kwargs),
                ControllerBenchMetric(
                    name=crosstrack_name, kind="final_relative_crosstrack_error_km", **rel_metric_kwargs
                ),
                ControllerBenchMetric(
                    name=radial_rate_name, kind="final_relative_radial_rate_km_s", **rel_metric_kwargs
                ),
                ControllerBenchMetric(
                    name=intrack_rate_name, kind="final_relative_intrack_rate_km_s", **rel_metric_kwargs
                ),
                ControllerBenchMetric(
                    name=crosstrack_rate_name,
                    kind="final_relative_crosstrack_rate_km_s",
                    **rel_metric_kwargs,
                ),
                ControllerBenchMetric(name=dv_name, kind="total_dv_m_s", object_id=objective.object_id),
                ControllerBenchMetric(name=fuel_name, kind="fuel_used_kg", object_id=objective.object_id),
            ]
        )
        max_final_range_km = objective.max_final_range_km
        if max_final_range_km is None:
            max_final_range_km = objective.max_final_relative_distance_km
        max_final_speed_km_s = objective.max_final_speed_km_s
        if max_final_speed_km_s is None:
            max_final_speed_km_s = objective.max_final_relative_speed_km_s
        if max_final_range_km is not None:
            criteria.append(ControllerBenchPassCriterion(metric=final_range_name, op="<=", value=max_final_range_km))
        if max_final_speed_km_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=final_speed_name, op="<=", value=max_final_speed_km_s))
        if objective.max_radial_error_km is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=radial_name, op="<=", value=objective.max_radial_error_km)
            )
        if objective.max_intrack_error_km is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=intrack_name, op="<=", value=objective.max_intrack_error_km)
            )
        if objective.max_crosstrack_error_km is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=crosstrack_name, op="<=", value=objective.max_crosstrack_error_km)
            )
        if objective.max_radial_rate_km_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=radial_rate_name, op="<=", value=objective.max_radial_rate_km_s)
            )
        if objective.max_intrack_rate_km_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=intrack_rate_name, op="<=", value=objective.max_intrack_rate_km_s)
            )
        if objective.max_crosstrack_rate_km_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=crosstrack_rate_name,
                    op="<=",
                    value=objective.max_crosstrack_rate_km_s,
                )
            )
        if objective.max_total_dv_m_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=dv_name, op="<=", value=objective.max_total_dv_m_s))
        if objective.max_fuel_used_kg is not None:
            criteria.append(ControllerBenchPassCriterion(metric=fuel_name, op="<=", value=objective.max_fuel_used_kg))
    elif objective.kind == "nonzero_ric_hold":
        if not objective.object_id or not objective.reference_object_id:
            raise ValueError("nonzero_ric_hold objective requires object_id and reference_object_id.")
        if objective.target_relative_state_ric_m is None:
            raise ValueError("nonzero_ric_hold objective requires target_relative_state_ric_m.")
        if objective.max_final_position_error_m is None and objective.max_final_velocity_error_m_s is None:
            raise ValueError(
                "nonzero_ric_hold objective requires max_final_position_error_m or "
                "max_final_velocity_error_m_s so signed-axis response is a qualification gate."
            )
        position_name = _metric_name(prefix, "final_relative_position_error_m")
        velocity_name = _metric_name(prefix, "final_relative_velocity_error_m_s")
        kwargs = {
            "object_id": objective.object_id,
            "reference_object_id": objective.reference_object_id,
            "target_relative_state_ric_m": objective.target_relative_state_ric_m,
        }
        metrics.extend(
            [
                ControllerBenchMetric(name=position_name, kind="final_relative_position_error_m", **kwargs),
                ControllerBenchMetric(name=velocity_name, kind="final_relative_velocity_error_m_s", **kwargs),
            ]
        )
        if objective.max_final_position_error_m is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=position_name, op="<=", value=objective.max_final_position_error_m)
            )
        if objective.max_final_velocity_error_m_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=velocity_name,
                    op="<=",
                    value=objective.max_final_velocity_error_m_s,
                )
            )

    elif objective.kind == "waypoint_sequence":
        if not objective.object_id or not objective.reference_object_id:
            raise ValueError("waypoint_sequence objective requires object_id and reference_object_id.")
        if not objective.waypoints_ric_m:
            raise ValueError("waypoint_sequence objective requires waypoints_ric_m.")
        if objective.waypoint_tolerance_m is None or objective.waypoint_velocity_tolerance_m_s is None:
            raise ValueError(
                "waypoint_sequence objective requires waypoint_tolerance_m and waypoint_velocity_tolerance_m_s."
            )
        completed_name = _metric_name(prefix, "waypoint_sequence_completed")
        count_name = _metric_name(prefix, "waypoint_sequence_completed_count")
        kwargs = {
            "object_id": objective.object_id,
            "reference_object_id": objective.reference_object_id,
            "waypoints_ric_m": objective.waypoints_ric_m,
            "waypoint_tolerance_m": objective.waypoint_tolerance_m,
            "waypoint_velocity_tolerance_m_s": objective.waypoint_velocity_tolerance_m_s,
        }
        metrics.extend(
            [
                ControllerBenchMetric(name=completed_name, kind="waypoint_sequence_completed", **kwargs),
                ControllerBenchMetric(name=count_name, kind="waypoint_sequence_completed_count", **kwargs),
            ]
        )
        criteria.append(ControllerBenchPassCriterion(metric=completed_name, op="==", value=True))

    elif objective.kind in {"safe_rpo_approach", "passive_retreat"}:
        if not objective.object_id or not objective.reference_object_id:
            raise ValueError(f"{objective.kind} objective requires object_id and reference_object_id.")
        kwargs = {"object_id": objective.object_id, "reference_object_id": objective.reference_object_id}
        final_range_name = _metric_name(prefix, "final_relative_distance_km")
        min_range_name = _metric_name(prefix, "min_relative_distance_km")
        max_range_name = _metric_name(prefix, "max_relative_distance_km")
        final_speed_name = _metric_name(prefix, "final_relative_speed_km_s")
        dv_name = _metric_name(prefix, "total_dv_m_s")
        metrics.extend(
            [
                ControllerBenchMetric(name=final_range_name, kind="final_relative_distance_km", **kwargs),
                ControllerBenchMetric(name=min_range_name, kind="min_relative_distance_km", **kwargs),
                ControllerBenchMetric(name=max_range_name, kind="max_relative_distance_km", **kwargs),
                ControllerBenchMetric(name=final_speed_name, kind="final_relative_speed_km_s", **kwargs),
                ControllerBenchMetric(name=dv_name, kind="total_dv_m_s", object_id=objective.object_id),
            ]
        )
        if objective.min_final_range_km is not None:
            criteria.append(ControllerBenchPassCriterion(metric=final_range_name, op=">=", value=objective.min_final_range_km))
        if objective.max_final_range_km is not None:
            criteria.append(ControllerBenchPassCriterion(metric=final_range_name, op="<=", value=objective.max_final_range_km))
        if objective.min_range_km is not None:
            criteria.append(ControllerBenchPassCriterion(metric=min_range_name, op=">=", value=objective.min_range_km))
        if objective.max_range_km is not None:
            criteria.append(ControllerBenchPassCriterion(metric=max_range_name, op="<=", value=objective.max_range_km))
        if objective.max_final_speed_km_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=final_speed_name, op="<=", value=objective.max_final_speed_km_s))
        if objective.max_total_dv_m_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=dv_name, op="<=", value=objective.max_total_dv_m_s))

    elif objective.kind == "scheduled_burn_delivery":
        if not objective.object_id or objective.target_delta_v_m_s is None:
            raise ValueError("scheduled_burn_delivery objective requires object_id and target_delta_v_m_s.")
        error_name = _metric_name(prefix, "delta_v_error_m_s")
        metrics.append(
            ControllerBenchMetric(
                name=error_name,
                kind="scheduled_burn_delta_v_error_m_s",
                object_id=objective.object_id,
                target_value=objective.target_delta_v_m_s,
            )
        )
        if objective.max_delta_v_error_m_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=error_name, op="<=", value=objective.max_delta_v_error_m_s))

    elif objective.kind == "orbital_element_tracking":
        if not objective.object_id:
            raise ValueError("orbital_element_tracking objective requires object_id.")
        if objective.target_semimajor_axis_km is not None:
            sma_name = _metric_name(prefix, "final_semimajor_axis_error_km")
            metrics.append(
                ControllerBenchMetric(
                    name=sma_name,
                    kind="final_semimajor_axis_error_km",
                    object_id=objective.object_id,
                    target_value=objective.target_semimajor_axis_km,
                )
            )
            if objective.max_semimajor_axis_error_km is not None:
                criteria.append(
                    ControllerBenchPassCriterion(metric=sma_name, op="<=", value=objective.max_semimajor_axis_error_km)
                )
        if objective.target_eccentricity is not None:
            ecc_name = _metric_name(prefix, "final_eccentricity_error")
            metrics.append(
                ControllerBenchMetric(
                    name=ecc_name,
                    kind="final_eccentricity_error",
                    object_id=objective.object_id,
                    target_value=objective.target_eccentricity,
                )
            )
            if objective.max_eccentricity_error is not None:
                criteria.append(
                    ControllerBenchPassCriterion(metric=ecc_name, op="<=", value=objective.max_eccentricity_error)
                )
        if not metrics:
            raise ValueError("orbital_element_tracking requires a semimajor-axis or eccentricity target.")

    elif objective.kind == "navigation_estimation":
        if not objective.object_id:
            raise ValueError("navigation_estimation objective requires object_id.")
        position_name = _metric_name(prefix, "navigation_rms_position_error_m")
        velocity_name = _metric_name(prefix, "navigation_rms_velocity_error_m_s")
        kwargs = {"object_id": objective.object_id, "reference_object_id": objective.reference_object_id}
        metrics.extend(
            [
                ControllerBenchMetric(name=position_name, kind="navigation_rms_position_error_m", **kwargs),
                ControllerBenchMetric(name=velocity_name, kind="navigation_rms_velocity_error_m_s", **kwargs),
            ]
        )
        if objective.max_navigation_position_error_m is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=position_name, op="<=", value=objective.max_navigation_position_error_m)
            )
        if objective.max_navigation_velocity_error_m_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=velocity_name,
                    op="<=",
                    value=objective.max_navigation_velocity_error_m_s,
                )
            )

    elif objective.kind == "multirate_consistency":
        if not objective.object_id:
            raise ValueError("multirate_consistency objective requires object_id.")
        count_name = _metric_name(prefix, "fsw_invocation_count")
        jitter_name = _metric_name(prefix, "max_task_cadence_jitter_s")
        missed_task_name = _metric_name(prefix, "fsw_missed_task_release_count")
        missed_sensor_name = _metric_name(prefix, "fsw_missed_sensor_release_count")
        metrics.extend(
            [
                ControllerBenchMetric(name=count_name, kind="fsw_invocation_count", object_id=objective.object_id),
                ControllerBenchMetric(name=jitter_name, kind="max_task_cadence_jitter_s", object_id=objective.object_id),
                ControllerBenchMetric(name=missed_task_name, kind="fsw_missed_task_release_count", object_id=objective.object_id),
                ControllerBenchMetric(name=missed_sensor_name, kind="fsw_missed_sensor_release_count", object_id=objective.object_id),
            ]
        )
        if objective.min_invocation_count is not None:
            criteria.append(ControllerBenchPassCriterion(metric=count_name, op=">=", value=objective.min_invocation_count))
        if objective.max_cadence_jitter_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=jitter_name, op="<=", value=objective.max_cadence_jitter_s))
        if objective.max_missed_task_releases is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=missed_task_name,
                    op="<=",
                    value=objective.max_missed_task_releases,
                )
            )
        if objective.max_missed_sensor_releases is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=missed_sensor_name,
                    op="<=",
                    value=objective.max_missed_sensor_releases,
                )
            )

    elif objective.kind == "actuator_realization":
        if not objective.object_id:
            raise ValueError("actuator_realization objective requires object_id.")
        saturation_name = _metric_name(prefix, "actuator_saturation_fraction")
        force_name = _metric_name(prefix, "max_actuator_force_residual_n")
        torque_name = _metric_name(prefix, "max_actuator_torque_residual_n_m")
        accepted_name = _metric_name(prefix, "accepted_command_fraction")
        realized_name = _metric_name(prefix, "physical_realization_fraction")
        metrics.extend(
            [
                ControllerBenchMetric(name=saturation_name, kind="actuator_saturation_fraction", object_id=objective.object_id),
                ControllerBenchMetric(name=force_name, kind="max_actuator_force_residual_n", object_id=objective.object_id),
                ControllerBenchMetric(name=torque_name, kind="max_actuator_torque_residual_n_m", object_id=objective.object_id),
                ControllerBenchMetric(name=accepted_name, kind="accepted_command_fraction", object_id=objective.object_id),
                ControllerBenchMetric(name=realized_name, kind="physical_realization_fraction", object_id=objective.object_id),
            ]
        )
        if objective.max_actuator_saturation_fraction is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=saturation_name,
                    op="<=",
                    value=objective.max_actuator_saturation_fraction,
                )
            )
        if objective.max_realization_force_error_n is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=force_name, op="<=", value=objective.max_realization_force_error_n)
            )
        if objective.max_realization_torque_error_n_m is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=torque_name,
                    op="<=",
                    value=objective.max_realization_torque_error_n_m,
                )
            )
        if objective.min_accepted_command_fraction is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=accepted_name,
                    op=">=",
                    value=objective.min_accepted_command_fraction,
                )
            )
        if objective.min_physical_realization_fraction is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=realized_name,
                    op=">=",
                    value=objective.min_physical_realization_fraction,
                )
            )

    elif objective.kind == "flight_software_stack_conformance":
        if not objective.object_id:
            raise ValueError("flight_software_stack_conformance objective requires object_id.")
        boundary_name = _metric_name(prefix, "fsw_boundary_error_count")
        invocation_name = _metric_name(prefix, "fsw_invocation_count")
        audit_name = _metric_name(prefix, "fsw_audit_completeness_fraction")
        command_name = _metric_name(prefix, "fsw_command_count")
        profile_name = _metric_name(prefix, "fsw_profile_identity_consistent")
        stack_name = _metric_name(prefix, "fsw_stack_identity_consistent")
        metrics.extend(
            [
                ControllerBenchMetric(name=boundary_name, kind="fsw_boundary_error_count", object_id=objective.object_id),
                ControllerBenchMetric(name=invocation_name, kind="fsw_invocation_count", object_id=objective.object_id),
                ControllerBenchMetric(name=audit_name, kind="fsw_audit_completeness_fraction", object_id=objective.object_id),
                ControllerBenchMetric(name=command_name, kind="fsw_command_count", object_id=objective.object_id),
                ControllerBenchMetric(name=profile_name, kind="fsw_profile_identity_consistent", object_id=objective.object_id),
                ControllerBenchMetric(name=stack_name, kind="fsw_stack_identity_consistent", object_id=objective.object_id),
            ]
        )
        criteria.append(ControllerBenchPassCriterion(metric=boundary_name, op="==", value=0))
        criteria.append(
            ControllerBenchPassCriterion(metric=invocation_name, op=">=", value=objective.min_invocation_count or 1)
        )
        criteria.append(ControllerBenchPassCriterion(metric=profile_name, op="==", value=True))
        criteria.append(ControllerBenchPassCriterion(metric=stack_name, op="==", value=True))
        criteria.append(
            ControllerBenchPassCriterion(
                metric=audit_name,
                op=">=",
                value=(
                    objective.min_audit_completeness_fraction
                    if objective.min_audit_completeness_fraction is not None
                    else 1.0
                ),
            )
        )
        if objective.max_command_count is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=command_name, op="<=", value=objective.max_command_count)
            )
        if objective.expected_translation_mode:
            mode_name = _metric_name(prefix, "selected_mode_matches")
            metrics.append(
                ControllerBenchMetric(
                    name=mode_name,
                    kind="selected_mode_matches",
                    object_id=objective.object_id,
                    expected_value=objective.expected_translation_mode,
                )
            )
            criteria.append(ControllerBenchPassCriterion(metric=mode_name, op="==", value=True))
        if objective.expected_control_law:
            law_name = _metric_name(prefix, "control_law_matches")
            metrics.append(
                ControllerBenchMetric(
                    name=law_name,
                    kind="control_law_matches",
                    object_id=objective.object_id,
                    expected_value=objective.expected_control_law,
                )
            )
            criteria.append(ControllerBenchPassCriterion(metric=law_name, op="==", value=True))

    elif objective.kind == "executive_preemption_recovery":
        if not objective.object_id:
            raise ValueError("executive_preemption_recovery objective requires object_id.")
        for suffix, metric_kind, required in (
            ("preemption_seen", "executive_preemption_seen", objective.require_preemption),
            ("recovery_seen", "executive_recovery_seen", objective.require_recovery),
            ("primary_resume_seen", "executive_primary_resume_seen", objective.require_primary_resume),
        ):
            name = _metric_name(prefix, suffix)
            metrics.append(ControllerBenchMetric(name=name, kind=metric_kind, object_id=objective.object_id))
            if required:
                criteria.append(ControllerBenchPassCriterion(metric=name, op="==", value=True))

    elif objective.kind == "rocket_orbit_insertion":
        rocket_id = objective.object_id or "rocket"
        insertion_name = _metric_name(prefix, "rocket_insertion_achieved")
        insertion_time_name = _metric_name(prefix, "rocket_insertion_time_s")
        final_alt_name = _metric_name(prefix, "rocket_final_altitude_km")
        final_speed_name = _metric_name(prefix, "rocket_final_speed_km_s")
        apo_name = _metric_name(prefix, "rocket_final_apoapsis_alt_km")
        peri_name = _metric_name(prefix, "rocket_final_periapsis_alt_km")
        ecc_name = _metric_name(prefix, "rocket_final_eccentricity")
        prop_name = _metric_name(prefix, "rocket_final_propellant_remaining_fraction")
        q_name = _metric_name(prefix, "rocket_max_dynamic_pressure_pa")
        mach_name = _metric_name(prefix, "rocket_max_mach")
        alpha_name = _metric_name(prefix, "rocket_max_abs_alpha_deg")
        tvc_name = _metric_name(prefix, "rocket_max_tvc_gimbal_deg")
        aero_force_name = _metric_name(prefix, "rocket_max_aero_force_n")
        dv_name = _metric_name(prefix, "rocket_total_dv_m_s")
        burn_name = _metric_name(prefix, "rocket_burn_samples")
        metrics.extend(
            [
                ControllerBenchMetric(name=insertion_name, kind="rocket_insertion_achieved", object_id=rocket_id),
                ControllerBenchMetric(name=insertion_time_name, kind="rocket_insertion_time_s", object_id=rocket_id),
                ControllerBenchMetric(name=final_alt_name, kind="rocket_final_altitude_km", object_id=rocket_id),
                ControllerBenchMetric(name=final_speed_name, kind="rocket_final_speed_km_s", object_id=rocket_id),
                ControllerBenchMetric(name=apo_name, kind="rocket_final_apoapsis_alt_km", object_id=rocket_id),
                ControllerBenchMetric(name=peri_name, kind="rocket_final_periapsis_alt_km", object_id=rocket_id),
                ControllerBenchMetric(name=ecc_name, kind="rocket_final_eccentricity", object_id=rocket_id),
                ControllerBenchMetric(
                    name=prop_name,
                    kind="rocket_final_propellant_remaining_fraction",
                    object_id=rocket_id,
                ),
                ControllerBenchMetric(name=q_name, kind="rocket_max_dynamic_pressure_pa", object_id=rocket_id),
                ControllerBenchMetric(name=mach_name, kind="rocket_max_mach", object_id=rocket_id),
                ControllerBenchMetric(name=alpha_name, kind="rocket_max_abs_alpha_deg", object_id=rocket_id),
                ControllerBenchMetric(name=tvc_name, kind="rocket_max_tvc_gimbal_deg", object_id=rocket_id),
                ControllerBenchMetric(name=aero_force_name, kind="rocket_max_aero_force_n", object_id=rocket_id),
                ControllerBenchMetric(name=dv_name, kind="rocket_total_dv_m_s", object_id=rocket_id),
                ControllerBenchMetric(name=burn_name, kind="rocket_burn_samples", object_id=rocket_id),
            ]
        )
        if objective.require_insertion:
            criteria.append(ControllerBenchPassCriterion(metric=insertion_name, op="==", value=True))
        if objective.max_insertion_time_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=insertion_time_name, op="<=", value=objective.max_insertion_time_s)
            )
        if objective.max_final_apoapsis_error_km is not None:
            if objective.target_altitude_km is None:
                raise ValueError("rocket_orbit_insertion max_final_apoapsis_error_km requires target_altitude_km.")
            tol = float(objective.max_final_apoapsis_error_km)
            target = float(objective.target_altitude_km)
            criteria.extend(
                [
                    ControllerBenchPassCriterion(metric=apo_name, op=">=", value=target - tol),
                    ControllerBenchPassCriterion(metric=apo_name, op="<=", value=target + tol),
                ]
            )
        if objective.max_final_periapsis_error_km is not None:
            if objective.target_altitude_km is None:
                raise ValueError("rocket_orbit_insertion max_final_periapsis_error_km requires target_altitude_km.")
            tol = float(objective.max_final_periapsis_error_km)
            target = float(objective.target_altitude_km)
            criteria.extend(
                [
                    ControllerBenchPassCriterion(metric=peri_name, op=">=", value=target - tol),
                    ControllerBenchPassCriterion(metric=peri_name, op="<=", value=target + tol),
                ]
            )
        if objective.max_final_eccentricity is not None:
            criteria.append(ControllerBenchPassCriterion(metric=ecc_name, op="<=", value=objective.max_final_eccentricity))
        if objective.min_propellant_remaining_fraction is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=prop_name,
                    op=">=",
                    value=objective.min_propellant_remaining_fraction,
                )
            )
        if objective.max_dynamic_pressure_pa is not None:
            criteria.append(ControllerBenchPassCriterion(metric=q_name, op="<=", value=objective.max_dynamic_pressure_pa))
        if objective.max_mach is not None:
            criteria.append(ControllerBenchPassCriterion(metric=mach_name, op="<=", value=objective.max_mach))
        if objective.max_abs_alpha_deg is not None:
            criteria.append(ControllerBenchPassCriterion(metric=alpha_name, op="<=", value=objective.max_abs_alpha_deg))
        if objective.max_tvc_gimbal_deg is not None:
            criteria.append(ControllerBenchPassCriterion(metric=tvc_name, op="<=", value=objective.max_tvc_gimbal_deg))
        if objective.max_aero_force_n is not None:
            criteria.append(
                ControllerBenchPassCriterion(metric=aero_force_name, op="<=", value=objective.max_aero_force_n)
            )
        if objective.max_total_dv_m_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=dv_name, op="<=", value=objective.max_total_dv_m_s))
    else:
        raise ValueError(f"Unsupported controller bench objective kind: {objective.kind}")

    if objective.require_not_terminated_early:
        term_name = _metric_name(prefix, "terminated_early")
        metrics.append(
            ControllerBenchMetric(
                name=term_name,
                source_path="summary.terminated_early",
            )
        )
        criteria.append(ControllerBenchPassCriterion(metric=term_name, op="==", value=False))

    return tuple(metrics), tuple(criteria)


def _resolve_metric_specs(suite: ControllerBenchConfig, case: ControllerBenchCase) -> tuple[ControllerBenchMetric, ...]:
    metrics = list(case.metrics if case.metrics else suite.metrics)
    objectives = case.objectives if case.objectives else suite.objectives
    for objective in objectives:
        objective_metrics, _ = _objective_specifications(objective)
        metrics.extend(objective_metrics)
    return _dedupe_metrics(metrics)


def _resolve_pass_criteria(
    suite: ControllerBenchConfig,
    case: ControllerBenchCase,
    objectives: tuple[ControllerBenchObjective, ...],
) -> tuple[ControllerBenchPassCriterion, ...]:
    criteria = list(case.pass_criteria if case.pass_criteria else suite.pass_criteria)
    for objective in objectives:
        _, objective_criteria = _objective_specifications(objective)
        criteria.extend(objective_criteria)
    return tuple(criteria)


def _evaluate_objectives(
    metrics: dict[str, Any],
    objectives: tuple[ControllerBenchObjective, ...],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for objective in objectives:
        prefix = _slug(objective.name or objective.kind)
        objective_metrics, objective_criteria = _objective_specifications(objective)
        scoped_metrics = {metric.name: metrics.get(metric.name) for metric in objective_metrics}
        passed, failed = _evaluate_pass_criteria(scoped_metrics, objective_criteria)
        results.append(
            {
                "name": objective.name or objective.kind,
                "kind": objective.kind,
                "passed": bool(passed),
                "failed_criteria": failed,
                "metrics": scoped_metrics,
                "metric_prefix": prefix,
            }
        )
    return results


def _run_single_bench_case(
    suite: ControllerBenchConfig,
    case: ControllerBenchCase,
    variant: ControllerVariant,
) -> dict[str, Any]:
    run, _ = _run_single_bench_case_with_payload(suite=suite, case=case, variant=variant)
    return run


def _run_single_bench_case_with_payload(
    suite: ControllerBenchConfig,
    case: ControllerBenchCase,
    variant: ControllerVariant,
    step_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = load_simulation_yaml(case.config_path)
    if analysis_study_type(scenario) != "single_run":
        raise ValueError(f"Controller bench case '{case.name}' must be a single-run scenario.")

    root = scenario.to_dict()
    run_outdir = suite.output_dir / case.name / variant.name
    _apply_controller_variant(root, suite.controller_target, variant)
    materialized_root = deepcopy(root)
    _apply_benchmark_output_overrides(root, run_outdir, suite)
    cfg = scenario_config_from_dict(root)
    payload = _run_single_config(cfg, step_callback=step_callback)
    payload["config_path"] = str(case.config_path.resolve())
    target_materialized = dict(
        dict(materialized_root.get("objects", {}) or {}).get(suite.controller_target.object_id, {}) or {}
    )
    fsw_materialized = dict(target_materialized.get("flight_software", {}) or {})
    if fsw_materialized.get("profile"):
        from sim.flight_software.profiles import materialize_use_case_profile

        fsw_materialized = materialize_use_case_profile(
            str(fsw_materialized["profile"]),
            params=dict(fsw_materialized.get("params", {}) or {}),
            hardware_profile=fsw_materialized.get("hardware_profile"),
            task_period_s=fsw_materialized.get("task_period_s"),
        ).to_config()
    payload["controller_bench_context"] = {
        "object_id": suite.controller_target.object_id,
        "task_period_s": fsw_materialized.get("task_period_s"),
        "sensor_period_s": dict(target_materialized.get("knowledge", {}) or {}).get("refresh_rate_s"),
        "profile_id": fsw_materialized.get("profile"),
        "stack_id": fsw_materialized.get("stack"),
    }

    objectives = case.objectives if case.objectives else suite.objectives
    metric_specs = _resolve_metric_specs(suite, case)
    metrics = {metric.name: evaluate_metric(metric, payload) for metric in metric_specs}
    criteria = _resolve_pass_criteria(suite, case, objectives)
    passed, failed_criteria = _evaluate_pass_criteria(metrics, criteria)
    objective_results = _evaluate_objectives(metrics, objectives)
    artifacts = {
        "output_dir": str(run_outdir),
        "summary_json": str(run_outdir / "master_run_summary.json"),
    }
    if suite.save_run_payloads:
        artifacts["run_log_json"] = str(run_outdir / "master_run_log.json")
    linear_feedback_artifacts = write_linear_feedback_diagnostics(
        payload, suite.controller_target.object_id, run_outdir
    )
    if linear_feedback_artifacts:
        artifacts["linear_feedback"] = linear_feedback_artifacts
    engineering_summary: dict[str, Any] = {}
    if suite.optimization.enabled:
        from sim.controller_lab.engineering import analyze_frame_sign_probe, analyze_stack_conformance

        reference_id = next(
            (
                objective.reference_object_id
                for objective in objectives
                if objective.object_id == suite.controller_target.object_id and objective.reference_object_id
            ),
            "",
        )
        analyses: dict[str, Any] = {}
        if suite.engineering.enabled and suite.engineering.stack_conformance:
            analyses["stack_conformance"] = analyze_stack_conformance(
                payload, materialized_root, suite.controller_target.object_id
            )
        if suite.engineering.enabled and suite.engineering.frame_sign_probe:
            analyses["frame_sign_probe"] = analyze_frame_sign_probe(
                payload,
                materialized_root,
                suite.controller_target.object_id,
                reference_id,
            )
        engineering_summary = {"analyses": analyses}
    else:
        engineering_artifacts, engineering_summary = write_run_engineering_artifacts(
            payload=payload,
            scenario_root=materialized_root,
            suite=suite,
            variant=variant,
            case_name=case.name,
            objectives=objectives,
            run_outdir=run_outdir,
        )
        artifacts.update(engineering_artifacts)
        if not passed and suite.engineering.failure_reduction:
            artifacts.update(
                write_minimal_reproducer(
                    suite=suite,
                    variant=variant,
                    case_name=case.name,
                    case_config_path=case.config_path,
                    run_outdir=run_outdir,
                )
            )
    analyses = dict(engineering_summary.get("analyses", {}) or {})
    stack_conformance = dict(analyses.get("stack_conformance", {}) or {})
    if any(objective.kind == "flight_software_stack_conformance" for objective in objectives) and not bool(
        stack_conformance.get("passed", False)
    ):
        passed = False
        failure = "engineering.stack_conformance == true"
        if failure not in failed_criteria:
            failed_criteria.append(failure)
        for objective_result in objective_results:
            if objective_result.get("kind") == "flight_software_stack_conformance":
                objective_result["passed"] = False
                objective_result.setdefault("failed_criteria", []).append(failure)
    if any(objective.kind == "nonzero_ric_hold" for objective in objectives):
        frame_probe = dict(analyses.get("frame_sign_probe", {}) or {})
        # RMOE feedback regulates orbital-element centers, amplitudes, and
        # drift in a priority sequence, so Cartesian first-command alignment
        # is explicitly not applicable. Missing or unexcited evidence for a
        # Cartesian feedback law remains release-blocking.
        if frame_probe.get("status") != "not_applicable" and frame_probe.get("directionally_correct") is not True:
            passed = False
            failure = "engineering.frame_sign_probe.directionally_correct == true"
            if failure not in failed_criteria:
                failed_criteria.append(failure)
            for objective_result in objective_results:
                if objective_result.get("kind") == "nonzero_ric_hold":
                    objective_result["passed"] = False
                    objective_result.setdefault("failed_criteria", []).append(failure)
    run = {
        "variant_name": variant.name,
        "case_name": case.name,
        "description": case.description,
        "metrics": metrics,
        "passed": bool(passed),
        "failed_criteria": failed_criteria,
        "objective_results": objective_results,
        "artifacts": artifacts,
        "config_path": str(case.config_path.resolve()),
        "engineering": dict(engineering_summary.get("analyses", {}) or {}),
    }
    run["constraint_ledger"] = _constraint_ledger_from_payload(payload, metrics, objectives)
    run["failure_analysis"] = _failure_analysis_for_run(run, metrics, criteria, objectives, payload)
    return run, payload

__all__ = [name for name in globals() if not name.startswith("__")]
