# ruff: noqa: F401,F403,F405,I001
from sim.observation_normalization import *


def _validate_kalman_sigma(value: float, *, name: str, allow_zero: bool) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        domain = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {domain}.") from exc
    domain_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not np.isfinite(parsed) or not domain_ok:
        domain = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {domain}.")
    return parsed


def _row_covariance(
    row: Mapping[str, Any],
    dimension: int,
    *,
    default_sigma: float = 1.0,
) -> np.ndarray:
    uncertainty = dict(row.get("uncertainty", {}) or {})
    matrix = uncertainty.get("matrix")
    if matrix is not None:
        covariance = np.asarray(matrix, dtype=float)
        if covariance.shape[0] >= dimension and covariance.shape[1] >= dimension:
            return np.array(covariance[:dimension, :dimension], dtype=float)
    position_sigma = _validate_kalman_sigma(
        default_sigma if row.get("position_sigma_km") is None else row.get("position_sigma_km"),
        name="position_sigma_km",
        allow_zero=False,
    )
    if dimension == 3:
        return np.eye(3) * position_sigma**2
    velocity_sigma = _validate_kalman_sigma(
        1.0e-3 if row.get("velocity_sigma_km_s") is None else row.get("velocity_sigma_km_s"),
        name="velocity_sigma_km_s",
        allow_zero=False,
    )
    return np.diag(
        [position_sigma**2] * 3
        + [velocity_sigma**2] * 3
    )


def _fit_partition_observations(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in observations]
    declared = {str(row.get("partition", "unassigned") or "unassigned").strip().lower() for row in rows}
    if declared <= {"unassigned"}:
        return rows
    return [row for row in rows if str(row.get("partition", "unassigned") or "unassigned").strip().lower() == "fit"]


def _has_explicit_observation_partitions(observations: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(row.get("partition", "unassigned") or "unassigned").strip().lower() != "unassigned"
        for row in observations
    )


def fit_state_from_position_observations(
    packet: ObservationPacket | Mapping[str, Any],
    *,
    object_id: str | None = None,
    role: str | None = None,
    epoch: str = "first",
) -> dict[str, Any]:
    packet_obj = observation_packet_from_dict(packet.to_dict() if isinstance(packet, ObservationPacket) else packet)
    input_observations = packet_obj.observations
    observations = _fit_partition_observations(input_observations)
    errors = _position_observation_errors(observations)
    if errors:
        raise ValueError("; ".join(errors))
    epoch_time_s = _estimate_epoch_s(observations, epoch)
    t = np.array([float(row["time_s"]) - epoch_time_s for row in observations], dtype=float)
    positions = np.array([row["position_eci_km"] for row in observations], dtype=float)
    sigmas = np.array(
        [float(np.sqrt(np.mean(np.diag(_row_covariance(row, 3))))) for row in observations],
        dtype=float,
    )
    has_velocity_observations = all("velocity_eci_km_s" in row for row in observations)
    velocities = (
        np.array([row["velocity_eci_km_s"] for row in observations], dtype=float) if has_velocity_observations else None
    )
    normal = np.zeros((6, 6), dtype=float)
    rhs = np.zeros(6, dtype=float)
    for index, row in enumerate(observations):
        h_position = np.hstack((np.eye(3), float(t[index]) * np.eye(3)))
        z_parts = [positions[index]]
        h_parts = [h_position]
        dimension = 3
        if velocities is not None:
            h_parts.append(np.hstack((np.zeros((3, 3)), np.eye(3))))
            z_parts.append(velocities[index])
            dimension = 6
        h = np.vstack(h_parts)
        z = np.concatenate(z_parts)
        covariance_i = _row_covariance(row, dimension)
        covariance_inv = np.linalg.pinv(covariance_i, hermitian=True)
        normal += h.T @ covariance_inv @ h
        rhs += h.T @ covariance_inv @ z
    covariance = np.linalg.pinv(normal, hermitian=True)
    state = covariance @ rhs
    fitted = positions * 0.0
    fitted_velocity = positions * 0.0
    for index, time_offset in enumerate(t):
        fitted[index] = state[:3] + float(time_offset) * state[3:]
        fitted_velocity[index] = state[3:]
    residuals = positions - fitted
    velocity_residuals = velocities - fitted_velocity if velocities is not None else None
    rms_km = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    residual_norms = np.linalg.norm(residuals, axis=1)
    velocity_residual_norms = np.linalg.norm(velocity_residuals, axis=1) if velocity_residuals is not None else None
    out_object_id = str(object_id or packet_obj.to_dict().get("object_id") or "target")
    out_role = str(role or out_object_id)
    jd_epoch = _epoch_jd_from_time(observations, epoch_time_s)
    mission_packet = _state_estimate_packet(
        object_id=out_object_id,
        role=out_role,
        state=state,
        covariance=covariance,
        epoch_jd_utc=jd_epoch,
        method="batch_position_fit",
        source_packet=packet_obj,
        diagnostics={
            "epoch": epoch,
            "epoch_time_s": float(epoch_time_s),
            "rms_position_residual_km": rms_km,
            "max_position_residual_km": float(np.max(residual_norms)),
            "used_velocity_observations": bool(has_velocity_observations),
            "velocity_observation_count": len(observations) if has_velocity_observations else 0,
            "rms_velocity_residual_km_s": float(np.sqrt(np.mean(velocity_residual_norms**2)))
            if velocity_residual_norms is not None
            else None,
            "max_velocity_residual_km_s": float(np.max(velocity_residual_norms))
            if velocity_residual_norms is not None
            else None,
            "observation_count": len(observations),
            "input_observation_count": len(input_observations),
            "partition_policy": "explicit_fit_only"
            if _has_explicit_observation_partitions(input_observations)
            else "all_unassigned",
            "arc_length_s": float(observations[-1]["time_s"] - observations[0]["time_s"]),
            "quality_flags": _estimate_quality_flags(
                observation_count=len(observations),
                arc_length_s=float(observations[-1]["time_s"] - observations[0]["time_s"]),
                rms_position_residual_km=rms_km,
                position_sigmas_km=sigmas,
            ),
            "residual_samples": _residual_samples(observations, residuals, velocity_residuals),
        },
        warnings=[
            "State estimate comes from a constant-velocity least-squares fit to ECI position observations; it is preliminary OD, not a full orbital batch least-squares solution."
        ],
    )
    return {
        "status": "ready_with_warnings",
        "method": "batch_position_fit",
        "object_id": out_object_id,
        "epoch_time_s": float(epoch_time_s),
        "epoch_jd_utc": jd_epoch,
        "state_eci_km_s": state.tolist(),
        "covariance_6x6": covariance.tolist(),
        "diagnostics": mission_packet.to_dict()["objects"][out_object_id]["state_estimate"]["diagnostics"],
        "mission_input_packet": mission_packet.to_dict(),
    }


def kalman_filter_position_observations(
    packet: ObservationPacket | Mapping[str, Any],
    *,
    object_id: str | None = None,
    role: str | None = None,
    process_accel_sigma_km_s2: float = 1.0e-7,
    initial_position_sigma_km: float = 1.0,
    initial_velocity_sigma_km_s: float = 1.0e-3,
    default_position_sigma_km: float = 1.0,
    initial_velocity_eci_km_s: Sequence[float] | None = None,
    initial_state_eci_km_s: Sequence[float] | None = None,
    initialization_mode: str = "causal",
) -> dict[str, Any]:
    """Run the preliminary constant-velocity position filter.

    The default ``causal`` mode initializes at the first retained observation
    using that observation's position and a zero velocity prior unless
    ``initial_velocity_eci_km_s`` is supplied.  Because that first position is
    the observation-seeded prior, its row is initialization-only and its
    configured prior covariance is retained; its innovation and NIS are not
    valid measurement-update evidence.  Supply an explicit six-component
    ``initial_state_eci_km_s`` prior at the first observation epoch to
    assimilate the first observation as a measurement update.  Each later
    history row then depends only on observations through that row.  ``batch``
    is an explicit compatibility mode for the former full-arc least-squares
    initialization; its history is future-conditioned and must not be treated
    as causal filter evidence.
    """

    normalized_initialization_mode = str(initialization_mode or "causal").strip().lower()
    if normalized_initialization_mode not in {"causal", "batch"}:
        raise ValueError("initialization_mode must be 'causal' or 'batch'.")
    if initial_velocity_eci_km_s is None:
        initial_velocity = np.zeros(3, dtype=float)
    else:
        try:
            initial_velocity = np.asarray(initial_velocity_eci_km_s, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial_velocity_eci_km_s must be a finite length-3 vector.") from exc
        if initial_velocity.size != 3 or not np.all(np.isfinite(initial_velocity)):
            raise ValueError("initial_velocity_eci_km_s must be a finite length-3 vector.")
    if initial_state_eci_km_s is None:
        initial_state_prior = None
    else:
        try:
            initial_state_prior = np.asarray(initial_state_eci_km_s, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial_state_eci_km_s must be a finite length-6 vector.") from exc
        if initial_state_prior.size != 6 or not np.all(np.isfinite(initial_state_prior)):
            raise ValueError("initial_state_eci_km_s must be a finite length-6 vector.")
    if initial_state_eci_km_s is not None and initial_velocity_eci_km_s is not None:
        raise ValueError("initial_state_eci_km_s and initial_velocity_eci_km_s cannot be supplied together.")
    if normalized_initialization_mode == "batch" and (
        initial_velocity_eci_km_s is not None or initial_state_eci_km_s is not None
    ):
        raise ValueError("explicit initial priors are only supported with initialization_mode='causal'.")
    process_accel_sigma = _validate_kalman_sigma(
        process_accel_sigma_km_s2,
        name="process_accel_sigma_km_s2",
        allow_zero=True,
    )
    initial_position_sigma = _validate_kalman_sigma(
        initial_position_sigma_km,
        name="initial_position_sigma_km",
        allow_zero=True,
    )
    initial_velocity_sigma = _validate_kalman_sigma(
        initial_velocity_sigma_km_s,
        name="initial_velocity_sigma_km_s",
        allow_zero=True,
    )
    default_position_sigma = _validate_kalman_sigma(
        default_position_sigma_km,
        name="default_position_sigma_km",
        allow_zero=False,
    )
    packet_obj = observation_packet_from_dict(packet.to_dict() if isinstance(packet, ObservationPacket) else packet)
    input_observations = packet_obj.observations
    observations = _fit_partition_observations(input_observations)
    errors = _position_observation_errors(observations)
    if errors:
        raise ValueError("; ".join(errors))
    for row in observations:
        _row_covariance(row, 3, default_sigma=default_position_sigma)
    if normalized_initialization_mode == "batch":
        fit = fit_state_from_position_observations(packet_obj, object_id=object_id, role=role, epoch="first")
        x = np.array(fit["state_eci_km_s"], dtype=float)
        initialization_source = "batch_position_fit"
        first_row_is_initialization_only = False
    elif initial_state_prior is not None:
        x = initial_state_prior.copy()
        initialization_source = "explicit_full_state_prior"
        first_row_is_initialization_only = False
    else:
        first_position = np.asarray(observations[0]["position_eci_km"], dtype=float).reshape(3)
        x = np.hstack((first_position, initial_velocity))
        initialization_source = "first_position_observation_plus_velocity_prior"
        first_row_is_initialization_only = True
    initial_state = x.copy()
    p = np.diag(
        [
            initial_position_sigma**2,
            initial_position_sigma**2,
            initial_position_sigma**2,
            initial_velocity_sigma**2,
            initial_velocity_sigma**2,
            initial_velocity_sigma**2,
        ]
    )
    h = np.zeros((3, 6), dtype=float)
    h[:, :3] = np.eye(3)
    i6 = np.eye(6)
    history: list[dict[str, Any]] = []
    previous_time_s = float(observations[0]["time_s"])
    if first_row_is_initialization_only:
        history.append(
            {
                "time_s": previous_time_s,
                "innovation_km": None,
                "innovation_norm_km": None,
                "nis": None,
                "position_sigma_trace_km2": float(np.trace(p[:3, :3])),
                "velocity_sigma_trace_km2_s2": float(np.trace(p[3:, 3:])),
                "state_eci_km_s": x.tolist(),
                "row_type": "initialization",
                "measurement_update_applied": False,
                "innovation_valid": False,
            }
        )
        rows_to_update = observations[1:]
    else:
        rows_to_update = observations
    for row in rows_to_update:
        t_s = float(row["time_s"])
        dt_s = max(t_s - previous_time_s, 0.0)
        f, q = _constant_velocity_matrices(dt_s, process_accel_sigma)
        x_pred = f @ x
        p_pred = f @ p @ f.T + q
        z = np.array(row["position_eci_km"], dtype=float)
        r = _row_covariance(row, 3, default_sigma=default_position_sigma)
        innovation = z - h @ x_pred
        s = h @ p_pred @ h.T + r
        try:
            k = np.linalg.solve(s.T, (p_pred @ h.T).T).T
            s_y = np.linalg.solve(s, innovation)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s)
            k = p_pred @ h.T @ s_inv
            s_y = s_inv @ innovation
        x = x_pred + k @ innovation
        p = (i6 - k @ h) @ p_pred @ (i6 - k @ h).T + k @ r @ k.T
        p = 0.5 * (p + p.T)
        history.append(
            {
                "time_s": t_s,
                "innovation_km": innovation.tolist(),
                "innovation_norm_km": float(np.linalg.norm(innovation)),
                "nis": float(innovation.T @ s_y),
                "position_sigma_trace_km2": float(np.trace(p[:3, :3])),
                "velocity_sigma_trace_km2_s2": float(np.trace(p[3:, 3:])),
                "state_eci_km_s": x.tolist(),
                "row_type": "measurement_update",
                "measurement_update_applied": True,
                "innovation_valid": True,
            }
        )
        previous_time_s = t_s
    out_object_id = str(object_id or packet_obj.to_dict().get("object_id") or "target")
    out_role = str(role or out_object_id)
    jd_epoch = _epoch_jd_from_time(observations, float(observations[-1]["time_s"]))
    valid_history = [row for row in history if bool(row.get("innovation_valid", True))]
    innovation_norms = [float(row["innovation_norm_km"]) for row in valid_history]
    nis_values = [float(row["nis"]) for row in valid_history]
    mean_nis = float(np.mean(nis_values)) if nis_values else None
    max_nis = float(np.max(nis_values)) if nis_values else None
    mean_innovation_norm = float(np.mean(innovation_norms)) if innovation_norms else None
    max_innovation_norm = float(np.max(innovation_norms)) if innovation_norms else None
    if normalized_initialization_mode == "batch":
        filter_warning = (
            "State estimate uses an explicitly requested full-arc batch initializer before a linear constant-velocity "
            "Kalman filter; its history is future-conditioned and is not causal filter evidence."
        )
    elif first_row_is_initialization_only:
        filter_warning = (
            "State estimate comes from a causal linear constant-velocity Kalman filter initialized from the first "
            "position observation and an explicit velocity prior; the first row is initialization-only, so its "
            "innovation and NIS are not measurement-update evidence; it is not an orbital-dynamics EKF."
        )
    else:
        filter_warning = (
            "State estimate comes from a causal linear constant-velocity Kalman filter initialized from an explicit "
            "six-component state prior; the first observation is assimilated as a measurement update; it is not an "
            "orbital-dynamics EKF."
        )
    mission_packet = _state_estimate_packet(
        object_id=out_object_id,
        role=out_role,
        state=x,
        covariance=p,
        epoch_jd_utc=jd_epoch,
        method="kalman_position_filter",
        source_packet=packet_obj,
        diagnostics={
            "process_accel_sigma_km_s2": process_accel_sigma,
            "initialization_mode": normalized_initialization_mode,
            "initialization_source": initialization_source,
            "initialization_only_row_count": int(sum(not bool(row.get("innovation_valid", True)) for row in history)),
            "initial_state_eci_km_s": initial_state.tolist(),
            "mean_innovation_norm_km": mean_innovation_norm,
            "max_innovation_norm_km": max_innovation_norm,
            "mean_nis": mean_nis,
            "max_nis": max_nis,
            "valid_innovation_count": len(valid_history),
            "final_covariance_trace": float(np.trace(p)),
            "final_position_sigma_trace_km2": float(np.trace(p[:3, :3])),
            "final_velocity_sigma_trace_km2_s2": float(np.trace(p[3:, 3:])),
            "observation_count": len(observations),
            "input_observation_count": len(input_observations),
            "partition_policy": "explicit_fit_only"
            if _has_explicit_observation_partitions(input_observations)
            else "all_unassigned",
            "innovation_samples": [
                {
                    "time_s": float(row["time_s"]),
                    "innovation_norm_km": (
                        None if row["innovation_norm_km"] is None else float(row["innovation_norm_km"])
                    ),
                    "nis": None if row["nis"] is None else float(row["nis"]),
                    "row_type": row["row_type"],
                    "measurement_update_applied": bool(row["measurement_update_applied"]),
                    "innovation_valid": bool(row["innovation_valid"]),
                }
                for row in history
            ],
            "quality_flags": _kalman_quality_flags(
                observation_count=len(observations),
                mean_nis=float(mean_nis) if mean_nis is not None else float("nan"),
                max_nis=float(max_nis) if max_nis is not None else float("nan"),
                final_covariance_trace=float(np.trace(p)),
            ),
        },
        warnings=[filter_warning],
    )
    return {
        "status": "ready_with_warnings",
        "method": "kalman_position_filter",
        "object_id": out_object_id,
        "epoch_time_s": float(observations[-1]["time_s"]),
        "epoch_jd_utc": jd_epoch,
        "state_eci_km_s": x.tolist(),
        "covariance_6x6": p.tolist(),
        "history": history,
        "diagnostics": mission_packet.to_dict()["objects"][out_object_id]["state_estimate"]["diagnostics"],
        "mission_input_packet": mission_packet.to_dict(),
    }
