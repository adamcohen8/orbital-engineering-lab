"""Directed terminal-to-terminal free-space link analysis v0.1."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.event_refinement import (
    AvailabilityEvaluator,
    RefinedTransition,
    SampledAvailabilityInterval,
    availability_intervals,
    refine_availability_transitions,
)
from sim.analysis.global_coverage import _validated_quaternion
from sim.dynamics.orbit.frames import (
    FrameContext,
    eci_to_ecef_rotation_context,
    eci_to_ecef_rotation_derivative_context,
)
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.geodesy import (
    WGS84_A_KM,
    WGS84_B_KM,
    ecef_to_enu_rotation,
    geodetic_to_ecef_km,
)
from sim.utils.quaternion import quaternion_to_dcm_bn

DIRECTED_LINK_CONTRACT_VERSION = "oel.directed-link-analysis.v0.1"
SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380_649e-23
LINK_REASON_NAMES = (
    "invalid_input",
    "state_unavailable",
    "attitude_unavailable",
    "earth_occulted",
    "below_elevation_mask",
    "beyond_max_range",
    "tx_outside_pattern",
    "rx_outside_pattern",
    "negative_margin",
    "available",
)
_ANGULAR_TOLERANCE_RAD = 1.0e-12
_RANGE_TOLERANCE_KM = 1.0e-9


def _required_id(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _finite_float(value: Any, field_name: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@dataclass(frozen=True)
class TerminalPattern:
    kind: str
    gain_dbi: float
    half_angle_rad: float | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip().lower()
        if kind not in {"constant", "axisymmetric_hard_cone"}:
            raise ValueError("Terminal pattern kind must be constant or axisymmetric_hard_cone.")
        gain = _finite_float(self.gain_dbi, "gain_dbi")
        if kind == "constant":
            if self.half_angle_rad is not None:
                raise ValueError("A constant terminal pattern must not declare half_angle_rad.")
            angle = None
        else:
            if self.half_angle_rad is None:
                raise ValueError("A directional terminal pattern requires half_angle_rad.")
            angle = _finite_float(self.half_angle_rad, "half_angle_rad")
            if not 0.0 < angle <= np.pi:
                raise ValueError("half_angle_rad must be within (0, pi].")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "gain_dbi", gain)
        object.__setattr__(self, "half_angle_rad", angle)

    @property
    def attitude_independent(self) -> bool:
        return self.kind == "constant"


@dataclass(frozen=True)
class LinkTerminal:
    terminal_id: str
    asset_id: str
    parent_frame: str
    quat_parent_from_terminal: tuple[float, float, float, float]
    pattern: TerminalPattern

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal_id", _required_id(self.terminal_id, "terminal_id"))
        object.__setattr__(self, "asset_id", _required_id(self.asset_id, "asset_id"))
        frame = str(self.parent_frame or "").strip().lower()
        if frame not in {"body", "enu"}:
            raise ValueError("parent_frame must be body or enu.")
        object.__setattr__(self, "parent_frame", frame)
        mounting = _validated_quaternion(
            self.quat_parent_from_terminal,
            "quat_parent_from_terminal",
        )
        object.__setattr__(
            self,
            "quat_parent_from_terminal",
            tuple(float(item) for item in mounting),
        )
        if not isinstance(self.pattern, TerminalPattern):
            raise ValueError("pattern must be a validated TerminalPattern.")


@dataclass(frozen=True)
class DirectedLinkConfig:
    analysis_id: str
    link_id: str
    tx_terminal: LinkTerminal
    rx_terminal: LinkTerminal
    carrier_frequency_hz: float
    tx_power_w: float
    data_rate_bps: float
    system_noise_temperature_k: float
    required_eb_n0_db: float
    tx_line_loss_db: float = 0.0
    rx_line_loss_db: float = 0.0
    misc_loss_db: float = 0.0
    min_fixed_site_elevation_rad: float | None = None
    max_range_km: float | None = None
    transition_time_tolerance_s: float | None = None
    transition_max_iterations: int | None = None
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_id", _required_id(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "link_id", _required_id(self.link_id, "link_id"))
        if not isinstance(self.tx_terminal, LinkTerminal) or not isinstance(
            self.rx_terminal, LinkTerminal
        ):
            raise ValueError("tx_terminal and rx_terminal must be validated LinkTerminal values.")
        if self.tx_terminal.terminal_id == self.rx_terminal.terminal_id:
            raise ValueError("Transmitting and receiving terminal IDs must differ.")
        for field_name in (
            "carrier_frequency_hz",
            "tx_power_w",
            "data_rate_bps",
            "system_noise_temperature_k",
        ):
            value = _finite_float(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "required_eb_n0_db",
            _finite_float(self.required_eb_n0_db, "required_eb_n0_db"),
        )
        for field_name in ("tx_line_loss_db", "rx_line_loss_db", "misc_loss_db"):
            value = _finite_float(getattr(self, field_name), field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be nonnegative.")
            object.__setattr__(self, field_name, value)
        if self.min_fixed_site_elevation_rad is not None:
            elevation = _finite_float(
                self.min_fixed_site_elevation_rad,
                "min_fixed_site_elevation_rad",
            )
            if not -0.5 * np.pi <= elevation <= 0.5 * np.pi:
                raise ValueError("min_fixed_site_elevation_rad must be within [-pi/2, pi/2].")
            object.__setattr__(self, "min_fixed_site_elevation_rad", elevation)
        if self.max_range_km is not None:
            maximum_range = _finite_float(self.max_range_km, "max_range_km")
            if maximum_range <= 0.0:
                raise ValueError("max_range_km must be positive.")
            object.__setattr__(self, "max_range_km", maximum_range)
        refinement_values = (
            self.transition_time_tolerance_s,
            self.transition_max_iterations,
        )
        if (refinement_values[0] is None) != (refinement_values[1] is None):
            raise ValueError("Transition refinement tolerance and iteration limit must be declared together.")
        if refinement_values[0] is not None:
            tolerance = _finite_float(refinement_values[0], "transition_time_tolerance_s")
            iterations = refinement_values[1]
            if tolerance <= 0.0:
                raise ValueError("transition_time_tolerance_s must be positive.")
            if (
                isinstance(iterations, (bool, np.bool_))
                or int(iterations) != iterations
                or int(iterations) <= 0
            ):
                raise ValueError("transition_max_iterations must be a positive integer.")
            object.__setattr__(self, "transition_time_tolerance_s", tolerance)
            object.__setattr__(self, "transition_max_iterations", int(iterations))
        backend = normalize_numeric_backend(self.numeric_backend, error_message="numeric_backend must be python or rust.")
        object.__setattr__(self, "numeric_backend", backend)


@dataclass(frozen=True)
class LinkEndpointHistory:
    asset_id: str
    state_provider_id: str
    endpoint_kind: str
    times_s: np.ndarray
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    dcm_parent_from_eci: np.ndarray | None
    attitude_source_kind: str
    attitude_provider_id: str | None
    fixed_site_elevation_reference: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", _required_id(self.asset_id, "asset_id"))
        object.__setattr__(
            self,
            "state_provider_id",
            _required_id(self.state_provider_id, "state_provider_id"),
        )
        kind = str(self.endpoint_kind or "").strip().lower()
        if kind not in {"spacecraft", "fixed_wgs84_site"}:
            raise ValueError("endpoint_kind must be spacecraft or fixed_wgs84_site.")
        object.__setattr__(self, "endpoint_kind", kind)
        times, positions, velocities = _validated_history_arrays(
            self.times_s,
            self.position_eci_km,
            self.velocity_eci_km_s,
        )
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "position_eci_km", positions)
        object.__setattr__(self, "velocity_eci_km_s", velocities)
        source_kind = str(self.attitude_source_kind or "").strip().lower()
        if source_kind not in {"achieved", "replay", "analytic_ideal", "not_required"}:
            raise ValueError("Unsupported attitude_source_kind.")
        provider = None if self.attitude_provider_id is None else str(self.attitude_provider_id).strip()
        matrices = self.dcm_parent_from_eci
        if matrices is not None:
            matrices = np.asarray(matrices, dtype=float)
            if matrices.shape != (times.size, 3, 3) or not np.all(np.isfinite(matrices)):
                raise ValueError("dcm_parent_from_eci must be finite with shape (samples, 3, 3).")
            gram = np.einsum("nij,nkj->nik", matrices, matrices)
            if np.any(np.max(np.abs(gram - np.eye(3)), axis=(1, 2)) > 1.0e-10) or np.any(
                np.abs(np.linalg.det(matrices) - 1.0) > 1.0e-10
            ):
                raise ValueError("dcm_parent_from_eci must contain proper orthonormal DCMs.")
            object.__setattr__(self, "dcm_parent_from_eci", matrices)
        if not isinstance(self.fixed_site_elevation_reference, (bool, np.bool_)):
            raise ValueError("fixed_site_elevation_reference must be boolean.")
        fixed_reference = bool(self.fixed_site_elevation_reference)
        object.__setattr__(self, "fixed_site_elevation_reference", fixed_reference)
        if kind == "fixed_wgs84_site":
            if matrices is None or not fixed_reference:
                raise ValueError("Fixed WGS84 site evidence requires ENU DCM and elevation reference.")
            if source_kind != "not_required" or provider:
                raise ValueError("Fixed WGS84 sites must not declare spacecraft attitude evidence.")
        else:
            if fixed_reference:
                raise ValueError("Spacecraft evidence cannot be a fixed-site elevation reference.")
            if matrices is None and (source_kind != "not_required" or provider):
                raise ValueError("Missing spacecraft attitude requires not_required and no provider.")
            if matrices is not None and (source_kind == "not_required" or not provider):
                raise ValueError("Spacecraft attitude DCMs require a named physical or assumed provider.")
        object.__setattr__(self, "attitude_source_kind", source_kind)
        object.__setattr__(self, "attitude_provider_id", provider or None)


@dataclass(frozen=True)
class LinkSampleLedger:
    time_s: np.ndarray
    range_km: np.ndarray
    range_rate_km_s: np.ndarray
    fixed_site_elevation_rad: np.ndarray
    tx_off_axis_rad: np.ndarray
    rx_off_axis_rad: np.ndarray
    earth_clear: np.ndarray
    elevation_pass: np.ndarray
    range_pass: np.ndarray
    tx_pattern_pass: np.ndarray
    rx_pattern_pass: np.ndarray
    tx_power_dbw: np.ndarray
    tx_gain_dbi: np.ndarray
    rx_gain_dbi: np.ndarray
    free_space_path_loss_db: np.ndarray
    eirp_dbw: np.ndarray
    received_power_dbw: np.ndarray
    noise_density_dbw_hz: np.ndarray
    cn0_db_hz: np.ndarray
    eb_n0_db: np.ndarray
    margin_db: np.ndarray
    margin_pass: np.ndarray
    available: np.ndarray
    primary_reason: tuple[str, ...]


@dataclass(frozen=True)
class FreeSpaceLedger:
    tx_power_dbw: np.ndarray
    tx_gain_dbi: np.ndarray
    rx_gain_dbi: np.ndarray
    free_space_path_loss_db: np.ndarray
    eirp_dbw: np.ndarray
    received_power_dbw: np.ndarray
    noise_density_dbw_hz: np.ndarray
    cn0_db_hz: np.ndarray
    eb_n0_db: np.ndarray
    margin_db: np.ndarray
    margin_pass: np.ndarray


@dataclass(frozen=True)
class DirectedLinkResult:
    config: DirectedLinkConfig
    frame_metadata: dict[str, Any]
    samples: LinkSampleLedger
    transitions: tuple[RefinedTransition, ...]
    intervals: tuple[SampledAvailabilityInterval, ...]
    windows: tuple[LinkWindowEvidence, ...]
    summary: dict[str, Any]
    input_evidence_sha256: str
    semantic_sha256: str
    refinement_provider_id: str | None = None


@dataclass(frozen=True)
class DirectedLinkArtifacts:
    output_dir: Path
    manifest_json: Path
    summary_json: Path
    samples_csv: Path
    intervals_csv: Path
    transitions_json: Path
    evidence_packet_json: Path
    margin_plot_png: Path | None
    margin_plot_quality_json: Path | None


@dataclass(frozen=True)
class LinkWindowEvidence:
    interval_index: int
    start_s: float
    end_s: float
    duration_s: float
    start_censored: bool
    end_censored: bool
    acquisition_disposition: str
    loss_disposition: str
    acquisition_reason: str
    loss_reason: str
    minimum_margin_db: float
    mean_margin_db: float
    maximum_margin_db: float
    minimum_range_km: float
    maximum_fixed_site_elevation_rad: float
    estimated_delivered_data_bits: float


def spacecraft_endpoint_history(
    *,
    asset_id: str,
    state_provider_id: str,
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    velocities_eci_km_s: np.ndarray,
    attitudes_quat_bn: np.ndarray | None,
    attitude_source_kind: str,
    attitude_provider_id: str | None,
) -> LinkEndpointHistory:
    times, positions, velocities = _validated_history_arrays(
        times_s,
        positions_eci_km,
        velocities_eci_km_s,
    )
    source_kind = str(attitude_source_kind or "").strip().lower()
    if source_kind not in {"achieved", "replay", "analytic_ideal", "not_required"}:
        raise ValueError("Unsupported attitude_source_kind.")
    parent_dcm: np.ndarray | None = None
    provider = None if attitude_provider_id is None else str(attitude_provider_id).strip()
    if attitudes_quat_bn is not None:
        attitudes = np.asarray(attitudes_quat_bn, dtype=float)
        if attitudes.shape != (times.size, 4) or not np.all(np.isfinite(attitudes)):
            raise ValueError("attitudes_quat_bn must be finite with shape (samples, 4).")
        norms = np.linalg.norm(attitudes, axis=1)
        if np.any(np.abs(norms - 1.0) > 1.0e-10):
            raise ValueError("attitudes_quat_bn must be normalized within 1e-10.")
        if source_kind == "not_required" or not provider:
            raise ValueError("Supplied attitudes require a physical or explicitly assumed provider.")
        parent_dcm = np.asarray([quaternion_to_dcm_bn(value) for value in attitudes])
    elif source_kind != "not_required" or provider:
        raise ValueError("Missing attitudes require attitude_source_kind=not_required and no provider.")
    return LinkEndpointHistory(
        asset_id=_required_id(asset_id, "asset_id"),
        state_provider_id=_required_id(state_provider_id, "state_provider_id"),
        endpoint_kind="spacecraft",
        times_s=times,
        position_eci_km=positions,
        velocity_eci_km_s=velocities,
        dcm_parent_from_eci=parent_dcm,
        attitude_source_kind=source_kind,
        attitude_provider_id=provider or None,
    )


def fixed_wgs84_site_history(
    *,
    asset_id: str,
    state_provider_id: str,
    times_s: np.ndarray,
    geodetic_latitude_deg: float,
    longitude_deg: float,
    ellipsoidal_height_km: float,
    frame_context: FrameContext,
) -> LinkEndpointHistory:
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size < 1 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least one finite epoch.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    latitude = _finite_float(geodetic_latitude_deg, "geodetic_latitude_deg")
    longitude = _finite_float(longitude_deg, "longitude_deg")
    height = _finite_float(ellipsoidal_height_km, "ellipsoidal_height_km")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("geodetic_latitude_deg must be within [-90, 90].")
    if height < 0.0:
        raise ValueError("ellipsoidal_height_km must be nonnegative for a fixed site.")
    if frame_context.jd_utc_start is None:
        raise ValueError("A fixed WGS84 site requires an absolute UTC epoch.")
    site_ecef = geodetic_to_ecef_km(latitude, longitude, height)
    ecef_from_enu = ecef_to_enu_rotation(latitude, longitude)
    positions = np.empty((times.size, 3), dtype=float)
    velocities = np.empty_like(positions)
    parent_dcm = np.empty((times.size, 3, 3), dtype=float)
    for index, time_s in enumerate(times):
        ecef_from_eci = eci_to_ecef_rotation_context(float(time_s), frame_context)
        rotation_derivative = eci_to_ecef_rotation_derivative_context(
            float(time_s), frame_context
        )
        positions[index] = ecef_from_eci.T @ np.array(site_ecef, dtype=float)
        velocities[index] = ecef_from_eci.T @ (
            np.zeros(3) - rotation_derivative @ positions[index]
        )
        parent_dcm[index] = ecef_from_enu @ ecef_from_eci
    return LinkEndpointHistory(
        asset_id=_required_id(asset_id, "asset_id"),
        state_provider_id=_required_id(state_provider_id, "state_provider_id"),
        endpoint_kind="fixed_wgs84_site",
        times_s=times,
        position_eci_km=positions,
        velocity_eci_km_s=velocities,
        dcm_parent_from_eci=parent_dcm,
        attitude_source_kind="not_required",
        attitude_provider_id=None,
        fixed_site_elevation_reference=True,
    )


def _validated_history_arrays(
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    velocities_eci_km_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float)
    positions = np.asarray(positions_eci_km, dtype=float)
    velocities = np.asarray(velocities_eci_km_s, dtype=float)
    if times.ndim != 1 or times.size < 1 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least one finite epoch.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    if positions.shape != (times.size, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_eci_km must be finite with shape (samples, 3).")
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocities_eci_km_s must be finite with shape (samples, 3).")
    return times, positions, velocities


def _earth_occulted_segment(
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    *,
    numeric_backend: str = "rust",
) -> np.ndarray:
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown directed-link numeric backend {numeric_backend!r}.")
    if backend == "rust":
        from sim.rust_coverage_backend import earth_occulted

        return earth_occulted(tx_position, rx_position)
    axes = np.array([WGS84_A_KM, WGS84_A_KM, WGS84_B_KM], dtype=float)
    tx = tx_position / axes[None, :]
    delta = (rx_position - tx_position) / axes[None, :]
    quadratic = np.einsum("ij,ij->i", delta, delta)
    linear = 2.0 * np.einsum("ij,ij->i", tx, delta)
    constant = np.einsum("ij,ij->i", tx, tx) - 1.0
    discriminant = linear * linear - 4.0 * quadratic * constant
    tangent_or_crossing = discriminant >= -1.0e-15
    root = np.sqrt(np.clip(discriminant, 0.0, None))
    first = (-linear - root) / (2.0 * quadratic)
    second = (-linear + root) / (2.0 * quadratic)
    epsilon = 1.0e-12
    open_segment_hit = ((first > epsilon) & (first < 1.0 - epsilon)) | (
        (second > epsilon) & (second < 1.0 - epsilon)
    )
    return tangent_or_crossing & open_segment_hit


def _pattern_geometry(
    terminal: LinkTerminal,
    history: LinkEndpointHistory,
    peer_direction_eci: np.ndarray,
    *,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, np.ndarray]:
    count = peer_direction_eci.shape[0]
    if terminal.pattern.attitude_independent:
        return np.zeros(count, dtype=float), np.ones(count, dtype=bool)
    if history.dcm_parent_from_eci is None:
        raise ValueError(f"Directional terminal {terminal.terminal_id!r} requires attitude evidence.")
    if str(numeric_backend).strip().lower() == "rust":
        from sim.rust_coverage_backend import terminal_pattern

        return terminal_pattern(
            peer_direction_eci,
            history.dcm_parent_from_eci,
            terminal.quat_parent_from_terminal,
            float(terminal.pattern.half_angle_rad),
        )
    terminal_from_parent = quaternion_to_dcm_bn(
        np.asarray(terminal.quat_parent_from_terminal, dtype=float)
    ).T
    direction_terminal = np.einsum(
        "ij,njk,nk->ni",
        terminal_from_parent,
        history.dcm_parent_from_eci,
        peer_direction_eci,
    )
    off_axis = np.arccos(np.clip(direction_terminal[:, 2], -1.0, 1.0))
    passed = off_axis <= float(terminal.pattern.half_angle_rad) + _ANGULAR_TOLERANCE_RAD
    return off_axis, passed


def _primary_reasons(
    earth_clear: np.ndarray,
    elevation_pass: np.ndarray,
    range_pass: np.ndarray,
    tx_pattern_pass: np.ndarray,
    rx_pattern_pass: np.ndarray,
    margin_pass: np.ndarray,
) -> tuple[str, ...]:
    reasons = np.full(earth_clear.shape, "available", dtype=object)
    unresolved = np.ones(earth_clear.shape, dtype=bool)
    for name, gate in (
        ("earth_occulted", earth_clear),
        ("below_elevation_mask", elevation_pass),
        ("beyond_max_range", range_pass),
        ("tx_outside_pattern", tx_pattern_pass),
        ("rx_outside_pattern", rx_pattern_pass),
        ("negative_margin", margin_pass),
    ):
        failed = unresolved & (~gate)
        reasons[failed] = name
        unresolved &= gate
    return tuple(str(value) for value in reasons)


def free_space_link_ledger(
    range_km: np.ndarray,
    *,
    carrier_frequency_hz: float,
    tx_power_w: float,
    tx_gain_dbi: float | np.ndarray,
    rx_gain_dbi: float | np.ndarray,
    data_rate_bps: float,
    system_noise_temperature_k: float,
    required_eb_n0_db: float,
    tx_line_loss_db: float = 0.0,
    rx_line_loss_db: float = 0.0,
    misc_loss_db: float = 0.0,
    numeric_backend: str = "rust",
) -> FreeSpaceLedger:
    """Evaluate the one authoritative v0.1 free-space RF equation path."""

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown directed-link numeric backend {numeric_backend!r}.")
    ranges = np.asarray(range_km, dtype=float)
    if ranges.size == 0 or np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("range_km must contain positive finite values.")
    positive_inputs = {
        "carrier_frequency_hz": carrier_frequency_hz,
        "tx_power_w": tx_power_w,
        "data_rate_bps": data_rate_bps,
        "system_noise_temperature_k": system_noise_temperature_k,
    }
    normalized_positive: dict[str, float] = {}
    for field_name, raw in positive_inputs.items():
        value = _finite_float(raw, field_name)
        if value <= 0.0:
            raise ValueError(f"{field_name} must be positive.")
        normalized_positive[field_name] = value
    required = _finite_float(required_eb_n0_db, "required_eb_n0_db")
    losses: dict[str, float] = {}
    for field_name, raw in {
        "tx_line_loss_db": tx_line_loss_db,
        "rx_line_loss_db": rx_line_loss_db,
        "misc_loss_db": misc_loss_db,
    }.items():
        value = _finite_float(raw, field_name)
        if value < 0.0:
            raise ValueError(f"{field_name} must be nonnegative.")
        losses[field_name] = value
    tx_gain = np.broadcast_to(np.asarray(tx_gain_dbi, dtype=float), ranges.shape).copy()
    rx_gain = np.broadcast_to(np.asarray(rx_gain_dbi, dtype=float), ranges.shape).copy()
    if np.any(~np.isfinite(tx_gain)) or np.any(~np.isfinite(rx_gain)):
        raise ValueError("Terminal gains must be finite and broadcastable to range_km.")
    if backend == "rust":
        from sim.rust_coverage_backend import free_space_link_ledger as rust_free_space_link_ledger

        return rust_free_space_link_ledger(
            ranges,
            carrier_frequency_hz=normalized_positive["carrier_frequency_hz"],
            tx_power_w=normalized_positive["tx_power_w"],
            tx_gain_dbi=tx_gain,
            rx_gain_dbi=rx_gain,
            data_rate_bps=normalized_positive["data_rate_bps"],
            system_noise_temperature_k=normalized_positive["system_noise_temperature_k"],
            required_eb_n0_db=required,
            tx_line_loss_db=losses["tx_line_loss_db"],
            rx_line_loss_db=losses["rx_line_loss_db"],
            misc_loss_db=losses["misc_loss_db"],
        )
    tx_power_dbw = np.full(
        ranges.shape,
        10.0 * np.log10(normalized_positive["tx_power_w"]),
    )
    path_loss = 20.0 * np.log10(
        4.0
        * np.pi
        * ranges
        * 1000.0
        * normalized_positive["carrier_frequency_hz"]
        / SPEED_OF_LIGHT_M_S
    )
    eirp = tx_power_dbw + tx_gain - losses["tx_line_loss_db"]
    received = (
        eirp
        + rx_gain
        - path_loss
        - losses["rx_line_loss_db"]
        - losses["misc_loss_db"]
    )
    noise_density = np.full(
        ranges.shape,
        10.0
        * np.log10(
            BOLTZMANN_J_K * normalized_positive["system_noise_temperature_k"]
        ),
    )
    cn0 = received - noise_density
    eb_n0 = cn0 - 10.0 * np.log10(normalized_positive["data_rate_bps"])
    margin = eb_n0 - required
    return FreeSpaceLedger(
        tx_power_dbw=tx_power_dbw,
        tx_gain_dbi=tx_gain,
        rx_gain_dbi=rx_gain,
        free_space_path_loss_db=path_loss,
        eirp_dbw=eirp,
        received_power_dbw=received,
        noise_density_dbw_hz=noise_density,
        cn0_db_hz=cn0,
        eb_n0_db=eb_n0,
        margin_db=margin,
        margin_pass=margin >= -1.0e-10,
    )


def evaluate_directed_link(
    config: DirectedLinkConfig,
    *,
    tx_history: LinkEndpointHistory,
    rx_history: LinkEndpointHistory,
    frame_context: FrameContext,
    evaluator_at_time: AvailabilityEvaluator | None = None,
    refinement_provider_id: str | None = None,
) -> DirectedLinkResult:
    """Evaluate one directed same-epoch link with a complete RF term ledger."""

    if not isinstance(config, DirectedLinkConfig):
        raise ValueError("config must be a validated DirectedLinkConfig.")
    if not isinstance(tx_history, LinkEndpointHistory) or not isinstance(
        rx_history, LinkEndpointHistory
    ):
        raise ValueError("Endpoint histories must be validated LinkEndpointHistory values.")
    if tx_history.asset_id != config.tx_terminal.asset_id:
        raise ValueError("Transmitting history asset does not match tx_terminal.asset_id.")
    if rx_history.asset_id != config.rx_terminal.asset_id:
        raise ValueError("Receiving history asset does not match rx_terminal.asset_id.")
    if config.tx_terminal.parent_frame == "enu" and tx_history.endpoint_kind != "fixed_wgs84_site":
        raise ValueError("An ENU transmitting terminal requires fixed-site endpoint evidence.")
    if config.rx_terminal.parent_frame == "enu" and rx_history.endpoint_kind != "fixed_wgs84_site":
        raise ValueError("An ENU receiving terminal requires fixed-site endpoint evidence.")
    if config.tx_terminal.parent_frame == "body" and tx_history.endpoint_kind != "spacecraft":
        raise ValueError("A body-frame transmitting terminal requires spacecraft evidence.")
    if config.rx_terminal.parent_frame == "body" and rx_history.endpoint_kind != "spacecraft":
        raise ValueError("A body-frame receiving terminal requires spacecraft evidence.")
    if tx_history.endpoint_kind == "fixed_wgs84_site" and rx_history.endpoint_kind == "fixed_wgs84_site":
        raise ValueError("Directed Link Analysis v0.1 does not support fixed-site to fixed-site links.")
    if not np.array_equal(tx_history.times_s, rx_history.times_s):
        raise ValueError("Endpoint histories must use identical analysis epochs.")
    if frame_context.jd_utc_start is None:
        raise ValueError("Directed Link Analysis v0.1 requires an absolute UTC epoch.")

    times = tx_history.times_s
    axes = np.array([WGS84_A_KM, WGS84_A_KM, WGS84_B_KM])
    endpoint_positions_ecef: dict[str, np.ndarray] = {
        "Transmitting": np.empty_like(tx_history.position_eci_km),
        "Receiving": np.empty_like(rx_history.position_eci_km),
    }
    for index, time_s in enumerate(times):
        rotation = eci_to_ecef_rotation_context(float(time_s), frame_context)
        endpoint_positions_ecef["Transmitting"][index] = (
            rotation @ tx_history.position_eci_km[index]
        )
        endpoint_positions_ecef["Receiving"][index] = (
            rotation @ rx_history.position_eci_km[index]
        )
    for label, history in (("Transmitting", tx_history), ("Receiving", rx_history)):
        positions_ecef = endpoint_positions_ecef[label]
        ellipsoid_level = np.sum((positions_ecef / axes[None, :]) ** 2, axis=1)
        if history.endpoint_kind == "spacecraft" and np.any(ellipsoid_level <= 1.0):
            index = int(np.flatnonzero(ellipsoid_level <= 1.0)[0])
            raise ValueError(f"{label} spacecraft is on or inside WGS84 at sample {index}.")
    relative = rx_history.position_eci_km - tx_history.position_eci_km
    if config.numeric_backend == "rust":
        from sim.rust_coverage_backend import endpoint_kinematics

        ranges, range_rate = endpoint_kinematics(
            tx_history.position_eci_km,
            rx_history.position_eci_km,
            tx_history.velocity_eci_km_s,
            rx_history.velocity_eci_km_s,
        )
    else:
        ranges = np.linalg.norm(relative, axis=1)
        relative_velocity = rx_history.velocity_eci_km_s - tx_history.velocity_eci_km_s
        range_rate = np.einsum("ij,ij->i", relative_velocity, relative / ranges[:, None])
    if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("Endpoint range must be positive and finite at every epoch.")
    tx_to_rx = relative / ranges[:, None]
    rx_to_tx = -tx_to_rx
    tx_off_axis, tx_pattern_pass = _pattern_geometry(
        config.tx_terminal,
        tx_history,
        tx_to_rx,
        numeric_backend=config.numeric_backend,
    )
    rx_off_axis, rx_pattern_pass = _pattern_geometry(
        config.rx_terminal,
        rx_history,
        rx_to_tx,
        numeric_backend=config.numeric_backend,
    )
    earth_clear = ~_earth_occulted_segment(
        endpoint_positions_ecef["Transmitting"],
        endpoint_positions_ecef["Receiving"],
        numeric_backend=config.numeric_backend,
    )
    fixed_site_elevation = np.full(times.shape, np.nan, dtype=float)
    elevation_pass = np.ones(times.shape, dtype=bool)
    site_history: LinkEndpointHistory | None = None
    site_to_peer: np.ndarray | None = None
    if tx_history.fixed_site_elevation_reference:
        site_history, site_to_peer = tx_history, tx_to_rx
    elif rx_history.fixed_site_elevation_reference:
        site_history, site_to_peer = rx_history, rx_to_tx
    if site_history is not None and site_to_peer is not None:
        if site_history.dcm_parent_from_eci is None:
            raise ValueError("Fixed-site history is missing its ENU frame evidence.")
        enu_direction = np.einsum(
            "nij,nj->ni",
            site_history.dcm_parent_from_eci,
            site_to_peer,
        )
        fixed_site_elevation = np.arctan2(
            enu_direction[:, 2],
            np.hypot(enu_direction[:, 0], enu_direction[:, 1]),
        )
        if config.min_fixed_site_elevation_rad is not None:
            elevation_pass = (
                fixed_site_elevation
                >= config.min_fixed_site_elevation_rad - _ANGULAR_TOLERANCE_RAD
            )
    elif config.min_fixed_site_elevation_rad is not None:
        raise ValueError("min_fixed_site_elevation_rad requires a fixed-site endpoint.")
    range_pass = np.ones(times.shape, dtype=bool)
    if config.max_range_km is not None:
        range_pass = ranges <= config.max_range_km + _RANGE_TOLERANCE_KM

    rf = free_space_link_ledger(
        ranges,
        carrier_frequency_hz=config.carrier_frequency_hz,
        tx_power_w=config.tx_power_w,
        tx_gain_dbi=config.tx_terminal.pattern.gain_dbi,
        rx_gain_dbi=config.rx_terminal.pattern.gain_dbi,
        data_rate_bps=config.data_rate_bps,
        system_noise_temperature_k=config.system_noise_temperature_k,
        required_eb_n0_db=config.required_eb_n0_db,
        tx_line_loss_db=config.tx_line_loss_db,
        rx_line_loss_db=config.rx_line_loss_db,
        misc_loss_db=config.misc_loss_db,
        numeric_backend=config.numeric_backend,
    )
    margin_pass = rf.margin_pass
    available = (
        earth_clear
        & elevation_pass
        & range_pass
        & tx_pattern_pass
        & rx_pattern_pass
        & margin_pass
    )
    reasons = _primary_reasons(
        earth_clear,
        elevation_pass,
        range_pass,
        tx_pattern_pass,
        rx_pattern_pass,
        margin_pass,
    )
    samples = LinkSampleLedger(
        time_s=times,
        range_km=ranges,
        range_rate_km_s=range_rate,
        fixed_site_elevation_rad=fixed_site_elevation,
        tx_off_axis_rad=tx_off_axis,
        rx_off_axis_rad=rx_off_axis,
        earth_clear=earth_clear,
        elevation_pass=elevation_pass,
        range_pass=range_pass,
        tx_pattern_pass=tx_pattern_pass,
        rx_pattern_pass=rx_pattern_pass,
        tx_power_dbw=rf.tx_power_dbw,
        tx_gain_dbi=rf.tx_gain_dbi,
        rx_gain_dbi=rf.rx_gain_dbi,
        free_space_path_loss_db=rf.free_space_path_loss_db,
        eirp_dbw=rf.eirp_dbw,
        received_power_dbw=rf.received_power_dbw,
        noise_density_dbw_hz=rf.noise_density_dbw_hz,
        cn0_db_hz=rf.cn0_db_hz,
        eb_n0_db=rf.eb_n0_db,
        margin_db=rf.margin_db,
        margin_pass=margin_pass,
        available=available,
        primary_reason=reasons,
    )
    if evaluator_at_time is not None and config.transition_time_tolerance_s is None:
        raise ValueError("evaluator_at_time requires transition refinement configuration.")
    if evaluator_at_time is None and refinement_provider_id is not None:
        raise ValueError("refinement_provider_id requires evaluator_at_time.")
    normalized_refinement_provider_id = (
        None
        if evaluator_at_time is None
        else str(refinement_provider_id or "caller_supplied_evaluator").strip()
    )
    if times.size == 1:
        if evaluator_at_time is not None:
            raise ValueError("Scalar link evaluation does not perform transition refinement.")
        transitions = ()
        intervals = ()
    else:
        transitions = refine_availability_transitions(
            times,
            available,
            reasons,
            evaluator_at_time=evaluator_at_time,
            time_tolerance_s=config.transition_time_tolerance_s if evaluator_at_time else None,
            max_iterations=config.transition_max_iterations if evaluator_at_time else None,
            numeric_backend=config.numeric_backend,
        )
        intervals = availability_intervals(
            times,
            available,
            reasons,
            transitions=transitions,
        )
    windows = _link_windows(samples, intervals, config.data_rate_bps)
    frame_metadata = frame_context.metadata(sample_t_s=float(times[0]))
    input_hash = _input_hash(tx_history, rx_history, frame_metadata)
    semantic_hash = _semantic_hash(
        config, input_hash, samples, intervals, transitions, normalized_refinement_provider_id
    )
    summary = _link_summary(config, samples, intervals)
    summary["transition_refinement"] = {
        "enabled": evaluator_at_time is not None,
        "method": "provider_bisection" if evaluator_at_time is not None else "sample_bounded",
        "provider_id": normalized_refinement_provider_id,
        "time_tolerance_s": config.transition_time_tolerance_s,
        "max_iterations": config.transition_max_iterations,
        "transition_count": len(transitions),
    }
    return DirectedLinkResult(
        config=config,
        frame_metadata=frame_metadata,
        samples=samples,
        transitions=transitions,
        intervals=intervals,
        windows=windows,
        summary=summary,
        input_evidence_sha256=input_hash,
        semantic_sha256=semantic_hash,
        refinement_provider_id=normalized_refinement_provider_id,
    )


def _hash_arrays(identity: dict[str, Any], arrays: tuple[tuple[str, Any, str], ...]) -> str:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for name, values, dtype in arrays:
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _input_hash(
    tx: LinkEndpointHistory,
    rx: LinkEndpointHistory,
    frame_metadata: dict[str, Any],
) -> str:
    arrays: list[tuple[str, Any, str]] = [
        ("times_s", tx.times_s, "<f8"),
        ("tx_position_eci_km", tx.position_eci_km, "<f8"),
        ("tx_velocity_eci_km_s", tx.velocity_eci_km_s, "<f8"),
        ("rx_position_eci_km", rx.position_eci_km, "<f8"),
        ("rx_velocity_eci_km_s", rx.velocity_eci_km_s, "<f8"),
    ]
    if tx.dcm_parent_from_eci is not None:
        arrays.append(("tx_dcm_parent_from_eci", tx.dcm_parent_from_eci, "<f8"))
    if rx.dcm_parent_from_eci is not None:
        arrays.append(("rx_dcm_parent_from_eci", rx.dcm_parent_from_eci, "<f8"))
    identity = {
        "schema": "oel.directed-link-input-evidence.v1",
        "frame": frame_metadata,
        "tx_asset_id": tx.asset_id,
        "tx_endpoint_kind": tx.endpoint_kind,
        "tx_state_provider_id": tx.state_provider_id,
        "tx_attitude_source_kind": tx.attitude_source_kind,
        "tx_attitude_provider_id": tx.attitude_provider_id,
        "tx_fixed_site_elevation_reference": tx.fixed_site_elevation_reference,
        "rx_asset_id": rx.asset_id,
        "rx_endpoint_kind": rx.endpoint_kind,
        "rx_state_provider_id": rx.state_provider_id,
        "rx_attitude_source_kind": rx.attitude_source_kind,
        "rx_attitude_provider_id": rx.attitude_provider_id,
        "rx_fixed_site_elevation_reference": rx.fixed_site_elevation_reference,
    }
    return _hash_arrays(identity, tuple(arrays))


def _normalized_config(config: DirectedLinkConfig) -> dict[str, Any]:
    record = asdict(config)
    record["contract_version"] = DIRECTED_LINK_CONTRACT_VERSION
    return record


def recompute_directed_link_semantic_sha256(
    *,
    normalized_config: Mapping[str, Any],
    input_evidence_sha256: str,
    time_s: Sequence[float] | np.ndarray,
    range_km: Sequence[float] | np.ndarray,
    margin_db: Sequence[float] | np.ndarray,
    available: Sequence[bool] | np.ndarray,
    primary_reason: Sequence[str],
    intervals: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    refinement_provider_id: str | None,
) -> str:
    """Rebuild the directed-link semantic identity from retained evidence."""

    config = dict(normalized_config)
    if config.get("contract_version") != DIRECTED_LINK_CONTRACT_VERSION:
        raise ValueError("Unsupported normalized directed-link contract version.")
    input_digest = str(input_evidence_sha256 or "").strip().lower()
    if len(input_digest) != 64 or any(character not in "0123456789abcdef" for character in input_digest):
        raise ValueError("input_evidence_sha256 must be a lowercase SHA-256 digest.")
    times = np.asarray(time_s, dtype=float)
    ranges = np.asarray(range_km, dtype=float)
    margins = np.asarray(margin_db, dtype=float)
    mask = np.asarray(available, dtype=bool)
    reasons = tuple(str(value) for value in primary_reason)
    if (
        times.ndim != 1
        or times.size == 0
        or ranges.shape != times.shape
        or margins.shape != times.shape
        or mask.shape != times.shape
        or len(reasons) != times.size
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(ranges))
        or np.any(~np.isfinite(margins))
    ):
        raise ValueError("Retained directed-link semantic sample columns are inconsistent.")
    try:
        reason_codes = np.asarray([LINK_REASON_NAMES.index(value) for value in reasons])
        interval_rows = np.asarray(
            [
                [float(value["start_s"]), float(value["end_s"]), float(value["duration_s"])]
                for value in intervals
            ],
            dtype=float,
        ).reshape(-1, 3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Retained directed-link intervals or reason codes are invalid.") from exc
    if np.any(~np.isfinite(interval_rows)):
        raise ValueError("Retained directed-link interval rows must be finite.")
    return _hash_arrays(
        {
            **config,
            "input_evidence_sha256": input_digest,
            "refinement_provider_id": refinement_provider_id,
            "transitions": [dict(value) for value in transitions],
        },
        (
            ("time_s", times, "<f8"),
            ("range_km", ranges, "<f8"),
            ("margin_db", margins, "<f8"),
            ("available", mask, "|u1"),
            ("primary_reason_code", reason_codes, "|u1"),
            ("interval_rows", interval_rows, "<f8"),
        ),
    )


def _semantic_hash(
    config: DirectedLinkConfig,
    input_hash: str,
    samples: LinkSampleLedger,
    intervals: tuple[SampledAvailabilityInterval, ...],
    transitions: tuple[RefinedTransition, ...],
    refinement_provider_id: str | None = None,
) -> str:
    return recompute_directed_link_semantic_sha256(
        normalized_config=_normalized_config(config),
        input_evidence_sha256=input_hash,
        time_s=samples.time_s,
        range_km=samples.range_km,
        margin_db=samples.margin_db,
        available=samples.available,
        primary_reason=samples.primary_reason,
        intervals=[asdict(value) for value in intervals],
        transitions=[asdict(value) for value in transitions],
        refinement_provider_id=refinement_provider_id,
    )


def _link_summary(
    config: DirectedLinkConfig,
    samples: LinkSampleLedger,
    intervals: tuple[SampledAvailabilityInterval, ...],
) -> dict[str, Any]:
    duration = float(samples.time_s[-1] - samples.time_s[0])
    available_duration = float(sum(interval.duration_s for interval in intervals))
    reason_counts = {name: samples.primary_reason.count(name) for name in LINK_REASON_NAMES}
    return {
        "contract_version": DIRECTED_LINK_CONTRACT_VERSION,
        "analysis_id": config.analysis_id,
        "link_id": config.link_id,
        "status": "complete",
        "direction": {
            "tx_asset_id": config.tx_terminal.asset_id,
            "tx_terminal_id": config.tx_terminal.terminal_id,
            "rx_asset_id": config.rx_terminal.asset_id,
            "rx_terminal_id": config.rx_terminal.terminal_id,
        },
        "sample_count": int(samples.time_s.size),
        "horizon_start_s": float(samples.time_s[0]),
        "horizon_end_s": float(samples.time_s[-1]),
        "horizon_duration_s": duration,
        "available_sample_count": int(np.count_nonzero(samples.available)),
        "available_duration_s": available_duration,
        "sampled_available_fraction": (
            float(samples.available[0]) if duration == 0.0 else available_duration / duration
        ),
        "interval_count": len(intervals),
        "required_eb_n0_db": config.required_eb_n0_db,
        "minimum_fixed_site_elevation_deg": (
            None
            if config.min_fixed_site_elevation_rad is None
            else float(np.rad2deg(config.min_fixed_site_elevation_rad))
        ),
        "maximum_range_km_threshold": config.max_range_km,
        "estimated_delivered_data_bits": available_duration * config.data_rate_bps,
        "margin_db": {
            "minimum": float(np.min(samples.margin_db)),
            "mean": float(np.mean(samples.margin_db)),
            "maximum": float(np.max(samples.margin_db)),
        },
        "range_km": {
            "minimum": float(np.min(samples.range_km)),
            "maximum": float(np.max(samples.range_km)),
        },
        "primary_reason_sample_count": reason_counts,
        "claim_limits": [
            "Same-epoch one-way free-space engineering feasibility only.",
            "No atmosphere, weather, interference, polarization, terrain, protocols, or hardware assurance.",
            "Range rate is retained as geometry evidence and is not a Doppler claim.",
            "Sampled windows do not imply between-sample continuity unless provider-refined.",
        ],
    }


def _link_windows(
    samples: LinkSampleLedger,
    intervals: tuple[SampledAvailabilityInterval, ...],
    data_rate_bps: float,
) -> tuple[LinkWindowEvidence, ...]:
    windows: list[LinkWindowEvidence] = []
    for interval in intervals:
        selected = (
            (samples.time_s >= interval.start_s - 1.0e-12)
            & (samples.time_s < interval.end_s - 1.0e-12)
            & samples.available
        )
        if interval.end_censored:
            selected |= (
                (samples.time_s >= interval.start_s - 1.0e-12)
                & (samples.time_s <= interval.end_s + 1.0e-12)
                & samples.available
            )
        if not np.any(selected):
            available_indices = np.flatnonzero(samples.available)
            if not available_indices.size:
                raise RuntimeError("Link interval has no supporting available sample.")
            nearest = available_indices[
                np.argmin(
                    np.abs(
                        samples.time_s[available_indices]
                        - 0.5 * (interval.start_s + interval.end_s)
                    )
                )
            ]
            selected[int(nearest)] = True
        elevation = samples.fixed_site_elevation_rad[selected]
        finite_elevation = elevation[np.isfinite(elevation)]
        windows.append(
            LinkWindowEvidence(
                interval_index=interval.interval_index,
                start_s=interval.start_s,
                end_s=interval.end_s,
                duration_s=interval.duration_s,
                start_censored=interval.start_censored,
                end_censored=interval.end_censored,
                acquisition_disposition=interval.acquisition_disposition,
                loss_disposition=interval.loss_disposition,
                acquisition_reason=interval.acquisition_reason,
                loss_reason=interval.loss_reason,
                minimum_margin_db=float(np.min(samples.margin_db[selected])),
                mean_margin_db=float(np.mean(samples.margin_db[selected])),
                maximum_margin_db=float(np.max(samples.margin_db[selected])),
                minimum_range_km=float(np.min(samples.range_km[selected])),
                maximum_fixed_site_elevation_rad=(
                    float("nan")
                    if not finite_elevation.size
                    else float(np.max(finite_elevation))
                ),
                estimated_delivered_data_bits=interval.duration_s * data_rate_bps,
            )
        )
    return tuple(windows)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_directed_link_artifacts(
    result: DirectedLinkResult,
    output_dir: str | Path,
    *,
    include_margin_plot: bool = False,
    plot_scenario_name: str = "",
    orbit_binding: Mapping[str, Any] | None = None,
) -> DirectedLinkArtifacts:
    """Write deterministic link samples, windows, summary, and evidence packet."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Directed-link output directory already exists: {destination}")
    destination.mkdir(parents=True)
    summary_path = destination / "link_summary.json"
    samples_path = destination / "link_samples.csv"
    intervals_path = destination / "link_intervals.csv"
    transitions_path = destination / "link_transitions.json"
    packet_path = destination / "link_evidence_packet.json"
    plot_path = destination / "link_margin.png" if include_margin_plot else None
    plot_quality_path = destination / "link_margin.quality.json" if include_margin_plot else None
    manifest_path = destination / "link_analysis_manifest.json"
    _json_dump(summary_path, result.summary)
    samples = result.samples
    sample_fields = (
        "time_s",
        "range_km",
        "range_rate_km_s",
        "fixed_site_elevation_rad",
        "tx_off_axis_rad",
        "rx_off_axis_rad",
        "earth_clear",
        "elevation_pass",
        "range_pass",
        "tx_pattern_pass",
        "rx_pattern_pass",
        "tx_power_dbw",
        "tx_gain_dbi",
        "rx_gain_dbi",
        "free_space_path_loss_db",
        "eirp_dbw",
        "received_power_dbw",
        "noise_density_dbw_hz",
        "cn0_db_hz",
        "eb_n0_db",
        "margin_db",
        "margin_pass",
        "available",
        "primary_reason",
    )
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(sample_fields)
        for index in range(samples.time_s.size):
            row: list[Any] = []
            for field_name in sample_fields:
                value = getattr(samples, field_name)
                item = value[index]
                if isinstance(item, (bool, np.bool_)):
                    row.append(str(bool(item)).lower())
                elif isinstance(item, (float, np.floating)):
                    row.append("" if not np.isfinite(item) else f"{float(item):.17g}")
                else:
                    row.append(item)
            writer.writerow(row)
    interval_fields = tuple(LinkWindowEvidence.__dataclass_fields__)
    with intervals_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=interval_fields, lineterminator="\n")
        writer.writeheader()
        for window in result.windows:
            row = asdict(window)
            if not np.isfinite(row["maximum_fixed_site_elevation_rad"]):
                row["maximum_fixed_site_elevation_rad"] = ""
            writer.writerow(row)
    _json_dump(
        transitions_path,
        {
            "schema": "oel.directed-link-transitions.v1",
            "analysis_id": result.config.analysis_id,
            "link_id": result.config.link_id,
            "transitions": [asdict(value) for value in result.transitions],
        },
    )
    packet_payload = {
            "schema": "oel.directed-link-evidence-packet.v1",
            "analysis_id": result.config.analysis_id,
            "link_id": result.config.link_id,
            "semantic_sha256": result.semantic_sha256,
            "input_evidence_sha256": result.input_evidence_sha256,
            "summary": result.summary,
            "citations": [
                {"artifact": samples_path.name, "rows": int(samples.time_s.size)},
                {"artifact": intervals_path.name, "rows": len(result.windows)},
                {"artifact": transitions_path.name, "rows": len(result.transitions)},
            ],
        }
    if orbit_binding is not None:
        packet_payload["orbit_binding"] = dict(orbit_binding)
    _json_dump(packet_path, packet_payload)
    if plot_path is not None:
        from sim.analysis.link_plotting import write_link_margin_plot

        write_link_margin_plot(result, plot_path, scenario_name=plot_scenario_name)
    artifact_paths = [
        summary_path,
        samples_path,
        intervals_path,
        transitions_path,
        packet_path,
    ]
    if plot_path is not None:
        artifact_paths.append(plot_path)
    if plot_quality_path is not None:
        artifact_paths.append(plot_quality_path)
    manifest_payload = {
            "contract_version": DIRECTED_LINK_CONTRACT_VERSION,
            "analysis_id": result.config.analysis_id,
            "link_id": result.config.link_id,
            "status": "complete",
            "normalized_config": _normalized_config(result.config),
            "frame": result.frame_metadata,
            "input_evidence_sha256": result.input_evidence_sha256,
            "semantic_sha256": result.semantic_sha256,
            "artifacts": {
                path.name: {"sha256": _file_hash(path)} for path in artifact_paths
            },
            "claim_limits": result.summary["claim_limits"],
            "refinement_provider_id": result.refinement_provider_id,
        }
    if orbit_binding is not None:
        manifest_payload["orbit_binding"] = dict(orbit_binding)
    _json_dump(manifest_path, manifest_payload)
    return DirectedLinkArtifacts(
        output_dir=destination,
        manifest_json=manifest_path,
        summary_json=summary_path,
        samples_csv=samples_path,
        intervals_csv=intervals_path,
        transitions_json=transitions_path,
        evidence_packet_json=packet_path,
        margin_plot_png=plot_path,
        margin_plot_quality_json=plot_quality_path,
    )


def evaluate_directed_link_sample(
    config: DirectedLinkConfig,
    *,
    tx_history: LinkEndpointHistory,
    rx_history: LinkEndpointHistory,
    frame_context: FrameContext,
) -> DirectedLinkResult:
    """Evaluate exactly one runtime sample through the authoritative batch path."""

    if tx_history.times_s.size != 1 or rx_history.times_s.size != 1:
        raise ValueError("Scalar link evaluation requires one-sample endpoint histories.")
    return evaluate_directed_link(
        config,
        tx_history=tx_history,
        rx_history=rx_history,
        frame_context=frame_context,
    )


__all__ = [
    "BOLTZMANN_J_K",
    "DIRECTED_LINK_CONTRACT_VERSION",
    "LINK_REASON_NAMES",
    "SPEED_OF_LIGHT_M_S",
    "DirectedLinkArtifacts",
    "DirectedLinkConfig",
    "DirectedLinkResult",
    "FreeSpaceLedger",
    "LinkEndpointHistory",
    "LinkSampleLedger",
    "LinkTerminal",
    "LinkWindowEvidence",
    "TerminalPattern",
    "evaluate_directed_link",
    "evaluate_directed_link_sample",
    "free_space_link_ledger",
    "fixed_wgs84_site_history",
    "recompute_directed_link_semantic_sha256",
    "spacecraft_endpoint_history",
    "write_directed_link_artifacts",
]
