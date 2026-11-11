"""Authoritative Pro RF ledger layered over OEL's public free-space evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.analysis.directed_link import SPEED_OF_LIGHT_M_S, free_space_link_ledger
from sim.analysis.event_refinement import SampledAvailabilityInterval, availability_intervals
from sim.pro_communications.atmosphere import AtmosphericLossLedger, evaluate_atmospheric_losses
from sim.pro_communications.contracts import (
    PRO_COMMUNICATIONS_SCHEMA_VERSION,
    RF_CONVENTIONS_VERSION,
    LinkBudgetProfile,
    ResourceLimits,
)

PRO_LINK_LEDGER_VERSION = "oel.pro-link-ledger.v0.2"
_BOLTZMANN_J_K = 1.380649e-23


@dataclass(frozen=True)
class ProLinkEnvironmentEvidence:
    """Optional content-identified terrain and bounded external-interference histories."""

    source_id: str
    terrain_clear: np.ndarray | None = None
    interference_temperature_k: np.ndarray | None = None
    interference_power_dbw: np.ndarray | None = None
    interference_bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        source = str(self.source_id or "").strip()
        if not source:
            raise ValueError("source_id must be a non-empty string.")
        object.__setattr__(self, "source_id", source)
        if self.terrain_clear is not None:
            terrain = _boolean_values(self.terrain_clear, "terrain_clear")
            if terrain.ndim != 1 or terrain.size < 2:
                raise ValueError("terrain_clear must be a one-dimensional boolean history.")
            object.__setattr__(self, "terrain_clear", terrain)
        if self.interference_temperature_k is not None:
            temperature = np.asarray(self.interference_temperature_k, dtype=float)
            if temperature.ndim != 1 or temperature.size < 2 or np.any(~np.isfinite(temperature)):
                raise ValueError("interference_temperature_k must be a finite one-dimensional history.")
            if np.any(temperature < 0.0):
                raise ValueError("interference_temperature_k must be nonnegative.")
            object.__setattr__(self, "interference_temperature_k", temperature)
        if self.interference_power_dbw is not None:
            power = np.asarray(self.interference_power_dbw, dtype=float)
            if power.ndim != 1 or power.size < 2 or np.any(~np.isfinite(power)):
                raise ValueError("interference_power_dbw must be a finite one-dimensional history.")
            object.__setattr__(self, "interference_power_dbw", power)
            if self.interference_bandwidth_hz is None:
                raise ValueError("interference_power_dbw requires interference_bandwidth_hz.")
        if self.interference_bandwidth_hz is not None:
            bandwidth = float(self.interference_bandwidth_hz)
            if not np.isfinite(bandwidth) or bandwidth <= 0.0:
                raise ValueError("interference_bandwidth_hz must be positive and finite.")
            if self.interference_power_dbw is None:
                raise ValueError("interference_bandwidth_hz is valid only with interference_power_dbw.")
            object.__setattr__(self, "interference_bandwidth_hz", bandwidth)


@dataclass(frozen=True)
class ProLinkSampleLedger:
    time_s: np.ndarray
    range_km: np.ndarray
    range_rate_km_s: np.ndarray
    elevation_rad: np.ndarray
    tx_off_axis_rad: np.ndarray
    rx_off_axis_rad: np.ndarray
    tx_gain_dbi: np.ndarray
    rx_gain_dbi: np.ndarray
    tx_pattern_pass: np.ndarray
    rx_pattern_pass: np.ndarray
    earth_clear: np.ndarray
    elevation_pass: np.ndarray
    range_pass: np.ndarray
    terrain_clear: np.ndarray
    atmospheric_applicability_pass: np.ndarray
    free_space_path_loss_db: np.ndarray
    gaseous_loss_db: np.ndarray
    rain_loss_db: np.ndarray
    cloud_loss_db: np.ndarray
    polarization_loss_db: np.ndarray
    pointing_loss_db: np.ndarray
    implementation_loss_db: np.ndarray
    legacy_misc_loss_db: np.ndarray
    total_named_propagation_loss_db: np.ndarray
    tx_power_dbw: np.ndarray
    eirp_dbw: np.ndarray
    received_power_dbw: np.ndarray
    noise_density_dbw_hz: np.ndarray
    external_interference_density_dbw_hz: np.ndarray
    effective_noise_density_dbw_hz: np.ndarray
    cn0_db_hz: np.ndarray
    eb_n0_db: np.ndarray
    required_eb_n0_db: np.ndarray
    margin_db: np.ndarray
    margin_pass: np.ndarray
    acquisition_threshold_pass: np.ndarray
    tracking_threshold_pass: np.ndarray
    receiver_locked: np.ndarray
    selected_data_rate_bps: np.ndarray
    selected_mode_id: tuple[str, ...]
    available: np.ndarray
    primary_reason: tuple[str, ...]
    doppler_shift_hz: np.ndarray
    received_frequency_hz: np.ndarray


@dataclass(frozen=True)
class ProLinkBudgetResult:
    profile: LinkBudgetProfile
    samples: ProLinkSampleLedger
    intervals: tuple[SampledAvailabilityInterval, ...]
    atmospheric_evidence: AtmosphericLossLedger
    summary: dict[str, Any]
    semantic_sha256: str


@dataclass(frozen=True)
class ProRfClosureLedger:
    tx_gain_dbi: np.ndarray
    rx_gain_dbi: np.ndarray
    tx_pattern_pass: np.ndarray
    rx_pattern_pass: np.ndarray
    earth_clear: np.ndarray
    elevation_pass: np.ndarray
    range_pass: np.ndarray
    terrain_clear: np.ndarray
    atmospheric_applicability_pass: np.ndarray
    free_space_path_loss_db: np.ndarray
    gaseous_loss_db: np.ndarray
    rain_loss_db: np.ndarray
    cloud_loss_db: np.ndarray
    total_named_propagation_loss_db: np.ndarray
    tx_power_dbw: np.ndarray
    eirp_dbw: np.ndarray
    received_power_dbw: np.ndarray
    noise_density_dbw_hz: np.ndarray
    external_interference_density_dbw_hz: np.ndarray
    effective_noise_density_dbw_hz: np.ndarray
    cn0_db_hz: np.ndarray
    eb_n0_db: np.ndarray
    margin_db: np.ndarray
    margin_pass: np.ndarray
    selected_data_rate_bps: np.ndarray
    selected_mode_id: np.ndarray
    available: np.ndarray
    primary_reason: np.ndarray
    atmospheric_evidence: AtmosphericLossLedger


def evaluate_pro_rf_closure(
    profile: LinkBudgetProfile,
    *,
    range_km: np.ndarray,
    elevation_rad: np.ndarray,
    tx_off_axis_rad: np.ndarray | float = 0.0,
    rx_off_axis_rad: np.ndarray | float = 0.0,
    earth_clear: np.ndarray | bool = True,
    terrain_clear: np.ndarray | bool = True,
    interference_temperature_k: np.ndarray | float = 0.0,
    interference_power_dbw: np.ndarray | float | None = None,
    interference_bandwidth_hz: float | None = None,
    minimum_elevation_rad: float = np.deg2rad(5.0),
    maximum_range_km: float | None = None,
    numeric_backend: str = "rust",
) -> ProRfClosureLedger:
    """Evaluate the authoritative Pro RF closure equations for any broadcast shape."""

    if not isinstance(profile, LinkBudgetProfile):
        raise ValueError("profile must be a LinkBudgetProfile.")
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown communications numeric backend {numeric_backend!r}.")
    base_values = [
        np.asarray(range_km, dtype=float),
        np.asarray(elevation_rad, dtype=float),
        np.asarray(tx_off_axis_rad, dtype=float),
        np.asarray(rx_off_axis_rad, dtype=float),
        _boolean_values(earth_clear, "earth_clear"),
        _boolean_values(terrain_clear, "terrain_clear"),
        np.asarray(interference_temperature_k, dtype=float),
    ]
    if interference_power_dbw is None:
        ranges, elevation, tx_angle, rx_angle, clear, terrain, interference_temperature = np.broadcast_arrays(
            *base_values
        )
        interference_power = None
    else:
        (
            ranges,
            elevation,
            tx_angle,
            rx_angle,
            clear,
            terrain,
            interference_temperature,
            interference_power,
        ) = np.broadcast_arrays(*base_values, np.asarray(interference_power_dbw, dtype=float))
    if ranges.size == 0 or np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("range_km must contain positive finite values.")
    if np.any(~np.isfinite(elevation)) or np.any((elevation < -0.5 * np.pi) | (elevation > 0.5 * np.pi)):
        raise ValueError("elevation_rad must be finite within [-pi/2, pi/2].")
    if np.any(~np.isfinite(tx_angle)) or np.any(~np.isfinite(rx_angle)):
        raise ValueError("Terminal off-axis angles must be finite.")
    if np.any((tx_angle < 0.0) | (tx_angle > np.pi)) or np.any((rx_angle < 0.0) | (rx_angle > np.pi)):
        raise ValueError("Terminal off-axis angles must be within [0, pi].")
    if np.any(~np.isfinite(interference_temperature)) or np.any(interference_temperature < 0.0):
        raise ValueError("interference_temperature_k must be finite and nonnegative.")
    if interference_power is not None:
        if np.any(~np.isfinite(interference_power)):
            raise ValueError("interference_power_dbw must be finite.")
        bandwidth = float(interference_bandwidth_hz) if interference_bandwidth_hz is not None else np.nan
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("interference_power_dbw requires positive finite interference_bandwidth_hz.")
    elif interference_bandwidth_hz is not None:
        raise ValueError("interference_bandwidth_hz is valid only with interference_power_dbw.")
    minimum_elevation = float(minimum_elevation_rad)
    if not np.isfinite(minimum_elevation) or not -0.5 * np.pi <= minimum_elevation <= 0.5 * np.pi:
        raise ValueError("minimum_elevation_rad must be finite within [-pi/2, pi/2].")

    tx_gain, tx_pattern_pass = profile.transmitter.antenna.evaluate(np.rad2deg(tx_angle))
    rx_gain, rx_pattern_pass = profile.receiver.antenna.evaluate(np.rad2deg(rx_angle))
    elevation_pass = elevation >= minimum_elevation - 1.0e-12
    if maximum_range_km is None:
        range_pass = np.ones(ranges.shape, dtype=bool)
    else:
        maximum_range = float(maximum_range_km)
        if not np.isfinite(maximum_range) or maximum_range <= 0.0:
            raise ValueError("maximum_range_km must be positive and finite.")
        range_pass = ranges <= maximum_range + 1.0e-9

    public_ledger = free_space_link_ledger(
        ranges,
        carrier_frequency_hz=profile.carrier_frequency_hz,
        tx_power_w=profile.transmitter.power_w,
        tx_gain_dbi=tx_gain,
        rx_gain_dbi=rx_gain,
        data_rate_bps=profile.waveform.data_rate_bps,
        system_noise_temperature_k=profile.receiver.resolved_system_noise_temperature_k,
        required_eb_n0_db=profile.waveform.required_eb_n0_db,
        tx_line_loss_db=profile.transmitter.line_loss_db,
        rx_line_loss_db=profile.receiver.line_loss_db,
        misc_loss_db=profile.legacy_misc_loss_db,
        numeric_backend=backend,
    )

    applicability = np.ones(ranges.shape, dtype=bool)
    if any(
        model != "none"
        for model in (
            profile.atmosphere.gaseous_model,
            profile.atmosphere.rain_model,
            profile.atmosphere.cloud_model,
        )
    ):
        applicability = elevation >= np.deg2rad(5.0) - 1.0e-12
    gaseous = np.full(ranges.shape, np.nan, dtype=float)
    rain = np.full(ranges.shape, np.nan, dtype=float)
    cloud = np.full(ranges.shape, np.nan, dtype=float)
    editions: tuple[str, ...] = ()
    statistical_components: tuple[str, ...] = ()
    if np.any(applicability):
        evaluated = evaluate_atmospheric_losses(
            profile.atmosphere,
            carrier_frequency_hz=profile.carrier_frequency_hz,
            elevation_rad=elevation[applicability],
            numeric_backend=backend,
        )
        gaseous[applicability] = evaluated.gaseous_db
        rain[applicability] = evaluated.rain_db
        cloud[applicability] = evaluated.cloud_db
        editions = evaluated.model_editions
        statistical_components = evaluated.statistical_components
    atmospheric = AtmosphericLossLedger(
        gaseous_db=gaseous,
        rain_db=rain,
        cloud_db=cloud,
        total_db=gaseous + rain + cloud,
        statistical_components=statistical_components,
        model_editions=editions,
    )
    fixed_loss = profile.polarization_loss_db + profile.pointing_loss_db + profile.implementation_loss_db
    total_additional = atmospheric.total_db + fixed_loss
    received = public_ledger.received_power_dbw - total_additional
    thermal_density_w_hz = _BOLTZMANN_J_K * profile.receiver.resolved_system_noise_temperature_k
    external_density_w_hz = _BOLTZMANN_J_K * interference_temperature
    if interference_power is not None:
        external_density_w_hz = external_density_w_hz + 10.0 ** (interference_power / 10.0) / bandwidth
    effective_density_w_hz = thermal_density_w_hz + external_density_w_hz
    external_density_dbw_hz = np.full(ranges.shape, -np.inf, dtype=float)
    positive_external = external_density_w_hz > 0.0
    external_density_dbw_hz[positive_external] = 10.0 * np.log10(external_density_w_hz[positive_external])
    effective_density_dbw_hz = 10.0 * np.log10(effective_density_w_hz)
    cn0 = received - effective_density_dbw_hz
    eb_n0 = cn0 - 10.0 * np.log10(profile.waveform.data_rate_bps)
    margin = eb_n0 - profile.waveform.required_eb_n0_db
    margin_pass = margin >= -1.0e-10
    selected_rate = profile.waveform.selected_rate_bps(eb_n0)
    if profile.waveform.rate_policy == "fixed":
        selected_mode = np.where(selected_rate > 0.0, profile.waveform.waveform_id, "")
        data_rate_reason = "negative_margin"
    else:
        selected_mode = np.full(ranges.shape, "", dtype=object)
        for mode in profile.waveform.adaptive_modes:
            selected_mode = np.where(eb_n0 >= mode.required_eb_n0_db - 1.0e-10, mode.mode_id, selected_mode)
        data_rate_reason = "no_supported_waveform_mode"
    data_rate_pass = selected_rate > 0.0
    available = (
        clear
        & terrain
        & elevation_pass
        & range_pass
        & tx_pattern_pass
        & rx_pattern_pass
        & applicability
        & data_rate_pass
    )
    reasons = np.full(ranges.shape, "available", dtype=object)
    unresolved = np.ones(ranges.shape, dtype=bool)
    for reason, gate in (
        ("earth_occulted", clear),
        ("terrain_blocked", terrain),
        ("below_elevation_mask", elevation_pass),
        ("beyond_max_range", range_pass),
        ("tx_outside_pattern", tx_pattern_pass),
        ("rx_outside_pattern", rx_pattern_pass),
        ("atmospheric_model_outside_applicability", applicability),
        (data_rate_reason, data_rate_pass),
    ):
        failed = unresolved & ~gate
        reasons[failed] = reason
        unresolved &= gate
    return ProRfClosureLedger(
        tx_gain_dbi=tx_gain,
        rx_gain_dbi=rx_gain,
        tx_pattern_pass=tx_pattern_pass,
        rx_pattern_pass=rx_pattern_pass,
        earth_clear=clear,
        elevation_pass=elevation_pass,
        range_pass=range_pass,
        terrain_clear=terrain,
        atmospheric_applicability_pass=applicability,
        free_space_path_loss_db=public_ledger.free_space_path_loss_db,
        gaseous_loss_db=gaseous,
        rain_loss_db=rain,
        cloud_loss_db=cloud,
        total_named_propagation_loss_db=atmospheric.total_db,
        tx_power_dbw=public_ledger.tx_power_dbw,
        eirp_dbw=public_ledger.eirp_dbw,
        received_power_dbw=received,
        noise_density_dbw_hz=public_ledger.noise_density_dbw_hz,
        external_interference_density_dbw_hz=external_density_dbw_hz,
        effective_noise_density_dbw_hz=effective_density_dbw_hz,
        cn0_db_hz=cn0,
        eb_n0_db=eb_n0,
        margin_db=margin,
        margin_pass=margin_pass,
        selected_data_rate_bps=selected_rate,
        selected_mode_id=selected_mode,
        available=available,
        primary_reason=reasons,
        atmospheric_evidence=atmospheric,
    )


def _one_dimensional(values: Any, name: str, *, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a one-dimensional finite array with at least two samples.")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must contain {length} samples.")
    return array


def _boolean_values(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain boolean values or numeric 0/1 only.")
    return array.astype(bool)


def _boolean(values: Any, name: str, length: int) -> np.ndarray:
    array = _boolean_values(values, name)
    if array.shape != (length,):
        raise ValueError(f"{name} must contain {length} boolean samples.")
    return array


@dataclass(frozen=True)
class _PreparedProLinkInputs:
    times: np.ndarray
    ranges: np.ndarray
    elevation: np.ndarray
    tx_angle: np.ndarray
    rx_angle: np.ndarray
    rates: np.ndarray
    clear: np.ndarray
    minimum_elevation: float


def _prepare_pro_link_inputs(
    *,
    times_s: np.ndarray,
    range_km: np.ndarray,
    elevation_rad: np.ndarray,
    tx_off_axis_rad: np.ndarray | None,
    rx_off_axis_rad: np.ndarray | None,
    range_rate_km_s: np.ndarray | None,
    earth_clear: np.ndarray | None,
    minimum_elevation_rad: float,
) -> _PreparedProLinkInputs:
    times = _one_dimensional(times_s, "times_s")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    count = times.size
    ranges = _one_dimensional(range_km, "range_km", length=count)
    if np.any(ranges <= 0.0):
        raise ValueError("range_km must contain positive values.")
    elevation = _one_dimensional(elevation_rad, "elevation_rad", length=count)
    if np.any((elevation < -0.5 * np.pi) | (elevation > 0.5 * np.pi)):
        raise ValueError("elevation_rad must be within [-pi/2, pi/2].")
    minimum_elevation = float(minimum_elevation_rad)
    if not np.isfinite(minimum_elevation) or not -0.5 * np.pi <= minimum_elevation <= 0.5 * np.pi:
        raise ValueError("minimum_elevation_rad must be finite within [-pi/2, pi/2].")
    tx_angle = (
        np.zeros(count, dtype=float)
        if tx_off_axis_rad is None
        else _one_dimensional(tx_off_axis_rad, "tx_off_axis_rad", length=count)
    )
    rx_angle = (
        np.zeros(count, dtype=float)
        if rx_off_axis_rad is None
        else _one_dimensional(rx_off_axis_rad, "rx_off_axis_rad", length=count)
    )
    if np.any((tx_angle < 0.0) | (tx_angle > np.pi)) or np.any((rx_angle < 0.0) | (rx_angle > np.pi)):
        raise ValueError("Terminal off-axis angles must be within [0, pi].")
    rates = (
        np.zeros(count, dtype=float)
        if range_rate_km_s is None
        else _one_dimensional(range_rate_km_s, "range_rate_km_s", length=count)
    )
    clear = np.ones(count, dtype=bool) if earth_clear is None else _boolean(earth_clear, "earth_clear", count)
    return _PreparedProLinkInputs(
        times=times,
        ranges=ranges,
        elevation=elevation,
        tx_angle=tx_angle,
        rx_angle=rx_angle,
        rates=rates,
        clear=clear,
        minimum_elevation=minimum_elevation,
    )


def _semantic_hash(
    profile_sha256: str,
    samples: ProLinkSampleLedger,
    environment_source_id: str | None,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "contract_version": PRO_LINK_LEDGER_VERSION,
                "schema_version": PRO_COMMUNICATIONS_SCHEMA_VERSION,
                "rf_conventions_version": RF_CONVENTIONS_VERSION,
                "profile_sha256": profile_sha256,
                "environment_source_id": environment_source_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for name in (
        "time_s",
        "range_km",
        "range_rate_km_s",
        "elevation_rad",
        "tx_gain_dbi",
        "rx_gain_dbi",
        "free_space_path_loss_db",
        "gaseous_loss_db",
        "rain_loss_db",
        "cloud_loss_db",
        "received_power_dbw",
        "cn0_db_hz",
        "eb_n0_db",
        "margin_db",
        "external_interference_density_dbw_hz",
        "effective_noise_density_dbw_hz",
        "selected_data_rate_bps",
    ):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(getattr(samples, name), dtype="<f8").tobytes())
    for name in (
        "tx_pattern_pass",
        "rx_pattern_pass",
        "earth_clear",
        "elevation_pass",
        "range_pass",
        "terrain_clear",
        "atmospheric_applicability_pass",
        "margin_pass",
        "acquisition_threshold_pass",
        "tracking_threshold_pass",
        "receiver_locked",
        "available",
    ):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(getattr(samples, name), dtype=np.uint8).tobytes())
    digest.update("\0".join(samples.primary_reason).encode())
    digest.update("\0".join(samples.selected_mode_id).encode())
    return digest.hexdigest()


def evaluate_pro_link_budget(
    profile: LinkBudgetProfile,
    *,
    times_s: np.ndarray,
    range_km: np.ndarray,
    elevation_rad: np.ndarray,
    tx_off_axis_rad: np.ndarray | None = None,
    rx_off_axis_rad: np.ndarray | None = None,
    range_rate_km_s: np.ndarray | None = None,
    earth_clear: np.ndarray | None = None,
    environment_evidence: ProLinkEnvironmentEvidence | None = None,
    minimum_elevation_rad: float = np.deg2rad(5.0),
    maximum_range_km: float | None = None,
    resource_limits: ResourceLimits | None = None,
    _prepared_inputs: _PreparedProLinkInputs | None = None,
    numeric_backend: str = "rust",
) -> ProLinkBudgetResult:
    """Evaluate a one-way Pro link from authoritative same-epoch geometry evidence."""

    if not isinstance(profile, LinkBudgetProfile):
        raise ValueError("profile must be a LinkBudgetProfile.")
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown communications numeric backend {numeric_backend!r}.")
    prepared = _prepared_inputs or _prepare_pro_link_inputs(
        times_s=times_s,
        range_km=range_km,
        elevation_rad=elevation_rad,
        tx_off_axis_rad=tx_off_axis_rad,
        rx_off_axis_rad=rx_off_axis_rad,
        range_rate_km_s=range_rate_km_s,
        earth_clear=earth_clear,
        minimum_elevation_rad=minimum_elevation_rad,
    )
    times = prepared.times
    ranges = prepared.ranges
    elevation = prepared.elevation
    tx_angle = prepared.tx_angle
    rx_angle = prepared.rx_angle
    rates = prepared.rates
    clear = prepared.clear
    minimum_elevation = prepared.minimum_elevation
    count = times.size
    limits = resource_limits or ResourceLimits()
    limits.enforce(time_samples=count, links=1, terminals=2)
    environment = environment_evidence
    if environment is not None and not isinstance(environment, ProLinkEnvironmentEvidence):
        raise ValueError("environment_evidence must be a ProLinkEnvironmentEvidence record.")

    def _environment_history(field_name: str, default: np.ndarray) -> np.ndarray:
        if environment is None or getattr(environment, field_name) is None:
            return default
        values = np.asarray(getattr(environment, field_name))
        if values.shape != (count,):
            raise ValueError(f"environment_evidence.{field_name} must contain {count} samples.")
        return values

    terrain = _environment_history("terrain_clear", np.ones(count, dtype=bool)).astype(bool)
    interference_temperature = _environment_history(
        "interference_temperature_k",
        np.zeros(count, dtype=float),
    ).astype(float)
    interference_power = (
        None
        if environment is None or environment.interference_power_dbw is None
        else _environment_history("interference_power_dbw", np.empty(count)).astype(float)
    )
    rf = evaluate_pro_rf_closure(
        profile,
        range_km=ranges,
        elevation_rad=elevation,
        tx_off_axis_rad=tx_angle,
        rx_off_axis_rad=rx_angle,
        earth_clear=clear,
        terrain_clear=terrain,
        interference_temperature_k=interference_temperature,
        interference_power_dbw=interference_power,
        interference_bandwidth_hz=None if environment is None else environment.interference_bandwidth_hz,
        minimum_elevation_rad=minimum_elevation,
        maximum_range_km=maximum_range_km,
        numeric_backend=backend,
    )
    atmospheric = rf.atmospheric_evidence
    editions = atmospheric.model_editions
    statistical_components = atmospheric.statistical_components
    margin = rf.margin_db
    geometry_and_equipment_pass = (
        rf.earth_clear
        & rf.terrain_clear
        & rf.elevation_pass
        & rf.range_pass
        & rf.tx_pattern_pass
        & rf.rx_pattern_pass
        & rf.atmospheric_applicability_pass
    )
    if profile.waveform.acquisition_required_eb_n0_db is None:
        acquisition_pass = geometry_and_equipment_pass.copy()
        tracking_pass = geometry_and_equipment_pass.copy()
        receiver_locked = geometry_and_equipment_pass.copy()
    else:
        acquisition_pass = geometry_and_equipment_pass & (
            rf.eb_n0_db >= float(profile.waveform.acquisition_required_eb_n0_db) - 1.0e-10
        )
        tracking_pass = geometry_and_equipment_pass & (
            rf.eb_n0_db >= float(profile.waveform.tracking_required_eb_n0_db) - 1.0e-10
        )
        receiver_locked = np.zeros(count, dtype=bool)
        locked = False
        for index in range(count):
            locked = bool(tracking_pass[index] if locked else acquisition_pass[index])
            receiver_locked[index] = locked
    available = rf.available & receiver_locked
    selected_rate = np.where(receiver_locked, rf.selected_data_rate_bps, 0.0)
    selected_mode = tuple(str(mode) if receiver_locked[index] else "" for index, mode in enumerate(rf.selected_mode_id))
    reasons = np.asarray(rf.primary_reason, dtype=object).copy()
    reasons[rf.available & ~receiver_locked] = "receiver_not_acquired_or_tracking_lost"
    required = np.full(count, profile.waveform.required_eb_n0_db)
    polarization = np.full(count, profile.polarization_loss_db)
    pointing = np.full(count, profile.pointing_loss_db)
    implementation = np.full(count, profile.implementation_loss_db)
    legacy = np.full(count, profile.legacy_misc_loss_db)
    doppler = -profile.carrier_frequency_hz * (rates * 1000.0) / SPEED_OF_LIGHT_M_S
    samples = ProLinkSampleLedger(
        time_s=times,
        range_km=ranges,
        range_rate_km_s=rates,
        elevation_rad=elevation,
        tx_off_axis_rad=tx_angle,
        rx_off_axis_rad=rx_angle,
        tx_gain_dbi=rf.tx_gain_dbi,
        rx_gain_dbi=rf.rx_gain_dbi,
        tx_pattern_pass=rf.tx_pattern_pass,
        rx_pattern_pass=rf.rx_pattern_pass,
        earth_clear=rf.earth_clear,
        elevation_pass=rf.elevation_pass,
        range_pass=rf.range_pass,
        terrain_clear=rf.terrain_clear,
        atmospheric_applicability_pass=rf.atmospheric_applicability_pass,
        free_space_path_loss_db=rf.free_space_path_loss_db,
        gaseous_loss_db=rf.gaseous_loss_db,
        rain_loss_db=rf.rain_loss_db,
        cloud_loss_db=rf.cloud_loss_db,
        polarization_loss_db=polarization,
        pointing_loss_db=pointing,
        implementation_loss_db=implementation,
        legacy_misc_loss_db=legacy,
        total_named_propagation_loss_db=rf.total_named_propagation_loss_db,
        tx_power_dbw=rf.tx_power_dbw,
        eirp_dbw=rf.eirp_dbw,
        received_power_dbw=rf.received_power_dbw,
        noise_density_dbw_hz=rf.noise_density_dbw_hz,
        external_interference_density_dbw_hz=rf.external_interference_density_dbw_hz,
        effective_noise_density_dbw_hz=rf.effective_noise_density_dbw_hz,
        cn0_db_hz=rf.cn0_db_hz,
        eb_n0_db=rf.eb_n0_db,
        required_eb_n0_db=required,
        margin_db=margin,
        margin_pass=rf.margin_pass,
        acquisition_threshold_pass=acquisition_pass,
        tracking_threshold_pass=tracking_pass,
        receiver_locked=receiver_locked,
        selected_data_rate_bps=selected_rate,
        selected_mode_id=selected_mode,
        available=available,
        primary_reason=tuple(str(value) for value in reasons),
        doppler_shift_hz=doppler,
        received_frequency_hz=profile.carrier_frequency_hz + doppler,
    )
    intervals = availability_intervals(times, available, samples.primary_reason)
    available_duration = float(sum(interval.duration_s for interval in intervals))
    horizon = float(times[-1] - times[0])
    integrated_data_bits = float(np.dot(selected_rate[:-1], np.diff(times)))
    finite_margin = margin[np.isfinite(margin)]
    profile_sha256 = profile.semantic_sha256
    summary = {
        "schema_version": PRO_COMMUNICATIONS_SCHEMA_VERSION,
        "contract_version": PRO_LINK_LEDGER_VERSION,
        "rf_conventions_version": RF_CONVENTIONS_VERSION,
        "profile_id": profile.profile_id,
        "profile_sha256": profile_sha256,
        "sample_count": int(count),
        "window_count": len(intervals),
        "available_duration_s": available_duration,
        "sample_bounded_available_fraction": available_duration / horizon,
        "estimated_delivered_data_bits": integrated_data_bits,
        "rate_policy": profile.waveform.rate_policy,
        "selected_rate_minimum_bps": float(np.min(selected_rate)),
        "selected_rate_maximum_bps": float(np.max(selected_rate)),
        "acquisition_tracking_modeled": profile.waveform.acquisition_required_eb_n0_db is not None,
        "environment_source_id": None if environment is None else environment.source_id,
        "terrain_history_supplied": environment is not None and environment.terrain_clear is not None,
        "external_interference_supplied": environment is not None
        and (environment.interference_temperature_k is not None or environment.interference_power_dbw is not None),
        "minimum_margin_db": None if not finite_margin.size else float(np.min(finite_margin)),
        "maximum_margin_db": None if not finite_margin.size else float(np.max(finite_margin)),
        "atmospheric_model_editions": list(editions),
        "statistical_components": list(statistical_components),
        "claims": [
            "Same-epoch one-way engineering feasibility under the declared profiles.",
            "Sample-bounded availability and declared-rate delivery estimate.",
            "Stateful acquisition/tracking and profile-driven rate selection when explicitly configured.",
        ],
        "non_claims": [
            "No flight qualification or operational communications assurance.",
            "No packet, protocol, routing, latency, emitter-database, or regulatory assurance.",
            "Supplied terrain and interference histories are bounded external evidence, not independently predicted truth.",
            "Climatological rain attenuation is not deterministic epoch weather.",
        ],
    }
    semantic_sha256 = _semantic_hash(
        profile_sha256,
        samples,
        None if environment is None else environment.source_id,
    )
    summary["semantic_sha256"] = semantic_sha256
    return ProLinkBudgetResult(
        profile=profile,
        samples=samples,
        intervals=intervals,
        atmospheric_evidence=atmospheric,
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


__all__ = [
    "PRO_LINK_LEDGER_VERSION",
    "ProLinkEnvironmentEvidence",
    "ProLinkBudgetResult",
    "ProLinkSampleLedger",
    "evaluate_pro_link_budget",
]
