"""Versioned contracts for bounded local OEL communications analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from functools import cached_property
from typing import Any

import numpy as np

PRO_COMMUNICATIONS_SCHEMA_VERSION = "oel.pro-communications.v0.2"
RF_CONVENTIONS_VERSION = "oel.pro-rf-conventions.v0.2"
_RATE_RELATIVE_TOLERANCE = 1.0e-9
_POLARIZATIONS = {
    "unspecified",
    "linear_horizontal",
    "linear_vertical",
    "right_hand_circular",
    "left_hand_circular",
}


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _finite(value: Any, field_name: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _positive(value: Any, field_name: str) -> float:
    normalized = _finite(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _semantic_payload(value: Any) -> Any:
    """Build the same JSON shape as ``asdict`` without deep-copying scalars."""

    if is_dataclass(value):
        return {
            item.name: _semantic_payload(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_semantic_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _semantic_payload(item) for key, item in value.items()}
    return value


def _nonnegative(value: Any, field_name: str) -> float:
    normalized = _finite(value, field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be nonnegative.")
    return normalized


def _polarization(value: Any, field_name: str) -> str:
    normalized = str(value or "unspecified").strip().lower()
    if normalized not in _POLARIZATIONS:
        raise ValueError(f"{field_name} must be one of {sorted(_POLARIZATIONS)}.")
    return normalized


@dataclass(frozen=True)
class SourceProvenance:
    source_id: str
    source_version: str
    reference: str
    applicability: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("source_id", "source_version", "reference", "applicability"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        if self.sha256 is not None:
            digest = str(self.sha256).strip().lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest.")
            object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class AntennaPatternProfile:
    pattern_id: str
    kind: str
    peak_gain_dbi: float
    half_angle_deg: float | None = None
    off_axis_deg: tuple[float, ...] = ()
    gain_dbi: tuple[float, ...] = ()
    provenance: SourceProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern_id", _required_text(self.pattern_id, "pattern_id"))
        kind = str(self.kind or "").strip().lower()
        if kind not in {"constant", "axisymmetric_hard_cone", "tabulated_axisymmetric"}:
            raise ValueError(
                "Antenna pattern kind must be constant, axisymmetric_hard_cone, or tabulated_axisymmetric."
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "peak_gain_dbi", _finite(self.peak_gain_dbi, "peak_gain_dbi"))
        if kind == "constant":
            if self.half_angle_deg is not None or self.off_axis_deg or self.gain_dbi:
                raise ValueError("A constant antenna pattern cannot declare angular support.")
            return
        if kind == "axisymmetric_hard_cone":
            if self.half_angle_deg is None:
                raise ValueError("An axisymmetric hard-cone pattern requires half_angle_deg.")
            angle = _positive(self.half_angle_deg, "half_angle_deg")
            if angle > 180.0:
                raise ValueError("half_angle_deg must not exceed 180 degrees.")
            if self.off_axis_deg or self.gain_dbi:
                raise ValueError("A hard-cone pattern cannot declare tabulated points.")
            object.__setattr__(self, "half_angle_deg", angle)
            return
        if self.half_angle_deg is not None:
            raise ValueError("A tabulated antenna pattern uses its final angle as the support limit.")
        angles = np.asarray(self.off_axis_deg, dtype=float)
        gains = np.asarray(self.gain_dbi, dtype=float)
        if angles.ndim != 1 or gains.ndim != 1 or angles.size < 2 or angles.shape != gains.shape:
            raise ValueError("Tabulated antenna angles and gains must be matching one-dimensional sequences.")
        if np.any(~np.isfinite(angles)) or np.any(~np.isfinite(gains)):
            raise ValueError("Tabulated antenna angles and gains must be finite.")
        if angles[0] != 0.0 or np.any(np.diff(angles) <= 0.0) or angles[-1] > 180.0:
            raise ValueError(
                "Tabulated antenna angles must start at zero and strictly increase to at most 180 degrees."
            )
        if not np.isclose(gains[0], self.peak_gain_dbi, rtol=0.0, atol=1.0e-12):
            raise ValueError("The first tabulated antenna gain must equal peak_gain_dbi.")
        object.__setattr__(self, "off_axis_deg", tuple(float(value) for value in angles))
        object.__setattr__(self, "gain_dbi", tuple(float(value) for value in gains))

    def evaluate(self, off_axis_deg: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return gain and pattern-pass arrays without extrapolating tabulated data."""

        angles = np.asarray(off_axis_deg, dtype=float)
        if np.any(~np.isfinite(angles)) or np.any(angles < 0.0) or np.any(angles > 180.0):
            raise ValueError("off_axis_deg must contain finite values within [0, 180].")
        if self.kind == "constant":
            return np.full(angles.shape, self.peak_gain_dbi), np.ones(angles.shape, dtype=bool)
        if self.kind == "axisymmetric_hard_cone":
            passed = angles <= float(self.half_angle_deg) + 1.0e-12
            return np.full(angles.shape, self.peak_gain_dbi), passed
        support = np.asarray(self.off_axis_deg, dtype=float)
        gains = np.asarray(self.gain_dbi, dtype=float)
        passed = angles <= support[-1] + 1.0e-12
        values = np.interp(np.minimum(angles, support[-1]), support, gains)
        return values, passed


@dataclass(frozen=True)
class TransmitterProfile:
    transmitter_id: str
    power_w: float
    line_loss_db: float
    antenna: AntennaPatternProfile
    polarization: str = "unspecified"
    provenance: SourceProvenance | None = None
    commanded_power_w: float | None = None
    available_power_w: float | None = None
    amplifier_backoff_db: float = 0.0
    maximum_eirp_dbw: float | None = None
    minimum_frequency_hz: float | None = None
    maximum_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transmitter_id", _required_text(self.transmitter_id, "transmitter_id"))
        object.__setattr__(self, "power_w", _positive(self.power_w, "power_w"))
        object.__setattr__(self, "line_loss_db", _nonnegative(self.line_loss_db, "line_loss_db"))
        if not isinstance(self.antenna, AntennaPatternProfile):
            raise ValueError("antenna must be an AntennaPatternProfile.")
        object.__setattr__(self, "polarization", _polarization(self.polarization, "polarization"))
        backoff = _nonnegative(self.amplifier_backoff_db, "amplifier_backoff_db")
        object.__setattr__(self, "amplifier_backoff_db", backoff)
        power_pair = (self.commanded_power_w is not None, self.available_power_w is not None)
        if power_pair.count(True) == 1:
            raise ValueError("commanded_power_w and available_power_w must be declared together.")
        if all(power_pair):
            commanded = _positive(self.commanded_power_w, "commanded_power_w")
            available = _positive(self.available_power_w, "available_power_w")
            resolved = min(commanded, available) * 10.0 ** (-backoff / 10.0)
            if not np.isclose(resolved, self.power_w, rtol=1.0e-9, atol=0.0):
                raise ValueError(
                    "power_w must equal min(commanded_power_w, available_power_w) after amplifier_backoff_db."
                )
            object.__setattr__(self, "commanded_power_w", commanded)
            object.__setattr__(self, "available_power_w", available)
        elif backoff != 0.0:
            raise ValueError("amplifier_backoff_db requires commanded_power_w and available_power_w.")
        if self.maximum_eirp_dbw is not None:
            object.__setattr__(self, "maximum_eirp_dbw", _finite(self.maximum_eirp_dbw, "maximum_eirp_dbw"))
        self._normalize_frequency_envelope()

    def _normalize_frequency_envelope(self) -> None:
        declared = (self.minimum_frequency_hz is not None, self.maximum_frequency_hz is not None)
        if declared.count(True) == 1:
            raise ValueError("minimum_frequency_hz and maximum_frequency_hz must be declared together.")
        if all(declared):
            minimum = _positive(self.minimum_frequency_hz, "minimum_frequency_hz")
            maximum = _positive(self.maximum_frequency_hz, "maximum_frequency_hz")
            if minimum > maximum:
                raise ValueError("minimum_frequency_hz must not exceed maximum_frequency_hz.")
            object.__setattr__(self, "minimum_frequency_hz", minimum)
            object.__setattr__(self, "maximum_frequency_hz", maximum)


@dataclass(frozen=True)
class ReceiverProfile:
    receiver_id: str
    line_loss_db: float
    antenna: AntennaPatternProfile
    system_noise_temperature_k: float | None = None
    g_over_t_db_per_k: float | None = None
    polarization: str = "unspecified"
    provenance: SourceProvenance | None = None
    noise_equivalent_bandwidth_hz: float | None = None
    minimum_frequency_hz: float | None = None
    maximum_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "receiver_id", _required_text(self.receiver_id, "receiver_id"))
        object.__setattr__(self, "line_loss_db", _nonnegative(self.line_loss_db, "line_loss_db"))
        if not isinstance(self.antenna, AntennaPatternProfile):
            raise ValueError("antenna must be an AntennaPatternProfile.")
        declared = (self.system_noise_temperature_k is not None, self.g_over_t_db_per_k is not None)
        if declared.count(True) != 1:
            raise ValueError("Declare exactly one of system_noise_temperature_k or g_over_t_db_per_k.")
        if self.system_noise_temperature_k is not None:
            object.__setattr__(
                self,
                "system_noise_temperature_k",
                _positive(self.system_noise_temperature_k, "system_noise_temperature_k"),
            )
        if self.g_over_t_db_per_k is not None:
            object.__setattr__(
                self,
                "g_over_t_db_per_k",
                _finite(self.g_over_t_db_per_k, "g_over_t_db_per_k"),
            )
        object.__setattr__(self, "polarization", _polarization(self.polarization, "polarization"))
        if self.noise_equivalent_bandwidth_hz is not None:
            object.__setattr__(
                self,
                "noise_equivalent_bandwidth_hz",
                _positive(self.noise_equivalent_bandwidth_hz, "noise_equivalent_bandwidth_hz"),
            )
        declared_frequency = (self.minimum_frequency_hz is not None, self.maximum_frequency_hz is not None)
        if declared_frequency.count(True) == 1:
            raise ValueError("minimum_frequency_hz and maximum_frequency_hz must be declared together.")
        if all(declared_frequency):
            minimum = _positive(self.minimum_frequency_hz, "minimum_frequency_hz")
            maximum = _positive(self.maximum_frequency_hz, "maximum_frequency_hz")
            if minimum > maximum:
                raise ValueError("minimum_frequency_hz must not exceed maximum_frequency_hz.")
            object.__setattr__(self, "minimum_frequency_hz", minimum)
            object.__setattr__(self, "maximum_frequency_hz", maximum)

    @property
    def resolved_system_noise_temperature_k(self) -> float:
        if self.system_noise_temperature_k is not None:
            return self.system_noise_temperature_k
        return float(10.0 ** ((self.antenna.peak_gain_dbi - float(self.g_over_t_db_per_k)) / 10.0))


@dataclass(frozen=True)
class AdaptiveWaveformMode:
    mode_id: str
    data_rate_bps: float
    required_eb_n0_db: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode_id", _required_text(self.mode_id, "mode_id"))
        object.__setattr__(self, "data_rate_bps", _positive(self.data_rate_bps, "data_rate_bps"))
        object.__setattr__(self, "required_eb_n0_db", _finite(self.required_eb_n0_db, "required_eb_n0_db"))


@dataclass(frozen=True)
class WaveformProfile:
    waveform_id: str
    data_rate_bps: float
    required_eb_n0_db: float
    required_eb_n0_provenance: SourceProvenance
    symbol_rate_baud: float | None = None
    bits_per_symbol: float | None = None
    coding_rate: float | None = None
    modulation_id: str = "unspecified"
    coding_id: str = "unspecified"
    occupied_bandwidth_hz: float | None = None
    acquisition_required_eb_n0_db: float | None = None
    tracking_required_eb_n0_db: float | None = None
    rate_policy: str = "fixed"
    adaptive_modes: tuple[AdaptiveWaveformMode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "waveform_id", _required_text(self.waveform_id, "waveform_id"))
        object.__setattr__(self, "data_rate_bps", _positive(self.data_rate_bps, "data_rate_bps"))
        object.__setattr__(self, "required_eb_n0_db", _finite(self.required_eb_n0_db, "required_eb_n0_db"))
        object.__setattr__(self, "modulation_id", _required_text(self.modulation_id, "modulation_id"))
        object.__setattr__(self, "coding_id", _required_text(self.coding_id, "coding_id"))
        if not isinstance(self.required_eb_n0_provenance, SourceProvenance):
            raise ValueError("required_eb_n0_provenance must be a SourceProvenance record.")
        optional = (self.symbol_rate_baud, self.bits_per_symbol, self.coding_rate)
        if any(value is not None for value in optional) and not all(value is not None for value in optional):
            raise ValueError("symbol_rate_baud, bits_per_symbol, and coding_rate must be declared together.")
        if all(value is not None for value in optional):
            symbol_rate = _positive(self.symbol_rate_baud, "symbol_rate_baud")
            bits = _positive(self.bits_per_symbol, "bits_per_symbol")
            coding = _positive(self.coding_rate, "coding_rate")
            if coding > 1.0:
                raise ValueError("coding_rate must not exceed one.")
            derived = symbol_rate * bits * coding
            if not np.isclose(derived, self.data_rate_bps, rtol=_RATE_RELATIVE_TOLERANCE, atol=0.0):
                raise ValueError("Waveform symbol-rate fields do not reproduce data_rate_bps.")
            object.__setattr__(self, "symbol_rate_baud", symbol_rate)
            object.__setattr__(self, "bits_per_symbol", bits)
            object.__setattr__(self, "coding_rate", coding)
        if self.occupied_bandwidth_hz is not None:
            object.__setattr__(
                self,
                "occupied_bandwidth_hz",
                _positive(self.occupied_bandwidth_hz, "occupied_bandwidth_hz"),
            )
        closure_pair = (
            self.acquisition_required_eb_n0_db is not None,
            self.tracking_required_eb_n0_db is not None,
        )
        if closure_pair.count(True) == 1:
            raise ValueError("acquisition_required_eb_n0_db and tracking_required_eb_n0_db must be declared together.")
        if all(closure_pair):
            acquisition = _finite(self.acquisition_required_eb_n0_db, "acquisition_required_eb_n0_db")
            tracking = _finite(self.tracking_required_eb_n0_db, "tracking_required_eb_n0_db")
            if acquisition < tracking:
                raise ValueError("acquisition_required_eb_n0_db must be at least tracking_required_eb_n0_db.")
            object.__setattr__(self, "acquisition_required_eb_n0_db", acquisition)
            object.__setattr__(self, "tracking_required_eb_n0_db", tracking)
        policy = str(self.rate_policy or "fixed").strip().lower()
        if policy not in {"fixed", "highest_rate_meeting_eb_n0"}:
            raise ValueError("rate_policy must be fixed or highest_rate_meeting_eb_n0.")
        modes = tuple(self.adaptive_modes)
        if policy == "fixed" and modes:
            raise ValueError("adaptive_modes require rate_policy highest_rate_meeting_eb_n0.")
        if policy == "highest_rate_meeting_eb_n0":
            if len(modes) < 2 or any(not isinstance(mode, AdaptiveWaveformMode) for mode in modes):
                raise ValueError("Adaptive rate policy requires at least two AdaptiveWaveformMode records.")
            if len({mode.mode_id for mode in modes}) != len(modes):
                raise ValueError("Adaptive waveform mode IDs must be unique.")
            if len({mode.data_rate_bps for mode in modes}) != len(modes):
                raise ValueError("Adaptive waveform data rates must be unique.")
            primary = [
                mode
                for mode in modes
                if np.isclose(mode.data_rate_bps, self.data_rate_bps, rtol=1.0e-12, atol=0.0)
                and np.isclose(mode.required_eb_n0_db, self.required_eb_n0_db, rtol=0.0, atol=1.0e-12)
            ]
            if len(primary) != 1:
                raise ValueError("Exactly one adaptive mode must reproduce the primary rate and Eb/N0 threshold.")
            modes = tuple(sorted(modes, key=lambda mode: (mode.data_rate_bps, mode.mode_id)))
        object.__setattr__(self, "rate_policy", policy)
        object.__setattr__(self, "adaptive_modes", modes)

    def selected_rate_bps(self, eb_n0_db: float | np.ndarray) -> np.ndarray:
        """Return the deterministic supported information rate at each Eb/N0 value."""

        values = np.asarray(eb_n0_db, dtype=float)
        if np.any(np.isinf(values)):
            raise ValueError("eb_n0_db must not contain infinity.")
        if self.rate_policy == "fixed":
            return np.where(values >= self.required_eb_n0_db - 1.0e-10, self.data_rate_bps, 0.0)
        selected = np.zeros(values.shape, dtype=float)
        for mode in self.adaptive_modes:
            selected = np.where(values >= mode.required_eb_n0_db - 1.0e-10, mode.data_rate_bps, selected)
        return selected


@dataclass(frozen=True)
class AtmosphericProfile:
    profile_id: str
    gaseous_model: str = "none"
    total_pressure_hpa: float | None = None
    temperature_k: float | None = None
    water_vapour_density_g_m3: float | None = None
    rain_model: str = "none"
    rainfall_rate_mm_h: float | None = None
    rain_height_km: float | None = None
    station_height_km: float = 0.0
    station_latitude_deg: float | None = None
    polarization_tilt_deg: float = 0.0
    rain_exceedance_percentage: float = 0.01
    cloud_model: str = "none"
    integrated_cloud_liquid_water_kg_m2: float | None = None
    provenance: SourceProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required_text(self.profile_id, "profile_id"))
        gas = str(self.gaseous_model or "none").strip().lower()
        rain = str(self.rain_model or "none").strip().lower()
        cloud = str(self.cloud_model or "none").strip().lower()
        if gas not in {"none", "itu-r-p.676-13-annex2-instantaneous"}:
            raise ValueError("Unsupported gaseous_model.")
        if rain not in {"none", "itu-r-p.618-14-declared-r0.01"}:
            raise ValueError("Unsupported rain_model.")
        if cloud not in {"none", "itu-r-p.840-9-instantaneous"}:
            raise ValueError("Unsupported cloud_model.")
        object.__setattr__(self, "gaseous_model", gas)
        object.__setattr__(self, "rain_model", rain)
        object.__setattr__(self, "cloud_model", cloud)
        if gas != "none":
            for field_name in ("total_pressure_hpa", "temperature_k", "water_vapour_density_g_m3"):
                if getattr(self, field_name) is None:
                    raise ValueError(f"{gas} requires {field_name}.")
            object.__setattr__(self, "total_pressure_hpa", _positive(self.total_pressure_hpa, "total_pressure_hpa"))
            object.__setattr__(self, "temperature_k", _positive(self.temperature_k, "temperature_k"))
            object.__setattr__(
                self,
                "water_vapour_density_g_m3",
                _nonnegative(self.water_vapour_density_g_m3, "water_vapour_density_g_m3"),
            )
        if rain != "none":
            for field_name in ("rainfall_rate_mm_h", "rain_height_km", "station_latitude_deg"):
                if getattr(self, field_name) is None:
                    raise ValueError(f"{rain} requires {field_name}.")
            object.__setattr__(self, "rainfall_rate_mm_h", _nonnegative(self.rainfall_rate_mm_h, "rainfall_rate_mm_h"))
            object.__setattr__(self, "rain_height_km", _finite(self.rain_height_km, "rain_height_km"))
            latitude = _finite(self.station_latitude_deg, "station_latitude_deg")
            if not -90.0 <= latitude <= 90.0:
                raise ValueError("station_latitude_deg must be within [-90, 90].")
            object.__setattr__(self, "station_latitude_deg", latitude)
            percentage = _positive(self.rain_exceedance_percentage, "rain_exceedance_percentage")
            if not 0.001 <= percentage <= 5.0:
                raise ValueError("rain_exceedance_percentage must be within [0.001, 5].")
            object.__setattr__(self, "rain_exceedance_percentage", percentage)
        object.__setattr__(self, "station_height_km", _finite(self.station_height_km, "station_height_km"))
        tilt = _finite(self.polarization_tilt_deg, "polarization_tilt_deg")
        if not -180.0 <= tilt <= 180.0:
            raise ValueError("polarization_tilt_deg must be within [-180, 180].")
        object.__setattr__(self, "polarization_tilt_deg", tilt)
        if cloud != "none":
            if self.integrated_cloud_liquid_water_kg_m2 is None:
                raise ValueError(f"{cloud} requires integrated_cloud_liquid_water_kg_m2.")
            object.__setattr__(
                self,
                "integrated_cloud_liquid_water_kg_m2",
                _nonnegative(
                    self.integrated_cloud_liquid_water_kg_m2,
                    "integrated_cloud_liquid_water_kg_m2",
                ),
            )


@dataclass(frozen=True)
class LinkBudgetProfile:
    profile_id: str
    carrier_frequency_hz: float
    transmitter: TransmitterProfile
    receiver: ReceiverProfile
    waveform: WaveformProfile
    atmosphere: AtmosphericProfile
    polarization_loss_db: float = 0.0
    pointing_loss_db: float = 0.0
    implementation_loss_db: float = 0.0
    legacy_misc_loss_db: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required_text(self.profile_id, "profile_id"))
        object.__setattr__(
            self,
            "carrier_frequency_hz",
            _positive(self.carrier_frequency_hz, "carrier_frequency_hz"),
        )
        if not isinstance(self.transmitter, TransmitterProfile):
            raise ValueError("transmitter must be a TransmitterProfile.")
        if not isinstance(self.receiver, ReceiverProfile):
            raise ValueError("receiver must be a ReceiverProfile.")
        if not isinstance(self.waveform, WaveformProfile):
            raise ValueError("waveform must be a WaveformProfile.")
        if not isinstance(self.atmosphere, AtmosphericProfile):
            raise ValueError("atmosphere must be an AtmosphericProfile.")
        frequency = self.carrier_frequency_hz
        for endpoint_name, endpoint in (("transmitter", self.transmitter), ("receiver", self.receiver)):
            if endpoint.minimum_frequency_hz is not None and not (
                endpoint.minimum_frequency_hz <= frequency <= endpoint.maximum_frequency_hz
            ):
                raise ValueError(f"carrier_frequency_hz is outside the {endpoint_name} applicability envelope.")
        if self.transmitter.maximum_eirp_dbw is not None:
            peak_eirp = (
                10.0 * np.log10(self.transmitter.power_w)
                - self.transmitter.line_loss_db
                + self.transmitter.antenna.peak_gain_dbi
            )
            if peak_eirp > self.transmitter.maximum_eirp_dbw + 1.0e-10:
                raise ValueError("The configured transmitter operating point exceeds maximum_eirp_dbw.")
        named_total = 0.0
        for field_name in ("polarization_loss_db", "pointing_loss_db", "implementation_loss_db"):
            value = _nonnegative(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
            named_total += value
        legacy = _nonnegative(self.legacy_misc_loss_db, "legacy_misc_loss_db")
        if legacy > 0.0 and named_total > 0.0:
            raise ValueError("legacy_misc_loss_db cannot be combined with named Pro loss terms.")
        object.__setattr__(self, "legacy_misc_loss_db", legacy)

    @cached_property
    def semantic_sha256(self) -> str:
        payload = {
            "schema_version": PRO_COMMUNICATIONS_SCHEMA_VERSION,
            "rf_conventions_version": RF_CONVENTIONS_VERSION,
            "profile": _semantic_payload(self),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ResourceLimits:
    profile_id: str = "laptop-safe"
    max_time_samples: int = 2_000
    max_cells: int = 196_608
    max_links: int = 8
    max_terminals: int = 32
    max_constellation_members: int = 16
    max_schedule_opportunities: int = 500
    max_campaign_cases: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required_text(self.profile_id, "profile_id"))
        for field_name in (
            "max_time_samples",
            "max_cells",
            "max_links",
            "max_terminals",
            "max_constellation_members",
            "max_schedule_opportunities",
            "max_campaign_cases",
        ):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
            object.__setattr__(self, field_name, int(value))

    def enforce(self, **counts: int) -> None:
        mapping = {
            "time_samples": "max_time_samples",
            "cells": "max_cells",
            "links": "max_links",
            "terminals": "max_terminals",
            "constellation_members": "max_constellation_members",
            "schedule_opportunities": "max_schedule_opportunities",
            "campaign_cases": "max_campaign_cases",
        }
        unknown = sorted(set(counts) - set(mapping))
        if unknown:
            raise ValueError(f"Unknown resource dimensions: {unknown}.")
        for name, raw in counts.items():
            value = int(raw)
            limit = getattr(self, mapping[name])
            if value < 0:
                raise ValueError(f"Resource count {name} must be nonnegative.")
            if value > limit:
                raise ValueError(
                    f"Resource profile {self.profile_id} exceeded {name}: requested {value}, limit {limit}."
                )


__all__ = [
    "AdaptiveWaveformMode",
    "AntennaPatternProfile",
    "AtmosphericProfile",
    "LinkBudgetProfile",
    "PRO_COMMUNICATIONS_SCHEMA_VERSION",
    "RF_CONVENTIONS_VERSION",
    "ReceiverProfile",
    "ResourceLimits",
    "SourceProvenance",
    "TransmitterProfile",
    "WaveformProfile",
]
