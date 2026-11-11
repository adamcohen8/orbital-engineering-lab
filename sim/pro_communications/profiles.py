"""Strict JSON profile resolution for local communications workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from sim.pro_communications.contracts import (
    AdaptiveWaveformMode,
    AntennaPatternProfile,
    AtmosphericProfile,
    LinkBudgetProfile,
    ReceiverProfile,
    SourceProvenance,
    TransmitterProfile,
    WaveformProfile,
)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    return dict(value)


def _reject_unknown(record: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError(f"Unknown {field_name} fields: {unknown}.")


def source_provenance_from_mapping(value: Mapping[str, Any] | None) -> SourceProvenance | None:
    if value is None:
        return None
    record = _mapping(value, "provenance")
    _reject_unknown(record, {"source_id", "source_version", "reference", "applicability", "sha256"}, "provenance")
    return SourceProvenance(**record)


def antenna_profile_from_mapping(value: Mapping[str, Any]) -> AntennaPatternProfile:
    record = _mapping(value, "antenna")
    _reject_unknown(
        record,
        {"pattern_id", "kind", "peak_gain_dbi", "half_angle_deg", "off_axis_deg", "gain_dbi", "provenance"},
        "antenna",
    )
    record["provenance"] = source_provenance_from_mapping(record.get("provenance"))
    if "off_axis_deg" in record:
        record["off_axis_deg"] = tuple(record["off_axis_deg"])
    if "gain_dbi" in record:
        record["gain_dbi"] = tuple(record["gain_dbi"])
    return AntennaPatternProfile(**record)


def link_profile_from_mapping(value: Mapping[str, Any]) -> LinkBudgetProfile:
    record = _mapping(value, "link_profile")
    _reject_unknown(
        record,
        {
            "profile_id",
            "carrier_frequency_hz",
            "transmitter",
            "receiver",
            "waveform",
            "atmosphere",
            "polarization_loss_db",
            "pointing_loss_db",
            "implementation_loss_db",
            "legacy_misc_loss_db",
        },
        "link_profile",
    )
    transmitter = _mapping(record.pop("transmitter"), "transmitter")
    _reject_unknown(
        transmitter,
        {
            "transmitter_id",
            "power_w",
            "line_loss_db",
            "antenna",
            "polarization",
            "provenance",
            "commanded_power_w",
            "available_power_w",
            "amplifier_backoff_db",
            "maximum_eirp_dbw",
            "minimum_frequency_hz",
            "maximum_frequency_hz",
        },
        "transmitter",
    )
    transmitter["antenna"] = antenna_profile_from_mapping(transmitter["antenna"])
    transmitter["provenance"] = source_provenance_from_mapping(transmitter.get("provenance"))
    receiver = _mapping(record.pop("receiver"), "receiver")
    _reject_unknown(
        receiver,
        {
            "receiver_id",
            "line_loss_db",
            "antenna",
            "system_noise_temperature_k",
            "g_over_t_db_per_k",
            "polarization",
            "provenance",
            "noise_equivalent_bandwidth_hz",
            "minimum_frequency_hz",
            "maximum_frequency_hz",
        },
        "receiver",
    )
    receiver["antenna"] = antenna_profile_from_mapping(receiver["antenna"])
    receiver["provenance"] = source_provenance_from_mapping(receiver.get("provenance"))
    waveform = _mapping(record.pop("waveform"), "waveform")
    _reject_unknown(
        waveform,
        {
            "waveform_id",
            "data_rate_bps",
            "required_eb_n0_db",
            "required_eb_n0_provenance",
            "symbol_rate_baud",
            "bits_per_symbol",
            "coding_rate",
            "modulation_id",
            "coding_id",
            "occupied_bandwidth_hz",
            "acquisition_required_eb_n0_db",
            "tracking_required_eb_n0_db",
            "rate_policy",
            "adaptive_modes",
        },
        "waveform",
    )
    waveform["required_eb_n0_provenance"] = source_provenance_from_mapping(waveform.get("required_eb_n0_provenance"))
    adaptive_modes = waveform.get("adaptive_modes", ())
    if not isinstance(adaptive_modes, (list, tuple)):
        raise ValueError("adaptive_modes must be an array.")
    resolved_modes = []
    for value in adaptive_modes:
        mode = _mapping(value, "adaptive_mode")
        _reject_unknown(mode, {"mode_id", "data_rate_bps", "required_eb_n0_db"}, "adaptive_mode")
        resolved_modes.append(AdaptiveWaveformMode(**mode))
    waveform["adaptive_modes"] = tuple(resolved_modes)
    atmosphere = _mapping(record.pop("atmosphere"), "atmosphere")
    _reject_unknown(
        atmosphere,
        {
            "profile_id",
            "gaseous_model",
            "total_pressure_hpa",
            "temperature_k",
            "water_vapour_density_g_m3",
            "rain_model",
            "rainfall_rate_mm_h",
            "rain_height_km",
            "station_height_km",
            "station_latitude_deg",
            "polarization_tilt_deg",
            "rain_exceedance_percentage",
            "cloud_model",
            "integrated_cloud_liquid_water_kg_m2",
            "provenance",
        },
        "atmosphere",
    )
    atmosphere["provenance"] = source_provenance_from_mapping(atmosphere.get("provenance"))
    return LinkBudgetProfile(
        transmitter=TransmitterProfile(**transmitter),
        receiver=ReceiverProfile(**receiver),
        waveform=WaveformProfile(**waveform),
        atmosphere=AtmosphericProfile(**atmosphere),
        **record,
    )


def load_link_profiles(path: str | Path) -> dict[str, LinkBudgetProfile]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    record = _mapping(payload, "profile file")
    _reject_unknown(record, {"schema_version", "profiles"}, "profile file")
    if record.get("schema_version") != "oel.pro-communications-profile-file.v0.1":
        raise ValueError("Unsupported Pro communications profile-file schema_version.")
    profiles = _mapping(record.get("profiles"), "profiles")
    resolved = {str(key): link_profile_from_mapping(value) for key, value in sorted(profiles.items())}
    if any(key != profile.profile_id for key, profile in resolved.items()):
        raise ValueError("Every profile mapping key must equal its profile_id.")
    if not resolved:
        raise ValueError("A profile file must contain at least one profile.")
    return resolved


__all__ = [
    "antenna_profile_from_mapping",
    "link_profile_from_mapping",
    "load_link_profiles",
    "source_provenance_from_mapping",
]
