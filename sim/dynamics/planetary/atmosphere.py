"""OEL-owned deterministic atmosphere interpolation with explicit altitude datum."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .resources import DATA_ROOT, ResourceReceipt, read_json


def validate_atmosphere_options(config):
    model = config.get("model", "none")
    common = {"model", "out_of_domain"}
    options = {
        "none": common,
        "mars_reference_table": common | {"sha256"},
        "table": common | {"data_path", "sha256"},
        "exponential": common | {
            "altitude_datum", "altitude_domain_km", "reference_density_kg_m3",
            "reference_altitude_km", "scale_height_km", "wind_body_m_s",
        },
    }
    if model not in options:
        raise ValueError(f"Unsupported planetary atmosphere model {model!r}")
    unused = set(config) - options[model]
    if unused:
        raise ValueError(f"Planetary atmosphere {model!r} does not accept: {', '.join(sorted(unused))}")


@dataclass(frozen=True)
class AtmosphereSample:
    altitude_km: float
    density_kg_m3: float
    wind_body_m_s: np.ndarray
    temperature_k: float | None = None
    pressure_pa: float | None = None
    within_domain: bool = True


@dataclass(frozen=True)
class PreparedAtmosphere:
    model_id: str
    body_id: str
    frame_id: str
    definition: dict
    policy: str
    heights: np.ndarray | None = None
    densities: np.ndarray | None = None
    winds: np.ndarray | None = None
    receipt: object | None = None
    source_receipts: tuple = ()

    def sample(self, r_body, *, t_s=0.0):
        from .native import atmosphere_input, module

        height, rho, wind, inside = module().planetary_atmosphere_sample(
            atmosphere_input(self), np.asarray(r_body, dtype=float).tolist()
        )
        return AtmosphereSample(height, rho, np.asarray(wind), within_domain=inside)

    def metadata(self):
        return {
            "model": self.model_id,
            "body_id": self.body_id,
            "frame_id": self.frame_id,
            "out_of_domain": self.policy,
            "definition": deepcopy(self.definition),
            "sha256": None if self.receipt is None else self.receipt.sha256,
            "interpolation": "log_density_linear_wind",
            "thermodynamic_properties": "unavailable",
        }


def prepare_atmosphere(body, config, frame_id):
    validate_atmosphere_options(config)
    model = config.get("model", "none")
    policy = config.get("out_of_domain", "error")
    if policy not in {"error", "vacuum_above"}:
        raise ValueError(
            "Planetary atmosphere out_of_domain must be error or vacuum_above"
        )
    if model == "none":
        data = {
            "altitude_datum": {
                "kind": "radial_sphere",
                "radius_km": body.reference_radius_km,
            },
            "altitude_domain_km": [-body.reference_radius_km, 1.0e20],
        }
        return PreparedAtmosphere(model, body.body_id, frame_id, data, policy)
    if body.body_id == "sun":
        raise ValueError("Sun-centered neutral-atmosphere drag is unsupported")
    receipt = None
    if model == "exponential":
        data = {
            key: config[key]
            for key in (
                "altitude_datum",
                "altitude_domain_km",
                "reference_density_kg_m3",
                "reference_altitude_km",
                "scale_height_km",
            )
        }
        data["wind_body_m_s"] = config.get("wind_body_m_s", [0.0, 0.0, 0.0])
        data["source"] = "User-defined controlled atmosphere experiment"
        if data["scale_height_km"] <= 0 or data["reference_density_kg_m3"] <= 0:
            raise ValueError(
                "Exponential atmosphere requires positive scale height and reference density"
            )
        return PreparedAtmosphere(model, body.body_id, frame_id, data, policy)
    if model == "mars_reference_table":
        if body.body_id != "mars":
            raise ValueError("mars_reference_table requires Mars center")
        receipt, data = read_json(DATA_ROOT / "mars_reference_atmosphere.json", config.get("sha256"))
    elif model == "table":
        receipt, data = read_json(config["data_path"], config.get("sha256"))
    else:
        raise ValueError(f"Unsupported planetary atmosphere model {model!r}")
    if data.get("body_id") != body.body_id or data.get("frame_id") != frame_id:
        raise ValueError("Atmosphere resource body/frame mismatch")
    if (
        not data.get("source")
        or not data.get("conditions")
        or data.get("density_units") != "kg/m3"
    ):
        raise ValueError(
            "Atmosphere table requires source, fixed conditions, and kg/m3 density units"
        )
    if (
        data.get("wind_units") != "m/s"
        or data.get("wind_frame") != "body_fixed_cartesian"
    ):
        raise ValueError("Atmosphere table requires body-fixed Cartesian winds in m/s")
    h = np.array(data["altitude_km"], dtype=float)
    rho = np.array(data["density_kg_m3"], dtype=float)
    wind = np.array(data["wind_body_m_s"], dtype=float)
    if h.ndim != 1 or h.size < 2 or rho.shape != h.shape or wind.shape != (h.size, 3):
        raise ValueError("Invalid planetary atmosphere table dimensions")
    if (
        not all(np.all(np.isfinite(x)) for x in (h, rho, wind))
        or np.any(np.diff(h) <= 0)
        or np.any(rho <= 0)
    ):
        raise ValueError(
            "Atmosphere tables require ordered finite heights and positive finite density"
        )
    if list(data["altitude_domain_km"]) != [h[0], h[-1]]:
        raise ValueError("Atmosphere altitude domain must equal its table endpoints")
    for x in (h, rho, wind):
        x.setflags(write=False)
    source_receipts = ()
    if model == "mars_reference_table":
        source_receipts = tuple(
            ResourceReceipt.read(DATA_ROOT / product["file"], product["sha256"])[0]
            for product in data["source"]["products"]
        )
    return PreparedAtmosphere(
        model,
        body.body_id,
        frame_id,
        data,
        policy,
        h,
        rho,
        wind,
        receipt,
        source_receipts,
    )
