"""Single thermal node and bounded battery; SI except orbit km and battery Wh.

Earth radiation uses deterministic apparent-disk quadrature of a uniform
Lambertian sphere. Panels are electrically connected, thermally external to
the body node. This is an engineering model, not hardware qualification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.epoch import AU_KM
from sim.utils.quaternion import quaternion_to_dcm_bn

SIGMA = 5.670374419e-8
ENERGY_RATES = {
    "solar_generation": "solar_generation_w",
    "load_served": "load_served_w",
    "unmet_load": "unmet_load_w",
    "curtailed": "curtailed_power_w",
    "battery_charge": "battery_charge_w",
    "battery_discharge": "battery_discharge_w",
    "battery_loss": "battery_loss_w",
    "conversion_loss": "conversion_loss_w",
    "solar_heat": "solar_heat_w",
    "albedo_heat": "albedo_heat_w",
    "earth_ir_heat": "earth_ir_heat_w",
    "internal_heat": "internal_heat_w",
    "radiated_heat": "radiated_heat_w",
}


def _mapping(value, path, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
    return dict(value)


def _number(data, key, path, default=None, *, low=0.0, high=None, positive=False):
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path}.{key} must be a finite number")
    value = float(value)
    if value < low or (positive and value <= low) or (high is not None and value > high):
        raise ValueError(f"{path}.{key} outside allowed range")
    data[key] = value
    return value


def validate_resource_specs(specs: dict, path="specs") -> tuple[dict, dict]:
    """Validate without loading geometry, plugins, or executing a scenario."""
    numeric_backend = specs.get("numeric_backend", "rust")
    if numeric_backend not in {"python", "rust"}:
        raise ValueError(f"{path}.numeric_backend must be python or rust")
    thermal = _mapping(
        specs.get("thermal", {}),
        f"{path}.thermal",
        {
            "enabled",
            "initial_temperature_k",
            "heat_capacity_j_k",
            "solar_absorptivity",
            "infrared_emissivity",
            "projected_area_m2",
            "radiating_area_m2",
            "area_mode",
            "internal_heat_w",
            "earth_albedo",
            "earth_ir_w_m2",
            "solar_irradiance_w_m2",
            "max_step_s",
            "earth_quadrature_order",
        },
    )
    power = _mapping(
        specs.get("power", {}),
        f"{path}.power",
        {
            "enabled",
            "panels",
            "baseline_load_w",
            "load_heat_fraction",
            "conversion_efficiency",
            "battery_capacity_wh",
            "initial_soc",
            "charge_efficiency",
            "discharge_efficiency",
            "max_charge_w",
            "max_discharge_w",
            "solar_irradiance_w_m2",
            "max_step_s",
        },
    )
    for name, data in (("thermal", thermal), ("power", power)):
        if not isinstance(data.get("enabled", False), bool):
            raise ValueError(f"{path}.{name}.enabled must be boolean")
        if not data.get("enabled", False):
            if set(data) - {"enabled"}:
                raise ValueError(f"{path}.{name}: parameters require enabled: true")
            continue
        _number(data, "solar_irradiance_w_m2", f"{path}.{name}", 1361.0)
        _number(data, "max_step_s", f"{path}.{name}", 1.0, positive=True)
    if thermal.get("enabled"):
        p = f"{path}.thermal"
        for key in ("initial_temperature_k", "heat_capacity_j_k", "radiating_area_m2"):
            _number(thermal, key, p, positive=True)
        for key in ("solar_absorptivity", "infrared_emissivity"):
            _number(thermal, key, p, high=1.0)
        mode = thermal.setdefault("area_mode", "constant")
        if mode not in {"constant", "geometry"}:
            raise ValueError(f"{p}.area_mode must be constant or geometry")
        if mode == "constant":
            _number(thermal, "projected_area_m2", p, positive=True)
        elif "projected_area_m2" in thermal:
            raise ValueError(f"{p}: geometry and projected_area_m2 are mutually exclusive")
        _number(thermal, "internal_heat_w", p, 0.0)
        _number(thermal, "earth_albedo", p, 0.3, high=1.0)
        _number(thermal, "earth_ir_w_m2", p, 237.0)
        order = thermal.setdefault("earth_quadrature_order", 8)
        if isinstance(order, bool) or not isinstance(order, int) or not 4 <= order <= 64:
            raise ValueError(f"{p}.earth_quadrature_order must be an integer in [4, 64]")
    if power.get("enabled"):
        p = f"{path}.power"
        _number(power, "baseline_load_w", p)
        _number(power, "battery_capacity_wh", p, positive=True)
        _number(power, "initial_soc", p, high=1.0)
        _number(power, "load_heat_fraction", p, 1.0, high=1.0)
        for key, default in (
            ("conversion_efficiency", 0.95),
            ("charge_efficiency", 0.95),
            ("discharge_efficiency", 0.95),
        ):
            _number(power, key, p, default, high=1.0, positive=True)
        for key in ("max_charge_w", "max_discharge_w"):
            _number(power, key, p)
        panels = power.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError(f"{p}.panels must be a nonempty list")
        normalized = []
        for i, item in enumerate(panels):
            pp = f"{p}.panels[{i}]"
            panel = _mapping(item, pp, {"area_m2", "normal_body", "efficiency"})
            _number(panel, "area_m2", pp, positive=True)
            _number(panel, "efficiency", pp, high=1.0)
            n = np.asarray(panel.get("normal_body"), dtype=float)
            if n.shape != (3,) or not np.all(np.isfinite(n)) or np.linalg.norm(n) <= 0.0:
                raise ValueError(f"{pp}.normal_body must be a finite nonzero 3-vector")
            panel["normal_body"] = (n / np.linalg.norm(n)).tolist()
            normalized.append(panel)
        power["panels"] = normalized
    if thermal.get("enabled") and power.get("enabled"):
        if thermal["solar_irradiance_w_m2"] != power["solar_irradiance_w_m2"]:
            raise ValueError(f"{path}: thermal and power solar irradiance must agree")
    return thermal, power


@lru_cache(maxsize=16)
def _quadrature(order):
    x, w = np.polynomial.legendre.leggauss(order)
    phi = (np.arange(2 * order) + 0.5) * math.pi / order
    return x, w, phi


def earth_radiation(position, sun_position, order, *, numeric_backend="rust"):
    """Return source directions, solid angles and surface solar irradiance factors."""
    position_array = np.asarray(position, dtype=float)
    sun_array = np.asarray(sun_position, dtype=float)
    if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
        raise ValueError("position must be finite with shape (3,)")
    if sun_array.shape != (3,) or not np.all(np.isfinite(sun_array)):
        raise ValueError("sun_position must be finite with shape (3,)")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or not 4 <= order <= 64:
        raise ValueError("Earth-radiation quadrature order must be an integer in [4, 64]")
    if numeric_backend == "rust":
        from sim.rust_spacecraft_backend import earth_radiation as rust_earth_radiation

        return rust_earth_radiation(position_array, sun_array, order)
    if numeric_backend != "python":
        raise ValueError("spacecraft resource numeric_backend must be python or rust")
    r = position_array
    distance = float(np.linalg.norm(r))
    if distance <= EARTH_RADIUS_KM:
        raise ValueError("spacecraft resources require position above the Earth surface")
    axis = -r / distance
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(axis, helper)
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    x, w, phi = _quadrature(order)
    mu_min = math.sqrt(1.0 - (EARTH_RADIUS_KM / distance) ** 2)
    mu = mu_min + (x + 1.0) * (1.0 - mu_min) / 2.0
    rays = (
        mu[:, None, None] * axis
        + np.sqrt(1.0 - mu**2)[:, None, None] * (np.cos(phi)[None, :, None] * a + np.sin(phi)[None, :, None] * b)
    ).reshape(-1, 3)
    weights = np.repeat(w * (1.0 - mu_min) / 2.0 * math.pi / order, 2 * order)
    dot = rays @ r
    travel = -dot - np.sqrt(np.maximum(0.0, dot**2 - (distance**2 - EARTH_RADIUS_KM**2)))
    points = r + rays * travel[:, None]
    normals = points / EARTH_RADIUS_KM
    to_sun = sun_array - points
    ds = np.linalg.norm(to_sun, axis=1)
    illumination = np.maximum(0.0, np.sum(normals * to_sun / ds[:, None], axis=1)) * (AU_KM / ds) ** 2
    return rays, weights, illumination


@dataclass(frozen=True)
class SpacecraftResources:
    thermal: dict
    power: dict
    geometry: Any = None
    numeric_backend: str = "rust"

    @classmethod
    def from_specs(cls, specs, geometry=None):
        thermal, power = validate_resource_specs(specs)
        if not thermal.get("enabled") and not power.get("enabled"):
            return None
        numeric_backend = str(specs.get("numeric_backend", "rust"))
        if numeric_backend not in {"python", "rust"}:
            raise ValueError("spacecraft resource numeric_backend must be python or rust")
        if thermal.get("area_mode") == "geometry" and geometry is None:
            raise ValueError("specs.thermal.area_mode=geometry requires spacecraft geometry")
        return cls(thermal, power, geometry, numeric_backend)

    @property
    def max_step_s(self):
        return min(d["max_step_s"] for d in (self.thermal, self.power) if d.get("enabled"))

    def initial_state(self):
        result = {}
        if self.thermal.get("enabled"):
            result["temperature_k"] = self.thermal["initial_temperature_k"]
        if self.power.get("enabled"):
            result["battery_energy_wh"] = self.power["battery_capacity_wh"] * self.power["initial_soc"]
            result["battery_soc"] = self.power["initial_soc"]
        for key in ENERGY_RATES:
            owner = self.thermal if "heat" in key else self.power
            if owner.get("enabled"):
                result[key + "_energy_j"] = 0.0
        return result

    def _area(self, directions_body):
        if self.thermal.get("area_mode") != "geometry":
            return np.full(len(directions_body), self.thermal["projected_area_m2"])
        method = getattr(self.geometry, "projected_area_for_direction_m2", None)
        if method is None:
            method = self.geometry.projected_area_m2
        return np.array([method(-u) for u in directions_body])

    def advance(self, start, end, environment, dt_s):
        if self.numeric_backend == "rust":
            from sim.rust_spacecraft_backend import advance as rust_advance

            return rust_advance(self, start, end, environment, dt_s)
        return self._advance_python(start, end, environment, dt_s)

    def _advance_python(self, start, end, environment, dt_s):
        """Advance owned states with midpoint forcing and implicit radiative cooling.

        Returned rates are interval averages. Backward-Euler radiation ensures
        nonnegative temperatures without clipping energy; integration is first order.
        """
        h = float(dt_s)
        if not math.isfinite(h) or h <= 0.0:
            raise ValueError("resource timestep must be finite and positive")
        old = start.resource_state if start.resource_state is not None else self.initial_state()
        result = dict(old)
        position = (start.position_eci_km + end.position_eci_km) / 2.0
        q0, q1 = start.attitude_quat_bn, end.attitude_quat_bn
        q = q0 + (q1 if np.dot(q0, q1) >= 0.0 else -q1)
        q = q / np.linalg.norm(q)
        c_bn = quaternion_to_dcm_bn(q)
        t = (start.t_s + end.t_s) / 2.0
        geometry = resolve_srp_geometry(position, t, environment)
        shadow = srp_shadow_factor(position, t, environment, srp_geometry=geometry)
        sun_body = c_bn @ np.asarray(geometry["sun_dir_sc_eci"])
        settings = self.thermal if self.thermal.get("enabled") else self.power
        solar = settings["solar_irradiance_w_m2"] * float(geometry["distance_scale"]) * shadow
        result.update(
            time_s=float(end.t_s),
            interval_start_s=float(start.t_s),
            sunlit_fraction=shadow,
            solar_irradiance_w_m2=solar,
        )
        electrical_heat = 0.0
        if self.power.get("enabled"):
            p = self.power
            raw = solar * sum(
                panel["area_m2"] * panel["efficiency"] * max(0.0, np.dot(panel["normal_body"], sun_body))
                for panel in p["panels"]
            )
            generation = raw * p["conversion_efficiency"]
            demand = p["baseline_load_w"]
            energy = old["battery_energy_wh"]
            charge = min(
                max(generation - demand, 0.0),
                p["max_charge_w"],
                max(0.0, p["battery_capacity_wh"] - energy) * 3600.0 / (h * p["charge_efficiency"]),
            )
            discharge = min(
                max(demand - generation, 0.0),
                p["max_discharge_w"],
                max(0.0, energy) * 3600.0 * p["discharge_efficiency"] / h,
            )
            served = min(demand, generation + discharge)
            curtailed = max(0.0, generation - served - charge)
            battery_rate = charge * p["charge_efficiency"] - discharge / p["discharge_efficiency"]
            new_energy = energy + h * battery_rate / 3600.0
            battery_loss = charge * (1.0 - p["charge_efficiency"]) + discharge * (1.0 / p["discharge_efficiency"] - 1.0)
            conversion_loss = (generation - curtailed) * (1.0 / p["conversion_efficiency"] - 1.0)
            electrical_heat = served * p["load_heat_fraction"] + battery_loss + conversion_loss
            result.update(
                solar_generation_w=generation,
                load_demand_w=demand,
                load_served_w=served,
                unmet_load_w=demand - served,
                curtailed_power_w=curtailed,
                battery_charge_w=charge,
                battery_discharge_w=discharge,
                battery_energy_wh=new_energy,
                battery_soc=new_energy / p["battery_capacity_wh"],
                battery_loss_w=battery_loss,
                conversion_loss_w=conversion_loss,
                electrical_heat_w=electrical_heat,
                power_balance_residual_w=generation + discharge - served - charge - curtailed,
            )
        if self.thermal.get("enabled"):
            th = self.thermal
            direct = solar * self._area(np.array([sun_body]))[0] * th["solar_absorptivity"]
            rays, weights, surface_sun = earth_radiation(
                position, geometry["sun_pos_eci_km"], th["earth_quadrature_order"],
                numeric_backend="python",
            )
            weighted_area = self._area(rays @ c_bn.T) * weights / math.pi
            albedo = (
                th["solar_absorptivity"]
                * th["earth_albedo"]
                * th["solar_irradiance_w_m2"]
                * float(weighted_area @ surface_sun)
            )
            earth_ir = th["infrared_emissivity"] * th["earth_ir_w_m2"] * float(np.sum(weighted_area))
            internal = th["internal_heat_w"] + electrical_heat
            incoming = direct + albedo + earth_ir + internal
            capacity = th["heat_capacity_j_k"]
            radiation_coefficient = th["infrared_emissivity"] * SIGMA * th["radiating_area_m2"]
            rhs = old["temperature_k"] + h * incoming / capacity
            lo, hi = 0.0, rhs
            for _ in range(64):
                mid = (lo + hi) / 2.0
                if mid + h * radiation_coefficient * mid**4 / capacity > rhs:
                    hi = mid
                else:
                    lo = mid
            temperature = (lo + hi) / 2.0
            radiation = radiation_coefficient * temperature**4
            stored = capacity * (temperature - old["temperature_k"]) / h
            result.update(
                temperature_k=temperature,
                solar_heat_w=direct,
                albedo_heat_w=albedo,
                earth_ir_heat_w=earth_ir,
                internal_heat_w=internal,
                radiated_heat_w=radiation,
                stored_heat_w=stored,
                thermal_balance_residual_w=incoming - radiation - stored,
            )
        for key, rate in ENERGY_RATES.items():
            if rate in result:
                result[key + "_energy_j"] = old.get(key + "_energy_j", 0.0) + h * result[rate]
        return result
