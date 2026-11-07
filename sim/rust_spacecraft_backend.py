"""Optional Rust kernels for spacecraft mesh geometry and resources.

The adapter deliberately keeps Python responsible for file intake, scenario
validation, environment resolution, geometry-profile lookup and serialized
resource dictionaries.  Rust is selected explicitly and receives only finite,
already-normalized numerical arrays.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM


def _extension():
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:  # pragma: no cover - exercised without wheel
        raise RuntimeError(
            "Rust spacecraft backend requested but oel_rust_orbit is unavailable; "
            "install the optional Rust orbit wheel first"
        ) from exc


def _pack_f64(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(order="C")


def _unpack_f64(values: bytes) -> np.ndarray:
    return np.frombuffer(values, dtype="<f8").copy()


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def facet_projected_geometry(
    triangles_body_m: Any,
    directions_body: Any,
    normals_body: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected area and center of pressure for every direction."""

    triangles = np.asarray(triangles_body_m, dtype=np.float64)
    directions = np.asarray(directions_body, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError("triangles_body_m must be a nonempty Nx3x3 array")
    if len(triangles) > 100_000:
        raise ValueError("triangles_body_m must contain at most 100000 triangles")
    if directions.ndim != 2 or directions.shape[1] != 3 or len(directions) == 0:
        raise ValueError("directions_body must be a nonempty Nx3 array")
    if not np.all(np.isfinite(triangles)) or not np.all(np.isfinite(directions)):
        raise ValueError("triangles_body_m and directions_body must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("directions_body must contain nonzero vectors")
    if normals_body is None:
        normals = np.empty(0, dtype=np.float64)
    else:
        normals = np.asarray(normals_body, dtype=np.float64)
        if normals.shape != (len(triangles), 3) or not np.all(np.isfinite(normals)):
            raise ValueError("normals_body must be finite with shape (triangle_count, 3)")
    extension = _extension()
    function = getattr(extension, "spacecraft_facet_projected_batch", None)
    if function is None:
        raise RuntimeError("installed Rust wheel does not provide spacecraft facet kernels")
    byte_function = getattr(extension, "spacecraft_facet_projected_batch_bytes", None)
    if byte_function is not None:
        values = _unpack_f64(byte_function(_pack_f64(triangles), _pack_f64(normals), _pack_f64(directions)))
    else:
        flat = function(triangles.reshape(-1).tolist(), normals.reshape(-1).tolist(), directions.reshape(-1).tolist())
        values = np.asarray(flat, dtype=np.float64)
    if values.shape != (len(directions) * 4,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Rust spacecraft facet kernel returned an invalid result")
    return values[::4].copy(), values.reshape(len(directions), 4)[:, 1:].copy()


def silhouette_batch(triangles_body_m: Any, directions_body: Any, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate occlusion-aware raster silhouettes for a direction batch."""

    triangles = np.asarray(triangles_body_m, dtype=np.float64)
    directions = np.asarray(directions_body, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError("triangles_body_m must be a nonempty Nx3x3 array")
    if len(triangles) > 100_000:
        raise ValueError("triangles_body_m must contain at most 100000 triangles")
    if directions.ndim != 2 or directions.shape[1] != 3 or len(directions) == 0:
        raise ValueError("directions_body must be a nonempty Nx3 array")
    if not 16 <= int(resolution) <= 1024:
        raise ValueError("raster resolution must be 16..1024")
    if not np.all(np.isfinite(triangles)) or not np.all(np.isfinite(directions)):
        raise ValueError("triangles_body_m and directions_body must be finite")
    extension = _extension()
    function = getattr(extension, "spacecraft_silhouette_batch", None)
    if function is None:
        raise RuntimeError("installed Rust wheel does not provide spacecraft silhouette kernels")
    byte_function = getattr(extension, "spacecraft_silhouette_batch_bytes", None)
    if byte_function is not None:
        values = _unpack_f64(byte_function(_pack_f64(triangles), _pack_f64(directions), int(resolution)))
    else:
        flat = function(triangles.reshape(-1).tolist(), directions.reshape(-1).tolist(), int(resolution))
        values = np.asarray(flat, dtype=np.float64)
    if values.shape != (len(directions) * 4,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Rust spacecraft silhouette kernel returned an invalid result")
    return values[::4].copy(), values.reshape(len(directions), 4)[:, 1:].copy()


def earth_radiation(position: Any, sun_position: Any, order: int):
    """Run the Rust apparent-disk Earth-radiation quadrature."""

    position_array = _finite_array(position, (3,), "position")
    sun_array = _finite_array(sun_position, (3,), "sun_position")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or not 4 <= order <= 64:
        raise ValueError("Earth-radiation quadrature order must be an integer in [4, 64]")
    if float(np.linalg.norm(position_array)) <= EARTH_RADIUS_KM:
        raise ValueError("spacecraft resources require position above the Earth surface")
    extension = _extension()
    function = getattr(extension, "resource_earth_radiation", None)
    if function is None:
        raise RuntimeError("installed Rust wheel does not provide resource kernels")
    byte_function = getattr(extension, "resource_earth_radiation_bytes", None)
    if byte_function is not None:
        raw = byte_function(position_array.tolist(), sun_array.tolist(), int(order))
        rays, weights, illumination = (_unpack_f64(value) for value in raw)
        rays = rays.reshape(-1, 3)
    else:
        rays, weights, illumination = function(position_array.tolist(), sun_array.tolist(), int(order))
        rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(weights, dtype=np.float64)
        illumination = np.asarray(illumination, dtype=np.float64)
    if len(rays) != len(weights) or len(rays) != len(illumination):
        raise RuntimeError("Rust Earth-radiation kernel returned inconsistent lengths")
    if not np.all(np.isfinite(rays)) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(illumination)):
        raise RuntimeError("Rust Earth-radiation kernel returned non-finite values")
    return rays, weights, illumination


def advance(model, start, end, environment: dict, dt_s: float) -> dict[str, float]:
    """Advance one validated resource interval through native arithmetic."""

    from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
    from sim.utils.quaternion import quaternion_to_dcm_bn

    h = float(dt_s)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("resource timestep must be finite and positive")
    old = start.resource_state if start.resource_state is not None else model.initial_state()
    position = (np.asarray(start.position_eci_km, dtype=float) + np.asarray(end.position_eci_km, dtype=float)) / 2.0
    # Validate the public adapter inputs before eclipse geometry resolution.
    # The fused native path must preserve the same direct API errors as the
    # general Earth-radiation path for invalid finite vectors.
    position = _finite_array(position, (3,), "position")
    if "sun_pos_eci_km" in environment:
        _finite_array(environment["sun_pos_eci_km"], (3,), "sun_position")
    q0 = np.asarray(start.attitude_quat_bn, dtype=float)
    q1 = np.asarray(end.attitude_quat_bn, dtype=float)
    q = q0 + (q1 if float(np.dot(q0, q1)) >= 0.0 else -q1)
    q = q / np.linalg.norm(q)
    c_bn = quaternion_to_dcm_bn(q)
    t = (float(start.t_s) + float(end.t_s)) / 2.0
    geometry = resolve_srp_geometry(position, t, environment)
    shadow = srp_shadow_factor(position, t, environment, srp_geometry=geometry)
    sun_body = c_bn @ np.asarray(geometry["sun_dir_sc_eci"], dtype=float)
    settings = model.thermal if model.thermal.get("enabled") else model.power
    solar_irradiance = float(settings["solar_irradiance_w_m2"])
    direct_area = float(model._area(np.asarray([sun_body], dtype=float))[0]) if model.thermal.get("enabled") else 0.0

    extension = _extension()
    isotropic_function = getattr(extension, "resource_advance_isotropic", None)
    isotropic = isotropic_function is not None and model.thermal.get("area_mode") != "geometry"
    ray_data = np.empty(0, dtype=np.float64)
    if model.thermal.get("enabled") and isotropic:
        # Preserve the general Earth-radiation adapter's validation errors
        # and ordering before fusing its numerical work into native advance.
        _finite_array(position, (3,), "position")
        _finite_array(geometry["sun_pos_eci_km"], (3,), "sun_position")
    if model.thermal.get("enabled") and not isotropic:
        rays, weights, surface_sun = earth_radiation(
            position,
            np.asarray(geometry["sun_pos_eci_km"], dtype=float),
            int(model.thermal["earth_quadrature_order"]),
        )
        directions_body = rays @ c_bn.T
        areas = np.asarray(model._area(directions_body), dtype=np.float64)
        ray_data = np.column_stack((areas, weights, surface_sun, np.zeros(len(areas)))).reshape(-1)

    thermal_values = []
    if model.thermal.get("enabled"):
        thermal_values = [
            float(model.thermal["solar_absorptivity"]),
            float(model.thermal["infrared_emissivity"]),
            float(model.thermal.get("projected_area_m2", 0.0)),
            float(model.thermal["radiating_area_m2"]),
            float(model.thermal["internal_heat_w"]),
            float(model.thermal["earth_albedo"]),
            float(model.thermal["earth_ir_w_m2"]),
            solar_irradiance,
            float(model.thermal["heat_capacity_j_k"]),
        ]
    power_values = []
    panel_values = []
    if model.power.get("enabled"):
        power_values = [
            float(model.power["baseline_load_w"]),
            float(model.power["battery_capacity_wh"]),
            float(model.power["charge_efficiency"]),
            float(model.power["discharge_efficiency"]),
            float(model.power["conversion_efficiency"]),
            float(model.power["max_charge_w"]),
            float(model.power["max_discharge_w"]),
            float(model.power["load_heat_fraction"]),
            solar_irradiance,
        ]
        for panel in model.power["panels"]:
            panel_values.extend(
                [
                    float(panel["area_m2"]),
                    *np.asarray(panel["normal_body"], dtype=float).tolist(),
                    float(panel["efficiency"]),
                ]
            )
    function = getattr(extension, "resource_advance", None)
    if function is None:
        raise RuntimeError("installed Rust wheel does not provide resource kernels")
    arguments = (
        float(old.get("temperature_k", 0.0)),
        float(old.get("battery_energy_wh", 0.0)),
        bool(model.thermal.get("enabled")),
        bool(model.power.get("enabled")),
        thermal_values, power_values, panel_values, sun_body.tolist(),
        float(geometry["distance_scale"]), float(shadow), h, direct_area,
    )
    if isotropic:
        flat = isotropic_function(
            *arguments, position.tolist(),
            np.asarray(geometry["sun_pos_eci_km"], dtype=float).tolist(),
            int(model.thermal["earth_quadrature_order"]) if model.thermal.get("enabled") else 4,
        )
    else:
        flat = function(*arguments, ray_data.tolist())
    values = np.asarray(flat, dtype=np.float64)
    if values.shape != (22,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Rust resource kernel returned an invalid result")
    (
        generation,
        demand,
        served,
        unmet,
        curtailed,
        charge,
        discharge,
        battery_energy,
        battery_soc,
        battery_loss,
        conversion_loss,
        electrical_heat,
        power_residual,
        direct_heat,
        albedo_heat,
        earth_ir_heat,
        internal_heat,
        _incoming,
        temperature,
        radiated,
        stored,
        thermal_residual,
    ) = values.tolist()
    result = dict(old)
    result.update(
        time_s=float(end.t_s),
        interval_start_s=float(start.t_s),
        sunlit_fraction=float(shadow),
        solar_irradiance_w_m2=float(solar_irradiance * float(geometry["distance_scale"]) * float(shadow)),
    )
    if model.power.get("enabled"):
        result.update(
            solar_generation_w=generation,
            load_demand_w=demand,
            load_served_w=served,
            unmet_load_w=unmet,
            curtailed_power_w=curtailed,
            battery_charge_w=charge,
            battery_discharge_w=discharge,
            battery_energy_wh=battery_energy,
            battery_soc=battery_soc,
            battery_loss_w=battery_loss,
            conversion_loss_w=conversion_loss,
            electrical_heat_w=electrical_heat,
            power_balance_residual_w=power_residual,
        )
    if model.thermal.get("enabled"):
        result.update(
            temperature_k=temperature,
            solar_heat_w=direct_heat,
            albedo_heat_w=albedo_heat,
            earth_ir_heat_w=earth_ir_heat,
            internal_heat_w=internal_heat,
            radiated_heat_w=radiated,
            stored_heat_w=stored,
            thermal_balance_residual_w=thermal_residual,
        )
    from sim.spacecraft_resources.model import ENERGY_RATES

    for key, rate in ENERGY_RATES.items():
        if rate in result:
            result[key + "_energy_j"] = old.get(key + "_energy_j", 0.0) + h * result[rate]
    return result
