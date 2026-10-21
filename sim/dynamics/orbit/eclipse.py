from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM, SUN_RADIUS_KM
from sim.dynamics.orbit.epoch import AU_KM, resolve_sun_moon_positions


def ellipsoid_limb_angle(r_fixed_km, sun_fixed_km, equatorial_radius_km, flattening=1.0 / 298.257223563):
    """Apparent limb in the observer/centre/Sun plane of an oblate Earth.

    Intersect the ellipsoid quadratic with that plane, then solve the tangent
    condition r.T Q limb = 1. The positive transverse root faces the Sun.
    The disc-overlap approximation remains the same as the conical model.
    """
    r = np.asarray(r_fixed_km, dtype=float)
    sun = np.asarray(sun_fixed_km, dtype=float)
    distance = float(np.linalg.norm(r))
    u = r / distance
    transverse = sun - float(sun @ u) * u
    if np.linalg.norm(transverse) <= 1e-12 * max(float(np.linalg.norm(sun)), 1.0):
        axis = np.eye(3)[int(np.argmin(abs(u)))]
        transverse = axis - float(axis @ u) * u
    v = transverse / np.linalg.norm(transverse)
    q = np.array([1.0, 1.0, (1.0 - flattening) ** -2]) / equatorial_radius_km**2
    a, b, c = float((u * q) @ u), float((u * q) @ v), float((v * q) @ v)
    if a * distance**2 <= 1.0:
        return float(np.pi / 2)
    y = float(np.sqrt((1.0 - 1.0 / (a * distance**2)) / (c - b * b / a)))
    x = (1.0 / distance - b * y) / a
    return float(np.arctan2(y, distance - x))


def _resolve_sun_position_eci_km(env: dict, t_s: float) -> np.ndarray:
    if "sun_pos_eci_km" in env:
        return np.asarray(env["sun_pos_eci_km"], dtype=float)
    try:
        sun, _ = resolve_sun_moon_positions(env, t_s)
        if np.linalg.norm(sun) > 0.0:
            return np.asarray(sun, dtype=float)
    except RuntimeError:
        pass
    sun_dir = np.asarray(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0], dtype=float)), dtype=float)
    n = float(np.linalg.norm(sun_dir))
    if n <= 0.0:
        return np.array([AU_KM, 0.0, 0.0], dtype=float)
    return (sun_dir / n) * AU_KM


def resolve_srp_geometry(r_sc_eci_km: np.ndarray, t_s: float, env: dict) -> dict[str, object]:
    r_sc = np.asarray(r_sc_eci_km, dtype=float).reshape(3)
    r_norm2 = float(np.dot(r_sc, r_sc))
    r_norm = float(np.sqrt(r_norm2)) if r_norm2 > 0.0 else 0.0

    r_sun = _resolve_sun_position_eci_km(env, t_s).reshape(3)
    sun_norm2 = float(np.dot(r_sun, r_sun))
    sun_norm = float(np.sqrt(sun_norm2)) if sun_norm2 > 0.0 else 0.0

    rho = r_sun - r_sc
    rho_norm2 = float(np.dot(rho, rho))
    rho_norm = float(np.sqrt(rho_norm2)) if rho_norm2 > 0.0 else 0.0
    if rho_norm > 0.0:
        sun_dir_sc_eci = rho / rho_norm
        distance_scale = float((AU_KM / rho_norm) ** 2)
    else:
        sun_dir_sc_eci = np.zeros(3, dtype=float)
        distance_scale = 1.0

    return {
        "r_sc_eci_km": r_sc,
        "r_sc_norm_km": r_norm,
        "sun_pos_eci_km": r_sun,
        "sun_pos_norm_km": sun_norm,
        "rho_sc_to_sun_km": rho,
        "rho_norm_km": rho_norm,
        "sun_dir_sc_eci": sun_dir_sc_eci,
        "distance_scale": distance_scale,
    }


def _finite_disc_illumination(alpha: float, beta: float, gamma: float) -> float:
    """Return the visible fraction of the solar disc after circular overlap."""

    if beta <= 0.0:
        return 1.0
    if gamma >= alpha + beta:
        return 1.0
    if alpha > beta and gamma <= alpha - beta:
        return 0.0
    if beta > alpha and gamma <= beta - alpha:
        return float(max(0.0, 1.0 - (alpha * alpha) / (beta * beta)))
    if gamma <= 0.0:
        return 0.0 if alpha >= beta else float(max(0.0, 1.0 - (alpha * alpha) / (beta * beta)))

    # The penumbra is the complement of the overlap of the apparent Earth
    # and Sun discs. The two circular-segment terms minus the shared triangle
    # give the overlap area; divide by the full solar-disc area for illumination.
    denominator_earth = 2.0 * gamma * alpha
    denominator_sun = 2.0 * gamma * beta
    earth_angle = float(
        np.arccos(np.clip((gamma * gamma + alpha * alpha - beta * beta) / denominator_earth, -1.0, 1.0))
    )
    sun_angle = float(
        np.arccos(np.clip((gamma * gamma + beta * beta - alpha * alpha) / denominator_sun, -1.0, 1.0))
    )
    radicand = max(
        0.0,
        (-gamma + alpha + beta)
        * (gamma + alpha - beta)
        * (gamma - alpha + beta)
        * (gamma + alpha + beta),
    )
    overlap = (
        alpha * alpha * earth_angle
        + beta * beta * sun_angle
        - 0.5 * float(np.sqrt(radicand))
    )
    illumination = 1.0 - overlap / (float(np.pi) * beta * beta)
    return float(np.clip(illumination, 0.0, 1.0))


def srp_shadow_factor(
    r_sc_eci_km: np.ndarray,
    t_s: float,
    env: dict,
    earth_radius_km: float = EARTH_RADIUS_KM,
    sun_radius_km: float = SUN_RADIUS_KM,
    srp_geometry: dict[str, object] | None = None,
) -> float:
    """
    Returns illumination factor in [0, 1] for SRP.

    - 1.0: full sunlight
    - 0.0: full umbra
    - (0,1): penumbra transition
    """
    model = str(env.get("srp_shadow_model", "conical")).lower()
    if model in ("none", "off", "disabled"):
        return 1.0

    geometry = resolve_srp_geometry(r_sc_eci_km, t_s, env) if srp_geometry is None else srp_geometry
    r_sc = np.asarray(geometry["r_sc_eci_km"], dtype=float)
    r_norm = float(geometry["r_sc_norm_km"])
    if r_norm <= earth_radius_km:
        return 0.0

    rho_norm = float(geometry["rho_norm_km"])
    if rho_norm <= 0.0:
        return 1.0

    r_sun = np.asarray(geometry["sun_pos_eci_km"], dtype=float)
    sun_norm = float(geometry["sun_pos_norm_km"])
    s_hat = None
    if sun_norm > 0.0:
        s_hat = r_sun / sun_norm
        if float(np.dot(r_sc, s_hat)) >= 0.0:
            return 1.0

    if model in ("cylindrical", "cylinder"):
        if s_hat is None:
            s_hat = r_sun / max(sun_norm, 1e-12)
        r_sc_along_sun = float(np.dot(r_sc, s_hat))
        if r_sc_along_sun >= 0.0:
            return 1.0
        cross_track2 = max(0.0, float(np.dot(r_sc, r_sc)) - r_sc_along_sun * r_sc_along_sun)
        return 0.0 if cross_track2 < earth_radius_km * earth_radius_km else 1.0

    # Conical angular model (umbra + penumbra).
    # Apparent angular radii as seen from spacecraft.
    alpha = float(np.arcsin(np.clip(earth_radius_km / r_norm, -1.0, 1.0)))
    if model == "conical_ellipsoid":
        from sim.dynamics.orbit.frames import eci_to_ecef_rotation_context, frame_context_from_environment

        rotation = eci_to_ecef_rotation_context(t_s, frame_context_from_environment(env))
        alpha = ellipsoid_limb_angle(rotation @ r_sc, rotation @ r_sun, earth_radius_km)
    beta = float(np.arcsin(np.clip(sun_radius_km / rho_norm, -1.0, 1.0)))
    u_earth = -r_sc / r_norm
    u_sun = np.asarray(geometry["sun_dir_sc_eci"], dtype=float)
    gamma = float(np.arccos(np.clip(float(np.dot(u_earth, u_sun)), -1.0, 1.0)))

    if gamma >= alpha + beta:
        return 1.0

    # Complete occultation of Sun disk by Earth disk.
    if alpha > beta and gamma <= (alpha - beta):
        return 0.0

    # Rare annular-center case (Earth disk inside Sun disk).
    min_illum = 0.0
    if beta > alpha and gamma <= (beta - alpha):
        min_illum = max(0.0, 1.0 - (alpha * alpha) / (beta * beta))
        return float(min_illum)

    return _finite_disc_illumination(alpha, beta, gamma)
