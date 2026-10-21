"""Precision force descriptors and exact stage metadata for the native ONP plan.

Frame/resource policy stays with the existing Python owners. Rust evaluates
spacecraft-dependent acceleration using the same authoritative force kernels.
"""
from __future__ import annotations

import math
import os
from dataclasses import fields
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.environment import MOON_MU_KM3_S2, SUN_MU_KM3_S2
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.pro_perturbations.earth_radiation import REFERENCE_JD_UTC, EarthRadiationPressure, _quadrature
from sim.pro_perturbations.empirical import EmpiricalRICAcceleration
from sim.pro_perturbations.ocean_tides import OceanTides
from sim.pro_perturbations.schwarzschild import SchwarzschildAcceleration
from sim.pro_perturbations.solid_earth_tides import SolidEarthTides

# The public export omits this private owner. Do not hide a missing dependency
# or any other import failure inside an installed Pro implementation.
try:
    from sim.pro_perturbations.ecom import ECOM2Radiation
except ModuleNotFoundError as exc:
    if exc.name != "sim.pro_perturbations.ecom":
        raise
    ECOM2Radiation = None

_CLASSES = {
    EarthRadiationPressure: 9,
    SchwarzschildAcceleration: 10,
    SolidEarthTides: 11,
    OceanTides: 12,
    EmpiricalRICAcceleration: 13,
}
if ECOM2Radiation is not None:
    _CLASSES[ECOM2Radiation] = 14
_TIME_OWNERS = (FrameContext.at, eci_to_ecef_rotation_context, resolve_sun_moon_positions)


@lru_cache(maxsize=131072)
def _prepared_frame(context, time, resource, at_owner, rotation_owner):
    """Only immutable, state-independent frame rows; bind resources and hooks.

    Repeated OD integrations revisit identical epochs. Force coefficients and
    spacecraft state never enter this cache. The bound limits retained memory.
    """
    frame = at_owner(context, time)
    rotation = rotation_owner(time, frame)
    rotation.setflags(write=False)
    return frame, rotation


def precision_code(plugin):
    """Exact types only: subclasses retain their Python call contract."""
    code = _CLASSES.get(type(plugin))
    if code == 14:
        from sim.rust_orbit_backend import _extension
        return code if getattr(_extension(), "ONP_ECOM2_VERSION", 0) == 1 else None
    if code == 13:
        from sim.rust_orbit_backend import _extension
        if plugin.angular_frequency_rad_s and getattr(_extension(), "ONP_EMPIRICAL_RIC_HARMONIC_VERSION", 0)!=1:
            return None
        return code if getattr(_extension(), "ONP_EMPIRICAL_RIC_VERSION", 0) == 1 else None
    if code is not None and (code == 10 or type(plugin.frames) is FrameContext):
        return code
    return None


def precision_signature(plugin, ctx):
    code = precision_code(plugin)
    settings = tuple((f.name, getattr(plugin, f.name)) for f in fields(plugin) if f.init)
    tables = () if code != 12 else (id(plugin._factors), id(plugin._coefficients), plugin.coefficient_sha256)
    return code, settings, tables, float(ctx.area_m2)


def specifications(plugins, ctx):
    rows = []
    for plugin in plugins:
        code = precision_code(plugin)
        if code == 9:
            area = ctx.area_m2 if plugin.area_m2 is None else plugin.area_m2
            rows.append((code, [float(area), float(plugin.albedo), float(plugin.infrared)],
                         [array.ravel().tolist() for array in _quadrature(plugin.quadrature_order)]))
        elif code == 14:
            rows.append((code, list(plugin.coefficients_km_s2), []))
        elif code == 13:
            values=list(plugin.coefficients_km_s2)
            if plugin.angular_frequency_rad_s:
                values.extend([*plugin.cosine_km_s2,*plugin.sine_km_s2,plugin.angular_frequency_rad_s,plugin.reference_time_s])
            rows.append((code,values,[]))
        elif code == 10:
            rows.append((code, [], []))
        elif code == 11:
            rows.append((code, [float(plugin.tide_system == "zero_tide"), float(plugin.pole_tide), SUN_MU_KM3_S2, MOON_MU_KM3_S2], []))
        elif code == 12:
            rows.append((code, [float(plugin.degree), float(plugin.pole_tide)],
                         [plugin._factors.ravel().tolist(), plugin._coefficients.ravel().tolist()]))
    return rows


def wrap_stages(base, plugins, env, ctx, *, immutable_resources=None):
    selected = [plugin for plugin in plugins if precision_code(plugin) is not None]
    empty = np.eye(3).ravel().tolist() + [0.0] * 11 + [6378.137]
    # Cache only exact time-dependent frames. Resource signatures are checked
    # on each stage; spacecraft state, density and force outputs are never cached.
    cache = {}

    def stage(time, state):
        result = list(base(time, state))
        for plugin in selected:
            code = precision_code(plugin)
            if code == 14:
                row = list(empty)
                sun = result[13:16]
                if not any(sun):
                    sun, _ = resolve_sun_moon_positions(env, time)
                row[9:12] = list(sun)
                result.extend(row)
                continue
            if code in (10, 13):
                row=list(empty)
                if code == 13:
                    row[19] = float(time)
                result.extend(row)
                continue
            if code == 9:
                area = ctx.area_m2 if plugin.area_m2 is None else plugin.area_m2
                if not all(math.isfinite(v) for v in (area, ctx.cr, ctx.mass_kg)) or area < 0 or ctx.cr < 0 or ctx.mass_kg <= 0:
                    raise ValueError("Earth radiation requires positive mass and nonnegative finite area/Cr.")
                # Immutable rows may be reused after the fitted response moves
                # away from zero. Their geometry must not depend on that value.
                if (area == 0 or ctx.cr == 0) and immutable_resources is None:
                    result.extend(empty)
                    continue
            resource = None
            if plugin.frames.eop_path:
                path = os.path.expanduser(plugin.frames.eop_path)
                try:
                    stat = os.stat(path)
                    resource = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                except OSError:
                    resource = (path, None)
            key = (plugin.frames, float(time), resource)
            prepared = cache.get(key)
            if prepared is None:
                prepared = _prepared_frame(plugin.frames, float(time), resource,
                    FrameContext.at, eci_to_ecef_rotation_context)
                frame, rotation = prepared
                cache[key] = prepared
                while len(cache) > 8:
                    cache.pop(next(iter(cache)))
            frame, rotation = prepared
            sun = moon = np.zeros(3)
            if code in (9, 11):
                if (frame.jd_utc_start == env.get("jd_utc_start")
                        and any(result[13:16]) and any(result[16:19])):
                    # The shared base stage already evaluated these exact
                    # ephemerides at this epoch. Preserve separate-frame
                    # epochs and missing-body cases on the reference owner.
                    sun, moon = np.asarray(result[13:16]), np.asarray(result[16:19])
                else:
                    sun, moon = resolve_sun_moon_positions(dict(env, jd_utc_start=frame.jd_utc_start, _rust_numeric_backend="rust"), time)
                sun, moon = rotation @ sun, rotation @ moon
            jd = frame.jd_utc_start + float(time) / 86400
            elapsed = (frame.jd_utc_start - REFERENCE_JD_UTC) * 86400 + float(time) + frame.tt_minus_utc_s - 52.184
            result.extend(rotation.ravel().tolist() + sun.tolist() + moon.tolist() + [
                jd + frame.tt_minus_utc_s / 86400, jd + (frame.dut1_s or 0.0) / 86400,
                float(frame.xp_arcsec or 0.0), float(frame.yp_arcsec or 0.0), elapsed,
                float(env.get("spherical_harmonics_reference_radius_km", 6378.137)),
            ])
        return result

    # Only canonical, native, time-only input owners qualify. Custom hooks and
    # atmosphere retain ordinary per-state callback semantics and call order.
    current=(FrameContext.at,eci_to_ecef_rotation_context,resolve_sun_moon_positions)
    canonical=all(a is b and Path(a.__code__.co_filename).resolve().parent == Path(__file__).resolve().parent
                  for a,b in zip(current,_TIME_OWNERS,strict=True))
    if immutable_resources is not None and canonical and all(precision_code(p) in {9,10,11,12,13,14} for p in selected):
        resources=tuple(sorted(set(immutable_resources) | {os.path.expanduser(p.frames.eop_path)
            for p in selected if hasattr(p,"frames") and p.frames.eop_path}
            | {str(p.coeff_path) for p in selected if precision_code(p)==12}))
        settings=tuple((precision_code(p),None if precision_code(p)==14 else
                        tuple((f.name,getattr(p,f.name)) for f in fields(p) if f.init
                              and not (precision_code(p)==9 and f.name=="area_m2"))) for p in selected)
        stage._oel_time_only_key=(base,settings,current)
        stage._oel_time_only_resources=resources
    return stage


def immutable_sampled_stages(propagator, callback, state, times, widths, sample_ends=None):
    """Bounded exact-stage reuse across state/radiation sensitivity trials.

    The native table checks resources at every RK stage and restores the
    maintained callback on drift. Only the owner-generated metadata above can
    admit a time-only stage table; custom/state-dependent paths stay unchanged.
    """
    key=getattr(callback,"_oel_time_only_key",None)
    if key is None:
        return callback
    if (times.shape!=widths.shape or np.any(~np.isfinite(times)) or np.any(~np.isfinite(widths))
            or np.any(widths<=0) or np.any(np.diff(times)<0)
            or np.any((times==0)&np.signbit(times))):
        return callback  # Preserve ordinary schedule validation/error ownership.
    if sample_ends and (sample_ends[0]<=0 or sample_ends[-1]>len(widths)
                        or any(a>=b for a,b in zip(sample_ends,sample_ends[1:]))):
        return callback
    from sim.dynamics.orbit.rust_force_plan import _resource_signature
    from sim.rust_orbit_backend import _extension
    constructor=getattr(_extension(),"ONPImmutableStageTable",None)
    if constructor is None:
        return callback
    capacity=getattr(_extension(),"ONP_IMMUTABLE_STAGE_TABLE_MAX_EPOCHS",131072)
    if type(capacity) is not int or not 131072<=capacity<=262144 or len(times)>capacity:
        return callback
    paths=callback._oel_time_only_resources
    resources=tuple(_resource_signature(path) for path in paths)
    signature=(key,resources,times.tobytes(),widths.tobytes())
    cached=getattr(propagator,"_rust_precision_sampled_stage_cache",None)
    if cached is not None and cached[0]==signature:
        return cached[1]
    epochs=np.unique(np.r_[times,times+.5*widths,times+widths])
    if len(epochs)>capacity:
        return callback
    rows=[callback(float(t),state) for t in epochs]
    # Refuse a mixed-resource preparation snapshot, preserving ordinary policy.
    if resources!=tuple(_resource_signature(path) for path in paths):
        return callback
    table=constructor(epochs.tolist(),rows,list(paths),callback)
    propagator._rust_precision_sampled_stage_cache=(signature,table)
    return table
