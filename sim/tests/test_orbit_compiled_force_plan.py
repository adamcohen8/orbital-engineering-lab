from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

import sim.dynamics.orbit.propagator as propagator_module
from sim.acceleration.settings import acceleration_context, acceleration_settings_from_mode
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    lift_plugin,
    spherical_harmonics_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_planets_plugin,
    third_body_sun_plugin,
)
from sim.dynamics.orbit.spherical_harmonics import (
    SphericalHarmonicTerm,
    compile_spherical_harmonic_terms,
)


def _case() -> tuple[np.ndarray, dict, OrbitContext]:
    terms = [
        SphericalHarmonicTerm(
            n=2,
            m=0,
            c_nm=-4.841693259705e-4,
            normalized=True,
        )
    ]
    env = {
        "_compiled_spherical_harmonics_terms": compile_spherical_harmonic_terms(terms),
        "_parsed_spherical_harmonics_terms": terms,
        "spherical_harmonics_terms": terms,
        "spherical_harmonics_reference_radius_km": 6378.1363,
        "spherical_harmonics_frame_model": "iau76_80_eop",
        "density_frame_model": "iau76_80_eop",
        "drag_frame_model": "iau76_80_eop",
        "geodetic_model": "wgs84",
        "atmosphere_model": "nrlmsise00",
        "jd_utc_start": 2459669.5,
        "dut1_s": 0.1,
        "xp_arcsec": 0.05,
        "yp_arcsec": -0.03,
        "dat_s": 37.0,
        "tt_minus_utc_s": 69.184,
        "f107": 150.0,
        "f107a": 150.0,
        "ap": 4.0,
        "nrlmsise00_ap_a": [4.0] * 7,
        "srp_shadow_model": "conical",
        "sun_pos_eci_km": np.array([1.49e8, 2.0e6, -1.0e6], dtype=float),
        "moon_pos_eci_km": np.array([3.7e5, -8.0e4, 3.0e4], dtype=float),
    }
    state = np.array([700.0, 4400.0, 5100.0, -7.3, -1.1, 1.9], dtype=float)
    context = OrbitContext(mu_km3_s2=398600.4415, mass_kg=300.0, area_m2=1.0, cd=2.2, cr=1.2)
    return state, env, context


def _require_compiled_acceleration() -> None:
    settings = acceleration_settings_from_mode("auto", allow_env_override=False)
    if not settings.enabled:
        pytest.skip("compiled force-plan instrumentation requires the optional accel backend")


def _assert_compiled_steps_are_numerically_equivalent(plugins: list, step_count: int = 12) -> None:
    _require_compiled_acceleration()
    state, env, context = _case()
    command = np.zeros(3, dtype=float)
    with acceleration_context("auto", allow_env_override=False):
        reference = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto")
        assert reference._builtin_rk4_fast_path_enabled()
        accelerated = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto")
        for step in range(step_count):
            t_s = float(step * 10)
            expected = reference._propagate_builtin_rk4(
                x_eci=state,
                dt_s=10.0,
                t_s=t_s,
                command_accel_eci_km_s2=command,
                env=env,
                ctx=context,
            )
            actual = accelerated.propagate(state, 10.0, t_s, command, env, context)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-11)
            state = expected


def test_compiled_force_plan_is_numerically_equivalent_for_drag_and_full_force_subsets() -> None:
    _assert_compiled_steps_are_numerically_equivalent([spherical_harmonics_plugin, drag_plugin])
    _assert_compiled_steps_are_numerically_equivalent(
        [
            spherical_harmonics_plugin,
            drag_plugin,
            srp_plugin,
            third_body_sun_plugin,
            third_body_moon_plugin,
        ]
    )


@pytest.mark.parametrize("solar_input", ["explicit", "sampled", "analytic", "override"])
def test_fused_nrlmsise_uses_common_frame_and_current_solar_inputs(solar_input):
    _require_compiled_acceleration()
    state, env, context = _case()
    # This supported fused envelope uses simple drag velocity and a full EOP
    # density frame. The default EOP drag evaluator intentionally falls back.
    env["drag_frame_model"] = "simple"
    env.pop("moon_pos_eci_km")
    if solar_input == "sampled":
        env["sun_ephemeris_time_s"] = [0., 300.]
        env["sun_ephemeris_eci_km"] = [env["sun_pos_eci_km"], -env["sun_pos_eci_km"]]
    elif solar_input == "analytic":
        env.pop("sun_pos_eci_km")
    elif solar_input == "override":
        env.pop("sun_pos_eci_km")
        env["nrlmsise00_lst_hr"] = 6.0
    fused = OrbitPropagator(plugins=[drag_plugin], acceleration_mode="auto")
    reference = OrbitPropagator(plugins=[drag_plugin], acceleration_mode="off")
    with acceleration_context("auto", allow_env_override=False):
        for index in range(3):
            if solar_input == "override" and index:
                env["nrlmsise00_lst_hr"] += 6.0
            expected = reference._propagate_builtin_rk4(
                x_eci=state, dt_s=10., t_s=index*10., command_accel_eci_km_s2=np.zeros(3), env=env, ctx=context)
            actual = fused._try_propagate_compiled_builtin_rk4(
                x_eci=state, dt_s=10., t_s=index*10., command_accel_eci_km_s2=np.zeros(3), env=env, ctx=context)
            assert actual is not None
            np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-11)
            state = expected


def test_fused_nrlmsise_honors_environment_drag_coefficient_override() -> None:
    _require_compiled_acceleration()
    state, env, context = _case()
    env["drag_frame_model"] = "simple"
    env["drag_coefficient"] = 0.7
    fused = OrbitPropagator(plugins=[drag_plugin], acceleration_mode="auto")
    reference = OrbitPropagator(plugins=[drag_plugin], acceleration_mode="off")
    with acceleration_context("auto", allow_env_override=False):
        expected = reference._propagate_builtin_rk4(
            x_eci=state,
            dt_s=10.0,
            t_s=0.0,
            command_accel_eci_km_s2=np.zeros(3),
            env=env,
            ctx=context,
        )
        actual = fused._try_propagate_compiled_builtin_rk4(
            x_eci=state,
            dt_s=10.0,
            t_s=0.0,
            command_accel_eci_km_s2=np.zeros(3),
            env=env,
            ctx=context,
        )
    assert actual is not None
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-11)


def test_fused_solar_endpoint_reloads_rewritten_eop_and_rejects_deleted_file(tmp_path):
    _require_compiled_acceleration()
    state, env, context = _case()
    env.pop("sun_pos_eci_km")
    env.pop("moon_pos_eci_km")
    env["drag_frame_model"] = "simple"
    path = tmp_path / "eop.txt"
    path.write_text("NUM_OBSERVED_POINTS 2\n"
                    "2022 03 31 59669 .1 .2 -.1 0 0 0 0 0 37\n"
                    "2022 04 01 59670 .1 .2 -.1 0 0 0 0 0 37\n")
    env["density_eop_path"] = str(path)
    fused = OrbitPropagator(plugins=[drag_plugin], acceleration_mode="auto")
    reference = OrbitPropagator(plugins=[drag_plugin], acceleration_mode="off")
    args = dict(x_eci=state, dt_s=10., command_accel_eci_km_s2=np.zeros(3), env=env, ctx=context)
    with acceleration_context("auto", allow_env_override=False):
        assert fused._try_propagate_compiled_builtin_rk4(t_s=0., **args) is not None
        old_key = fused._compiled_endpoint_cache_key
        path.write_text(path.read_text().replace(".1 .2", "1.1 2.2"))
        assert fused._compiled_endpoint_environment_signature(env, (drag_plugin,)) != old_key[1]
        expected = reference._propagate_builtin_rk4(t_s=10., **args)
        actual = fused._try_propagate_compiled_builtin_rk4(t_s=10., **args)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-11)
        path.unlink()
        with pytest.raises(FileNotFoundError):
            fused._try_propagate_compiled_builtin_rk4(t_s=20., **args)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_fused_nonfinite_solar_override_keeps_reference_handling(value):
    _require_compiled_acceleration()
    state, env, context = _case()
    env.update(drag_frame_model="simple", nrlmsise00_lst_hr=value)
    prop = OrbitPropagator(plugins=[drag_plugin], acceleration_mode="auto")
    with acceleration_context("auto", allow_env_override=False):
        assert prop._try_propagate_compiled_builtin_rk4(
            x_eci=state, dt_s=10., t_s=0., command_accel_eci_km_s2=np.zeros(3), env=env, ctx=context) is None


def test_eop_drag_avoids_the_incomplete_compiled_rotation_path() -> None:
    _require_compiled_acceleration()
    state, env, context = _case()
    command = np.zeros(3, dtype=float)
    plugins = [spherical_harmonics_plugin, drag_plugin]
    with acceleration_context("auto", allow_env_override=False):
        reference = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto")
        expected = reference._propagate_builtin_rk4(
            x_eci=state,
            dt_s=10.0,
            t_s=0.0,
            command_accel_eci_km_s2=command,
            env=env,
            ctx=context,
        )
        accelerated = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto")
        original_rotation = OrbitPropagator._accelerated_compiled_rotation

        rotation_calls: list[float] = []

        def counted_rotation(env, t_s, *, model_key, path_key):
            rotation_calls.append(float(t_s))
            return original_rotation(env, t_s, model_key=model_key, path_key=path_key)

        with patch.object(OrbitPropagator, "_accelerated_compiled_rotation", staticmethod(counted_rotation)):
            actual = accelerated.propagate(state, 10.0, 0.0, command, env, context)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-11)
    assert rotation_calls == []


def test_compiled_force_plan_reuses_exact_previous_endpoint_environment() -> None:
    _require_compiled_acceleration()
    state, env, context = _case()
    env = dict(env)
    env.pop("sun_pos_eci_km")
    env.pop("moon_pos_eci_km")
    env["ephemeris_mode"] = "analytic_simple"
    # Exercise the compiled endpoint cache with its supported fixed-z drag
    # frame; the full EOP rotation-derivative path deliberately stays on the
    # authoritative Python evaluator.
    env["drag_frame_model"] = "simple"
    command = np.zeros(3, dtype=float)
    plugins = [
        spherical_harmonics_plugin,
        drag_plugin,
        srp_plugin,
        third_body_sun_plugin,
        third_body_moon_plugin,
    ]
    with acceleration_context("auto", allow_env_override=False):
        reference = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto", numeric_backend="python")
        expected_first = reference._propagate_builtin_rk4(
            x_eci=state,
            dt_s=10.0,
            t_s=0.0,
            command_accel_eci_km_s2=command,
            env=env,
            ctx=context,
        )
        expected_second = reference._propagate_builtin_rk4(
            x_eci=expected_first,
            dt_s=10.0,
            t_s=10.0,
            command_accel_eci_km_s2=command,
            env=env,
            ctx=context,
        )

        accelerated = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto", numeric_backend="python")
        original_resolver = propagator_module.resolve_sun_moon_positions
        with patch.object(
            propagator_module,
            "resolve_sun_moon_positions",
            wraps=original_resolver,
        ) as resolver:
            actual_first = accelerated.propagate(state, 10.0, 0.0, command, env, context)
            actual_second = accelerated.propagate(actual_first, 10.0, 10.0, command, env, context)

    np.testing.assert_allclose(actual_first, expected_first, rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(actual_second, expected_second, rtol=0.0, atol=2.0e-11)
    assert resolver.call_count == 5


def test_compiled_endpoint_cache_rejects_spice_configuration_and_callbacks() -> None:
    plugins = (srp_plugin, third_body_sun_plugin, third_body_moon_plugin)
    spice_env = {
        "ephemeris_mode": "spice",
        "jd_utc_start": 2459669.5,
        "spice_kernels": ["de440s.bsp"],
        "spice_frame": "J2000",
    }

    assert OrbitPropagator._compiled_endpoint_environment_signature(spice_env, plugins) is None
    assert (
        OrbitPropagator._compiled_endpoint_environment_signature(
            {
                "ephemeris_mode": "analytic_enhanced",
                "jd_utc_start": 2459669.5,
                "spice_ephemeris_callable": lambda _jd, _env: {},
            },
            plugins,
        )
        is None
    )


def test_compiled_force_plan_falls_back_for_disturbed_ap_and_lower_atmosphere() -> None:
    state, env, context = _case()
    command = np.zeros(3, dtype=float)
    with acceleration_context("auto", allow_env_override=False):
        disturbed_env = dict(env)
        disturbed_env["nrlmsise00_ap_a"] = [4.0, 12.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        propagator = OrbitPropagator(
            integrator="rk4",
            plugins=[spherical_harmonics_plugin, drag_plugin],
            acceleration_mode="auto",
        )
        assert (
            propagator._try_propagate_compiled_builtin_rk4(
                x_eci=state,
                dt_s=10.0,
                t_s=0.0,
                command_accel_eci_km_s2=command,
                env=disturbed_env,
                ctx=context,
            )
            is None
        )

        low_state = state.copy()
        low_state[:3] *= 6550.0 / np.linalg.norm(low_state[:3])
        assert (
            propagator._try_propagate_compiled_builtin_rk4(
                x_eci=low_state,
                dt_s=10.0,
                t_s=0.0,
                command_accel_eci_km_s2=command,
                env=env,
                ctx=context,
            )
            is None
        )


def _custom_acceleration(t_s: float, x_eci: np.ndarray, env: dict, context: OrbitContext) -> np.ndarray:
    del t_s, context
    return np.array([float(env["custom_acceleration_scale"]) * x_eci[0], -2.0e-12, 3.0e-12])


def _assert_staged_propagation_is_exact(
    *,
    integrator: str,
    plugins: list,
    env_updates: dict,
    dt_s: float = 10.0,
) -> None:
    state, env, context = _case()
    env = {**env, **env_updates}
    command = np.array([1.0e-9, -2.0e-9, 3.0e-9], dtype=float)
    kwargs = {
        "integrator": integrator,
        "plugins": plugins,
        "adaptive_atol": 1.0e-12,
        "adaptive_rtol": 1.0e-12,
    }
    with acceleration_context("off", allow_env_override=False):
        reference = OrbitPropagator(acceleration_mode="off", **kwargs)
        expected = reference.propagate(state, dt_s, 12.3, command, env, context)
    with acceleration_context("auto", allow_env_override=False):
        accelerated = OrbitPropagator(acceleration_mode="auto", **kwargs)
        actual = accelerated.propagate(state, dt_s, 12.3, command, env, context)

    assert np.array_equal(actual, expected)
    if integrator == "rkf78":
        assert accelerated.last_adaptive_step_info == reference.last_adaptive_step_info


def test_staged_compiled_path_is_exact_for_remaining_forces_and_custom_plugin() -> None:
    _assert_staged_propagation_is_exact(
        integrator="rk4",
        plugins=[
            j2_plugin,
            j3_plugin,
            j4_plugin,
            drag_plugin,
            lift_plugin,
            srp_plugin,
            third_body_sun_plugin,
            third_body_moon_plugin,
            third_body_planets_plugin,
            _custom_acceleration,
        ],
        env_updates={
            "density_kg_m3": 1.2e-12,
            "drag_coefficient": 0.73,
            "lift_coefficient": 0.3,
            "lift_direction_eci": [0.2, 0.8, -0.1],
            "third_body_planets": ["venus"] * 10 + ["mars"],
            "venus_pos_eci_km": np.array([5.0e7, 1.0e8, 2.0e7]),
            "mars_pos_eci_km": np.array([-7.0e7, 1.5e8, -3.0e7]),
            "custom_acceleration_scale": 1.0e-15,
        },
    )


def test_staged_compiled_path_preserves_rkf78_state_and_step_control_exactly() -> None:
    _assert_staged_propagation_is_exact(
        integrator="rkf78",
        plugins=[j2_plugin, drag_plugin, lift_plugin, third_body_planets_plugin, _custom_acceleration],
        env_updates={
            "density_kg_m3": 1.2e-12,
            "lift_coefficient": 0.3,
            "lift_direction_eci": [0.2, 0.8, -0.1],
            "third_body_planets": ["venus"],
            "venus_pos_eci_km": np.array([5.0e7, 1.0e8, 2.0e7]),
            "custom_acceleration_scale": 1.0e-15,
        },
    )


def test_staged_compiled_drag_is_exact_for_every_atmosphere_family() -> None:
    def density_callable(alt_km, lat_deg, lon_deg, dt_utc, env):
        del alt_km, lat_deg, lon_deg, dt_utc, env
        return 7.5e-12

    common = {
        "atmo_epoch_utc": datetime(2024, 3, 20, 12, tzinfo=timezone.utc),
        "geodetic_model": "wgs84",
        "drag_frame_model": "simple",
        "f107": 150.0,
        "f107a": 150.0,
        "ap": 4.0,
        "jacchia70_f10": 150.0,
        "jacchia70_f10b": 150.0,
        "jacchia70_ap": 4.0,
    }
    cases = [
        ("exponential", {}),
        ("ussa1976", {}),
        ("nrlmsise00", {"nrlmsise00_density_callable": density_callable}),
        ("msis86", {"msis86_density_callable": density_callable}),
        ("jacchia70", {"jacchia70_density_callable": density_callable}),
        ("jb2006", {"jb2006_density_callable": density_callable}),
        ("jb2008", {"jb2008_density_callable": density_callable}),
        ("harris_priester", {}),
    ]
    for model, model_env in cases:
        _assert_staged_propagation_is_exact(
            integrator="rk4",
            plugins=[j2_plugin, j3_plugin, j4_plugin, drag_plugin],
            env_updates={**common, **model_env, "atmosphere_model": model},
        )


def test_staged_compiled_density_callable_receives_original_environment() -> None:
    def density_callable(alt_km, lat_deg, lon_deg, dt_utc, env):
        del alt_km, lat_deg, lon_deg, dt_utc
        return 2.0e-12 if "jd_utc" in env else 1.0e-12

    _assert_staged_propagation_is_exact(
        integrator="rk4",
        plugins=[j2_plugin, j3_plugin, drag_plugin, srp_plugin, _custom_acceleration],
        env_updates={
            "nrlmsise00_density_callable": density_callable,
            "custom_acceleration_scale": 1.0e-15,
        },
        dt_s=10.0,
    )
