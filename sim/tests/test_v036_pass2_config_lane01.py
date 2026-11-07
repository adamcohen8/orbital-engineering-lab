"""Focused v0.36 config and scenario-builder regressions for review pass 2."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sim.config.scenario.validation import _validate_physics_runtime_settings
from sim.scenarios import ScenarioBuilder


def _agent(
    *,
    kind: str = "satellite",
    propagation_method: str = "special",
    general: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        kind=kind,
        propagation_method=propagation_method,
        general={} if general is None else general,
        initial_state={},
        runtime_profile="trajectory_only" if kind == "satellite" else "flight_software",
        knowledge={},
        specs={},
        force_models=[],
        reference_orbit={},
        bridge=None,
    )


def _runtime_config(*, objects: dict[str, SimpleNamespace], execution: dict | None = None, body: str = "earth"):
    return SimpleNamespace(
        simulator=SimpleNamespace(
            central_body=body,
            additional_gravity=None,
            execution={} if execution is None else execution,
            dynamics={
                "orbit": {
                    "model": "two_body",
                    "propagation_method": "special",
                    "integrator": "rk4",
                    "numeric_backend": "python",
                },
                "attitude": {"enabled": False, "numeric_backend": "python"},
                "rocket": {"numeric_backend": "python"},
                "reentry": {"numeric_backend": "python"},
            },
            collisions={"enabled": False, "numeric_backend": "python"},
            system_force_models=[],
            initial_jd_utc=None,
            environment={},
            frames={},
        ),
        analysis=SimpleNamespace(
            enabled=False,
            covariance=SimpleNamespace(enabled=False, numeric_backend="python"),
        ),
        objects=objects,
    )


def _fake_ogp_extension() -> SimpleNamespace:
    return SimpleNamespace(
        OGPContext=object,
        ogp_propagate_batch_parallel_bytes=lambda *_args, **_kwargs: None,
    )


@pytest.mark.parametrize(
    "general",
    [{}, {"numeric_backend": None}, {"numeric_backend": " Rust "}],
    ids=["omitted", "none", "mixed-case"],
)
def test_ogp_parallelism_uses_canonical_rust_backend_default(general, monkeypatch):
    monkeypatch.setattr("sim.rust_ogp_backend._extension", _fake_ogp_extension)
    cfg = _runtime_config(
        objects={"catalog": _agent(propagation_method="general", general=general)},
        execution={"ogp_parallelism": {"enabled": True, "workers": 1}},
    )

    _validate_physics_runtime_settings(cfg)


def test_ogp_parallelism_rejects_python_backend_before_native_probe(monkeypatch):
    monkeypatch.setattr("sim.rust_ogp_backend._extension", _fake_ogp_extension)
    cfg = _runtime_config(
        objects={
            "catalog": _agent(
                propagation_method="general",
                general={"numeric_backend": "python"},
            )
        },
        execution={"ogp_parallelism": {"enabled": True, "workers": 1}},
    )

    with pytest.raises(ValueError, match="ogp_parallelism"):
        _validate_physics_runtime_settings(cfg)


@pytest.mark.parametrize(
    ("dynamics", "objects", "expected"),
    [
        (
            {"orbit": {"numeric_backend": False}},
            {},
            "simulator.dynamics.orbit.numeric_backend must be python or rust",
        ),
        (
            {},
            {
                "sat": SimpleNamespace(
                    initial_state={},
                    general={"numeric_backend": False},
                    knowledge={"numeric_backend": "python"},
                )
            },
            "objects.sat.general.numeric_backend must be python or rust",
        ),
    ],
)
def test_authored_selector_errors_precede_optional_runtime_attributes(dynamics, objects, expected):
    safe_dynamics = {
        "attitude": {"numeric_backend": "python"},
        "rocket": {"numeric_backend": "python"},
        "reentry": {"numeric_backend": "python"},
    }
    safe_dynamics.update(dynamics)
    cfg = SimpleNamespace(
        simulator=SimpleNamespace(
            central_body="earth",
            additional_gravity=None,
            execution={},
            dynamics=safe_dynamics,
            collisions={"numeric_backend": "python"},
        ),
        analysis=SimpleNamespace(covariance=SimpleNamespace(numeric_backend="python")),
        objects=objects,
    )

    with pytest.raises(ValueError, match=expected) as error:
        _validate_physics_runtime_settings(cfg)
    assert str(error.value) == expected + "."


def test_planetary_ogp_parallelism_is_rejected_explicitly():
    cfg = _runtime_config(
        objects={"probe": _agent()},
        execution={"ogp_parallelism": {"enabled": True, "workers": 1}},
        body="mars",
    )

    with pytest.raises(ValueError, match=r"ogp_parallelism.*central_body=earth"):
        _validate_physics_runtime_settings(cfg)


@pytest.mark.parametrize(
    ("kind", "runtime_profile"),
    [("satellite", "trajectory_only"), ("rocket", "flight_software")],
)
def test_non_ogp_objects_do_not_probe_ogp_extension(kind, runtime_profile, monkeypatch):
    def unexpected_probe():
        raise AssertionError("non-OGP object triggered the OGP native probe")

    monkeypatch.setattr("sim.rust_ogp_backend._extension", unexpected_probe)
    obj = _agent(
        kind=kind,
        propagation_method="special",
        general={"numeric_backend": "rust"},
    )
    obj.runtime_profile = runtime_profile
    cfg = _runtime_config(objects={kind: obj})

    _validate_physics_runtime_settings(cfg)


def test_builder_to_dict_returns_a_deep_snapshot():
    builder = (
        ScenarioBuilder("deep_snapshot")
        .duration(60.0, dt_s=5.0, orbit_substep_s=1.0)
        .satellite(
            "target",
            mass_kg=100.0,
            position_eci_km=[7000.0, 0.0, 0.0],
            velocity_eci_km_s=[0.0, 7.5, 0.0],
            specs={"thermal": {"enabled": False}},
        )
        .ground_station("denver", lat_deg=39.7, lon_deg=-104.9)
    )
    snapshot = builder.to_dict()
    snapshot["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] = 99.0
    snapshot["objects"]["target"]["initial_state"]["position_eci_km"][0] = 1.0
    snapshot["objects"]["target"]["specs"]["thermal"]["enabled"] = True
    snapshot["ground_stations"][0]["lat_deg"] = 1.0

    current = builder.to_dict()
    assert current["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] == 1.0
    assert current["objects"]["target"]["initial_state"]["position_eci_km"][0] == 7000.0
    assert current["objects"]["target"]["specs"]["thermal"]["enabled"] is False
    assert current["ground_stations"][0]["lat_deg"] == pytest.approx(39.7)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda builder: builder.duration(True, dt_s=1.0), id="duration"),
        pytest.param(lambda builder: builder.duration(1.0, dt_s=True), id="dt"),
        pytest.param(
            lambda builder: builder.satellite(
                "target",
                mass_kg=True,
                position_eci_km=[7000.0, 0.0, 0.0],
                velocity_eci_km_s=[0.0, 7.5, 0.0],
            ),
            id="mass",
        ),
        pytest.param(
            lambda builder: builder.satellite(
                "target",
                position_eci_km=[True, 0.0, 0.0],
                velocity_eci_km_s=[0.0, 7.5, 0.0],
            ),
            id="position",
        ),
        pytest.param(
            lambda builder: builder.ground_station("station", lat_deg=True, lon_deg=0.0),
            id="ground-station",
        ),
    ],
)
def test_builder_rejects_booleans_in_physical_numeric_parameters(build):
    with pytest.raises(ValueError, match="must be numeric, not a boolean"):
        build(ScenarioBuilder("strict_numeric"))
