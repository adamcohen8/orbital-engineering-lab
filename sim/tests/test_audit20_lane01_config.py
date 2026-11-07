from __future__ import annotations

import math

import pytest

from sim.config.scenario.loader import scenario_config_from_dict
from sim.scenarios import ScenarioBuilder


def _base_config() -> dict:
    return {
        "objects": {
            "target": {
                "enabled": True,
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
        },
        "simulator": {"duration_s": 1.0, "dt_s": 1.0},
    }


@pytest.mark.parametrize("form", ["relative_to_target_ric", "relative_to_target_cislunar"])
@pytest.mark.parametrize("bad_state", [None, [0.0] * 5, [math.nan] * 6])
def test_nested_relative_state_is_checked_during_config_parse(form: str, bad_state) -> None:
    config = _base_config()
    config["objects"]["chaser"] = {
        "enabled": True,
        "initial_state": {
            "relative_to": "target",
            form: {} if bad_state is None else {"state": bad_state},
        },
    }

    with pytest.raises(ValueError, match=r"relative_to_target_(ric|cislunar).*state.*(length-6|finite)"):
        scenario_config_from_dict(config)


@pytest.mark.parametrize("method, args", [("duration", (math.nan,)), ("satellite", ())])
def test_scenario_builder_rejects_nonfinite_numeric_inputs(method: str, args: tuple) -> None:
    builder = ScenarioBuilder("finite_builder")
    if method == "duration":
        with pytest.raises(ValueError, match="finite"):
            builder.duration(*args, dt_s=1.0)
    else:
        with pytest.raises(ValueError, match="finite"):
            builder.satellite(
                "target",
                position_eci_km=[math.nan, 0.0, 0.0],
                velocity_eci_km_s=[0.0, 7.5, 0.0],
            )
