#!/usr/bin/env python3
"""Generate browser-preview contracts from canonical OEL game configs and physics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREVIEW_ROOT = ROOT / "web" / "rpo-trainer-preview"
CONFIG_ROOT = ROOT / "sim" / "game" / "configs"
CONTRACT_PATH = PREVIEW_ROOT / "fixtures" / "downloadable-game-contract.json"
TRAJECTORY_PATH = PREVIEW_ROOT / "fixtures" / "oel-level0-reference-trajectories.json"
TUTORIAL_CONFIG_PATH = CONFIG_ROOT / "game_training_rpo_00_tutorial.yaml"
SANDBOX_CONFIG_PATH = CONFIG_ROOT / "game_training_rpo_sandbox.yaml"
ARCADE_CONFIG_PATH = CONFIG_ROOT / "game_training_rpo_arcade_pursuit.yaml"
REFERENCE_TIMES_S = (0, 60, 300, 600)
REFERENCE_CASES = {
    "plus_in_track": [0.0, -0.8, 0.0, 0.0, 0.00025, 0.0],
    "plus_radial": [0.0, -0.8, 0.0, 0.00025, 0.0, 0.0],
    "plus_cross_track": [0.0, -0.8, 0.0, 0.0, 0.0, 0.00025],
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _ric_mapping(values: list[float]) -> dict[str, float]:
    keys = ("r_km", "i_km", "c_km", "rd_km_s", "id_km_s", "cd_km_s")
    if len(values) != len(keys):
        raise ValueError("A rectangular RIC state must contain six values.")
    return {key: float(value) for key, value in zip(keys, values)}


def build_downloadable_contract() -> dict[str, Any]:
    from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2

    tutorial = _load_yaml(TUTORIAL_CONFIG_PATH)
    sandbox = _load_yaml(SANDBOX_CONFIG_PATH)
    arcade = _load_yaml(ARCADE_CONFIG_PATH)

    tutorial_game = tutorial["metadata"]["game"]
    tutorial_training = tutorial_game["training"]
    guided_burn_dv = sorted({float(item["delta_v_m_s"]) for item in tutorial_training["guided_tutorial_burns"]})
    if len(guided_burn_dv) != 1:
        raise ValueError("Level 0 guided burns must share one delta-v for the browser contract.")

    sandbox_training = sandbox["metadata"]["game"]["training"]
    arcade_game = arcade["metadata"]["game"]
    arcade_training = arcade_game["training"]
    defense = arcade_game["defensive_target"]
    arcade_rules = arcade_game["arcade"]
    target_coes = arcade["objects"]["target"]["initial_state"]["coes"]
    chaser_ric = arcade["objects"]["chaser"]["initial_state"]["relative_to_target_ric"]["state"]

    return {
        "schema_version": 1,
        "generated_from": [
            str(TUTORIAL_CONFIG_PATH.relative_to(ROOT)),
            str(SANDBOX_CONFIG_PATH.relative_to(ROOT)),
            str(ARCADE_CONFIG_PATH.relative_to(ROOT)),
        ],
        "tutorial": {
            "title": tutorial_game["level_name"],
            "max_time_s": float(tutorial_training["max_time_s"]),
            "max_delta_v_m_s": float(tutorial_training["max_delta_v_m_s"]),
            "goal_range_km": float(tutorial_training["goal_range_km"]),
            "max_goal_speed_km_s": float(tutorial_training["max_goal_speed_km_s"]),
            "guided_burn_delta_v_m_s": guided_burn_dv[0],
            "guided_speed_multiplier": float(
                tutorial_training["guided_tutorial_speed_step"]["target_speed_multiplier"]
            ),
            "learning_goal": tutorial_training["learning_goal"],
            "player_brief": tutorial_training["player_brief"],
            "pass_criteria": tutorial_training["pass_criteria"],
            "instructor_notes": tutorial_training["instructor_notes"],
        },
        "sandbox": {
            "max_time_s": float(sandbox_training["max_time_s"]),
            "player_brief": sandbox_training["player_brief"],
            "instructor_notes": sandbox_training["instructor_notes"],
            "supports_target_orbit_edit": "target orbit" in sandbox_training["player_brief"].lower(),
            "supports_target_eccentricity": any(
                "eccentricity" in note.lower() for note in sandbox_training["instructor_notes"]
            ),
        },
        "arcade": {
            "challenge_id": arcade_training["scenario_id"],
            "title": arcade_game["level_name"],
            "mu_km3_s2": float(EARTH_MU_KM3_S2),
            "dt_s": float(arcade["simulator"]["dt_s"]),
            "max_time_s": float(arcade_training["max_time_s"]),
            "max_player_accel_km_s2": float(arcade_game["player_max_accel_km_s2"]),
            "max_delta_v_m_s": float(arcade_training["max_delta_v_m_s"]),
            "max_target_delta_v_m_s": float(defense["max_delta_v_m_s"]),
            "goal_range_km": float(arcade_training["goal_range_km"]),
            "difficulty": arcade_game["difficulty"],
            "target_coes": {key: float(value) for key, value in target_coes.items()},
            "chaser_initial_ric": _ric_mapping(chaser_ric),
            "target_defense": {
                key: defense[key]
                for key in (
                    "enabled",
                    "trigger_range_km",
                    "trigger_closing_speed_km_s",
                    "keepout_radius_km",
                    "max_accel_km_s2",
                    "max_delta_v_m_s",
                    "delta_v_ramp_after_round",
                    "delta_v_ramp_step_m_s",
                    "pulse_period_s",
                    "cross_track_bias",
                )
            },
            "arcade": arcade_rules,
        },
    }


def _rounded_state(values: Any) -> list[float]:
    # Frame transforms can produce either sign of zero across platforms.
    # Keep the JSON fixture canonical after rounding without changing precision.
    rounded = [round(float(value), 12) for value in values]
    return [0.0 if value == 0.0 else value for value in rounded]


def build_reference_trajectories() -> dict[str, Any]:
    from sim.api import SimulationConfig, SimulationSession
    from sim.game.training_geometry import relative_state_from_arrays

    base = SimulationConfig.from_yaml(TUTORIAL_CONFIG_PATH)
    cases: list[dict[str, Any]] = []
    for name, initial_state in REFERENCE_CASES.items():
        config = base.with_value(
            "chaser.initial_state.relative_to_target_ric.state", initial_state
        ).with_value("simulator.duration_s", float(REFERENCE_TIMES_S[-1]))
        session = SimulationSession.from_config(config)
        snapshot = session.reset()
        samples = [
            {
                "time_s": 0,
                "relative_ric_km_km_s": _rounded_state(
                    relative_state_from_arrays(snapshot.truth["target"], snapshot.truth["chaser"])
                ),
            }
        ]
        for step_index in range(1, REFERENCE_TIMES_S[-1] + 1):
            snapshot = session.step(dt_s=1.0)
            if step_index in REFERENCE_TIMES_S:
                samples.append(
                    {
                        "time_s": step_index,
                        "relative_ric_km_km_s": _rounded_state(
                            relative_state_from_arrays(snapshot.truth["target"], snapshot.truth["chaser"])
                        ),
                    }
                )
        cases.append(
            {
                "name": name,
                "initial_relative_ric_km_km_s": initial_state,
                "samples": samples,
            }
        )
    return {
        "schema_version": 1,
        "generated_from": str(TUTORIAL_CONFIG_PATH.relative_to(ROOT)),
        "oel_model": "two_body",
        "oel_step_s": 1.0,
        "browser_step_s": 0.1,
        "position_tolerance_km": 5.0e-5,
        "cases": cases,
    }


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_or_check(path: Path, payload: dict[str, Any], *, check: bool) -> bool:
    expected = _json_text(payload)
    if check:
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual != expected:
            print(f"stale fixture: {path.relative_to(ROOT)}")
            return False
        print(f"current fixture: {path.relative_to(ROOT)}")
        return True
    path.write_text(expected, encoding="utf-8")
    print(f"wrote fixture: {path.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if checked-in fixtures are stale.")
    args = parser.parse_args()
    results = (
        _write_or_check(CONTRACT_PATH, build_downloadable_contract(), check=args.check),
        _write_or_check(TRAJECTORY_PATH, build_reference_trajectories(), check=args.check),
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
