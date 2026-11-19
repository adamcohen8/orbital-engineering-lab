from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.optimize import brentq

from sim.api import SimulationConfig, SimulationSession
from sim.dynamics.docking import MatedAssembly
from sim.utils.quaternion import quaternion_to_dcm_bn


def _scenario(case, output_dir):
    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((root / "examples" / "configs" / f"public_docking_{case}.yaml").read_text())
    raw["outputs"]["output_dir"] = str(output_dir)
    return raw


def _run(raw):
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    result = session.run()
    evidence = json.loads((Path(raw["outputs"]["output_dir"]) / "docking.json").read_text())
    return result, evidence


@pytest.mark.parametrize(("case", "phase", "events"), [
    ("capture", "captured", ["first_contact", "soft_capture"]),
    ("rebound", "separated", ["first_contact", "rebound"]),
    ("mate", "mated", ["first_contact", "soft_capture", "hard_mate"]),
])
def test_docking_scenarios_publish_truth_and_review(tmp_path, case, phase, events):
    raw = _scenario(case, tmp_path)
    result, evidence = _run(raw)
    assert evidence["schema_version"] == 1
    assert result.summary["docking"]["phase"] == phase
    assert [row["event"] for row in evidence["events"]] == events
    assert evidence["summary"]["peak_sampled_force_n"] > 10
    assert not (tmp_path / "master_run_log.json").exists()
    with sqlite3.connect(tmp_path / "review" / "run.sqlite") as conn:
        rows = conn.execute("SELECT event_type FROM events WHERE source = 'docking_dynamics' ORDER BY time_s, event_id")
        assert [value for value, in rows] == [f"docking_{event}" for event in events]
        count, = conn.execute("SELECT COUNT(*) FROM docking_samples").fetchone()
        assert count == len(evidence["samples"])
    assert max(row["time_s"] for row in evidence["samples"]) == pytest.approx(5)
    if case == "rebound":
        assert evidence["summary"]["peak_sampled_torque_n_m"] > 1
        assert np.linalg.norm(result.truth["chaser"][-1, 10:13]) > 0.01
        first = evidence["events"][0]
        assert sum(load > 0 for load in first["site_loads_n"]) < 4
    if case == "mate":
        assert evidence["summary"]["mated_mass_kg"] == 200
        joined = [row for row in evidence["samples"] if row["phase"] == "mated"]
        assert joined[-1]["time_s"] - joined[0]["time_s"] > 2
        offsets = np.array([row["offset_receiver_m"] for row in joined])
        assert np.max(np.abs(offsets - offsets[0])) < 1e-8


@pytest.mark.parametrize("point_damping_n_s_m", [0, 25])
def test_centered_contact_matches_reduced_mass_oscillator(tmp_path, point_damping_n_s_m):
    raw = _scenario("capture", tmp_path)
    config = raw["simulator"]["docking"]
    del config["latch"]
    config["contact_damping_n_s_m"] = point_damping_n_s_m
    config["max_step_s"] = .0025
    result, evidence = _run(raw)

    stiffness = 4 * config["contact_stiffness_n_m"]
    damping = 4 * point_damping_n_s_m
    reduced_mass = 50
    decay = damping / (2 * reduced_mass)
    omega = np.sqrt(stiffness / reduced_mass - decay**2)
    first, separation = evidence["events"]
    speed = -first["normal_speed_m_s"]

    def oscillator(t):
        compression = speed * np.exp(-decay * t) * np.sin(omega * t) / omega
        compression_rate = speed * np.exp(-decay * t) * (
            np.cos(omega * t) - decay * np.sin(omega * t) / omega
        )
        return compression, compression_rate

    def normal_force(t):
        compression, rate = oscillator(t)
        return stiffness * compression + damping * rate

    period_end = np.pi / omega
    # The unilateral force cuts out when the bilateral solution reaches zero.
    unload = brentq(normal_force, period_end / 2, period_end) if damping else period_end
    compression, rate = oscillator(unload)
    exit_time = unload - compression / rate
    peak = max(normal_force(t) for t in np.linspace(0, unload, 1000))

    assert separation["event"] == "rebound"
    assert separation["time_s"] - first["time_s"] == pytest.approx(exit_time, abs=.0025)
    assert evidence["summary"]["peak_sampled_force_n"] == pytest.approx(peak, rel=3e-4)
    assert separation["normal_speed_m_s"] == pytest.approx(-rate, rel=3e-4)
    assert result.summary["docking"]["peak_sampled_torque_n_m"] < 1e-7


def test_impact_and_hard_mate_converge_when_step_is_halved(tmp_path):
    for case, last_event_tolerance_s in (("rebound", 0.006), ("mate", 0.0075)):
        raw = _scenario(case, tmp_path / case / "coarse")
        _, coarse = _run(raw)
        finer = deepcopy(raw)
        finer["outputs"]["output_dir"] = str(tmp_path / case / "fine")
        finer["simulator"]["docking"]["max_step_s"] *= 0.5
        _, fine = _run(finer)
        assert [row["event"] for row in coarse["events"]] == [row["event"] for row in fine["events"]]
        assert coarse["events"][0]["time_s"] == pytest.approx(fine["events"][0]["time_s"], abs=2e-7)
        assert coarse["events"][-1]["time_s"] == pytest.approx(fine["events"][-1]["time_s"], abs=last_event_tolerance_s)
        for field in ("peak_sampled_force_n", "peak_sampled_torque_n_m"):
            assert coarse["summary"][field] == pytest.approx(fine["summary"][field], rel=.015, abs=1e-8)


def _momentum(pair, masses, inertias, center):
    linear = np.sum(masses[:, None] * pair[:, 3:6] * 1000, axis=0)
    angular = sum(
        quaternion_to_dcm_bn(row[6:10]).T @ inertia @ row[10:13]
        + np.cross((row[:3] - center) * 1000, mass * row[3:6] * 1000)
        for row, mass, inertia in zip(pair, masses, inertias)
    )
    return linear, angular


def test_off_axis_internal_wrench_and_lock_conserve_momentum(tmp_path):
    raw = _scenario("rebound", tmp_path)
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.reset()
    stepper = session._engine.docking_stepper
    stepper._gravity = lambda _time, _pair: np.zeros((2, 3))
    pair = np.array([
        [0., .00003, .001016, -.00001, 0., -.0001, .04997916927067833, .9987502603949663, 0., 0., .01, 0., -.02],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., -.015, .02],
    ])
    stepper.phase = "contact"
    masses, inertias = stepper.masses, stepper.inertias
    center = np.average(pair[:, :3], axis=0, weights=masses)
    original = _momentum(pair, masses, inertias, center)
    initial_body_rates = pair[:, 10:13].copy()
    for index in range(200):
        pair = stepper._advance_pair(pair, index * .001, .001, contact=True)
    assert np.linalg.norm(pair[:, 10:13] - initial_body_rates) > 0.005
    np.testing.assert_allclose(_momentum(pair, masses, inertias, center), original, atol=1e-10, rtol=0)
    assembly, joined = MatedAssembly.lock(pair, masses, inertias)
    projected = assembly.pair_state(joined)
    np.testing.assert_allclose(_momentum(projected, masses, inertias, center), original, atol=1e-10, rtol=0)
    np.testing.assert_allclose(projected[:, :3], pair[:, :3], atol=1e-15, rtol=0)
    for index in range(100):
        joined = assembly.step(joined, index * .01, .01, stepper._gravity)
    after = assembly.pair_state(joined)
    np.testing.assert_allclose(_momentum(after, masses, inertias, center), original, atol=1e-9, rtol=0)
    assert np.linalg.norm(after[0, :3] - after[1, :3]) == pytest.approx(
        np.linalg.norm(pair[0, :3] - pair[1, :3]), abs=1e-14,
    )


def test_latch_releases_when_load_exceeds_limit(tmp_path):
    raw = _scenario("capture", tmp_path)
    raw["simulator"]["docking"]["latch"]["max_force_n"] = 5
    _, evidence = _run(raw)
    assert [row["event"] for row in evidence["events"]] == [
        "first_contact", "soft_capture", "latch_release", "rebound",
    ]


@pytest.mark.parametrize(("mutate", "error"), [
    (lambda d: d["simulator"]["docking"].update({"max_step_s": 0}), "max_step_s"),
    (lambda d: d["simulator"]["docking"]["receiver"].update({"object_id": "chaser"}), "distinct"),
    (lambda d: d["simulator"]["dynamics"]["orbit"].update({"j2": True}), "central gravity"),
    (lambda d: d["simulator"]["dynamics"]["attitude"].update({"enabled": False}), "attitude-enabled"),
    (lambda d: d["simulator"]["docking"].update({"typo": 1}), "typo"),
    (lambda d: d["objects"]["chaser"]["initial_state"].update({"position_eci_km": [7000, 0, .00099]}), "initially separated"),
])
def test_docking_config_rejects_unsupported_setup(tmp_path, mutate, error):
    raw = _scenario("capture", tmp_path)
    mutate(raw)
    with pytest.raises(ValueError, match=error):
        SimulationConfig.from_dict(raw)
