from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from sim.api import SimulationSession
from sim.performance.suite import (
    _effective_scenario_config,
    _merged_case_profile,
    load_performance_manifest,
)


def test_adaptive_trajectory_fixture_stays_within_approved_rounding_bound(tmp_path: Path) -> None:
    manifest = load_performance_manifest()
    case = next(item for item in manifest.cases if item.name == "adaptive_high_fidelity")
    profile = _merged_case_profile(manifest, case, "full", warmups=0, repeats=1)
    trajectory_cfg, _ = _effective_scenario_config(
        case,
        profile,
        output_dir=tmp_path / "trajectory",
    )
    passive_case = replace(
        case,
        base_overrides={
            **case.base_overrides,
            "objects.satellite.runtime_profile": "flight_software",
            "objects.satellite.flight_software": {
                "stack": "fsw.passive",
                "hardware_profile": "hardware.passive.v1",
                # Match the scenario sample cadence so the scheduler wrapper
                # does not split each 2 s adaptive propagation interval into
                # two separately rounded 1 s integrations.
                "task_period_s": 2.0,
                "params": {},
            },
        },
    )

    passive_cfg, _ = _effective_scenario_config(
        passive_case,
        profile,
        output_dir=tmp_path / "passive",
    )
    trajectory = SimulationSession.from_config(trajectory_cfg).run().truth["satellite"]
    passive = SimulationSession.from_config(passive_cfg).run().truth["satellite"]

    # With matching sample/task cadence, the passive scheduler changes no
    # physical integration boundaries; retain the strict parity contract.
    np.testing.assert_allclose(trajectory[:, :3], passive[:, :3], rtol=0.0, atol=6.0e-12)
    np.testing.assert_allclose(trajectory[:, 3:6], passive[:, 3:6], rtol=0.0, atol=8.0e-15)
    np.testing.assert_array_equal(trajectory[:, 6:], passive[:, 6:])
