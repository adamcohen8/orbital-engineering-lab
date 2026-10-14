from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue


@dataclass(frozen=True)
class SimulationSnapshot:
    step_index: int
    time_s: float
    truth: dict[str, np.ndarray]
    belief: dict[str, np.ndarray]
    applied_thrust: dict[str, np.ndarray]
    applied_torque: dict[str, np.ndarray]
    spacecraft_resources: dict[str, dict[str, float]] = field(default_factory=dict)

    ground_segment: dict = field(default_factory=dict)

    state_metadata: dict = field(default_factory=dict)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.truth.keys()))
