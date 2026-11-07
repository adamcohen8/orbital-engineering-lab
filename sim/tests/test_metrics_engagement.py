from __future__ import annotations

import unittest

import numpy as np

from sim.core.models import SimLog
from sim.metrics.engagement import compute_engagement_metrics


class TestEngagementMetrics(unittest.TestCase):
    def test_min_separation_ignores_invalid_samples_but_keeps_finite_tail(self):
        t_s = np.array([0.0, 1.0, 2.0], dtype=float)
        target_truth = np.zeros((3, 14), dtype=float)
        target_truth[:, 13] = 100.0
        chaser_truth = target_truth.copy()
        chaser_truth[:, 0] = [np.nan, 2.0, 8.0]
        zeros = np.zeros((3, 3), dtype=float)
        runtime = np.zeros(3, dtype=float)
        skipped = np.zeros(3, dtype=bool)
        log = SimLog(
            t_s=t_s,
            truth_by_object={"target": target_truth, "chaser": chaser_truth},
            belief_by_object={"target": target_truth[:, :6], "chaser": chaser_truth[:, :6]},
            applied_thrust_by_object={"target": zeros, "chaser": zeros},
            applied_torque_by_object={"target": zeros, "chaser": zeros},
            controller_runtime_ms_by_object={"target": runtime, "chaser": runtime},
            controller_skipped_by_object={"target": skipped, "chaser": skipped},
        )

        metrics = compute_engagement_metrics(log)

        self.assertAlmostEqual(metrics.min_separation_km, 2.0, places=9)

    def test_keepout_time_uses_pairwise_separation(self):
        t_s = np.array([0.0, 1.0, 2.0], dtype=float)
        target_truth = np.array(
            [
                [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            ],
            dtype=float,
        )
        chaser_truth = np.array(
            [
                [7005.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                [7002.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 89.0],
                [7008.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 88.0],
            ],
            dtype=float,
        )
        zeros = np.zeros((3, 3), dtype=float)
        runtime = np.zeros(3, dtype=float)
        skipped = np.zeros(3, dtype=bool)
        log = SimLog(
            t_s=t_s,
            truth_by_object={"target": target_truth, "chaser": chaser_truth},
            belief_by_object={"target": target_truth[:, :6], "chaser": chaser_truth[:, :6]},
            applied_thrust_by_object={"target": zeros, "chaser": zeros},
            applied_torque_by_object={"target": zeros, "chaser": zeros},
            controller_runtime_ms_by_object={"target": runtime, "chaser": runtime},
            controller_skipped_by_object={"target": skipped, "chaser": skipped},
        )

        metrics = compute_engagement_metrics(log, keepout_radius_km=3.0)

        self.assertAlmostEqual(metrics.min_separation_km, 2.0, places=9)
        self.assertAlmostEqual(metrics.time_inside_keepout_s, 1.0, places=9)
        self.assertAlmostEqual(metrics.fuel_used_kg_by_object["chaser"], 2.0, places=9)

    def test_keepout_time_counts_intervals_not_samples(self):
        t_s = np.array([0.0, 1.0, 2.0], dtype=float)
        target_truth = np.array(
            [
                [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            ],
            dtype=float,
        )
        chaser_truth = target_truth.copy()
        chaser_truth[:, 0] += 1.0
        zeros = np.zeros((3, 3), dtype=float)
        runtime = np.zeros(3, dtype=float)
        skipped = np.zeros(3, dtype=bool)
        log = SimLog(
            t_s=t_s,
            truth_by_object={"target": target_truth, "chaser": chaser_truth},
            belief_by_object={"target": target_truth[:, :6], "chaser": chaser_truth[:, :6]},
            applied_thrust_by_object={"target": zeros, "chaser": zeros},
            applied_torque_by_object={"target": zeros, "chaser": zeros},
            controller_runtime_ms_by_object={"target": runtime, "chaser": runtime},
            controller_skipped_by_object={"target": skipped, "chaser": skipped},
        )

        metrics = compute_engagement_metrics(log, keepout_radius_km=3.0)

        self.assertAlmostEqual(metrics.time_inside_keepout_s, 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
