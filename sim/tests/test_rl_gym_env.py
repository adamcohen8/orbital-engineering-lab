from __future__ import annotations

import unittest

import numpy as np

from machine_learning import (
    ActionField,
    AsyncVectorSimulationEnv,
    DirectActionAdapter,
    GymEnvConfig,
    GymSimulationEnv,
    LinearPolicy,
    MultiAgentEnvConfig,
    MultiAgentSimulationEnv,
    ObservationField,
    RangeTermination,
    RelativeDistanceReward,
    SelfPlayTrainerConfig,
    SyncVectorSimulationEnv,
    ThrustVectorToPointingAdapter,
    VectorEnvConfig,
    build_policy_card,
    classify_observation_fields,
    collect_multi_agent_rollout,
    collect_vector_rollout,
    make_sb3_vec_env,
    make_vector_env,
    run_self_play_training,
)
from sim.config import MonteCarloVariation


def _base_scenario() -> dict:
    return {
        "scenario_name": "rl_gym_env_test",
        "objects": {
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
            "chaser": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "relative_to_target_ric": {"frame": "rect", "state": [1.0, -2.0, 0.0, 0.0, 0.0, 0.0]},
                    "attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
                },
                "mission_execution": {
                    "module": "sim.mission.modules",
                    "class_name": "ControllerPointingExecution",
                    "params": {"alignment_tolerance_deg": 180.0},
                },
                "attitude_control": {
                    "module": "sim.control.attitude.baseline",
                    "class_name": "ReactionWheelPDController",
                    "params": {
                        "kp": [0.1, 0.1, 0.1],
                        "kd": [0.2, 0.2, 0.2],
                    },
                },
            },
        },
        "simulator": {
            "duration_s": 5.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": True}},
        },
        "outputs": {"output_dir": "outputs/test_rl_gym_env", "mode": "save"},
        "metadata": {"seed": 3},
    }


class TestGymSimulationEnv(unittest.TestCase):
    def test_default_enabled_objects_are_available_to_observation_probe(self):
        scenario = _base_scenario()
        scenario["objects"]["target"].pop("enabled")
        scenario["objects"]["chaser"].pop("enabled")
        env = GymSimulationEnv(
            GymEnvConfig(
                scenario=scenario,
                controlled_agent_id="chaser",
                action_fields=(),
            )
        )

        observation, _ = env.reset(seed=17)

        self.assertEqual(observation.shape, (6,))

    def test_default_observations_use_belief_not_truth(self):
        env = GymSimulationEnv(
            GymEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_id="chaser",
                action_fields=(),
            )
        )

        obs, _ = env.reset(seed=11)

        self.assertEqual(obs.shape, (6,))
        self.assertTrue(all(field.path.startswith("belief.chaser.state") for field in env.observation_fields))

    def test_reset_and_step_use_gymnasium_signature(self):
        env = GymSimulationEnv(
            GymEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_id="chaser",
                observation_fields=(
                    ObservationField("truth.chaser.position_eci_km"),
                    ObservationField("truth.target.position_eci_km"),
                ),
                action_fields=(
                    ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),
                    ActionField("thrust_eci_km_s2[1]", -1e-5, 1e-5),
                    ActionField("thrust_eci_km_s2[2]", -1e-5, 1e-5),
                ),
                action_adapter=DirectActionAdapter(),
                reward_fn=RelativeDistanceReward(),
                termination_fn=RangeTermination(capture_radius_km=1e-6),
            )
        )
        obs, info = env.reset(seed=11)
        self.assertEqual(obs.shape, (6,))
        self.assertIn("metrics", info)

        obs2, reward, terminated, truncated, info2 = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(obs2.shape, (6,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("metrics", info2)

    def test_nonzero_action_changes_propagated_velocity(self):
        env_cfg = GymEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_id="chaser",
            observation_fields=(
                ObservationField("truth.chaser.position_eci_km"),
                ObservationField("truth.chaser.velocity_eci_km_s"),
            ),
            action_fields=(
                ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[1]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[2]", -1e-5, 1e-5),
            ),
        )
        coast_env = GymSimulationEnv(env_cfg)
        thrust_env = GymSimulationEnv(env_cfg)
        coast_env.reset(seed=42)
        thrust_env.reset(seed=42)

        coast_obs, *_ = coast_env.step(np.zeros(3, dtype=np.float32))
        thrust_obs, *_ = thrust_env.step(np.array([1e-5, 0.0, 0.0], dtype=np.float32))

        velocity_delta_km_s = thrust_obs[3:] - coast_obs[3:]
        self.assertGreater(float(velocity_delta_km_s[0]), 5e-6)
        self.assertTrue(np.allclose(velocity_delta_km_s[1:], 0.0, atol=1e-9, rtol=0.0))

    def test_episode_variations_apply_to_scenario(self):
        env = GymSimulationEnv(
            GymEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_id="chaser",
                observation_fields=(ObservationField("truth.chaser.position_eci_km"),),
                action_fields=(),
                episode_variations=(
                    MonteCarloVariation(
                        parameter_path="objects.chaser.initial_state.relative_to_target_ric.state[0]",
                        mode="choice",
                        options=[0.5],
                    ),
                ),
            )
        )
        _, info = env.reset(seed=4)
        sampled = dict(info["sampled_parameters"])
        self.assertEqual(sampled["objects.chaser.initial_state.relative_to_target_ric.state[0]"], 0.5)

    def test_knowledge_observation_fields_use_schema_defaults_before_track_exists(self):
        scenario = _base_scenario()
        scenario["objects"]["chaser"]["knowledge"] = {"targets": ["target"]}
        env = GymSimulationEnv(
            GymEnvConfig(
                scenario=scenario,
                controlled_agent_id="chaser",
                observation_fields=(ObservationField("knowledge.chaser.target.state"),),
                action_fields=(),
            )
        )

        obs, _ = env.reset(seed=6)

        self.assertEqual(obs.shape, (6,))
        self.assertTrue(np.allclose(obs, np.zeros(6, dtype=np.float32)))

    def test_policy_card_labels_truth_observations_as_oracle_baseline(self):
        env_cfg = GymEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_id="chaser",
            observation_fields=(
                ObservationField("truth.chaser.position_eci_km"),
                ObservationField("belief.chaser.state[0]"),
            ),
            action_fields=(),
        )

        card = build_policy_card(env_cfg, policy_name="truth_debug_policy")

        self.assertEqual(card.observation_source["contract"], "oracle_baseline")
        self.assertTrue(card.privileged_data["uses_privileged_data"])
        self.assertIn("truth.chaser.position_eci_km", card.privileged_data["privileged_paths_by_agent"]["chaser"])

    def test_policy_card_defaults_to_observer_owned(self):
        card = build_policy_card(
            GymEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_id="chaser",
                action_fields=(),
            ),
            policy_name="belief_policy",
        )

        self.assertEqual(card.observation_source["contract"], "observer_owned")
        self.assertFalse(card.privileged_data["uses_privileged_data"])
        paths = [
            row["path"]
            for row in card.observation_source["by_agent"]["chaser"]["fields"]
        ]
        self.assertTrue(all(path.startswith("belief.chaser.state") for path in paths))

    def test_observation_classifier_marks_simulator_metrics_privileged(self):
        summary = classify_observation_fields((ObservationField("metrics.range_km"),))

        self.assertEqual(summary["policy_label"], "simulator_metric_baseline")
        self.assertTrue(summary["uses_privileged_data"])

    def test_policy_card_aggregate_preserves_metric_only_baseline_label(self):
        card = build_policy_card(
            GymEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_id="chaser",
                observation_fields=(ObservationField("metrics.range_km"),),
                action_fields=(),
            ),
            policy_name="metric_policy",
        )

        self.assertEqual(card.observation_source["contract"], "simulator_metric_baseline")

    def test_policy_card_aggregate_labels_metric_and_observer_agents_mixed(self):
        card = build_policy_card(
            MultiAgentEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_ids=("chaser", "target"),
                observation_fields_by_agent={
                    "chaser": (ObservationField("metrics.range_km"),),
                    "target": (ObservationField("belief.target.state[0]"),),
                },
            ),
            policy_name="mixed_policy",
        )

        self.assertEqual(card.observation_source["contract"], "mixed")

    def test_invalid_observation_path_for_disabled_agent_fails_fast(self):
        scenario = _base_scenario()
        scenario["objects"]["target"]["enabled"] = False

        with self.assertRaisesRegex(ValueError, "truth.target.position_eci_km"):
            GymSimulationEnv(
                GymEnvConfig(
                    scenario=scenario,
                    controlled_agent_id="chaser",
                    observation_fields=(ObservationField("truth.target.position_eci_km"),),
                    action_fields=(),
                )
            )

    def test_thrust_vector_adapter_supports_hybrid_pointing(self):
        env = GymSimulationEnv(
            GymEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_id="chaser",
                observation_fields=(ObservationField("truth.chaser.attitude_quat_bn"),),
                action_fields=(
                    ActionField("thrust_direction_eci[0]", -1.0, 1.0),
                    ActionField("thrust_direction_eci[1]", -1.0, 1.0),
                    ActionField("thrust_direction_eci[2]", -1.0, 1.0),
                    ActionField("throttle", 0.0, 1.0),
                ),
                action_adapter=ThrustVectorToPointingAdapter(),
            )
        )
        env.reset(seed=9)
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
        self.assertTrue(np.isfinite(info["metrics"]["time_s"]))

    def test_sync_vector_env_batches_resets_and_steps(self):
        env_cfg = GymEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_id="chaser",
            observation_fields=(ObservationField("truth.chaser.position_eci_km"),),
            action_fields=(ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
        )
        vec = make_vector_env(VectorEnvConfig(env_cfg=env_cfg, num_envs=2, parallel=False))
        self.assertIsInstance(vec, SyncVectorSimulationEnv)
        obs, infos = vec.reset(seed=10)
        self.assertEqual(obs.shape, (2, 3))
        self.assertEqual(len(infos["controlled_agent_id"]), 2)
        obs2, rewards, terminated, truncated, infos2 = vec.step(np.array([[0.0], [0.0]], dtype=np.float32))
        self.assertEqual(obs2.shape, (2, 3))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(terminated.shape, (2,))
        self.assertEqual(truncated.shape, (2,))
        self.assertEqual(len(infos2["metrics"]), 2)
        vec.close()

    def test_async_vector_env_batches_resets_and_steps(self):
        env_cfg = GymEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_id="chaser",
            observation_fields=(ObservationField("truth.chaser.position_eci_km"),),
            action_fields=(ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
        )
        vec = make_vector_env(VectorEnvConfig(env_cfg=env_cfg, num_envs=2, parallel=True))
        self.assertIsInstance(vec, AsyncVectorSimulationEnv)
        try:
            obs, infos = vec.reset(seed=20)
            self.assertEqual(obs.shape, (2, 3))
            self.assertEqual(len(infos["controlled_agent_id"]), 2)
            obs2, rewards, terminated, truncated, infos2 = vec.step(np.array([[0.0], [0.0]], dtype=np.float32))
            self.assertEqual(obs2.shape, (2, 3))
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
            self.assertEqual(len(infos2["metrics"]), 2)
        finally:
            vec.close()

    def test_collect_vector_rollout_returns_batched_trajectories(self):
        env_cfg = GymEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_id="chaser",
            observation_fields=(ObservationField("truth.chaser.position_eci_km"),),
            action_fields=(ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
        )
        vec = make_vector_env(VectorEnvConfig(env_cfg=env_cfg, num_envs=2, parallel=False, auto_reset=True))
        try:
            batch = collect_vector_rollout(
                vec,
                policy_fn=lambda obs: np.zeros((obs.shape[0], 1), dtype=np.float32),
                horizon=3,
            )
            self.assertEqual(batch.obs.shape, (3, 2, 3))
            self.assertEqual(batch.actions.shape, (3, 2, 1))
            self.assertEqual(batch.rewards.shape, (3, 2))
            self.assertEqual(batch.terminated.shape, (3, 2))
            self.assertEqual(batch.truncated.shape, (3, 2))
            self.assertEqual(batch.next_obs.shape, (2, 3))
        finally:
            vec.close()

    def test_make_sb3_vec_env_requires_dependency(self):
        env_cfg = GymEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_id="chaser",
            observation_fields=(ObservationField("truth.chaser.position_eci_km"),),
            action_fields=(ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
        )
        with self.assertRaises(ModuleNotFoundError):
            make_sb3_vec_env(VectorEnvConfig(env_cfg=env_cfg, num_envs=2, parallel=False))

    def test_multi_agent_env_steps_chaser_and_target_simultaneously(self):
        cfg = MultiAgentEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_ids=("chaser", "target"),
            observation_fields_by_agent={
                "chaser": (ObservationField("truth.chaser.position_eci_km"), ObservationField("metrics.range_km")),
                "target": (ObservationField("truth.target.position_eci_km"), ObservationField("metrics.range_km")),
            },
            action_fields_by_agent={
                "chaser": (ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
                "target": (ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
            },
        )
        env = MultiAgentSimulationEnv(cfg)
        obs, infos = env.reset(seed=5)
        self.assertEqual(set(obs.keys()), {"chaser", "target"})
        self.assertEqual(obs["chaser"].shape, (4,))
        self.assertEqual(obs["target"].shape, (4,))
        self.assertIn("metrics", infos["chaser"])
        step_obs, rewards, terminations, truncations, step_infos = env.step(
            {
                "chaser": np.array([0.0], dtype=np.float32),
                "target": np.array([0.0], dtype=np.float32),
            }
        )
        self.assertEqual(set(step_obs.keys()), {"chaser", "target"})
        self.assertEqual(set(rewards.keys()), {"chaser", "target"})
        self.assertEqual(set(terminations.keys()), {"chaser", "target"})
        self.assertEqual(set(truncations.keys()), {"chaser", "target"})
        self.assertIn("metrics", step_infos["target"])

    def test_multi_agent_default_observations_use_belief(self):
        env = MultiAgentSimulationEnv(
            MultiAgentEnvConfig(
                scenario=_base_scenario(),
                controlled_agent_ids=("chaser", "target"),
                action_fields_by_agent={"chaser": (), "target": ()},
            )
        )

        obs, _ = env.reset(seed=5)

        self.assertEqual(obs["chaser"].shape, (6,))
        self.assertEqual(obs["target"].shape, (6,))
        self.assertTrue(all(field.path.startswith("belief.chaser.state") for field in env._fields_for_agent("chaser")))

    def test_collect_multi_agent_rollout_returns_per_agent_batches(self):
        cfg = MultiAgentEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_ids=("chaser", "target"),
            observation_fields_by_agent={
                "chaser": (ObservationField("truth.chaser.position_eci_km"),),
                "target": (ObservationField("truth.target.position_eci_km"),),
            },
            action_fields_by_agent={
                "chaser": (ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
                "target": (ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
            },
        )
        env = MultiAgentSimulationEnv(cfg)
        batch = collect_multi_agent_rollout(
            env,
            policy_fns_by_agent={
                "chaser": lambda obs: np.zeros(1, dtype=np.float32),
                "target": lambda obs: np.zeros(1, dtype=np.float32),
            },
            horizon=3,
            reset_kwargs={"seed": 13},
        )
        self.assertEqual(batch.obs_by_agent["chaser"].shape, (3, 3))
        self.assertEqual(batch.actions_by_agent["target"].shape, (3, 1))
        self.assertEqual(batch.rewards_by_agent["chaser"].shape, (3,))
        self.assertEqual(batch.terminated_by_agent["target"].shape, (3,))

    def test_run_self_play_training_returns_history_and_pool(self):
        cfg = MultiAgentEnvConfig(
            scenario=_base_scenario(),
            controlled_agent_ids=("chaser", "target"),
            observation_fields_by_agent={
                "chaser": (ObservationField("truth.chaser.position_eci_km"),),
                "target": (ObservationField("truth.target.position_eci_km"),),
            },
            action_fields_by_agent={
                "chaser": (ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
                "target": (ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),),
            },
        )
        env = MultiAgentSimulationEnv(cfg)
        obs, _ = env.reset(seed=21)
        result = run_self_play_training(
            env,
            policies_by_agent={
                "chaser": LinearPolicy.random(
                    obs_dim=int(obs["chaser"].size), action_dim=1, rng=np.random.default_rng(1)
                ),
                "target": LinearPolicy.random(
                    obs_dim=int(obs["target"].size), action_dim=1, rng=np.random.default_rng(2)
                ),
            },
            trainer_cfg=SelfPlayTrainerConfig(
                update_mode="alternating",
                iterations=2,
                rollout_horizon=3,
                learning_rate=0.01,
                mutation_sigma=0.001,
                snapshot_interval=1,
                max_opponents=4,
                seed=9,
            ),
        )
        self.assertEqual(len(result["history"]), 2)
        self.assertIn("chaser", result["policies_by_agent"])
        self.assertGreaterEqual(len(result["opponent_pool"].snapshots_by_agent["chaser"]), 1)


if __name__ == "__main__":
    unittest.main()
