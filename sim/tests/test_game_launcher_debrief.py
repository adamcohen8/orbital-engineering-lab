from __future__ import annotations

# These owner-aligned tests share deterministic builders and compatibility
# imports from the adjacent support module.
# ruff: noqa: F403, F405
from sim.tests.game_mode_test_support import *


def test_game_launcher_discovers_ordered_training_levels() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "game" / "configs"
    options = discover_game_scenarios(config_dir)

    release_ids = [
        "rpo_00_tutorial",
        "rpo_01_coast_relative_motion",
        "rpo_02_vbar_approach",
        "rpo_03_rbar_approach",
        "rpo_04_rendezvous",
        "rpo_05_passive_cross_track_approach",
        "rpo_06_sun_angle_inspection",
        "rpo_07_elliptic_burn_then_approach",
        "rpo_08_elliptic_nmc",
        "rpo_09_elliptic_rendezvous",
        "rpo_11_evasive_target_survival",
        "rpo_10_defensive_target_demo",
        "rpo_bonus_cislunar_rendezvous",
        "rpo_sandbox",
    ]
    assert [option.scenario_id for option in options if option.scenario_id in release_ids] == release_ids
    by_id = {option.scenario_id: option for option in options}
    assert by_id["rpo_00_tutorial"].title == "Level 0 - Pilot Tutorial"
    assert by_id["rpo_01_coast_relative_motion"].title == "Level 1 - Relative Orbit"
    assert by_id["rpo_02_vbar_approach"].title == "Level 2 - V-Bar Approach"
    assert by_id["rpo_03_rbar_approach"].title == "Level 3 - R-Bar Approach"
    assert by_id["rpo_04_rendezvous"].title == "Level 4 - Rendezvous"
    assert by_id["rpo_05_passive_cross_track_approach"].title == "Level 5 - Safe Inspection"
    tutorial = by_id["rpo_00_tutorial"]
    assert tutorial.player_brief
    assert tutorial.pass_criteria
    assert tutorial.instructor_notes
    assert tutorial.time_budget_s == pytest.approx(18000.0)
    assert tutorial.delta_v_budget_m_s == pytest.approx(12.0)
    assert tutorial.path.name == "game_training_rpo_00_tutorial.yaml"
    assert by_id["rpo_05_passive_cross_track_approach"].path.name == "game_training_rpo_05_passive_cross_track_approach.yaml"
    assert by_id["rpo_06_sun_angle_inspection"].path.name == "game_training_rpo_06_sun_angle_inspection.yaml"
    assert by_id["rpo_06_sun_angle_inspection"].title == "Level 6 - Sun-Angle Inspection"
    assert by_id["rpo_07_elliptic_burn_then_approach"].path.name == "game_training_rpo_07_elliptic_burn_then_approach.yaml"
    assert by_id["rpo_07_elliptic_burn_then_approach"].title == "Level 7 - Elliptical Approach"
    assert by_id["rpo_08_elliptic_nmc"].path.name == "game_training_rpo_08_elliptic_nmc.yaml"
    assert by_id["rpo_08_elliptic_nmc"].title == "Level 8 - Elliptical NMC"
    operator_options = discover_game_scenarios_for_mode(config_dir, mode="operator")
    assert operator_options[0].title == "Level 0 - Operator Tutorial"
    operator_ids = {option.scenario_id for option in operator_options}
    assert "rpo_10_defensive_target_demo" not in operator_ids
    assert "rpo_arcade_pursuit" not in operator_ids
    assert "rpo_11_evasive_target_survival" in operator_ids
    assert by_id["rpo_09_elliptic_rendezvous"].path.name == "game_training_rpo_09_elliptic_rendezvous.yaml"
    assert by_id["rpo_09_elliptic_rendezvous"].title == "Level 9 - Elliptical Rendezvous"
    evasion = by_id["rpo_11_evasive_target_survival"]
    pursuit = by_id["rpo_10_defensive_target_demo"]
    assert evasion.path.name == "game_training_rpo_11_evasive_target_survival.yaml"
    assert evasion.title == "Level 10 - Evasion"
    assert pursuit.path.name == "game_training_rpo_10_defensive_target_demo.yaml"
    assert pursuit.title == "Level 11 - Pursuit"
    assert evasion.delta_v_budget_m_s == pytest.approx(25.0)
    assert evasion.target_delta_v_budget_m_s == pytest.approx(1.0)
    assert pursuit.target_delta_v_budget_m_s == pytest.approx(0.1)
    cislunar = by_id["rpo_bonus_cislunar_rendezvous"]
    assert cislunar.title == "Bonus Level - Cislunar Rendezvous"
    assert cislunar.path.name == "game_training_rpo_bonus_cislunar_rendezvous.yaml"
    assert cislunar.time_budget_s == pytest.approx(259200.0)
    assert cislunar.delta_v_budget_m_s == pytest.approx(75.0)
    sandbox = by_id["rpo_sandbox"]
    assert sandbox.title == "Sandbox"
    assert sandbox.path.name == "game_training_rpo_sandbox.yaml"
    assert sandbox.time_budget_s == pytest.approx(20000.0)
    assert sandbox.delta_v_budget_m_s is None


@pytest.mark.parametrize(
    ("config_name", "expected_title"),
    [
        ("game_training_rpo_02_vbar_approach.yaml", "RPO Trainer Level 2 - V-Bar Approach Debrief"),
        ("game_training_rpo_03_rbar_approach.yaml", "RPO Trainer Level 3 - R-Bar Approach Debrief"),
        ("game_training_rpo_04_rendezvous.yaml", "RPO Trainer Level 4 - Rendezvous Debrief"),
    ],
)
def test_game_debrief_titles_use_level_names_for_levels_2_to_4(config_name: str, expected_title: str) -> None:
    sim_cfg = SimulationConfig.from_yaml(Path(__file__).resolve().parents[1] / "game" / "configs" / config_name)
    training_cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))

    assert game_debrief._debrief_display_title(config=training_cfg, score=object()) == expected_title


def test_bonus_cislunar_rendezvous_uses_cr3bp_frame() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "sim"
        / "game"
        / "configs"
        / "game_training_rpo_bonus_cislunar_rendezvous.yaml"
    )
    config = SimulationConfig.from_yaml(path)

    assert game_runner._game_control_mode(config) == "moon_ric_translation"
    assert game_runner._game_relative_frame(config) == "moon_ric"
    assert game_runner._game_coast_prediction_model(config) == "cr3bp"
    assert game_runner._game_cr3bp_projection_mode(config) == "linearized"
    assert game_runner._game_camera_mode(config) == "rule_toggle_pair"
    assert game_runner._game_camera_rule_mode(config) == "current_pair"
    assert game_runner._game_camera_rule_toggle_enabled(config) is True
    assert game_runner._game_chaser_sprite_path(config) == Path("cislunar_chaser_sprite.png")
    assert game_runner._game_target_sprite_path(config) == Path("cislunar_target_sprite.png")
    assert (
        Path(__file__).resolve().parents[1] / "game" / "assets" / game_runner._game_chaser_sprite_path(config)
    ).is_file()
    assert (
        Path(__file__).resolve().parents[1] / "game" / "assets" / game_runner._game_target_sprite_path(config)
    ).is_file()
    assert game_runner._game_chaser_sprite_diameter_km(config) == pytest.approx(0.05)
    assert game_runner._game_target_sprite_diameter_km(config) == pytest.approx(0.12)
    assert game_runner._game_dashboard_fps_cap(config) == pytest.approx(45.0)
    assert game_runner._game_dashboard_high_speed_fps(config) == pytest.approx(45.0)
    assert game_runner._game_dashboard_high_speed_fps_max_multiple(config) == pytest.approx(100.0)
    assert game_runner._dashboard_fps_for_speed(
        100.0,
        fps_cap=game_runner._game_dashboard_fps_cap(config),
        high_speed_fps=game_runner._game_dashboard_high_speed_fps(config),
        high_speed_fps_max_multiple=game_runner._game_dashboard_high_speed_fps_max_multiple(config),
    ) == pytest.approx(45.0)
    assert game_runner._dashboard_fps_for_speed(
        2000.0,
        fps_cap=game_runner._game_dashboard_fps_cap(config),
        high_speed_fps=game_runner._game_dashboard_high_speed_fps(config),
        high_speed_fps_max_multiple=game_runner._game_dashboard_high_speed_fps_max_multiple(config),
    ) == pytest.approx(30.0)
    assert game_runner._game_target_centered_plot_planes(config) == ("RI", "RC")
    assert game_runner._game_target_centered_plot_axes(config) == {}
    assert game_runner._game_plot_prediction_full_trajectory_only(config) is True
    assert game_runner._game_initial_speed_multiple(config, None) == pytest.approx(10.0)
    assert game_runner._game_initial_speed_multiple(config, 1.0) == pytest.approx(10.0)
    assert game_runner._game_maneuver_control_speed_multiple(config) == pytest.approx(100.0)
    assert game_runner._game_two_rail_speed_control_enabled(config) is True
    assert game_runner._game_speed_dt_schedule(config) == ((10.0, 0.5), (25.0, 1.0), (50.0, 1.0), (100.0, 1.0))
    assert game_runner._game_coast_speed_dt_schedule(config) == (
        (10.0, 0.5),
        (25.0, 1.0),
        (50.0, 2.0),
        (100.0, 2.0),
        (200.0, 10.0),
        (500.0, 25.0),
        (1000.0, 50.0),
        (2000.0, 100.0),
    )
    assert game_runner._game_tick_dt_s(config, 10.0) == pytest.approx(0.5)
    assert game_runner._game_tick_dt_s(config, 25.0) == pytest.approx(1.0)
    assert game_runner._game_tick_dt_s(config, 50.0) == pytest.approx(1.0)
    assert game_runner._game_tick_dt_s(config, 100.0) == pytest.approx(1.0)
    assert game_runner._game_tick_dt_s(config, 1000.0) == pytest.approx(1.0)
    assert game_runner._game_coast_tick_dt_s(config, 100.0) == pytest.approx(2.0)
    assert game_runner._game_coast_tick_dt_s(config, 200.0) == pytest.approx(10.0)
    assert game_runner._game_coast_tick_dt_s(config, 1000.0) == pytest.approx(50.0)
    assert game_runner._game_coast_tick_dt_s(config, 2000.0) == pytest.approx(100.0)
    speed_options = game_runner._game_speed_multiplier_options(config)
    assert speed_options[:2] == pytest.approx((10.0, 25.0))
    assert 1.0 not in speed_options
    assert 2.0 not in speed_options
    assert 5.0 not in speed_options
    assert speed_options[-3:] == pytest.approx((500.0, 1000.0, 2000.0))
    assert game_runner._adjust_speed_multiple(200.0, 1, options=speed_options) == pytest.approx(500.0)
    assert game_runner._adjust_speed_multiple(500.0, 1, options=speed_options) == pytest.approx(1000.0)
    coast_state = KeyboardCommandState()
    burn_state = KeyboardCommandState(pitch=1.0)
    assert game_runner._effective_speed_multiple_for_control(
        config,
        1000.0,
        coast_state,
        control_mode="moon_ric_translation",
        options=speed_options,
    ) == pytest.approx(1000.0)
    assert game_runner._effective_speed_multiple_for_control(
        config,
        1000.0,
        burn_state,
        control_mode="moon_ric_translation",
        options=speed_options,
    ) == pytest.approx(100.0)
    assert game_runner._game_active_tick_dt_s(config, 1000.0, maneuver_active=False) == pytest.approx(50.0)
    assert game_runner._game_active_tick_dt_s(
        config,
        game_runner._effective_speed_multiple_for_control(
            config,
            1000.0,
            burn_state,
            control_mode="moon_ric_translation",
            options=speed_options,
        ),
        maneuver_active=True,
    ) == pytest.approx(1.0)
    assert game_runner._adjust_speed_multiple(1000.0, 1, options=speed_options) == pytest.approx(2000.0)
    assert game_runner._adjust_speed_multiple(2000.0, 1, options=speed_options) == pytest.approx(2000.0)
    assert game_runner._game_cr3bp_coast_prediction_horizon_s(config) == pytest.approx(21600.0)
    assert game_runner._game_cr3bp_active_prediction_horizon_s(config) is None
    assert game_runner._game_cr3bp_coast_prediction_horizon_mode(config) == "default"
    assert game_runner._game_cr3bp_coast_prediction_dt_s(config) == pytest.approx(1.0)
    assert game_runner._game_cr3bp_prediction_coast_update_interval_s(config) == pytest.approx(300.0)
    assert game_runner._game_show_target_hcw_path(config) is False
    assert game_runner._game_target_coast_prediction_horizon_s(config) == pytest.approx(1127210.360660)
    assert game_runner._game_target_coast_prediction_dt_s(config) == pytest.approx(600.0)
    assert game_runner._max_accel_from_config(config, "chaser") == pytest.approx(1.25e-6)
    assert config.scenario.simulator.dt_s == pytest.approx(10.0)
    assert config.scenario.simulator.dynamics["orbit"]["model"] == "cr3bp"
    assert config.scenario.simulator.dynamics["orbit"]["orbit_substep_s"] == pytest.approx(10.0)
    halo_initial = config.scenario.objects["target"].initial_state["cr3bp_halo"]
    assert halo_initial["family"] == "l2_nrho_southern"
    assert halo_initial["phase_time_s"] == pytest.approx(843600.0)
    assert halo_initial["phase_substep_s"] == pytest.approx(120.0)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    assert training_cfg.relative_frame == "moon_ric"
    assert training_cfg.max_goal_speed_km_s == pytest.approx(0.0001)
    assert training_cfg.hard_speed_limit_km_s == pytest.approx(0.0001)
    chaser_initial = config.scenario.objects["chaser"].initial_state["relative_to_target_ric"]
    assert chaser_initial["reference_frame"] == "moon_ric"

    session = SimulationSession.from_config(config)
    snapshot = session.reset()
    assert snapshot is not None
    chaser0 = np.array(snapshot.truth["chaser"], dtype=float).reshape(-1)[:6]
    target0 = np.array(snapshot.truth["target"], dtype=float).reshape(-1)[:6]
    rel0 = relative_moon_ric_state_from_arrays(target0, chaser0)
    target_moon0 = target0 - cr3bp_moon_state_km_s()
    assert target_moon0 == pytest.approx(
        [
            -120.88579680700822,
            -2920.2783735755906,
            2434.1051443969734,
            -0.052740452172463344,
            1.4146334563164559,
            0.6881647620057171,
        ]
    )
    assert np.linalg.norm(target_moon0[:3]) == pytest.approx(3803.6176212945975)
    assert rel0 == pytest.approx([-3.0, 4.0, 0.5, 0.0, 0.0, 2.0e-6])
    assert np.linalg.norm(rel0[:3]) == pytest.approx(5.024937810560445)

    stepped = session.step()
    target_motion = np.linalg.norm(
        np.array(stepped.truth["target"], dtype=float).reshape(-1)[:3]
        - np.array(snapshot.truth["target"], dtype=float).reshape(-1)[:3]
    )
    assert target_motion > 0.0


def test_level3_rbar_approach_uses_iss_target_sprite() -> None:
    path = Path(__file__).resolve().parents[2] / "sim" / "game" / "configs" / "game_training_rpo_03_rbar_approach.yaml"
    config = SimulationConfig.from_yaml(path)
    sprite_path = game_runner._game_target_sprite_path(config)

    assert sprite_path == Path("rpo_iss_target_sprite.png")
    assert (Path(__file__).resolve().parents[1] / "game" / "assets" / sprite_path).is_file()
    assert game_runner._game_target_sprite_diameter_km(config) == pytest.approx(0.11)
    assert RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {})).keepout_radius_km == pytest.approx(
        0.15
    )


def test_cr3bp_large_l1_halo_seed_is_available_for_cislunar_game() -> None:
    state = cr3bp_halo_seed_state_km_s(family="l1_northern_large")

    assert state - cr3bp_l1_state_km_s() == pytest.approx([-4288.472449806286, 0.0, 30752.0, 0.0, 0.198451917044, 0.0])


def test_cr3bp_l2_nrho_seed_is_available_for_cislunar_game() -> None:
    state = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")

    assert state - cr3bp_l1_state_km_s() == pytest.approx(
        [70894.61952879478, 0.0, -69817.0344, 0.0, -0.1042749723224868, 0.0]
    )


def test_cr3bp_moon_ric_transform_round_trips_for_nrho_target() -> None:
    target = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    rel = np.array([5.0, -20.0, -5.0, 0.001, -0.002, 0.003], dtype=float)
    chaser = game_pygame_dashboard._moon_ric_rect_state_to_cr3bp(rel, target)

    assert game_pygame_dashboard._cr3bp_state_to_moon_ric_rect(chaser, target) == pytest.approx(rel)


def test_cr3bp_moon_ric_batched_transform_matches_scalar_rows() -> None:
    target0 = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    references = np.vstack(
        [
            target0,
            propagate_cr3bp_state(target0, 600.0, 0.0),
            propagate_cr3bp_state(target0, 3600.0, 0.0),
        ]
    )
    rel_rows = np.array(
        [
            [5.0, -20.0, -5.0, 0.001, -0.002, 0.003],
            [4.0, -18.0, -4.0, 0.0008, -0.0015, 0.0025],
            [2.0, -12.0, -3.0, 0.0004, -0.0007, 0.0010],
        ],
        dtype=float,
    )
    deputies = np.vstack(
        [
            game_pygame_dashboard._moon_ric_rect_state_to_cr3bp(rel, reference)
            for rel, reference in zip(rel_rows, references)
        ]
    )
    scalar = np.vstack(
        [
            game_pygame_dashboard._cr3bp_state_to_moon_ric_rect(deputy, reference)
            for deputy, reference in zip(deputies, references)
        ]
    )

    batched = game_pygame_dashboard._cr3bp_states_to_moon_ric_rect_rows(deputies, references)

    np.testing.assert_allclose(batched, scalar, rtol=1.0e-12, atol=1.0e-12)


def test_cr3bp_physical_jacobian_matches_force_finite_difference() -> None:
    state = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern") + np.array(
        [100.0, -50.0, 20.0, 1.0e-4, -2.0e-4, 1.5e-4],
        dtype=float,
    )
    analytic = cr3bp_jacobian_physical(state)
    finite_difference = np.zeros((6, 6), dtype=float)
    perturbations = np.array([1.0e-2, 1.0e-2, 1.0e-2, 1.0e-8, 1.0e-8, 1.0e-8], dtype=float)

    for idx, step in enumerate(perturbations):
        delta = np.zeros(6, dtype=float)
        delta[idx] = step
        finite_difference[:, idx] = (
            cr3bp_derivative_physical(state + delta) - cr3bp_derivative_physical(state - delta)
        ) / (2.0 * step)

    assert analytic == pytest.approx(finite_difference, abs=1.0e-9)


def test_cr3bp_reference_stm_matches_propagated_finite_difference() -> None:
    reference0 = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern") + np.array(
        [100.0, -50.0, 20.0, 1.0e-4, -2.0e-4, 1.5e-4],
        dtype=float,
    )
    delta0 = np.array([1.0e-3, -2.0e-3, 5.0e-4, 1.0e-8, -2.0e-8, 1.5e-8], dtype=float)

    reference, stm = propagate_cr3bp_reference_stm(reference0, np.eye(6, dtype=float), 600.0, 0.0)
    deputy = propagate_cr3bp_state(reference0 + delta0, 600.0, 0.0)

    assert deputy - reference == pytest.approx(stm @ delta0, abs=1.0e-10)


def test_linearized_cr3bp_moon_ric_projection_tracks_nonlinear_for_small_offsets() -> None:
    target = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    rel0 = np.array([0.1, -0.2, 0.05, 1.0e-5, -2.0e-5, 1.5e-5], dtype=float)
    times = np.linspace(0.0, 600.0, 7, dtype=float)

    nonlinear = game_pygame_dashboard._nonlinear_cr3bp_moon_ric_coast_prediction(
        rel0, target_state=target, times=times, current_t_s=0.0
    )
    linearized = game_pygame_dashboard._linearized_cr3bp_moon_ric_coast_prediction(
        rel0, target_state=target, times=times, current_t_s=0.0
    )

    assert linearized.shape == nonlinear.shape
    assert linearized[0] == pytest.approx(rel0)
    assert linearized == pytest.approx(nonlinear, abs=2.0e-4)


def test_game_configs_and_optional_music_packaging_contract() -> None:
    def music_tracks(value: object) -> set[str]:
        if isinstance(value, dict):
            tracks = {str(track) for key, track in value.items() if key == "music_track" and track}
            for nested in value.values():
                tracks.update(music_tracks(nested))
            return tracks
        if isinstance(value, list):
            tracks: set[str] = set()
            for nested in value:
                tracks.update(music_tracks(nested))
            return tracks
        return set()

    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(text)
    setuptools_cfg = pyproject["tool"]["setuptools"]
    package_data = set(setuptools_cfg["package-data"]["sim"])
    exclude_package_data = set(setuptools_cfg["exclude-package-data"]["sim"])
    expected_music = {path.name for path in LEVEL_MUSIC_PATHS.values()}
    expected_music.update(
        {
            game_launcher.LAUNCHER_MUSIC_PATH.name,
            MISSION_SUCCESS_MUSIC_PATH.name,
            MISSION_FAILURE_MUSIC_PATH.name,
            ARCADE_ROUND_CLEAR_SOUND_PATH.name,
        }
    )
    for config_path in (root / "sim/game/configs").glob("*.yaml"):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        expected_music.update(music_tracks(cfg))
    expected_music.update(
        path.name for path in (root / "web/rpo-duel-prototype/public/assets").glob("*.wav")
    )
    package_music = {
        Path(pattern).name for pattern in package_data if pattern.startswith("game/music/") and pattern.endswith(".wav")
    }

    assert '"game/configs/*.yaml"' in text
    assert '"game/assets/*.png"' in text
    assert '"game/music/*.md"' in text
    assert "include-package-data = false" in text
    assert "game/music/*.wav" not in package_data
    assert "game/music/*.wav" in exclude_package_data
    assert package_music == set()
    try:
        from tools.export_public import DEFAULT_GAME_MUSIC_FILES
    except ModuleNotFoundError:
        DEFAULT_GAME_MUSIC_FILES = None
    if DEFAULT_GAME_MUSIC_FILES is not None:
        public_export_music = {Path(path).name for path in DEFAULT_GAME_MUSIC_FILES}
        assert public_export_music == expected_music


def test_training_game_configs_default_to_ric_translation(tmp_path: Path) -> None:
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {
        "enabled": True,
        "scenario_id": "custom_training",
        "target_object_id": "training_target",
        "chaser_object_id": "training_chaser",
    }
    cfg["metadata"]["game"].pop("control_mode", None)
    path = tmp_path / "training_default.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert game_runner._game_control_mode(SimulationConfig.from_yaml(path)) == "ric_translation"


def test_non_training_game_configs_default_to_attitude_thrust(tmp_path: Path) -> None:
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"].pop("training", None)
    cfg["metadata"]["game"].pop("control_mode", None)
    path = tmp_path / "legacy_default.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert game_runner._game_control_mode(SimulationConfig.from_yaml(path)) == "attitude_thrust"


def test_translation_control_modes_remains_available_from_input_module() -> None:
    from sim.game.input import TRANSLATION_CONTROL_MODES
    from sim.game.manual import TRANSLATION_CONTROL_MODES as MANUAL_TRANSLATION_CONTROL_MODES

    assert TRANSLATION_CONTROL_MODES is MANUAL_TRANSLATION_CONTROL_MODES


def test_game_boolean_metadata_parses_false_string(tmp_path: Path) -> None:
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["show_coast_prediction"] = "false"
    path = tmp_path / "coast_prediction_false.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert game_runner._game_show_coast_prediction(SimulationConfig.from_yaml(path)) is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("drag_coefficient", "nan", "finite number"),
        ("ballistic_coefficient_min_kg_m2", 0.0, "greater than zero"),
        ("ballistic_coefficient_initial_kg_m2", 500.0, "within the configured bounds"),
        ("lift_area_m2", -1.0, "nonnegative"),
        ("ri_pitch_max_deg", 100.0, "must not exceed 90"),
    ],
)
def test_invalid_aerodynamic_game_metadata_is_rejected(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["aerodynamic_control"] = {field: value}
    path = tmp_path / f"invalid_aero_{field}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        game_runner._game_aerodynamic_control_config(SimulationConfig.from_yaml(path))


def test_public_manual_rpo_example_uses_ric_translation_controls() -> None:
    path = Path(__file__).resolve().parents[2] / "examples" / "configs" / "public_manual_rpo_training.yaml"

    assert game_runner._game_control_mode(SimulationConfig.from_yaml(path)) == "ric_translation"


def test_dashboard_object_ids_follow_training_defaults() -> None:
    training_cfg = RPOTrainingConfig(
        enabled=True,
        target_object_id="training_target",
        chaser_object_id="training_chaser",
    )

    assert game_runner._dashboard_object_ids(training_cfg, {}) == ("training_target", "training_chaser")
    assert game_runner._dashboard_object_ids(
        training_cfg,
        {
            "battlespace_dashboard_target_object_id": "visual_target",
            "battlespace_dashboard_chaser_object_id": "visual_chaser",
        },
    ) == ("visual_target", "visual_chaser")


def test_training_briefing_lines_include_objective_and_assists() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_01_coast_relative_motion.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    lines = game_runner._training_briefing_lines(config, training_cfg, difficulty="hard")

    assert lines[0] == "rpo_01_coast_relative_motion"
    assert "Assists: Hard" in lines
    assert any(line.startswith("Objective:") for line in lines)
    assert any(line.startswith("Gate:") for line in lines)


def test_sandbox_config_is_open_ended_setup_mode() -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    score = RPOTrainingScore(
        scenario_id=training_cfg.scenario_id,
        learning_goal=training_cfg.learning_goal,
        samples=2,
        elapsed_s=12.0,
        closest_approach_km=3.0,
        final_range_km=3.0,
        final_goal_error_km=3.0,
        final_relative_speed_km_s=0.0,
        time_inside_keepout_s=0.0,
        approximate_delta_v_m_s=1.25,
        target_delta_v_m_s=0.0,
        burn_axes_satisfied=(),
        phase_burns_satisfied=(),
        speed_multiplier_changed=False,
        coast_after_burn_satisfied=False,
        coast_after_burn_s=0.0,
        guided_tutorial_burns_satisfied=(),
        guided_tutorial_burns_total=0,
        guided_tutorial_speed_satisfied=True,
        guided_tutorial_speed_target=None,
        achieved_time_s=None,
        min_goal_error_km=3.0,
        final_nmt_radial_amplitude_km=float("nan"),
        final_nmt_cross_track_amplitude_km=float("nan"),
        final_nmt_radial_amplitude_error_km=float("nan"),
        final_nmt_cross_track_amplitude_error_km=float("nan"),
        final_nmt_drift_velocity_error_km_s=float("nan"),
        goal_met=False,
        level_passed=False,
        level_failed=False,
        pass_fail_reasons=("Sandbox active; no pass/fail objective.",),
        keepout_violation=False,
        hard_speed_limit_violation=False,
        forbidden_region_violation=False,
        forbidden_region_names=(),
        approach_gate_violation=False,
        approach_gate_names=(),
        approach_gates_satisfied=0,
        approach_gates_total=0,
        inspection_gates_satisfied=0,
        inspection_gates_total=0,
        inspection_gate_names=(),
        hints=(),
    )

    assert game_runner._game_sandbox_enabled(config) is True
    assert game_runner._game_camera_rule_mode(config) == "full_trajectory"
    assert training_cfg.sandbox_mode is True
    assert training_cfg.max_time_s == pytest.approx(20000.0)
    assert training_cfg.max_delta_v_m_s is None
    assert config.scenario.simulator.duration_s == pytest.approx(20000.0)
    assert config.scenario.simulator.dt_s == pytest.approx(1.0)
    assert game_runner._game_target_centered_plot_planes(config) == ("RI", "RC")
    assert game_runner._sandbox_coast_prediction_model(game_runner._sandbox_setup_from_config(config)) == "hcw"
    assert "INFO dV Used 1.250 m/s" in game_runner._mission_metrics(training_cfg, score)
    assert game_runner._mission_checklist(training_cfg, score) == ("INFO Experiment Freely",)


def test_training_tracker_sandbox_stays_open_until_time_limit_then_succeeds() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="sandbox-unit",
        sandbox_mode=True,
        max_time_s=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -3.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])

    for time_s in (0.0, 2.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is True
    assert score.level_failed is False
    assert "Sandbox complete; time limit reached." in score.pass_fail_reasons

    active_cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="sandbox-active-unit",
        sandbox_mode=True,
        max_time_s=10.0,
    )
    active_tracker = RPOTrainingTracker(active_cfg)
    for time_s in (0.0, 2.0):
        active_tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )
    active_score = active_tracker.score()

    assert active_score.level_passed is False
    assert active_score.level_failed is False
    assert active_score.pass_fail_reasons == ("Sandbox active; no pass/fail objective.",)
    assert active_tracker.current_hint() == "Sandbox: Maneuver freely, coast, and watch the relative orbit respond."


def test_sandbox_setup_form_values_update_runtime_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    config = SimulationConfig.from_yaml(config_path)
    values = [
        "7200",
        "0.1",
        "30",
        "40",
        "50",
        "45",
        "1.2",
        "-4.5",
        "0.25",
        "0.3",
        "-0.4",
        "0.5",
    ]

    setup, error = game_runner._sandbox_setup_from_text_values(values)
    assert error == ""
    assert setup == SandboxSetupValues(
        target_a_km=7200.0,
        target_ecc=0.1,
        target_inc_deg=30.0,
        target_raan_deg=40.0,
        target_argp_deg=50.0,
        target_true_anomaly_deg=45.0,
        radial_km=1.2,
        in_track_km=-4.5,
        cross_track_km=0.25,
        radial_rate_m_s=0.3,
        in_track_rate_m_s=-0.4,
        cross_track_rate_m_s=0.5,
    )

    updated = game_runner._apply_sandbox_setup_to_config(config, setup)
    chaser = updated.scenario.objects["chaser"]
    target = updated.scenario.objects["target"]
    training_cfg = RPOTrainingConfig.from_metadata(dict(updated.scenario.metadata or {}))

    assert chaser.initial_state["relative_to_target_ric"]["state"] == pytest.approx(
        [1.2, -4.5, 0.25, 0.0003, -0.0004, 0.0005]
    )
    assert target.initial_state["coes"]["a_km"] == pytest.approx(7200.0)
    assert target.initial_state["coes"]["ecc"] == pytest.approx(0.1)
    assert target.initial_state["coes"]["inc_deg"] == pytest.approx(30.0)
    assert target.initial_state["coes"]["raan_deg"] == pytest.approx(40.0)
    assert target.initial_state["coes"]["argp_deg"] == pytest.approx(50.0)
    assert target.initial_state["coes"]["true_anomaly_deg"] == pytest.approx(45.0)
    assert game_runner._game_coast_prediction_model(updated) == "yamanaka_ankersen"
    assert game_runner._game_camera_rule_mode(updated) == "full_trajectory"
    assert game_runner._game_target_centered_plot_planes(updated) == ("RI", "RC")
    assert training_cfg.sandbox_mode is True
    assert training_cfg.max_time_s == pytest.approx(20000.0)
    assert training_cfg.max_delta_v_m_s is None
    assert updated.scenario.simulator.duration_s == pytest.approx(20000.0)
    assert updated.scenario.simulator.dt_s == pytest.approx(1.0)

    attempt_config = game_session._attempt_config_for_training_clock(updated, training_cfg)
    assert attempt_config.scenario.simulator.duration_s == pytest.approx(20000.0)
    assert attempt_config.scenario.simulator.dt_s == pytest.approx(1.0)

    session = SimulationSession.from_config(attempt_config)
    snapshot = session.step()
    assert {"target", "chaser"}.issubset(snapshot.truth)


def test_sandbox_projection_switches_for_any_nonzero_eccentricity() -> None:
    assert game_runner._sandbox_coast_prediction_model(SandboxSetupValues(target_ecc=0.0)) == "hcw"
    assert (
        game_runner._sandbox_coast_prediction_model(SandboxSetupValues(target_ecc=np.nextafter(0.0, 1.0)))
        == "yamanaka_ankersen"
    )


def test_sandbox_setup_form_validation() -> None:
    setup, error = game_runner._sandbox_setup_from_text_values(["0"] * 12)
    assert setup is None
    assert error == "Target Semimajor Axis must be positive."

    setup, error = game_runner._sandbox_setup_from_text_values(
        ["7000", "1", "45", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
    )
    assert setup is None
    assert error == "Target Eccentricity must satisfy 0 <= e < 1."


def test_sandbox_setup_form_uses_dedicated_two_column_screen(monkeypatch) -> None:
    class FakeEventSource:
        def __init__(self, batches: list[list[object]]) -> None:
            self._batches = list(batches)

        def get(self) -> list[object]:
            if not self._batches:
                return []
            return self._batches.pop(0)

    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        KEYUP = "keyup"
        MOUSEWHEEL = "mousewheel"
        K_ESCAPE = "escape"
        K_RETURN = "return"
        K_KP_ENTER = "kp_enter"
        K_SPACE = "space"
        K_TAB = "tab"
        K_DOWN = "down"
        K_UP = "up"
        K_LEFT = "left"
        K_RIGHT = "right"
        KMOD_SHIFT = 1
        K_PAGEUP = "pageup"
        K_PAGEDOWN = "pagedown"
        K_HOME = "home"
        K_END = "end"
        K_BACKSPACE = "backspace"
        K_DELETE = "delete"
        MOUSEBUTTONDOWN = "mousedown"

        class Rect:
            def __init__(self, x, y, w, h):
                self.x, self.y, self.w, self.h = x, y, w, h
                self.width, self.height = w, h

            @property
            def right(self):
                return self.x + self.w

        def __init__(self, batches: list[list[object]]) -> None:
            self.event = FakeEventSource(batches)
            self.event.get_grab = lambda: True
            self.event.set_grab = lambda value: None
            self.mouse = type(
                "Mouse",
                (),
                {
                    "get_visible": staticmethod(lambda: False),
                    "set_visible": staticmethod(lambda value: None),
                    "get_pos": staticmethod(lambda: (0, 0)),
                },
            )()
            self.display = type("Display", (), {"flip": staticmethod(lambda: None)})()

    class FakeDashboard:
        closed = False

        def __init__(self, batches: list[list[object]]) -> None:
            self.pygame = FakePygame(batches)
            self.screen = type("Screen", (), {"get_size": staticmethod(lambda: (1280, 800))})()
            self.font = object()
            self.small_font = object()
            self.large_font = object()

        def tick(self, _: float) -> None:
            return None

    tab = type(
        "KeyEvent",
        (),
        {"type": FakePygame.KEYDOWN, "key": FakePygame.K_TAB, "unicode": "", "mod": 0},
    )()
    enter = type("KeyEvent", (), {"type": FakePygame.KEYDOWN, "key": FakePygame.K_RETURN, "unicode": ""})()
    dashboard = FakeDashboard([[tab], [enter]])
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    config = SimulationConfig.from_yaml(config_path)
    draws: list[dict[str, object]] = []
    monkeypatch.setattr(
        "sim.game.launcher_widgets._draw_sandbox_setup_screen",
        lambda *_args, **kwargs: draws.append(kwargs),
    )

    setup = game_runner._run_sandbox_setup_form(
        dashboard,
        config=config,
        speed_multiple=1.0,
        level_title="Sandbox",
    )

    assert setup == game_runner._sandbox_setup_from_config(config)
    assert len(draws) == 1
    assert len(draws[0]["field_rects"]) == 12
    assert draws[0]["active_index"] == 1


def test_launcher_hit_test_accounts_for_scroll_offset() -> None:
    assert game_launcher._option_index_at_pos((60, 120), count=12, scroll_offset=4) is None
    assert game_launcher._option_index_at_pos((60, 140), count=12, scroll_offset=0) == 0
    assert game_launcher._option_index_at_pos((60, 140), count=12, scroll_offset=4) == 4
    assert game_launcher._option_index_at_pos((60, 200), count=12, scroll_offset=4) == 4
    assert game_launcher._option_index_at_pos((60, 214), count=12, scroll_offset=4) == 5
    assert game_launcher._option_index_at_pos((60, 204), count=12, scroll_offset=4) is None


def test_launcher_scroll_tracks_keyboard_selection() -> None:
    assert game_launcher._scroll_for_selection(0, 0, count=12, screen_height=680) == 0
    assert game_launcher._scroll_for_selection(6, 0, count=12, screen_height=680) == 1
    assert game_launcher._scroll_for_selection(11, 1, count=12, screen_height=680) == 6
    assert game_launcher._scroll_for_selection(4, 6, count=12, screen_height=680) == 4


def test_launcher_keyboard_selection_wraps_at_edges() -> None:
    assert game_launcher._advance_launcher_selection(0, -1, count=12) == 11
    assert game_launcher._advance_launcher_selection(11, 1, count=12) == 0
    assert game_launcher._advance_launcher_selection(4, 1, count=12) == 5
    assert game_launcher._advance_launcher_selection(4, -1, count=12) == 3
    assert game_launcher._advance_launcher_selection(0, -1, count=4) == 3
    assert game_launcher._advance_launcher_selection(3, 1, count=4) == 0
    assert game_launcher._advance_launcher_selection(0, -1, count=0) == 0


def test_launcher_difficulty_helpers_support_picker() -> None:
    assert game_launcher._difficulty_index("easy") == 0
    assert game_launcher._difficulty_index("normal") == 1
    assert game_launcher._difficulty_index("expert") == 3
    assert game_launcher._difficulty_index("unknown") == 0
    assert game_launcher._difficulty_at_pos((650, 94)) == "easy"
    assert game_launcher._difficulty_at_pos((908, 94)) == "extreme"
    assert game_launcher._difficulty_at_pos((500, 94)) is None


def test_launcher_progress_helpers_persist_user_state_without_mutating_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_PROGRESS_PATH", str(tmp_path / "progress.yaml"))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "game_training_rpo_01_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_01_demo"}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    original_yaml = path.read_text(encoding="utf-8")

    record_game_progress(path, "hard")
    options = discover_game_scenarios(config_dir)

    assert path.read_text(encoding="utf-8") == original_yaml
    assert game_launcher._game_progress_path().exists()
    assert options[0].completed_difficulties == ("hard",)
    assert options[0].high_score == 0
    assert game_launcher._progress_stars(options[0].completed_difficulties) == "★★★☆"

    record_game_progress(path, "medium", score=1200)
    record_game_progress(path, "easy", score=900)
    options = discover_game_scenarios(config_dir)

    assert options[0].completed_difficulties == ("easy", "medium", "hard")
    assert options[0].high_score == 1200

    record_game_progress(path, "extreme", score=2500, completed=False)
    options = discover_game_scenarios(config_dir)

    assert options[0].completed_difficulties == ("easy", "medium", "hard")
    assert options[0].high_score == 2500

    clear_game_progress(config_dir)
    options = discover_game_scenarios(config_dir)

    assert path.read_text(encoding="utf-8") == original_yaml
    assert options[0].completed_difficulties == ()
    assert options[0].high_score == 0
    assert game_launcher._progress_stars(options[0].completed_difficulties) == "☆☆☆☆"


def test_launcher_state_paths_preserve_existing_progress_and_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OEL_GAME_PROGRESS_PATH", raising=False)
    monkeypatch.delenv("OEL_GAME_SETTINGS_PATH", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    legacy = tmp_path / ".orbital_engagement_lab"
    legacy.mkdir()
    (legacy / "game_progress.yaml").write_text("scenarios: {}\n", encoding="utf-8")
    (legacy / "game_settings.yaml").write_text("frame_convention: {}\n", encoding="utf-8")

    assert game_launcher._game_progress_path() == legacy / "game_progress.yaml"
    assert game_launcher._game_settings_path() == legacy / "game_settings.yaml"

    current = tmp_path / ".orbital_engineering_lab"
    current.mkdir()
    (current / "game_progress.yaml").write_text("scenarios: {}\n", encoding="utf-8")
    assert game_launcher._game_progress_path() == current / "game_progress.yaml"
    assert game_launcher._game_settings_path() == legacy / "game_settings.yaml"


def test_launcher_progress_is_separate_for_operator_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_PROGRESS_PATH", str(tmp_path / "progress.yaml"))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "game_training_rpo_02_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_02_demo"}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    record_game_progress(path, "hard", score=100, mode="pilot")
    record_game_progress(path, "easy", score=700, mode="operator")
    pilot = discover_game_scenarios_for_mode(config_dir, mode="pilot")[0]
    operator = discover_game_scenarios_for_mode(config_dir, mode="operator")[0]

    assert pilot.completed_difficulties == ("hard",)
    assert pilot.high_score == 100
    assert operator.completed_difficulties == ("easy",)
    assert operator.high_score == 700


def test_launcher_settings_persist_frame_convention(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_SETTINGS_PATH", str(tmp_path / "settings.yaml"))
    plan = parse_operator_burn_plan("T= 50 s, 2.0 m/s R, 1.0 m/s I, 0.2 m/s C")

    game_launcher._save_game_settings(
        GameSettings(
            frame_convention=FrameConvention(
                positive_in_track="left",
                positive_cross_track="clockwise",
            ),
            presentation_mode="high_refresh",
            ask_frame_convention_on_launch=False,
            last_game_mode="operator",
            operator_burn_scripts={"rpo_01_relative_orbit": plan},
        )
    )
    loaded = game_launcher._load_game_settings()

    assert game_launcher._game_settings_path().exists()
    assert loaded.frame_convention == FrameConvention(
        positive_in_track="left",
        positive_cross_track="clockwise",
    )
    assert loaded.ask_frame_convention_on_launch is False
    assert loaded.presentation_mode == "high_refresh"
    assert loaded.last_game_mode == "operator"
    assert loaded.operator_burn_scripts["rpo_01_relative_orbit"].burns == plan.burns


def test_frame_convention_dialog_settings_preserve_saved_operator_scripts() -> None:
    plan = parse_operator_burn_plan("T= 50 s, 2.0 m/s R")
    settings = GameSettings(
        frame_convention=FrameConvention(),
        operator_burn_scripts={"rpo_01_relative_orbit": plan},
    )

    updated = game_launcher._frame_convention_dialog_settings(
        settings,
        frame_convention=frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE),
        presentation_mode="auto",
        dont_ask_again=True,
        selected_mode="operator",
    )

    assert updated.ask_frame_convention_on_launch is False
    assert updated.presentation_mode == "auto"
    assert updated.last_game_mode == "operator"
    assert updated.operator_burn_scripts["rpo_01_relative_orbit"].burns == plan.burns


def test_operator_burn_scripts_are_saved_by_scenario_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_SETTINGS_PATH", str(tmp_path / "settings.yaml"))
    plan = parse_operator_burn_plan("T= 10 s, 0.5 m/s R\nT= 40 s, -0.2 m/s I, 0.1 m/s C")

    assert game_launcher._load_saved_operator_burn_plan("rpo_02_vbar_approach") is None

    game_launcher._save_operator_burn_plan("rpo_02_vbar_approach", plan)
    loaded = game_launcher._load_saved_operator_burn_plan("rpo_02_vbar_approach")

    assert loaded is not None
    assert loaded.burns == plan.burns

    game_launcher._save_operator_burn_plan("rpo_02_vbar_approach", OperatorBurnPlan())

    assert game_launcher._load_saved_operator_burn_plan("rpo_02_vbar_approach") == OperatorBurnPlan()


def test_launcher_settings_persist_frame_convention_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_SETTINGS_PATH", str(tmp_path / "settings.yaml"))
    settings_path = game_launcher._game_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        f"frame_convention:\n  preset: {FRAME_CONVENTION_PRESET_SPACE_FORCE}\nask_frame_convention_on_launch: false\n",
        encoding="utf-8",
    )

    loaded = game_launcher._load_game_settings()

    assert loaded.frame_convention == FrameConvention(
        positive_in_track="left",
        positive_cross_track="clockwise",
    )
    assert loaded.ask_frame_convention_on_launch is False
    assert loaded.presentation_mode == "auto"
    assert loaded.last_game_mode is None


def test_space_force_frame_mirrors_only_display_in_track_axis() -> None:
    convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)

    assert frame_convention_display_axis_sign(convention, 0) == pytest.approx(1.0)
    assert frame_convention_display_axis_sign(convention, 1) == pytest.approx(-1.0)
    assert frame_convention_display_axis_sign(convention, 2) == pytest.approx(1.0)


def test_dashboard_frame_convention_does_not_flip_cislunar_axes() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
    dashboard.relative_frame = "ric"

    assert dashboard._axis_display_sign(1) == pytest.approx(-1.0)

    dashboard.relative_frame = "cislunar_l1"

    assert dashboard._axis_display_sign(1) == pytest.approx(1.0)


def test_dashboard_signed_axis_labels_include_positive_and_negative_directions() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "ric"

    assert dashboard._signed_axis_label_for_plot(0, 1) == "+R"
    assert dashboard._signed_axis_label_for_plot(0, -1) == "-R"
    assert dashboard._signed_axis_label_for_plot(1, 1) == "+I"
    assert dashboard._signed_axis_label_for_plot(2, -1) == "-C"


def test_operator_preview_signed_axis_labels_include_positive_and_negative_directions() -> None:
    assert game_launcher._operator_signed_axis_label(0, 1) == "+R"
    assert game_launcher._operator_signed_axis_label(1, -1) == "-I"
    assert game_launcher._operator_signed_axis_label(2, 1) == "+C"


def test_frame_convention_dialog_hit_testing() -> None:
    choices = game_launcher._frame_convention_dialog_choice_rects(1040, 680)
    graphics = game_launcher._frame_convention_dialog_graphics_rects(1040, 680)
    checkbox = game_launcher._frame_convention_dialog_checkbox_rect(1040, 680)
    continue_rect = game_launcher._frame_convention_dialog_continue_rect(1040, 680)

    assert (
        game_launcher._frame_convention_dialog_action(
            (choices["oel_default"][0] + 2, choices["oel_default"][1] + 2),
            width=1040,
            height=680,
        )
        == "oel_default"
    )
    assert (
        game_launcher._frame_convention_dialog_action(
            (choices["space_force"][0] + 2, choices["space_force"][1] + 2),
            width=1040,
            height=680,
        )
        == "space_force"
    )
    for mode in ("compatibility", "standard", "high_refresh", "auto"):
        assert (
            game_launcher._frame_convention_dialog_action(
                (graphics[mode][0] + 2, graphics[mode][1] + 2),
                width=1040,
                height=680,
            )
            == mode
        )
    assert (
        game_launcher._frame_convention_dialog_action((checkbox[0] + 3, checkbox[1] + 3), width=1040, height=680)
        == "dont_ask_again"
    )
    assert (
        game_launcher._frame_convention_dialog_action(
            (continue_rect[0] + 3, continue_rect[1] + 3), width=1040, height=680
        )
        == "continue"
    )
    assert game_launcher._frame_convention_dialog_action((10, 10), width=1040, height=680) is None


def test_clear_progress_suppresses_legacy_yaml_progress_without_mutating_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_PROGRESS_PATH", str(tmp_path / "progress.yaml"))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "game_training_rpo_01_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_01_demo"}
    cfg["metadata"]["game"]["progress"] = {"completed_difficulties": ["hard"]}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    original_yaml = path.read_text(encoding="utf-8")

    assert discover_game_scenarios(config_dir)[0].completed_difficulties == ("hard",)

    clear_game_progress(config_dir)
    options = discover_game_scenarios(config_dir)

    assert path.read_text(encoding="utf-8") == original_yaml
    assert options[0].completed_difficulties == ()


def test_clear_progress_button_hit_test() -> None:
    assert game_launcher._clear_progress_at_pos((860, 44)) is True
    assert game_launcher._clear_progress_at_pos((800, 44)) is False


def test_record_video_button_hit_test() -> None:
    assert game_launcher._record_video_at_pos((700, 44)) is True
    assert game_launcher._record_video_at_pos((650, 44)) is False


def test_music_button_hit_test() -> None:
    assert game_launcher._music_at_pos((536, 44)) is True
    assert game_launcher._music_at_pos((500, 44)) is False


def test_settings_button_hit_test() -> None:
    assert game_launcher._settings_button_at_pos((980, 642), width=1040, height=680) is True
    assert game_launcher._settings_button_at_pos((1008, 642), width=1040, height=680) is False


def test_launcher_widgets_import_shared_layout_helpers_directly() -> None:
    from sim.game import launcher_widgets

    helper_names = (
        "_frame_convention_dialog_checkbox_rect",
        "_frame_convention_dialog_choice_rects",
        "_frame_convention_dialog_continue_rect",
        "_frame_convention_dialog_rect",
        "_launcher_panel_height",
        "_mode_toggle_rect",
        "_preview_bounds",
        "_settings_button_rect",
        "_visible_option_count",
    )

    assert all(callable(getattr(launcher_widgets, name, None)) for name in helper_names)
    assert launcher_widgets._mode_toggle_rect(1040, 680) == (806, 624, 148, 34)


def test_operator_planning_resolves_widget_renderers_after_decomposition() -> None:
    from sim.game import launcher_widgets, operator_planning

    assert (
        operator_planning._operator_widget_renderer("_draw_operator_plan_screen")
        is launcher_widgets._draw_operator_plan_screen
    )
    assert (
        operator_planning._operator_widget_renderer("_draw_operator_prebrief_screen")
        is launcher_widgets._draw_operator_prebrief_screen
    )
    assert callable(operator_planning._text)


def test_start_screen_event_action_begins_on_any_non_escape_key() -> None:
    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        K_ESCAPE = "escape"

    ordinary_key = type("Event", (), {"type": FakePygame.KEYDOWN, "key": "return"})()
    escape_key = type("Event", (), {"type": FakePygame.KEYDOWN, "key": FakePygame.K_ESCAPE})()
    quit_event = type("Event", (), {"type": FakePygame.QUIT})()
    mouse_event = type("Event", (), {"type": "mouse"})()

    assert game_launcher._start_screen_event_action(FakePygame, ordinary_key) == "begin"
    assert game_launcher._start_screen_event_action(FakePygame, escape_key) == "quit"
    assert game_launcher._start_screen_event_action(FakePygame, quit_event) == "quit"
    assert game_launcher._start_screen_event_action(FakePygame, mouse_event) == "ignore"


def test_choose_game_launch_can_skip_start_screen(monkeypatch) -> None:
    calls: list[bool] = []

    monkeypatch.setattr(game_launcher, "_load_game_settings", lambda: GameSettings())
    monkeypatch.setattr(
        game_launcher, "discover_game_scenarios_for_mode", lambda config_dir=None, *, mode="pilot": ("option",)
    )

    def fake_run_launcher(options, *, show_start_screen=True, initial_mode="pilot"):
        calls.append(bool(show_start_screen))
        return None

    monkeypatch.setattr(game_launcher, "_run_launcher", fake_run_launcher)

    game_launcher.choose_game_launch(show_start_screen=False)

    assert calls == [False]


def test_choose_game_launch_opens_in_initial_mode(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(game_launcher, "_load_game_settings", lambda: GameSettings())

    def fake_discover(config_dir=None, *, mode="pilot"):
        calls.append(("discover", mode))
        return ("option",)

    def fake_run_launcher(options, *, show_start_screen=True, initial_mode="pilot"):
        calls.append(("run", initial_mode))
        return None

    monkeypatch.setattr(game_launcher, "discover_game_scenarios_for_mode", fake_discover)
    monkeypatch.setattr(game_launcher, "_run_launcher", fake_run_launcher)

    game_launcher.choose_game_launch(show_start_screen=False, initial_mode="operator")

    assert calls == [("discover", "operator"), ("run", "operator")]


def test_choose_game_launch_prefers_saved_last_mode(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(game_launcher, "_load_game_settings", lambda: GameSettings(last_game_mode="operator"))

    def fake_discover(config_dir=None, *, mode="pilot"):
        calls.append(("discover", mode))
        return ("option",)

    def fake_run_launcher(options, *, show_start_screen=True, initial_mode="pilot"):
        calls.append(("run", initial_mode))
        return None

    monkeypatch.setattr(game_launcher, "discover_game_scenarios_for_mode", fake_discover)
    monkeypatch.setattr(game_launcher, "_run_launcher", fake_run_launcher)

    game_launcher.choose_game_launch(show_start_screen=False, initial_mode="pilot")

    assert calls == [("discover", "operator"), ("run", "operator")]


def test_operator_selection_defers_script_screen_to_gameplay(monkeypatch) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "game" / "configs"
    option = next(
        option
        for option in discover_game_scenarios_for_mode(config_dir, mode="operator")
        if option.scenario_id == "rpo_01_coast_relative_motion"
    )
    calls: list[str] = []

    class FakePygame:
        pass

    def fail_prebrief(*_args, **_kwargs):
        raise AssertionError("operator launch should not show the prebrief screen")

    def fail_plan_screen(*_args, **_kwargs):
        calls.append("plan")
        raise AssertionError("operator launch should defer the script screen to gameplay")

    monkeypatch.setattr(game_launcher, "_enter_operator_fullscreen", lambda pygame, screen: None)
    monkeypatch.setattr(game_launcher, "_run_operator_prebrief_screen", fail_prebrief)
    monkeypatch.setattr(game_launcher, "_run_operator_plan_screen", fail_plan_screen)

    selection = game_launcher._selection_for_launch(
        FakePygame(),
        object(),
        object(),
        option=option,
        difficulty="easy",
        music_enabled=False,
        record_video=False,
        mode="operator",
        frame_convention=FrameConvention(),
        presentation_mode="high_refresh",
        font=_FixedWidthFont(),
        small_font=_FixedWidthFont(),
        title_font=_FixedWidthFont(),
    )

    assert calls == []
    assert selection is not None
    assert selection.operator_burn_plan is None
    assert selection.presentation_mode == "high_refresh"
    assert selection.skip_initial_briefing is True


def test_start_screen_artwork_rect_fits_screen_without_distortion() -> None:
    rect = game_launcher._start_artwork_rect((1672, 941), (1040, 680))
    x, y, width, height = rect

    assert x >= 0
    assert y >= 0
    assert width <= 1040
    assert height <= 680
    assert width == 1040
    assert height < 680
    assert width / height == pytest.approx(1672 / 941, rel=0.02)


def test_launcher_hides_tutorial_progress_text() -> None:
    option = _launcher_option_with_long_preview()
    tutorial = replace(option, scenario_id="rpo_00_tutorial", title="Level 0 - Pilot Tutorial", level_number=0)

    assert game_launcher._show_progress_text(tutorial) is False
    assert game_launcher._show_progress_text(option) is True


def test_launcher_preview_wraps_text_to_pixel_width() -> None:
    font = _FixedWidthFont()
    lines = game_launcher._wrap_text_px(
        "Use small pulses and long coast arcs to shape the relative orbit.",
        font,
        160,
    )

    assert len(lines) > 1
    assert all(font.size(line)[0] <= 160 for line in lines)


def test_launcher_preview_wraps_long_budget_line() -> None:
    font = _FixedWidthFont()
    option = replace(
        _launcher_option_with_long_preview(),
        time_budget_s=7200.0,
        delta_v_budget_m_s=1.0,
        goal_speed_km_s=0.00005,
        target_delta_v_budget_m_s=2.0,
    )

    lines = game_launcher._wrapped_budget_lines(option, font, 300)

    assert len(lines) > 1
    assert all(font.size(line)[0] <= 300 for line in lines)
    assert not lines[-1].endswith("...")


def test_launcher_preview_truncates_long_unbroken_words_to_pixel_width() -> None:
    font = _FixedWidthFont()
    text = game_launcher._fit_text_px("supercalifragilisticexpialidocious", font, 80)

    assert text.endswith("...")
    assert font.size(text)[0] <= 80


def test_launcher_preview_scroll_clamps_to_scrollable_content() -> None:
    font = _FixedWidthFont()
    option = _launcher_option_with_long_preview()
    bounds = game_launcher._preview_bounds(1040, 320)
    content_height = game_launcher._preview_content_height(option, font=font, small_font=font, width_px=bounds[2] - 40)

    scroll = game_launcher._clamp_preview_scroll_px(
        100_000, option=option, font=font, small_font=font, preview_bounds=bounds
    )

    assert content_height > bounds[3] - 40
    assert scroll == content_height - (bounds[3] - 40)


def test_launcher_preview_scroll_is_zero_when_content_fits() -> None:
    font = _FixedWidthFont()
    option = GameScenarioOption(
        path=Path("level.yaml"),
        scenario_id="rpo_test",
        title="Level Test",
        description="",
        learning_goal="Short objective.",
        player_brief="Short brief.",
        pass_criteria=("Short pass.",),
        instructor_notes=(),
        difficulty="easy",
        time_budget_s=None,
        delta_v_budget_m_s=None,
        goal_speed_km_s=None,
        target_delta_v_budget_m_s=None,
        completed_difficulties=(),
        high_score=0,
        level_number=1,
    )

    scroll = game_launcher._clamp_preview_scroll_px(
        120, option=option, font=font, small_font=font, preview_bounds=(490, 124, 420, 480)
    )

    assert scroll == 0


def test_game_recording_path_uses_scenario_difficulty_and_attempt(tmp_path: Path) -> None:
    path = game_recording_path(
        scenario_name="RPO 09 Defensive Target Demo",
        difficulty="Hard",
        attempt_index=3,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 14, 12, 30, 45),
    )

    assert path == tmp_path / "rpo_09_defensive_target_demo_hard_20260514_123045_attempt03.mp4"


def test_game_clip_recording_path_uses_clip_folder_and_index(tmp_path: Path) -> None:
    path = game_clip_recording_path(
        scenario_name="RPO 01 Relative Orbit",
        difficulty="Easy",
        clip_index=2,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 27, 9, 8, 7),
    )

    assert path == tmp_path / "clips" / "rpo_01_relative_orbit_easy_20260527_090807_clip02.mp4"


def test_game_debrief_writer_exports_summary_and_replay(tmp_path: Path) -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-debrief",
        learning_goal="test",
        goal_range_km=0.25,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target_state, "chaser": chaser_state},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float), "target": np.zeros(3, dtype=float)},
            },
        )()
    )
    score = tracker.score()
    path = game_debrief_path(
        scenario_id=cfg.scenario_id,
        difficulty="easy",
        attempt_index=2,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 14, 12, 30, 45),
    )

    out = write_game_debrief(
        path,
        config=cfg,
        score=score,
        difficulty="easy",
        objective_checklist=game_runner._mission_checklist(cfg, score),
        arcade_score=123,
        arcade_seed=456,
        arcade_round_index=7,
        recording_path=tmp_path / "attempt.mp4",
        replay_history=tracker_replay_history(tracker),
    )
    summary_path = out.parent / "summary.json"
    payload = yaml.safe_load(summary_path.read_text(encoding="utf-8"))

    assert out == tmp_path / "unit_debrief" / "attempt_002_easy_20260514_123045" / "report.md"
    assert summary_path == out.parent / "summary.json"
    assert payload["scenario_id"] == "unit-debrief"
    assert payload["level_passed"] is True
    assert payload["score"]["arcade_score"] == 123
    assert payload["score"]["arcade_seed"] == 456
    assert payload["score"]["arcade_round_index"] == 7
    assert payload["artifacts"]["recording_path"].endswith("attempt.mp4")
    assert payload["artifacts"]["report_path"].endswith("report.md")
    assert payload["artifacts"]["summary_path"].endswith("summary.json")
    assert "ric_2d" in payload["artifacts"]["plot_paths"]
    assert "mission_timeline" in payload["artifacts"]["plot_paths"]
    assert (out.parent / "plots" / "ric_2d_plots.png").exists()
    assert (out.parent / "plots" / "mission_timeline.png").exists()
    assert "Pass/Failure" in out.read_text(encoding="utf-8")
    assert "# Unit Debrief" in out.read_text(encoding="utf-8")
    assert "## Event Timeline" not in out.read_text(encoding="utf-8")
    assert "![Mission Timeline](plots/mission_timeline.png)" in out.read_text(encoding="utf-8")
    assert "![2D RIC Plots](plots/ric_2d_plots.png)" in out.read_text(encoding="utf-8")
    assert payload["replay"]["time_s"] == [0.0]
    assert payload["replay"]["relative_ric"][0][:3] == pytest.approx([0.0, -0.2, 0.0])


def test_game_debrief_header_uses_rpo_trainer_level_name(tmp_path: Path) -> None:
    sim_cfg = SimulationConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_sun_angle_inspection.yaml"
    )
    cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
    score = type(
        "Score",
        (),
        {
            "scenario_id": cfg.scenario_id,
            "learning_goal": cfg.learning_goal,
            "samples": 1,
            "elapsed_s": 0.0,
            "closest_approach_km": 1.0,
            "final_range_km": 1.0,
            "final_goal_error_km": 1.0,
            "final_relative_speed_km_s": 0.0,
            "time_inside_keepout_s": 0.0,
            "approximate_delta_v_m_s": 0.0,
            "level_passed": True,
            "level_failed": False,
            "pass_fail_reasons": ("All pass criteria satisfied.",),
        },
    )()
    out = write_game_debrief(
        game_debrief_path(
            scenario_id=cfg.scenario_id,
            difficulty="easy",
            attempt_index=1,
            output_dir=tmp_path,
            timestamp=datetime(2026, 5, 22, 12, 0, 0),
        ),
        config=cfg,
        score=score,
        difficulty="easy",
        objective_checklist=(),
        replay_history={
            "time_s": [0.0],
            "relative_ric": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "chaser_thrust_ric_km_s2": [[0.0, 0.0, 0.0]],
        },
    )
    payload = yaml.safe_load((out.parent / "summary.json").read_text(encoding="utf-8"))

    assert cfg.level_name == "Level 6 - Sun-Angle Inspection"
    assert payload["display_title"] == "RPO Trainer Level 6 - Sun-Angle Inspection Debrief"
    assert out.read_text(encoding="utf-8").startswith("# RPO Trainer Level 6 - Sun-Angle Inspection Debrief\n")


def test_tracker_replay_history_uses_array_backed_replay_stream() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-replay",
        learning_goal="test",
        goal_range_km=0.25,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.1, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 12.0,
                "truth": {"target": target_state, "chaser": chaser_state},
                "applied_thrust": {
                    "chaser": np.array([0.0, 1.0e-6, 0.0], dtype=float),
                    "target": np.array([0.0, 0.0, 2.0e-6], dtype=float),
                },
            },
        )()
    )

    expected = tracker_replay_history(tracker)
    tracker.t_s.clear()
    tracker.rel_ric_hist.clear()
    tracker.thrust_hist.clear()
    tracker.thrust_ric_hist.clear()
    tracker.target_thrust_hist.clear()
    replay = tracker_replay_history(tracker)

    assert replay["time_s"] == expected["time_s"] == [12.0]
    np.testing.assert_allclose(replay["relative_ric"], expected["relative_ric"])
    np.testing.assert_allclose(replay["chaser_thrust_ric_km_s2"], expected["chaser_thrust_ric_km_s2"])
    np.testing.assert_allclose(replay["target_thrust_eci_km_s2"], [[0.0, 0.0, 2.0e-6]])


def test_aerodynamic_controls_are_preserved_in_replay_and_debrief_metrics() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="aerodynamic-debrief",
        learning_goal="test",
        goal_range_km=0.25,
    )
    tracker = RPOTrainingTracker(cfg)
    provider = type("AerodynamicProvider", (), {})()
    provider.control_mode = "aerodynamic"
    provider.command_state = KeyboardCommandState(pitch=1.0, roll=-1.0)
    provider.aerodynamic_drag_coefficient = 2.0
    provider.aerodynamic_lift_coefficient = 0.5
    provider.aerodynamic_lift_area_m2 = 25.0
    provider.ballistic_coefficient_kg_m2 = 100.0
    provider.lift_bank_angle_deg = 10.0

    target_state6 = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state6 = ric_rect_state_to_eci(rel_ric, target_state6[:3], target_state6[3:])
    state_tail = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5000.0], dtype=float)
    target_state = np.hstack((target_state6, state_tail))
    chaser_state = np.hstack((chaser_state6, state_tail))

    for time_s, bc, bank in ((0.0, 100.0, 10.0), (2.0, 120.0, 30.0)):
        provider.ballistic_coefficient_kg_m2 = bc
        provider.lift_bank_angle_deg = bank
        snapshot = SimulationSnapshot(
            step_index=int(time_s / 2.0),
            time_s=time_s,
            truth={"target": target_state, "chaser": chaser_state},
            belief={},
            applied_thrust={"target": np.zeros(3), "chaser": np.zeros(3)},
            applied_torque={},
        )
        tracker.record(snapshot, control_telemetry_provider=provider)

    replay = tracker_replay_history(tracker)
    payload = game_debrief.game_debrief_payload(
        config=cfg,
        score=tracker.score(),
        difficulty="easy",
        replay_history=replay,
    )

    assert replay["aerodynamic_ballistic_coefficient_kg_m2"] == pytest.approx([100.0, 120.0])
    assert replay["aerodynamic_drag_area_m2"] == pytest.approx([25.0, 5000.0 / 240.0])
    assert replay["aerodynamic_lift_coefficient"] == pytest.approx([0.5, 0.5])
    assert replay["aerodynamic_lift_area_m2"] == pytest.approx([25.0, 25.0])
    assert replay["aerodynamic_lift_bank_angle_deg"] == pytest.approx([10.0, 30.0])
    assert payload["metrics"]["burn_count"] == 0
    assert payload["metrics"]["active_control_time_s"] == pytest.approx(2.0)
    assert payload["metrics"]["aerodynamic_control_time_s"] == pytest.approx(2.0)
    assert payload["metrics"]["aerodynamic_adjustment_count"] == 1
    assert payload["metrics"]["final_ballistic_coefficient_kg_m2"] == pytest.approx(120.0)
    assert payload["metrics"]["final_drag_area_m2"] == pytest.approx(5000.0 / 240.0)
    assert payload["metrics"]["final_lift_bank_angle_deg"] == pytest.approx(30.0)
    assert any(event["label"] == "Aerodynamic control input" for event in payload["event_timeline"])


def test_game_debrief_attempt_index_counts_level_folders(tmp_path: Path) -> None:
    first = game_debrief_path(
        scenario_id="Unit Debrief",
        difficulty="easy",
        attempt_index=1,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 14, 12, 30, 45),
    )
    first.parent.mkdir(parents=True)

    assert next_game_debrief_attempt_index(scenario_id="Unit Debrief", output_dir=tmp_path) == 2


def test_game_debrief_is_disabled_for_sandbox_and_arcade_modes() -> None:
    sandbox_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    sandbox_config = SimulationConfig.from_yaml(sandbox_path)
    sandbox_training = RPOTrainingConfig.from_metadata(dict(sandbox_config.scenario.metadata or {}))

    arcade_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    arcade_config = SimulationConfig.from_yaml(arcade_path)
    arcade_training = RPOTrainingConfig.from_metadata(dict(arcade_config.scenario.metadata or {}))

    normal_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_01_coast_relative_motion.yaml"
    )
    normal_config = SimulationConfig.from_yaml(normal_path)
    normal_cfg = RPOTrainingConfig.from_metadata(dict(normal_config.scenario.metadata or {}))

    assert game_runner._game_debrief_enabled(sandbox_config, sandbox_training, arcade_enabled=False) is False
    assert game_runner._game_debrief_enabled(arcade_config, arcade_training, arcade_enabled=True) is False
    assert game_runner._game_debrief_enabled(normal_config, normal_cfg, arcade_enabled=False) is True


@pytest.mark.parametrize(
    "config_name,expected_enabled",
    [
        ("game_training_rpo_00_tutorial.yaml", True),
        ("game_training_rpo_01_coast_relative_motion.yaml", True),
        ("game_training_rpo_02_vbar_approach.yaml", True),
        ("game_training_rpo_03_rbar_approach.yaml", True),
        ("game_training_rpo_04_rendezvous.yaml", True),
        ("game_training_rpo_05_passive_cross_track_approach.yaml", True),
        ("game_training_rpo_06_sun_angle_inspection.yaml", True),
        ("game_training_rpo_07_elliptic_burn_then_approach.yaml", True),
        ("game_training_rpo_08_elliptic_nmc.yaml", True),
        ("game_training_rpo_09_elliptic_rendezvous.yaml", True),
        ("game_training_rpo_10_defensive_target_demo.yaml", True),
        ("game_training_rpo_11_evasive_target_survival.yaml", True),
        ("game_training_rpo_arcade_pursuit.yaml", False),
        ("game_training_rpo_sandbox.yaml", False),
    ],
)
def test_game_debrief_policy_and_writer_across_levels(
    config_name: str,
    expected_enabled: bool,
    tmp_path: Path,
) -> None:
    config = SimulationConfig.from_yaml(Path(__file__).resolve().parents[1] / "game" / "configs" / config_name)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    enabled = game_runner._game_debrief_enabled(
        config,
        training_cfg,
        arcade_enabled=game_arcade._game_arcade_enabled(config),
    )

    assert enabled is expected_enabled
    if not enabled:
        return

    score = RPOTrainingScore(
        scenario_id=training_cfg.scenario_id,
        learning_goal=training_cfg.learning_goal,
        samples=3,
        elapsed_s=2.0,
        closest_approach_km=0.25,
        final_range_km=0.3,
        final_goal_error_km=0.05,
        final_relative_speed_km_s=0.0001,
        time_inside_keepout_s=0.0,
        approximate_delta_v_m_s=0.04,
        target_delta_v_m_s=0.0,
        burn_axes_satisfied=tuple(training_cfg.required_burn_axes),
        phase_burns_satisfied=tuple(burn.name for burn in training_cfg.required_phase_burns),
        speed_multiplier_changed=bool(training_cfg.require_speed_multiplier_change),
        coast_after_burn_satisfied=training_cfg.required_coast_after_burn_s is None,
        coast_after_burn_s=0.0,
        guided_tutorial_burns_satisfied=tuple(burn.name for burn in training_cfg.guided_tutorial_burns),
        guided_tutorial_burns_total=len(training_cfg.guided_tutorial_burns),
        guided_tutorial_speed_satisfied=True,
        guided_tutorial_speed_target=(
            None
            if training_cfg.guided_tutorial_speed_step is None
            else training_cfg.guided_tutorial_speed_step.target_speed_multiplier
        ),
        achieved_time_s=2.0,
        min_goal_error_km=0.04,
        final_nmt_radial_amplitude_km=training_cfg.goal_nmt_radial_amplitude_km or 0.0,
        final_nmt_cross_track_amplitude_km=training_cfg.goal_nmt_cross_track_amplitude_km,
        final_nmt_radial_amplitude_error_km=0.0,
        final_nmt_cross_track_amplitude_error_km=0.0,
        final_nmt_drift_velocity_error_km_s=0.0,
        goal_met=True,
        level_passed=True,
        level_failed=False,
        pass_fail_reasons=("All pass criteria satisfied.",),
        keepout_violation=False,
        hard_speed_limit_violation=False,
        forbidden_region_violation=False,
        forbidden_region_names=(),
        approach_gate_violation=False,
        approach_gate_names=(),
        approach_gates_satisfied=len(training_cfg.approach_gates),
        approach_gates_total=len(training_cfg.approach_gates),
        inspection_gates_satisfied=len(training_cfg.inspection_gates),
        inspection_gates_total=len(training_cfg.inspection_gates),
        inspection_gate_names=tuple(gate.name for gate in training_cfg.inspection_gates),
        hints=(),
    )
    replay = {
        "time_s": [0.0, 1.0, 2.0],
        "relative_ric": [
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.7, 0.05, 0.0001, 0.0, 0.0],
            [0.0, -0.3, 0.0, 0.0, 0.0001, 0.0],
        ],
        "chaser_thrust_ric_km_s2": [
            [0.0, 0.0, 0.0],
            [0.00001, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "target_thrust_eci_km_s2": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    }
    out = write_game_debrief(
        game_debrief_path(
            scenario_id=training_cfg.scenario_id,
            difficulty="easy",
            attempt_index=1,
            output_dir=tmp_path,
            timestamp=datetime(2026, 5, 22, 12, 0, 0),
        ),
        config=training_cfg,
        score=score,
        difficulty="easy",
        objective_checklist=game_runner._mission_checklist(training_cfg, score),
        replay_history=replay,
    )

    assert out.exists()
    assert (out.parent / "summary.json").exists()
    assert (out.parent / "plots" / "mission_timeline.png").exists()
    assert (out.parent / "plots" / "ric_2d_plots.png").exists()
    assert "## Stats Summary" in out.read_text(encoding="utf-8")


def test_tutorial_debrief_history_can_scope_to_final_free_maneuver_phase() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="rpo_00_tutorial",
        guided_tutorial_burns=(
            GuidedTutorialBurnConfig(
                name="plus_in_track",
                axis="in_track",
                sign=1,
                delta_v_m_s=0.25,
            ),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    guided_rel = np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=float)
    final_rel = np.array([0.0, -0.25, 0.0, 0.0, 0.0, 0.0], dtype=float)

    for time_s, rel_ric in ((1.0, guided_rel),):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {
                        "target": target_state,
                        "chaser": ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:]),
                    },
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )
    tracker.mark_guided_tutorial_burn_complete("plus_in_track")
    tracker.clear(reset_guided_tutorial_progress=False)
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 99.0,
                "truth": {
                    "target": target_state,
                    "chaser": ric_rect_state_to_eci(final_rel, target_state[:3], target_state[3:]),
                },
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    replay = tracker_replay_history(tracker)

    assert tracker.guided_tutorial_burns_satisfied() == ("plus_in_track",)
    assert replay["time_s"] == [99.0]
    assert replay["relative_ric"][0][:3] == pytest.approx([0.0, -0.25, 0.0])


def test_game_debrief_ric_plot_axes_keep_radial_vertical() -> None:
    assert game_debrief._plane_axes("RI") == (1, 0, "I", "R")
    assert game_debrief._plane_axes("RC") == (2, 0, "C", "R")
    assert game_debrief._plane_axes("IC") == (1, 2, "I", "C")


def test_game_debrief_cumulative_delta_v_matches_sampled_accel_integral() -> None:
    thrust_km_s2 = np.array(
        [
            [0.001, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [np.nan, 0.0, 0.0],
            [0.003, 0.0, 0.0],
        ],
        dtype=float,
    )
    t_s = np.array([0.0, 2.0, 5.0, 8.0], dtype=float)

    cumulative = game_debrief._cumulative_delta_v_m_s(thrust_km_s2, t_s)

    assert cumulative == pytest.approx([0.0, 4.0, 4.0, 13.0])


def test_training_tracker_cached_delta_v_matches_batch_integral_exactly() -> None:
    tracker = RPOTrainingTracker(RPOTrainingConfig(enabled=True, scenario_id="cached-delta-v"))
    target = np.array(
        [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
        dtype=float,
    )
    chaser = target.copy()
    chaser[1] = -1.0
    time_s = np.array([0.0, 2.0, 5.0, 5.0, 8.0], dtype=float)
    thrust = np.array(
        [
            [0.001, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [np.nan, 0.0, 0.0],
            [0.003, 0.0, 0.0],
            [0.001, 0.002, 0.003],
        ],
        dtype=float,
    )
    target_thrust = thrust[:, ::-1].copy()

    for idx, sample_time_s in enumerate(time_s):
        tracker.record(
            type(
                "Snapshot",
                (),
                {
                    "time_s": float(sample_time_s),
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {
                        "chaser": thrust[idx],
                        "target": target_thrust[idx],
                    },
                },
            )()
        )

    score = tracker.score()

    assert score.approximate_delta_v_m_s == game_training._integrated_delta_v_m_s(thrust, time_s)
    assert score.target_delta_v_m_s == game_training._integrated_delta_v_m_s(target_thrust, time_s)


def test_training_tracker_cached_geometry_matches_batch_score_exactly() -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_04_rendezvous.yaml")
    training = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    session = SimulationSession.from_config(game_session._attempt_config_for_training_clock(config, training))
    snapshot = session.reset()
    tracker = RPOTrainingTracker(training)

    for index in range(8):
        if index:
            snapshot = session.step()
        tracker.record(snapshot)

    count = tracker._history_count
    rel = tracker._rel_array[:count]
    ranges = np.linalg.norm(rel[:, :3], axis=1)
    speeds = np.linalg.norm(rel[:, 3:6], axis=1)
    goal_error = np.linalg.norm(rel[:, :3] - training.goal_relative_ric_km.reshape(1, 3), axis=1)
    score = tracker.score()

    assert np.array_equal(tracker._range_array[:count], ranges)
    assert np.array_equal(tracker._speed_array[:count], speeds)
    assert np.array_equal(tracker._goal_error_array[:count], goal_error)
    assert score.closest_approach_km == float(np.min(ranges))
    assert score.final_range_km == float(ranges[-1])
    assert score.final_relative_speed_km_s == float(speeds[-1])
    assert score.final_goal_error_km == float(goal_error[-1])
    assert score.min_goal_error_km == float(np.min(goal_error))


def test_training_tracker_charges_terminal_step_burn_before_passing() -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_04_rendezvous.yaml")
    base_training = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    command_state = KeyboardCommandState(pitch=1.0)
    session, _, initial_snapshot = game_runner._start_game_attempt(
        config,
        command_state=command_state,
        training_cfg=base_training,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )
    terminal_snapshot = session.step(dt_s=1.0)
    initial_rel = relative_state_from_arrays(
        initial_snapshot.truth["target"],
        initial_snapshot.truth["chaser"],
    )
    terminal_rel = relative_state_from_arrays(
        terminal_snapshot.truth["target"],
        terminal_snapshot.truth["chaser"],
    )
    tracker = RPOTrainingTracker(
        RPOTrainingConfig(
            enabled=True,
            scenario_id="terminal-burn-budget",
            goal_relative_ric_km=terminal_rel[:3],
            goal_radius_km=1.0e-9,
            max_goal_speed_km_s=1.0,
            max_delta_v_m_s=0.005,
        )
    )

    tracker.record(initial_snapshot)
    tracker.record(terminal_snapshot)
    score = tracker.score()

    assert np.linalg.norm(initial_rel[:3] - terminal_rel[:3]) > 1.0e-9
    assert score.approximate_delta_v_m_s == pytest.approx(0.01)
    assert score.level_passed is False
    assert score.level_failed is True


def test_game_debrief_timeline_uses_burn_intervals() -> None:
    cfg = RPOTrainingConfig(enabled=True, scenario_id="timeline-unit")
    score = RPOTrainingScore(
        scenario_id=cfg.scenario_id,
        learning_goal="",
        samples=5,
        elapsed_s=4.0,
        closest_approach_km=0.2,
        final_range_km=0.3,
        final_goal_error_km=0.0,
        final_relative_speed_km_s=0.0,
        time_inside_keepout_s=0.0,
        approximate_delta_v_m_s=0.0,
        target_delta_v_m_s=0.0,
        burn_axes_satisfied=(),
        phase_burns_satisfied=(),
        speed_multiplier_changed=False,
        coast_after_burn_satisfied=True,
        coast_after_burn_s=0.0,
        guided_tutorial_burns_satisfied=(),
        guided_tutorial_burns_total=0,
        guided_tutorial_speed_satisfied=True,
        guided_tutorial_speed_target=None,
        achieved_time_s=4.0,
        min_goal_error_km=0.0,
        final_nmt_radial_amplitude_km=0.0,
        final_nmt_cross_track_amplitude_km=0.0,
        final_nmt_radial_amplitude_error_km=0.0,
        final_nmt_cross_track_amplitude_error_km=0.0,
        final_nmt_drift_velocity_error_km_s=0.0,
        goal_met=True,
        level_passed=True,
        level_failed=False,
        pass_fail_reasons=("All pass criteria satisfied.",),
        keepout_violation=False,
        hard_speed_limit_violation=False,
        forbidden_region_violation=False,
        forbidden_region_names=(),
        approach_gate_violation=False,
        approach_gate_names=(),
        approach_gates_satisfied=0,
        approach_gates_total=0,
        inspection_gates_satisfied=0,
        inspection_gates_total=0,
        inspection_gate_names=(),
        hints=(),
    )
    replay = {
        "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
        "relative_ric": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 5,
        "chaser_thrust_ric_km_s2": [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
        ],
    }

    events = game_debrief._event_timeline(config=cfg, score=score, replay_history=replay)
    burn_events = [event for event in events if event.get("kind") == "interval"]

    assert game_debrief._active_segments(np.array([False, True, True, False, True])) == [(1, 2), (4, 4)]
    assert burn_events[0]["start_time_s"] == pytest.approx(0.0)
    assert burn_events[0]["end_time_s"] == pytest.approx(2.0)
    assert burn_events[0]["label"] == "Control input"
    assert burn_events[1]["start_time_s"] == pytest.approx(3.0)
    assert burn_events[1]["end_time_s"] == pytest.approx(4.0)


def test_open_game_debrief_folder_opens_report_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []
    report = tmp_path / "attempt_001" / "report.md"
    report.parent.mkdir()
    report.write_text("# Debrief\n", encoding="utf-8")

    monkeypatch.setattr("sim.game.debrief.open_folder", lambda path: calls.append(Path(path)))

    assert open_game_debrief_folder(report) is True
    assert calls == [report.parent]


def test_game_frame_recorder_finishes_or_discards_with_fake_writer(tmp_path: Path) -> None:
    class FakeWriter:
        def __init__(self, path: Path):
            self.path = path
            self.frames: list[np.ndarray] = []
            self.closed = False

        def append_data(self, frame: np.ndarray) -> None:
            self.frames.append(np.array(frame, dtype=np.uint8))

        def close(self) -> None:
            self.closed = True
            self.path.write_bytes(b"fake-mp4")

    writers: list[FakeWriter] = []

    def factory(path: Path, fps: float) -> FakeWriter:
        assert fps == 12.0
        writer = FakeWriter(path)
        writers.append(writer)
        return writer

    path = tmp_path / "attempt.mp4"
    recorder = GameFrameRecorder.start(path, fps=12.0, writer_factory=factory)
    recorder.capture_frame(np.zeros((2, 3, 3), dtype=np.uint8))

    assert recorder.finish() == path
    assert recorder.saved is True
    assert recorder.frames_written == 1
    assert writers[-1].closed is True
    assert path.exists()

    recorder = GameFrameRecorder.start(path, fps=12.0, writer_factory=factory)
    recorder.capture_frame(np.zeros((2, 3, 4), dtype=np.uint8))
    recorder.discard()

    assert recorder.saved is False
    assert writers[-1].closed is True
    assert not path.exists()


def test_add_looped_audio_to_video_muxes_audio_with_ffmpeg(tmp_path: Path) -> None:
    video = tmp_path / "attempt.mp4"
    audio = tmp_path / "level.wav"
    video.write_bytes(b"silent-video")
    audio.write_bytes(b"level-audio")
    captured: dict[str, list[str]] = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = [str(part) for part in cmd]
        assert kwargs["check"] is True
        Path(cmd[-1]).write_bytes(b"muxed-video")

    out = add_looped_audio_to_video(video, audio, ffmpeg_exe="/usr/local/bin/ffmpeg", runner=fake_runner)

    assert out == video
    assert video.read_bytes() == b"muxed-video"
    cmd = captured["cmd"]
    assert cmd[:6] == ["/usr/local/bin/ffmpeg", "-y", "-i", str(video), "-stream_loop", "-1"]
    assert cmd[6:8] == ["-i", str(audio)]
    assert "-shortest" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_add_level_music_to_recording_uses_training_level_track(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "attempt.mp4"
    video.write_bytes(b"silent-video")
    calls: list[tuple[Path, Path]] = []

    def fake_add_audio(recording_path: Path, music_path: Path) -> Path:
        calls.append((recording_path, music_path))
        return recording_path

    monkeypatch.setattr(game_recording_controller, "add_looped_audio_to_video", fake_add_audio)
    cfg = RPOTrainingConfig(enabled=True, scenario_id="rpo_00_tutorial")

    assert game_runner._add_level_music_to_recording(video, cfg) == video
    assert calls == [(video, LEVEL_MUSIC_PATHS["rpo_00_tutorial"])]


def test_add_level_music_to_recording_prefers_arcade_override(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "attempt.mp4"
    video.write_bytes(b"silent-video")
    calls: list[tuple[Path, Path]] = []

    def fake_add_audio(recording_path: Path, music_path: Path) -> Path:
        calls.append((recording_path, music_path))
        return recording_path

    monkeypatch.setattr(game_recording_controller, "add_looped_audio_to_video", fake_add_audio)
    cfg = RPOTrainingConfig(enabled=True, scenario_id="rpo_arcade_pursuit")
    boss_track = game_runner.GAME_MUSIC_DIR / "28_high_shred_boss_riff.wav"

    assert game_runner._add_level_music_to_recording(video, cfg, override_level_path=boss_track) == video
    assert calls == [(video, boss_track)]


def test_game_recording_defaults_to_recording_fps() -> None:
    assert game_runner.run_game_mode.__kwdefaults__["recording_fps"] == game_runner.GAME_RECORDING_FPS


def test_start_game_recorder_disables_when_writer_start_fails(tmp_path: Path, monkeypatch) -> None:
    config = SimulationConfig.from_dict(_game_config(tmp_path))
    monkeypatch.setattr(
        game_recording_controller.GameFrameRecorder,
        "start",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer unavailable")),
    )

    recorder = game_runner._start_game_recorder(
        enabled=True,
        config=config,
        difficulty="easy",
        attempt_index=1,
        output_dir=tmp_path,
        fps=30.0,
    )

    assert recorder is None


def test_start_game_clip_recorder_uses_clip_path(tmp_path: Path, monkeypatch) -> None:
    config = SimulationConfig.from_dict(_game_config(tmp_path))
    starts: list[Path] = []

    def fake_start(path, **kwargs):
        starts.append(Path(path))
        return "recorder"

    monkeypatch.setattr(game_recording_controller.GameFrameRecorder, "start", fake_start)

    recorder = game_runner._start_game_clip_recorder(
        enabled=True,
        config=config,
        difficulty="easy",
        clip_index=4,
        output_dir=tmp_path,
        fps=30.0,
    )

    assert recorder == "recorder"
    assert starts[0].parent == tmp_path / "clips"
    assert starts[0].name.startswith(f"{config.scenario.scenario_name}_easy_")
    assert starts[0].name.endswith("_clip04.mp4")


def test_recording_hold_frame_count_uses_duration_and_fps() -> None:
    assert game_recording_controller.recording_hold_frame_count(duration_s=3.0, fps=30.0) == 90
    assert game_recording_controller.recording_hold_frame_count(duration_s=2.5, fps=24.0) == 60
    assert game_recording_controller.recording_hold_frame_count(duration_s=-1.0, fps=30.0) == 0


def test_next_available_recording_path_avoids_existing_clip(tmp_path: Path) -> None:
    base = tmp_path / "clip.mp4"
    base.write_bytes(b"existing")

    assert game_recording_controller.next_available_recording_path(base) == tmp_path / "clip_02.mp4"

    (tmp_path / "clip_02.mp4").write_bytes(b"existing")

    assert game_recording_controller.next_available_recording_path(base) == tmp_path / "clip_03.mp4"


def test_start_game_recorder_avoids_existing_full_attempt_path(tmp_path: Path, monkeypatch) -> None:
    config = SimulationConfig.from_dict(_game_config(tmp_path))
    base = tmp_path / "attempt.mp4"
    base.write_bytes(b"existing")
    starts: list[Path] = []
    monkeypatch.setattr(game_recording_controller, "game_recording_path", lambda **_kwargs: base)
    monkeypatch.setattr(
        game_recording_controller.GameFrameRecorder,
        "start",
        lambda path, **_kwargs: starts.append(Path(path)) or "recorder",
    )

    recorder = game_runner._start_game_recorder(
        enabled=True,
        config=config,
        difficulty="easy",
        attempt_index=1,
        output_dir=tmp_path,
        fps=30.0,
    )

    assert recorder == "recorder"
    assert starts == [tmp_path / "attempt_02.mp4"]


def test_recording_controller_capture_hold_repeats_current_frame(tmp_path: Path, monkeypatch) -> None:
    config = SimulationConfig.from_dict(_game_config(tmp_path))
    captures: list[object] = []

    class FakeRecorder:
        saved = False

    def fake_capture(recorder, dashboard):
        captures.append(dashboard)
        return recorder

    monkeypatch.setattr(game_recording_controller, "safe_capture_recording_frame", fake_capture)
    controller = game_recording_controller.GameRecordingController(
        enabled=True,
        config=config,
        difficulty="easy",
        fps=10.0,
        recorder=FakeRecorder(),
    )
    dashboard = object()

    assert controller.capture_hold(dashboard, duration_s=3.0) is controller.recorder
    assert captures == [dashboard] * 30


def test_safe_capture_recording_frame_discards_and_disables_on_capture_failure(monkeypatch) -> None:
    class FakeRecorder:
        saved = False

        def __init__(self) -> None:
            self.discarded = False

        def discard(self) -> None:
            self.discarded = True

    recorder = FakeRecorder()
    monkeypatch.setattr(
        game_recording_controller,
        "capture_recording_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    returned = game_runner._safe_capture_recording_frame(recorder, type("Dashboard", (), {"screen": object()})())

    assert returned is None
    assert recorder.discarded is True


def test_finish_game_recording_discards_and_disables_on_finalize_failure() -> None:
    class FakeRecorder:
        def __init__(self) -> None:
            self.discarded = False

        def finish(self) -> Path:
            raise RuntimeError("encode failed")

        def discard(self) -> None:
            self.discarded = True

    recorder = FakeRecorder()

    returned = game_runner._finish_game_recording(recorder, RPOTrainingConfig(enabled=True))

    assert returned is None
    assert recorder.discarded is True


def test_recording_controller_restart_does_not_delete_saved_recording(tmp_path: Path, monkeypatch) -> None:
    class FakeRecorder:
        saved = True

        def __init__(self) -> None:
            self.discarded = False

        def discard(self) -> None:
            self.discarded = True

    saved_recorder = FakeRecorder()
    starts: list[int] = []

    def fake_start(**kwargs):
        starts.append(int(kwargs["attempt_index"]))
        return None

    monkeypatch.setattr(game_recording_controller, "start_game_recorder", fake_start)
    controller = game_recording_controller.GameRecordingController(
        enabled=True,
        config=SimulationConfig.from_dict(_game_config(tmp_path)),
        difficulty="easy",
        output_dir=tmp_path,
        recorder=saved_recorder,
    )

    controller.restart()

    assert saved_recorder.discarded is False
    assert starts == [2]
    assert controller.attempt_index == 2
