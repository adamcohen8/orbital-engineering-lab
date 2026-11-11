# ruff: noqa: F401,F403,F405,I001
from sim.observation_normalization import *
from sim.observation_ingestion import *
from sim.observation_fitting import *
from sim.observation_tle import *

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate preliminary OEL states from structured observation packets.")
    sub = parser.add_subparsers(dest="command", required=True)

    sample_sgp4 = sub.add_parser("sample-sgp4", help="Sample a TLE with OEL SGP4 into an observation packet.")
    _add_sgp4_sampling_args(sample_sgp4)
    sample_sgp4.add_argument("--out", required=True, help="Output observation packet JSON.")

    sgp4_screen = sub.add_parser(
        "tle-maneuver-screen",
        help="Sample a TLE with SGP4, then run the dynamics OD maneuver screen.",
    )
    _add_sgp4_sampling_args(sgp4_screen)
    sgp4_screen.add_argument("--output-dir", required=True, help="Directory for the generated packet and OD reports.")
    sgp4_screen.add_argument("--packet-out", help="Optional generated observation packet path.")
    sgp4_screen.add_argument("--role")
    sgp4_screen.add_argument("--scenario-name", default="sgp4_tle_maneuver_screen")
    sgp4_screen.add_argument("--fit-duration-s", type=float)
    sgp4_screen.add_argument("--holdout-duration-s", type=float)
    sgp4_screen.add_argument("--estimate", default="state")
    sgp4_screen.add_argument("--dynamics-model", default="two_body")
    sgp4_screen.add_argument("--j2", action="store_true")
    sgp4_screen.add_argument("--j3", action="store_true")
    sgp4_screen.add_argument("--j4", action="store_true")
    sgp4_screen.add_argument("--drag", action="store_true")
    sgp4_screen.add_argument("--srp", action="store_true")
    sgp4_screen.add_argument("--third-body-sun", action="store_true")
    sgp4_screen.add_argument("--third-body-moon", action="store_true")
    sgp4_screen.add_argument("--atmosphere-model", default="")
    sgp4_screen.add_argument("--orbit-force-model-json")
    sgp4_screen.add_argument("--area-m2", type=float)
    sgp4_screen.add_argument("--drag-area-m2", type=float)
    sgp4_screen.add_argument("--srp-area-m2", type=float)
    sgp4_screen.add_argument("--solar-area-m2", type=float)
    sgp4_screen.add_argument("--cd", type=float)
    sgp4_screen.add_argument("--cr", type=float)
    sgp4_screen.add_argument("--max-nfev", type=int, default=24)
    _add_robust_batch_arguments(sgp4_screen)
    sgp4_screen.add_argument("--maneuver-frame", choices=("ric", "eci"), default="ric")
    sgp4_screen.add_argument("--max-maneuver-dv-m-s", type=float, default=25.0)
    sgp4_screen.add_argument("--maneuver-min-delta-v-m-s", type=float, default=0.05)
    sgp4_screen.add_argument("--maneuver-min-improvement-ratio", type=float, default=0.35)
    sgp4_screen.add_argument("--maneuver-burn-duration-s", type=float)
    sgp4_screen.add_argument("--maneuver-guard-observations", type=int, default=3)
    sgp4_screen.add_argument("--maneuver-max-candidates", type=int, default=12)
    sgp4_screen.add_argument("--maneuver-max-nfev", type=int)

    inspect_parser = sub.add_parser("inspect", help="Inspect an observation packet and estimator readiness.")
    inspect_parser.add_argument("packet")
    inspect_parser.add_argument("--json", action="store_true")

    fit = sub.add_parser("fit-position", help="Estimate an initial state with a constant-velocity batch fit.")
    fit.add_argument("--packet", required=True)
    fit.add_argument("--out", required=True, help="Output mission input packet JSON.")
    fit.add_argument("--estimate-json", help="Optional full estimate diagnostics JSON.")
    fit.add_argument("--object-id")
    fit.add_argument("--role")
    fit.add_argument("--epoch", default="first", choices=("first", "last", "mean"))

    kf = sub.add_parser("kalman-position", help="Estimate state with a constant-velocity Kalman position filter.")
    kf.add_argument("--packet", required=True)
    kf.add_argument("--out", required=True, help="Output mission input packet JSON.")
    kf.add_argument("--estimate-json", help="Optional full filter diagnostics JSON.")
    kf.add_argument("--object-id")
    kf.add_argument("--role")
    kf.add_argument("--process-accel-sigma-km-s2", type=float, default=1.0e-7)
    kf.add_argument("--initial-position-sigma-km", type=float, default=1.0)
    kf.add_argument("--initial-velocity-sigma-km-s", type=float, default=1.0e-3)
    kf.add_argument("--default-position-sigma-km", type=float, default=1.0)

    fit_orbit = sub.add_parser("fit-orbit", help="Fit an initial state with OEL orbital dynamics in the residual loop.")
    fit_orbit.add_argument("--packet", required=True)
    fit_orbit.add_argument("--output-dir", required=True, help="Directory for OD reports and artifacts.")
    fit_orbit.add_argument("--state-packet-out", help="Optional path for fitted mission input packet JSON.")
    fit_orbit.add_argument("--object-id")
    fit_orbit.add_argument("--role")
    fit_orbit.add_argument("--scenario-name", default="dynamics_orbit_determination")
    fit_orbit.add_argument("--fit-duration-s", type=float)
    fit_orbit.add_argument("--holdout-duration-s", type=float)
    fit_orbit.add_argument("--dt-s", type=float)
    fit_orbit.add_argument("--estimate", default="state")
    fit_orbit.add_argument("--dynamics-model", default="two_body")
    fit_orbit.add_argument("--j2", action="store_true")
    fit_orbit.add_argument("--j3", action="store_true")
    fit_orbit.add_argument("--j4", action="store_true")
    fit_orbit.add_argument("--drag", action="store_true")
    fit_orbit.add_argument("--srp", action="store_true")
    fit_orbit.add_argument("--third-body-sun", action="store_true")
    fit_orbit.add_argument("--third-body-moon", action="store_true")
    fit_orbit.add_argument("--atmosphere-model", default="")
    fit_orbit.add_argument("--spherical-harmonics-degree", type=int)
    fit_orbit.add_argument("--spherical-harmonics-order", type=int)
    fit_orbit.add_argument("--spherical-harmonics-source", default="")
    fit_orbit.add_argument("--spherical-harmonics-coeff-path")
    fit_orbit.add_argument("--spherical-harmonics-frame-model")
    fit_orbit.add_argument("--spherical-harmonics-eop-path")
    fit_orbit.add_argument(
        "--orbit-force-model-json",
        help="JSON object merged into simulator.dynamics.orbit for OD candidate propagations.",
    )
    fit_orbit.add_argument("--mass-kg", type=float)
    fit_orbit.add_argument("--area-m2", type=float)
    fit_orbit.add_argument("--drag-area-m2", type=float)
    fit_orbit.add_argument("--srp-area-m2", type=float)
    fit_orbit.add_argument("--solar-area-m2", type=float)
    fit_orbit.add_argument("--geometry-profile-path")
    fit_orbit.add_argument("--cd", type=float)
    fit_orbit.add_argument("--cr", type=float)
    fit_orbit.add_argument(
        "--attitude-source",
        default="none",
        choices=(
            "none",
            "observed-history",
            "observed_history",
            "modeled-inline",
            "modeled_inline",
            "modeled-replay",
            "modeled_replay",
        ),
        help="Attitude source for attitude-aware force estimation.",
    )
    fit_orbit.add_argument("--attitude-mode", default="sun_track")
    fit_orbit.add_argument("--sun-track-body-axis", nargs=3, type=float, metavar=("BX", "BY", "BZ"))
    fit_orbit.add_argument(
        "--attitude-controller",
        default="surrogate_snap",
        choices=("surrogate_snap", "reaction_wheel_pd"),
        help="Controller used when attitude-source is modeled-inline or modeled-replay.",
    )
    fit_orbit.add_argument("--max-nfev", type=int, default=24)
    _add_robust_batch_arguments(fit_orbit)
    fit_orbit.add_argument(
        "--detect-maneuvers", action="store_true", help="Run conservative single-maneuver OD model comparison."
    )
    fit_orbit.add_argument("--maneuver-frame", choices=("ric", "eci"), default="ric")
    fit_orbit.add_argument("--max-maneuver-dv-m-s", type=float, default=25.0)
    fit_orbit.add_argument("--maneuver-min-delta-v-m-s", type=float, default=0.05)
    fit_orbit.add_argument("--maneuver-min-improvement-ratio", type=float, default=0.35)
    fit_orbit.add_argument("--maneuver-burn-duration-s", type=float)
    fit_orbit.add_argument("--maneuver-guard-observations", type=int, default=3)
    fit_orbit.add_argument("--maneuver-max-candidates", type=int, default=12)
    fit_orbit.add_argument("--maneuver-max-nfev", type=int)

    args = parser.parse_args(argv)
    try:
        if args.command == "sample-sgp4":
            line1, line2 = _tle_lines_from_args(args)
            packet = sample_sgp4_observations_from_tle(
                object_id=args.object_id,
                line1=line1,
                line2=line2,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                mass_kg=args.mass_kg,
                start_jd_utc=args.start_jd_utc,
                source_label=args.source_label,
                position_sigma_km=args.position_sigma_km,
                velocity_sigma_km_s=args.velocity_sigma_km_s,
                include_velocity=not bool(args.no_velocity),
                require_checksum=bool(args.require_checksum),
                output_frame=args.output_frame,
                frame_transform=args.frame_transform,
            )
            packet.write_json(args.out)
            print(args.out)
            return 0
        if args.command == "tle-maneuver-screen":
            from sim.estimation.orbit_determination import solve_dynamics_orbit_determination

            line1, line2 = _tle_lines_from_args(args)
            output_root = Path(args.output_dir).expanduser()
            packet_path = (
                Path(args.packet_out).expanduser() if args.packet_out else output_root / "observation_packet.json"
            )
            packet = sample_sgp4_observations_from_tle(
                object_id=args.object_id,
                line1=line1,
                line2=line2,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                mass_kg=args.mass_kg,
                start_jd_utc=args.start_jd_utc,
                source_label=args.source_label,
                position_sigma_km=args.position_sigma_km,
                velocity_sigma_km_s=args.velocity_sigma_km_s,
                include_velocity=not bool(args.no_velocity),
                require_checksum=bool(args.require_checksum),
                output_frame=args.output_frame,
                frame_transform=args.frame_transform,
            )
            packet.write_json(packet_path)
            result = solve_dynamics_orbit_determination(
                packet,
                output_dir=output_root,
                object_id=args.object_id,
                role=args.role,
                scenario_name=args.scenario_name,
                fit_duration_s=args.fit_duration_s,
                holdout_duration_s=args.holdout_duration_s,
                dt_s=args.dt_s,
                estimate=args.estimate,
                dynamics_model=args.dynamics_model,
                j2=bool(args.j2),
                drag=bool(args.drag),
                srp=bool(args.srp),
                atmosphere_model=args.atmosphere_model.strip() or None,
                orbit_force_model=_orbit_force_model_from_args(args),
                base_specs=_base_specs_from_args(args),
                max_nfev=int(args.max_nfev),
                robust_loss=args.robust_loss,
                robust_f_scale=float(args.robust_f_scale),
                sigma_clip_threshold=args.sigma_clip_threshold,
                detect_maneuvers=True,
                maneuver_frame=args.maneuver_frame,
                max_maneuver_dv_m_s=float(args.max_maneuver_dv_m_s),
                maneuver_min_delta_v_m_s=float(args.maneuver_min_delta_v_m_s),
                maneuver_min_improvement_ratio=float(args.maneuver_min_improvement_ratio),
                maneuver_burn_duration_s=args.maneuver_burn_duration_s,
                maneuver_guard_observations=int(args.maneuver_guard_observations),
                maneuver_max_candidates=int(args.maneuver_max_candidates),
                maneuver_max_nfev=args.maneuver_max_nfev,
            )
            result["generated_observation_packet_path"] = str(packet_path)
            _write_json(result["report_json_path"], result)
            print(result["report_json_path"])
            return 0
        if args.command == "inspect":
            summary = inspect_observation_packet(load_observation_packet(args.packet))
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_observation_inspection(summary)
            return 0
        if args.command == "fit-position":
            estimate = fit_state_from_position_observations(
                load_observation_packet(args.packet),
                object_id=args.object_id,
                role=args.role,
                epoch=args.epoch,
            )
            _write_json(args.out, estimate["mission_input_packet"])
            if args.estimate_json:
                _write_json(args.estimate_json, _estimate_payload_without_packet(estimate))
            print(args.out)
            return 0
        if args.command == "kalman-position":
            estimate = kalman_filter_position_observations(
                load_observation_packet(args.packet),
                object_id=args.object_id,
                role=args.role,
                process_accel_sigma_km_s2=args.process_accel_sigma_km_s2,
                initial_position_sigma_km=args.initial_position_sigma_km,
                initial_velocity_sigma_km_s=args.initial_velocity_sigma_km_s,
                default_position_sigma_km=args.default_position_sigma_km,
            )
            _write_json(args.out, estimate["mission_input_packet"])
            if args.estimate_json:
                _write_json(args.estimate_json, _estimate_payload_without_packet(estimate))
            print(args.out)
            return 0
        if args.command == "fit-orbit":
            from sim.estimation.orbit_determination import solve_dynamics_orbit_determination

            result = solve_dynamics_orbit_determination(
                load_observation_packet(args.packet),
                output_dir=args.output_dir,
                object_id=args.object_id,
                role=args.role,
                scenario_name=args.scenario_name,
                fit_duration_s=args.fit_duration_s,
                holdout_duration_s=args.holdout_duration_s,
                dt_s=args.dt_s,
                estimate=args.estimate,
                dynamics_model=args.dynamics_model,
                j2=bool(args.j2),
                drag=bool(args.drag),
                srp=bool(args.srp),
                atmosphere_model=args.atmosphere_model.strip() or None,
                orbit_force_model=_orbit_force_model_from_args(args),
                base_specs=_base_specs_from_args(args),
                attitude_source=args.attitude_source,
                attitude_mode=args.attitude_mode,
                attitude_body_axis=args.sun_track_body_axis,
                attitude_controller=args.attitude_controller,
                max_nfev=int(args.max_nfev),
                robust_loss=args.robust_loss,
                robust_f_scale=float(args.robust_f_scale),
                sigma_clip_threshold=args.sigma_clip_threshold,
                detect_maneuvers=bool(args.detect_maneuvers),
                maneuver_frame=args.maneuver_frame,
                max_maneuver_dv_m_s=float(args.max_maneuver_dv_m_s),
                maneuver_min_delta_v_m_s=float(args.maneuver_min_delta_v_m_s),
                maneuver_min_improvement_ratio=float(args.maneuver_min_improvement_ratio),
                maneuver_burn_duration_s=args.maneuver_burn_duration_s,
                maneuver_guard_observations=int(args.maneuver_guard_observations),
                maneuver_max_candidates=int(args.maneuver_max_candidates),
                maneuver_max_nfev=args.maneuver_max_nfev,
            )
            if args.state_packet_out:
                source = Path(result["fitted_mission_input_packet_path"])
                target = Path(args.state_packet_out).expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            print(result["report_json_path"])
            return 0
    except Exception as exc:
        print(f"observation tool failed: {exc}", file=sys.stderr)
        return 2
    return 2

def _add_sgp4_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--object-id", required=True)
    tle_group = parser.add_mutually_exclusive_group(required=True)
    tle_group.add_argument(
        "--tle-file",
        help="TLE file containing line 1 and line 2, optionally after a title line.",
    )
    tle_group.add_argument("--line1", help="TLE line 1; requires --line2.")
    parser.add_argument("--line2", help="TLE line 2 when using --line1.")
    parser.add_argument("--source-label", default="sgp4_tle")
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--dt-s", type=float, required=True)
    parser.add_argument("--start-jd-utc", type=float)
    parser.add_argument("--mass-kg", type=float, default=1.0)
    parser.add_argument("--position-sigma-km", type=float, default=1.0)
    parser.add_argument("--velocity-sigma-km-s", type=float, default=1.0e-3)
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--require-checksum", action="store_true")
    parser.add_argument("--output-frame", default="eci")
    parser.add_argument("--frame-transform")


def _add_robust_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="linear",
    )
    parser.add_argument("--robust-f-scale", type=float, default=1.0)
    parser.add_argument("--sigma-clip-threshold", type=float)


def _tle_lines_from_args(args: Any) -> tuple[str, str]:
    if getattr(args, "tle_file", None):
        return _read_tle_lines(args.tle_file)
    line1 = str(getattr(args, "line1", "") or "").strip()
    line2 = str(getattr(args, "line2", "") or "").strip()
    if not line1 or not line2:
        raise ValueError("--line1 requires --line2.")
    return line1, line2


def _read_tle_lines(path: str | Path) -> tuple[str, str]:
    lines = [line.strip() for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
    for idx, line in enumerate(lines[:-1]):
        if line.startswith("1 ") and lines[idx + 1].startswith("2 "):
            return line, lines[idx + 1]
    raise ValueError("TLE file must contain adjacent line 1 and line 2 records.")


def _estimate_payload_without_packet(estimate: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(estimate))
    payload.pop("mission_input_packet", None)
    return payload


def _base_specs_from_args(args: Any) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for attr, key in (
        ("mass_kg", "mass_kg"),
        ("area_m2", "area_m2"),
        ("drag_area_m2", "drag_area_m2"),
        ("srp_area_m2", "srp_area_m2"),
        ("solar_area_m2", "solar_area_m2"),
        ("cd", "cd"),
        ("cr", "cr"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            specs[key] = float(value)
    geometry_profile_path = getattr(args, "geometry_profile_path", None)
    if geometry_profile_path:
        specs["geometry"] = {"profile_path": str(geometry_profile_path)}
    return specs


def _orbit_force_model_from_args(args: Any) -> dict[str, Any]:
    force_model: dict[str, Any] = {}
    raw_json = getattr(args, "orbit_force_model_json", None)
    if raw_json:
        try:
            loaded = json.loads(str(raw_json))
        except json.JSONDecodeError as exc:
            raise ValueError(f"--orbit-force-model-json must be a JSON object: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError("--orbit-force-model-json must be a JSON object.")
        force_model.update(dict(loaded))
    for attr, key in (
        ("j3", "j3"),
        ("j4", "j4"),
        ("third_body_sun", "third_body_sun"),
        ("third_body_moon", "third_body_moon"),
    ):
        if bool(getattr(args, attr, False)):
            force_model[key] = True
    sh_degree = getattr(args, "spherical_harmonics_degree", None)
    sh_order = getattr(args, "spherical_harmonics_order", None)
    sh_source = str(getattr(args, "spherical_harmonics_source", "") or "").strip()
    sh_coeff_path = getattr(args, "spherical_harmonics_coeff_path", None)
    sh_frame_model = getattr(args, "spherical_harmonics_frame_model", None)
    sh_eop_path = getattr(args, "spherical_harmonics_eop_path", None)
    if any(
        value not in (None, "")
        for value in (sh_degree, sh_order, sh_source, sh_coeff_path, sh_frame_model, sh_eop_path)
    ):
        sh = dict(force_model.get("spherical_harmonics", {}) or {})
        sh["enabled"] = True
        if sh_degree is not None:
            sh["degree"] = int(sh_degree)
        if sh_order is not None:
            sh["order"] = int(sh_order)
        if sh_source:
            sh["source"] = sh_source
        if sh_coeff_path:
            sh["coeff_path"] = str(sh_coeff_path)
        if sh_frame_model:
            sh["frame_model"] = str(sh_frame_model)
        if sh_eop_path:
            sh["eop_path"] = str(sh_eop_path)
        force_model["spherical_harmonics"] = sh
    return force_model


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
