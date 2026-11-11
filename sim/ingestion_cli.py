# ruff: noqa: F401,F403,F405,I001
from sim.ingestion_normalization import *
from sim.ingestion_sources import *
from sim.scenario_builders import *

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize structured mission inputs and build OEL scenario YAML.")
    sub = parser.add_subparsers(dest="command", required=True)

    tle = sub.add_parser("tle", help="Normalize TLE lines as a mission input packet.")
    tle.add_argument("--object-id", required=True)
    tle.add_argument("--line1", required=True)
    tle.add_argument("--line2", required=True)
    tle.add_argument("--role")
    tle.add_argument("--source-label", default="agent")
    tle.add_argument("--require-checksum", action="store_true")
    tle.add_argument("--no-propagate-to-initial-epoch", action="store_true")
    tle.add_argument("--reference-jd-utc", type=float)
    tle.add_argument("--out", required=True)

    sv = sub.add_parser("state-vector", help="Normalize an explicit ECI state vector.")
    sv.add_argument("--object-id", required=True)
    sv.add_argument("--position", nargs=3, required=True, type=float, metavar=("X", "Y", "Z"))
    sv.add_argument("--velocity", nargs=3, required=True, type=float, metavar=("VX", "VY", "VZ"))
    sv.add_argument("--role")
    sv.add_argument("--source-label", default="agent")
    sv.add_argument("--frame", default="eci")
    sv.add_argument("--epoch-jd-utc", type=float)
    sv.add_argument("--position-units", default="km")
    sv.add_argument("--velocity-units", default="km/s")
    sv.add_argument("--out", required=True)

    coes = sub.add_parser("coes", help="Normalize classical orbital elements.")
    coes.add_argument("--object-id", required=True)
    coes.add_argument("--a-km", required=True, type=float)
    coes.add_argument("--ecc", required=True, type=float)
    coes.add_argument("--inc-deg", required=True, type=float)
    coes.add_argument("--raan-deg", required=True, type=float)
    coes.add_argument("--argp-deg", required=True, type=float)
    coes.add_argument("--true-anomaly-deg", required=True, type=float)
    coes.add_argument("--role")
    coes.add_argument("--source-label", default="agent")
    coes.add_argument("--epoch-jd-utc", type=float)
    coes.add_argument("--out", required=True)

    ric = sub.add_parser("relative-ric", help="Normalize a chaser relative RIC state.")
    ric.add_argument("--object-id", required=True)
    ric.add_argument("--reference-id", default="target")
    ric.add_argument("--state", nargs=6, required=True, type=float, metavar=("R", "I", "C", "RDOT", "IDOT", "CDOT"))
    ric.add_argument("--frame", default="rect")
    ric.add_argument("--role", default="chaser")
    ric.add_argument("--source-label", default="agent")
    ric.add_argument("--position-units", default="km")
    ric.add_argument("--velocity-units", default="km/s")
    ric.add_argument("--out", required=True)

    merge = sub.add_parser("merge", help="Merge mission input packets.")
    merge.add_argument("--packet", action="append", required=True)
    merge.add_argument("--out", required=True)

    inspect_parser = sub.add_parser("inspect", help="Inspect a mission input packet and report build readiness.")
    inspect_parser.add_argument("packet")
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.add_argument("--markdown", action="store_true")
    inspect_parser.add_argument("--markdown-out")

    prop = sub.add_parser("build-propagation", help="Build propagation scenario YAML from a packet.")
    _add_build_args(prop)

    rpo = sub.add_parser("build-rpo", help="Build basic RPO scenario YAML from a packet.")
    _add_build_args(rpo)
    rpo.add_argument("--target-id", default="target")
    rpo.add_argument("--chaser-id", default="chaser")

    args = parser.parse_args(argv)
    try:
        if args.command == "tle":
            packet = ingest_tle(
                object_id=args.object_id,
                line1=args.line1,
                line2=args.line2,
                role=args.role,
                source_label=args.source_label,
                require_checksum=args.require_checksum,
                propagate_to_initial_epoch=not args.no_propagate_to_initial_epoch,
                reference_jd_utc=args.reference_jd_utc,
            )
            print(packet.write_json(args.out))
            return 0
        if args.command == "state-vector":
            packet = ingest_state_vector(
                object_id=args.object_id,
                position=args.position,
                velocity=args.velocity,
                role=args.role,
                source_label=args.source_label,
                frame=args.frame,
                epoch_jd_utc=args.epoch_jd_utc,
                position_units=args.position_units,
                velocity_units=args.velocity_units,
            )
            print(packet.write_json(args.out))
            return 0
        if args.command == "coes":
            packet = ingest_coes(
                object_id=args.object_id,
                coes={
                    "a_km": args.a_km,
                    "ecc": args.ecc,
                    "inc_deg": args.inc_deg,
                    "raan_deg": args.raan_deg,
                    "argp_deg": args.argp_deg,
                    "true_anomaly_deg": args.true_anomaly_deg,
                },
                role=args.role,
                source_label=args.source_label,
                epoch_jd_utc=args.epoch_jd_utc,
            )
            print(packet.write_json(args.out))
            return 0
        if args.command == "relative-ric":
            packet = ingest_relative_ric_state(
                object_id=args.object_id,
                reference_id=args.reference_id,
                state=args.state,
                frame=args.frame,
                role=args.role,
                source_label=args.source_label,
                position_units=args.position_units,
                velocity_units=args.velocity_units,
            )
            print(packet.write_json(args.out))
            return 0
        if args.command == "merge":
            packet = merge_packets(*(load_packet(path) for path in args.packet))
            print(packet.write_json(args.out))
            return 0
        if args.command == "inspect":
            packet = load_packet(args.packet)
            summary = inspect_packet(packet)
            if args.markdown or args.markdown_out:
                markdown = render_ingestion_summary(packet)
                if args.markdown_out:
                    target = Path(args.markdown_out).expanduser()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(markdown, encoding="utf-8")
                    print(target)
                else:
                    print(markdown, end="")
                return 0
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_inspection(summary)
            return 0
        if args.command == "build-propagation":
            artifact = build_basic_propagation_scenario(
                load_packet(args.packet),
                scenario_name=args.scenario_name,
                output_dir=args.output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                initial_jd_utc=args.initial_jd_utc,
                dynamics_model=args.dynamics_model,
                j2=args.j2,
                review_detail=args.review_detail,
            )
            print(artifact.write(args.scenario_path))
            return 0
        if args.command == "build-rpo":
            artifact = build_basic_rpo_scenario(
                load_packet(args.packet),
                scenario_name=args.scenario_name,
                output_dir=args.output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                target_id=args.target_id,
                chaser_id=args.chaser_id,
                initial_jd_utc=args.initial_jd_utc,
                dynamics_model=args.dynamics_model,
                j2=args.j2,
                review_detail=args.review_detail,
            )
            print(artifact.write(args.scenario_path))
            return 0
    except Exception as exc:
        print(f"ingestion failed: {exc}", file=sys.stderr)
        return 2
    return 2
