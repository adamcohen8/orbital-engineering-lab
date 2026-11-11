from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sim.estimation.productization import safely_inspect_od_input, write_standard_od_plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and productize OEL orbit-determination evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Safely inspect a JSON OD input without executing it.")
    inspect_parser.add_argument("input")

    plot_parser = subparsers.add_parser("plot", help="Create standard plots from a sequential or relative OD report.")
    plot_parser.add_argument("report")
    plot_parser.add_argument("--output-dir", required=True)

    adapters_parser = subparsers.add_parser("adapters", help="List registered OD adapters and their trust state.")
    adapters_parser.add_argument("--json", action="store_true")

    covariance_parser = subparsers.add_parser(
        "materialize-covariance-study",
        help="Materialize and validate a Pro covariance study from an accepted state product without executing it.",
    )
    covariance_parser.add_argument("--state-product", required=True, type=Path)
    covariance_parser.add_argument("--scenario-name", required=True)
    covariance_parser.add_argument("--output", required=True, type=Path)
    covariance_parser.add_argument("--run-output-dir", required=True, type=Path)
    covariance_parser.add_argument("--duration-s", required=True, type=float)
    covariance_parser.add_argument("--dt-s", required=True, type=float)
    covariance_parser.add_argument("--position-step-km", type=float, default=1.0e-3)
    covariance_parser.add_argument("--velocity-step-km-s", type=float, default=1.0e-6)
    covariance_parser.add_argument("--process-noise-acceleration-sigma-km-s2", type=float, default=0.0)
    covariance_parser.add_argument("--manifest", type=Path)
    covariance_parser.add_argument("--trust-plugins", action="store_true")
    covariance_parser.add_argument("--overwrite", action="store_true")
    covariance_parser.add_argument("--json", action="store_true")

    sequential_emit_parser = subparsers.add_parser(
        "emit-sequential-state-products",
        help="Emit filtered and smoothed state products from a completed sequential OD artifact directory.",
    )
    sequential_emit_parser.add_argument("--artifact-dir", required=True, type=Path)
    sequential_emit_parser.add_argument("--json", action="store_true")

    sequential_parser = subparsers.add_parser(
        "materialize-sequential-onp",
        help="Materialize one explicitly selected sequential OD state product without executing it.",
    )
    sequential_parser.add_argument("--product-dir", required=True, type=Path)
    sequential_parser.add_argument("--selection", required=True, choices=("filtered", "smoothed"))
    sequential_parser.add_argument("--scenario-name", required=True)
    sequential_parser.add_argument("--output", required=True, type=Path)
    sequential_parser.add_argument("--run-output-dir", required=True, type=Path)
    sequential_parser.add_argument("--duration-s", required=True, type=float)
    sequential_parser.add_argument("--dt-s", required=True, type=float)
    sequential_parser.add_argument("--manifest", type=Path)
    sequential_parser.add_argument("--trust-plugins", action="store_true")
    sequential_parser.add_argument("--overwrite", action="store_true")
    sequential_parser.add_argument("--json", action="store_true")

    relative_emit_parser = subparsers.add_parser(
        "emit-relative-state-product",
        help="Bind a relative OD report to an accepted chief state product.",
    )
    relative_emit_parser.add_argument("--report", required=True, type=Path)
    relative_emit_parser.add_argument("--chief-state-product", required=True, type=Path)
    relative_emit_parser.add_argument("--chief-id", required=True)
    relative_emit_parser.add_argument("--deputy-id", required=True)
    relative_emit_parser.add_argument("--output", required=True, type=Path)
    relative_emit_parser.add_argument("--json", action="store_true")

    relative_parser = subparsers.add_parser(
        "materialize-relative-onp",
        help="Materialize an accepted chief and rectangular-RIC relative product without executing them.",
    )
    relative_parser.add_argument("--relative-state-product", required=True, type=Path)
    relative_parser.add_argument("--chief-state-product", required=True, type=Path)
    relative_parser.add_argument("--scenario-name", required=True)
    relative_parser.add_argument("--output", required=True, type=Path)
    relative_parser.add_argument("--run-output-dir", required=True, type=Path)
    relative_parser.add_argument("--duration-s", required=True, type=float)
    relative_parser.add_argument("--dt-s", required=True, type=float)
    relative_parser.add_argument("--manifest", type=Path)
    relative_parser.add_argument("--trust-plugins", action="store_true")
    relative_parser.add_argument("--overwrite", action="store_true")
    relative_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            payload = safely_inspect_od_input(args.input)
            print(json.dumps(payload, indent=2))
            return 0 if payload.get("status") == "safe_inspected" else 2
        if args.command == "plot":
            report = json.loads(Path(args.report).expanduser().read_text(encoding="utf-8"))
            payload = write_standard_od_plots(report, args.output_dir)
            print(json.dumps({"plots": payload}, indent=2))
            return 0
        if args.command == "adapters":
            from sim.estimation.productization import default_od_adapter_registry

            rows = default_od_adapter_registry().list()
            if args.json:
                print(json.dumps({"adapters": rows}, indent=2))
            else:
                for row in rows:
                    print(f"{row['adapter_id']}: {row['packet_kind']} [{row['trust']}]")
            return 0
        if args.command == "materialize-covariance-study":
            from sim.interchange.adapters.covariance import materialize_covariance_study

            payload = materialize_covariance_study(
                args.state_product,
                scenario_name=args.scenario_name,
                scenario_path=args.output,
                output_dir=args.run_output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                position_step_km=args.position_step_km,
                velocity_step_km_s=args.velocity_step_km_s,
                process_noise_acceleration_sigma_km_s2=args.process_noise_acceleration_sigma_km_s2,
                manifest_path=args.manifest,
                trust_plugins=args.trust_plugins,
                overwrite=args.overwrite,
            )
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"status: {payload.get('status', '')}")
                print(f"scenario_path: {payload.get('scenario_path', '')}")
                print(f"manifest_path: {payload.get('manifest_path', '')}")
                print("execution_occurred: false")
                print(f"next_action: {payload.get('recommended_next_action', '')}")
            return 0 if payload.get("status") == "materialized" else 2
        if args.command == "emit-sequential-state-products":
            from sim.interchange.adapters.sequential_od import emit_sequential_od_state_products

            payload = emit_sequential_od_state_products(args.artifact_dir)
            print(json.dumps(payload, indent=2, sort_keys=True) if args.json else _compact_output(payload))
            return 0 if payload.get("status") == "emitted" else 2
        if args.command == "materialize-sequential-onp":
            from sim.interchange.materialization import materialize_onp

            product_path = args.product_dir / f"state_estimate_{args.selection}_product.json"
            payload = materialize_onp(
                product_path,
                scenario_name=args.scenario_name,
                scenario_path=args.output,
                output_dir=args.run_output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                manifest_path=args.manifest,
                trust_plugins=args.trust_plugins,
                overwrite=args.overwrite,
            )
            payload["sequential_selection"] = args.selection
            print(json.dumps(payload, indent=2, sort_keys=True) if args.json else _compact_output(payload))
            return 0 if payload.get("status") == "materialized" else 2
        if args.command == "emit-relative-state-product":
            from sim.interchange.adapters.relative_od import emit_relative_od_state_product

            payload = emit_relative_od_state_product(
                args.report,
                args.chief_state_product,
                chief_id=args.chief_id,
                deputy_id=args.deputy_id,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True) if args.json else _compact_output(payload))
            return 0 if payload.get("status") == "emitted" else 2
        if args.command == "materialize-relative-onp":
            from sim.interchange.adapters.relative_od import materialize_relative_onp

            payload = materialize_relative_onp(
                args.relative_state_product,
                args.chief_state_product,
                scenario_name=args.scenario_name,
                scenario_path=args.output,
                output_dir=args.run_output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                manifest_path=args.manifest,
                trust_plugins=args.trust_plugins,
                overwrite=args.overwrite,
            )
            print(json.dumps(payload, indent=2, sort_keys=True) if args.json else _compact_output(payload))
            return 0 if payload.get("status") == "materialized" else 2
    except Exception as exc:
        print(f"OD workflow failed: {exc}", file=sys.stderr)
        return 2
    return 2


def _compact_output(payload: dict) -> str:
    keys = ("status", "product_path", "scenario_path", "manifest_path", "recommended_next_action")
    return "\n".join(f"{key}: {payload[key]}" for key in keys if key in payload)


if __name__ == "__main__":
    raise SystemExit(main())
