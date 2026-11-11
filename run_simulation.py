# ruff: noqa: E402 -- CLI thread policy must run before NumPy-backed imports.
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sim.doctor import (
    doctor_requested,
    require_supported_interpreter,
)
from sim.doctor import print_doctor_report as _print_doctor_report

if __name__ == "__main__":
    if doctor_requested(sys.argv[1:]):
        raise SystemExit(0 if _print_doctor_report(source_root=Path(__file__).resolve().parent) else 1)
    require_supported_interpreter()

from sim.platform_compat import open_folder
from sim.runtime_environment import configure_headless_runtime, configure_runtime_caches

_NATIVE_MATH_THREAD_ENV_VARS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _configure_cli_native_math_threads(default_threads: str = "1") -> None:
    value = str(default_threads).strip()
    if not value.isdigit() or int(value) <= 0:
        raise ValueError("default native math thread count must be a positive integer.")
    for name in _NATIVE_MATH_THREAD_ENV_VARS:
        os.environ.setdefault(name, value)


if __name__ == "__main__":
    _configure_cli_native_math_threads()
    configure_runtime_caches()
    _HEADLESS_RUNTIME_STATUS = configure_headless_runtime()

from sim.config import load_simulation_yaml, validate_scenario_plugins
from sim.execution import run_simulation_config_file
from sim.execution.study import analysis_study_type
from sim.installation.resources import quickstart_config_path
from sim.security.sealed_mode import SealedModePolicy, validate_sealed_mode

QUICKSTART_CONFIG = quickstart_config_path()


def _print_preflight(config_path: str, cfg, errors: list[str], *, import_plugins: bool) -> None:
    print("")
    print("=" * 72)
    print("SIMULATION PREFLIGHT")
    print("=" * 72)
    print(f"Config   : {Path(config_path).resolve()}")
    print(f"Scenario : {cfg.scenario_name}")
    mode = analysis_study_type(cfg)
    print(f"Mode     : {mode if mode in {'monte_carlo', 'sensitivity', 'covariance'} else 'Single Run'}")
    print(f"Timing   : duration={float(cfg.simulator.duration_s):.1f} s, dt={float(cfg.simulator.dt_s):.3f} s")
    if errors:
        print("Status   : INVALID")
        for err in errors:
            print(f"- {err}")
    else:
        print(f"Status   : {'OK' if import_plugins else 'STRUCTURALLY OK'}")
        if not import_plugins:
            print("Plugins  : SKIPPED (safe validation does not import plugins)")
    print("=" * 72)


def _prepare_cli_output_directory(cfg, *, overwrite: bool) -> None:
    # Refuse ambiguous evidence reuse and archive only with explicit consent.

    output_dir = Path(cfg.outputs.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"Configured output path is not a directory: {output_dir}")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    free_bytes = int(shutil.disk_usage(parent).free)
    object_count = max(len(list(getattr(cfg, "objects", ()) or ())), 1)
    samples = max(int(round(float(cfg.simulator.duration_s) / float(cfg.simulator.dt_s))) + 1, 1)
    estimated_floor = max(64 * 1024 * 1024, samples * object_count * 2048)
    if free_bytes < estimated_floor:
        raise SystemExit(
            "Output preflight found insufficient filesystem headroom; refusing to start before evidence files are opened. "
            f"Available={free_bytes / (1024 ** 2):.1f} MiB, required floor={estimated_floor / (1024 ** 2):.1f} MiB, "
            f"output={output_dir}. Free space or choose a smaller run."
        )
    if not output_dir.is_dir() or not any(output_dir.iterdir()):
        return
    if not overwrite:
        raise SystemExit(
            "Configured output directory is not empty; refusing to overwrite or mix prior evidence: "
            f"{output_dir}\nUse a distinct outputs.output_dir, or rerun with --overwrite-output "
            "to archive the existing directory before execution."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = output_dir.with_name(f"{output_dir.name}.previous-{stamp}")
    suffix = 2
    while archived.exists():
        archived = output_dir.with_name(f"{output_dir.name}.previous-{stamp}-{suffix}")
        suffix += 1
    output_dir.replace(archived)
    print(f"Archived Output: {archived}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output_replacement_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "oel.output-replacement.v1",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "replacement_authorized_by": "cli:--overwrite-output",
                "archived_output_dir": str(archived),
                "new_output_dir": str(output_dir),
                "prior_evidence_deleted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _output_index_path(out: dict, cfg) -> str:
    run = dict(out.get("run", {}) or {})
    for candidate in (run.get("output_index_md"), out.get("output_index_md")):
        text = str(candidate or "").strip()
        if text:
            return text
    index_path = Path(str(run.get("output_dir") or cfg.outputs.output_dir)) / "index.md"
    return str(index_path) if index_path.exists() else ""


def _open_output_folder(path_text: str | Path) -> bool:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.is_file():
        path = path.parent
    if not path.exists():
        print(f"Open Output: skipped; folder does not exist: {path}")
        return False
    try:
        open_folder(path)
        print(f"Open Output: {path}")
        return True
    except Exception as exc:
        print(f"Open Output: failed to open {path} ({exc})")
        return False


def _print_single_run_summary(out: dict) -> None:
    run = dict(out.get("run", {}) or {})
    objects = [str(item) for item in list(run.get("objects", []) or [])]
    print("")
    print("=" * 72)
    print("SIMULATION COMPLETED")
    print("=" * 72)
    print(f"Scenario : {out.get('scenario_name', run.get('scenario_name', 'unknown'))}")
    print(f"Samples  : {run.get('samples', 0)}")
    print(f"Duration : {float(run.get('duration_s', 0.0)):.1f} s")
    print(f"Output   : {run.get('output_dir', '')}")
    if "rocket" in objects and "rocket_insertion_achieved" in run:
        if bool(run.get("rocket_insertion_achieved", False)):
            print(f"Insertion  : achieved at t={run.get('rocket_insertion_time_s')}")
        else:
            print("Insertion  : not achieved")
    print("=" * 72)


def _plain_progress_reporter():
    last_percent = -10
    last_reported = time.monotonic()

    def report(step: int, total: int) -> None:
        nonlocal last_percent, last_reported
        if total <= 0:
            return
        percent = min(100, max(0, int(100 * int(step) / int(total))))
        now = time.monotonic()
        if percent >= last_percent + 10 or now - last_reported >= 30.0 or int(step) >= int(total):
            print(f"Simulation progress: {int(step)}/{int(total)} steps ({percent}%)", file=sys.stderr)
            last_percent = percent
            last_reported = now

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public-core Orbital Engineering Lab scenario.")
    parser.add_argument("--config", default="", help="Path to a simulation scenario YAML file.")
    parser.add_argument("--campaign", type=Path, help="Run or validate a config queue.")
    parser.add_argument("--controller-bench", type=Path, help="Run or validate a controller benchmark.")
    parser.add_argument("--quickstart", action="store_true", help="Run the bundled five-minute quickstart scenario.")
    parser.add_argument(
        "--doctor", action="store_true", help="Check the local Python environment and quickstart readiness."
    )
    parser.add_argument(
        "--open-output", action="store_true", help="Open the output folder after a successful simulation run."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the scenario config and exit without running the simulator.",
    )
    parser.add_argument(
        "--safe-validate",
        action="store_true",
        help="Validate schema and plugin pointer shape without importing configured plugin modules.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help=(
            "Archive a non-empty configured output directory to a timestamped sibling before execution. "
            "Without this flag, ordinary scenario runs refuse non-empty output directories."
        ),
    )
    parser.add_argument(
        "--sealed-mode",
        action="store_true",
        help="Apply a restricted profile for shared/classroom/government use.",
    )
    parser.add_argument(
        "--allow-untrusted-plugin-imports",
        action="store_true",
        help="In --sealed-mode, allow scenario plugin modules outside the built-in OEL allowlist.",
    )
    parser.add_argument(
        "--allow-hosted-ai",
        action="store_true",
        help="In --sealed-mode, allow live hosted AI providers.",
    )
    parser.add_argument(
        "--allow-custom-ai-endpoints",
        action="store_true",
        help="In --sealed-mode, allow custom AI endpoints.",
    )
    parser.add_argument(
        "--allow-high-detail-outputs",
        action="store_true",
        help="In --sealed-mode, allow full logs, full review stores, raw MC runs, or non-summary AI packets.",
    )
    args = parser.parse_args()
    if args.doctor:
        if not _print_doctor_report(source_root=Path(__file__).resolve().parent):
            raise SystemExit(1)
        return
    if sum(bool(value) for value in (args.config, args.quickstart, args.campaign, args.controller_bench)) != 1:
        parser.error("Choose one of --config, --quickstart, --campaign, or --controller-bench.")
    validate_only = args.validate_only or args.safe_validate
    if args.campaign or args.controller_bench:
        # Structured workflows retain the same process-wide sealed policy as scenario runs.
        if args.sealed_mode:
            os.environ["OEL_SEALED_MODE"] = "1"
        if args.campaign:
            from sim.execution.campaigns import run_config_campaign, validate_config_campaign

            if validate_only:
                result = validate_config_campaign(args.campaign, import_plugins=not (args.safe_validate or args.sealed_mode))
                print(json.dumps(result, indent=2, default=str))
                if not result["valid"]:
                    raise SystemExit(1)
                return
            result = run_config_campaign(campaign_path=args.campaign, import_plugins=not args.sealed_mode)
        else:
            from sim.controller_lab import run_controller_bench, validate_controller_bench_config

            if args.safe_validate or args.sealed_mode:
                parser.error("Use --validate-only on a trusted controller bench config.")
            if validate_only:
                result = validate_controller_bench_config(args.controller_bench)
                print(json.dumps(result, indent=2, default=str))
                if not result["passed"]:
                    raise SystemExit(1)
                return
            result = run_controller_bench(config_path=args.controller_bench)
        print(json.dumps(result, indent=2, default=str))
        return
    config_path = str(QUICKSTART_CONFIG if args.quickstart else args.config)
    if not config_path:
        raise SystemExit("Provide --config PATH or use --quickstart.")

    cfg = load_simulation_yaml(config_path)
    import_plugins = bool(not args.safe_validate and not args.sealed_mode)
    errors = (
        list(validate_scenario_plugins(cfg, import_plugins=import_plugins))
        if bool(cfg.simulator.plugin_validation.get("strict", True))
        else []
    )
    if args.sealed_mode:
        errors.extend(
            validate_sealed_mode(
                cfg,
                SealedModePolicy(
                    allow_untrusted_plugin_imports=bool(args.allow_untrusted_plugin_imports),
                    allow_hosted_ai=bool(args.allow_hosted_ai),
                    allow_custom_ai_endpoints=bool(args.allow_custom_ai_endpoints),
                    allow_non_loopback_sil=False,
                    allow_high_detail_outputs=bool(args.allow_high_detail_outputs),
                ),
            )
        )
    if args.validate_only or args.safe_validate:
        _print_preflight(config_path, cfg, errors, import_plugins=import_plugins)
        if errors:
            raise SystemExit(1)
        return
    if errors:
        msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
        raise SystemExit(msg)

    if analysis_study_type(cfg) in {"monte_carlo", "sensitivity", "covariance"}:
        out = run_simulation_config_file(config_path)
        print(json.dumps(out.get("artifacts", {}), indent=2, default=str))
        return
    _prepare_cli_output_directory(cfg, overwrite=bool(args.overwrite_output))
    out = run_simulation_config_file(config_path, step_callback=_plain_progress_reporter())
    run = dict(out.get("run", {}) or {})
    print("")
    print("=" * 72)
    print("SIMULATION COMPLETED")
    print("=" * 72)
    print(f"Scenario : {out.get('scenario_name', run.get('scenario_name', 'unknown'))}")
    print(f"Samples  : {run.get('samples', 0)}")
    print(f"Duration : {float(run.get('duration_s', 0.0)):.1f} s")
    print(f"Output   : {run.get('output_dir') or cfg.outputs.output_dir}")
    index_path = _output_index_path(out, cfg)
    if index_path:
        print(f"Start Here: {index_path}")
    if args.open_output:
        _open_output_folder(run.get("output_dir") or cfg.outputs.output_dir)
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(
            "OEL input/configuration error: "
            f"{exc}\nHint: run this config with --safe-validate for structural inspection, "
            "then use --validate-only after its referenced plugins and paths are trusted."
        ) from None
