from __future__ import annotations

import argparse
import json

from sim.scale.monte_carlo import run_monte_carlo_campaign, run_monte_carlo_parallel_cpu, run_monte_carlo_serial
from sim.scale.sensitivity import run_sensitivity_campaign, run_sensitivity_parallel_cpu, run_sensitivity_serial


def handle_materialize_sensitivity(args: argparse.Namespace) -> int:
    payload = run_sensitivity_campaign(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"trials_recorded: {payload.get('trials_recorded')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_run_sensitivity_serial(args: argparse.Namespace) -> int:
    payload = run_sensitivity_serial(
        args.config,
        limit=int(args.limit or 0),
        propagate_sgp4=bool(args.propagate_sgp4),
        duration_s=args.duration_s,
        step_s=args.step_s,
        retry_failed=bool(args.retry_failed),
        propagation_summary_backend=str(args.propagation_summary_backend or "scalar_trial"),
        chunk_duration_s=args.chunk_duration_s,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"trials_seen: {payload.get('trials_seen')}")
        print(f"trials_pending: {payload.get('trials_pending')}")
        print(f"results_recorded: {payload.get('results_recorded')}")
        print(f"status_counts: {payload.get('status_counts')}")
    return 0 if payload.get("status") in {"completed", "blocked"} else 2

def handle_run_sensitivity_parallel_cpu(args: argparse.Namespace) -> int:
    payload = run_sensitivity_parallel_cpu(
        args.config,
        limit=int(args.limit or 0),
        workers=int(args.workers or 0),
        propagate_sgp4=bool(args.propagate_sgp4),
        duration_s=args.duration_s,
        step_s=args.step_s,
        retry_failed=bool(args.retry_failed),
        propagation_summary_backend=str(args.propagation_summary_backend or "scalar_trial"),
        chunk_duration_s=args.chunk_duration_s,
        backend=getattr(args, "backend", "parallel_cpu"),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"trials_seen: {payload.get('trials_seen')}")
        print(f"trials_pending: {payload.get('trials_pending')}")
        print(f"results_recorded: {payload.get('results_recorded')}")
        print(f"status_counts: {payload.get('status_counts')}")
    return 0 if payload.get("status") in {"completed", "blocked"} else 2

def handle_materialize_monte_carlo(args: argparse.Namespace) -> int:
    payload = run_monte_carlo_campaign(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"trials_recorded: {payload.get('trials_recorded')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_run_monte_carlo_serial(args: argparse.Namespace) -> int:
    payload = run_monte_carlo_serial(
        args.config,
        limit=int(args.limit or 0),
        propagate_sgp4=bool(args.propagate_sgp4),
        duration_s=args.duration_s,
        step_s=args.step_s,
        retry_failed=bool(args.retry_failed),
        sgp4_batch_backend=str(args.sgp4_batch_backend or ""),
        propagation_summary_backend=str(args.propagation_summary_backend or "scalar_trial"),
        chunk_duration_s=args.chunk_duration_s,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"trials_seen: {payload.get('trials_seen')}")
        print(f"trials_pending: {payload.get('trials_pending')}")
        print(f"results_recorded: {payload.get('results_recorded')}")
        print(f"status_counts: {payload.get('status_counts')}")
    return 0 if payload.get("status") in {"completed", "blocked"} else 2

def handle_run_monte_carlo_parallel_cpu(args: argparse.Namespace) -> int:
    payload = run_monte_carlo_parallel_cpu(
        args.config,
        limit=int(args.limit or 0),
        workers=int(args.workers or 0),
        propagate_sgp4=bool(args.propagate_sgp4),
        duration_s=args.duration_s,
        step_s=args.step_s,
        retry_failed=bool(args.retry_failed),
        sgp4_batch_backend=str(args.sgp4_batch_backend or ""),
        propagation_summary_backend=str(args.propagation_summary_backend or "scalar_trial"),
        chunk_duration_s=args.chunk_duration_s,
        backend=getattr(args, "backend", "parallel_cpu"),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"trials_seen: {payload.get('trials_seen')}")
        print(f"trials_pending: {payload.get('trials_pending')}")
        print(f"results_recorded: {payload.get('results_recorded')}")
        print(f"status_counts: {payload.get('status_counts')}")
    return 0 if payload.get("status") in {"completed", "blocked"} else 2

COMMAND_HANDLERS = {
    'materialize-sensitivity': handle_materialize_sensitivity,
    'run-sensitivity-serial': handle_run_sensitivity_serial,
    'run-sensitivity-parallel-cpu': handle_run_sensitivity_parallel_cpu,
    'materialize-monte-carlo': handle_materialize_monte_carlo,
    'run-monte-carlo-serial': handle_run_monte_carlo_serial,
    'run-monte-carlo-parallel-cpu': handle_run_monte_carlo_parallel_cpu,
}
