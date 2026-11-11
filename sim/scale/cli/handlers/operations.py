from __future__ import annotations

import argparse
import json

from sim.scale.handoff import run_hpop_handoff, run_onp_handoff
from sim.scale.onp_execution import run_onp_refinement_queue
from sim.scale.onp_refinement import run_onp_refinement_materialization
from sim.scale.propagation import run_ogp_propagation
from sim.scale.recompute import (
    run_observation_recompute_plan,
    run_recompute_plan,
    run_recompute_propagation,
    run_recompute_screening,
)
from sim.scale.refinement import run_candidate_refinement
from sim.scale.screening import run_pair_screening
from sim.scale.workflow import run_catalog_workflow, run_recompute_workflow, run_refresh_to_onp_workflow


def handle_propagate_ogp(args: argparse.Namespace) -> int:
    payload = run_ogp_propagation(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        ogp = dict(payload.get("ogp", {}) or {})
        counts = dict(ogp.get("object_counts", {}) or {})
        if counts:
            print(
                f"ogp_objects: total={counts.get('total')} sgp4={counts.get('sgp4')} sdp4={counts.get('sdp4')}"
            )
        print(f"products_inserted: {payload.get('products_inserted')}")
        print(f"samples_inserted: {payload.get('samples_inserted')}")
        print(f"invalidations_completed: {payload.get('invalidations_completed')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_plan_recompute(args: argparse.Namespace) -> int:
    payload = run_recompute_plan(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store', {}).get('path')}")
        print(f"open_invalidations: {payload.get('invalidation', {}).get('open_items')}")
        print(f"recompute_chunks: {payload.get('propagation', {}).get('recompute_chunk_count')}")
        print(f"affected_pair_windows: {payload.get('screening', {}).get('affected_pair_window_count')}")
        print(f"affected_candidates: {payload.get('candidates', {}).get('affected_candidate_count')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_plan_observation_recompute(args: argparse.Namespace) -> int:
    payload = run_observation_recompute_plan(
        args.store,
        object_ids=tuple(str(item) for item in args.object_id if str(item).strip()),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = dict(payload.get("summary", {}) or {})
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store', {}).get('path')}")
        print(f"ground_measurement_invalidations: {summary.get('ground_measurement_invalidations')}")
        print(f"objects_quality_allows_recompute: {summary.get('objects_quality_allows_recompute')}")
        print(f"objects_need_quality_assessment: {summary.get('objects_need_quality_assessment')}")
        print(f"objects_only_rejected: {summary.get('objects_only_rejected')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_run_recompute_propagation(args: argparse.Namespace) -> int:
    payload = run_recompute_propagation(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"execution_status: {payload.get('execution_status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"planned_tasks: {payload.get('propagation', {}).get('planned_task_count')}")
        print(f"products_inserted: {payload.get('propagation', {}).get('products_inserted')}")
        print(f"invalidations_completed: {payload.get('invalidation', {}).get('invalidations_completed')}")
    return 0 if payload.get("status") in {"completed", "warning"} else 2

def handle_run_recompute_screening(args: argparse.Namespace) -> int:
    payload = run_recompute_screening(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"product_pair_count: {payload.get('targeting', {}).get('product_pair_count')}")
        print(f"pair_windows_inserted: {payload.get('pair_windows_inserted')}")
        print(f"candidate_promotions_inserted: {payload.get('candidate_promotions_inserted')}")
        print(f"candidates_superseded: {payload.get('candidates_superseded')}")
        print(f"hpop_packets_marked_stale: {payload.get('hpop_packets_marked_stale')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_screen_pairs(args: argparse.Namespace) -> int:
    payload = run_pair_screening(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"pair_windows_inserted: {payload.get('pair_windows_inserted')}")
        print(f"candidate_promotions_inserted: {payload.get('candidate_promotions_inserted')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_refine_candidates(args: argparse.Namespace) -> int:
    payload = run_candidate_refinement(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"refinements_inserted: {payload.get('refinements_inserted')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_handoff_hpop(args: argparse.Namespace) -> int:
    payload = run_onp_handoff(args.config) if args.command == "handoff-onp" else run_hpop_handoff(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"packets_inserted: {payload.get('packets_inserted')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_materialize_onp_refinement(args: argparse.Namespace) -> int:
    payload = run_onp_refinement_materialization(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"cases_materialized: {payload.get('cases_materialized')}")
        print(f"validated_cases: {payload.get('validated_cases')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_run_onp_refinement_queue(args: argparse.Namespace) -> int:
    payload = run_onp_refinement_queue(
        args.config,
        workers=args.workers,
        backend=getattr(args, "backend", None),
        retry_failed=args.retry_failed,
        recover_incomplete=args.recover_incomplete,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"batch_run_id: {payload.get('batch_run_id')}")
        print(f"cases_claimed: {payload.get('cases_claimed')}")
        print(f"cases_completed: {payload.get('cases_completed')}")
        print(f"cases_failed: {payload.get('cases_failed')}")
        print(f"comparison_csv: {payload.get('artifacts', {}).get('comparison_csv')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_run_catalog_workflow(args: argparse.Namespace) -> int:
    payload = run_catalog_workflow(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"workflow_json: {payload.get('artifacts', {}).get('workflow_json')}")
        print(f"final_health: {payload.get('final_health')}")
    return 0 if payload.get("status") in {"completed", "warning"} else 2

def handle_run_recompute_workflow(args: argparse.Namespace) -> int:
    payload = run_recompute_workflow(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"workflow_json: {payload.get('artifacts', {}).get('workflow_json')}")
        print(f"final_health: {payload.get('final_health')}")
    return 0 if payload.get("status") in {"completed", "warning"} else 2

def handle_run_refresh_to_onp_workflow(args: argparse.Namespace) -> int:
    payload = run_refresh_to_onp_workflow(
        args.config,
        workers=args.workers,
        backend=getattr(args, "backend", None),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"workflow_json: {payload.get('artifacts', {}).get('workflow_json')}")
        print(f"comparison_json: {payload.get('artifacts', {}).get('comparison_json')}")
        print(f"final_health: {payload.get('final_health')}")
    return 0 if payload.get("status") in {"completed", "warning"} else 2

COMMAND_HANDLERS = {
    'propagate-ogp': handle_propagate_ogp,
    'propagate-sgp4': handle_propagate_ogp,
    'plan-recompute': handle_plan_recompute,
    'plan-observation-recompute': handle_plan_observation_recompute,
    'run-recompute-propagation': handle_run_recompute_propagation,
    'run-recompute-screening': handle_run_recompute_screening,
    'screen-pairs': handle_screen_pairs,
    'refine-candidates': handle_refine_candidates,
    'handoff-onp': handle_handoff_hpop,
    'handoff-hpop': handle_handoff_hpop,
    'materialize-onp-refinement': handle_materialize_onp_refinement,
    'run-onp-refinement-queue': handle_run_onp_refinement_queue,
    'run-catalog-workflow': handle_run_catalog_workflow,
    'run-recompute-workflow': handle_run_recompute_workflow,
    'run-refresh-to-onp-workflow': handle_run_refresh_to_onp_workflow,
}
