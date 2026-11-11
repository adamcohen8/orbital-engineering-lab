from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.scale.ingest import ingest_ogp_mean_element_product, ingest_tle_file
from sim.scale.planner import run_dry_run
from sim.scale.validation import run_scale_validation


def handle_dry_run(args: argparse.Namespace) -> int:
    payload = run_dry_run(args.config, output_dir=Path(args.output_dir) if args.output_dir else None)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"planner_json: {payload.get('artifacts', {}).get('planner_json')}")
        print(f"planner_markdown: {payload.get('artifacts', {}).get('planner_markdown')}")
        print(f"store_path: {payload.get('store', {}).get('path')}")
    return 0 if payload.get("status") == "dry_run" else 2

def handle_validate_scale(args: argparse.Namespace) -> int:
    payload = run_scale_validation(args.config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"tier: {payload.get('tier')}")
        print(f"evidence_json: {payload.get('artifacts', {}).get('evidence_json')}")
        print(f"evidence_markdown: {payload.get('artifacts', {}).get('evidence_markdown')}")
    return 0 if payload.get("status") == "passed" else 2

def handle_ingest_tle(args: argparse.Namespace) -> int:
    payload = ingest_tle_file(
        args.tle_file,
        store_path=args.store,
        source_label=args.source_label,
        object_id_prefix=args.object_id_prefix,
        require_checksum=bool(args.require_checksum),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"records_seen: {payload.get('records_seen')}")
        print(f"tle_records_inserted: {payload.get('tle_records_inserted')}")
        print(f"current_states_updated: {payload.get('current_states_updated')}")
        print(f"invalidations_inserted: {payload.get('invalidations_inserted')}")
    return 0 if payload.get("status") == "completed" else 2


def handle_ingest_ogp_product(args: argparse.Namespace) -> int:
    payload = ingest_ogp_mean_element_product(
        args.product,
        store_path=args.store,
        object_id=args.object_id,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"object_id: {payload.get('object_id')}")
        print(f"source_product_id: {payload.get('source_product_id')}")
        print("tle_text_generated: false")
    return 0 if payload.get("status") == "completed" else 2

COMMAND_HANDLERS = {
    'dry-run': handle_dry_run,
    'validate-scale': handle_validate_scale,
    'ingest-tle': handle_ingest_tle,
    'ingest-ogp-product': handle_ingest_ogp_product,
}
