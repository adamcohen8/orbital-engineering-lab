from __future__ import annotations

import argparse


def register_operations_parsers(sub: argparse._SubParsersAction) -> None:
    propagate_ogp_parser = sub.add_parser("propagate-ogp", help="Write mixed OGP propagation chunks.")
    propagate_ogp_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    propagate_ogp_parser.add_argument("--json", action="store_true", help="Print the propagation payload as JSON.")

    propagate_parser = sub.add_parser("propagate-sgp4", help="Compatibility alias for mixed OGP propagation chunks.")
    propagate_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    propagate_parser.add_argument("--json", action="store_true", help="Print the propagation payload as JSON.")

    recompute_parser = sub.add_parser("plan-recompute", help="Plan incremental recompute work from open invalidations.")
    recompute_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    recompute_parser.add_argument("--json", action="store_true", help="Print the recompute plan payload as JSON.")

    observation_recompute_parser = sub.add_parser(
        "plan-observation-recompute",
        help="Plan observation recompute work from open new-observation invalidations and quality rows.",
    )
    observation_recompute_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    observation_recompute_parser.add_argument(
        "--object-id", action="append", default=[], help="Optional object ID; repeatable."
    )
    observation_recompute_parser.add_argument(
        "--json", action="store_true", help="Print the observation recompute plan as JSON."
    )

    recompute_propagation_parser = sub.add_parser(
        "run-recompute-propagation",
        help="Execute only OGP propagation chunks selected by the recompute planner.",
    )
    recompute_propagation_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    recompute_propagation_parser.add_argument(
        "--json", action="store_true", help="Print the recompute execution payload as JSON."
    )

    recompute_screening_parser = sub.add_parser(
        "run-recompute-screening",
        help="Rescreen only pair windows affected by superseded propagation products.",
    )
    recompute_screening_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    recompute_screening_parser.add_argument(
        "--json", action="store_true", help="Print the recompute screening payload as JSON."
    )

    screen_parser = sub.add_parser("screen-pairs", help="Screen completed OGP propagation chunks by sampled distance.")
    screen_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    screen_parser.add_argument("--json", action="store_true", help="Print the screening payload as JSON.")

    refine_parser = sub.add_parser("refine-candidates", help="Refine promoted OGP candidates on a tighter sample grid.")
    refine_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    refine_parser.add_argument("--json", action="store_true", help="Print the refinement payload as JSON.")

    onp_handoff_parser = sub.add_parser("handoff-onp", help="Write manifest-only ONP handoff packets.")
    onp_handoff_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    onp_handoff_parser.add_argument("--json", action="store_true", help="Print the handoff payload as JSON.")

    handoff_parser = sub.add_parser(
        "handoff-hpop",
        help="Compatibility alias for the legacy HPOP-named ONP handoff command.",
    )
    handoff_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    handoff_parser.add_argument("--json", action="store_true", help="Print the handoff payload as JSON.")

    onp_materialize_parser = sub.add_parser(
        "materialize-onp-refinement",
        help="Generate and validate focused ONP scenarios from current handoff packets.",
    )
    onp_materialize_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    onp_materialize_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the materialization payload as JSON.",
    )

    onp_queue_parser = sub.add_parser(
        "run-onp-refinement-queue",
        help="Resume validated ONP cases and write OGP-vs-ONP review evidence.",
    )
    onp_queue_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    onp_queue_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker slots; zero chooses a bounded CPU count and one runs serially.",
    )
    onp_queue_parser.add_argument(
        "--backend",
        choices=("process_pool", "thread_pool", "serial"),
        default=None,
        help="Execution transport; defaults to process_pool for multiple workers. thread_pool is explicit opt-in.",
    )
    onp_queue_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Create a new attempt for cases whose latest attempt failed or was blocked.",
    )
    onp_queue_parser.add_argument(
        "--recover-incomplete",
        action="store_true",
        help="Mark abandoned claimed/running attempts interrupted before resuming them.",
    )
    onp_queue_parser.add_argument("--limit", type=int, help="Maximum number of cases to claim in this batch.")
    onp_queue_parser.add_argument("--json", action="store_true", help="Print the execution payload as JSON.")

    catalog_workflow_parser = sub.add_parser(
        "run-catalog-workflow",
        help="Run ordered catalog OGP propagation, screening, refinement, and ONP handoff.",
    )
    catalog_workflow_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    catalog_workflow_parser.add_argument("--json", action="store_true", help="Print the workflow payload as JSON.")

    recompute_workflow_parser = sub.add_parser(
        "run-recompute-workflow",
        help="Run ordered targeted recompute propagation, rescreening, refinement, and ONP handoff.",
    )
    recompute_workflow_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    recompute_workflow_parser.add_argument("--json", action="store_true", help="Print the workflow payload as JSON.")

    refresh_onp_parser = sub.add_parser(
        "run-refresh-to-onp-workflow",
        help="Run targeted recompute through ONP materialization, execution, and before/after comparison.",
    )
    refresh_onp_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    refresh_onp_parser.add_argument("--workers", type=int, default=1)
    refresh_onp_parser.add_argument(
        "--backend",
        choices=("process_pool", "thread_pool", "serial"),
        default=None,
        help="ONP queue execution transport; thread_pool is explicit opt-in.",
    )
    refresh_onp_parser.add_argument("--json", action="store_true")
