from __future__ import annotations

import argparse


def register_catalog_parsers(sub: argparse._SubParsersAction) -> None:
    dry_run_parser = sub.add_parser("dry-run", help="Write an OEL Scale planner dry-run without propagation.")
    dry_run_parser.add_argument("config", help="OEL Scale YAML config.")
    dry_run_parser.add_argument("--output-dir", help="Override dry-run artifact output directory.")
    dry_run_parser.add_argument("--json", action="store_true", help="Print the dry-run payload as JSON.")

    validation_parser = sub.add_parser(
        "validate-scale",
        help="Run a named synthetic Scale correctness and performance validation tier.",
    )
    validation_parser.add_argument("config", help="Scale validation YAML config.")
    validation_parser.add_argument("--json", action="store_true", help="Print the evidence packet as JSON.")

    ingest_tle_parser = sub.add_parser("ingest-tle", help="Ingest normalized TLE records into an OEL Scale store.")
    ingest_tle_parser.add_argument("tle_file", help="TLE file in two-line or name-plus-two-line format.")
    ingest_tle_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    ingest_tle_parser.add_argument("--source-label", default="", help="Source label for the ingest batch.")
    ingest_tle_parser.add_argument("--object-id-prefix", default="norad", help="Prefix for generated object IDs.")
    ingest_tle_parser.add_argument("--require-checksum", action="store_true", help="Require valid TLE checksums.")
    ingest_tle_parser.add_argument("--json", action="store_true", help="Print the ingest payload as JSON.")

    ingest_ogp_parser = sub.add_parser(
        "ingest-ogp-product",
        help="Ingest an accepted OGP mean-element Product Envelope into Scale without constructing TLE text.",
    )
    ingest_ogp_parser.add_argument("product", help="OGP mean-element Product Envelope JSON.")
    ingest_ogp_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    ingest_ogp_parser.add_argument("--object-id", help="Optional exact identity assertion; it may not relabel the product.")
    ingest_ogp_parser.add_argument("--json", action="store_true")
