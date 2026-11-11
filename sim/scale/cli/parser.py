from __future__ import annotations

import argparse

from sim.scale.cli.parsers.campaigns import register_campaigns_parsers
from sim.scale.cli.parsers.catalog import register_catalog_parsers
from sim.scale.cli.parsers.observations import register_observations_parsers
from sim.scale.cli.parsers.operations import register_operations_parsers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded local OEL Scale workflows.")
    sub = parser.add_subparsers(dest="command", required=True)
    register_catalog_parsers(sub)
    register_observations_parsers(sub)
    register_operations_parsers(sub)
    register_campaigns_parsers(sub)
    return parser
