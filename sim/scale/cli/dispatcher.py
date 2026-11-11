from __future__ import annotations

import argparse
import sys

from sim.scale.cli.registry import COMMAND_HANDLERS


def dispatch(args: argparse.Namespace) -> int:
    try:
        handler = COMMAND_HANDLERS.get(str(args.command))
        if handler is None:
            return 2
        return int(handler(args))
    except Exception as exc:
        print(f"OEL Scale command failed: {exc}", file=sys.stderr)
        return 2
