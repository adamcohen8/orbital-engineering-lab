"""Registry-driven OEL Scale command-line implementation."""

from sim.scale.cli.dispatcher import dispatch
from sim.scale.cli.parser import build_parser
from sim.scale.cli.registry import COMMAND_HANDLERS

__all__ = ["COMMAND_HANDLERS", "build_parser", "dispatch"]
