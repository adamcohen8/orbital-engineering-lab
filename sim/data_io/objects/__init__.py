"""Stable shared object IO facade; focused owners implement each operation."""

from .composition import compose_object, create_scenario, write_composed_scenario
from .continuation import materialize_continuation
from .contracts import ImportedObject
from .discovery import describe_export, describe_import
from .exports import export_object
from .importers import import_object

__all__ = [
    "ImportedObject",
    "import_object",
    "compose_object",
    "create_scenario",
    "write_composed_scenario",
    "export_object",
    "describe_import",
    "describe_export",
    "materialize_continuation",
]
