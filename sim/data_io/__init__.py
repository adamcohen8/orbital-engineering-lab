"""Modular, offline data interchange and completed-evidence post-processing.

Readers/writers never start simulations. Derived products live in separate
operations and consume saved evidence, never live simulator internals.
"""

from .contracts import AttitudeHistory, DataIOError, Trajectory

__all__ = ["AttitudeHistory", "DataIOError", "Trajectory"]
