"""OEL Flight Software Development and Verification Kit."""

from .candidate import (
    CandidateValidationError,
    inspect_candidate,
    load_candidate,
    validate_candidate,
)
from .contracts import (
    FSWDK_CONTRACT_VERSION,
    FlightSoftwareCandidate,
    FSWDKIssue,
    FSWDKReceipt,
)
from .external import run_external_smoke, validate_external_compatibility

__all__ = [
    "CandidateValidationError",
    "FSWDK_CONTRACT_VERSION",
    "FSWDKIssue",
    "FSWDKReceipt",
    "FlightSoftwareCandidate",
    "inspect_candidate",
    "load_candidate",
    "run_external_smoke",
    "validate_candidate",
    "validate_external_compatibility",
]
