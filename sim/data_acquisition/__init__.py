"""Optional local-first acquisition. Importing this package never uses the network."""
from .contracts import AcquisitionError, Request
from .service import AcquisitionService

__all__ = ["AcquisitionError", "AcquisitionService", "Request"]
