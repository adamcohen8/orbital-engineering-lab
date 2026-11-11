"""Limit nested native math threads without changing process-global settings.

Accelerate on macOS 15+ exposes a per-thread BLAS/LAPACK threading setting.
Other math libraries retain their startup configuration; this helper never
changes their process-global runtime settings while simulations are running.
"""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator


@lru_cache(maxsize=1)
def _accelerate_api():
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        getter = library.BLASGetThreading
        setter = library.BLASSetThreading
    except (OSError, AttributeError):
        return None
    getter.argtypes = []
    getter.restype = ctypes.c_uint
    setter.argtypes = [ctypes.c_uint]
    setter.restype = ctypes.c_int
    return library, getter, setter


@contextmanager
def thread_math_limits() -> Iterator[None]:
    """Use one Accelerate math thread for this worker, then restore its setting."""

    api = _accelerate_api()
    if api is None:
        yield
        return
    _, getter, setter = api
    previous = getter()
    # Apple's thread_api.h defines SINGLE_THREADED as 1, MULTI_THREADED as 0.
    if setter(1) != 0:
        raise RuntimeError("Accelerate rejected the worker's single-thread math setting.")
    try:
        yield
    finally:
        if setter(previous) != 0:
            raise RuntimeError("Accelerate could not restore the worker's math setting.")
