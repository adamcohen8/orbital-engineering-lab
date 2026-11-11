"""Content-bound P.676 arithmetic adapter; applicability and ledgers stay in Python."""

from functools import lru_cache
from importlib import import_module

import numpy as np


@lru_cache(maxsize=2)
def _context(oxygen: bytes, water: bytes):
    try:
        cls = import_module("oel_rust_orbit").P676Context
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Rust P.676 requires an oel_rust_orbit wheel with P676Context") from exc
    return cls(oxygen, water)


@lru_cache(maxsize=8)
def _prepared_specific(context, conditions: bytes):
    """Bound reuse to the native table owner and all four physical inputs."""
    return tuple(np.frombuffer(context.specific_batch(conditions), dtype="<f8"))


def _constant_values(values):
    if not values[0].size:
        return None
    scalars = tuple(float(value.flat[0]) for value in values)
    if all(np.all(value == scalar) for value, scalar in zip(values, scalars)):
        return scalars
    return None


def specific_batch(frequency, pressure, temperature, density, oxygen, water):
    context = _context(np.asarray(oxygen, dtype="<f8").tobytes(),
                       np.asarray(water, dtype="<f8").tobytes())
    shape = frequency.shape
    if frequency.size > 1_000_000:
        raise ValueError("invalid bounded P.676 case batch")
    constants = _constant_values((frequency, pressure, temperature, density))
    if constants is not None:
        # Constructing the content-bound context remains outside the value cache,
        # so an unavailable native owner cannot be hidden by a warmed value.
        result = _prepared_specific(context, np.asarray(constants, dtype="<f8").tobytes())
        return tuple(np.full(shape, value) for value in result)
    cases = np.column_stack([value.reshape(-1) for value in
                             (frequency, pressure, temperature, density)]).astype("<f8", copy=False)
    raw = context.specific_batch(cases.tobytes())
    values = np.frombuffer(raw, dtype="<f8").copy().reshape(-1, 2)
    return values[:, 0].reshape(shape), values[:, 1].reshape(shape)


def slant_batch(frequency, elevation, pressure, temperature, density, specific, coefficients, oxygen, water):
    context = _context(np.asarray(oxygen, dtype="<f8").tobytes(),
                       np.asarray(water, dtype="<f8").tobytes())
    shape = frequency.shape
    constants = _constant_values((frequency, pressure, temperature, density,
                                 specific.oxygen_db_per_km, specific.water_vapour_db_per_km,
                                 *(coefficients[..., index] for index in range(4))))
    prepared = getattr(context, "slant_prepared", None)
    if constants is not None and prepared is not None:
        raw = prepared(np.asarray(constants, dtype="<f8").tobytes(),
                       np.asarray(elevation, dtype="<f8").tobytes())
        values = np.frombuffer(raw, dtype="<f8").copy().reshape(-1, 4)
        return tuple(values[:, index].reshape(shape) for index in range(4))
    cases = np.column_stack([value.reshape(-1) for value in (
        frequency, elevation, pressure, temperature, density,
        specific.oxygen_db_per_km, specific.water_vapour_db_per_km,
        *(coefficients[..., index] for index in range(4)),
    )]).astype("<f8", copy=False)
    raw = context.slant_batch(cases.tobytes())
    values = np.frombuffer(raw, dtype="<f8").copy().reshape(-1, 4)
    return tuple(values[:, index].reshape(shape) for index in range(4))
