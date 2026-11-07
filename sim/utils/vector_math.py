"""Fixed-size scalar reductions shared by Python numeric reference owners."""

from __future__ import annotations

import math


def dot3(left, right) -> float:
    """Use the native three-vector operation order, independent of BLAS/FMA."""
    a, b, c = map(float, left)
    x, y, z = map(float, right)
    return a * x + b * y + c * z


def norm3(value) -> float:
    return math.sqrt(dot3(value, value))
