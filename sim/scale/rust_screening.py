"""Optional native numeric pair discovery for Scale's sampled-distance screen."""

from __future__ import annotations

import math
import sys
from array import array
from typing import TYPE_CHECKING

from sim.rust_orbit_backend import _extension

if TYPE_CHECKING:
    from sim.scale.store_models import PropagationProductSummary, PropagationSampleRecord


def rust_spatial_candidate_pairs(
    products: list[PropagationProductSummary],
    *,
    samples_by_product: dict[str, list[PropagationSampleRecord]],
    distance_threshold_km: float,
    distance_comparison: str = "exact_sqrt",
) -> list[tuple[PropagationProductSummary, PropagationProductSummary, tuple[float, float, float]]]:
    """Return the Python screening contract from one numeric Rust batch call."""

    if distance_comparison not in {"exact_sqrt", "squared_distance"}:
        raise ValueError("distance_comparison must be exact_sqrt or squared_distance.")
    ordered = sorted(products, key=lambda item: item.propagation_product_id)
    index_by_id = {item.propagation_product_id: index for index, item in enumerate(ordered)}
    object_indices: list[int] = []
    sample_indices: list[int] = []
    states = array("d")
    for product_id, samples in samples_by_product.items():
        object_index = index_by_id.get(product_id)
        if object_index is None:
            continue
        for sample in samples:
            if sample.error:
                continue
            x, y, z = sample.pos_x_km, sample.pos_y_km, sample.pos_z_km
            if x is None or y is None or z is None:
                continue
            x, y, z = float(x), float(y), float(z)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            vx, vy, vz = sample.vel_x_km_s, sample.vel_y_km_s, sample.vel_z_km_s
            if vx is None or vy is None or vz is None:
                continue
            jd_utc = float(sample.jd_utc)
            vx, vy, vz = float(vx), float(vy), float(vz)
            if not all(math.isfinite(value) for value in (jd_utc, vx, vy, vz)):
                continue
            object_indices.append(object_index)
            sample_indices.append(int(sample.sample_index))
            states.extend((jd_utc, x, y, z, vx, vy, vz))
    native = _extension()
    if not hasattr(native, "screen_sampled_pairs"):
        raise RuntimeError("Scale Rust screening requires an oel_rust_orbit wheel with screen_sampled_pairs.")
    byte_function = getattr(native, "screen_sampled_pairs_bytes", None)
    if byte_function is not None:
        if sys.byteorder != "little":
            states.byteswap()
        hits = byte_function(
            object_indices, sample_indices, states.tobytes(),
            float(distance_threshold_km), distance_comparison == "squared_distance",
        )
    else:
        hits = native.screen_sampled_pairs(
            object_indices, sample_indices, states.tolist(),
            float(distance_threshold_km), distance_comparison == "squared_distance",
        )
    return [
        (ordered[left], ordered[right], (distance, jd_utc, relative_speed))
        for left, right, distance, jd_utc, relative_speed in hits
    ]
