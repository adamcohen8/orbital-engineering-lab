"""Immutable body identities; availability is distinct from qualification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BodyDefinition:
    body_id: str
    center_id: int
    mu_km3_s2: float
    reference_radius_km: float
    solid_surface: bool = True
    constants_source: str = "NAIF gm_de440.tpc physical-center GM; IAU reference radii"


# The harmonic dataset supplies its own GM. These are point-mass defaults.
BODIES = {
    "mars": BodyDefinition("mars", 499, 42828.37362069909, 3396.19),
    "moon": BodyDefinition("moon", 301, 4902.80011845755, 1737.4),
    "sun": BodyDefinition("sun", 10, 132712440041.27942, 695700.0, False),
    "mercury": BodyDefinition("mercury", 199, 22031.868551400003, 2439.7),
    "venus": BodyDefinition("venus", 299, 324858.592, 6051.8),
    "jupiter": BodyDefinition("jupiter", 599, 126686531.9003704, 71492.0, False),
    "saturn": BodyDefinition("saturn", 699, 37931206.23436167, 60268.0, False),
    "uranus": BodyDefinition("uranus", 799, 5793951.256527211, 25559.0, False),
    "neptune": BodyDefinition("neptune", 899, 6835103.145462294, 24764.0, False),
}


def body_definition(body_id: str) -> BodyDefinition:
    try:
        return BODIES[body_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported planetary central_body {body_id!r}; choose {', '.join(BODIES)}."
        ) from exc
