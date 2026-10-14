"""Export the public numerical crate without private orchestration modules."""

from __future__ import annotations

import re
from pathlib import Path

PUBLIC_ORBIT_MODULES = frozenset(
    {
        "actuation",
        "adaptive",
        "attitude",
        "attitude_ekf",
        "cr3bp",
        "control",
        "collisions",
        "communications",
        "conjunction",
        "conjunction_probability",
        "covariance",
        "covariance_orbit",
        "coverage",
        "estimation",
        "environment",
        "force_context",
        "stage_context",
        "analytic_bodies",
        "force_plan",
        "planetary",
        "additional_gravity",
        "additional_gravity_binding",
        "planetary_binding",
        "harmonics",
        "iod",
        "links",
        "measurements",
        "optical_collection",
        "mission_schedule",
        "mechanical",
        "nrlmsise00_coeff",
        "nrlmsise00_model",
        "nrlmsise00_native",
        "nrlmsise00_stage",
        "numeric_bytes",
        "ogp",
        "output",
        "relative",
        "resources",
        "spacecraft",
        "targeting",
        "target_coast",
        "target_rollouts",
        "tasking_search",
        "passive",
        "perturbations",
        "precision_forces",
        "precision_plan",
        "solid_tides",
        "ocean_tides",
        "zonal",
        "tracking",
        "vehicle",
    }
)


def patch_public_native_core(destination: Path) -> None:
    """Keep authoritative source bodies unchanged; trim only export wiring."""
    path = destination / "rust/oel-orbit/src/lib.rs"
    if not path.exists():
        return  # Minimal exporter fixtures; canonical integrity requires it.
    text = path.read_text(encoding="utf-8")
    header, body = text.split("const EARTH_RADIUS_KM:", 1)
    found = set()

    def declaration(match):
        name = match.group(1)
        if name in PUBLIC_ORBIT_MODULES:
            found.add(name)
            return match.group(0)
        return ""

    header = re.sub(r"(?:#\[[^\n]*\]\n)*(?:pub )?mod (\w+);\n", declaration, header)
    if found != PUBLIC_ORBIT_MODULES:
        raise ValueError(f"Public native module inventory drift: {sorted(PUBLIC_ORBIT_MODULES - found)}")
    # Sampled-pair screening belongs to the separate non-public workflow.
    start = '    #[cfg(feature = "pro")]\n    #[pyfunction]\n    fn screen_sampled_pairs_bytes('
    end = "    fn parameters("
    if start in body:
        if body.count(start) != 1 or end not in body.split(start, 1)[1]:
            raise ValueError("Public native screening boundary drift")
        first, rest = body.split(start, 1)
        _, last = rest.split(end, 1)
        body = first + end + last
    elif "screen_sampled_pairs" in body:
        raise ValueError("Public native screening boundary drift")
    body = body.replace('    #[cfg(feature = "pro")]\n    use super::screening;\n', "")
    body = re.sub(
        r"^(?:        #\[[^\n]*\]\n)*        super::(\w+)::register(?:_py)?\(module\)\?;\n",
        lambda m: m.group(0) if m.group(1) in PUBLIC_ORBIT_MODULES else "",
        body,
        flags=re.MULTILINE,
    )
    body = re.sub(
        r"^(?:        #\[[^\n]*\]\n)*        module.add_function\(wrap_pyfunction!\(screen_sampled_pairs(?:_bytes)?, module\)\?\)\?;\n",
        "",
        body,
        flags=re.MULTILINE,
    )
    path.write_text(header + "const EARTH_RADIUS_KM:" + body, encoding="utf-8")
    validation = destination / "sim/config/scenario/validation.py"
    if validation.exists():
        text = validation.read_text(encoding="utf-8")
        # Covariance campaigns remain private. Disabled campaign defaults must
        # not require their absent native implementation in the public core.
        text = text.replace(
            '    if covariance_backend == "rust":\n',
            '    if cfg.analysis.enabled and cfg.analysis.covariance.enabled and covariance_backend == "rust":\n',
        )
        validation.write_text(text, encoding="utf-8")
