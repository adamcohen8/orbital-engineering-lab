"""Run the synthetic release checks against an installed OEL runtime."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "test_public_precision_forces.py",
    "test_rust_precision_force_plan.py",
    "test_rust_schwarzschild_prepared.py",
    "test_rust_ocean_tides.py",
    "test_rust_earth_radiation_optimization.py",
    "test_orbit_atmosphere_models.py::TestOrbitAtmosphereModels::test_harris_priester_density_is_local_and_f107_sensitive",
    "test_orbit_atmosphere_models.py::TestOrbitAtmosphereModels::test_harris_priester_cached_coefficient_paths_are_isolated",
    "test_orbit_atmosphere_models.py::TestOrbitAtmosphereModels::test_harris_priester_uses_wgs84_geodetic_height_when_requested",
    "test_ccsds_tdm_tracking_od.py",
    "test_pro_tracking_od.py",
    "test_relative_od.py",
    "test_continuous_tracking_od.py",
    "test_ecom.py",
    "test_precision_od.py",
    "test_precision_covariance.py",
    "test_pro_slr_od.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path, help="Write a local machine-readable test report.")
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("Use python -I with the installed OEL environment.")

    try:
        import oel_rust_orbit as native
        import pytest

        import sim
    except ImportError as exc:
        parser.error(f"Install the public OEL runtime and pytest first: {exc}")

    prefix = Path(sys.prefix).resolve()
    for module in (sim, native):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix) or "site-packages" not in path.parts:
            parser.error("OEL and its native extension must come from the installed runtime.")
    if find_spec("sim.licensing") is not None or find_spec("sim.intent_hypothesis") is not None:
        parser.error("Use a clean public OEL install.")
    for capability in ("ONP_PRECISION_FORCE_PLAN_VERSION", "ONP_ECOM2_VERSION", "SLR_EARTH_SHAPIRO_VERSION"):
        if getattr(native, capability, 0) < 1:
            parser.error(f"The installed native wheel lacks {capability}.")

    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    options = ["-q", "--import-mode=importlib", "-p", "no:cacheprovider", "-o", "addopts="]
    if args.junitxml:
        options.append(f"--junitxml={args.junitxml.resolve()}")
    return int(pytest.main([*options, *(str(ROOT / "sim/tests" / test) for test in TESTS)]))


if __name__ == "__main__":
    raise SystemExit(main())
