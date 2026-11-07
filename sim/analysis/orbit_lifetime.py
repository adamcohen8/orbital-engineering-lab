"""Bounded deterministic ONP orbit-decay and lifetime analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.rust_numeric import numeric_backend as validate_numeric_backend
from sim.analysis.rust_numeric import require_native
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin, j2_plugin
from sim.utils.io import SafeReadError, read_regular_file_nofollow
from sim.utils.vector_math import dot3, norm3

ORBIT_LIFETIME_PROBLEM_SCHEMA = "oel.orbit_lifetime_problem.v1"
ORBIT_LIFETIME_EVIDENCE_SCHEMA = "oel.orbit_lifetime_evidence.v1"
ORBIT_LIFETIME_MANIFEST_SCHEMA = "oel.orbit_lifetime_manifest.v1"
ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA = "oel.orbit_lifetime_comparison_problem.v1"
ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA = "oel.orbit_lifetime_comparison_evidence.v1"
ORBIT_LIFETIME_COMPARISON_MANIFEST_SCHEMA = "oel.orbit_lifetime_comparison_manifest.v1"

MAX_LIFETIME_DURATION_S = 90.0 * 86400.0
MAX_LIFETIME_INTEGRATION_STEPS = 500_000
MAX_LIFETIME_OUTPUT_SAMPLES = 200_000
MAX_LIFETIME_COMPARISON_CASES = 8
MAX_LIFETIME_JSON_BYTES = 16 * 1024 * 1024
MAX_LIFETIME_ARTIFACT_BYTES = 128 * 1024 * 1024
MIN_LIFETIME_EPOCH_JD_UTC = 1721425.5
MAX_LIFETIME_EPOCH_JD_UTC = 5373393.5
HARRIS_PRIESTER_SUPPORTED_F107 = (65.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0)
_DIGEST_CHARACTERS = set("0123456789abcdef")
_CASE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_FILES = (
    "sim/analysis/orbit_lifetime.py",
    "sim/dynamics/orbit/accelerations.py",
    "sim/dynamics/orbit/atmosphere.py",
    "sim/dynamics/orbit/elements.py",
    "sim/dynamics/orbit/environment.py",
    "sim/dynamics/orbit/epoch.py",
    "sim/dynamics/orbit/frames.py",
    "sim/dynamics/orbit/propagator.py",
    "sim/utils/vector_math.py",
)
_ATMOSPHERE_IMPLEMENTATION_FILES = {
    "harris_priester": (
        "sim/dynamics/orbit/harris_priester_backend.py",
        "sim/dynamics/orbit/data/harris_priester_hpop.csv",
    ),
    "nrlmsise00": (
        "sim/dynamics/orbit/nrlmsise00_backend.py",
        "sim/dynamics/orbit/nrlmsise00_coeff.py",
    ),
}
_SINGLE_FILES = {
    "normalized_problem.json",
    "orbit_lifetime_summary.json",
    "orbit_lifetime_timeseries.csv",
    "orbit_lifetime_events.csv",
}
_COMPARISON_FILES = {
    "normalized_comparison.json",
    "orbit_lifetime_comparison_summary.json",
    "orbit_lifetime_comparison.csv",
}


class OrbitLifetimeError(ValueError):
    """Raised when a lifetime input or retained evidence record is invalid."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrbitLifetimeError(f"{field} must be a non-empty string.")
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise OrbitLifetimeError(f"{field} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OrbitLifetimeError(f"{field} must be a finite number.") from exc
    if not math.isfinite(result):
        raise OrbitLifetimeError(f"{field} must be a finite number.")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise OrbitLifetimeError(f"{field} must be an integer.")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise OrbitLifetimeError(f"{field} must be a boolean.")
    return bool(value)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OrbitLifetimeError(f"{field} must be a JSON object.")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: set[str], field: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        raise OrbitLifetimeError(f"{field} is missing required fields: {', '.join(missing)}.")
    if unknown:
        raise OrbitLifetimeError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _vector(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise OrbitLifetimeError(f"{field} must contain exactly {length} finite numbers.")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _digest(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def _valid_digest(value: Any, field: str) -> str:
    result = _text(value, field).lower()
    if len(result) != 64 or any(character not in _DIGEST_CHARACTERS for character in result):
        raise OrbitLifetimeError(f"{field} must be a lowercase SHA-256 digest.")
    return result


def _file_identity(relative_path: str) -> dict[str, Any]:
    path = _PROJECT_ROOT / relative_path
    content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_LIFETIME_ARTIFACT_BYTES)
    return {
        "path": relative_path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _implementation_identity(atmosphere_models: Sequence[str], numeric_backend: str = "rust") -> dict[str, Any]:
    paths = set(_IMPLEMENTATION_FILES)
    paths.add("sim/analysis/rust_numeric.py")
    if numeric_backend == "rust":
        native_sources = _PROJECT_ROOT / "rust/oel-orbit"
        if (native_sources / "Cargo.toml").is_file():
            paths.update(str(path.relative_to(_PROJECT_ROOT)) for path in (native_sources / "src").rglob("*.rs"))
            paths.update(("rust/oel-orbit/Cargo.toml", "rust/oel-orbit/Cargo.lock"))
        paths.update(("sim/dynamics/orbit/rust_force_plan.py", "sim/dynamics/orbit/rust_stage_context.py",
                      "sim/rust_environment_backend.py"))
    for model in atmosphere_models:
        paths.update(_ATMOSPHERE_IMPLEMENTATION_FILES.get(model, ()))
    files = [_file_identity(path) for path in sorted(paths)]
    identity = {
        "algorithm_id": "oel.orbit_lifetime.v1",
        "source_tree_sha256": _digest(files),
        "files": files,
    }
    if numeric_backend == "rust":
        native = require_native("ONPForceContext.rk4")
        from sim.rust_orbit_backend import native_extension_path

        content = read_regular_file_nofollow(
            native_extension_path(), min_bytes=1, max_bytes=64 * 1024 * 1024
        )
        identity["native_runtime"] = {
            "module": "oel_rust_orbit", "version": str(native.__version__),
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        }
    return identity


@dataclass(frozen=True)
class LifetimeAtmosphere:
    model: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LifetimeAtmosphere:
        raw = _mapping(value, "atmosphere")
        _exact(raw, {"model", "parameters"}, "atmosphere")
        return cls(model=_text(raw["model"], "atmosphere.model").lower(), parameters=raw["parameters"])

    def __post_init__(self) -> None:
        model = _text(self.model, "atmosphere.model").lower()
        aliases = {"harris-priester": "harris_priester", "hp": "harris_priester"}
        model = aliases.get(model, model)
        allowed = {"constant", "exponential", "ussa1976", "nrlmsise00", "harris_priester"}
        if model not in allowed:
            raise OrbitLifetimeError(
                "atmosphere.model must be constant, exponential, ussa1976, nrlmsise00, or harris_priester."
            )
        raw = _mapping(self.parameters, "atmosphere.parameters")
        expected = {
            "constant": {"density_kg_m3"},
            "exponential": {
                "reference_density_kg_m3",
                "reference_altitude_km",
                "scale_height_km",
                "ceiling_altitude_km",
            },
            "ussa1976": set(),
            "nrlmsise00": {"f107", "f107a", "ap", "ap_a"},
            "harris_priester": {"f107"},
        }[model]
        _exact(raw, expected, "atmosphere.parameters")
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "ap_a":
                normalized[key] = list(_vector(value, 7, "atmosphere.parameters.ap_a"))
            else:
                normalized[key] = _finite(value, f"atmosphere.parameters.{key}")
        positive = {
            "density_kg_m3",
            "reference_density_kg_m3",
            "scale_height_km",
            "ceiling_altitude_km",
            "f107",
            "f107a",
        }
        for key in positive & set(normalized):
            if normalized[key] <= 0.0:
                raise OrbitLifetimeError(f"atmosphere.parameters.{key} must be positive.")
        if model == "harris_priester" and normalized["f107"] not in HARRIS_PRIESTER_SUPPORTED_F107:
            supported = ", ".join(f"{value:g}" for value in HARRIS_PRIESTER_SUPPORTED_F107)
            raise OrbitLifetimeError(
                f"atmosphere.parameters.f107 must select a supported Harris-Priester table: {supported}."
            )
        if "reference_altitude_km" in normalized and normalized["reference_altitude_km"] < 0.0:
            raise OrbitLifetimeError("atmosphere.parameters.reference_altitude_km must be nonnegative.")
        if "ap" in normalized and normalized["ap"] < 0.0:
            raise OrbitLifetimeError("atmosphere.parameters.ap must be nonnegative.")
        if "ap_a" in normalized and any(value < 0.0 for value in normalized["ap_a"]):
            raise OrbitLifetimeError("atmosphere.parameters.ap_a values must be nonnegative.")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "parameters", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "parameters": dict(self.parameters)}

    def altitude_domain(self) -> dict[str, Any]:
        domains: dict[str, tuple[float, bool, float | None, bool]] = {
            "constant": (0.0, True, None, True),
            "exponential": (0.0, True, None, True),
            "ussa1976": (0.0, True, 1000.0, True),
            "nrlmsise00": (0.0, True, 1000.0, True),
            "harris_priester": (110.0, False, 2000.0, False),
        }
        minimum, minimum_inclusive, maximum, maximum_inclusive = domains[self.model]
        if self.model == "exponential":
            maximum = float(self.parameters["ceiling_altitude_km"])
        return {
            "minimum_altitude_km": minimum,
            "minimum_inclusive": minimum_inclusive,
            "maximum_altitude_km": maximum,
            "maximum_inclusive": maximum_inclusive,
        }

    def effective_record(self) -> dict[str, Any]:
        return {
            "normalized_atmosphere": self.to_dict(),
            "altitude_domain": self.altitude_domain(),
            "harris_priester_f107_table": (
                self.parameters["f107"] if self.model == "harris_priester" else None
            ),
        }

    def environment(self, epoch_jd_utc: float) -> dict[str, Any]:
        env: dict[str, Any] = {
            "jd_utc_start": epoch_jd_utc,
            "drag_frame_model": "simple",
            "density_frame_model": "simple",
            "geodetic_model": "wgs84",
            "ephemeris_mode": "analytic_enhanced",
        }
        if self.model == "constant":
            env["density_kg_m3"] = float(self.parameters["density_kg_m3"])
            return env
        env["atmosphere_model"] = self.model
        if self.model == "exponential":
            env.update(
                {
                    "exponential_reference_density_kg_m3": self.parameters["reference_density_kg_m3"],
                    "exponential_reference_altitude_km": self.parameters["reference_altitude_km"],
                    "exponential_scale_height_km": self.parameters["scale_height_km"],
                    "exponential_ceiling_altitude_km": self.parameters["ceiling_altitude_km"],
                }
            )
        elif self.model == "nrlmsise00":
            env.update(
                {
                    "f107": self.parameters["f107"],
                    "f107a": self.parameters["f107a"],
                    "ap": self.parameters["ap"],
                    "nrlmsise00_ap_a": list(self.parameters["ap_a"]),
                }
            )
        elif self.model == "harris_priester":
            env["harris_priester_f107"] = self.parameters["f107"]
        return env

    def density(self, state: np.ndarray, t_s: float, epoch_jd_utc: float) -> float:
        try:
            env = self.environment(epoch_jd_utc)
            if self.model == "constant":
                return float(env["density_kg_m3"])
            return float(density_from_model(self.model, state[:3], t_s, env=env))
        except OrbitLifetimeError:
            raise
        except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
            raise OrbitLifetimeError(
                f"Atmosphere model {self.model!r} failed at elapsed time {float(t_s):.9g} s: {exc}"
            ) from exc


@dataclass(frozen=True)
class LifetimeThresholds:
    warning_altitude_km: float
    disposal_altitude_km: float
    reentry_altitude_km: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LifetimeThresholds:
        raw = _mapping(value, "thresholds")
        _exact(
            raw,
            {"warning_altitude_km", "disposal_altitude_km", "reentry_altitude_km"},
            "thresholds",
        )
        return cls(**{key: _finite(raw[key], f"thresholds.{key}") for key in raw})

    def __post_init__(self) -> None:
        for field in ("warning_altitude_km", "disposal_altitude_km", "reentry_altitude_km"):
            object.__setattr__(self, field, _finite(getattr(self, field), f"thresholds.{field}"))
        if not (
            80.0 <= self.reentry_altitude_km
            < self.disposal_altitude_km
            < self.warning_altitude_km
            <= 2000.0
        ):
            raise OrbitLifetimeError(
                "threshold altitudes must satisfy 80 <= reentry < disposal < warning <= 2000 km."
            )

    def items(self) -> tuple[tuple[str, float], ...]:
        return (
            ("warning", self.warning_altitude_km),
            ("disposal", self.disposal_altitude_km),
            ("reentry", self.reentry_altitude_km),
        )


@dataclass(frozen=True)
class OrbitLifetimeProblem:
    analysis_id: str
    asset_id: str
    epoch_jd_utc: float
    initial_position_eci_km: tuple[float, float, float]
    initial_velocity_eci_km_s: tuple[float, float, float]
    duration_s: float
    integration_step_s: float
    output_step_s: float
    transition_time_tolerance_s: float
    transition_max_iterations: int
    mass_kg: float
    drag_area_m2: float
    drag_coefficient: float
    drag_enabled: bool
    include_j2: bool
    stop_at_reentry: bool
    atmosphere: LifetimeAtmosphere
    thresholds: LifetimeThresholds
    schema_version: str = ORBIT_LIFETIME_PROBLEM_SCHEMA

    numeric_backend: str = "rust"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OrbitLifetimeProblem:
        raw = _mapping(value, "orbit-lifetime problem")
        fields = {
            "schema_version",
            "analysis_id",
            "asset_id",
            "epoch_jd_utc",
            "initial_position_eci_km",
            "initial_velocity_eci_km_s",
            "duration_s",
            "integration_step_s",
            "output_step_s",
            "transition_time_tolerance_s",
            "transition_max_iterations",
            "mass_kg",
            "drag_area_m2",
            "drag_coefficient",
            "drag_enabled",
            "include_j2",
            "stop_at_reentry",
            "atmosphere",
            "thresholds",
        }
        backend = raw.pop("numeric_backend", "rust")
        _exact(raw, fields, "orbit-lifetime problem")
        if raw["schema_version"] != ORBIT_LIFETIME_PROBLEM_SCHEMA:
            raise OrbitLifetimeError(f"Unsupported orbit-lifetime schema {raw['schema_version']!r}.")
        return cls(
            numeric_backend=backend,
            schema_version=raw["schema_version"],
            analysis_id=_text(raw["analysis_id"], "analysis_id"),
            asset_id=_text(raw["asset_id"], "asset_id"),
            epoch_jd_utc=_finite(raw["epoch_jd_utc"], "epoch_jd_utc"),
            initial_position_eci_km=_vector(raw["initial_position_eci_km"], 3, "initial_position_eci_km"),
            initial_velocity_eci_km_s=_vector(
                raw["initial_velocity_eci_km_s"], 3, "initial_velocity_eci_km_s"
            ),
            duration_s=_finite(raw["duration_s"], "duration_s"),
            integration_step_s=_finite(raw["integration_step_s"], "integration_step_s"),
            output_step_s=_finite(raw["output_step_s"], "output_step_s"),
            transition_time_tolerance_s=_finite(
                raw["transition_time_tolerance_s"], "transition_time_tolerance_s"
            ),
            transition_max_iterations=_integer(raw["transition_max_iterations"], "transition_max_iterations"),
            mass_kg=_finite(raw["mass_kg"], "mass_kg"),
            drag_area_m2=_finite(raw["drag_area_m2"], "drag_area_m2"),
            drag_coefficient=_finite(raw["drag_coefficient"], "drag_coefficient"),
            drag_enabled=_boolean(raw["drag_enabled"], "drag_enabled"),
            include_j2=_boolean(raw["include_j2"], "include_j2"),
            stop_at_reentry=_boolean(raw["stop_at_reentry"], "stop_at_reentry"),
            atmosphere=LifetimeAtmosphere.from_mapping(raw["atmosphere"]),
            thresholds=LifetimeThresholds.from_mapping(raw["thresholds"]),
        )

    def __post_init__(self) -> None:
        try:
            validate_numeric_backend(self.numeric_backend)
        except ValueError as exc:
            raise OrbitLifetimeError(str(exc)) from exc
        if self.schema_version != ORBIT_LIFETIME_PROBLEM_SCHEMA:
            raise OrbitLifetimeError(f"Unsupported orbit-lifetime schema {self.schema_version!r}.")
        object.__setattr__(self, "analysis_id", _text(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "asset_id", _text(self.asset_id, "asset_id"))
        for field in (
            "epoch_jd_utc",
            "duration_s",
            "integration_step_s",
            "output_step_s",
            "transition_time_tolerance_s",
            "mass_kg",
            "drag_area_m2",
            "drag_coefficient",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        object.__setattr__(
            self,
            "initial_position_eci_km",
            _vector(self.initial_position_eci_km, 3, "initial_position_eci_km"),
        )
        object.__setattr__(
            self,
            "initial_velocity_eci_km_s",
            _vector(self.initial_velocity_eci_km_s, 3, "initial_velocity_eci_km_s"),
        )
        _integer(self.transition_max_iterations, "transition_max_iterations")
        _boolean(self.drag_enabled, "drag_enabled")
        _boolean(self.include_j2, "include_j2")
        _boolean(self.stop_at_reentry, "stop_at_reentry")
        if not isinstance(self.atmosphere, LifetimeAtmosphere) or not isinstance(
            self.thresholds, LifetimeThresholds
        ):
            raise OrbitLifetimeError("atmosphere and thresholds must be validated contract values.")
        if not 0.0 < self.duration_s <= MAX_LIFETIME_DURATION_S:
            raise OrbitLifetimeError(
                f"duration_s must be positive and no greater than {MAX_LIFETIME_DURATION_S}."
            )
        if not MIN_LIFETIME_EPOCH_JD_UTC <= self.epoch_jd_utc <= MAX_LIFETIME_EPOCH_JD_UTC:
            raise OrbitLifetimeError(
                "epoch_jd_utc must permit UTC conversion throughout the maximum 90-day lifetime horizon."
            )
        if not 0.0 < self.integration_step_s <= 120.0:
            raise OrbitLifetimeError("integration_step_s must lie in (0, 120].")
        if self.output_step_s < self.integration_step_s:
            raise OrbitLifetimeError("output_step_s must be at least integration_step_s.")
        ratio = self.output_step_s / self.integration_step_s
        if abs(ratio - round(ratio)) > 1.0e-12:
            raise OrbitLifetimeError("output_step_s must be an integer multiple of integration_step_s.")
        integration_count = int(math.ceil(self.duration_s / self.integration_step_s))
        output_count = int(math.ceil(self.duration_s / self.output_step_s)) + 1
        if integration_count > MAX_LIFETIME_INTEGRATION_STEPS:
            raise OrbitLifetimeError(
                f"The lifetime run may not exceed {MAX_LIFETIME_INTEGRATION_STEPS} integration steps."
            )
        if output_count > MAX_LIFETIME_OUTPUT_SAMPLES:
            raise OrbitLifetimeError(
                f"The lifetime run may not exceed {MAX_LIFETIME_OUTPUT_SAMPLES} output samples."
            )
        if not 0.0 < self.transition_time_tolerance_s < self.integration_step_s:
            raise OrbitLifetimeError(
                "transition_time_tolerance_s must be positive and smaller than integration_step_s."
            )
        if self.transition_max_iterations <= 0:
            raise OrbitLifetimeError("transition_max_iterations must be positive.")
        if self.mass_kg <= 0.0 or self.drag_area_m2 <= 0.0 or self.drag_coefficient <= 0.0:
            raise OrbitLifetimeError("mass_kg, drag_area_m2, and drag_coefficient must be positive.")
        state = self.initial_state()
        radius = float(np.linalg.norm(state[:3]))
        if radius <= EARTH_RADIUS_KM + self.thresholds.reentry_altitude_km:
            raise OrbitLifetimeError("The initial state must begin above the declared reentry threshold.")
        try:
            elements = rv_to_coe_eci(state[:3], state[3:])
        except (ArithmeticError, ValueError) as exc:
            raise OrbitLifetimeError(f"The initial state does not define a supported elliptical orbit: {exc}") from exc
        if elements.a_km * (1.0 - elements.ecc) <= EARTH_RADIUS_KM:
            raise OrbitLifetimeError("The initial osculating orbit intersects Earth.")
        domain = self.atmosphere.altitude_domain()
        minimum = float(domain["minimum_altitude_km"])
        reentry = self.thresholds.reentry_altitude_km
        if reentry < minimum or (reentry == minimum and not domain["minimum_inclusive"]):
            qualifier = "above" if not domain["minimum_inclusive"] else "at or above"
            raise OrbitLifetimeError(
                f"thresholds.reentry_altitude_km must be {qualifier} the {self.atmosphere.model} "
                f"atmosphere lower domain limit of {minimum:g} km."
            )
        maximum = domain["maximum_altitude_km"]
        apogee = elements.a_km * (1.0 + elements.ecc) - EARTH_RADIUS_KM
        if maximum is not None and (
            apogee > float(maximum) or (apogee == float(maximum) and not domain["maximum_inclusive"])
        ):
            qualifier = "below" if not domain["maximum_inclusive"] else "at or below"
            raise OrbitLifetimeError(
                f"The initial osculating apogee must be {qualifier} the {self.atmosphere.model} "
                f"atmosphere upper domain limit of {float(maximum):g} km."
            )

    def initial_state(self) -> np.ndarray:
        return np.asarray((*self.initial_position_eci_km, *self.initial_velocity_eci_km_s), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["initial_position_eci_km"] = list(self.initial_position_eci_km)
        value["initial_velocity_eci_km_s"] = list(self.initial_velocity_eci_km_s)
        value["atmosphere"] = self.atmosphere.to_dict()
        return value


@dataclass(frozen=True)
class LifetimeComparisonCase:
    case_id: str
    atmosphere: LifetimeAtmosphere

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LifetimeComparisonCase:
        raw = _mapping(value, "comparison case")
        _exact(raw, {"case_id", "atmosphere"}, "comparison case")
        return cls(
            case_id=_text(raw["case_id"], "comparison case_id"),
            atmosphere=LifetimeAtmosphere.from_mapping(raw["atmosphere"]),
        )

    def __post_init__(self) -> None:
        case_id = _text(self.case_id, "comparison case_id")
        if _CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise OrbitLifetimeError(
                "comparison case_id must match ^[a-z][a-z0-9_-]{0,63}$."
            )
        if not isinstance(self.atmosphere, LifetimeAtmosphere):
            raise OrbitLifetimeError("comparison atmosphere must be validated.")
        object.__setattr__(self, "case_id", case_id)


@dataclass(frozen=True)
class OrbitLifetimeComparisonProblem:
    comparison_id: str
    base_problem: OrbitLifetimeProblem
    cases: tuple[LifetimeComparisonCase, ...]
    schema_version: str = ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OrbitLifetimeComparisonProblem:
        raw = _mapping(value, "orbit-lifetime comparison")
        _exact(raw, {"schema_version", "comparison_id", "base_problem", "cases"}, "orbit-lifetime comparison")
        if raw["schema_version"] != ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA:
            raise OrbitLifetimeError(
                f"Unsupported orbit-lifetime comparison schema {raw['schema_version']!r}."
            )
        cases = raw["cases"]
        if not isinstance(cases, list):
            raise OrbitLifetimeError("comparison cases must be a JSON array.")
        return cls(
            schema_version=raw["schema_version"],
            comparison_id=_text(raw["comparison_id"], "comparison_id"),
            base_problem=OrbitLifetimeProblem.from_mapping(raw["base_problem"]),
            cases=tuple(LifetimeComparisonCase.from_mapping(item) for item in cases),
        )

    def __post_init__(self) -> None:
        if self.schema_version != ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA:
            raise OrbitLifetimeError(
                f"Unsupported orbit-lifetime comparison schema {self.schema_version!r}."
            )
        object.__setattr__(self, "comparison_id", _text(self.comparison_id, "comparison_id"))
        if not isinstance(self.base_problem, OrbitLifetimeProblem):
            raise OrbitLifetimeError("base_problem must be a validated OrbitLifetimeProblem.")
        try:
            cases = tuple(self.cases)
        except TypeError as exc:
            raise OrbitLifetimeError("cases must be an iterable of validated cases.") from exc
        if not 2 <= len(cases) <= MAX_LIFETIME_COMPARISON_CASES:
            raise OrbitLifetimeError(
                f"A comparison must contain between 2 and {MAX_LIFETIME_COMPARISON_CASES} cases."
            )
        if any(not isinstance(item, LifetimeComparisonCase) for item in cases):
            raise OrbitLifetimeError("cases must contain validated LifetimeComparisonCase values.")
        identifiers = [item.case_id for item in cases]
        if len(identifiers) != len(set(identifiers)):
            raise OrbitLifetimeError("comparison case_id values must be unique.")
        object.__setattr__(self, "cases", cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_id": self.comparison_id,
            "base_problem": self.base_problem.to_dict(),
            "cases": [
                {"case_id": item.case_id, "atmosphere": item.atmosphere.to_dict()}
                for item in sorted(self.cases, key=lambda item: item.case_id)
            ],
        }


@dataclass(frozen=True)
class LifetimeSample:
    sample_index: int
    time_s: float
    elapsed_days: float
    position_x_km: float
    position_y_km: float
    position_z_km: float
    velocity_x_km_s: float
    velocity_y_km_s: float
    velocity_z_km_s: float
    altitude_km: float
    density_kg_m3: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    perigee_altitude_km: float
    apogee_altitude_km: float
    specific_energy_km2_s2: float
    angular_momentum_km2_s: float
    drag_acceleration_m_s2: float
    semi_major_axis_change_km: float


@dataclass(frozen=True)
class LifetimeEvent:
    event_index: int
    threshold_kind: str
    threshold_altitude_km: float
    time_s: float
    jd_utc: float
    altitude_km: float
    bracket_start_s: float
    bracket_end_s: float
    iterations: int
    disposition: str
    position_x_km: float
    position_y_km: float
    position_z_km: float
    velocity_x_km_s: float
    velocity_y_km_s: float
    velocity_z_km_s: float

    def state(self) -> np.ndarray:
        return np.asarray(
            (
                self.position_x_km,
                self.position_y_km,
                self.position_z_km,
                self.velocity_x_km_s,
                self.velocity_y_km_s,
                self.velocity_z_km_s,
            ),
            dtype=float,
        )


@dataclass(frozen=True)
class OrbitLifetimeResult:
    problem: OrbitLifetimeProblem
    samples: tuple[LifetimeSample, ...]
    events: tuple[LifetimeEvent, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class OrbitLifetimeComparisonResult:
    problem: OrbitLifetimeComparisonProblem
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class OrbitLifetimeArtifacts:
    output_dir: Path
    manifest_json: Path
    problem_json: Path
    summary_json: Path
    timeseries_csv: Path
    events_csv: Path


@dataclass(frozen=True)
class OrbitLifetimeComparisonArtifacts:
    output_dir: Path
    manifest_json: Path
    problem_json: Path
    summary_json: Path
    comparison_csv: Path


def _propagator(problem: OrbitLifetimeProblem) -> tuple[OrbitPropagator, dict[str, Any], OrbitContext]:
    plugins = [drag_plugin] if problem.drag_enabled else []
    if problem.include_j2:
        plugins.insert(0, j2_plugin)
    propagator = OrbitPropagator(
        model="two_body", integrator="rk4", plugins=plugins, acceleration_mode="off",
        numeric_backend=problem.numeric_backend,
    )
    env = problem.atmosphere.environment(problem.epoch_jd_utc)
    env.update(
        {
            "drag_area_m2": problem.drag_area_m2,
            "drag_coefficient": problem.drag_coefficient,
        }
    )
    context = OrbitContext(
        mu_km3_s2=EARTH_MU_KM3_S2,
        mass_kg=problem.mass_kg,
        area_m2=problem.drag_area_m2,
        cd=problem.drag_coefficient,
    )
    return propagator, env, context


def _altitude(state: np.ndarray) -> float:
    return norm3(state[:3]) - EARTH_RADIUS_KM


def _radial_velocity_km_s(state: np.ndarray) -> float:
    radius = norm3(state[:3])
    if radius <= 0.0:
        return 0.0
    return dot3(state[:3], state[3:]) / radius


def _propagate_checked(
    problem: OrbitLifetimeProblem,
    propagator: OrbitPropagator,
    state: np.ndarray,
    duration_s: float,
    start_time_s: float,
    env: dict[str, Any],
    context: OrbitContext,
) -> np.ndarray:
    try:
        if problem.numeric_backend == "rust":
            from sim.dynamics.orbit.rust_force_plan import make_plan, native_force_plan_codes
            prepared = getattr(propagator, "_lifetime_native_plan", None)
            if prepared is None and native_force_plan_codes(propagator, env) is not None:
                propagator.numeric_backend = "rust"
                prepared = make_plan(propagator, state, start_time_s, env, context)
                propagator._lifetime_native_plan = prepared
            if prepared is not None and prepared[-1] is not None:
                propagated = np.asarray(prepared[-1].rk4(
                    state.tolist(), start_time_s, duration_s, [0.0, 0.0, 0.0], prepared[-2]
                ), dtype=float)
            else:
                # Atmosphere owners outside the prepared native envelope keep
                # their real stage callbacks through OEL's native integrator.
                propagated = propagator.propagate(
                    state, duration_s, start_time_s, np.zeros(3), env, context,
                )
        else:
            propagated = propagator.propagate(
                state,
                duration_s,
                start_time_s,
                np.zeros(3),
                env,
                context,
            )
    except OrbitLifetimeError:
        raise
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        raise OrbitLifetimeError(
            f"ONP propagation with atmosphere model {problem.atmosphere.model!r} failed from "
            f"{float(start_time_s):.9g} s over {float(duration_s):.9g} s: {exc}"
        ) from exc
    if np.any(~np.isfinite(propagated)):
        raise OrbitLifetimeError("ONP propagation produced a non-finite state.")
    return propagated


def _drag_acceleration_checked(
    problem: OrbitLifetimeProblem,
    time_s: float,
    state: np.ndarray,
    env: dict[str, Any],
    context: OrbitContext,
) -> np.ndarray:
    if not problem.drag_enabled:
        return np.zeros(3)
    try:
        drag = drag_plugin(time_s, state, env, context)
    except OrbitLifetimeError:
        raise
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        raise OrbitLifetimeError(
            f"Atmospheric drag model {problem.atmosphere.model!r} failed at elapsed time "
            f"{float(time_s):.9g} s: {exc}"
        ) from exc
    if np.any(~np.isfinite(drag)):
        raise OrbitLifetimeError("Atmospheric drag evaluation produced a non-finite acceleration.")
    return drag


def _sample(
    problem: OrbitLifetimeProblem,
    state: np.ndarray,
    time_s: float,
    sample_index: int,
    initial_a_km: float,
) -> LifetimeSample:
    try:
        elements = rv_to_coe_eci(state[:3], state[3:])
    except (ArithmeticError, ValueError) as exc:
        raise OrbitLifetimeError(
            f"Osculating orbit metrics failed at elapsed time {float(time_s):.9g} s: {exc}"
        ) from exc
    density = problem.atmosphere.density(state, time_s, problem.epoch_jd_utc)
    _, env, context = _propagator(problem)
    drag = _drag_acceleration_checked(problem, time_s, state, env, context)
    radius = norm3(state[:3])
    speed2 = dot3(state[3:], state[3:])
    energy = 0.5 * speed2 - EARTH_MU_KM3_S2 / radius
    angular_momentum = norm3(np.cross(state[:3], state[3:]))
    return LifetimeSample(
        sample_index=sample_index,
        time_s=float(time_s),
        elapsed_days=float(time_s / 86400.0),
        position_x_km=float(state[0]),
        position_y_km=float(state[1]),
        position_z_km=float(state[2]),
        velocity_x_km_s=float(state[3]),
        velocity_y_km_s=float(state[4]),
        velocity_z_km_s=float(state[5]),
        altitude_km=_altitude(state),
        density_kg_m3=density,
        semi_major_axis_km=elements.a_km,
        eccentricity=elements.ecc,
        inclination_deg=elements.inc_deg,
        perigee_altitude_km=elements.a_km * (1.0 - elements.ecc) - EARTH_RADIUS_KM,
        apogee_altitude_km=elements.a_km * (1.0 + elements.ecc) - EARTH_RADIUS_KM,
        specific_energy_km2_s2=energy,
        angular_momentum_km2_s=angular_momentum,
        drag_acceleration_m_s2=norm3(drag) * 1000.0,
        semi_major_axis_change_km=elements.a_km - initial_a_km,
    )


def _refine_threshold(
    problem: OrbitLifetimeProblem,
    left_state: np.ndarray,
    left_time: float,
    right_state: np.ndarray,
    right_time: float,
    kind: str,
    threshold: float,
    *,
    prefer_above: bool = False,
) -> LifetimeEvent:
    lo_t = float(left_time)
    hi_t = float(right_time)
    lo_state = np.array(left_state, dtype=float, copy=True)
    hi_state = np.array(right_state, dtype=float, copy=True)
    iterations = 0
    while hi_t - lo_t > problem.transition_time_tolerance_s:
        if iterations >= problem.transition_max_iterations:
            raise OrbitLifetimeError(
                f"Threshold refinement for {kind!r} exceeded transition_max_iterations."
            )
        mid_t = 0.5 * (lo_t + hi_t)
        propagator, env, context = _propagator(problem)
        mid_state = _propagate_checked(
            problem,
            propagator,
            left_state,
            mid_t - left_time,
            left_time,
            env,
            context,
        )
        if _altitude(mid_state) <= threshold:
            hi_t, hi_state = mid_t, mid_state
        else:
            lo_t, lo_state = mid_t, mid_state
        iterations += 1
    if prefer_above:
        event_t, event_state = lo_t, lo_state
    else:
        event_t, event_state = min(
            ((lo_t, lo_state), (hi_t, hi_state)),
            key=lambda item: abs(_altitude(item[1]) - threshold),
        )
    return LifetimeEvent(
        event_index=-1,
        threshold_kind=kind,
        threshold_altitude_km=threshold,
        time_s=event_t,
        jd_utc=problem.epoch_jd_utc + event_t / 86400.0,
        altitude_km=_altitude(event_state),
        bracket_start_s=lo_t,
        bracket_end_s=hi_t,
        iterations=iterations,
        disposition="provider_refined",
        position_x_km=float(event_state[0]),
        position_y_km=float(event_state[1]),
        position_z_km=float(event_state[2]),
        velocity_x_km_s=float(event_state[3]),
        velocity_y_km_s=float(event_state[4]),
        velocity_z_km_s=float(event_state[5]),
    )


def _refine_radial_minimum(
    problem: OrbitLifetimeProblem,
    left_state: np.ndarray,
    left_time: float,
    right_state: np.ndarray,
    right_time: float,
) -> tuple[float, np.ndarray]:
    lo_t = float(left_time)
    hi_t = float(right_time)
    lo_state = np.array(left_state, dtype=float, copy=True)
    hi_state = np.array(right_state, dtype=float, copy=True)
    best_t, best_state = min(
        ((lo_t, lo_state), (hi_t, hi_state)),
        key=lambda item: _altitude(item[1]),
    )
    iterations = 0
    while hi_t - lo_t > problem.transition_time_tolerance_s:
        if iterations >= problem.transition_max_iterations:
            raise OrbitLifetimeError("Interior radial-minimum refinement exceeded transition_max_iterations.")
        mid_t = 0.5 * (lo_t + hi_t)
        propagator, env, context = _propagator(problem)
        mid_state = _propagate_checked(
            problem,
            propagator,
            left_state,
            mid_t - left_time,
            left_time,
            env,
            context,
        )
        if _altitude(mid_state) < _altitude(best_state):
            best_t, best_state = mid_t, mid_state
        if _radial_velocity_km_s(mid_state) >= 0.0:
            hi_t, hi_state = mid_t, mid_state
        else:
            lo_t, lo_state = mid_t, mid_state
        iterations += 1
    for candidate_t, candidate_state in ((lo_t, lo_state), (hi_t, hi_state)):
        if _altitude(candidate_state) < _altitude(best_state):
            best_t, best_state = candidate_t, candidate_state
    return best_t, best_state


def _find_downward_crossing(
    problem: OrbitLifetimeProblem,
    left_state: np.ndarray,
    left_time: float,
    right_state: np.ndarray,
    right_time: float,
    kind: str,
    threshold: float,
    *,
    prefer_above: bool = False,
) -> LifetimeEvent | None:
    left_altitude = _altitude(left_state)
    right_altitude = _altitude(right_state)
    if left_altitude <= threshold:
        return None
    if right_altitude <= threshold:
        return _refine_threshold(
            problem,
            left_state,
            left_time,
            right_state,
            right_time,
            kind,
            threshold,
            prefer_above=prefer_above,
        )
    if _radial_velocity_km_s(left_state) < 0.0 <= _radial_velocity_km_s(right_state):
        minimum_time, minimum_state = _refine_radial_minimum(
            problem,
            left_state,
            left_time,
            right_state,
            right_time,
        )
        if _altitude(minimum_state) <= threshold:
            return _refine_threshold(
                problem,
                left_state,
                left_time,
                minimum_state,
                minimum_time,
                kind,
                threshold,
                prefer_above=prefer_above,
            )
    return None


def assess_orbit_lifetime(
    problem: OrbitLifetimeProblem | Mapping[str, Any],
) -> OrbitLifetimeResult:
    """Propagate one declared ONP drag case and retain bounded lifetime evidence."""

    parsed = problem if isinstance(problem, OrbitLifetimeProblem) else OrbitLifetimeProblem.from_mapping(problem)
    if parsed.numeric_backend == "rust":
        require_native("ONPForceContext.rk4", "ONPStageContext")
        try:
            from sim.dynamics.orbit.rust_force_plan import make_plan  # noqa: F401
        except ImportError as exc:
            raise OrbitLifetimeError("Rust lifetime force adapter is unavailable in this installation.") from exc
    state = parsed.initial_state()
    initial_elements = rv_to_coe_eci(state[:3], state[3:])
    samples = [_sample(parsed, state, 0.0, 0, initial_elements.a_km)]
    events: list[LifetimeEvent] = []
    reached: set[str] = set()
    initial_altitude = _altitude(state)
    for kind, threshold in parsed.thresholds.items():
        if initial_altitude <= threshold:
            events.append(
                LifetimeEvent(
                    event_index=-1,
                    threshold_kind=kind,
                    threshold_altitude_km=threshold,
                    time_s=0.0,
                    jd_utc=parsed.epoch_jd_utc,
                    altitude_km=initial_altitude,
                    bracket_start_s=0.0,
                    bracket_end_s=0.0,
                    iterations=0,
                    disposition="initial_state_at_or_below",
                    position_x_km=float(state[0]),
                    position_y_km=float(state[1]),
                    position_z_km=float(state[2]),
                    velocity_x_km_s=float(state[3]),
                    velocity_y_km_s=float(state[4]),
                    velocity_z_km_s=float(state[5]),
                )
            )
            reached.add(kind)

    propagator, env, context = _propagator(parsed)
    time_s = 0.0
    next_output_s = min(parsed.output_step_s, parsed.duration_s)
    integration_steps = 0
    integrated_drag_work = 0.0
    stop_reason = "horizon_complete"
    while time_s < parsed.duration_s - 1.0e-12:
        dt_s = min(parsed.integration_step_s, parsed.duration_s - time_s)
        left_state = state
        left_time = time_s
        left_drag = _drag_acceleration_checked(parsed, left_time, left_state, env, context)
        right_state = _propagate_checked(
            parsed,
            propagator,
            left_state,
            dt_s,
            left_time,
            env,
            context,
        )
        right_time = left_time + dt_s
        right_drag = _drag_acceleration_checked(parsed, right_time, right_state, env, context)
        step_drag_work = 0.5 * dt_s * (
            float(np.dot(left_state[3:], left_drag)) + float(np.dot(right_state[3:], right_drag))
        )
        integration_steps += 1
        step_events: list[LifetimeEvent] = []
        for kind, threshold in parsed.thresholds.items():
            if kind not in reached:
                event = _find_downward_crossing(
                    parsed,
                    left_state,
                    left_time,
                    right_state,
                    right_time,
                    kind,
                    threshold,
                )
                if event is not None:
                    step_events.append(event)
                    reached.add(kind)
        events.extend(step_events)
        reentry = next((item for item in step_events if item.threshold_kind == "reentry"), None)
        if reentry is not None and parsed.stop_at_reentry:
            state = reentry.state()
            time_s = reentry.time_s
            event_drag = _drag_acceleration_checked(parsed, time_s, state, env, context)
            integrated_drag_work += 0.5 * (time_s - left_time) * (
                float(np.dot(left_state[3:], left_drag)) + float(np.dot(state[3:], event_drag))
            )
            stop_reason = "reentry_threshold_reached"
            samples.append(_sample(parsed, state, time_s, len(samples), initial_elements.a_km))
            break
        domain_floor = float(parsed.atmosphere.altitude_domain()["minimum_altitude_km"])
        terminal_kind = "earth_surface" if domain_floor <= 0.0 else "atmosphere_domain_limit"
        terminal = _find_downward_crossing(
            parsed,
            left_state,
            left_time,
            right_state,
            right_time,
            terminal_kind,
            domain_floor,
            prefer_above=True,
        )
        if terminal is not None:
            events.append(terminal)
            state = terminal.state()
            time_s = terminal.time_s
            event_drag = _drag_acceleration_checked(parsed, time_s, state, env, context)
            integrated_drag_work += 0.5 * (time_s - left_time) * (
                float(np.dot(left_state[3:], left_drag)) + float(np.dot(state[3:], event_drag))
            )
            stop_reason = (
                "earth_surface_reached" if terminal_kind == "earth_surface" else "atmosphere_domain_limit_reached"
            )
            samples.append(_sample(parsed, state, time_s, len(samples), initial_elements.a_km))
            break
        integrated_drag_work += step_drag_work
        state = right_state
        time_s = right_time
        if time_s >= next_output_s - 1.0e-9 or time_s >= parsed.duration_s - 1.0e-9:
            samples.append(_sample(parsed, state, time_s, len(samples), initial_elements.a_km))
            while next_output_s <= time_s + 1.0e-9:
                next_output_s += parsed.output_step_s

    if abs(samples[-1].time_s - time_s) > 1.0e-9:
        samples.append(_sample(parsed, state, time_s, len(samples), initial_elements.a_km))
    ordered_events = tuple(
        replace(item, event_index=index)
        for index, item in enumerate(
            sorted(events, key=lambda item: (item.time_s, -item.threshold_altitude_km, item.threshold_kind))
        )
    )
    final_sample = samples[-1]
    initial_sample = samples[0]
    event_by_kind = {item.threshold_kind: item for item in ordered_events}
    threshold_summary = {
        kind: {
            "altitude_km": threshold,
            "reached": kind in event_by_kind,
            "time_s": None if kind not in event_by_kind else event_by_kind[kind].time_s,
            "elapsed_days": None if kind not in event_by_kind else event_by_kind[kind].time_s / 86400.0,
            "disposition": None if kind not in event_by_kind else event_by_kind[kind].disposition,
        }
        for kind, threshold in parsed.thresholds.items()
    }
    energy_change = final_sample.specific_energy_km2_s2 - initial_sample.specific_energy_km2_s2
    accounting_applicable = not parsed.include_j2
    energy_residual = energy_change - integrated_drag_work if accounting_applicable else None
    problem_hash = _digest(parsed.to_dict())
    result_hash = _digest(
        {
            "problem_semantic_sha256": problem_hash,
            "samples": [asdict(item) for item in samples],
            "events": [asdict(item) for item in ordered_events],
        }
    )
    summary = {
        "schema_version": ORBIT_LIFETIME_EVIDENCE_SCHEMA,
        "analysis_id": parsed.analysis_id,
        "asset_id": parsed.asset_id,
        "status": "completed",
        "outcome": stop_reason,
        "problem_semantic_sha256": problem_hash,
        "result_semantic_sha256": result_hash,
        "propagator": {
            "family": "ONP",
            "integrator": "rk4",
            "force_models": [
                *(["J2"] if parsed.include_j2 else []),
                *(["drag"] if parsed.drag_enabled else []),
            ],
            "state_frame": "eci",
            "drag_frame_model": "simple",
        },
        "atmosphere": parsed.atmosphere.to_dict(),
        "atmosphere_effective": parsed.atmosphere.effective_record(),
        "implementation_identity": _implementation_identity((parsed.atmosphere.model,), parsed.numeric_backend),
        "spacecraft": {
            "mass_kg": parsed.mass_kg,
            "drag_area_m2": parsed.drag_area_m2,
            "drag_coefficient": parsed.drag_coefficient,
            "ballistic_coefficient_kg_m2": parsed.mass_kg / (
                parsed.drag_coefficient * parsed.drag_area_m2
            ),
        },
        "resource_use": {
            "integration_steps": integration_steps,
            "output_samples": len(samples),
            "event_count": len(ordered_events),
            "propagated_duration_s": time_s,
        },
        "initial": {
            "altitude_km": initial_sample.altitude_km,
            "semi_major_axis_km": initial_sample.semi_major_axis_km,
            "perigee_altitude_km": initial_sample.perigee_altitude_km,
            "apogee_altitude_km": initial_sample.apogee_altitude_km,
        },
        "final": {
            "time_s": final_sample.time_s,
            "elapsed_days": final_sample.elapsed_days,
            "altitude_km": final_sample.altitude_km,
            "semi_major_axis_km": final_sample.semi_major_axis_km,
            "perigee_altitude_km": final_sample.perigee_altitude_km,
            "apogee_altitude_km": final_sample.apogee_altitude_km,
        },
        "extrema": {
            "minimum_altitude_km": min(item.altitude_km for item in samples),
            "minimum_perigee_altitude_km": min(item.perigee_altitude_km for item in samples),
            "maximum_density_kg_m3": max(item.density_kg_m3 for item in samples),
            "maximum_drag_acceleration_m_s2": max(item.drag_acceleration_m_s2 for item in samples),
        },
        "changes": {
            "semi_major_axis_km": final_sample.semi_major_axis_km - initial_sample.semi_major_axis_km,
            "specific_energy_km2_s2": energy_change,
            "angular_momentum_km2_s": final_sample.angular_momentum_km2_s
            - initial_sample.angular_momentum_km2_s,
        },
        "energy_accounting": {
            "applicable": accounting_applicable,
            "integrated_drag_work_km2_s2": integrated_drag_work,
            "kepler_energy_change_km2_s2": energy_change,
            "residual_km2_s2": energy_residual,
            "reason": None if accounting_applicable else "J2 potential is not included in Kepler-energy closure.",
        },
        "thresholds": threshold_summary,
        "claim_limits": [
            "This is deterministic engineering evidence for one declared initial state, spacecraft, atmosphere, and horizon.",
            "A threshold not reached within the horizon is not evidence of infinite lifetime or long-term compliance.",
            "The v1 workflow uses frozen scalar weather inputs and does not predict or ingest current space weather.",
            "Propagation terminates rather than continuing beyond the selected atmosphere model or Earth-surface domain.",
            "The result is not orbit-custody, disposal-compliance, reentry-risk, flight-qualification, or operational authority.",
        ],
    }
    return OrbitLifetimeResult(parsed, tuple(samples), ordered_events, summary)


def compare_orbit_lifetime_models(
    problem: OrbitLifetimeComparisonProblem | Mapping[str, Any],
) -> OrbitLifetimeComparisonResult:
    """Run bounded atmosphere cases with identical non-atmosphere inputs."""

    parsed = (
        problem
        if isinstance(problem, OrbitLifetimeComparisonProblem)
        else OrbitLifetimeComparisonProblem.from_mapping(problem)
    )
    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in sorted(parsed.cases, key=lambda item: item.case_id):
        case_problem = replace(
            parsed.base_problem,
            analysis_id=f"{parsed.base_problem.analysis_id}.{case.case_id}",
            atmosphere=case.atmosphere,
        )
        result = assess_orbit_lifetime(case_problem)
        reentry = result.summary["thresholds"]["reentry"]
        row = {
            "case_id": case.case_id,
            "atmosphere_model": case.atmosphere.model,
            "outcome": result.summary["outcome"],
            "reentry_reached": reentry["reached"],
            "reentry_time_s": reentry["time_s"],
            "reentry_elapsed_days": reentry["elapsed_days"],
            "propagated_duration_s": result.summary["resource_use"]["propagated_duration_s"],
            "final_semi_major_axis_km": result.summary["final"]["semi_major_axis_km"],
            "minimum_altitude_km": result.summary["extrema"]["minimum_altitude_km"],
            "minimum_perigee_altitude_km": result.summary["extrema"]["minimum_perigee_altitude_km"],
            "maximum_density_kg_m3": result.summary["extrema"]["maximum_density_kg_m3"],
            "result_semantic_sha256": result.summary["result_semantic_sha256"],
        }
        rows.append(row)
        case_summaries.append({"case_id": case.case_id, "atmosphere": case.atmosphere.to_dict(), "result": result.summary})
    semantic_comparison = parsed.to_dict()
    del semantic_comparison["base_problem"]["atmosphere"]
    comparison_hash = _digest(semantic_comparison)
    result_hash = _digest({"comparison_semantic_sha256": comparison_hash, "rows": rows})
    summary = {
        "schema_version": ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA,
        "comparison_id": parsed.comparison_id,
        "status": "completed",
        "comparison_semantic_sha256": comparison_hash,
        "result_semantic_sha256": result_hash,
        "case_count": len(rows),
        "identical_non_atmosphere_inputs": True,
        "implementation_identity": _implementation_identity(
            tuple(case.atmosphere.model for case in parsed.cases), parsed.base_problem.numeric_backend
        ),
        "cases": case_summaries,
        "claim_limits": [
            "Cases differ only by the explicitly retained atmosphere record; all other physical inputs are identical.",
            "The unused base_problem atmosphere is excluded from comparison semantic identity.",
            "Model spread is sensitivity evidence, not calibrated uncertainty or a probability distribution.",
            "A horizon-complete case does not establish lifetime beyond the retained propagation horizon.",
        ],
    }
    return OrbitLifetimeComparisonResult(parsed, tuple(rows), summary)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _csv_bytes(values: Sequence[Any], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for value in values:
        writer.writerow(asdict(value) if hasattr(value, "__dataclass_fields__") else value)
    return stream.getvalue().encode()


def _receipt(path: Path, root: Path) -> dict[str, Any]:
    content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_LIFETIME_ARTIFACT_BYTES)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_atomic(files: Mapping[str, bytes], manifest: dict[str, Any], output_dir: str | Path) -> Path:
    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise OrbitLifetimeError("output_dir must not be a symbolic link.")
    destination = destination_input.resolve()
    if destination.exists():
        raise OrbitLifetimeError(f"output_dir must not already exist: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        for name, content in files.items():
            (temporary / name).write_bytes(content)
        manifest["artifacts"] = [_receipt(temporary / name, temporary) for name in sorted(files)]
        manifest_name = (
            "orbit_lifetime_comparison_manifest.json"
            if manifest["schema_version"] == ORBIT_LIFETIME_COMPARISON_MANIFEST_SCHEMA
            else "orbit_lifetime_manifest.json"
        )
        (temporary / manifest_name).write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def write_orbit_lifetime_artifacts(
    result: OrbitLifetimeResult,
    output_dir: str | Path,
) -> OrbitLifetimeArtifacts:
    files = {
        "normalized_problem.json": _json_bytes(result.problem.to_dict()),
        "orbit_lifetime_summary.json": _json_bytes(result.summary),
        "orbit_lifetime_timeseries.csv": _csv_bytes(
            result.samples, tuple(LifetimeSample.__dataclass_fields__)
        ),
        "orbit_lifetime_events.csv": _csv_bytes(result.events, tuple(LifetimeEvent.__dataclass_fields__)),
    }
    manifest = {
        "schema_version": ORBIT_LIFETIME_MANIFEST_SCHEMA,
        "analysis_id": result.problem.analysis_id,
        "asset_id": result.problem.asset_id,
        "status": result.summary["status"],
        "outcome": result.summary["outcome"],
        "problem_semantic_sha256": result.summary["problem_semantic_sha256"],
        "result_semantic_sha256": result.summary["result_semantic_sha256"],
        "implementation_identity": result.summary["implementation_identity"],
        "claim_limits": result.summary["claim_limits"],
    }
    destination = _write_atomic(files, manifest, output_dir)
    return OrbitLifetimeArtifacts(
        output_dir=destination,
        manifest_json=destination / "orbit_lifetime_manifest.json",
        problem_json=destination / "normalized_problem.json",
        summary_json=destination / "orbit_lifetime_summary.json",
        timeseries_csv=destination / "orbit_lifetime_timeseries.csv",
        events_csv=destination / "orbit_lifetime_events.csv",
    )


def write_orbit_lifetime_comparison_artifacts(
    result: OrbitLifetimeComparisonResult,
    output_dir: str | Path,
) -> OrbitLifetimeComparisonArtifacts:
    fields = tuple(result.rows[0])
    files = {
        "normalized_comparison.json": _json_bytes(result.problem.to_dict()),
        "orbit_lifetime_comparison_summary.json": _json_bytes(result.summary),
        "orbit_lifetime_comparison.csv": _csv_bytes(result.rows, fields),
    }
    manifest = {
        "schema_version": ORBIT_LIFETIME_COMPARISON_MANIFEST_SCHEMA,
        "comparison_id": result.problem.comparison_id,
        "status": result.summary["status"],
        "comparison_semantic_sha256": result.summary["comparison_semantic_sha256"],
        "result_semantic_sha256": result.summary["result_semantic_sha256"],
        "implementation_identity": result.summary["implementation_identity"],
        "claim_limits": result.summary["claim_limits"],
    }
    destination = _write_atomic(files, manifest, output_dir)
    return OrbitLifetimeComparisonArtifacts(
        output_dir=destination,
        manifest_json=destination / "orbit_lifetime_comparison_manifest.json",
        problem_json=destination / "normalized_comparison.json",
        summary_json=destination / "orbit_lifetime_comparison_summary.json",
        comparison_csv=destination / "orbit_lifetime_comparison.csv",
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_LIFETIME_JSON_BYTES)
        value = json.loads(content.decode("utf-8"), parse_constant=_reject_constant)
    except (SafeReadError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OrbitLifetimeError(f"Could not read {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise OrbitLifetimeError(f"{field} must be a JSON object.")
    return value


def _verify_receipts(root: Path, manifest: Mapping[str, Any], expected_files: set[str]) -> None:
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list) or len(receipts) != len(expected_files):
        raise OrbitLifetimeError("Lifetime manifest has an invalid receipt set.")
    received: set[str] = set()
    for item in receipts:
        receipt = _mapping(item, "artifact receipt")
        _exact(receipt, {"path", "bytes", "sha256"}, "artifact receipt")
        name = _text(receipt["path"], "artifact receipt path")
        if name not in expected_files or name in received or Path(name).name != name:
            raise OrbitLifetimeError("Lifetime manifest contains an invalid artifact path.")
        received.add(name)
        path = root / name
        try:
            content = read_regular_file_nofollow(
                path, min_bytes=1, max_bytes=MAX_LIFETIME_ARTIFACT_BYTES
            )
        except SafeReadError as exc:
            raise OrbitLifetimeError(f"Could not safely read retained lifetime artifact {name}: {exc}") from exc
        if len(content) != receipt.get("bytes") or hashlib.sha256(content).hexdigest() != receipt.get("sha256"):
            raise OrbitLifetimeError(f"Artifact receipt mismatch for {name}.")
        _valid_digest(receipt["sha256"], f"artifact receipt {name} sha256")
    if received != expected_files:
        raise OrbitLifetimeError("Lifetime manifest does not bind the exact artifact set.")


def verify_orbit_lifetime_artifacts(evidence_dir: str | Path) -> dict[str, Any]:
    root_input = Path(evidence_dir).expanduser()
    if root_input.is_symlink():
        raise OrbitLifetimeError("Orbit-lifetime evidence directory must not be a symbolic link.")
    root = root_input.resolve()
    if not root.is_dir():
        raise OrbitLifetimeError(f"Orbit-lifetime evidence directory does not exist: {root}.")
    expected = {*_SINGLE_FILES, "orbit_lifetime_manifest.json"}
    if {item.name for item in root.iterdir()} != expected:
        raise OrbitLifetimeError("Orbit-lifetime evidence contains an unexpected artifact set.")
    manifest = _read_json(root / "orbit_lifetime_manifest.json", "orbit-lifetime manifest")
    manifest_fields = {
        "schema_version",
        "analysis_id",
        "asset_id",
        "status",
        "outcome",
        "problem_semantic_sha256",
        "result_semantic_sha256",
        "implementation_identity",
        "artifacts",
        "claim_limits",
    }
    _exact(manifest, manifest_fields, "orbit-lifetime manifest")
    if manifest["schema_version"] != ORBIT_LIFETIME_MANIFEST_SCHEMA:
        raise OrbitLifetimeError("Orbit-lifetime manifest has an unsupported schema version.")
    _verify_receipts(root, manifest, _SINGLE_FILES)
    retained_problem = _read_json(root / "normalized_problem.json", "normalized lifetime problem")
    problem = OrbitLifetimeProblem.from_mapping(retained_problem)
    if retained_problem != problem.to_dict():
        raise OrbitLifetimeError("Retained orbit-lifetime problem is not canonically normalized.")
    result = assess_orbit_lifetime(problem)
    retained_summary = _read_json(root / "orbit_lifetime_summary.json", "orbit-lifetime summary")
    if retained_summary != result.summary:
        raise OrbitLifetimeError("Retained orbit-lifetime summary differs from authoritative replay.")
    expected_csv = {
        "orbit_lifetime_timeseries.csv": _csv_bytes(
            result.samples, tuple(LifetimeSample.__dataclass_fields__)
        ),
        "orbit_lifetime_events.csv": _csv_bytes(
            result.events, tuple(LifetimeEvent.__dataclass_fields__)
        ),
    }
    for name, content in expected_csv.items():
        if read_regular_file_nofollow(
            root / name, min_bytes=1, max_bytes=MAX_LIFETIME_ARTIFACT_BYTES
        ) != content:
            raise OrbitLifetimeError(f"Retained {name} differs from authoritative replay.")
    for field in manifest_fields - {"schema_version", "artifacts"}:
        if manifest[field] != result.summary[field]:
            raise OrbitLifetimeError(f"Orbit-lifetime manifest field {field!r} differs from replay.")
    return {
        "schema_version": ORBIT_LIFETIME_EVIDENCE_SCHEMA,
        "status": "verified",
        "analysis_id": problem.analysis_id,
        "asset_id": problem.asset_id,
        "outcome": result.summary["outcome"],
        "problem_semantic_sha256": result.summary["problem_semantic_sha256"],
        "result_semantic_sha256": result.summary["result_semantic_sha256"],
    }


def verify_orbit_lifetime_comparison_artifacts(evidence_dir: str | Path) -> dict[str, Any]:
    root_input = Path(evidence_dir).expanduser()
    if root_input.is_symlink():
        raise OrbitLifetimeError("Orbit-lifetime comparison directory must not be a symbolic link.")
    root = root_input.resolve()
    if not root.is_dir():
        raise OrbitLifetimeError(f"Orbit-lifetime comparison directory does not exist: {root}.")
    expected = {*_COMPARISON_FILES, "orbit_lifetime_comparison_manifest.json"}
    if {item.name for item in root.iterdir()} != expected:
        raise OrbitLifetimeError("Orbit-lifetime comparison contains an unexpected artifact set.")
    manifest = _read_json(
        root / "orbit_lifetime_comparison_manifest.json", "orbit-lifetime comparison manifest"
    )
    fields = {
        "schema_version",
        "comparison_id",
        "status",
        "comparison_semantic_sha256",
        "result_semantic_sha256",
        "implementation_identity",
        "artifacts",
        "claim_limits",
    }
    _exact(manifest, fields, "orbit-lifetime comparison manifest")
    if manifest["schema_version"] != ORBIT_LIFETIME_COMPARISON_MANIFEST_SCHEMA:
        raise OrbitLifetimeError("Orbit-lifetime comparison manifest has an unsupported schema version.")
    _verify_receipts(root, manifest, _COMPARISON_FILES)
    retained_problem = _read_json(root / "normalized_comparison.json", "normalized comparison")
    problem = OrbitLifetimeComparisonProblem.from_mapping(retained_problem)
    if retained_problem != problem.to_dict():
        raise OrbitLifetimeError("Retained orbit-lifetime comparison is not canonically normalized.")
    result = compare_orbit_lifetime_models(problem)
    retained_summary = _read_json(
        root / "orbit_lifetime_comparison_summary.json", "orbit-lifetime comparison summary"
    )
    if retained_summary != result.summary:
        raise OrbitLifetimeError("Retained comparison summary differs from authoritative replay.")
    expected_csv = _csv_bytes(result.rows, tuple(result.rows[0]))
    if read_regular_file_nofollow(
        root / "orbit_lifetime_comparison.csv", min_bytes=1, max_bytes=MAX_LIFETIME_ARTIFACT_BYTES
    ) != expected_csv:
        raise OrbitLifetimeError("Retained orbit-lifetime comparison CSV differs from authoritative replay.")
    for field in fields - {"schema_version", "artifacts"}:
        if manifest[field] != result.summary[field]:
            raise OrbitLifetimeError(f"Orbit-lifetime comparison manifest field {field!r} differs from replay.")
    return {
        "schema_version": ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA,
        "status": "verified",
        "comparison_id": problem.comparison_id,
        "case_count": len(result.rows),
        "comparison_semantic_sha256": result.summary["comparison_semantic_sha256"],
        "result_semantic_sha256": result.summary["result_semantic_sha256"],
    }


__all__ = [
    "HARRIS_PRIESTER_SUPPORTED_F107",
    "MAX_LIFETIME_EPOCH_JD_UTC",
    "MAX_LIFETIME_ARTIFACT_BYTES",
    "MAX_LIFETIME_COMPARISON_CASES",
    "MAX_LIFETIME_DURATION_S",
    "MAX_LIFETIME_INTEGRATION_STEPS",
    "MAX_LIFETIME_JSON_BYTES",
    "MAX_LIFETIME_OUTPUT_SAMPLES",
    "MIN_LIFETIME_EPOCH_JD_UTC",
    "ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA",
    "ORBIT_LIFETIME_COMPARISON_MANIFEST_SCHEMA",
    "ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA",
    "ORBIT_LIFETIME_EVIDENCE_SCHEMA",
    "ORBIT_LIFETIME_MANIFEST_SCHEMA",
    "ORBIT_LIFETIME_PROBLEM_SCHEMA",
    "LifetimeAtmosphere",
    "LifetimeComparisonCase",
    "LifetimeEvent",
    "LifetimeSample",
    "LifetimeThresholds",
    "OrbitLifetimeArtifacts",
    "OrbitLifetimeComparisonArtifacts",
    "OrbitLifetimeComparisonProblem",
    "OrbitLifetimeComparisonResult",
    "OrbitLifetimeError",
    "OrbitLifetimeProblem",
    "OrbitLifetimeResult",
    "assess_orbit_lifetime",
    "compare_orbit_lifetime_models",
    "verify_orbit_lifetime_artifacts",
    "verify_orbit_lifetime_comparison_artifacts",
    "write_orbit_lifetime_artifacts",
    "write_orbit_lifetime_comparison_artifacts",
]
