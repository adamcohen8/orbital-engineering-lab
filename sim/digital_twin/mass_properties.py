from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VALID_INERTIA_REFERENCE_POINTS = {"center_of_mass", "body_origin", "body_frame_origin", "unknown"}
VALID_FRAMES = {"body", "body_frame", "principal_axes", "unknown"}
VALID_SOURCES = {"user_supplied", "cad_export", "oel_estimate", "mesh_uniform_density_estimate", "preset", "unknown"}
VALID_CONFIDENCE = {"high", "medium", "low", "assumed", "unknown"}

DEFAULT_SATELLITE_INERTIA_KG_M2 = np.diag([120.0, 100.0, 80.0])


@dataclass(frozen=True)
class MassPropertyValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class MassPropertyAudit:
    mass_kg: float | None
    center_of_mass_body_m: np.ndarray | None
    inertia_kg_m2: np.ndarray | None
    principal_moments_kg_m2: np.ndarray | None
    principal_axes_body: np.ndarray | None
    inertia_reference_point: str
    frame: str
    source: str
    confidence: str
    validation: MassPropertyValidationResult


@dataclass(frozen=True)
class MassPropertyImport:
    snippet: dict[str, Any]
    audit: MassPropertyAudit


def validate_mass_properties(
    specs: dict[str, Any],
    *,
    path: str = "specs.mass_properties",
) -> MassPropertyValidationResult:
    errors: list[str] = _validate_object_mass_fields(specs, path)
    warnings: list[str] = []
    raw_mp = specs.get("mass_properties")
    if raw_mp in (None, ""):
        return MassPropertyValidationResult(errors=errors, warnings=warnings)
    if not isinstance(raw_mp, dict):
        return MassPropertyValidationResult(errors=[*errors, f"{path}: must be a mapping/object."], warnings=warnings)

    mp = dict(raw_mp)

    inertia = _array3x3_or_error(mp.get("inertia_kg_m2"), f"{path}.inertia_kg_m2", errors)
    if inertia is not None:
        sym_tol = float(mp.get("symmetry_tolerance", 1e-9) or 1e-9)
        if not np.allclose(inertia, inertia.T, rtol=1e-9, atol=sym_tol):
            errors.append(f"{path}.inertia_kg_m2: must be symmetric within tolerance.")
        inertia_sym = 0.5 * (inertia + inertia.T)
        eigvals = np.linalg.eigvalsh(inertia_sym)
        if np.any(eigvals <= 0.0):
            errors.append(f"{path}.inertia_kg_m2: principal moments must be positive.")
        elif not _satisfies_triangle_inequality(eigvals):
            errors.append(f"{path}.inertia_kg_m2: principal moments must satisfy inertia triangle inequalities.")

    _vector3_or_error(mp.get("center_of_mass_body_m"), f"{path}.center_of_mass_body_m", errors, required=False)
    _finite_float_or_error(mp.get("mass_kg"), f"{path}.mass_kg", errors, required=False, min_value=0.0)

    inertia_reference = str(mp.get("inertia_reference_point", "unknown") or "unknown").strip().lower()
    if inertia_reference not in VALID_INERTIA_REFERENCE_POINTS:
        errors.append(
            f"{path}.inertia_reference_point: must be one of {sorted(VALID_INERTIA_REFERENCE_POINTS)}."
        )
    frame = str(mp.get("frame", "body") or "body").strip().lower()
    if frame not in VALID_FRAMES:
        errors.append(f"{path}.frame: must be one of {sorted(VALID_FRAMES)}.")
    source = str(mp.get("source", "unknown") or "unknown").strip().lower()
    if source not in VALID_SOURCES:
        errors.append(f"{path}.source: must be one of {sorted(VALID_SOURCES)}.")
    confidence = str(mp.get("confidence", "unknown") or "unknown").strip().lower()
    if confidence not in VALID_CONFIDENCE:
        errors.append(f"{path}.confidence: must be one of {sorted(VALID_CONFIDENCE)}.")

    spec_mass = _object_mass_kg(specs)
    mp_mass = _finite_float(mp.get("mass_kg"))
    if spec_mass is not None and mp_mass is not None:
        tolerance = max(1e-6, 1e-6 * abs(spec_mass))
        if abs(spec_mass - mp_mass) > tolerance:
            warnings.append(
                f"{path}.mass_kg ({mp_mass:g}) differs from object mass ({spec_mass:g}); "
                "runtime mass still follows object mass fields."
            )
    if inertia is not None and "center_of_mass_body_m" not in mp:
        warnings.append(f"{path}.center_of_mass_body_m is not set; inertia reference is harder to audit.")
    if inertia is not None and inertia_reference != "center_of_mass":
        errors.append(f"{path}.inertia_reference_point: runtime inertia must be referenced to center_of_mass.")
    if inertia is not None and frame not in {"body", "body_frame"}:
        errors.append(f"{path}.frame: runtime inertia must be expressed in the body frame.")

    return MassPropertyValidationResult(errors=errors, warnings=warnings)


def _validate_object_mass_fields(specs: dict[str, Any], path: str) -> list[str]:
    """Validate the mass fields consumed by the runtime before auditing inertia."""

    object_path = path[:-len(".mass_properties")] if path.endswith(".mass_properties") else path
    errors: list[str] = []
    if "dry_mass_kg" in specs or "fuel_mass_kg" in specs:
        dry = _mass_component_or_error(
            specs.get("dry_mass_kg", 0.0),
            f"{object_path}.dry_mass_kg",
            errors,
            required="dry_mass_kg" in specs,
        )
        fuel = _mass_component_or_error(
            specs.get("fuel_mass_kg", 0.0),
            f"{object_path}.fuel_mass_kg",
            errors,
            required="fuel_mass_kg" in specs,
        )
        if dry is not None and fuel is not None and dry + fuel <= 0.0:
            errors.append(f"{object_path}: dry_mass_kg + fuel_mass_kg must be > 0.")
    elif "mass_kg" in specs:
        mass = _mass_component_or_error(specs.get("mass_kg"), f"{object_path}.mass_kg", errors, required=True)
        if mass is not None and mass <= 0.0:
            errors.append(f"{object_path}.mass_kg: must be > 0.")
    return errors


def resolve_inertia_kg_m2(specs: dict[str, Any], *, default: np.ndarray | None = None) -> np.ndarray:
    raw_mp = specs.get("mass_properties")
    if raw_mp in (None, ""):
        return np.array(DEFAULT_SATELLITE_INERTIA_KG_M2 if default is None else default, dtype=float)
    result = validate_mass_properties(specs)
    if result.errors:
        raise ValueError("; ".join(result.errors))
    mp = dict(raw_mp)
    if "inertia_kg_m2" not in mp:
        return np.array(DEFAULT_SATELLITE_INERTIA_KG_M2 if default is None else default, dtype=float)
    inertia = np.array(mp.get("inertia_kg_m2"), dtype=float).reshape(3, 3)
    return 0.5 * (inertia + inertia.T)


def resolve_center_of_mass_body_m(specs: dict[str, Any], *, default: np.ndarray | None = None) -> np.ndarray:
    raw_mp = specs.get("mass_properties")
    fallback = np.zeros(3, dtype=float) if default is None else np.array(default, dtype=float).reshape(3)
    if raw_mp in (None, ""):
        return fallback
    result = validate_mass_properties(specs)
    if result.errors:
        raise ValueError("; ".join(result.errors))
    mp = dict(raw_mp)
    if "center_of_mass_body_m" not in mp:
        return fallback
    return _vector3(mp.get("center_of_mass_body_m"))


def audit_mass_properties(specs: dict[str, Any]) -> MassPropertyAudit:
    validation = validate_mass_properties(specs)
    raw_mp = specs.get("mass_properties")
    mp = dict(raw_mp) if isinstance(raw_mp, dict) else {}
    inertia = None
    moments = None
    axes = None
    if "inertia_kg_m2" in mp and not validation.errors:
        inertia = np.array(mp.get("inertia_kg_m2"), dtype=float).reshape(3, 3)
        inertia = 0.5 * (inertia + inertia.T)
        moments, axes = np.linalg.eigh(inertia)
    return MassPropertyAudit(
        mass_kg=_object_mass_kg(specs),
        center_of_mass_body_m=None
        if mp.get("center_of_mass_body_m") is None
        else _vector3(mp.get("center_of_mass_body_m")),
        inertia_kg_m2=inertia,
        principal_moments_kg_m2=moments,
        principal_axes_body=axes,
        inertia_reference_point=str(mp.get("inertia_reference_point", "unknown") or "unknown").strip().lower(),
        frame=str(mp.get("frame", "body") or "body").strip().lower(),
        source=str(mp.get("source", "unknown") or "unknown").strip().lower(),
        confidence=str(mp.get("confidence", "unknown") or "unknown").strip().lower(),
        validation=validation,
    )


def import_mass_properties(
    path: str | Path,
    *,
    source: str = "cad_export",
    confidence: str = "high",
    frame: str = "body",
    inertia_reference_point: str = "center_of_mass",
) -> MassPropertyImport:
    raw = _load_mapping(path)
    snippet = normalized_mass_property_snippet(
        raw,
        source=source,
        confidence=confidence,
        frame=frame,
        inertia_reference_point=inertia_reference_point,
    )
    audit = audit_mass_properties(snippet)
    if audit.validation.errors:
        raise ValueError("; ".join(audit.validation.errors))
    return MassPropertyImport(snippet=snippet, audit=audit)


def normalized_mass_property_snippet(
    raw: dict[str, Any],
    *,
    source: str = "cad_export",
    confidence: str = "high",
    frame: str = "body",
    inertia_reference_point: str = "center_of_mass",
) -> dict[str, Any]:
    data = dict(raw or {})
    if "specs" in data and isinstance(data["specs"], dict):
        return normalized_mass_property_snippet(
            dict(data["specs"]),
            source=source,
            confidence=confidence,
            frame=frame,
            inertia_reference_point=inertia_reference_point,
        )

    existing_mp = dict(data.get("mass_properties", {}) or {})
    mass = _first_present(data, ("mass_kg", "mass", "total_mass_kg", "wet_mass_kg"))
    if mass is None:
        mass = _first_present(existing_mp, ("mass_kg", "mass", "total_mass_kg"))
    com = _first_present(
        data,
        (
            "center_of_mass_body_m",
            "center_of_mass_m",
            "center_of_mass",
            "com_body_m",
            "com_m",
            "com",
            "center_of_gravity_body_m",
            "cg_body_m",
        ),
    )
    if com is None:
        com = _first_present(
            existing_mp,
            (
                "center_of_mass_body_m",
                "center_of_mass_m",
                "center_of_mass",
                "com_body_m",
                "com_m",
                "com",
                "center_of_gravity_body_m",
                "cg_body_m",
            ),
        )
    inertia = _first_present(data, ("inertia_kg_m2", "inertia_matrix_kg_m2", "inertia_matrix", "inertia"))
    if inertia is None:
        inertia = _first_present(existing_mp, ("inertia_kg_m2", "inertia_matrix_kg_m2", "inertia_matrix", "inertia"))

    mp: dict[str, Any] = {}
    if mass is not None:
        mp["mass_kg"] = float(mass)
    if com is not None:
        mp["center_of_mass_body_m"] = _vector3(com).tolist()
    if inertia is not None:
        mp["inertia_kg_m2"] = np.array(inertia, dtype=float).reshape(3, 3).tolist()
    mp["inertia_reference_point"] = str(existing_mp.get("inertia_reference_point", inertia_reference_point))
    mp["frame"] = str(existing_mp.get("frame", frame))
    mp["source"] = str(existing_mp.get("source", source))
    mp["confidence"] = str(existing_mp.get("confidence", confidence))

    snippet: dict[str, Any] = {}
    if mass is not None:
        snippet["mass_kg"] = float(mass)
    snippet["mass_properties"] = mp
    return snippet


def mass_property_report_markdown(
    specs: dict[str, Any],
    *,
    title: str = "Mass Properties Audit",
    source_path: str | Path | None = None,
) -> str:
    audit = audit_mass_properties(specs)
    lines = [f"# {title}", ""]
    if source_path is not None:
        lines.extend([f"- Source file: `{source_path}`"])
    lines.extend(
        [
            f"- Source: `{audit.source}`",
            f"- Confidence: `{audit.confidence}`",
            f"- Frame: `{audit.frame}`",
            f"- Inertia reference point: `{audit.inertia_reference_point}`",
        ]
    )
    if audit.mass_kg is not None:
        lines.append(f"- Object mass: {audit.mass_kg:.9g} kg")
    if audit.center_of_mass_body_m is not None:
        lines.append(f"- Center of mass body-frame: {_fmt_vec(audit.center_of_mass_body_m)} m")
        lines.append(f"- COM offset magnitude: {float(np.linalg.norm(audit.center_of_mass_body_m)):.9g} m")
    else:
        lines.append("- Center of mass body-frame: not provided")

    lines.append("")
    lines.append("## Validation")
    if audit.validation.errors:
        lines.extend(f"- ERROR: {err}" for err in audit.validation.errors)
    else:
        lines.append("- No blocking mass-property validation errors.")
    if audit.validation.warnings:
        lines.extend(f"- WARNING: {warning}" for warning in audit.validation.warnings)
    else:
        lines.append("- No mass-property warnings.")

    lines.append("")
    lines.append("## Inertia")
    if audit.inertia_kg_m2 is None:
        lines.append("No valid inertia matrix is available.")
    else:
        lines.append("Inertia matrix, kg m^2:")
        lines.append("")
        lines.append("```text")
        for row in audit.inertia_kg_m2:
            lines.append("[" + ", ".join(f"{float(v): .9g}" for v in row) + "]")
        lines.append("```")
        if audit.principal_moments_kg_m2 is not None:
            lines.append("")
            lines.append("Principal moments, kg m^2:")
            for idx, moment in enumerate(audit.principal_moments_kg_m2, start=1):
                lines.append(f"- I{idx}: {float(moment):.9g}")
        if audit.principal_axes_body is not None:
            lines.append("")
            lines.append("Principal axes as body-frame columns:")
            lines.append("")
            lines.append("```text")
            for row in audit.principal_axes_body:
                lines.append("[" + ", ".join(f"{float(v): .9g}" for v in row) + "]")
            lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _array3x3_or_error(value: Any, path: str, errors: list[str]) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.array(value, dtype=float)
    except (TypeError, ValueError):
        errors.append(f"{path}: must be a finite 3x3 numeric matrix.")
        return None
    if arr.shape != (3, 3):
        errors.append(f"{path}: must be a 3x3 matrix.")
        return None
    if not np.all(np.isfinite(arr)):
        errors.append(f"{path}: must contain only finite values.")
        return None
    return arr


def _vector3_or_error(value: Any, path: str, errors: list[str], *, required: bool) -> np.ndarray | None:
    if value is None:
        if required:
            errors.append(f"{path}: is required.")
        return None
    try:
        vec = _vector3(value)
    except (TypeError, ValueError):
        errors.append(f"{path}: must be a length-3 finite numeric vector.")
        return None
    if not np.all(np.isfinite(vec)):
        errors.append(f"{path}: must contain only finite values.")
        return None
    return vec


def _finite_float_or_error(
    value: Any,
    path: str,
    errors: list[str],
    *,
    required: bool,
    min_value: float | None = None,
) -> float | None:
    if value is None:
        if required:
            errors.append(f"{path}: is required.")
        return None
    parsed = _finite_float(value)
    if parsed is None:
        errors.append(f"{path}: must be a finite number.")
        return None
    if min_value is not None and parsed < min_value:
        errors.append(f"{path}: must be >= {min_value:g}.")
    return parsed


def _mass_component_or_error(value: Any, path: str, errors: list[str], *, required: bool) -> float | None:
    if value is None:
        if required:
            errors.append(f"{path}: must be a finite number.")
        return None
    if isinstance(value, (bool, np.bool_)):
        errors.append(f"{path}: must be a finite number.")
        return None
    return _finite_float_or_error(value, path, errors, required=required, min_value=0.0)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _vector3(value: Any) -> np.ndarray:
    vec = np.array(value, dtype=float).reshape(-1)
    if vec.shape != (3,):
        raise ValueError("Expected length-3 vector.")
    return vec


def _object_mass_kg(specs: dict[str, Any]) -> float | None:
    if "dry_mass_kg" in specs or "fuel_mass_kg" in specs:
        dry = _finite_float(specs.get("dry_mass_kg", 0.0))
        fuel = _finite_float(specs.get("fuel_mass_kg", 0.0))
        if dry is None or fuel is None:
            return None
        return dry + fuel
    return _finite_float(specs.get("mass_kg"))


def _satisfies_triangle_inequality(moments: np.ndarray, *, tolerance: float = 1e-9) -> bool:
    vals = np.sort(np.array(moments, dtype=float).reshape(3))
    return bool(vals[0] + vals[1] + tolerance >= vals[2])


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def _load_mapping(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required to import YAML mass-property files.") from exc
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Mass-property input root must be a mapping/object.")
    return dict(raw)


def _fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.9g}" for v in vec.reshape(3)) + "]"
