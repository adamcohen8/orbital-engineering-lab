from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sim.digital_twin.mass_properties import audit_mass_properties, mass_property_report_markdown
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile

TWIN_SCHEMA_V0 = "oel.spacecraft_twin.v0"


@dataclass(frozen=True)
class TwinGeometrySummary:
    path: Path
    sample_count: int
    area_min_m2: float
    area_max_m2: float
    area_mean_m2: float
    confidence: str = "unknown"


@dataclass(frozen=True)
class TwinValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    geometry_summary: TwinGeometrySummary | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SpacecraftTwinPackage:
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> SpacecraftTwinPackage:
        p = Path(path).expanduser().resolve()
        raw = _load_yaml_mapping(p, "spacecraft twin package")
        return cls(path=p, raw=raw)

    @property
    def root_dir(self) -> Path:
        return self.path.parent

    @property
    def schema(self) -> str:
        return str(self.raw.get("schema", "") or "")

    @property
    def object_id(self) -> str:
        return str(self.raw.get("object_id", "") or "").strip()

    @property
    def display_name(self) -> str:
        return str(self.raw.get("display_name", self.object_id) or self.object_id)

    @property
    def version(self) -> str:
        return str(self.raw.get("version", "0.0.0") or "0.0.0")

    def resolve_artifact_path(
        self,
        section_name: str,
        *,
        key: str = "path",
        required: bool = False,
    ) -> Path | None:
        section = self.raw.get(section_name)
        if isinstance(section, str):
            raw_path = section
        elif isinstance(section, dict):
            raw_path = section.get(key)
        else:
            raw_path = None
        if raw_path in (None, ""):
            if required:
                raise ValueError(f"{section_name}.{key} is required.")
            return None
        return _resolve_package_path(self.root_dir, str(raw_path))

    def object_path(self) -> Path:
        return self.resolve_artifact_path("object", required=True)  # type: ignore[return-value]

    def geometry_profile_path(self) -> Path | None:
        geometry = self.raw.get("geometry")
        if not isinstance(geometry, dict):
            return None
        raw_path = geometry.get("area_profile_path", geometry.get("profile_path"))
        if raw_path in (None, ""):
            return None
        return _resolve_package_path(self.root_dir, str(raw_path))

    def source_mesh_path(self) -> Path | None:
        geometry = self.raw.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("source_mesh_path") in (None, ""):
            return None
        return _resolve_package_path(self.root_dir, str(geometry["source_mesh_path"]))

    def geometry_confidence(self) -> str:
        geometry = self.raw.get("geometry")
        if not isinstance(geometry, dict):
            return "unknown"
        return str(geometry.get("confidence", "unknown") or "unknown")

    def mass_properties_path(self) -> Path | None:
        return self.resolve_artifact_path("mass_properties", required=False)

    def source_evidence_path(self) -> Path | None:
        return self.resolve_artifact_path("source_evidence", required=False)

    def assumptions_path(self) -> Path | None:
        return self.resolve_artifact_path("assumptions", required=False)

    def report_path(self) -> Path:
        validation = self.raw.get("validation")
        raw_path = "validation_report.md"
        if isinstance(validation, dict) and validation.get("report_path") not in (None, ""):
            raw_path = str(validation["report_path"])
        return _resolve_package_path(self.root_dir, raw_path)

    def assembled_object(self) -> dict[str, Any]:
        object_block = _load_yaml_mapping(self.object_path(), "spacecraft twin object")
        if "objects" in object_block and self.object_id in dict(object_block.get("objects", {}) or {}):
            object_block = dict(object_block["objects"][self.object_id])
        elif self.object_id in object_block and isinstance(object_block[self.object_id], dict):
            object_block = dict(object_block[self.object_id])
        else:
            object_block = dict(object_block)

        assembled = _deep_merge_dicts(
            {
                "kind": "satellite",
                "enabled": True,
                "role": "agent",
                "specs": {},
            },
            object_block,
        )
        specs = dict(assembled.get("specs", {}) or {})
        mass_path = self.mass_properties_path()
        if mass_path is not None and mass_path.is_file():
            mass_snippet = _load_yaml_mapping(mass_path, "spacecraft twin mass properties")
            specs = _deep_merge_dicts(specs, dict(mass_snippet or {}))
        geometry_profile = self.geometry_profile_path()
        if geometry_profile is not None:
            geometry = dict(specs.get("geometry", {}) or {})
            geometry["profile_path"] = str(geometry_profile)
            specs["geometry"] = geometry
        assembled["specs"] = specs
        return assembled

    def scenario_object_block(self) -> dict[str, Any]:
        if not self.object_id:
            raise ValueError("Twin package object_id is required.")
        return {"objects": {self.object_id: self.assembled_object()}}

    def validate(self) -> TwinValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        missing: list[str] = []
        geometry_summary: TwinGeometrySummary | None = None

        if self.schema != TWIN_SCHEMA_V0:
            errors.append(f"schema must be {TWIN_SCHEMA_V0!r}.")
        if not self.object_id:
            errors.append("object_id is required.")
        if not self.display_name:
            warnings.append("display_name is empty.")

        for label, path in self._referenced_paths().items():
            if path is None:
                continue
            if not _is_relative_to(path, self.root_dir):
                warnings.append(f"{label} is outside the twin package root: {path}")
            if label == "validation.report_path":
                continue
            if not path.is_file():
                errors.append(f"{label} file does not exist: {path}")

        assembled: dict[str, Any] | None = None
        try:
            assembled = self.assembled_object()
            if not isinstance(assembled.get("specs", {}), dict):
                errors.append("assembled object specs must be a mapping/object.")
        except Exception as exc:
            errors.append(f"failed to assemble object: {exc}")

        if assembled is not None:
            specs = dict(assembled.get("specs", {}) or {})
            audit = audit_mass_properties(specs)
            errors.extend(audit.validation.errors)
            warnings.extend(audit.validation.warnings)
            missing.extend(_missing_object_inputs(specs))

        profile_path = self.geometry_profile_path()
        if profile_path is not None and profile_path.is_file():
            try:
                profile = GeometryAreaProfile.load(profile_path)
                geometry_summary = TwinGeometrySummary(
                    path=profile_path,
                    sample_count=int(profile.projected_area_m2.size),
                    area_min_m2=float(np.min(profile.projected_area_m2)),
                    area_max_m2=float(np.max(profile.projected_area_m2)),
                    area_mean_m2=float(np.mean(profile.projected_area_m2)),
                    confidence=self.geometry_confidence(),
                )
            except Exception as exc:
                errors.append(f"failed to load geometry area profile: {exc}")
        elif profile_path is None:
            missing.append("geometry_area_profile")

        if self.source_evidence_path() is None:
            missing.append("source_evidence")
        if self.assumptions_path() is None:
            missing.append("assumptions")

        return TwinValidationResult(
            errors=_dedupe(errors),
            warnings=_dedupe(warnings),
            missing_inputs=_dedupe(missing),
            geometry_summary=geometry_summary,
        )

    def report_markdown(self) -> str:
        validation = self.validate()
        lines = [
            f"# Spacecraft Twin Validation: {self.display_name}",
            "",
            f"- Schema: `{self.schema}`",
            f"- Object ID: `{self.object_id}`",
            f"- Version: `{self.version}`",
            f"- Package: `{self.path}`",
            "",
            "## Artifact Inventory",
        ]
        for label, path in self._referenced_paths().items():
            if path is None:
                lines.append(f"- {label}: not configured")
            else:
                status = "present" if path.is_file() else "missing"
                lines.append(f"- {label}: `{path}` ({status})")

        lines.extend(["", "## Validation"])
        if validation.errors:
            lines.extend(f"- ERROR: {err}" for err in validation.errors)
        else:
            lines.append("- No blocking twin-package validation errors.")
        if validation.warnings:
            lines.extend(f"- WARNING: {warning}" for warning in validation.warnings)
        else:
            lines.append("- No twin-package warnings.")

        lines.extend(["", "## Missing Inputs"])
        if validation.missing_inputs:
            lines.extend(f"- {item}" for item in validation.missing_inputs)
        else:
            lines.append("- No missing input categories detected by the v0 checklist.")

        lines.extend(["", "## Geometry"])
        if validation.geometry_summary is None:
            lines.append("No valid geometry area profile is available.")
        else:
            geom = validation.geometry_summary
            lines.extend(
                [
                    f"- Profile: `{geom.path}`",
                    f"- Confidence: `{geom.confidence}`",
                    f"- Samples: {geom.sample_count}",
                    f"- Projected area min/mean/max: {geom.area_min_m2:.9g} / "
                    f"{geom.area_mean_m2:.9g} / {geom.area_max_m2:.9g} m^2",
                ]
            )

        lines.extend(["", "## Mass Properties"])
        try:
            lines.append(
                mass_property_report_markdown(
                    self.assembled_object().get("specs", {}) or {},
                    title="Mass Properties",
                    source_path=self.mass_properties_path(),
                ).strip()
            )
        except Exception as exc:
            lines.append(f"Mass-property report unavailable: {exc}")

        lines.extend(["", "## Suggested Next Steps"])
        if validation.errors:
            lines.append("- Resolve blocking validation errors before using this twin in a scenario.")
        if validation.missing_inputs:
            lines.append("- Fill or explicitly accept missing input categories before treating the twin as review-ready.")
        if not validation.errors and not validation.missing_inputs:
            lines.append("- Twin package is ready for scenario assembly under the v0 checklist.")
        lines.append("")
        return "\n".join(lines)

    def write_report(self, path: str | Path | None = None) -> Path:
        out = self.report_path() if path in (None, "") else Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.report_markdown(), encoding="utf-8")
        return out

    def write_object_yaml(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_yaml(out, self.scenario_object_block())
        return out

    def _referenced_paths(self) -> dict[str, Path | None]:
        return {
            "object": self.resolve_artifact_path("object", required=False),
            "geometry.area_profile_path": self.geometry_profile_path(),
            "geometry.source_mesh_path": self.source_mesh_path(),
            "mass_properties": self.mass_properties_path(),
            "source_evidence": self.source_evidence_path(),
            "assumptions": self.assumptions_path(),
            "validation.report_path": self.report_path(),
        }


def _missing_object_inputs(specs: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    mass_props = dict(specs.get("mass_properties", {}) or {})
    if "inertia_kg_m2" not in mass_props:
        missing.append("mass_properties.inertia_kg_m2")
    if "center_of_mass_body_m" not in mass_props:
        missing.append("mass_properties.center_of_mass_body_m")
    geometry = dict(specs.get("geometry", {}) or {})
    if not any(geometry.get(key) not in (None, "") for key in ("profile_path", "area_profile_path")):
        missing.append("geometry.area_profile")
    if not any(key in specs for key in ("actuator_preset", "actuators", "actuator_model")):
        missing.append("actuator_config")
    if not any(key in specs for key in ("thruster", "max_thrust_n", "propulsion", "thrusters")):
        missing.append("propulsion_config")
    if not any(key in specs for key in ("sensors", "sensor_model", "sensor_error")):
        missing.append("sensor_config")
    return missing


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load spacecraft twin YAML files.") from exc
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{label} YAML root must be a mapping/object.")
    return dict(raw)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to write spacecraft twin YAML files.") from exc
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _resolve_package_path(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
