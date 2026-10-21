from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def dump_scenario_yaml(data: Mapping[str, Any]) -> str:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to write simulation YAML configs.") from exc
    return yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=False)


def _float_list(value: Iterable[Any], *, length: int, field_name: str) -> list[float]:
    out = [float(item) for item in value]
    if len(out) != int(length):
        raise ValueError(f"{field_name} must contain exactly {length} numeric values.")
    return out


def _merge_dict(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(out.get(key), Mapping) and isinstance(value, Mapping):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _pruned(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _pruned(item)
            if cleaned in (None, {}, []):
                continue
            out[str(key)] = cleaned
        return out
    if isinstance(value, list):
        return [item for item in (_pruned(item) for item in value) if item not in (None, {}, [])]
    return value


def _clean_agent_section(section: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _pruned(section)
    if cleaned.get("enabled") is True:
        cleaned.pop("enabled", None)
    return cleaned


def _clean_object_map(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for object_id, section in value.items():
        if not isinstance(section, Mapping):
            continue
        cleaned = _clean_agent_section(dict(section))
        if cleaned.get("object_id") == object_id:
            cleaned.pop("object_id", None)
        if cleaned:
            out[str(object_id)] = cleaned
    return out


def _drop_section_defaults(section: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in section.items():
        if key in defaults and value == defaults[key]:
            continue
        cleaned = _pruned(value)
        if cleaned in (None, {}, []):
            continue
        out[str(key)] = cleaned
    return out


def _clean_simulator(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    out: dict[str, Any] = {}
    for key, item in raw.items():
        if key == "acceleration" and isinstance(item, Mapping):
            cleaned = _drop_section_defaults(item, {"mode": "off", "warmup": False, "env_override": True})
        elif key == "plugin_validation" and isinstance(item, Mapping):
            cleaned = _drop_section_defaults(item, {"strict": True, "strict_runtime": False})
        elif key == "termination" and isinstance(item, Mapping):
            cleaned = _drop_section_defaults(item, {"earth_impact_enabled": True, "earth_radius_km": 6378.137})
        else:
            cleaned = _pruned(item)
        if cleaned in (None, {}, []):
            continue
        out[str(key)] = cleaned
    return out


def _clean_outputs(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    out: dict[str, Any] = {}
    for key, item in raw.items():
        if key == "stats" and isinstance(item, Mapping):
            cleaned = _drop_section_defaults(
                item,
                {"print_summary": True, "save_json": True, "save_full_log": True, "save_history_npz": False},
            )
        elif key == "plots" and isinstance(item, Mapping):
            cleaned = _drop_section_defaults(item, {"figure_ids": [], "dpi": 150, "style": "oel_dark"})
        elif key == "animations" and isinstance(item, Mapping):
            cleaned = _drop_section_defaults(item, {"types": [], "fps": 30.0})
        elif key == "monte_carlo" and isinstance(item, Mapping):
            cleaned = _drop_section_defaults(
                item,
                {
                    "save_histograms": False,
                    "display_histograms": False,
                    "save_ops_dashboard": True,
                    "display_ops_dashboard": False,
                    "save_iteration_summaries": False,
                    "success_termination_reasons": ["rocket_orbit_insertion"],
                },
            )
        elif key == "review" and isinstance(item, Mapping):
            if not bool(item.get("enabled", False)):
                continue
            cleaned = _drop_section_defaults(item, {"strict": False})
            if bool(cleaned.get("enabled", False)) and "detail" not in cleaned:
                cleaned["detail"] = "standard"
        else:
            cleaned = _pruned(item)
        if cleaned in (None, {}, []):
            continue
        out[str(key)] = cleaned
    return out


def _clean_artifact_dict(root: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(root)
    out: dict[str, Any] = {}
    for key in (
        "scenario_name",
        "scenario_description",
        "metadata",
        "objects",
        "ground_stations",
        "simulator",
        "outputs",
        "monte_carlo",
        "analysis",
    ):
        if key not in raw:
            continue
        value = raw[key]
        if key == "scenario_description" and str(value or "") == "":
            continue
        if key == "objects" and isinstance(value, Mapping):
            cleaned_objects = _clean_object_map(value)
            if cleaned_objects:
                out[key] = cleaned_objects
            continue
        if key == "simulator" and isinstance(value, Mapping):
            cleaned = _clean_simulator(value)
            if cleaned:
                out[key] = cleaned
            continue
        if key == "outputs" and isinstance(value, Mapping):
            cleaned = _clean_outputs(value)
            if cleaned:
                out[key] = cleaned
            continue
        if key == "monte_carlo" and isinstance(value, Mapping) and not bool(value.get("enabled", False)):
            continue
        if (
            key == "analysis"
            and isinstance(value, Mapping)
            and not bool(value.get("enabled", False))
            and not bool(dict(value.get("mission_recovery", {}) or {}).get("enabled", False))
        ):
            continue
        cleaned = _pruned(value)
        if cleaned in (None, {}, []):
            continue
        out[key] = cleaned
    return out


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    severity: str
    message: str
    hint: str | None = None
    allowed_values: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "severity": self.severity,
            "message": self.message,
            "hint": self.hint,
            "allowed_values": list(self.allowed_values),
        }


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    status: str
    scenario_name: str | None = None
    scenario_description: str | None = None
    config_path: str | None = None
    study_type: str | None = None
    issues: tuple[ValidationIssue, ...] = ()
    summary: Mapping[str, Any] | None = None

    @classmethod
    def from_validation_dict(cls, report: Mapping[str, Any]) -> ValidationReport:
        issues = [_validation_issue_from_error(error) for error in list(report.get("errors", []) or [])]
        plugins = report.get("plugins")
        if isinstance(plugins, Mapping) and str(plugins.get("status", "")).lower() == "warn":
            for error in list(plugins.get("errors", []) or []):
                issue = _validation_issue_from_error(error)
                issues.append(
                    ValidationIssue(
                        path=issue.path,
                        severity="warning",
                        message=issue.message,
                        hint=issue.hint,
                        allowed_values=issue.allowed_values,
                    )
                )
        summary_keys = (
            "objects",
            "duration_s",
            "dt_s",
            "output_dir",
            "plugins",
            "generated",
        )
        summary = {key: report.get(key) for key in summary_keys if key in report}
        return cls(
            ok=bool(report.get("ok", False)),
            status=str(report.get("status", "failed") or "failed"),
            scenario_name=(None if report.get("scenario_name") is None else str(report.get("scenario_name"))),
            scenario_description=(
                None if report.get("scenario_description") is None else str(report.get("scenario_description"))
            ),
            config_path=(None if report.get("config_path") is None else str(report.get("config_path"))),
            study_type=(None if report.get("study_type") is None else str(report.get("study_type"))),
            issues=tuple(issues),
            summary=summary,
        )

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "config_path": self.config_path,
            "study_type": self.study_type,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": dict(self.summary or {}),
        }


def _validation_issue_from_error(error: Any) -> ValidationIssue:
    message = str(error)
    path = _validation_path(message)
    allowed_values = _validation_allowed_values(message)
    return ValidationIssue(
        path=path,
        severity="error",
        message=message,
        hint=_validation_hint(message),
        allowed_values=allowed_values,
    )


_PATH_TOKEN_RE = re.compile(
    r"(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9*]+\])?(?:\.[A-Za-z_][A-Za-z0-9_*]*(?:\[[0-9*]+\])?)*)"
)


def _validation_path(message: str) -> str:
    text = str(message or "")
    for raw in re.split(r"[\s:]+", text):
        token = raw.strip("`'\".,;:()")
        if _looks_like_validation_path(token):
            return token
    for match in _PATH_TOKEN_RE.finditer(text):
        token = match.group("path").strip("`'\".,;:()")
        if _looks_like_validation_path(token):
            return token
    return ""


def _looks_like_validation_path(token: str) -> bool:
    if not token:
        return False
    if "." not in token and "[" not in token:
        return False
    if token.lower() in {"one.of", "mapping.object"}:
        return False
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9*]+\])?(?:\.[A-Za-z_][A-Za-z0-9_*]*(?:\[[0-9*]+\])?)*", token))


def _validation_allowed_values(message: str) -> tuple[str, ...]:
    text = str(message or "")
    match = re.search(r"must be (?:one of|one of:)\s*:?\s*(?P<values>[^.]+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"must be ['\"](?P<a>[^'\"]+)['\"] or ['\"](?P<b>[^'\"]+)['\"]", text, flags=re.IGNORECASE)
        if match:
            return (match.group("a"), match.group("b"))
        return ()
    values_text = match.group("values")
    values_text = values_text.replace(" or ", ", ")
    out: list[str] = []
    for item in values_text.split(","):
        value = item.strip().strip("`'\" .;:")
        if value:
            out.append(value)
    return tuple(out)


def _validation_hint(message: str) -> str | None:
    lower = message.lower()
    path = _validation_path(message)
    allowed_values = _validation_allowed_values(message)
    if "integer multiple" in lower and ("duration_s" in lower or "dt_s" in lower):
        return "Set simulator.duration_s and simulator.dt_s so duration_s is a positive integer multiple of dt_s."
    if path == "outputs.review.detail":
        return "Use outputs.review.detail set to compact, standard, or full."
    if path == "outputs.mode":
        return "Use outputs.mode set to interactive, save, or both."
    if path.startswith("ground_stations["):
        if path.endswith(".lat_deg"):
            return "Set station latitude in degrees between -90 and 90."
        if path.endswith(".lon_deg"):
            return "Set station longitude in degrees."
        if path.endswith(".max_range_km"):
            return "Set max_range_km to a positive distance or omit it."
        return "Check the ground-station id, latitude, longitude, and optional access limits."
    if "relative_to_target_ric" in path:
        if path.endswith(".frame"):
            return "Use relative_to_target_ric.frame set to rect or curv."
        if path.endswith(".state"):
            return "Provide a six-element finite numeric RIC state [R, I, C, Rdot, Idot, Cdot]."
    if "initial_state" in path:
        return "Check the object's initial-state block and use a supported state form."
    if "module" in lower or "class_name" in lower or "function" in lower or "plugin" in lower:
        return "Check the configured module, class_name/function, params shape, and plugin contract."
    if allowed_values:
        joined = ", ".join(allowed_values)
        return f"Use one of: {joined}."
    if "must be positive" in lower:
        return "Set this value to a positive number."
    if "must be one of" in lower:
        return "Use one of the allowed values named in the validation message."
    if "must be a mapping" in lower or "must be a mapping/object" in lower:
        return "Use a YAML mapping/object for this section."
    if "plugin validation failed" in lower:
        return "Check the configured module, class/function name, params shape, and plugin contract."
    if "review query" in lower:
        return "Use a read-only SELECT or WITH query against the generated review store."
    return None


@dataclass(frozen=True)
class ScenarioArtifact:
    """Validated, portable scenario artifact backed by OEL YAML semantics."""

    config: Any

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioArtifact:
        from sim.api import SimulationConfig

        return cls(SimulationConfig.from_dict(dict(data)))

    @classmethod
    def from_config(cls, config: Any) -> ScenarioArtifact:
        from sim.api import SimulationConfig
        from sim.config import SimulationScenarioConfig

        if isinstance(config, SimulationConfig):
            return cls(config)
        if isinstance(config, SimulationScenarioConfig):
            return cls(SimulationConfig(config))
        return cls.from_dict(dict(config))

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        path_policy: Any | None = None,
        allow_external_config_paths: bool = False,
        allow_external_ai_prompt_files: bool = False,
    ) -> ScenarioArtifact:
        from sim.api import SimulationConfig

        return cls(
            SimulationConfig.from_yaml(
                path,
                path_policy=path_policy,
                allow_external_config_paths=allow_external_config_paths,
                allow_external_ai_prompt_files=allow_external_ai_prompt_files,
            )
        )

    @property
    def source_path(self) -> Path | None:
        return self.config.source_path

    @property
    def scenario_name(self) -> str:
        return self.config.scenario_name

    def to_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    def to_artifact_dict(self) -> dict[str, Any]:
        return _clean_artifact_dict(self.to_dict())

    def to_yaml_text(self) -> str:
        return dump_scenario_yaml(self.to_artifact_dict())

    def write(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml_text(), encoding="utf-8")
        return target

    def to_config(self) -> Any:
        return self.config

    def to_scenario_config(self) -> Any:
        return self.config.to_scenario_config()

    def validate(self, workspace: Any | None = None) -> dict[str, Any]:
        from sim.api import SimulationWorkspace

        ws = workspace or SimulationWorkspace()
        return ws.validate(self)

    def validate_report(self, workspace: Any | None = None) -> ValidationReport:
        from sim.api import SimulationWorkspace

        ws = workspace or SimulationWorkspace()
        return ws.validate_report(self)

    def run(
        self,
        workspace: Any | None = None,
        *,
        step_callback: Any | None = None,
    ) -> Any:
        from sim.api import SimulationWorkspace

        ws = workspace or SimulationWorkspace()
        return ws.run(self, step_callback=step_callback)

    def with_seed(self, seed: int) -> ScenarioArtifact:
        return ScenarioArtifact(self.config.with_seed(seed))

    def with_value(self, parameter_path: str, value: Any) -> ScenarioArtifact:
        return ScenarioArtifact(self.config.with_value(parameter_path, value))

    def with_output_dir(self, output_dir: str | Path) -> ScenarioArtifact:
        return ScenarioArtifact(self.config.with_output_dir(output_dir))


class ScenarioBuilder:
    """Small domain-facing scenario authoring helper that emits YAML artifacts."""

    def __init__(self, scenario_name: str, *, description: str = "") -> None:
        name = str(scenario_name or "").strip()
        if not name:
            raise ValueError("scenario_name must be non-empty.")
        self._root: dict[str, Any] = {
            "scenario_name": name,
            "scenario_description": str(description or ""),
            "objects": {},
            "simulator": {
                "dynamics": {
                    "attitude": {"enabled": False},
                },
            },
            "outputs": {
                "mode": "save",
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
        }

    @property
    def scenario_name(self) -> str:
        return str(self._root["scenario_name"])

    def description(self, text: str) -> ScenarioBuilder:
        self._root["scenario_description"] = str(text or "")
        return self

    def metadata(self, **values: Any) -> ScenarioBuilder:
        self._root.setdefault("metadata", {}).update(values)
        return self

    def duration(
        self,
        duration_s: float,
        *,
        dt_s: float,
        initial_jd_utc: float | None = None,
        orbit_substep_s: float | None = None,
        attitude_substep_s: float | None = None,
    ) -> ScenarioBuilder:
        simulator = self._root.setdefault("simulator", {})
        simulator["duration_s"] = float(duration_s)
        simulator["dt_s"] = float(dt_s)
        if initial_jd_utc is not None:
            simulator["initial_jd_utc"] = float(initial_jd_utc)
        if orbit_substep_s is not None:
            dynamics = simulator.setdefault("dynamics", {})
            orbit = dynamics.setdefault("orbit", {})
            orbit["orbit_substep_s"] = float(orbit_substep_s)
        if attitude_substep_s is not None:
            dynamics = simulator.setdefault("dynamics", {})
            attitude = dynamics.setdefault("attitude", {})
            attitude["attitude_substep_s"] = float(attitude_substep_s)
        return self

    def outputs(
        self,
        output_dir: str | Path,
        *,
        mode: str = "save",
        plots: bool | Mapping[str, Any] = False,
        animations: bool | Mapping[str, Any] = False,
        stats: Mapping[str, Any] | None = None,
    ) -> ScenarioBuilder:
        outputs = self._root.setdefault("outputs", {})
        outputs["output_dir"] = str(output_dir)
        outputs["mode"] = str(mode)
        outputs["plots"] = self._output_toggle(plots, disabled_extra={"figure_ids": []})
        outputs["animations"] = self._output_toggle(animations, disabled_extra={"types": []})
        if stats is not None:
            outputs["stats"] = dict(stats)
        else:
            outputs.setdefault("stats", {"print_summary": False, "save_json": False, "save_full_log": False})
        return self

    def review(self, *, enabled: bool = True, detail: str = "standard") -> ScenarioBuilder:
        self._root.setdefault("outputs", {})["review"] = {
            "enabled": bool(enabled),
            "detail": str(detail),
        }
        return self

    def satellite(
        self,
        object_id: str,
        *,
        mass_kg: float | None = None,
        position_eci_km: Iterable[Any] | None = None,
        velocity_eci_km_s: Iterable[Any] | None = None,
        coes: Mapping[str, Any] | None = None,
        initial_state: Mapping[str, Any] | None = None,
        preset: str | None = None,
        specs: Mapping[str, Any] | None = None,
        role: str | None = None,
        enabled: bool = True,
    ) -> ScenarioBuilder:
        oid = str(object_id or "").strip()
        if not oid:
            raise ValueError("object_id must be non-empty.")
        state = self._initial_state(
            initial_state=initial_state,
            position_eci_km=position_eci_km,
            velocity_eci_km_s=velocity_eci_km_s,
            coes=coes,
        )
        spec_values = dict(specs or {})
        if mass_kg is not None:
            spec_values["mass_kg"] = float(mass_kg)
        entry: dict[str, Any] = {
            "kind": "satellite",
            "enabled": bool(enabled),
            "role": str(role or oid),
            "specs": spec_values,
            "initial_state": state,
        }
        if preset:
            entry["preset"] = str(preset)
        self._root.setdefault("objects", {})[oid] = entry
        return self

    def import_object(self, imported: Any, *, object_id: str | None = None, replace: bool = False, allow_model_change: bool = False) -> ScenarioBuilder:
        """Attach a selected external state through the modular object IO owner."""
        from sim.data_io.objects.composition import compose_object
        self._root = compose_object(self._root, imported, object_id=object_id, replace=replace, allow_model_change=allow_model_change)
        return self

    def target_satellite(self, **kwargs: Any) -> ScenarioBuilder:
        return self.satellite("target", role="target", **kwargs)

    def chaser_relative_ric(
        self,
        state: Iterable[Any],
        *,
        reference: str = "target",
        frame: str = "rect",
        object_id: str = "chaser",
        mass_kg: float | None = None,
        preset: str | None = None,
        specs: Mapping[str, Any] | None = None,
        role: str = "chaser",
        enabled: bool = True,
    ) -> ScenarioBuilder:
        reference_id = str(reference or "").strip()
        if reference_id != "target":
            raise ValueError("chaser_relative_ric currently supports reference='target'.")
        frame_key = str(frame or "rect").strip().lower()
        if frame_key in {"ric", "ric_rect", "rect", "rectangular"}:
            frame_value = "rect"
        elif frame_key in {"ric_curv", "curv", "curvilinear"}:
            frame_value = "curv"
        else:
            raise ValueError("frame must be one of 'rect' or 'curv'.")
        return self.satellite(
            object_id,
            mass_kg=mass_kg,
            initial_state={
                "relative_to_target_ric": {
                    "frame": frame_value,
                    "state": _float_list(state, length=6, field_name="state"),
                }
            },
            preset=preset,
            specs=specs,
            role=role,
            enabled=enabled,
        )

    def ground_station(
        self,
        station_id: str,
        *,
        lat_deg: float,
        lon_deg: float,
        alt_km: float = 0.0,
        min_elevation_deg: float = 0.0,
        max_range_km: float | None = None,
        enabled: bool = True,
    ) -> ScenarioBuilder:
        sid = str(station_id or "").strip()
        if not sid:
            raise ValueError("station_id must be non-empty.")
        station: dict[str, Any] = {
            "id": sid,
            "lat_deg": float(lat_deg),
            "lon_deg": float(lon_deg),
            "alt_km": float(alt_km),
            "min_elevation_deg": float(min_elevation_deg),
            "enabled": bool(enabled),
        }
        if max_range_km is not None:
            station["max_range_km"] = float(max_range_km)
        stations = self._root.setdefault("ground_stations", [])
        if not isinstance(stations, list):
            raise TypeError("ground_stations is not a list.")
        stations.append(station)
        return self

    def orbit_controller(
        self,
        object_id: str,
        *,
        module: str,
        class_name: str | None = None,
        function: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> ScenarioBuilder:
        if bool(class_name) == bool(function):
            raise ValueError("Provide exactly one of class_name or function.")
        pointer: dict[str, Any] = {
            "kind": "python",
            "module": str(module),
            "params": dict(params or {}),
        }
        if class_name:
            pointer["class_name"] = str(class_name)
        if function:
            pointer["function"] = str(function)
        self._object_entry(object_id)["orbit_control"] = pointer
        return self

    def to_dict(self) -> dict[str, Any]:
        return _merge_dict(self._root, {})

    def artifact(self) -> ScenarioArtifact:
        return ScenarioArtifact.from_dict(self.to_dict())

    @staticmethod
    def _output_toggle(value: bool | Mapping[str, Any], *, disabled_extra: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        out = {"enabled": bool(value)}
        if not bool(value):
            out.update(dict(disabled_extra))
        return out

    @staticmethod
    def _initial_state(
        *,
        initial_state: Mapping[str, Any] | None,
        position_eci_km: Iterable[Any] | None,
        velocity_eci_km_s: Iterable[Any] | None,
        coes: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        provided = [
            initial_state is not None,
            position_eci_km is not None or velocity_eci_km_s is not None,
            coes is not None,
        ]
        if sum(1 for item in provided if item) != 1:
            raise ValueError("Provide exactly one initial state form: initial_state, position/velocity, or coes.")
        if initial_state is not None:
            return dict(initial_state)
        if coes is not None:
            return {"coes": dict(coes)}
        if position_eci_km is None or velocity_eci_km_s is None:
            raise ValueError("position_eci_km and velocity_eci_km_s must be provided together.")
        return {
            "position_eci_km": _float_list(position_eci_km, length=3, field_name="position_eci_km"),
            "velocity_eci_km_s": _float_list(velocity_eci_km_s, length=3, field_name="velocity_eci_km_s"),
        }

    def _object_entry(self, object_id: str) -> dict[str, Any]:
        oid = str(object_id or "").strip()
        objects = self._root.setdefault("objects", {})
        if oid not in objects:
            raise KeyError(f"Unknown object_id {oid!r}. Add the object before configuring it.")
        entry = objects[oid]
        if not isinstance(entry, dict):
            raise TypeError(f"Object entry {oid!r} is not a mapping.")
        return entry
