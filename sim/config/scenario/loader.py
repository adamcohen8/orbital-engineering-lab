from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.config.scenario.analysis import (
    _monte_carlo_from_analysis,
    _parse_analysis_section,
)
from sim.config.scenario.models import (
    SimulationScenarioConfig,
)
from sim.config.scenario.objects import (
    _parse_agent_section,
    _parse_ground_stations_section,
    _parse_objects_section,
)
from sim.config.scenario.outputs import (
    _parse_outputs_section,
)
from sim.config.scenario.paths import (
    _validate_config_read_paths,
)
from sim.config.scenario.presets import _resolve_agent_presets
from sim.config.scenario.primitives import (
    _ROOT_UNSUPPORTED_ALIASES,
    _as_dict,
    _enforce_strict_booleans,
    _reject_unknown_fields,
    _reject_unsupported_aliases,
)
from sim.config.scenario.simulator import (
    _parse_simulator_section,
)
from sim.config.scenario.validation import (
    _validate_object_references,
    _validate_orbital_analysis_references,
    _validate_physics_runtime_settings,
)
from sim.schema_versions import LEGACY_SCENARIO_SCHEMA_VERSION, SCENARIO_SCHEMA_VERSION
from sim.security import ConfigPathPolicy

__all__ = [
    'scenario_config_from_dict',
    'load_simulation_yaml',
    'load_simulation_yaml_bytes',
]

def scenario_config_from_dict(
    data: dict[str, Any],
    source_path: str | Path | None = None,
    path_policy: ConfigPathPolicy | None = None,
) -> SimulationScenarioConfig:
    root = _as_dict(data, "root")
    if str(_as_dict(root.get("simulator"), "simulator").get("central_body", "earth")).strip().lower() != "earth":
        from .planetary import apply_planetary_defaults
        root = apply_planetary_defaults(root)
    _reject_unsupported_aliases(root, "", _ROOT_UNSUPPORTED_ALIASES)
    _reject_unknown_fields(
        root,
        "root",
        {
            "schema_version",
            "scenario_name",
            "scenario_description",
            "objects",
            "ground_stations",
            "ground_segment",
            "simulator",
            "outputs",
            "analysis",
            "metadata",
        },
    )
    schema_version = str(root.get("schema_version", LEGACY_SCENARIO_SCHEMA_VERSION) or "").strip()
    if schema_version not in {SCENARIO_SCHEMA_VERSION, LEGACY_SCENARIO_SCHEMA_VERSION}:
        raise ValueError(
            f"schema_version must be {SCENARIO_SCHEMA_VERSION!r}; "
            f"unversioned legacy configs are represented as {LEGACY_SCENARIO_SCHEMA_VERSION!r}."
        )
    base_dir = None if source_path is None else Path(source_path).expanduser().resolve().parent
    if path_policy is None and source_path is not None:
        path_policy = ConfigPathPolicy.default(config_path=source_path)
    root = _resolve_agent_presets(root, base_dir=base_dir, path_policy=path_policy)
    _validate_config_read_paths(root, path_policy)
    _enforce_strict_booleans(root)
    analysis = _parse_analysis_section(root.get("analysis"))
    normalized_mc = _monte_carlo_from_analysis(analysis)
    objects = _parse_objects_section(root.get("objects"))
    rocket = objects.get("rocket", _parse_agent_section(None, role="rocket", object_id="rocket", default_kind="rocket"))
    chaser = objects.get("chaser", _parse_agent_section(None, role="chaser", object_id="chaser", default_kind="satellite"))
    target = objects.get(
        "target",
        _parse_agent_section(
            None,
            role="target",
            object_id="target",
            default_kind="satellite",
            default_enabled=False if objects else None,
        ),
    )
    if not objects and target.enabled:
        objects = {"target": target}
    rocket = objects.get("rocket", rocket)
    chaser = objects.get("chaser", chaser)
    target = objects.get("target", target)
    from sim.ground_segment.config import normalize_config, validate_scenario

    cfg = SimulationScenarioConfig(
        schema_version=schema_version,
        scenario_name=str(root.get("scenario_name", "unnamed_scenario")),
        scenario_description=str(root.get("scenario_description", "") or ""),
        rocket=rocket,
        chaser=chaser,
        target=target,
        objects=objects,
        ground_stations=_parse_ground_stations_section(root.get("ground_stations")),
        ground_segment=normalize_config(root.get("ground_segment")),
        simulator=_parse_simulator_section(root.get("simulator")),
        outputs=_parse_outputs_section(root.get("outputs"), path_policy=path_policy),
        monte_carlo=normalized_mc,
        analysis=analysis,
        metadata=dict(root.get("metadata", {}) or {}),
    )
    if source_path is not None:
        object.__setattr__(cfg, "source_path", Path(source_path).expanduser().resolve())
    for object_id, section in dict(cfg.objects or {}).items():
        if bool(dict(section.reference_orbit or {}).get("enabled", False)) and (not bool(section.enabled)):
            raise ValueError(f"{object_id}.reference_orbit.enabled requires {object_id}.enabled to be true.")
    validate_scenario(cfg)
    _validate_physics_runtime_settings(cfg)
    _validate_object_references(cfg)
    _validate_orbital_analysis_references(cfg)
    return cfg
def load_simulation_yaml_bytes(
    content: bytes,
    *,
    source_path: str | Path,
    path_policy: ConfigPathPolicy | None = None,
    allow_external_config_paths: bool = False,
    allow_external_ai_prompt_files: bool = False,
) -> SimulationScenarioConfig:
    """Parse a scenario from exact caller-supplied bytes with path provenance."""

    p = Path(source_path)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Simulation YAML must be UTF-8: {p}") from exc
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to load simulation YAML configs. Install with `pip install pyyaml`."
        ) from exc

    class _UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def _construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        # Check authored keys before expanding YAML merge keys.  An explicit
        # key may override an inherited value, while two explicit keys in the
        # same mapping remain an error.
        authored_keys: set[Any] = set()
        for key_node, _value_node in node.value:
            if getattr(key_node, "tag", None) == "tag:yaml.org,2002:merge":
                continue
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in authored_keys
            except TypeError as exc:
                raise ValueError("YAML mapping keys must be hashable.") from exc
            if duplicate:
                mark = getattr(key_node, "start_mark", None)
                location = "" if mark is None else f" at line {mark.line + 1}, column {mark.column + 1}"
                raise ValueError(f"Duplicate YAML mapping key {key!r}{location}.")
            authored_keys.add(key)
        loader.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_unique_mapping,
    )
    raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("Simulation YAML root must be a mapping/object.")
    policy = path_policy or ConfigPathPolicy.default(
        config_path=p,
        allow_external_config_paths=allow_external_config_paths,
        allow_external_ai_prompt_files=allow_external_ai_prompt_files,
    )
    return scenario_config_from_dict(raw, source_path=p, path_policy=policy)


def load_simulation_yaml(
    path: str | Path,
    *,
    path_policy: ConfigPathPolicy | None = None,
    allow_external_config_paths: bool = False,
    allow_external_ai_prompt_files: bool = False,
) -> SimulationScenarioConfig:
    p = Path(path)
    return load_simulation_yaml_bytes(
        p.read_bytes(),
        source_path=p,
        path_policy=path_policy,
        allow_external_config_paths=allow_external_config_paths,
        allow_external_ai_prompt_files=allow_external_ai_prompt_files,
    )
