from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.config.scenario.primitives import _as_dict
from sim.security import ConfigPathPolicy

_AGENT_PRESET_KEYS = ("preset", "preset_yaml", "preset_path")
_AGENT_FRAGMENT_KEYS = {
    "object_id",
    "kind",
    "enabled",
    "role",
    "runtime_profile",
    "propagation_method",
    "general",
    "specs",
    "initial_state",
    "reference_orbit",
    "flight_software",
    "guidance",
    "base_guidance",
    "guidance_modifiers",
    "orbit_control",
    "attitude_control",
    "mission_strategy",
    "mission_execution",
    "mission_objectives",
    "force_models",
    "bridge",
    "knowledge",
}
_PRESET_METADATA_KEYS = {
    "name",
    "description",
    "preset_type",
    "object_type",
    "version",
    "metadata",
}


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _resolve_preset_path(
    preset_ref: str,
    base_dir: Path | None,
    role: str,
    path_policy: ConfigPathPolicy | None = None,
) -> Path:
    ref_path = Path(preset_ref).expanduser()
    candidates: list[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        if base_dir is not None:
            candidates.append(base_dir / ref_path)
        candidates.append(Path.cwd() / ref_path)
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / ref_path)
        candidates.append(repo_root / "sim" / "presets" / "objects" / ref_path)
        if ref_path.suffix == "":
            candidates.append(repo_root / "sim" / "presets" / "objects" / ref_path.with_suffix(".yaml"))

    for candidate in candidates:
        if path_policy is None:
            if candidate.is_file():
                return candidate
            continue
        try:
            # Resolve the logical candidate here and perform the one
            # descriptor-bound content read in ``_load_yaml_mapping``.  A
            # separate retained-file check would authorize one inode and then
            # reopen the visible path for YAML parsing, leaving a replacement
            # window between the two operations.
            return path_policy.resolve_input_file(candidate, purpose=f"{role}.preset")
        except FileNotFoundError:
            continue

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve {role} preset YAML '{preset_ref}'. Checked: {checked}")


def _load_yaml_mapping(
    path: Path,
    section_name: str,
    *,
    path_policy: ConfigPathPolicy | None = None,
) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to load simulation YAML configs. Install with `pip install pyyaml`."
        ) from exc
    if path_policy is None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        _logical, text = path_policy.read_input_text(
            path,
            purpose=section_name,
            max_bytes=2_000_000,
        )
        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{section_name} YAML root must be a mapping/object.")
    return dict(raw)


def _agent_fragment_from_preset(preset: dict[str, Any], role: str, preset_path: Path) -> dict[str, Any]:
    if "specs" in preset:
        fragment = {k: v for k, v in preset.items() if k not in _PRESET_METADATA_KEYS}
        fragment["specs"] = dict(fragment.get("specs", {}) or {})
        return fragment

    if any(k in preset for k in _AGENT_FRAGMENT_KEYS - {"specs"}):
        return {k: v for k, v in preset.items() if k not in _PRESET_METADATA_KEYS}

    specs = {k: v for k, v in preset.items() if k not in _PRESET_METADATA_KEYS}
    if not specs:
        raise ValueError(f"{role} preset YAML '{preset_path}' does not define specs.")
    return {"specs": specs}


def _resolve_agent_preset(
    value: Any,
    role: str,
    base_dir: Path | None,
    path_policy: ConfigPathPolicy | None = None,
) -> dict[str, Any]:
    d = _as_dict(value, role)
    preset_ref = next((d.get(key) for key in _AGENT_PRESET_KEYS if d.get(key) is not None), None)
    if preset_ref is None:
        return d
    if not isinstance(preset_ref, str) or not preset_ref.strip():
        raise ValueError(f"{role}.preset must be a non-empty YAML file path.")

    preset_path = _resolve_preset_path(
        preset_ref.strip(),
        base_dir=base_dir,
        role=role,
        path_policy=path_policy,
    )
    preset_raw = _load_yaml_mapping(
        preset_path,
        f"{role} preset",
        path_policy=path_policy,
    )
    preset_fragment = _agent_fragment_from_preset(preset_raw, role=role, preset_path=preset_path)
    local_fragment = {k: v for k, v in d.items() if k not in _AGENT_PRESET_KEYS}
    merged = _deep_merge_dicts(preset_fragment, local_fragment)

    local_specs = local_fragment.get("specs")
    merged_specs = merged.get("specs")
    if isinstance(local_specs, dict) and isinstance(merged_specs, dict):
        if "mass_kg" in local_specs and "dry_mass_kg" not in local_specs and "fuel_mass_kg" not in local_specs:
            merged_specs.pop("dry_mass_kg", None)
            merged_specs.pop("fuel_mass_kg", None)

    return merged


def _resolve_agent_presets(
    root: dict[str, Any],
    base_dir: Path | None,
    path_policy: ConfigPathPolicy | None = None,
) -> dict[str, Any]:
    resolved = dict(root)
    objects_raw = resolved.get("objects")
    if isinstance(objects_raw, dict):
        resolved_objects = {}
        for object_id, object_value in objects_raw.items():
            resolved_objects[str(object_id)] = _resolve_agent_preset(
                object_value,
                role=f"objects.{object_id}",
                base_dir=base_dir,
                path_policy=path_policy,
            )
        resolved["objects"] = resolved_objects
    return resolved
