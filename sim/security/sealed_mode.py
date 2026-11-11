from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sim.config.object_refs import configured_objects, object_parameter_prefix
from sim.config.plugin_specs import iter_nested_plugin_specs, plugin_spec_field

_HOSTED_AI_PROVIDERS = {"anthropic", "claude", "gemini", "google", "openai"}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_TRUSTED_PLUGIN_PREFIXES = ("sim.", "integrations.cfs_sil.")

_DEFAULT_AI_ENDPOINTS = {
    "ollama": "http://localhost:11434",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "claude": "https://api.anthropic.com/v1",
}


@dataclass(frozen=True)
class SealedModePolicy:
    allow_untrusted_plugin_imports: bool = False
    allow_hosted_ai: bool = False
    allow_custom_ai_endpoints: bool = False
    allow_non_loopback_sil: bool = False
    allow_high_detail_outputs: bool = False
    allow_gravity_model_downloads: bool = False
    trusted_plugin_prefixes: tuple[str, ...] = _TRUSTED_PLUGIN_PREFIXES


def sealed_mode_enabled(explicit: bool = False) -> bool:
    return bool(explicit or os.environ.get("OEL_SEALED_MODE", "").strip().lower() in {"1", "true", "yes", "on"})


def validate_sealed_mode(
    cfg: Any,
    policy: SealedModePolicy | None = None,
    *,
    offline_ai_operation: bool = False,
) -> list[str]:
    policy = policy or SealedModePolicy()
    errors: list[str] = []
    errors.extend(_validate_plugin_modules(cfg, policy))
    errors.extend(
        _validate_ai_section(
            dict(getattr(getattr(cfg, "outputs", None), "ai_report", {}) or {}),
            "outputs.ai_report",
            policy,
            enabled_default=False,
            offline_operation=offline_ai_operation,
        )
    )
    errors.extend(
        _validate_ai_section(
            dict(getattr(getattr(cfg, "outputs", None), "ai_config", {}) or {}),
            "outputs.ai_config",
            policy,
            enabled_default=True,
            offline_operation=offline_ai_operation,
        )
    )
    errors.extend(_validate_sil_networking(cfg, policy))
    errors.extend(_validate_gravity_model_downloads(cfg, policy))
    errors.extend(_validate_output_retention(cfg, policy, offline_ai_operation=offline_ai_operation))
    return errors


def _validate_plugin_modules(cfg: Any, policy: SealedModePolicy) -> list[str]:
    if policy.allow_untrusted_plugin_imports:
        return []
    errors: list[str] = []
    for index, pointer in enumerate(getattr(getattr(cfg, "simulator", None), "system_force_models", []) or []):
        pointer_path = f"simulator.system_force_models[{index}]"
        errors.extend(_validate_plugin_module(pointer, pointer_path, policy))
        for nested_path, nested_pointer in iter_nested_plugin_specs(pointer, pointer_path):
            errors.extend(_validate_plugin_module(nested_pointer, nested_path, policy))
    for object_id, agent in configured_objects(cfg).items():
        if not getattr(agent, "enabled", False):
            continue
        base_path = object_parameter_prefix(str(object_id))
        for field_name, pointer in _plugin_pointers(agent):
            pointer_path = f"{base_path}.{field_name}"
            errors.extend(_validate_plugin_module(pointer, pointer_path, policy))
            for nested_path, nested_pointer in iter_nested_plugin_specs(pointer, pointer_path):
                errors.extend(_validate_plugin_module(nested_pointer, nested_path, policy))
    return errors


def _validate_plugin_module(pointer: Any, path: str, policy: SealedModePolicy) -> list[str]:
    module = str(plugin_spec_field(pointer, "module", "") or "").strip()
    if not module:
        return []
    if module.startswith(policy.trusted_plugin_prefixes):
        if _module_resolves_from_trusted_installation(module):
            return []
        return [
            f"{path}: sealed mode blocks plugin module '{module}' because it does not resolve "
            "from the trusted OEL installation tree. Remove the shadowing module or pass "
            "--allow-untrusted-plugin-imports for an explicitly trusted scenario."
        ]
    return [
        f"{path}: sealed mode blocks plugin module '{module}'. "
        "Use built-in OEL modules or pass --allow-untrusted-plugin-imports for a trusted scenario."
    ]


def _module_resolves_from_trusted_installation(module: str) -> bool:
    """Resolve a dotted module without importing it and verify its selected origin."""

    search_path = None
    spec = None
    parts = module.split(".")
    for index in range(len(parts)):
        fullname = ".".join(parts[: index + 1])
        spec = PathFinder.find_spec(fullname, search_path)
        if spec is None:
            return False
        if index < len(parts) - 1:
            locations = spec.submodule_search_locations
            if locations is None:
                return False
            search_path = list(locations)
    if spec is None:
        return False
    trusted_root = Path(__file__).resolve().parents[2]
    origins: list[Path] = []
    if spec.origin not in (None, "built-in", "frozen"):
        origins.append(Path(str(spec.origin)).resolve())
    if spec.submodule_search_locations is not None:
        origins.extend(Path(str(location)).resolve() for location in spec.submodule_search_locations)
    return bool(origins) and all(_path_is_within(origin, trusted_root) for origin in origins)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _plugin_pointers(agent: Any) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    flight_software = getattr(agent, "flight_software", None)
    if flight_software is not None and getattr(flight_software, "module", None):
        out.append(("flight_software", flight_software))
    for field_name in (
        "guidance",
        "base_guidance",
        "orbit_control",
        "attitude_control",
        "mission_strategy",
        "mission_execution",
    ):
        pointer = getattr(agent, field_name, None)
        if pointer is not None:
            out.append((field_name, pointer))
    for idx, pointer in enumerate(getattr(agent, "guidance_modifiers", []) or []):
        out.append((f"guidance_modifiers[{idx}]", pointer))
    for idx, pointer in enumerate(getattr(agent, "mission_objectives", []) or []):
        out.append((f"mission_objectives[{idx}]", pointer))
    for idx, pointer in enumerate(getattr(agent, "force_models", []) or []):
        out.append((f"force_models[{idx}]", pointer))
    bridge = getattr(agent, "bridge", None)
    if bridge is not None and getattr(bridge, "enabled", False):
        out.append(("bridge", bridge))
    return out


def _validate_ai_section(
    ai_cfg: dict[str, Any],
    path: str,
    policy: SealedModePolicy,
    *,
    enabled_default: bool,
    offline_operation: bool = False,
) -> list[str]:
    errors: list[str] = []
    if offline_operation:
        return errors
    provider = str(ai_cfg.get("provider", "ollama") or "ollama").strip().lower()
    live_call_enabled = bool(ai_cfg.get("enabled", enabled_default)) and not bool(ai_cfg.get("dry_run", False))
    if live_call_enabled and provider in _HOSTED_AI_PROVIDERS and not policy.allow_hosted_ai:
        errors.append(
            f"{path}.provider: sealed mode blocks hosted AI provider '{provider}'. "
            "Use provider='ollama', set dry_run: true, or pass --allow-hosted-ai for an approved environment."
        )
    endpoint = str(ai_cfg.get("endpoint", "") or "").strip().rstrip("/")
    if endpoint and _default_endpoint(provider) != endpoint and not policy.allow_custom_ai_endpoints:
        errors.append(
            f"{path}.endpoint: sealed mode blocks custom AI endpoint '{endpoint}'. "
            "Use the built-in endpoint or pass --allow-custom-ai-endpoints for a trusted gateway."
        )
    return errors


def _default_endpoint(provider: str) -> str:
    return str(_DEFAULT_AI_ENDPOINTS.get(provider, "") or "").rstrip("/")


def _validate_sil_networking(cfg: Any, policy: SealedModePolicy) -> list[str]:
    if policy.allow_non_loopback_sil:
        return []
    errors: list[str] = []
    for object_id, agent in configured_objects(cfg).items():
        if not getattr(agent, "enabled", False):
            continue
        for field_name in ("flight_software", "bridge"):
            bridge = getattr(agent, field_name, None)
            if bridge is None or (field_name == "bridge" and not getattr(bridge, "enabled", False)):
                continue
            module = str(getattr(bridge, "module", "") or "")
            class_name = str(getattr(bridge, "class_name", "") or "")
            if not module.startswith("integrations.cfs_sil.") and not class_name.startswith("CfsV2"):
                continue
            errors.append(
                f"{object_parameter_prefix(str(object_id))}.{field_name}: sealed mode blocks all cFS/SIL "
                "socket networking, including loopback UDP. Use an in-memory adapter or explicitly opt into "
                "SIL networking for an isolated test environment."
            )
    return errors


def _is_loopback_host(host: str) -> bool:
    parsed = urlparse(host if "://" in host else f"//{host}")
    normalized = (parsed.hostname or host or "").strip().lower()
    return normalized in _LOOPBACK_HOSTS


def _validate_gravity_model_downloads(cfg: Any, policy: SealedModePolicy) -> list[str]:
    if policy.allow_gravity_model_downloads:
        return []
    dynamics = dict(getattr(getattr(cfg, "simulator", None), "dynamics", {}) or {})
    orbit = dict(dynamics.get("orbit", {}) or {})
    spherical_harmonics = dict(orbit.get("spherical_harmonics", {}) or {})
    if not bool(spherical_harmonics.get("enabled", False)):
        return []
    source = str(
        spherical_harmonics.get("source", spherical_harmonics.get("model", "")) or ""
    ).strip().lower()
    has_explicit_coefficients = spherical_harmonics.get("coeff_path") not in (None, "") or (
        spherical_harmonics.get("source_path") not in (None, "")
    )
    if source != "egm96" or has_explicit_coefficients or not bool(spherical_harmonics.get("allow_download", True)):
        return []
    return [
        "simulator.dynamics.orbit.spherical_harmonics.allow_download: sealed mode blocks gravity-model downloads. "
        "Set allow_download: false to require a verified cached copy, provide an explicit coefficient path, or pass "
        "--allow-gravity-model-downloads for an approved environment."
    ]


def _validate_output_retention(
    cfg: Any,
    policy: SealedModePolicy,
    *,
    offline_ai_operation: bool = False,
) -> list[str]:
    if policy.allow_high_detail_outputs:
        return []
    outputs = getattr(cfg, "outputs", None)
    errors: list[str] = []
    if not offline_ai_operation:
        stats = dict(getattr(outputs, "stats", {}) or {})
        if bool(stats.get("save_full_log", True)):
            errors.append(
                "outputs.stats.save_full_log: sealed mode blocks full run logs. "
                "Set save_full_log: false or pass --allow-high-detail-outputs for approved retention."
            )
        review = dict(getattr(outputs, "review", {}) or {})
        if str(review.get("detail", "standard") or "standard").strip().lower() == "full":
            errors.append(
                "outputs.review.detail: sealed mode blocks detail='full'. "
                "Use compact/standard review detail or pass --allow-high-detail-outputs for approved retention."
            )
        monte_carlo = dict(getattr(outputs, "monte_carlo", {}) or {})
        if bool(monte_carlo.get("save_raw_runs", False)):
            errors.append(
                "outputs.monte_carlo.save_raw_runs: sealed mode blocks raw Monte Carlo run payloads. "
                "Set save_raw_runs: false or pass --allow-high-detail-outputs for approved retention."
            )
    ai_report = dict(getattr(outputs, "ai_report", {}) or {})
    if bool(ai_report.get("enabled", False)):
        data_scope = str(ai_report.get("data_scope", "summary_only") or "summary_only").strip().lower()
        if data_scope != "summary_only":
            errors.append(
                "outputs.ai_report.data_scope: sealed mode blocks AI report data_scope values beyond summary_only. "
                "Use summary_only or pass --allow-high-detail-outputs for approved retention."
            )
    return errors
