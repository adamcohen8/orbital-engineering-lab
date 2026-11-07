"""Non-executing scenario validation receipts for study-plan preflight."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sim.api import SimulationWorkspace
from sim.config.scenario.loader import load_simulation_yaml_bytes
from sim.resource_limits import estimate_resource_requirements
from sim.security import ConfigPathPolicy

from .contracts import canonical_sha256

MAX_CONFIG_BYTES = 2_000_000


def validate_config_path(
    config_ref: str,
    config_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    content: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Safely validate one configuration and return a receipt plus normalized data.

    ``content`` is an optional descriptor-bound snapshot supplied by an
    adapter that already authorized the input. When present, validation parses
    those exact bytes instead of reopening ``config_path``; the path remains
    provenance for relative references.
    """

    source = Path(config_path).expanduser().resolve()
    root = Path(workspace_root or Path.cwd()).expanduser().resolve()
    if content is None and not source.is_file():
        raise FileNotFoundError("The selected study configuration is not a regular file.")
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise PermissionError("Study configuration must be inside the selected local workspace.") from exc
    raw = bytes(content) if content is not None else source.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("Study configuration exceeds the 2 MB validation boundary.")

    workspace = SimulationWorkspace(
        workspace_root=root,
        read_roots=(root,),
        write_roots=(root,),
        allow_config_dir_writes=True,
    )
    try:
        if content is None:
            config = workspace.load(source)
        else:
            path_policy = ConfigPathPolicy.default(
                config_path=source,
                workspace_root=root,
                read_roots=(root,),
                write_roots=(root,),
                allow_config_dir_writes=True,
                require_secure_reads=True,
            )
            config = load_simulation_yaml_bytes(
                raw,
                source_path=source,
                path_policy=path_policy,
            )
        validation = workspace.validate(config, import_plugins=False)
        normalized = config.to_dict()
        # ``SimulationWorkspace.load`` returns the public ``SimulationConfig``
        # wrapper, while ``load_simulation_yaml_bytes`` returns the underlying
        # ``SimulationScenarioConfig``.  The resource estimator accepts the
        # latter directly; calling ``to_scenario_config`` unconditionally makes
        # the descriptor-bound path fail closed for every valid MCP config.
        to_scenario_config = getattr(config, "to_scenario_config", None)
        scenario = to_scenario_config() if callable(to_scenario_config) else config
        estimate = asdict(estimate_resource_requirements(scenario))
        estimate["action"] = (
            "refuse"
            if estimate["risk"] == "unsafe"
            else "advisory" if estimate["risk"] in {"moderate", "heavy"} else "proceed"
        )
        valid = bool(validation.get("ok"))
        errors = [str(item) for item in validation.get("errors", [])]
    except Exception as exc:
        normalized = None
        estimate = {"action": "refuse", "risk": "unsafe", "reason": "configuration_invalid"}
        valid = False
        errors = [str(exc)]

    receipt = {
        "schema": "oel.study_config_validation_receipt.v1",
        "config_ref": str(config_ref),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_config_sha256": None if normalized is None else canonical_sha256(normalized),
        "valid": valid,
        "safe_validation_only": True,
        "plugins_imported": False,
        "execution_advanced": False,
        "charge_created": False,
        "errors": errors,
        "resource_estimate": estimate,
    }
    return receipt, normalized


__all__ = ["MAX_CONFIG_BYTES", "validate_config_path"]
