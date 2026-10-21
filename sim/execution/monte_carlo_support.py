"""Shared scalar helpers; campaign-analysis entry points require OEL Pro."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.presets.thrusters import BASIC_CHEMICAL_BOTTOM_Z
from sim.pro_features import require_pro_feature


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def quantile_stats(
    values: list[float] | np.ndarray, quantiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
        for q in quantiles:
            out[f"p{int(q)}"] = float("nan")
        return out
    out = {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for q in quantiles:
        out[f"p{int(q)}"] = float(np.percentile(arr, q))
    return out


def coerce_numeric_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in value.items():
        fv = safe_float(v)
        if np.isfinite(fv):
            out[str(k)] = fv
    return out


def get_git_commit_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return out or None


def infer_model_profile(root_cfg: dict[str, Any]) -> str:
    metadata = dict(root_cfg.get("metadata", {}) or {})
    simulator = dict(root_cfg.get("simulator", {}) or {})
    dynamics = dict(simulator.get("dynamics", {}) or {})
    environment = dict(simulator.get("environment", {}) or {})
    for src in (metadata, simulator, dynamics, environment):
        for key in ("profile", "profile_name", "fidelity_profile"):
            val = src.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return "custom"


def resolve_satellite_isp_s(specs: dict[str, Any]) -> float:
    if "isp_s" in specs:
        return float(specs.get("isp_s", 0.0))
    if "thruster_isp_s" in specs:
        return float(specs.get("thruster_isp_s", 0.0))
    thr = str(specs.get("thruster", "")).strip().upper()
    if thr in ("BASIC_CHEMICAL_BOTTOM_Z", "BASIC_CHEMICAL_Z_BOTTOM"):
        return float(BASIC_CHEMICAL_BOTTOM_Z.isp_s)
    return 0.0


def satellite_initial_delta_v_budget_m_s(agent_cfg: Any) -> float:
    specs = dict(getattr(agent_cfg, "specs", {}) or {})
    dry_mass_kg = safe_float(specs.get("dry_mass_kg"))
    fuel_mass_kg = safe_float(specs.get("fuel_mass_kg"))
    if not (np.isfinite(dry_mass_kg) and np.isfinite(fuel_mass_kg)):
        return float("nan")
    if dry_mass_kg <= 0.0 or fuel_mass_kg < 0.0:
        return float("nan")
    m0_kg = dry_mass_kg + fuel_mass_kg
    if m0_kg <= dry_mass_kg:
        return 0.0
    isp_s = resolve_satellite_isp_s(specs)
    if isp_s <= 0.0:
        return float("nan")
    return float(isp_s * 9.80665 * np.log(m0_kg / dry_mass_kg))


def assess_mc_run(*, run_entry: dict[str, Any], gates: Any, success_termination_reasons: set[str], require_rocket_insertion: bool) -> dict[str, Any]:
    require_pro_feature('campaigns')


def build_parameter_sensitivity_rankings(run_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require_pro_feature('campaigns')


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def extract_baseline_metrics(payload: dict[str, Any]) -> dict[str, float]:
    require_pro_feature('campaigns')


def aggregate_knowledge_consistency_from_runs(run_details: list[dict[str, Any]]) -> dict[str, Any]:
    require_pro_feature('campaigns')


def aggregate_knowledge_detection_from_runs(run_details: list[dict[str, Any]]) -> dict[str, Any]:
    require_pro_feature('campaigns')


def build_baseline_comparison(current_payload: dict[str, Any], baseline_payload: dict[str, Any]) -> dict[str, Any]:
    require_pro_feature('campaigns')


def write_commander_brief_markdown(path: Path, brief: dict[str, Any], *, title: str='Monte Carlo Commander Brief') -> None:
    require_pro_feature('campaigns')


def mc_initial_relative_ric_curv_samples(cfg: SimulationScenarioConfig, run_details: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    require_pro_feature('campaigns')


_safe_float = safe_float


_quantile_stats = quantile_stats


_coerce_numeric_map = coerce_numeric_map


_get_git_commit_sha = get_git_commit_sha


_infer_model_profile = infer_model_profile


_assess_mc_run = assess_mc_run


_build_parameter_sensitivity_rankings = build_parameter_sensitivity_rankings


_load_json_file = load_json_file


_aggregate_knowledge_consistency_from_runs = aggregate_knowledge_consistency_from_runs


_aggregate_knowledge_detection_from_runs = aggregate_knowledge_detection_from_runs


_build_baseline_comparison = build_baseline_comparison


_write_commander_brief_markdown = write_commander_brief_markdown


_mc_initial_relative_ric_curv_samples = mc_initial_relative_ric_curv_samples


_satellite_initial_delta_v_budget_m_s = satellite_initial_delta_v_budget_m_s
