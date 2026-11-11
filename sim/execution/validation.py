from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.config import SimulationScenarioConfig, scenario_config_from_dict, validate_scenario_plugins
from sim.execution.study import analysis_study_type


def prepare_batch_run_configs(cfg: SimulationScenarioConfig) -> list[dict[str, Any]]:
    study_type = analysis_study_type(cfg)
    if study_type not in {"monte_carlo", "sensitivity"}:
        return []
    root = cfg.to_dict()
    outdir = Path(cfg.outputs.output_dir)
    if study_type == "sensitivity":
        from sim.execution.sensitivity import prepare_sensitivity_runs

        sensitivity_method = str(cfg.analysis.sensitivity.method or "one_at_a_time").strip().lower()
        return prepare_sensitivity_runs(cfg=cfg, root=root, outdir=outdir, sensitivity_method=sensitivity_method)

    from sim.execution.campaigns import prepare_monte_carlo_runs

    return prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)


def validate_generated_batch_configs(
    cfg: SimulationScenarioConfig,
    *,
    import_plugins: bool = True,
    source_path: str | Path | None = None,
    path_policy: Any | None = None,
) -> dict[str, Any]:
    strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
    try:
        prepared = prepare_batch_run_configs(cfg)
    except Exception as exc:
        return {
            "run_count": 0,
            "errors": [
                {
                    "iteration": None,
                    "parameter_path": None,
                    "parameter_value": None,
                    "error": str(exc),
                }
            ],
        }

    if analysis_study_type(cfg) == "sensitivity":
        from sim.execution.sensitivity import validate_prepared_sensitivity_runs

        return validate_prepared_sensitivity_runs(
            prepared=prepared,
            strict_plugins=strict_plugins,
            import_plugins=import_plugins,
            source_path=source_path,
            path_policy=path_policy,
        )

    report, _validated = validate_prepared_monte_carlo_runs(
        prepared=prepared,
        strict_plugins=strict_plugins,
        import_plugins=import_plugins,
        source_path=source_path,
        path_policy=path_policy,
    )
    return report


def validate_prepared_monte_carlo_runs(
    *,
    prepared: list[dict[str, Any]],
    strict_plugins: bool,
    import_plugins: bool = True,
    source_path: str | Path | None = None,
    path_policy: Any | None = None,
) -> tuple[dict[str, Any], list[SimulationScenarioConfig | None]]:
    """Validate every sampled scenario before a Monte Carlo worker starts."""

    errors: list[dict[str, Any]] = []
    validated: list[SimulationScenarioConfig | None] = []
    for item in prepared:
        iteration = int(item.get("iteration", len(validated)))
        sampled = dict(item.get("sampled_parameters", {}) or {})
        try:
            run_cfg = scenario_config_from_dict(
                dict(item.get("config_dict", {}) or {}),
                source_path=source_path,
                path_policy=path_policy,
            )
            plugin_errors = validate_scenario_plugins(run_cfg, import_plugins=import_plugins) if strict_plugins else []
        except Exception as exc:
            validated.append(None)
            plugin_errors = [str(exc)]
        else:
            validated.append(run_cfg)
        for error in plugin_errors:
            errors.append(
                {
                    "iteration": iteration,
                    "parameter_path": next(iter(sampled), None) if len(sampled) == 1 else None,
                    "parameter_value": next(iter(sampled.values()), None) if len(sampled) == 1 else None,
                    "sampled_parameters": sampled,
                    "error": str(error),
                }
            )
    return {"run_count": len(prepared), "errors": errors}, validated


def raise_for_monte_carlo_preflight_errors(report: dict[str, Any]) -> None:
    errors = list(report.get("errors", []) or [])
    if not errors:
        return
    details = [
        f"iteration {error.get('iteration')} ({dict(error.get('sampled_parameters', {}) or {})}): {error.get('error')}"
        for error in errors[:10]
    ]
    if len(errors) > 10:
        details.append(f"... and {len(errors) - 10} more error(s)")
    raise ValueError("Monte Carlo generated-run preflight failed:\n- " + "\n- ".join(details))
