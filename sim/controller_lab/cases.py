# ruff: noqa: F401,F403,F405,I001
import json

from .bench_common import *
from .config import *
from .variants import *
from .engineering import lint_materialized_scenario

def _apply_benchmark_output_overrides(root: dict[str, Any], outdir: Path, suite: ControllerBenchConfig) -> None:
    outputs = root.setdefault("outputs", {})
    outputs["output_dir"] = str(outdir)
    outputs["mode"] = "save"
    stats = outputs.setdefault("stats", {})
    stats["print_summary"] = bool(suite.print_individual_run_summaries)
    stats["save_json"] = True
    stats["save_full_log"] = bool(suite.save_run_payloads)
    plots = outputs.setdefault("plots", {})
    if suite.disable_plots:
        plots["enabled"] = False
        plots["figure_ids"] = []
    animations = outputs.setdefault("animations", {})
    if suite.disable_animations:
        animations["enabled"] = False
        animations["types"] = []


def _select_controller_bench_variants(
    suite: ControllerBenchConfig,
    compare_names: list[str] | None = None,
    *,
    errors: list[str] | None = None,
) -> list[ControllerVariant]:
    variants = list(suite.variants)
    if not compare_names:
        return variants
    wanted = {str(name).strip() for name in compare_names if str(name).strip()}
    available = {variant.name for variant in variants}
    missing = sorted(wanted - available)
    if missing:
        valid = ", ".join(sorted(available)) or "(none)"
        message = f"Unknown controller variant(s) in --compare: {', '.join(missing)}. Valid variants: {valid}."
        if errors is not None:
            errors.append(message)
            return [variant for variant in variants if variant.name in wanted]
        raise ValueError(message)
    selected = [variant for variant in variants if variant.name in wanted]
    if not selected:
        message = "No matching controller variants selected."
        if errors is not None:
            errors.append(message)
            return []
        raise ValueError(message)
    return selected


def _controller_bench_preflight_error(errors: list[str]) -> ValueError:
    if len(errors) == 1:
        return ValueError(errors[0])
    return ValueError("Controller bench preflight failed:\n- " + "\n- ".join(errors))


def _paths_overlap(left: str, right: str) -> bool:
    left_parts = tuple(part for part in str(left).split(".") if part)
    right_parts = tuple(part for part in str(right).split(".") if part)
    return bool(left_parts) and bool(right_parts) and (
        left_parts == right_parts
        or left_parts[: len(right_parts)] == right_parts
        or right_parts[: len(left_parts)] == left_parts
    )


def _optimization_masking_errors(
    suite: ControllerBenchConfig,
    base_variant: ControllerVariant,
) -> list[str]:
    """Reject variant settings that overwrite a PSO parameter after injection."""

    target = suite.controller_target
    errors: list[str] = []
    for pointer in base_variant.pointers:
        if pointer.object_id == target.object_id and pointer.slot == target.slot:
            errors.append(
                f"optimization parameter injection is masked by variant '{base_variant.name}' pointer "
                f"for controller target {target.object_id}.{target.slot}; remove that specific pointer."
            )
            break

    scenario_prefixes = (
        f"objects.{target.object_id}.{target.slot}",
        f"{target.object_id}.{target.slot}",
    )
    for parameter in suite.optimization.parameters:
        relative_path = str(parameter.path).strip()
        if relative_path.startswith("controller."):
            relative_path = relative_path[len("controller.") :]
        optimized_paths = tuple(f"{prefix}.{relative_path}" for prefix in scenario_prefixes if relative_path)
        for override in base_variant.overrides:
            override_path = str(override.path).strip()
            if any(_paths_overlap(override_path, optimized_path) for optimized_path in optimized_paths):
                errors.append(
                    f"optimization parameter '{parameter.name}' at '{parameter.path}' is masked by variant "
                    f"'{base_variant.name}' override '{override_path}'."
                )
    return errors


def _validate_controller_bench_case(
    suite: ControllerBenchConfig,
    case: ControllerBenchCase,
    variants: list[ControllerVariant],
    *,
    import_plugins: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "name": case.name,
        "config_path": str(case.config_path),
        "description": case.description,
        "study_type": "",
        "target_object": suite.controller_target.object_id,
        "target_slot": suite.controller_target.slot,
        "variant_checks": 0,
        "semantic_lint_findings": [],
    }
    try:
        scenario = load_simulation_yaml(case.config_path)
    except Exception as exc:
        errors.append(f"case '{case.name}': failed to load config '{case.config_path}': {exc}")
        return summary, errors

    study_type = analysis_study_type(scenario)
    summary["study_type"] = study_type
    if study_type != "single_run":
        errors.append(f"case '{case.name}': controller bench cases must be single-run scenarios, got '{study_type}'.")

    object_id = str(suite.controller_target.object_id)
    slot = str(suite.controller_target.slot)
    if slot not in {"flight_software", "base_guidance"}:
        errors.append(
            "controller_target.slot must be 'flight_software' or 'base_guidance', "
            f"got '{slot}'."
        )
    objects = configured_objects(scenario)
    agent = objects.get(object_id)
    if agent is None:
        available = ", ".join(sorted(objects.keys())) or "(none)"
        errors.append(f"case '{case.name}': target object '{object_id}' was not found. Available objects: {available}.")
        return summary, errors
    if not bool(getattr(agent, "enabled", False)):
        errors.append(f"case '{case.name}': target object '{object_id}' is disabled.")
    if not hasattr(agent, slot):
        errors.append(f"case '{case.name}': target object '{object_id}' has no controller slot '{slot}'.")
    if str(getattr(agent, "kind", "satellite")) == "satellite" and slot != "flight_software":
        errors.append(
            f"case '{case.name}': satellite variants must replace a complete flight_software stack; "
            "component-only controller benchmarking is not supported by GNC v2."
        )

    root = scenario.to_dict()
    for variant in variants:
        variant_root = deepcopy(root)
        _apply_controller_variant(variant_root, suite.controller_target, variant)
        try:
            variant_cfg = scenario_config_from_dict(variant_root)
        except Exception as exc:
            errors.append(f"case '{case.name}', variant '{variant.name}': injected scenario config is invalid: {exc}")
            continue
        plugin_errors = validate_scenario_plugins(variant_cfg, import_plugins=import_plugins)
        summary["variant_checks"] = int(summary["variant_checks"]) + 1
        for plugin_error in plugin_errors:
            errors.append(f"case '{case.name}', variant '{variant.name}': {plugin_error}")
        if suite.engineering.enabled and suite.engineering.semantic_lint:
            objectives = case.objectives if case.objectives else suite.objectives
            findings = lint_materialized_scenario(
                variant_root,
                suite=suite,
                variant=variant,
                case_name=case.name,
                objectives=objectives,
            )
            summary["semantic_lint_findings"].extend(findings)
            for finding in findings:
                if finding.get("severity") == "error":
                    errors.append(
                        f"case '{case.name}', variant '{variant.name}': "
                        f"[{finding.get('code')}] {finding.get('message')}"
                    )
    return summary, errors


def validate_controller_bench_config(
    config_path: str | Path,
    *,
    compare_names: list[str] | None = None,
    import_plugins: bool = True,
) -> dict[str, Any]:
    require_pro_feature(FEATURE_CONTROLLER_BENCH)
    from tempfile import TemporaryDirectory

    validation_workspace = TemporaryDirectory(prefix="oel-controller-bench-validate-")
    suite = load_controller_bench_config(config_path, generated_cases_root=validation_workspace.name)
    errors: list[str] = []
    if not suite.cases:
        errors.append("Controller bench suite must define at least one case.")
    if not suite.variants:
        errors.append("Controller bench suite must define at least one controller variant.")
    if suite.optimization.enabled and compare_names:
        errors.append("Controller bench optimization cannot be combined with --compare.")
    selected = _select_controller_bench_variants(suite, compare_names, errors=errors) if suite.variants else []
    if suite.optimization.enabled and suite.optimization.base_variant not in {
        variant.name for variant in suite.variants
    }:
        errors.append("optimization.base_variant must name one declared controller variant.")
    if suite.optimization.enabled:
        base_variant = next(
            (variant for variant in suite.variants if variant.name == suite.optimization.base_variant),
            None,
        )
        if base_variant is not None:
            errors.extend(_optimization_masking_errors(suite, base_variant))
    seen_variants: dict[str, str] = {}
    for variant in selected:
        signature = json.dumps(
            {
                "pointer": variant.pointer,
                "pointers": [
                    {"object_id": item.object_id, "slot": item.slot, "pointer": item.pointer}
                    for item in variant.pointers
                ],
                "overrides": [{"path": item.path, "value": item.value} for item in variant.overrides],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        duplicate = seen_variants.get(signature)
        if duplicate is not None:
            errors.append(
                f"Controller variants '{duplicate}' and '{variant.name}' are behaviorally identical; "
                "change the stack parameters or remove the duplicate."
            )
        else:
            seen_variants[signature] = variant.name
    case_summaries: list[dict[str, Any]] = []
    if suite.cases and selected:
        for case in suite.cases:
            objectives = case.objectives if case.objectives else suite.objectives
            try:
                from sim.controller_lab.objectives import _resolve_metric_specs, _resolve_pass_criteria

                _resolve_metric_specs(suite, case)
                criteria = _resolve_pass_criteria(suite, case, objectives)
                for criterion in criteria:
                    if criterion.op not in {"<=", "<", ">=", ">", "==", "!="}:
                        raise ValueError(f"Unsupported pass-criterion op: {criterion.op}")
            except (TypeError, ValueError) as exc:
                errors.append(f"case '{case.name}': objective/pass-criteria contract is invalid: {exc}")
            case_summary, case_errors = _validate_controller_bench_case(
                suite,
                case,
                selected,
                import_plugins=import_plugins,
            )
            case_summaries.append(case_summary)
            errors.extend(case_errors)

    scheduled_runs = len(suite.cases) * len(selected)
    semantic_findings = [
        finding
        for case_summary in case_summaries
        for finding in list(case_summary.get("semantic_lint_findings", []) or [])
    ]
    workspace_root = Path(validation_workspace.name).resolve()
    for case_summary in case_summaries:
        case_path = Path(str(case_summary.get("config_path", "") or ""))
        if case_path.is_absolute() and case_path.is_relative_to(workspace_root):
            case_summary["config_path"] = f"generated://controller-bench/{case_summary['name']}"
    envelope_manifest = deepcopy(dict(suite.envelope_manifest or {}))
    for generated in list(envelope_manifest.get("generated_cases", []) or []):
        generated["config_path"] = f"generated://controller-bench/{generated.get('name', 'case')}"
    result = {
        "passed": not errors,
        "errors": errors,
        "suite_name": suite.suite_name,
        "description": suite.description,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "output_dir": str(suite.output_dir),
        "plot_mode": suite.plot_mode,
        "controller_target": {
            "object_id": suite.controller_target.object_id,
            "slot": suite.controller_target.slot,
        },
        "cases_total": len(suite.cases),
        "variants_total": len(suite.variants),
        "variants_selected": len(selected),
        "scheduled_runs": scheduled_runs,
        "selected_variants": [variant.name for variant in selected],
        "cases": case_summaries,
        "optimization_enabled": bool(suite.optimization.enabled),
        "envelope": envelope_manifest,
        "semantic_lint": {
            "enabled": bool(suite.engineering.enabled and suite.engineering.semantic_lint),
            "finding_count": len(semantic_findings),
            "error_count": sum(item.get("severity") == "error" for item in semantic_findings),
            "warning_count": sum(item.get("severity") == "warning" for item in semantic_findings),
            "findings": semantic_findings,
        },
    }
    validation_workspace.cleanup()
    return result

__all__ = [name for name in globals() if not name.startswith("__")]
