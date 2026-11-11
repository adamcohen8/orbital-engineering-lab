from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from sim.scale.campaign_execution import build_campaign_trial_result
from sim.scale.planner import ScalePlannerConfig, resolve_selected_object_ids
from sim.scale.store import (
    CampaignTrialResultRecord,
    SensitivityCampaignRecord,
    SensitivityTrialRecord,
    load_current_tle_states,
)


def _run_artifact_stem(*, backend: str, run_id: str) -> str:
    return f"monte_carlo_{backend}_{_run_id_suffix(run_id)}"


def _run_id_suffix(run_id: str) -> str:
    suffix = str(run_id).rsplit(":", 1)[-1]
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in suffix)
    return safe[:16] if safe else _digest({"run_id": run_id})[:16]


def _execution_contract(backend: str) -> str:
    if backend == "serial":
        return "sgp4_monte_carlo_serial_v1"
    if backend == "parallel_cpu":
        return "sgp4_monte_carlo_parallel_cpu_v1"
    return f"sgp4_monte_carlo_{backend}_v1"


def build_monte_carlo_campaign(
    cfg: ScalePlannerConfig,
) -> tuple[SensitivityCampaignRecord, list[SensitivityTrialRecord]]:
    if cfg.monte_carlo_target != "selected_objects":
        raise ValueError("Scale Monte Carlo v1 supports only monte_carlo.target: selected_objects.")
    if cfg.monte_carlo_trial_count <= 0:
        raise ValueError("scale_catalog.monte_carlo.trial_count must be positive.")
    target_states = load_current_tle_states(
        cfg.store_path, object_ids=resolve_selected_object_ids(cfg)
    )
    target_object_ids = tuple(state.object_id for state in target_states)
    target_state_inputs = [
        {
            "object_id": state.object_id,
            "od_solution_id": state.od_solution_id,
            "input_digest": state.input_digest,
        }
        for state in target_states
    ]
    trial_parameters = _sample_trial_parameter_sets(cfg)
    campaign_digest = _digest(
        {
            "workflow_name": cfg.name,
            "campaign_type": cfg.monte_carlo_campaign_type,
            "target": cfg.monte_carlo_target,
            "uncertainty_interpretation": cfg.monte_carlo_uncertainty_interpretation,
            "seed": cfg.monte_carlo_seed,
            "trial_count": cfg.monte_carlo_trial_count,
            "parameters": list(cfg.monte_carlo_parameters),
            "derived_parameters": list(cfg.monte_carlo_derived_parameters),
            "frame": cfg.monte_carlo_frame,
            "trial_builder": cfg.monte_carlo_trial_builder,
            "outputs": cfg.monte_carlo_outputs,
            "propagation": {
                "start_jd_utc": cfg.start_jd_utc,
                "duration_s": cfg.duration_s,
                "step_s": cfg.step_s,
            },
            "target_object_ids": list(target_object_ids),
            "target_state_inputs": target_state_inputs,
        }
    )
    campaign_id = f"monte_carlo:{campaign_digest}"
    campaign = SensitivityCampaignRecord(
        campaign_id=campaign_id,
        workflow_name=cfg.name,
        campaign_type=cfg.monte_carlo_campaign_type,
        target=cfg.monte_carlo_target,
        uncertainty_interpretation=cfg.monte_carlo_uncertainty_interpretation,
        seed=cfg.monte_carlo_seed,
        requested_trial_count=cfg.monte_carlo_trial_count,
        realized_trial_count=len(trial_parameters),
        sweep_mode="random",
        parameters=cfg.monte_carlo_parameters,
        frame=cfg.monte_carlo_frame,
        trial_builder=cfg.monte_carlo_trial_builder,
        outputs=cfg.monte_carlo_outputs,
        output_dir=str(cfg.monte_carlo_output_dir),
        input_digest=campaign_digest,
        status="planned",
        warnings=_campaign_warnings(cfg, target_object_ids=target_object_ids),
    )
    trials = [
        _trial_record(
            campaign_id=campaign_id,
            trial_index=index,
            parameters=params,
            target_object_ids=target_object_ids,
            frame=cfg.monte_carlo_frame,
            trial_builder=cfg.monte_carlo_trial_builder,
            campaign_digest=campaign_digest,
        )
        for index, params in enumerate(trial_parameters)
    ]
    return campaign, trials


def _trial_result(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trial: SensitivityTrialRecord,
    run_id: str,
    backend: str,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
) -> CampaignTrialResultRecord:
    return build_campaign_trial_result(
        cfg,
        campaign=campaign,
        trial=trial,
        run_id=run_id,
        backend=backend,
        trial_builder=cfg.monte_carlo_trial_builder,
        result_prefix="monte_carlo_result",
        workflow_label="Monte Carlo",
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
    )


def _sample_trial_parameter_sets(cfg: ScalePlannerConfig) -> list[dict[str, Any]]:
    _validate_parameter_definitions(
        cfg.monte_carlo_parameters,
        cfg.monte_carlo_derived_parameters,
    )
    rng = np.random.default_rng(int(cfg.monte_carlo_seed))
    out: list[dict[str, Any]] = []
    for _ in range(cfg.monte_carlo_trial_count):
        row: dict[str, Any] = {}
        for parameter in cfg.monte_carlo_parameters:
            name = _parameter_name(parameter, label="Monte Carlo parameter")
            row[name] = _sample_parameter(parameter, rng)
        _apply_derived_parameters(row, cfg.monte_carlo_derived_parameters)
        out.append(row)
    return out


def _validate_parameter_definitions(
    parameters: tuple[dict[str, Any], ...],
    derived_parameters: tuple[dict[str, Any], ...],
) -> None:
    names: set[str] = set()
    for parameter in parameters:
        name = _parameter_name(parameter, label="Monte Carlo parameter")
        if name in names:
            raise ValueError("Monte Carlo parameter names must be unique.")
        names.add(name)

        mode = _parameter_mode(parameter)
        if mode == "fixed" and "value" not in parameter:
            raise ValueError(f"Monte Carlo parameter {name!r} fixed mode requires a value key.")
        _validate_parameter_sampling(parameter, name=name, mode=mode)

    derived_names: set[str] = set()
    available_names = set(names)
    for derived in derived_parameters:
        name = _parameter_name(derived, label="Monte Carlo derived parameter")
        if name in derived_names:
            raise ValueError("Monte Carlo derived parameter names must be unique.")
        if name in names:
            raise ValueError("Monte Carlo parameter and derived parameter names must be unique.")
        terms = [str(term).strip() for term in list(derived.get("terms", []) or []) if str(term).strip()]
        if not terms:
            raise ValueError(f"Monte Carlo derived parameter {name!r} requires terms.")
        missing_terms = [term for term in terms if term not in available_names]
        if missing_terms:
            raise ValueError(
                f"Monte Carlo derived parameter {name!r} references unknown term(s): "
                f"{', '.join(missing_terms)}."
            )
        derived_names.add(name)
        available_names.add(name)


def _parameter_name(parameter: dict[str, Any], *, label: str) -> str:
    raw_name = parameter.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"Every {label} requires a non-empty name.")
    return raw_name.strip()


def _parameter_mode(parameter: dict[str, Any]) -> str:
    if "mode" in parameter:
        raw_mode = parameter["mode"]
    elif "distribution" in parameter:
        raw_mode = parameter["distribution"]
    else:
        raw_mode = "fixed"
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise ValueError("Monte Carlo parameter mode must be a non-empty string.")
    mode = raw_mode.strip().lower()
    if mode not in {"fixed", "choice", "uniform", "normal"}:
        raise ValueError(f"Unsupported Monte Carlo parameter mode: {mode}")
    return mode


def _finite_parameter_float(value: Any, *, parameter_name: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Monte Carlo parameter {parameter_name!r} {field_name} must be a finite number."
        ) from exc
    if not np.isfinite(parsed):
        raise ValueError(
            f"Monte Carlo parameter {parameter_name!r} {field_name} must be a finite number."
        )
    return parsed


def _validate_parameter_sampling(parameter: dict[str, Any], *, name: str, mode: str) -> None:
    if mode == "choice":
        values = list(parameter.get("options", parameter.get("values", [])) or [])
        if not values:
            raise ValueError(f"Monte Carlo parameter {name!r} choice mode requires options.")
        for value in values:
            if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(value):
                raise ValueError(f"Monte Carlo parameter {name!r} choice values must be finite.")
        return
    if mode == "uniform":
        if parameter.get("low") is None or parameter.get("high") is None:
            raise ValueError(f"Monte Carlo parameter {name!r} uniform mode requires low/high.")
        low = _finite_parameter_float(parameter["low"], parameter_name=name, field_name="low")
        high = _finite_parameter_float(parameter["high"], parameter_name=name, field_name="high")
        if high < low:
            raise ValueError(f"Monte Carlo parameter {name!r} uniform mode requires high >= low.")
        return
    if mode == "normal":
        if parameter.get("mean") is None or parameter.get("std") is None:
            raise ValueError(f"Monte Carlo parameter {name!r} normal mode requires mean/std.")
        _finite_parameter_float(parameter["mean"], parameter_name=name, field_name="mean")
        std = _finite_parameter_float(parameter["std"], parameter_name=name, field_name="std")
        if std < 0.0:
            raise ValueError(f"Monte Carlo parameter {name!r} normal mode requires std >= 0.")


def _sample_parameter(parameter: dict[str, Any], rng: np.random.Generator) -> Any:
    mode = _parameter_mode(parameter)
    name = str(parameter.get("name", "")).strip() or "<unnamed>"
    _validate_parameter_sampling(parameter, name=name, mode=mode)
    if mode == "fixed":
        if "value" not in parameter:
            raise ValueError(
                f"Monte Carlo parameter {parameter.get('name')!r} fixed mode requires a value key."
            )
        return parameter.get("value")
    if mode == "choice":
        values = list(parameter.get("options", parameter.get("values", [])) or [])
        return values[int(rng.integers(0, len(values)))]
    if mode == "uniform":
        low = _finite_parameter_float(parameter["low"], parameter_name=name, field_name="low")
        high = _finite_parameter_float(parameter["high"], parameter_name=name, field_name="high")
        return float(rng.uniform(low, high))
    if mode == "normal":
        mean = _finite_parameter_float(parameter["mean"], parameter_name=name, field_name="mean")
        std = _finite_parameter_float(parameter["std"], parameter_name=name, field_name="std")
        return float(rng.normal(mean, std))
    raise ValueError(f"Unsupported Monte Carlo parameter mode: {mode}")


def _apply_derived_parameters(row: dict[str, Any], derived_parameters: tuple[dict[str, Any], ...]) -> None:
    names = set(row)
    for derived in derived_parameters:
        name = _parameter_name(derived, label="Monte Carlo derived parameter")
        if name in names:
            raise ValueError("Monte Carlo parameter and derived parameter names must be unique.")
        names.add(name)
        op = str(derived.get("op", "add") or "add").strip().lower()
        terms = [str(term).strip() for term in list(derived.get("terms", []) or []) if str(term).strip()]
        if op != "add":
            raise ValueError("Monte Carlo derived parameters v1 supports only op: add.")
        if not terms:
            raise ValueError(f"Monte Carlo derived parameter {name!r} requires terms.")
        missing_terms = [term for term in terms if term not in row]
        if missing_terms:
            raise ValueError(
                f"Monte Carlo derived parameter {name!r} references unknown term(s): "
                f"{', '.join(missing_terms)}."
            )
        try:
            derived_value = float(sum(float(row[term]) for term in terms))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Monte Carlo derived parameter {name!r} terms must be numeric.") from exc
        if not np.isfinite(derived_value):
            raise ValueError(f"Monte Carlo derived parameter {name!r} must be finite.")
        row[name] = derived_value


def _trial_record(
    *,
    campaign_id: str,
    trial_index: int,
    parameters: dict[str, Any],
    target_object_ids: tuple[str, ...],
    frame: dict[str, Any],
    trial_builder: dict[str, Any],
    campaign_digest: str,
) -> SensitivityTrialRecord:
    digest = _digest(
        {
            "campaign_digest": campaign_digest,
            "trial_index": trial_index,
            "parameters": parameters,
            "target_object_ids": list(target_object_ids),
        }
    )
    builder_kind = str(trial_builder.get("kind", "external") or "external")
    return SensitivityTrialRecord(
        trial_id=f"monte_carlo_trial:{digest}",
        campaign_id=campaign_id,
        trial_index=trial_index,
        parameters=parameters,
        target_object_ids=target_object_ids,
        frame=frame,
        builder_status=f"pending_{builder_kind}_builder",
        trial_product_refs=(),
        input_digest=digest,
        status="planned",
    )


def _campaign_warnings(cfg: ScalePlannerConfig, *, target_object_ids: tuple[str, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    if cfg.monte_carlo_uncertainty_interpretation != "analyst_defined_monte_carlo":
        warnings.append("Only analyst-defined Monte Carlo is supported in v1.")
    if not target_object_ids:
        warnings.append("No current target objects were available for this Monte Carlo campaign.")
    if not cfg.monte_carlo_trial_builder:
        warnings.append("No trial_builder was provided; trials are sampled parameter rows only.")
    return tuple(warnings)
def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
