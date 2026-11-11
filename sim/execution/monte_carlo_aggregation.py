# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *
from .monte_carlo_campaign_execution import *
from .monte_carlo_checkpoints import *
from .monte_carlo_review import *
from .metrics import extract_study_metrics, normalize_study_metric_specs
from .campaign_retention import (
    policy_from_config, preflight_campaign_retention, retire_iteration_review, write_campaign_dataset,
)

def run_monte_carlo_campaign(
    *,
    config_path: str | Path,
    cfg: SimulationScenarioConfig,
    path_policy: Any | None = None,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Run a Monte Carlo campaign and produce the canonical aggregate payload."""
    require_pro_feature(FEATURE_CAMPAIGNS)
    if not can_run_monte_carlo_campaign(cfg):
        raise ValueError("run_monte_carlo_campaign only supports Monte Carlo configs.")

    root = cfg.to_dict()
    # Report writers return paths rooted here; make them unambiguous when the
    # review manifest resolves relative artifact names against the run directory.
    outdir = Path(cfg.outputs.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    mc_out_cfg = dict(cfg.outputs.monte_carlo or {})
    relative_range_writer = MonteCarloRelativeRangePlotWriter.from_config(
        outdir=outdir,
        plots_cfg=dict(cfg.outputs.plots or {}),
        mc_out_cfg=mc_out_cfg,
        scenario_name=str(cfg.scenario_name or ""),
    )
    repo_root = Path(__file__).resolve().parents[2]
    runs = []
    run_details: list[dict[str, Any]] = []
    closest_approach_km_runs: list[float] = []
    duration_runs_s: list[float] = []
    guardrail_event_runs: list[int] = []
    total_dv_runs_m_s: list[float] = []
    failure_mode_counts: dict[str, int] = {}
    success_termination_reasons = {
        str(x) for x in (mc_out_cfg.get("success_termination_reasons", ["rocket_orbit_insertion"]) or [])
    }
    require_rocket_insertion = bool(mc_out_cfg.get("require_rocket_insertion", False))
    raw_gates = mc_out_cfg.get("gates", {}) or {}
    gates: Any = raw_gates if isinstance(raw_gates, list) else dict(raw_gates or {})
    metric_specs = normalize_study_metric_specs(cfg.analysis.metrics)
    retention_policy = policy_from_config(root)
    preflight_campaign_retention(retention_policy, int(cfg.monte_carlo.iterations))
    dataset_records: list[dict[str, Any]] = []
    dv_budget_m_s_by_object: dict[str, float] = {}
    for object_id, agent_cfg in iter_object_sections(cfg, enabled_only=True):
        if not bool(getattr(agent_cfg, "enabled", False)) or str(getattr(agent_cfg, "kind", "satellite")) == "rocket":
            continue
        dv_budget = satellite_initial_delta_v_budget_m_s(agent_cfg)
        if np.isfinite(dv_budget):
            dv_budget_m_s_by_object[str(object_id)] = float(dv_budget)
    varies_metadata_seed = any(str(v.parameter_path) == "metadata.seed" for v in cfg.monte_carlo.variations)
    total_iters = int(cfg.monte_carlo.iterations)
    parallel_enabled = bool(cfg.monte_carlo.parallel_enabled)
    strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))

    campaign_result = run_monte_carlo_runs(
        cfg=cfg,
        root=root,
        outdir=outdir,
        strict_plugins=strict_plugins,
        mc_out_cfg=mc_out_cfg,
        step_callback=step_callback,
        batch_callback=batch_callback,
        batch_progress_callback=batch_progress_callback,
        relative_range_writer=relative_range_writer,
        backend=backend,
        source_path=config_path if path_policy is not None else None,
        path_policy=path_policy,
    )
    prepared = list(campaign_result.get("prepared", []) or [])
    completed = dict(campaign_result.get("completed", {}) or {})
    parallel_active = bool(campaign_result.get("parallel_active", False))
    parallel_workers = int(campaign_result.get("parallel_workers", 1) or 1)
    parallel_backend = str(campaign_result.get("parallel_backend", "serial") or "serial")
    parallel_fallback_reason = campaign_result.get("parallel_fallback_reason")
    checkpoint_meta = dict(campaign_result.get("checkpoint", {}) or {})

    for p in sorted(prepared, key=lambda x: int(x["iteration"])):
        i = int(p["iteration"])
        cres = dict(completed.get(i, {}) or {})
        ro_summary = dict(cres.get("summary", {}) or {})
        closest_approach_km = safe_float(cres.get("closest_approach_km"))
        closest_approach_km_runs.append(closest_approach_km)
        run_entry = dict(cres.get("payload", {}) or {})
        run_entry["summary"] = ro_summary
        run_entry["closest_approach_km"] = closest_approach_km
        run_entry["derived"] = dict(cres.get("derived", {}) or {})
        if "metric_cache" in cres:
            run_entry["metric_cache"] = dict(cres["metric_cache"])
        study_metrics = extract_study_metrics(run_entry, metric_specs)
        assessment = assess_mc_run(
            run_entry=run_entry,
            gates=gates,
            success_termination_reasons=success_termination_reasons,
            require_rocket_insertion=require_rocket_insertion,
        )
        if retention_policy:
            run_dir = outdir / f"mc_run_{i:04d}"
            keep_full = (retention_policy.get("mode", "stats") == "full"
                         or i in retention_policy.get("full_iterations", [])
                         or bool(retention_policy.get("full_on_failure", False) and not assessment["pass"]))
            review_path = run_dir / "review" / "run.sqlite"
            if keep_full and not review_path.is_file():
                raise FileNotFoundError(f"Retained full campaign review store missing: {review_path}")
            if not keep_full:
                retire_iteration_review(
                    run_dir, expected_sha256=dict(cres.get("retained_evidence", {}) or {}).get("source_review_sha256")
                )
            dataset_records.append({
                "iteration": i, "seed": int(p["seed"]),
                "sampled_parameters": dict(p["sampled_parameters"]),
                "bundle_options": dict(p.get("bundle_options", {}) or {}),
                "comparison": p.get("comparison"), "config_hash": p.get("config_hash"),
                "status": "completed", "passed": bool(assessment["pass"]),
                "failure_reasons": list(assessment["fail_reasons"]),
                "metrics": study_metrics,
                "retained_evidence": dict(cres.get("retained_evidence", {}) or {}),
                "full_review": str(review_path.relative_to(outdir)) if keep_full else None,
            })
        duration_runs_s.append(float(assessment["duration_s"]))
        guardrail_event_runs.append(int(assessment["guardrail_events"]))
        total_dv_runs_m_s.append(float(assessment["total_dv_m_s_total"]))
        run_detail = {
            "iteration": i,
            "seed": int(p["seed"]),
            "sampled_parameters": dict(p["sampled_parameters"]),
            "bundle_options": dict(p.get("bundle_options", {}) or {}),
            "comparison": p.get("comparison"),
            "sampling_attempts": int(p.get("sampling_attempts", 1)),
            "summary": ro_summary,
            "derived": dict(cres.get("derived", {}) or {}),
            "metrics": study_metrics,
            "pass": bool(assessment["pass"]),
            "fail_reasons": list(assessment["fail_reasons"]),
            "duration_s": float(assessment["duration_s"]),
            "closest_approach_km": float(assessment["closest_approach_km"])
            if np.isfinite(safe_float(assessment["closest_approach_km"]))
            else float("nan"),
            "guardrail_events": int(assessment["guardrail_events"]),
            "termination_reason": str(assessment["termination_reason"]),
            "terminated_early": bool(assessment["terminated_early"]),
            "rocket_insertion_achieved": bool(assessment["rocket_insertion_achieved"]),
            "total_dv_m_s_total": float(assessment["total_dv_m_s_total"]),
            "total_dv_m_s_by_object": dict(assessment["total_dv_m_s_by_object"]),
            "metric_gate_results": list(assessment.get("metric_gate_results", []) or []),
            "gate_metrics": dict(assessment.get("gate_metrics", {}) or {}),
            "delta_v_remaining_m_s_by_object": {},
        }
        dv_rem = dict(run_detail["delta_v_remaining_m_s_by_object"])
        for oid, dv_budget in dv_budget_m_s_by_object.items():
            dv_used = safe_float(dict(run_detail["total_dv_m_s_by_object"]).get(oid), default=0.0)
            dv_rem[oid] = float(max(float(dv_budget) - max(float(dv_used), 0.0), 0.0))
        run_detail["delta_v_remaining_m_s_by_object"] = dv_rem
        for reason in run_detail["fail_reasons"]:
            failure_mode_counts[str(reason)] = int(failure_mode_counts.get(str(reason), 0) + 1)
        run_details.append(run_detail)
        entry = {
            "iteration": i,
            "sampled_parameters": dict(p["sampled_parameters"]),
            "bundle_options": dict(p.get("bundle_options", {}) or {}),
            "comparison": p.get("comparison"),
            "summary": ro_summary,
            "metrics": study_metrics,
            "closest_approach_km": closest_approach_km,
            "assessment": assessment,
        }
        runs.append(entry)
        if bool(cfg.outputs.monte_carlo.get("save_iteration_summaries", False)):
            write_json(str(outdir / f"master_monte_carlo_run_{i:04d}.json"), entry)

    from sim.reporting.monte_carlo import (
        apply_monte_carlo_baseline_comparison,
        build_monte_carlo_report_payload,
        write_monte_carlo_report_artifacts,
    )

    report_context = build_monte_carlo_report_payload(
        cfg=cfg,
        config_path=config_path,
        root=root,
        repo_root=repo_root,
        runs=runs,
        run_details=run_details,
        closest_approach_km_runs=closest_approach_km_runs,
        duration_runs_s=duration_runs_s,
        total_dv_runs_m_s=total_dv_runs_m_s,
        guardrail_event_runs=guardrail_event_runs,
        failure_mode_counts=failure_mode_counts,
        dv_budget_m_s_by_object=dv_budget_m_s_by_object,
        gates=gates,
        mc_out_cfg=mc_out_cfg,
        varies_metadata_seed=varies_metadata_seed,
        parallel_active=parallel_active,
        parallel_enabled=parallel_enabled,
        total_iters=total_iters,
        parallel_workers=parallel_workers,
        parallel_fallback_reason=parallel_fallback_reason,
    )
    agg = report_context["agg"]
    agg["parallel_backend"] = parallel_backend
    agg["hierarchical_execution"] = dict(campaign_result.get("hierarchical_execution", {}) or {})
    if checkpoint_meta:
        agg["checkpoint"] = checkpoint_meta
    commander_brief = report_context["commander_brief"]
    study_brief = report_context["study_brief"]
    analyst_pack = report_context["analyst_pack"]
    durations_s = report_context["durations_s"]
    ca_finite = report_context["ca_finite"]
    all_obj_ids = report_context["all_obj_ids"]
    dv_by_object = report_context["dv_by_object"]
    dv_remaining_m_s_by_object = report_context["dv_remaining_m_s_by_object"]
    run_details = report_context["run_details"]
    keepout_threshold = report_context["keepout_threshold"]
    failure_mode_counts = report_context["failure_mode_counts"]

    from sim.reporting.monte_carlo_plots import write_monte_carlo_plot_artifacts

    agg = apply_monte_carlo_baseline_comparison(
        agg=agg,
        commander_brief=commander_brief,
        study_brief=study_brief,
        config_path=config_path,
        baseline_summary_json=str(mc_out_cfg.get("baseline_summary_json", "")).strip(),
    )
    relative_range_artifacts = relative_range_writer.finalize()
    if relative_range_artifacts:
        agg.setdefault("artifacts", {}).update(relative_range_artifacts)
    agg = write_monte_carlo_plot_artifacts(
        cfg=cfg,
        outdir=outdir,
        agg=agg,
        runs=runs,
        run_details=run_details,
        relative_range_series_runs=[],
        durations_s=durations_s,
        ca_finite=ca_finite,
        all_obj_ids=all_obj_ids,
        dv_by_object=dv_by_object,
        dv_remaining_m_s_by_object=dv_remaining_m_s_by_object,
        dv_budget_m_s_by_object=dv_budget_m_s_by_object,
        failure_mode_counts=failure_mode_counts,
        keepout_threshold=keepout_threshold,
        gates=gates,
        mc_out_cfg=mc_out_cfg,
    )
    agg = write_monte_carlo_report_artifacts(
        cfg=cfg,
        outdir=outdir,
        agg=agg,
        commander_brief=commander_brief,
        study_brief=study_brief,
        analyst_pack=analyst_pack,
        run_details=run_details,
        mc_out_cfg=mc_out_cfg,
    )
    if retention_policy:
        agg.setdefault("artifacts", {}).update(
            write_campaign_dataset(outdir, policy=retention_policy, records=dataset_records)
        )

    from sim.reporting.ai_reports import write_ai_report_artifacts

    agg = write_ai_report_artifacts(
        cfg=cfg,
        config_path=config_path,
        outdir=outdir,
        payload=agg,
        payload_kind="monte_carlo",
    )
    from sim.review import write_workflow_review

    artifacts = dict(agg.get("artifacts", {}) or {})
    review_artifacts = dict(artifacts)
    range_plot_status = review_artifacts.pop("relative_range_timeseries_status", None)
    review_summary = {
        "iterations": len(runs),
        "pass_rate": dict(agg.get("aggregate_stats", {}) or {}).get("pass_rate"),
        "closest_approach_km_mean": dict(agg.get("aggregate_stats", {}) or {}).get(
            "closest_approach_km_mean"
        ),
        "total_dv_m_s_mean": dict(agg.get("aggregate_stats", {}) or {}).get("total_dv_m_s_mean"),
    }
    if range_plot_status is not None:
        review_summary["relative_range_timeseries_status"] = range_plot_status
    review_outputs = write_workflow_review(
        output_dir=outdir,
        workflow_type="monte_carlo",
        title=str(agg.get("scenario_name", cfg.scenario_name) or "monte_carlo"),
        scenario_name=str(agg.get("scenario_name", cfg.scenario_name) or "monte_carlo"),
        status="complete",
        summary=review_summary,
        artifacts=review_artifacts,
        recommended_queries=[
            {
                "name": "workflow_metadata",
                "description": "Monte Carlo workflow metadata.",
                "sql": "SELECT * FROM workflow_metadata",
            },
            {
                "name": "campaign_runs",
                "description": "Run-level campaign status and key metrics.",
                "sql": "SELECT iteration, passed, closest_approach_km, total_dv_m_s, output_dir FROM campaign_runs",
            },
            {
                "name": "campaign_metrics",
                "description": "Flattened run-level metrics.",
                "sql": "SELECT iteration, metric_name, metric_value FROM campaign_metrics",
            },
        ],
        recommended_review_order=[
            "Open the commander brief or Monte Carlo summary JSON first.",
            "Query campaign_runs for run-level pass/fail and key metrics.",
            "Query campaign_metrics when a derived metric needs per-run inspection.",
        ],
        source_config=str(config_path),
        provenance={"writer": "sim.execution.campaigns.run_monte_carlo_simulation"},
        tables={
            "campaign_runs": _monte_carlo_review_run_rows(runs),
            "campaign_metrics": _monte_carlo_review_metric_rows(runs),
        },
    )
    artifacts.update(review_outputs)
    agg["artifacts"] = artifacts
    from sim.reporting.output_index import write_output_index

    index_path = write_output_index(
        outdir=outdir,
        workflow="monte_carlo",
        title=str(agg.get("scenario_name", cfg.scenario_name) or "monte_carlo"),
        payload=agg,
        artifacts=artifacts,
    )
    artifacts["output_index_md"] = str(index_path)
    agg["artifacts"] = artifacts
    summary_json = str(dict(agg.get("artifacts", {}) or {}).get("summary_json", "") or "")
    if summary_json:
        write_json(summary_json, agg)
    return agg
