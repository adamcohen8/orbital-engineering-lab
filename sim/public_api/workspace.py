from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

from sim.config import (
    SimulationScenarioConfig,
    enabled_object_ids,
    scenario_config_from_dict,
    validate_scenario_plugins,
)
from sim.execution.study import analysis_study_type
from sim.execution.validation import validate_generated_batch_configs
from sim.public_api.config import (
    MetricCallback,
    SimulationConfig,
    _api_sealed_policy,
)
from sim.public_api.feature_routing import _require_private_workflow
from sim.public_api.results import (
    MetricStudyResult,
    SimulationResult,
    _aggregate_custom_metrics,
)
from sim.public_api.session import HostedSimulationSession, SimulationSession
from sim.scenarios import ScenarioArtifact, ValidationReport
from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue
from sim.security import ConfigPathPolicy
from sim.security.sealed_mode import SealedModePolicy, _validate_ai_section, validate_sealed_mode


class SimulationWorkspace:
    """Higher-level programmatic facade for CLI-equivalent workflows."""

    def __init__(
        self,
        *,
        allow_external_config_paths: bool = False,
        allow_external_ai_prompt_files: bool = False,
        sealed_mode: bool = False,
        sealed_policy: SealedModePolicy | None = None,
        read_roots: Iterable[str | Path] = (),
        write_roots: Iterable[str | Path] = (),
        workspace_root: str | Path | None = None,
        allow_config_dir_writes: bool = True,
    ) -> None:
        self.allow_external_config_paths = bool(allow_external_config_paths)
        self.allow_external_ai_prompt_files = bool(allow_external_ai_prompt_files)
        self.read_roots = tuple(read_roots)
        self.write_roots = tuple(write_roots)
        self.workspace_root = workspace_root
        self.allow_config_dir_writes = bool(allow_config_dir_writes)
        self._enforce_workspace_paths = bool(
            workspace_root is not None or self.read_roots or self.write_roots
        )
        self._sealed_policy = _api_sealed_policy(sealed_mode=sealed_mode, sealed_policy=sealed_policy)

    def _path_policy_for(self, path: str | Path | None = None) -> ConfigPathPolicy:
        return ConfigPathPolicy.default(
            config_path=path,
            workspace_root=self.workspace_root,
            read_roots=self.read_roots,
            write_roots=self.write_roots,
            allow_external_config_paths=self.allow_external_config_paths,
            allow_external_ai_prompt_files=self.allow_external_ai_prompt_files,
            allow_config_dir_writes=self.allow_config_dir_writes,
        )

    def load(self, path: str | Path) -> SimulationConfig:
        config = SimulationConfig.from_yaml(path, path_policy=self._path_policy_for(path))
        self._enforce_sealed_mode(config)
        return config

    def from_dict(self, data: dict[str, Any]) -> SimulationConfig:
        config = SimulationConfig.from_dict(
            data,
            path_policy=self._path_policy_for() if self._enforce_workspace_paths else None,
        )
        self._enforce_sealed_mode(config)
        return config

    def session(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> SimulationSession:
        return SimulationSession.from_config(self._coerce_config(config), sealed_policy=self._sealed_policy)

    def artifact(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> ScenarioArtifact:
        if isinstance(config, ScenarioArtifact):
            return config
        if isinstance(config, (str, Path)):
            return ScenarioArtifact(self.load(config))
        return ScenarioArtifact.from_config(config)

    def save_config(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        path: str | Path,
        *,
        validate: bool = True,
    ) -> Path:
        artifact = self.artifact(config)
        if validate:
            report = self.validate(artifact)
            if not bool(report.get("ok", False)):
                errors = [str(item) for item in list(report.get("errors", []) or [])]
                detail = "\n- " + "\n- ".join(errors) if errors else ""
                raise ValueError(f"Cannot save invalid scenario artifact.{detail}")
        target = Path(path).expanduser()
        if self._enforce_workspace_paths:
            target = self._path_policy_for(self._config_path_text(config)).resolve_output_file(
                target,
                purpose="scenario artifact output",
            )
        return artifact.write(target)

    def run(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        step_callback: Any | None = None,
    ) -> SimulationResult:
        return self.session(config).run(step_callback=step_callback)

    def run_payload(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        step_callback: Any | None = None,
    ) -> dict[str, Any]:
        return self.run(config, step_callback=step_callback).payload

    def inspect_od_input(self, value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        """Safely inspect a versioned OD JSON input without authorizing execution."""

        if isinstance(value, (str, Path)):
            value = self._path_policy_for(value).resolve_input_file(value, purpose="OD input inspection")
        inspect_input = _require_private_workflow(
            "sim.estimation.productization",
            "safely_inspect_od_input",
            "OD input inspection",
        )
        return inspect_input(value)

    def run_sequential_od(
        self,
        packet: str | Path | Mapping[str, Any],
        *,
        initial_state_eci_km_s: Any,
        initial_covariance: Any,
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the shared EKF/UKF/RTS OD workflow through the workspace facade."""

        packet_value: Mapping[str, Any]
        if isinstance(packet, (str, Path)):
            packet_path = self._path_policy_for(packet).resolve_input_file(packet, purpose="sequential OD packet")
            packet_value = json.loads(packet_path.read_text(encoding="utf-8"))
        else:
            packet_value = packet
        resolved_output = output_dir
        if output_dir is not None and self._enforce_workspace_paths:
            resolved_output = self._path_policy_for().resolve_output_dir(output_dir, purpose="sequential OD output")
        runner = _require_private_workflow(
            "sim.estimation.sequential_od",
            "run_sequential_orbit_od",
            "Sequential OD",
        )
        return runner(
            packet_value,
            initial_state_eci_km_s=initial_state_eci_km_s,
            initial_covariance=initial_covariance,
            output_dir=resolved_output,
            **kwargs,
        )

    def run_integrated_relative_od(self, **kwargs: Any) -> dict[str, Any]:
        """Run nonlinear-baseline relative batch/sequential OD through the workspace facade."""

        values = dict(kwargs)
        output_dir = values.get("output_dir")
        if output_dir is not None and self._enforce_workspace_paths:
            values["output_dir"] = self._path_policy_for().resolve_output_dir(
                output_dir,
                purpose="integrated relative OD output",
            )
        runner = _require_private_workflow(
            "sim.estimation.relative_od_integrated",
            "run_integrated_relative_od",
            "Integrated relative OD",
        )
        return runner(**values)

    def package_od_evidence(
        self,
        *,
        task_id: str,
        capability_id: str,
        report: str | Path | Mapping[str, Any],
        claim_level: str,
        sources: Iterable[str | Path],
        reproduction_commands: Iterable[str],
        estimator: Mapping[str, Any],
        propagator: Mapping[str, Any],
        frame_policy: Mapping[str, Any],
        output_dir: str | Path,
        source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        handling_classification: str = "private",
    ) -> dict[str, Any]:
        """Package an OD report with provenance, freshness, and reproduction metadata."""

        policy = self._path_policy_for(report if isinstance(report, (str, Path)) else None)
        if isinstance(report, (str, Path)):
            report_path = policy.resolve_input_file(report, purpose="OD source report")
            report_value = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report_path = None
            report_value = dict(report)
        source_paths = [policy.resolve_input_file(path, purpose="OD dataset source") for path in sources]
        target = policy.resolve_output_dir(output_dir, purpose="OD evidence output")
        target.mkdir(parents=True, exist_ok=True)
        build_manifest = _require_private_workflow(
            "sim.estimation.productization",
            "build_od_dataset_manifest",
            "OD evidence packaging",
        )
        normalize_solution = _require_private_workflow(
            "sim.estimation.productization",
            "normalize_od_solution",
            "OD evidence packaging",
        )
        build_packet = _require_private_workflow(
            "sim.estimation.productization",
            "build_od_agent_evidence_packet",
            "OD evidence packaging",
        )
        manifest = build_manifest(
            dataset_id=f"od:{task_id}",
            sources=source_paths,
            source_metadata=source_metadata,
            handling_classification=handling_classification,
            output_path=target / "od_dataset_manifest.json",
        )
        fingerprints = {row["path"]: row["sha256"] for row in manifest["sources"]}
        solution = normalize_solution(
            capability_id=capability_id,
            report=report_value,
            claim_level=claim_level,
            input_fingerprints=fingerprints,
            estimator=estimator,
            propagator=propagator,
            frame_policy=frame_policy,
            output_path=target / "od_solution.json",
        )
        packet = build_packet(
            task_id=task_id,
            solution=solution,
            dataset_manifest=manifest,
            reproduction_commands=list(reproduction_commands),
            artifacts=(() if report_path is None else (report_path,)),
            output_path=target / "agent_evidence_packet.json",
        )
        return packet

    def evaluate_metrics(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...],
        *,
        step_callback: Any | None = None,
    ) -> dict[str, Any]:
        return self.run(config, step_callback=step_callback).evaluate_metrics(metrics)

    def run_monte_carlo_metrics(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...],
        *,
        step_callback: Any | None = None,
        batch_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        sim_config = self._coerce_config(config)
        cfg = sim_config.to_scenario_config()
        if analysis_study_type(cfg) != "monte_carlo":
            raise ValueError("run_monte_carlo_metrics() requires a Monte Carlo scenario.")
        from sim.execution.campaigns import prepare_monte_carlo_runs

        root = cfg.to_dict()
        prepared = prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=Path(cfg.outputs.output_dir))
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        runs: list[dict[str, Any]] = []
        total = len(prepared)
        for done, item in enumerate(prepared, start=1):
            iteration = int(item["iteration"])
            config_dict = dict(item["config_dict"])
            config_dict.setdefault("analysis", {})["enabled"] = False
            # Keep the same source provenance and workspace path policy on
            # every generated child.  Batch preparation may change an
            # allowlisted input path; reconstructing without the parent's
            # policy would let that child bypass the workspace read/write
            # roots before the sealed/plugin checks below.
            run_cfg = scenario_config_from_dict(
                config_dict,
                source_path=sim_config.source_path,
                path_policy=sim_config.path_policy,
            )
            run_config = SimulationConfig(
                run_cfg,
                source_path=sim_config.source_path,
                path_policy=sim_config.path_policy,
            )
            # Re-apply the workspace policy to every generated child before any
            # plugin-importing validation or engine construction.  Batch
            # variation paths can change an existing plugin pointer after the
            # base config has passed sealed-mode validation.
            self._enforce_sealed_mode(run_config)
            if strict_plugins:
                errors = validate_scenario_plugins(
                    run_cfg,
                    import_plugins=(
                        self._sealed_policy is None
                        or self._sealed_policy.allow_untrusted_plugin_imports
                    ),
                )
                if errors:
                    msg = f"Plugin validation failed in Monte Carlo iteration {iteration}:\n- " + "\n- ".join(errors)
                    raise ValueError(msg)
            run_result = SimulationSession.from_config(
                run_config,
                sealed_policy=self._sealed_policy,
            ).run(step_callback=step_callback)
            custom_metrics = run_result.evaluate_metrics(metrics)
            runs.append(
                {
                    "iteration": iteration,
                    "seed": int(item.get("seed", run_cfg.metadata.get("seed", 0))),
                    "sampled_parameters": dict(item.get("sampled_parameters", {}) or {}),
                    "summary": run_result.summary,
                    "metrics": custom_metrics,
                }
            )
            if batch_callback is not None:
                batch_callback(done, total)
        run_metrics = [dict(row.get("metrics", {}) or {}) for row in runs]
        return MetricStudyResult({
            "scenario_name": cfg.scenario_name,
            "monte_carlo": {"enabled": True, "iterations": int(len(runs))},
            "runs": runs,
            "custom_metrics": _aggregate_custom_metrics(run_metrics),
        })

    def sweep(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        parameter: str,
        values: Iterable[Any],
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...] | None = None,
        output_dir_template: str | None = None,
        step_callback: Any | None = None,
        batch_callback: Callable[[int, int], None] | None = None,
    ) -> MetricStudyResult:
        _require_private_workflow("sim.execution.sensitivity", "prepare_sensitivity_runs", "Parameter sweeps")
        base = self._coerce_config(config)
        sweep_values = list(values)
        runs: list[dict[str, Any]] = []
        total = len(sweep_values)
        for idx, value in enumerate(sweep_values):
            cfg_i = base.with_value(parameter, value)
            if output_dir_template:
                cfg_i = cfg_i.with_output_dir(
                    str(output_dir_template).format(index=idx, value=value, scenario=cfg_i.scenario_name)
                )
            self._enforce_sealed_mode(cfg_i)
            result = SimulationSession.from_config(
                cfg_i,
                sealed_policy=self._sealed_policy,
            ).run(step_callback=step_callback)
            custom_metrics = result.evaluate_metrics(metrics) if metrics is not None else {}
            runs.append(
                {
                    "iteration": idx,
                    "seed": result.config.scenario.metadata.get("seed"),
                    "sampled_parameters": {str(parameter): value},
                    "summary": result.summary,
                    "metrics": custom_metrics,
                }
            )
            if batch_callback is not None:
                batch_callback(idx + 1, total)
        run_metrics = [dict(row.get("metrics", {}) or {}) for row in runs]
        return MetricStudyResult({
            "scenario_name": base.scenario_name,
            "sweep": {"parameter": str(parameter), "values": sweep_values, "run_count": int(len(runs))},
            "runs": runs,
            "custom_metrics": _aggregate_custom_metrics(run_metrics),
        })

    def validate(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        import_plugins: bool = True,
    ) -> dict[str, Any]:
        try:
            sim_config = self._coerce_config(config)
        except Exception as exc:
            return {
                "ok": False,
                "status": "failed",
                "errors": [str(exc)],
                "config_path": self._config_path_text(config),
            }

        cfg = sim_config.to_scenario_config()
        study_type = analysis_study_type(cfg)
        plugin_errors = list(validate_scenario_plugins(cfg, import_plugins=import_plugins))
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        generated: dict[str, Any] = {"run_count": 0, "errors": []}
        if (not plugin_errors or not strict_plugins) and study_type in {"monte_carlo", "sensitivity"}:
            generated = validate_generated_batch_configs(
                cfg,
                import_plugins=import_plugins,
                source_path=sim_config.source_path,
                path_policy=sim_config.path_policy,
            )

        generated_errors = list(generated.get("errors", []) or [])
        errors: list[Any] = []
        if strict_plugins:
            errors.extend(plugin_errors)
        errors.extend(generated_errors)
        return {
            "ok": not errors,
            "status": "ok" if not errors else "failed",
            "config_path": str(sim_config.source_path) if sim_config.source_path is not None else None,
            "scenario_name": cfg.scenario_name,
            "scenario_description": cfg.scenario_description,
            "study_type": study_type,
            "objects": enabled_object_ids(cfg),
            "duration_s": float(cfg.simulator.duration_s),
            "dt_s": float(cfg.simulator.dt_s),
            "output_dir": str(cfg.outputs.output_dir),
            "plugins": {
                "strict": strict_plugins,
                "status": "ok" if not plugin_errors else ("failed" if strict_plugins else "warn"),
                "errors": plugin_errors,
            },
            "generated": generated,
            "errors": errors,
        }

    def validate_safe(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> dict[str, Any]:
        """Validate structure and plugin pointers without importing plugin modules."""

        return self.validate(config, import_plugins=False)

    def validate_candidate_config(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        trust_plugins: bool = False,
    ) -> dict[str, Any]:
        """Validate an agent-authored candidate without treating inspection as execution permission.

        Safe validation always runs first. Plugin-importing validation runs only when the
        caller explicitly marks the candidate's referenced code and paths as trusted.
        """

        safe = self.validate_safe(config)
        trusted: dict[str, Any] = {
            "ok": None,
            "status": "not_run",
            "reason": "safe_validation_failed" if not bool(safe.get("ok", False)) else "trust_not_granted",
        }
        if bool(safe.get("ok", False)) and trust_plugins:
            try:
                trusted = self.validate(config, import_plugins=True)
            except Exception as exc:
                trusted = {"ok": False, "status": "failed", "errors": [str(exc)]}

        if not bool(safe.get("ok", False)):
            status = "failed"
            ok = False
        elif not trust_plugins:
            status = "safe_only"
            ok = True
        else:
            ok = bool(trusted.get("ok", False))
            status = "ok" if ok else "failed"

        return {
            "workflow": "agent_authored_candidate_validation",
            "ok": ok,
            "status": status,
            "execution_authorized": False,
            "trust_plugins": bool(trust_plugins),
            "safe_validation": safe,
            "trusted_validation": trusted,
            "next_step": (
                "Review referenced plugins, modules, and external paths before trusted validation or execution."
                if status == "safe_only"
                else "Candidate is validated but execution remains a separate caller decision."
                if status == "ok"
                else "Correct the reported validation errors before execution."
            ),
        }

    def validate_report(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> ValidationReport:
        return ValidationReport.from_validation_dict(self.validate(config))

    def validate_controller_bench(
        self,
        config_path: str | Path,
        *,
        compare_names: list[str] | None = None,
    ) -> dict[str, Any]:
        from sim.controller_lab import validate_controller_bench_config

        report = validate_controller_bench_config(config_path, compare_names=compare_names)
        errors = list(report.get("errors", []) or [])
        report["ok"] = not errors
        report["status"] = "ok" if not errors else "failed"
        return report

    def run_controller_bench(
        self,
        config_path: str | Path,
        *,
        compare_names: list[str] | None = None,
    ) -> dict[str, Any]:
        from sim.controller_lab import run_controller_bench

        return run_controller_bench(config_path, compare_names=compare_names)

    def estimate_ai_report_cost(
        self,
        config_path: str | Path,
        *,
        output_dir: str | Path = "",
        controller_bench: bool = False,
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if controller_bench:
            estimate = _require_private_workflow(
                "run_simulation",
                "_estimate_ai_report_from_controller_bench",
                "AI report workflows",
            )
            return estimate(str(config_path), output_dir=str(output_dir or ""), ai_options=dict(ai_options or {}))
        estimate = _require_private_workflow("run_simulation", "_estimate_ai_report_from_outputs", "AI report workflows")
        return estimate(str(config_path), output_dir=str(output_dir or ""), ai_options=dict(ai_options or {}))

    def create_ai_report(
        self,
        config_path: str | Path,
        *,
        output_dir: str | Path = "",
        controller_bench: bool = False,
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = dict(ai_options or {})
        if self._sealed_policy is not None:
            errors = _validate_ai_section(
                options,
                "ai_options",
                self._sealed_policy,
                enabled_default=True,
            )
            if errors:
                raise ValueError("Sealed mode policy violations:\n- " + "\n- ".join(errors))
        allow_custom_endpoint = bool(options.get("allow_custom_endpoint", False))
        if controller_bench:
            create = _require_private_workflow(
                "run_simulation",
                "_create_ai_report_from_controller_bench",
                "AI report workflows",
            )
            return create(
                str(config_path),
                output_dir=str(output_dir or ""),
                ai_options=options,
                allow_custom_endpoint=allow_custom_endpoint,
            )
        create = _require_private_workflow("run_simulation", "_create_ai_report_from_outputs", "AI report workflows")
        return create(
            str(config_path),
            output_dir=str(output_dir or ""),
            ai_options=options,
            allow_custom_endpoint=allow_custom_endpoint,
        )

    def prepare_report_packet(
        self,
        config_path: str | Path,
        *,
        output_dir: str | Path = "",
        controller_bench: bool = False,
        report_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare deterministic evidence and instructions for an external coding agent."""

        function_name = (
            "_prepare_report_packet_from_controller_bench"
            if controller_bench
            else "_prepare_report_packet_from_outputs"
        )
        prepare = _require_private_workflow("run_simulation", function_name, "Agent report workflows")
        return prepare(
            str(config_path),
            output_dir=str(output_dir or ""),
            report_options=dict(report_options or {}),
        )

    def audit_report(
        self,
        report_path: str | Path,
        packet_path: str | Path,
        *,
        output_dir: str | Path = "",
        author: str = "coding_agent",
        model: str = "",
        fail_on_quality: bool = False,
    ) -> dict[str, Any]:
        """Render and audit an agent-authored report against an OEL evidence packet."""

        audit = _require_private_workflow("run_simulation", "_audit_agent_report", "Agent report workflows")
        return audit(
            str(report_path),
            str(packet_path),
            output_dir=str(output_dir or ""),
            author=str(author or "coding_agent"),
            model=str(model or ""),
            fail_on_quality=bool(fail_on_quality),
        )

    def estimate_ai_config_cost(
        self,
        config_path: str | Path,
        *,
        prompt: str = "",
        prompt_file: str | Path = "",
        output_dir: str | Path = "",
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        warnings.warn(
            "estimate_ai_config_cost() is a legacy provider adapter. Prefer authoring with a coding agent "
            "and validate_candidate_config().",
            FutureWarning,
            stacklevel=2,
        )
        estimate = _require_private_workflow("run_simulation", "_estimate_ai_config_cost", "AI config workflows")

        return estimate(
            str(config_path),
            prompt=str(prompt or ""),
            prompt_file=str(prompt_file or ""),
            output_dir=str(output_dir or ""),
            ai_options=dict(ai_options or {}),
        )

    def create_ai_config(
        self,
        config_path: str | Path,
        *,
        prompt: str = "",
        prompt_file: str | Path = "",
        output_dir: str | Path = "",
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        warnings.warn(
            "create_ai_config() is a legacy provider adapter. Prefer authoring with a coding agent "
            "and validate_candidate_config().",
            FutureWarning,
            stacklevel=2,
        )
        create = _require_private_workflow("run_simulation", "_create_ai_config_draft", "AI config workflows")
        options = dict(ai_options or {})
        if self._sealed_policy is not None:
            errors = _validate_ai_section(
                options,
                "ai_options",
                self._sealed_policy,
                enabled_default=True,
            )
            if errors:
                raise ValueError("Sealed mode policy violations:\n- " + "\n- ".join(errors))

        return create(
            str(config_path),
            prompt=str(prompt or ""),
            prompt_file=str(prompt_file or ""),
            output_dir=str(output_dir or ""),
            ai_options=options,
            allow_custom_endpoint=bool(options.get("allow_custom_endpoint", False)),
        )

    @staticmethod
    def _config_path_text(config: Any) -> str | None:
        if isinstance(config, (str, Path)):
            return str(Path(config).expanduser())
        if isinstance(config, ScenarioArtifact) and config.source_path is not None:
            return str(config.source_path)
        return None

    def _coerce_config(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> SimulationConfig:
        if isinstance(config, (str, Path)):
            coerced = SimulationConfig.from_yaml(config, path_policy=self._path_policy_for(config))
        elif not self._enforce_workspace_paths:
            coerced = SimulationSession._coerce_config(config)
        else:
            source_path = self._config_path_text(config)
            if isinstance(config, ScenarioArtifact):
                raw = config.to_dict()
            elif isinstance(config, SimulationConfig):
                raw = config.to_dict()
                source_path = str(config.source_path) if config.source_path is not None else source_path
            elif isinstance(config, SimulationScenarioConfig):
                raw = config.to_dict()
            elif isinstance(config, dict):
                raw = dict(config)
            else:
                raise TypeError(f"Unsupported config type: {type(config).__name__}")
            coerced = SimulationConfig.from_dict(
                raw,
                source_path=source_path,
                path_policy=self._path_policy_for(source_path),
            )
        self._enforce_sealed_mode(coerced)
        return coerced

    def _enforce_sealed_mode(self, config: SimulationConfig) -> None:
        if self._sealed_policy is None:
            return
        errors = validate_sealed_mode(config.to_scenario_config(), self._sealed_policy)
        if errors:
            raise ValueError("Sealed mode validation failed:\n- " + "\n- ".join(errors))


TrustedSimulationWorkspace = SimulationWorkspace


class HostedSimulationWorkspace(SimulationWorkspace):
    """Hosted facade with sealed mode and structural-first validation enforced."""

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.pop("sealed_mode", True) is not True:
            raise ValueError("HostedSimulationWorkspace cannot disable sealed mode.")
        if kwargs.pop("allow_config_dir_writes", False) is not False:
            raise ValueError("HostedSimulationWorkspace cannot allow config-directory writes outside workspace roots.")
        super().__init__(sealed_mode=True, allow_config_dir_writes=False, **kwargs)

    def session(self, config) -> HostedSimulationSession:
        return HostedSimulationSession.from_config(
            self._coerce_config(config),
            sealed_policy=self._sealed_policy,
        )

    def validate(self, config, *, import_plugins: bool = False) -> dict[str, Any]:
        if import_plugins:
            raise ValueError("HostedSimulationWorkspace validation cannot import plugin modules.")
        return super().validate(config, import_plugins=False)
