from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from sim.acceleration.settings import acceleration_context_from_config
from sim.config import SimulationScenarioConfig, load_simulation_yaml, validate_scenario_plugins
from sim.config.scenario.loader import load_simulation_yaml_bytes
from sim.pro_features import FEATURE_CAMPAIGNS, require_pro_feature
from sim.security import ConfigPathPolicy
from sim.security.sealed_mode import sealed_mode_enabled, validate_sealed_mode
from sim.single_run import _coerce_noninteractive_for_automation, _run_single_config, _SingleRunEngine

from . import campaigns, covariance, sensitivity
from .study import analysis_study_type

StepCallback = Callable[[int, int], None]
BatchCallback = Callable[[int, int], None]
BatchProgressCallback = Callable[[dict[str, Any]], None]


class SimulationExecutionService:
    """Canonical entrypoint for executing simulation configs.

    Single-run and batch-analysis workflows enter through this facade so callers
    do not need to know which execution module owns a specific study type.
    """

    @staticmethod
    def _validate_execution_boundary(cfg: SimulationScenarioConfig) -> None:
        """Enforce process-wide sealed policy before any execution-time imports."""

        if not sealed_mode_enabled():
            return
        errors = validate_sealed_mode(cfg)
        if errors:
            raise ValueError("Sealed mode validation failed:\n- " + "\n- ".join(errors))

    @staticmethod
    def _effective_path_policy(
        *,
        source_path: str | Path | None,
        path_policy: ConfigPathPolicy | None,
    ) -> ConfigPathPolicy | None:
        """Retain the loader's default trust boundary through batch execution."""

        if path_policy is not None or source_path is None:
            return path_policy
        return ConfigPathPolicy.default(config_path=source_path)

    def load_config(
        self,
        config_path: str | Path,
        *,
        path_policy: ConfigPathPolicy | None = None,
    ) -> SimulationScenarioConfig:
        return _coerce_noninteractive_for_automation(
            load_simulation_yaml(config_path, path_policy=path_policy)
        )

    def load_config_bytes(
        self,
        content: bytes,
        *,
        source_path: str | Path,
        path_policy: ConfigPathPolicy | None = None,
    ) -> SimulationScenarioConfig:
        return _coerce_noninteractive_for_automation(
            load_simulation_yaml_bytes(content, source_path=source_path, path_policy=path_policy)
        )

    def validate_config(self, cfg: SimulationScenarioConfig) -> list[str]:
        self._validate_execution_boundary(cfg)
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        if not strict_plugins:
            return []
        return list(validate_scenario_plugins(cfg))

    def study_type(self, cfg: SimulationScenarioConfig) -> str:
        return analysis_study_type(cfg)

    def is_batch_analysis(self, cfg: SimulationScenarioConfig) -> bool:
        return self.study_type(cfg) in {"monte_carlo", "sensitivity"}

    def is_analysis_workflow(self, cfg: SimulationScenarioConfig) -> bool:
        return self.study_type(cfg) in {"monte_carlo", "sensitivity", "covariance"}

    def create_engine(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: StepCallback | None = None,
        history_mode: str = "full",
        initial_history_capacity: int = 4096,
        max_history_samples: int = 4096,
    ) -> _SingleRunEngine:
        self._validate_execution_boundary(cfg)
        with acceleration_context_from_config(cfg):
            return _SingleRunEngine(
                cfg,
                step_callback=step_callback,
                history_mode=history_mode,
                initial_history_capacity=initial_history_capacity,
                max_history_samples=max_history_samples,
            )

    def run_single(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        self._validate_execution_boundary(cfg)
        if cfg.outputs.campaign_retention:
            raise ValueError(
                "outputs.campaign_retention applies to Monte Carlo or sensitivity studies; "
                "for a config queue, set it in the campaign YAML outputs"
            )
        return _run_single_config(cfg, step_callback=step_callback)

    def run_session_payload(
        self,
        cfg: SimulationScenarioConfig,
        *,
        source_path: str | Path | None = None,
        path_policy: ConfigPathPolicy | None = None,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        self._validate_execution_boundary(cfg)
        path_policy = self._effective_path_policy(source_path=source_path, path_policy=path_policy)
        if self.study_type(cfg) == "covariance":
            return covariance.run_covariance_analysis(
                config_path=source_path,
                cfg=cfg,
                step_callback=step_callback,
            )
        if self.is_batch_analysis(cfg):
            return self.run_batch_from_config(
                cfg,
                source_path=source_path,
                path_policy=path_policy,
                step_callback=step_callback,
            )
        return self.run_single(cfg, step_callback=step_callback)

    def run_config_file(
        self,
        config_path: str | Path,
        *,
        path_policy: ConfigPathPolicy | None = None,
        step_callback: StepCallback | None = None,
        batch_callback: BatchCallback | None = None,
        batch_progress_callback: BatchProgressCallback | None = None,
    ) -> dict[str, Any]:
        path = Path(config_path).expanduser().resolve()
        path_policy = self._effective_path_policy(source_path=path, path_policy=path_policy)
        cfg = self.load_config(path, path_policy=path_policy)
        return self.run_config(
            cfg,
            source_path=path,
            path_policy=path_policy,
            step_callback=step_callback,
            batch_callback=batch_callback,
            batch_progress_callback=batch_progress_callback,
        )

    def run_config(
        self,
        cfg: SimulationScenarioConfig,
        *,
        source_path: str | Path,
        path_policy: ConfigPathPolicy | None = None,
        step_callback: StepCallback | None = None,
        batch_callback: BatchCallback | None = None,
        batch_progress_callback: BatchProgressCallback | None = None,
    ) -> dict[str, Any]:
        path = Path(source_path).expanduser().resolve()
        path_policy = self._effective_path_policy(source_path=path, path_policy=path_policy)
        errors = self.validate_config(cfg)
        if errors:
            msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
            raise ValueError(msg)

        if self.is_batch_analysis(cfg):
            require_pro_feature(FEATURE_CAMPAIGNS)
            if self.study_type(cfg) == "sensitivity":
                return sensitivity.run_sensitivity_analysis(
                    config_path=path,
                    cfg=cfg,
                    path_policy=path_policy,
                    step_callback=step_callback,
                    batch_callback=batch_callback,
                    batch_progress_callback=batch_progress_callback,
                )
            if self.study_type(cfg) == "monte_carlo" and campaigns.can_run_monte_carlo_campaign(cfg):
                return campaigns.run_monte_carlo_campaign(
                    config_path=path,
                    cfg=cfg,
                    path_policy=path_policy,
                    step_callback=step_callback,
                    batch_callback=batch_callback,
                    batch_progress_callback=batch_progress_callback,
                )
            raise ValueError(f"Unsupported batch analysis study_type: {self.study_type(cfg)!r}")
        if self.study_type(cfg) == "covariance":
            return covariance.run_covariance_analysis(
                config_path=path,
                cfg=cfg,
                step_callback=step_callback,
            )

        payload = self.run_single(cfg, step_callback=step_callback)
        return self.wrap_single_file_payload(payload=payload, cfg=cfg, config_path=path)

    def run_batch_from_config(
        self,
        cfg: SimulationScenarioConfig,
        *,
        source_path: str | Path | None = None,
        path_policy: ConfigPathPolicy | None = None,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        self._validate_execution_boundary(cfg)
        path_policy = self._effective_path_policy(source_path=source_path, path_policy=path_policy)
        temp_dir = str(Path(source_path).expanduser().resolve().parent) if source_path is not None else str(Path.cwd())
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".yaml",
                mode="w",
                delete=False,
                encoding="utf-8",
                dir=temp_dir,
            ) as tmp:
                tmp_path = Path(tmp.name)
                yaml.safe_dump(cfg.to_dict(), tmp, sort_keys=False)
            if self.study_type(cfg) == "sensitivity":
                require_pro_feature(FEATURE_CAMPAIGNS)
                payload = sensitivity.run_sensitivity_analysis(
                    config_path=tmp_path,
                    cfg=cfg,
                    path_policy=path_policy,
                    step_callback=step_callback,
                )
            elif self.study_type(cfg) == "monte_carlo" and campaigns.can_run_monte_carlo_campaign(cfg):
                require_pro_feature(FEATURE_CAMPAIGNS)
                payload = campaigns.run_monte_carlo_campaign(
                    config_path=tmp_path,
                    cfg=cfg,
                    path_policy=path_policy,
                    step_callback=step_callback,
                )
            else:
                raise ValueError(f"Unsupported batch analysis study_type: {self.study_type(cfg)!r}")
            if source_path is not None and isinstance(payload, dict):
                payload["config_path"] = str(Path(source_path).expanduser().resolve())
            return payload
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def wrap_single_file_payload(
        self,
        *,
        payload: dict[str, Any],
        cfg: SimulationScenarioConfig,
        config_path: str | Path,
    ) -> dict[str, Any]:
        result = dict(payload)
        result.update({
            "config_path": str(Path(config_path).expanduser().resolve()),
            "scenario_name": cfg.scenario_name,
            "scenario_description": cfg.scenario_description,
            "monte_carlo": {"enabled": False},
            "run": dict(payload.get("summary", {}) or {}),
        })
        return result


_DEFAULT_SERVICE = SimulationExecutionService()


def create_single_run_engine(
    cfg: SimulationScenarioConfig,
    *,
    step_callback: StepCallback | None = None,
    history_mode: str = "full",
    initial_history_capacity: int = 4096,
    max_history_samples: int = 4096,
) -> _SingleRunEngine:
    return _DEFAULT_SERVICE.create_engine(
        cfg,
        step_callback=step_callback,
        history_mode=history_mode,
        initial_history_capacity=initial_history_capacity,
        max_history_samples=max_history_samples,
    )


def run_simulation_scenario(
    cfg: SimulationScenarioConfig,
    *,
    source_path: str | Path | None = None,
    path_policy: ConfigPathPolicy | None = None,
    step_callback: StepCallback | None = None,
) -> dict[str, Any]:
    return _DEFAULT_SERVICE.run_session_payload(
        cfg,
        source_path=source_path,
        path_policy=path_policy,
        step_callback=step_callback,
    )


def run_simulation_config_file(
    config_path: str | Path,
    *,
    path_policy: ConfigPathPolicy | None = None,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
) -> dict[str, Any]:
    return _DEFAULT_SERVICE.run_config_file(
        config_path,
        path_policy=path_policy,
        step_callback=step_callback,
        batch_callback=batch_callback,
        batch_progress_callback=batch_progress_callback,
    )
