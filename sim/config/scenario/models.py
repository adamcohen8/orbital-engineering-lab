from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from sim.schema_versions import LEGACY_SCENARIO_SCHEMA_VERSION

__all__ = [
    '_plain_config_data',
    'AlgorithmPointer',
    'BridgePointer',
    'FlightSoftwareSection',
    'AgentSection',
    'GroundStationSection',
    '_TypedConfigDict',
    'SimulatorAccelerationSection',
    'SimulatorExecutionSection',
    'SimulatorFramesSection',
    'SimulatorSection',
    'SimulatorDynamicsSection',
    'SimulatorEnvironmentSection',
    'SimulatorPluginValidationSection',
    'SimulatorTerminationSection',
    'OutputStatsSection',
    'OutputPlotsSection',
    'OutputAnimationsSection',
    'OutputMonteCarloSection',
    'OutputAIReportSection',
    'OutputAIConfigSection',
    'OutputResourceLimitsSection',
    'OutputReviewSection',
    'OutputOrbitalAnalysisSection',
    'OutputsSection',
    'MonteCarloVariation',
    'MonteCarloBundleOption',
    'MonteCarloBundle',
    'MonteCarloComparison',
    'MonteCarloConstraint',
    'MonteCarloCorrelatedNormal',
    'MonteCarloSection',
    'AnalysisExecutionSection',
    'AnalysisBaselineSection',
    'AnalysisMonteCarloSection',
    'SensitivityParameter',
    'SensitivitySection',
    'CovarianceObjectSection',
    'CovarianceCollisionScreeningSection',
    'CovariancePairSection',
    'CovarianceFiniteDifferenceSection',
    'CovarianceProcessNoiseSection',
    'CovarianceSection',
    'MissionRecoverySection',
    'AnalysisSection',
    'SimulationScenarioConfig',
]


def _plain_config_data(value: Any) -> Any:
    if isinstance(value, AlgorithmPointer) and value.builtin:
        return {
            "kind": str(value.kind),
            "builtin": str(value.builtin),
            "params": _plain_config_data(value.params),
        }
    if isinstance(value, AgentSection):
        data = {item.name: _plain_config_data(getattr(value, item.name)) for item in fields(value)}
        if value.runtime_profile == "flight_software":
            data.pop("runtime_profile", None)
        if not value.force_models:
            data.pop("force_models", None)
        return data
    if isinstance(value, SimulatorSection):
        data = {item.name: _plain_config_data(getattr(value, item.name)) for item in fields(value)}
        # The omitted and explicit Earth cases retain their established digest.
        if value.central_body == "earth":
            data.pop("central_body", None)
        else:
            data.pop("frames", None)
            data.pop("termination", None)
        if value.planetary is None:
            data.pop("planetary", None)
        if value.additional_gravity is None:
            data.pop("additional_gravity", None)
        if not value.system_force_models:
            data.pop("system_force_models", None)
        if not value.collisions:
            data.pop("collisions", None)
        return data
    if isinstance(value, OutputsSection):
        data = {item.name: _plain_config_data(getattr(value, item.name)) for item in fields(value)}
        if not value.campaign_retention:
            data.pop("campaign_retention", None)
        return data
    if isinstance(value, MonteCarloConstraint):
        data = {item.name: _plain_config_data(getattr(value, item.name)) for item in fields(value)}
        # A path-backed constraint has no constant value.  Omit the dataclass
        # default so round-tripping the normalized config preserves the
        # authored one-of contract and does not look like an explicit null.
        if value.right_path is not None and value.right_value is None:
            data.pop("right_value", None)
        return data
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain_config_data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {_plain_config_data(k): _plain_config_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_config_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_config_data(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class AlgorithmPointer:
    kind: str = "python"
    builtin: str | None = None
    module: str | None = None
    class_name: str | None = None
    function: str | None = None
    file: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgePointer:
    enabled: bool = False
    mode: str = "sil"
    endpoint: str | None = None
    module: str | None = None
    class_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlightSoftwareSection:
    profile: str | None = None
    stack: str | None = None
    module: str | None = None
    class_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    task_period_s: float | None = None
    hardware_profile: str | None = None
    mission_load: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentSection:
    object_id: str = ""
    kind: str = "satellite"
    enabled: bool = True
    role: str = "agent"
    runtime_profile: str = "flight_software"
    propagation_method: str = ""
    general: dict[str, Any] = field(default_factory=dict)
    specs: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    reference_orbit: dict[str, Any] = field(default_factory=dict)
    flight_software: FlightSoftwareSection | None = None
    guidance: AlgorithmPointer | None = None
    base_guidance: AlgorithmPointer | None = None
    guidance_modifiers: list[AlgorithmPointer] = field(default_factory=list)
    orbit_control: AlgorithmPointer | None = None
    attitude_control: AlgorithmPointer | None = None
    mission_strategy: AlgorithmPointer | None = None
    mission_execution: AlgorithmPointer | None = None
    mission_objectives: list[AlgorithmPointer] = field(default_factory=list)
    force_models: list[AlgorithmPointer] = field(default_factory=list)
    bridge: BridgePointer | None = None
    knowledge: dict[str, Any] = field(default_factory=dict)
    force_models: list[AlgorithmPointer] = field(default_factory=list)


@dataclass(frozen=True)
class GroundStationSection:
    id: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = 0.0
    max_range_km: float | None = None
    enabled: bool = True
    measurements: dict[str, Any] = field(default_factory=dict)


class _TypedConfigDict(dict):
    _defaults: dict[str, Any] = {}

    def __init__(self, value: Any = None, **overrides: Any) -> None:
        data = deepcopy(self._defaults)
        if value is not None:
            data.update(deepcopy(dict(value)))
        data.update(deepcopy(overrides))
        super().__init__(data)


class SimulatorAccelerationSection(_TypedConfigDict):
    _defaults = {
        "mode": "off",
        "warmup": False,
        "env_override": True,
    }

    @property
    def mode(self) -> str:
        return str(self.get("mode", "off") or "off")

    @property
    def warmup(self) -> bool:
        return bool(self.get("warmup", False))

    @property
    def env_override(self) -> bool:
        return bool(self.get("env_override", True))


class SimulatorExecutionSection(_TypedConfigDict):
    _defaults = {
        "policy": "configured",
        "object_parallelism": {
            "enabled": False,
            "backend": "serial",
            "workers": 0,
            "max_workers": 0,
            "reserve_workers": 1,
            "min_objects": 3,
        },
        "runtime_profiler": {
            "enabled": True,
        },
        "controller": {
            "orbit_budget_ms": 2.0,
            "attitude_budget_ms": 2.0,
            "deadline_policy": "record",
        },
    }

    @property
    def object_parallelism(self) -> dict[str, Any]:
        return dict(self.get("object_parallelism", {}) or {})

    @property
    def runtime_profiler(self) -> dict[str, Any]:
        return dict(self.get("runtime_profiler", {}) or {})

    @property
    def controller(self) -> dict[str, Any]:
        return dict(self.get("controller", {}) or {})


class SimulatorFramesSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {
        "model": "simple_gmst",
        "eop_path": None,
        "eop_extrapolation": "error",
        "time_scale_model": "utc_only",
        "tt_minus_utc_s": None,
        "dut1_s": None,
        "xp_arcsec": None,
        "yp_arcsec": None,
        "dat_s": None,
        "ddpsi_rad": 0.0,
        "ddeps_rad": 0.0,
    }

    @property
    def model(self) -> str:
        return str(self.get("model", "simple_gmst") or "simple_gmst")

    @property
    def eop_path(self) -> str | None:
        value = self.get("eop_path")
        return None if value in (None, "") else str(value)


@dataclass(frozen=True)
class SimulatorSection:
    duration_s: float = 3600.0
    dt_s: float = 1.0
    initial_jd_utc: float | None = None
    resource_profile: str | None = None
    acceleration: SimulatorAccelerationSection = field(default_factory=lambda: SimulatorAccelerationSection())
    execution: SimulatorExecutionSection = field(default_factory=lambda: SimulatorExecutionSection())
    frames: SimulatorFramesSection = field(default_factory=lambda: SimulatorFramesSection())
    dynamics: SimulatorDynamicsSection = field(default_factory=lambda: SimulatorDynamicsSection())
    environment: SimulatorEnvironmentSection = field(default_factory=lambda: SimulatorEnvironmentSection())
    plugin_validation: SimulatorPluginValidationSection = field(
        default_factory=lambda: SimulatorPluginValidationSection()
    )
    termination: SimulatorTerminationSection = field(default_factory=lambda: SimulatorTerminationSection())
    system_force_models: list[AlgorithmPointer] = field(default_factory=list)
    collisions: dict[str, Any] = field(default_factory=dict)
    central_body: str = "earth"
    planetary: dict[str, Any] | None = None
    additional_gravity: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceleration", SimulatorAccelerationSection(self.acceleration))
        object.__setattr__(self, "execution", SimulatorExecutionSection(self.execution))
        object.__setattr__(self, "frames", SimulatorFramesSection(self.frames))
        object.__setattr__(self, "dynamics", SimulatorDynamicsSection(self.dynamics))
        object.__setattr__(self, "environment", SimulatorEnvironmentSection(self.environment))
        object.__setattr__(self, "plugin_validation", SimulatorPluginValidationSection(self.plugin_validation))
        object.__setattr__(self, "termination", SimulatorTerminationSection(self.termination))

class SimulatorDynamicsSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def orbit(self) -> dict[str, Any]:
        return dict(self.get("orbit", {}) or {})

    @property
    def attitude(self) -> dict[str, Any]:
        return dict(self.get("attitude", {}) or {})

    @property
    def rocket(self) -> dict[str, Any]:
        return dict(self.get("rocket", {}) or {})


class SimulatorEnvironmentSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def atmosphere_env(self) -> dict[str, Any]:
        return dict(self.get("atmosphere_env", {}) or {})


class SimulatorPluginValidationSection(_TypedConfigDict):
    _defaults = {"strict": True, "strict_runtime": False}

    @property
    def strict(self) -> bool:
        return bool(self.get("strict", True))

    @property
    def strict_runtime(self) -> bool:
        return bool(self.get("strict_runtime", False))


class SimulatorTerminationSection(_TypedConfigDict):
    _defaults = {
        "earth_impact_enabled": True,
        "earth_radius_km": 6378.137,
        "by_object": {},
    }

    @property
    def earth_impact_enabled(self) -> bool:
        return bool(self.get("earth_impact_enabled", True))

    @property
    def earth_radius_km(self) -> float:
        return float(self.get("earth_radius_km", 6378.137))

    @property
    def by_object(self) -> dict[str, Any]:
        return dict(self.get("by_object", {}) or {})


class OutputStatsSection(_TypedConfigDict):
    _defaults = {
        "print_summary": True,
        "save_json": True,
        "save_full_log": True,
        "save_history_npz": False,
        "controller_debug": False,
    }

    @property
    def print_summary(self) -> bool:
        return bool(self.get("print_summary", True))

    @property
    def save_json(self) -> bool:
        return bool(self.get("save_json", True))

    @property
    def save_full_log(self) -> bool:
        return bool(self.get("save_full_log", True))

    @property
    def save_csv(self) -> bool:
        return bool(self.get("save_csv", False))

    @property
    def save_history_npz(self) -> bool:
        return bool(self.get("save_history_npz", False))

    @property
    def controller_debug(self) -> bool:
        return bool(self.get("controller_debug", False))


class OutputPlotsSection(_TypedConfigDict):
    _defaults = {
        "enabled": True,
        "figure_ids": [],
        "dpi": 150,
        "style": "oel_dark",
    }

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", True))

    @property
    def figure_ids(self) -> list[Any]:
        return list(self.get("figure_ids", []) or [])

    @property
    def dpi(self) -> int:
        return int(self.get("dpi", 150))

    @property
    def style(self) -> str:
        return str(self.get("style", "oel_dark") or "oel_dark")


class OutputAnimationsSection(_TypedConfigDict):
    _defaults = {
        "enabled": False,
        "types": [],
        "fps": 30.0,
    }

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", False))

    @property
    def types(self) -> list[Any]:
        return list(self.get("types", []) or [])

    @property
    def fps(self) -> float:
        return float(self.get("fps", 30.0))

    @property
    def style(self) -> str | None:
        value = self.get("style")
        if value in (None, ""):
            return None
        return str(value)


class OutputMonteCarloSection(_TypedConfigDict):
    _defaults = {
        "save_histograms": False,
        "display_histograms": False,
        "save_ops_dashboard": True,
        "display_ops_dashboard": False,
        "save_iteration_summaries": False,
        "success_termination_reasons": ["rocket_orbit_insertion"],
    }

    @property
    def save_histograms(self) -> bool:
        return bool(self.get("save_histograms", False))

    @property
    def display_histograms(self) -> bool:
        return bool(self.get("display_histograms", False))

    @property
    def save_iteration_summaries(self) -> bool:
        return bool(self.get("save_iteration_summaries", False))

    @property
    def success_termination_reasons(self) -> list[Any]:
        return list(self.get("success_termination_reasons", ["rocket_orbit_insertion"]) or [])


class OutputAIReportSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", False))

    @property
    def provider(self) -> str:
        return str(self.get("provider", "ollama") or "ollama")

    @property
    def model(self) -> str:
        return str(self.get("model", "") or "")


class OutputAIConfigSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", True))

    @property
    def provider(self) -> str:
        return str(self.get("provider", "ollama") or "ollama")

    @property
    def model(self) -> str:
        return str(self.get("model", "") or "")


class OutputResourceLimitsSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {
        "max_history_memory_mb": None,
    }

    @property
    def max_history_memory_mb(self) -> float | None:
        value = self.get("max_history_memory_mb")
        if value in (None, ""):
            return None
        return float(value)


class OutputReviewSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {
        "enabled": False,
        "detail": "standard",
        "strict": False,
    }

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", False))

    @property
    def detail(self) -> str:
        return str(self.get("detail", "standard") or "standard")

    @property
    def strict(self) -> bool:
        return bool(self.get("strict", False))


class OutputOrbitalAnalysisSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {
        "enabled": False,
        "coverage": [],
        "directed_links": [],
    }

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", False))

    @property
    def coverage(self) -> list[dict[str, Any]]:
        return [dict(value or {}) for value in list(self.get("coverage", []) or [])]

    @property
    def directed_links(self) -> list[dict[str, Any]]:
        return [dict(value or {}) for value in list(self.get("directed_links", []) or [])]


@dataclass(frozen=True)
class OutputsSection:
    output_dir: str = "outputs"
    mode: str = "interactive"
    stats: OutputStatsSection = field(default_factory=OutputStatsSection)
    plots: OutputPlotsSection = field(default_factory=OutputPlotsSection)
    animations: OutputAnimationsSection = field(default_factory=OutputAnimationsSection)
    monte_carlo: OutputMonteCarloSection = field(default_factory=OutputMonteCarloSection)
    campaign_retention: dict[str, Any] = field(default_factory=dict)
    ai_report: OutputAIReportSection = field(default_factory=OutputAIReportSection)
    ai_config: OutputAIConfigSection = field(default_factory=OutputAIConfigSection)
    review: OutputReviewSection = field(default_factory=OutputReviewSection)
    orbital_analysis: OutputOrbitalAnalysisSection = field(default_factory=OutputOrbitalAnalysisSection)
    resource_limits: OutputResourceLimitsSection = field(default_factory=OutputResourceLimitsSection)
    campaign_retention: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stats", OutputStatsSection(self.stats))
        object.__setattr__(self, "plots", OutputPlotsSection(self.plots))
        object.__setattr__(self, "animations", OutputAnimationsSection(self.animations))
        object.__setattr__(self, "monte_carlo", OutputMonteCarloSection(self.monte_carlo))
        object.__setattr__(self, "ai_report", OutputAIReportSection(self.ai_report))
        object.__setattr__(self, "ai_config", OutputAIConfigSection(self.ai_config))
        object.__setattr__(self, "review", OutputReviewSection(self.review))
        object.__setattr__(self, "orbital_analysis", OutputOrbitalAnalysisSection(self.orbital_analysis))
        object.__setattr__(self, "resource_limits", OutputResourceLimitsSection(self.resource_limits))


@dataclass(frozen=True)
class MonteCarloVariation:
    parameter_path: str
    mode: str = "choice"
    options: list[Any] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    weights: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class MonteCarloBundleOption:
    label: str
    values: dict[str, Any]
    weight: float = 1.0


@dataclass(frozen=True)
class MonteCarloBundle:
    name: str
    options: list[MonteCarloBundleOption] = field(default_factory=list)


@dataclass(frozen=True)
class MonteCarloComparison:
    bundle: str
    runs_per_option: int
    paired: bool = True


@dataclass(frozen=True)
class MonteCarloConstraint:
    left_path: str
    op: str
    right_path: str | None = None
    right_value: Any = None


@dataclass(frozen=True)
class MonteCarloCorrelatedNormal:
    name: str
    parameter_paths: list[str]
    means: list[float]
    stds: list[float]
    correlation: list[list[float]]


@dataclass(frozen=True)
class MonteCarloSection:
    enabled: bool = False
    iterations: int = 1
    base_seed: int = 0
    parallel_enabled: bool = False
    parallel_workers: int = 0
    variations: list[MonteCarloVariation] = field(default_factory=list)
    bundles: list[MonteCarloBundle] = field(default_factory=list)
    comparison: MonteCarloComparison | None = None
    constraints: list[MonteCarloConstraint] = field(default_factory=list)
    correlated_normals: list[MonteCarloCorrelatedNormal] = field(default_factory=list)
    max_attempts: int = 1000


@dataclass(frozen=True)
class AnalysisExecutionSection:
    parallel_enabled: bool = False
    parallel_workers: int = 0
    failure_policy: str = "fail_fast"


@dataclass(frozen=True)
class AnalysisBaselineSection:
    enabled: bool = False
    mode: str = "none"
    summary_json: str = ""


@dataclass(frozen=True)
class AnalysisMonteCarloSection:
    iterations: int = 1
    base_seed: int = 0
    variations: list[MonteCarloVariation] = field(default_factory=list)
    bundles: list[MonteCarloBundle] = field(default_factory=list)
    comparison: MonteCarloComparison | None = None
    constraints: list[MonteCarloConstraint] = field(default_factory=list)
    correlated_normals: list[MonteCarloCorrelatedNormal] = field(default_factory=list)
    max_attempts: int = 1000


@dataclass(frozen=True)
class SensitivityParameter:
    parameter_path: str
    values: list[Any] = field(default_factory=list)
    distribution: str = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass(frozen=True)
class SensitivitySection:
    method: str = "one_at_a_time"
    samples: int = 0
    seed: int = 0
    parameters: list[SensitivityParameter] = field(default_factory=list)


@dataclass(frozen=True)
class CovarianceObjectSection:
    enabled: bool = True
    frame: str = "eci"
    covariance: list[list[float]] = field(default_factory=list)
    diagonal: list[float] = field(default_factory=list)
    position_sigma_km: float | None = None
    velocity_sigma_km_s: float | None = None


@dataclass(frozen=True)
class CovarianceCollisionScreeningSection:
    enabled: bool = False
    hard_body_radius_km: float = 0.01
    method: str = "small_object"


@dataclass(frozen=True)
class CovariancePairSection:
    deputy_id: str
    chief_id: str
    collision_screening: CovarianceCollisionScreeningSection = field(
        default_factory=CovarianceCollisionScreeningSection
    )


@dataclass(frozen=True)
class CovarianceFiniteDifferenceSection:
    position_step_km: float = 1e-3
    velocity_step_km_s: float = 1e-6


@dataclass(frozen=True)
class CovarianceProcessNoiseSection:
    enabled: bool = False
    acceleration_sigma_km_s2: float = 0.0


@dataclass(frozen=True)
class CovarianceSection:
    enabled: bool = True
    objects: dict[str, CovarianceObjectSection] = field(default_factory=dict)
    pairs: list[CovariancePairSection] = field(default_factory=list)
    finite_difference: CovarianceFiniteDifferenceSection = field(default_factory=CovarianceFiniteDifferenceSection)
    process_noise: CovarianceProcessNoiseSection = field(default_factory=CovarianceProcessNoiseSection)
    write_review_tables: bool = True
    numeric_backend: str = "rust"


@dataclass(frozen=True)
class MissionRecoverySection:
    enabled: bool = False
    object_id: str = ""
    goal: str = "orbit_shape"
    assessment_time_s: float | str = "final"
    slot_tolerance_deg: float = 1.0
    max_phasing_orbits: int = 5000
    planner: dict[str, Any] = field(default_factory=dict)
    propulsion: dict[str, Any] = field(default_factory=dict)
    element_tolerances: dict[str, float] = field(default_factory=dict)
    target_orbit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisSection:
    enabled: bool = False
    study_type: str = "monte_carlo"
    execution: AnalysisExecutionSection = field(default_factory=AnalysisExecutionSection)
    metrics: list[Any] = field(default_factory=list)
    baseline: AnalysisBaselineSection = field(default_factory=AnalysisBaselineSection)
    monte_carlo: AnalysisMonteCarloSection = field(default_factory=AnalysisMonteCarloSection)
    sensitivity: SensitivitySection = field(default_factory=SensitivitySection)
    covariance: CovarianceSection = field(default_factory=CovarianceSection)
    mission_recovery: MissionRecoverySection = field(default_factory=MissionRecoverySection)
    orbital_delivery: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationScenarioConfig:
    schema_version: str = LEGACY_SCENARIO_SCHEMA_VERSION
    scenario_name: str = "unnamed_scenario"
    scenario_description: str = ""
    rocket: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="rocket"))
    chaser: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="chaser"))
    target: AgentSection = field(default_factory=lambda: AgentSection(enabled=True, role="target"))
    objects: dict[str, AgentSection] = field(default_factory=dict)
    ground_stations: list[GroundStationSection] = field(default_factory=list)
    simulator: SimulatorSection = field(default_factory=SimulatorSection)
    outputs: OutputsSection = field(default_factory=OutputsSection)
    monte_carlo: MonteCarloSection = field(default_factory=MonteCarloSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_segment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = _plain_config_data(self)
        if not self.ground_segment:
            data.pop("ground_segment", None)
        for legacy_key in ("rocket", "chaser", "target", "monte_carlo"):
            data.pop(legacy_key, None)
        return data
