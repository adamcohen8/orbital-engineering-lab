"""Static ownership map for runtime construction and single-run execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeFamily:
    name: str
    module: str
    facade: str
    capabilities: tuple[str, ...]
    patch_seams: tuple[str, ...] = ()


RUNTIME_CONSTRUCTION_FAMILIES: tuple[RuntimeFamily, ...] = (
    RuntimeFamily("models", "sim.runtime.models", "sim.runtime_support", ("AgentRuntime",)),
    RuntimeFamily(
        "compatibility",
        "sim.runtime.compat",
        "sim.runtime_support",
        ("_module_obj", "_compatible_keyword_args", "_call_with_compat_kwargs"),
    ),
    RuntimeFamily(
        "state_initialization",
        "sim.runtime.state_initialization",
        "sim.runtime_support",
        ("_apply_relative_init_from_reference", "_apply_relative_cislunar_init_from_reference"),
        ("_default_truth_from_agent",),
    ),
    RuntimeFamily(
        "actuators",
        "sim.runtime.actuator_factory",
        "sim.runtime_support",
        ("_build_satellite_actuator_stack_from_specs", "_resolve_satellite_inertia_kg_m2"),
    ),
    RuntimeFamily(
        "satellites",
        "sim.runtime.satellite_factory",
        "sim.runtime_support",
        ("_build_orbit_propagator",),
        ("_create_satellite_runtime",),
    ),
    RuntimeFamily(
        "rockets",
        "sim.runtime.rocket_factory",
        "sim.runtime_support",
        ("_resolve_rocket_stack", "_build_rocket_guidance"),
        ("_create_rocket_runtime",),
    ),
    RuntimeFamily(
        "knowledge",
        "sim.runtime.knowledge_factory",
        "sim.runtime_support",
        (),
        ("_build_knowledge_base",),
    ),
    RuntimeFamily(
        "missions",
        "sim.runtime.mission_runtime",
        "sim.runtime_support",
        ("_deploy_from_rocket", "_run_mission_modules", "_run_mission_strategy", "_run_mission_execution"),
    ),
    RuntimeFamily(
        "commands",
        "sim.runtime.commands",
        "sim.runtime_support",
        ("_combine_commands", "_command_to_dict"),
    ),
)


SINGLE_RUN_COLLABORATORS: tuple[tuple[str, str], ...] = (
    ("runtime profiling", "sim.execution.runtime_profile"),
    ("object-worker transport", "sim.execution.object_workers"),
    ("execution-local budgets", "sim.execution.worker_budget"),
    ("object-step coordination", "sim.execution.object_step_coordinator"),
    ("history storage", "sim.execution.single_run_history"),
    ("passive native histories", "sim.execution.passive_history"),
    ("payload assembly", "sim.reporting.run_payload_assembly"),
)
