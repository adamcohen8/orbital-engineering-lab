"""Static ownership maps for the OEL Scale data platform."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleFamily:
    name: str
    module: str
    capabilities: tuple[str, ...]
    facade: str


SCALE_STORE_FAMILIES: tuple[ScaleFamily, ...] = (
    ScaleFamily("models", "sim.scale.store_models", ("StoreSummary", "OnpRefinementRunRecord"), "sim.scale.store"),
    ScaleFamily("connection", "sim.scale.store_connection", ("summarize_store",), "sim.scale.store"),
    ScaleFamily("schema", "sim.scale.store_schema", ("initialize_store", "_create_schema"), "sim.scale.store"),
    ScaleFamily("catalog", "sim.scale.store_catalog", ("ingest_tle_records", "load_current_tle_states"), "sim.scale.store"),
    ScaleFamily(
        "measurements",
        "sim.scale.store_measurements",
        ("ingest_track_observation_records", "load_ground_station_measurement_rows"),
        "sim.scale.store",
    ),
    ScaleFamily("orbit_determination", "sim.scale.store_od", ("write_od_solution_records", "write_maneuver_flags", "load_exact_od_assessment_inputs"), "sim.scale.store"),
    ScaleFamily(
        "propagation",
        "sim.scale.store_propagation",
        ("write_propagation_products", "load_completed_propagation_products"),
        "sim.scale.store",
    ),
    ScaleFamily("screening", "sim.scale.store_screening", ("write_screening_results",), "sim.scale.store"),
    ScaleFamily(
        "refinement",
        "sim.scale.store_refinement",
        ("claim_onp_refinement_runs", "recover_incomplete_onp_refinement_runs"),
        "sim.scale.store",
    ),
    ScaleFamily("campaigns", "sim.scale.store_campaigns", ("write_sensitivity_campaign",), "sim.scale.store"),
)


SCALE_MONTE_CARLO_FAMILIES: tuple[ScaleFamily, ...] = (
    ScaleFamily("planning", "sim.scale.monte_carlo_planning", ("build_monte_carlo_campaign",), "sim.scale.monte_carlo"),
    ScaleFamily(
        "batch_parity",
        "sim.scale.monte_carlo_batch_parity",
        ("_build_batch_propagation_parity",),
        "sim.scale.monte_carlo",
    ),
    ScaleFamily("quality", "sim.scale.monte_carlo_quality", ("_evaluate_quality_gates",), "sim.scale.monte_carlo"),
    ScaleFamily("reporting", "sim.scale.monte_carlo_reporting", ("_write_serial_run_summary",), "sim.scale.monte_carlo"),
    ScaleFamily(
        "execution",
        "sim.scale.monte_carlo_execution",
        ("run_monte_carlo_campaign", "run_monte_carlo_serial", "run_monte_carlo_parallel_cpu"),
        "sim.scale.monte_carlo",
    ),
)


SCALE_CLI_FAMILIES: tuple[ScaleFamily, ...] = (
    ScaleFamily(
        "catalog",
        "sim.scale.cli.handlers.catalog",
        ("dry-run", "validate-scale", "ingest-tle", "ingest-ogp-product"),
        "sim.scale.__main__",
    ),
    ScaleFamily(
        "observations",
        "sim.scale.cli.handlers.observations",
        (
            "ingest-ground-measurements",
            "export-ground-measurement-packet",
            "fit-ground-measurement-od",
            "fit-ground-sensor-od",
            "assess-ground-measurement-quality",
            "detect-maneuvers",
            "fit-sgp4-od",
            "fit-ogp-od",
            "freeze-ogp-od",
            "assess-frozen-ogp-od",
            "fit-ground-sgp4-sensor-od",
            "fit-ground-ogp-sensor-od",
            "run-synthetic-sgp4-sensor-od",
            "run-synthetic-ogp-sensor-od",
            "run-synthetic-ground-sensor-od",
        ),
        "sim.scale.__main__",
    ),
    ScaleFamily(
        "operations",
        "sim.scale.cli.handlers.operations",
        (
            "propagate-ogp",
            "propagate-sgp4",
            "plan-recompute",
            "plan-observation-recompute",
            "run-recompute-propagation",
            "run-recompute-screening",
            "screen-pairs",
            "refine-candidates",
            "handoff-onp",
            "handoff-hpop",
            "materialize-onp-refinement",
            "run-onp-refinement-queue",
            "run-catalog-workflow",
            "run-recompute-workflow",
            "run-refresh-to-onp-workflow",
        ),
        "sim.scale.__main__",
    ),
    ScaleFamily(
        "campaigns",
        "sim.scale.cli.handlers.campaigns",
        (
            "materialize-sensitivity",
            "run-sensitivity-serial",
            "run-sensitivity-parallel-cpu",
            "materialize-monte-carlo",
            "run-monte-carlo-serial",
            "run-monte-carlo-parallel-cpu",
        ),
        "sim.scale.__main__",
    ),
)
