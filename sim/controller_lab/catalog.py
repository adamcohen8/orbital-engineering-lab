from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from sim.installation.resources import ResourceNotFoundError, resource_path


@dataclass(frozen=True)
class ControllerBenchCatalogEntry:
    """Machine-readable controller-bench starting point for agents."""

    suite_id: str
    title: str
    description: str
    suite_path: str
    intent_tags: tuple[str, ...]
    target_object: str
    target_slot: str
    objective_families: tuple[str, ...]
    runtime_tier: str = "standard"
    maturity: str = "supported"
    answers: tuple[str, ...] = ()
    does_not_prove: tuple[str, ...] = ()
    evidence_queries: tuple[str, ...] = (
        "workflow_metadata",
        "controller_bench_runs",
        "controller_bench_leaderboard",
        "controller_bench_failures",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CATALOG: tuple[ControllerBenchCatalogEntry, ...] = (
    ControllerBenchCatalogEntry(
        suite_id="flight_software_conformance",
        title="Flight-Software Stack Conformance Bench",
        description="Exercise all built-in satellite stack families, translation modes, relative laws, and representative cadences.",
        suite_path="configs/controller_bench_fsw_conformance.yaml",
        intent_tags=("gnc", "flight_software", "conformance", "modes", "multirate", "actuators"),
        target_object="chaser",
        target_slot="flight_software",
        objective_families=(
            "flight_software_stack_conformance",
            "actuator_realization",
            "multirate_consistency",
        ),
        runtime_tier="fast",
        answers=(
            "Does every built-in satellite flight-software stack and translation mode execute through typed boundaries?",
            "Are configured task releases preserved and actuator commands physically realized?",
        ),
        does_not_prove=("Global performance, robustness, or flight qualification for any stack.",),
    ),
    ControllerBenchCatalogEntry(
        suite_id="flight_software_frame_cadence",
        title="Flight-Software Frame And Cadence Stress Bench",
        description="Probe both signs of every RIC axis for all relative laws at several independent software cadences.",
        suite_path="configs/controller_bench_fsw_frame_cadence.yaml",
        intent_tags=("gnc", "frames", "sign", "ric", "multirate", "cadence", "conformance"),
        target_object="chaser",
        target_slot="flight_software",
        objective_families=("flight_software_stack_conformance", "multirate_consistency"),
        runtime_tier="standard",
        answers=(
            "Are RIC frame/sign responses consistent across positive and negative axis probes?",
            "Do controller outcomes remain well formed across supported task cadences?",
        ),
        does_not_prove=("Stability outside the signed-axis envelope or arbitrary task schedules.",),
    ),
    ControllerBenchCatalogEntry(
        suite_id="rpo_hcw_rendezvous",
        title="RPO HCW LQR Rendezvous Bench",
        description="Compare HCW LQR orbit-control variants across nominal and cross-track RPO cases.",
        suite_path="configs/controller_bench_rendezvous.yaml",
        intent_tags=("rpo", "rendezvous", "orbit_control", "controller_selection"),
        target_object="chaser",
        target_slot="flight_software",
        objective_families=("relative_rendezvous",),
        runtime_tier="fast",
        answers=(
            "Which HCW LQR variant satisfies the configured safe-rendezvous objective most often?",
            "Which cases fail and what failure modes are recorded?",
        ),
        does_not_prove=(
            "Operational rendezvous safety outside the configured cases.",
            "Controller superiority without the stated objective thresholds.",
        ),
    ),
    ControllerBenchCatalogEntry(
        suite_id="terminal_rendezvous_box",
        title="Terminal RIC Rendezvous Box Bench",
        description="Compare orbit controllers against terminal RIC position/rate box criteria.",
        suite_path="configs/controller_bench_terminal_rendezvous_box.yaml",
        intent_tags=("rpo", "terminal_box", "rendezvous", "orbit_control"),
        target_object="chaser",
        target_slot="flight_software",
        objective_families=("terminal_rendezvous_box",),
        runtime_tier="standard",
        answers=(
            "Which variant best satisfies terminal RIC component and rate limits?",
            "Which terminal box criteria fail by case?",
        ),
        does_not_prove=("Docking qualification or contact dynamics.",),
    ),
    ControllerBenchCatalogEntry(
        suite_id="attitude_hold",
        title="Attitude Hold Bench",
        description="Compare attitude controllers across hold and rate-recovery cases.",
        suite_path="configs/controller_bench_attitude.yaml",
        intent_tags=("attitude", "adcs", "settling", "controller_selection"),
        target_object="target",
        target_slot="flight_software",
        objective_families=("metric_criteria",),
        runtime_tier="fast",
        answers=(
            "Which attitude-control variant passes the configured final-error and rate criteria?",
            "Which recovery cases fail and what retuning action is suggested?",
        ),
        does_not_prove=("Flight-qualified ADCS performance or actuator margin beyond recorded metrics.",),
    ),
    ControllerBenchCatalogEntry(
        suite_id="attitude_slew_settle",
        title="Attitude Slew-And-Settle Bench",
        description="Evaluate attitude controllers against explicit slew-settle objective criteria.",
        suite_path="configs/controller_bench_attitude_slew_settle.yaml",
        intent_tags=("attitude", "adcs", "slew", "settle", "controller_selection"),
        target_object="target",
        target_slot="flight_software",
        objective_families=("attitude_slew_settle",),
        runtime_tier="standard",
        answers=("Which attitude variant settles fastest while meeting the configured error/rate limits?",),
        does_not_prove=("Robustness outside the configured attitude cases.",),
    ),
    ControllerBenchCatalogEntry(
        suite_id="rocket_gnc",
        title="Rocket GNC Orbit-Insertion Bench",
        description="Compare rocket base-guidance pitch programs across insertion and ascent-load cases.",
        suite_path="configs/controller_bench_rocket_gnc.yaml",
        intent_tags=("rocket", "gnc", "launch", "insertion", "base_guidance"),
        target_object="rocket",
        target_slot="base_guidance",
        objective_families=("rocket_orbit_insertion",),
        runtime_tier="heavy",
        answers=(
            "Which pitch-program variant reaches the target insertion envelope?",
            "Which variant best trades insertion time, max-Q, angle of attack, TVC demand, and propellant margin?",
        ),
        does_not_prove=("Validated launch-vehicle design or operational launch analysis.",),
    ),
    ControllerBenchCatalogEntry(
        suite_id="aero_assist_stack",
        title="Aero-Assist Integrated Stack Bench",
        description="Compare integrated orbit, attitude, and aero-property controller stacks.",
        suite_path="configs/controller_bench_aero_assist.yaml",
        intent_tags=("aero_assist", "integrated_stack", "orbit_control", "attitude_control"),
        target_object="chaser",
        target_slot="flight_software",
        objective_families=("relative_rendezvous", "integrated_stack"),
        runtime_tier="standard",
        answers=("Which integrated atmospheric-pass stack performs best under the configured scenario?",),
        does_not_prove=("Atmospheric flight qualification or aero-model validation outside the configured evidence.",),
    ),
)


def list_catalog_entries() -> tuple[ControllerBenchCatalogEntry, ...]:
    available = []
    for entry in _CATALOG:
        try:
            resource_path(entry.suite_path)
        except ResourceNotFoundError:
            continue
        available.append(entry)
    return tuple(available)


def get_catalog_entry(suite_id: str) -> ControllerBenchCatalogEntry | None:
    wanted = str(suite_id or "").strip().lower()
    for entry in _CATALOG:
        if entry.suite_id.lower() == wanted:
            return entry
    return None


def plan_controller_bench(intent: str, *, limit: int = 5) -> dict[str, Any]:
    """Rank catalog entries against a plain-language controller-bench intent."""

    query = str(intent or "").strip().lower()
    tokens = {tok for tok in query.replace("_", " ").replace("-", " ").split() if tok}
    ranked: list[dict[str, Any]] = []
    for entry in list_catalog_entries():
        haystack = " ".join(
            [
                entry.suite_id,
                entry.title,
                entry.description,
                " ".join(entry.intent_tags),
                " ".join(entry.objective_families),
                entry.target_object,
                entry.target_slot,
            ]
        ).lower()
        score = 0
        matched: list[str] = []
        for token in sorted(tokens):
            if token and token in haystack:
                score += 1
                matched.append(token)
        for tag in entry.intent_tags:
            if tag.lower() in query:
                score += 3
                matched.append(tag)
        if score or not query:
            ranked.append({"score": score, "matched_terms": sorted(set(matched)), "entry": entry.to_dict()})
    ranked.sort(key=lambda item: (-int(item["score"]), item["entry"]["suite_id"]))
    return {
        "intent": intent,
        "matches": ranked[: max(1, int(limit))],
        "guidance": (
            "Validate the selected suite with --controller-bench --validate-only before running. "
            "Use the configured objective families as the boundary for any controller-selection claim."
        ),
    }


def scaffold_from_catalog(
    suite_id: str,
    output_path: str | Path,
    *,
    suite_name: str = "",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a catalog suite to a new path with optional suite/output overrides."""

    entry = get_catalog_entry(suite_id)
    if entry is None:
        raise ValueError(f"Unknown controller-bench catalog suite id: {suite_id}")
    source = resource_path(entry.suite_path)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Catalog suite YAML root must be a mapping: {source}")
    for case in list(data.get("cases", []) or []):
        if not isinstance(case, dict):
            continue
        case_path = case.get("config_path")
        if not case_path:
            continue
        case_config = Path(str(case_path)).expanduser()
        if not case_config.is_absolute():
            case["config_path"] = str((source.parent / case_config).resolve())
    if suite_name:
        data["suite_name"] = str(suite_name)
    if output_dir is not None:
        data["output_dir"] = str(output_dir)
    data.setdefault("metadata", {})
    if isinstance(data["metadata"], dict):
        data["metadata"]["scaffolded_from"] = entry.suite_id
        data["metadata"]["source_suite_path"] = entry.suite_path
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return {
        "suite_id": entry.suite_id,
        "source_suite": str(source),
        "output_path": str(destination),
        "suite_name": data.get("suite_name"),
        "output_dir": data.get("output_dir"),
    }
