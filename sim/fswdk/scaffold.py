"""Candidate workspace scaffolding for the FSWDK."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sim.gnc_workbench import (
    ScaffoldRequest,
    scaffold_attitude_controller_workbench,
    scaffold_orbit_controller_workbench,
)

from .candidate import ROOT, inspect_candidate


@dataclass(frozen=True, slots=True)
class CandidateScaffoldResult:
    root_dir: Path
    manifest_path: Path
    files_written: tuple[Path, ...]
    candidate: dict[str, Any]


def _slug(value: str) -> str:
    slug = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")
    return slug or "custom_fsw"


def _write_new(
    path: Path,
    content: str,
    *,
    force: bool,
    writer: Any | None = None,
    logical_root: Path | None = None,
) -> bool:
    if path.is_symlink():
        raise PermissionError(f"Refusing to write through symbolic link: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return False
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing FSWDK file: {path}")
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    else:
        root = logical_root or path.parent
        relative = path.relative_to(root)
        payload = content.encode("utf-8")
        if path.exists() and force:
            writer.update_existing_bytes(relative, payload)
        else:
            writer.write_bytes(relative, payload, parents=True)
    return True


def scaffold_candidate(
    name: str,
    *,
    template: str = "adcs",
    project_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    class_name: str | None = None,
    force: bool = False,
    writer: Any | None = None,
) -> CandidateScaffoldResult:
    root = Path(project_root).expanduser().resolve()
    candidate_id = _slug(name)
    destination = Path(output_dir) if output_dir is not None else root / "local_experiments" / "fswdk" / candidate_id
    destination = destination.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("FSWDK candidate output must remain inside the project root.") from exc
    if writer is not None and Path(
        getattr(writer, "logical_root", writer.path)
    ).expanduser().resolve() != destination:
        raise PermissionError("FSWDK scaffold writer must be bound to the logical candidate output directory.")
    if template not in {"adcs", "rpo"}:
        raise ValueError("FSWDK template must be one of: adcs, rpo")
    envelope_path = destination / "configs" / "declared_envelope.yaml"
    bench_path = destination / "configs" / "candidate_bench.yaml"
    qualification_path = destination / "qualification.yaml"
    manifest_path = destination / "candidate.yaml"
    guide_path = destination / "FSWDK.md"
    # Validate our extra files before the workbench writes its own scaffold.
    for path in (envelope_path, bench_path, qualification_path, manifest_path, guide_path):
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("FSWDK candidate output must remain inside the project root.") from exc
    request = ScaffoldRequest(
        name=name,
        project_root=root,
        output_dir=destination,
        class_name=class_name,
        force=force,
        writer=writer,
    )
    workbench = (
        scaffold_attitude_controller_workbench(request)
        if template == "adcs"
        else scaffold_orbit_controller_workbench(request)
    )
    if writer is None:
        physical_scenario_path = workbench.scenario_path
    else:
        writer_root = Path(writer.path).expanduser().resolve()
        logical_root = Path(getattr(writer, "logical_root", destination)).expanduser().resolve()
        physical_scenario_path = writer_root / workbench.scenario_path.relative_to(logical_root)
    smoke_config = yaml.safe_load(physical_scenario_path.read_text(encoding="utf-8"))
    smoke_config["simulator"]["duration_s"] = 20.0
    smoke_config["outputs"].setdefault("stats", {})["print_summary"] = False
    smoke_payload = yaml.safe_dump(smoke_config, sort_keys=False).encode("utf-8")
    if writer is None:
        workbench.scenario_path.write_bytes(smoke_payload)
    else:
        writer.update_existing_bytes(workbench.scenario_path.relative_to(destination), smoke_payload)
    configs_dir = destination / "configs"
    outputs_root = root / "outputs" / "fswdk" / candidate_id
    bench_output = Path(os.path.relpath(outputs_root / "bench", configs_dir))
    smoke_relative = workbench.scenario_path.relative_to(destination)
    test_relative = workbench.test_path.parent.relative_to(destination)
    envelope = {
        "schema_version": "oel.fswdk.envelope.v1",
        "domain": "attitude" if template == "adcs" else "rpo",
        "simulation_only": True,
        "cases": ["nominal_smoke"],
        "task_period_s": [0.1, 0.2, 0.5] if template == "adcs" else [0.1, 0.5, 1.0],
        "assumptions": [
            "Candidate-visible inputs are restricted to the typed onboard boundary.",
            "Evidence applies only to the generated deterministic cases until explicitly expanded.",
        ],
        "limitations": [
            "The scaffold is not flight-qualified software.",
            "The initial smoke uses the workbench hardware and model assumptions declared in its scenario.",
        ],
    }
    candidate_pointer = {
        "module": workbench.module_path,
        "class_name": workbench.class_name,
        "hardware_profile": "hardware.ideal_wrench.v1",
        "task_period_s": 0.2 if template == "adcs" else 0.5,
        "params": {"satellite_id": "target" if template == "adcs" else "chaser"},
    }
    if template == "rpo":
        candidate_pointer["params"]["reference_object_id"] = "target"
    baseline_pointer: dict[str, Any] = {
        "stack": "fsw.attitude_reference" if template == "adcs" else "fsw.rpo_reference",
        "hardware_profile": "hardware.ideal_wrench.v1",
        "task_period_s": candidate_pointer["task_period_s"],
        "params": (
            {"reference_mode": "quaternion", "quaternion_bn": [1.0, 0.0, 0.0, 0.0], "max_torque_n_m": 0.05}
            if template == "adcs"
            else {"reference_object_id": "target", "translation_mode": "ric_hold", "max_acceleration_m_s2": 0.05}
        ),
    }
    objective = (
        {
            "kind": "attitude_slew_settle",
            "name": "point_and_settle",
            "object_id": "target",
            "desired_quat_bn": [1.0, 0.0, 0.0, 0.0],
            "settle_error_deg": 45.0,
            "settle_rate_norm_rad_s": 0.1,
            "settle_hold_s": 2.0,
            "max_final_attitude_error_deg": 60.0,
            "max_body_rate_norm_rad_s": 0.2,
            "max_total_torque_norm_n_m_s": 10.0,
            "require_not_terminated_early": True,
        }
        if template == "adcs"
        else {
            "kind": "nonzero_ric_hold",
            "name": "relative_hold",
            "object_id": "chaser",
            "reference_object_id": "target",
            "target_relative_state_ric_m": [0.0, -2000.0, 0.0, 0.0, 0.0, 0.0],
            "max_final_position_error_m": 5000.0,
            "max_final_velocity_error_m_s": 10.0,
        }
    )
    bench = {
        "suite_name": f"{candidate_id}_fswdk_bench",
        "description": f"FSWDK candidate-versus-reference comparison for {name}.",
        "output_dir": str(bench_output),
        "plot_mode": "save",
        "controller_target": {
            "object_id": "target" if template == "adcs" else "chaser",
            "slot": "flight_software",
        },
        "save_run_payloads": True,
        "disable_plots": True,
        "disable_animations": True,
        "print_individual_run_summaries": False,
        "engineering": {
            "enabled": True,
            "semantic_lint": True,
            "stack_conformance": True,
            "cadence_analysis": True,
            "aligned_timeline": True,
            "authority_analysis": True,
            "failure_reduction": True,
            "evidence_manifest": True,
        },
        "variants": [
            {
                "name": "reference_baseline",
                "controller": baseline_pointer,
                "sweep": {"controller.task_period_s": envelope["task_period_s"]},
            },
            {
                "name": "candidate",
                "controller": candidate_pointer,
                "sweep": {"controller.task_period_s": envelope["task_period_s"]},
            },
        ],
        "objectives": [objective],
        "cases": [{"name": "nominal_smoke", "config_path": workbench.scenario_path.name}],
    }
    qualification = {
        "schema_version": "oel.fswdk.qualification.v1",
        "candidate_id": candidate_id,
        "target": "simulation_qualified_candidate",
        "envelope": str(envelope_path.relative_to(destination)),
        "max_evidence_age_s": 86400.0,
        "requirements": [
            {
                "id": "FSW-CONTRACT-001",
                "statement": "The candidate implements the complete typed lifecycle contract.",
                "verification": "component contract test",
            },
            {
                "id": "FSW-SMOKE-001",
                "statement": "The candidate completes the deterministic nominal smoke scenario.",
                "verification": "OEL deterministic simulation",
            },
            {
                "id": "FSW-BENCH-001",
                "statement": "The candidate satisfies the declared Controller Bench objectives at all cadences.",
                "verification": "candidate-versus-reference Controller Bench",
            },
            {
                "id": "FSW-EVIDENCE-001",
                "statement": "Required gate evidence is complete, current, and content-addressed.",
                "verification": "artifact integrity inspection",
            },
        ],
        "gates": [
            {
                "id": "candidate_contract",
                "kind": "contract_test",
                "required": True,
                "requirement_ids": ["FSW-CONTRACT-001"],
            },
            {
                "id": "deterministic_smoke",
                "kind": "smoke",
                "required": True,
                "requirement_ids": ["FSW-SMOKE-001"],
            },
            {
                "id": "candidate_bench",
                "kind": "controller_bench",
                "required": True,
                "requirement_ids": ["FSW-BENCH-001"],
            },
            {
                "id": "evidence_complete",
                "kind": "artifact",
                "required": True,
                "requirement_ids": ["FSW-EVIDENCE-001"],
            },
        ],
        "non_claims": [
            "Passing this specification is simulation qualification inside the declared envelope only.",
            "The result is not flight qualification, certification, or operational approval.",
        ],
    }
    manifest = {
        "schema_version": "oel.fswdk.candidate.v1",
        "candidate_id": candidate_id,
        "revision": "0.1.0",
        "kind": "python_stack",
        "source": {
            "root": ".",
            "entrypoint": {"module": workbench.module_path, "class_name": workbench.class_name},
            "revision_id": "uncommitted-scaffold",
        },
        "interfaces": {
            "onboard_contract": "oel.fsw.boundary.v1",
            "hardware_profile": "hardware.ideal_wrench.v1",
            "task_period_s": candidate_pointer["task_period_s"],
        },
        "claims": {
            "intended_use": "attitude_control" if template == "adcs" else "rendezvous_and_proximity_operations",
            "declared_envelope": str(envelope_path.relative_to(destination)),
        },
        "verification": {
            "component_suite": str(test_relative),
            "smoke_case": str(smoke_relative),
            "bench_suite": str(bench_path.relative_to(destination)),
            "qualification_spec": str(qualification_path.relative_to(destination)),
        },
        "handling": {"classification": "private", "hosted_ai_allowed": False, "owner": ""},
    }
    guide = f"""# {name} FSWDK Candidate

This workspace contains a complete `{template}` flight-software candidate for
simulation-based development and verification. It is not flight-qualified
software.

```bash
.venv/bin/python -m sim.fswdk inspect {manifest_path.relative_to(root)}
.venv/bin/python -m sim.fswdk validate {manifest_path.relative_to(root)} --trusted-import
.venv/bin/python -m sim.fswdk test {manifest_path.relative_to(root)}
.venv/bin/python -m sim.fswdk simulate {manifest_path.relative_to(root)}
.venv/bin/python -m sim.fswdk bench {manifest_path.relative_to(root)}
.venv/bin/python -m sim.fswdk qualify {manifest_path.relative_to(root)}
```
"""
    generated = {
        envelope_path: yaml.safe_dump(envelope, sort_keys=False),
        bench_path: yaml.safe_dump(bench, sort_keys=False),
        qualification_path: yaml.safe_dump(qualification, sort_keys=False),
        manifest_path: yaml.safe_dump(manifest, sort_keys=False),
        guide_path: guide,
    }
    written = list(workbench.files_written)
    for path, content in generated.items():
        if _write_new(path, content, force=force, writer=writer, logical_root=destination):
            written.append(path)
    if writer is None or Path(writer.path).expanduser().resolve() == destination:
        inspection_manifest = manifest_path
        inspection_root = root
    else:
        writer_root = Path(writer.path).expanduser().resolve()
        logical_root = Path(getattr(writer, "logical_root", destination)).expanduser().resolve()
        inspection_manifest = writer_root / manifest_path.relative_to(logical_root)
        inspection_root = writer_root
    candidate = inspect_candidate(inspection_manifest, workspace_root=inspection_root)
    return CandidateScaffoldResult(destination, manifest_path, tuple(sorted(written)), candidate)
