"""Public, dependency-clean ADCS and RPO candidate scaffolds."""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml

from .candidate import ROOT, inspect_candidate
from .contracts import CANDIDATE_SCHEMA_ID

_IDENT_RE = re.compile(r"[^0-9A-Za-z_]+")


@dataclass(frozen=True, slots=True)
class CandidateScaffoldResult:
    root_dir: Path
    manifest_path: Path
    files_written: tuple[Path, ...]
    candidate: dict[str, Any]


def _slug(value: str) -> str:
    raw = re.sub(r"_+", "_", _IDENT_RE.sub("_", str(value).strip().lower()).strip("_"))
    raw = raw or "custom_fsw"
    if raw[0].isdigit():
        raw = f"stack_{raw}"
    return f"{raw}_stack" if keyword.iskeyword(raw) or len(raw) < 2 else raw


def _class_name(value: str) -> str:
    name = "".join(part[:1].upper() + part[1:] for part in _slug(value).split("_") if part)
    return name if name.endswith("FlightSoftwareStack") else f"{name}FlightSoftwareStack"


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
            raise FileExistsError(f"Refusing to overwrite existing FSW authoring file: {path}")
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


def _module_root(destination: Path, workspace_root: Path) -> tuple[str, ...]:
    relative = destination.relative_to(workspace_root)
    invalid = [part for part in relative.parts if not part.isidentifier() or keyword.iskeyword(part)]
    if invalid:
        raise ValueError(
            "Candidate output path must use importable Python identifiers; invalid component(s): "
            + ", ".join(invalid)
        )
    return relative.parts


def scaffold_candidate(
    name: str,
    *,
    template: str = "adcs",
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    class_name: str | None = None,
    force: bool = False,
    writer: Any | None = None,
) -> CandidateScaffoldResult:
    root = Path(workspace_root).expanduser().resolve()
    candidate_id = _slug(name)
    destination = Path(output_dir) if output_dir is not None else root / "fsw_candidates" / candidate_id
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Candidate output must remain inside the authorized workspace.") from exc
    if writer is not None and Path(
        getattr(writer, "logical_root", writer.path)
    ).expanduser().resolve() != destination:
        raise PermissionError("FSW scaffold writer must be bound to the logical candidate output directory.")
    if template not in {"adcs", "rpo"}:
        raise ValueError("Public FSW template must be one of: adcs, rpo")
    module_parts = _module_root(destination, root)
    stack_class = _class_name(class_name or name)
    stack_filename = f"{candidate_id}_stack.py"
    module_path = ".".join((*module_parts, "stacks", stack_filename.removesuffix(".py")))
    stack_path = destination / "stacks" / stack_filename
    scenario_path = destination / "configs" / f"{candidate_id}_smoke.yaml"
    test_path = destination / "tests" / f"test_{candidate_id}_stack.py"
    manifest_path = destination / "candidate.yaml"
    readme_path = destination / "README.md"
    scenario = _scenario(
        candidate_id=candidate_id,
        module_path=module_path,
        class_name=stack_class,
        template=template,
    )
    manifest = {
        "schema_version": CANDIDATE_SCHEMA_ID,
        "candidate_id": candidate_id,
        "revision": "0.1.0",
        "kind": "python_stack",
        "source": {
            "root": ".",
            "revision_id": "uncommitted-public-scaffold",
            "entrypoint": {"module": module_path, "class_name": stack_class},
        },
        "interfaces": {
            "onboard_contract": "oel.fsw.boundary.v1",
            "hardware_profile": "hardware.ideal_wrench.v1",
            "task_period_s": 0.2 if template == "adcs" else 0.5,
        },
        "claims": {
            "intended_use": "attitude_control" if template == "adcs" else "rendezvous_and_proximity_operations"
        },
        "verification": {
            "component_suite": str(test_path.parent.relative_to(destination)),
            "smoke_case": str(scenario_path.relative_to(destination)),
        },
        "handling": {"classification": "public", "hosted_ai_allowed": True, "owner": ""},
    }
    files = {
        destination / "__init__.py": "",
        destination / "stacks" / "__init__.py": "",
        destination / "tests" / "__init__.py": "",
        destination / "configs" / "__init__.py": "",
        stack_path: _stack_template(stack_class, template),
        scenario_path: yaml.safe_dump(scenario, sort_keys=False),
        test_path: _test_template(module_path, stack_class, template),
        manifest_path: yaml.safe_dump(manifest, sort_keys=False),
        readme_path: _readme(name, manifest_path.relative_to(root), template),
    }
    for path in files:
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("Candidate output must remain inside the authorized workspace.") from exc
    written: list[Path] = []
    for path, content in files.items():
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


def _stack_template(class_name: str, template: str) -> str:
    if template == "adcs":
        return dedent(
            f'''\
            """Editable public GNC v2 attitude flight-software stack."""

            from sim.flight_software import AttitudeReferenceFlightSoftwareStack, AttitudeReferenceStackConfig, FrameId
            from sim.gnc.attitude_v2 import (
                AttitudeAllocatorConfig,
                AttitudeAllocatorKind,
                AttitudeReferenceConfig,
                AttitudeReferenceMode,
                QuaternionTorqueController,
            )


            class {class_name}(AttitudeReferenceFlightSoftwareStack):
                """Customize public navigation, guidance, control, or allocation here."""

                def __init__(self, satellite_id: str = "target", max_torque_n_m: float = 0.05) -> None:
                    body = FrameId(f"OEL/BODY/{{satellite_id}}", "frames-v1")
                    inertial = FrameId("OEL/ECI/J2000", "frames-v1")
                    actuator = FrameId(f"OEL/ACTUATOR/{{satellite_id}}/wrench", "frames-v1")
                    super().__init__(AttitudeReferenceStackConfig(
                        satellite_id,
                        body,
                        inertial,
                        AttitudeAllocatorConfig(
                            satellite_id,
                            AttitudeAllocatorKind.IDEAL_WRENCH,
                            "wrench",
                            actuator,
                            limits=(max_torque_n_m, max_torque_n_m, max_torque_n_m),
                        ),
                        reference=AttitudeReferenceConfig(
                            AttitudeReferenceMode.QUATERNION,
                            validity_ticks=1_000_000_000,
                        ),
                        controller=QuaternionTorqueController(max_torque_n_m=max_torque_n_m),
                    ))
            '''
        )
    return dedent(
        f'''\
        """Editable public GNC v2 RPO flight-software stack."""

        from sim.flight_software import (
            FrameId,
            GoalDefinition,
            GoalMode,
            RpoReferenceFlightSoftwareStack,
            RpoReferenceStackConfig,
        )
        from sim.gnc.executive_v2 import ReferenceExecutiveConfig
        from sim.gnc.navigation_v2 import NavigationInitializationMode
        from sim.gnc.orbit_v2 import (
            TranslationAllocatorConfig,
            TranslationAllocatorKind,
            TranslationControlConfig,
            TranslationMode,
        )


        class {class_name}(RpoReferenceFlightSoftwareStack):
            """Customize public navigation, guidance, control, or allocation here."""

            def __init__(
                self,
                satellite_id: str = "chaser",
                reference_object_id: str = "target",
                assumed_mass_kg: float = 200.0,
                max_acceleration_m_s2: float = 0.05,
            ) -> None:
                body = FrameId(f"OEL/BODY/{{satellite_id}}", "frames-v1")
                inertial = FrameId("OEL/ECI/J2000", "frames-v1")
                relative = FrameId(f"OEL/RIC/{{reference_object_id}}", "frames-v1")
                actuator = FrameId(f"OEL/ACTUATOR/{{satellite_id}}/wrench", "frames-v1")
                control = TranslationControlConfig(
                    TranslationMode.RIC_HOLD,
                    assumed_mass_kg,
                    max_acceleration_m_s2,
                    target_id=reference_object_id,
                    target_relative_state_ric=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    validity_ticks=1_000_000_000,
                )
                allocator = TranslationAllocatorConfig(
                    satellite_id,
                    TranslationAllocatorKind.IDEAL_WRENCH,
                    "wrench",
                    actuator,
                    assumed_mass_kg * max_acceleration_m_s2,
                )
                goal = GoalDefinition("hold", "ric_hold", GoalMode.MAINTENANCE, target_frame=relative)
                executive = ReferenceExecutiveConfig(goal, TranslationMode.RIC_HOLD.value)
                super().__init__(RpoReferenceStackConfig(
                    satellite_id,
                    body,
                    inertial,
                    relative,
                    NavigationInitializationMode.IDEAL,
                    control,
                    allocator,
                    executive,
                ))
        '''
    )


def _scenario(
    *,
    candidate_id: str,
    module_path: str,
    class_name: str,
    template: str,
) -> dict[str, Any]:
    controlled_id = "target" if template == "adcs" else "chaser"
    controlled: dict[str, Any] = {
        "enabled": True,
        "role": controlled_id,
        "kind": "satellite",
        "specs": {
            "dry_mass_kg": 180.0,
            "fuel_mass_kg": 0.0 if template == "adcs" else 20.0,
            "flight_software_hardware": {
                "actuator_id": "wrench",
                "max_force_n": 0.0 if template == "adcs" else 10.0,
                "max_torque_n_m": 0.05,
            },
        },
        "initial_state": {
            "coes": {
                "a_km": 7000.0,
                "ecc": 0.0,
                "inc_deg": 45.0,
                "raan_deg": 0.0,
                "argp_deg": 0.0,
                "true_anomaly_deg": 0.0,
            },
            "attitude_quat_bn": [0.9961947, 0.0871557, 0.0, 0.0] if template == "adcs" else [1.0, 0.0, 0.0, 0.0],
            "angular_rate_body_rad_s": [0.0, 0.01, 0.0] if template == "adcs" else [0.0, 0.0, 0.0],
        },
        "flight_software": {
            "module": module_path,
            "class_name": class_name,
            "hardware_profile": "hardware.ideal_wrench.v1",
            "task_period_s": 0.2 if template == "adcs" else 0.5,
            "params": {"satellite_id": controlled_id},
        },
    }
    objects = {controlled_id: controlled}
    if template == "rpo":
        controlled["initial_state"] = {
            "relative_to": "target",
            "relative_to_target_ric": {"frame": "curv", "state": [0.0, -2.0, 0.0, 0.0, 0.0, 0.0]},
        }
        controlled["flight_software"]["params"]["reference_object_id"] = "target"
        objects["target"] = {
            "enabled": True,
            "role": "target",
            "kind": "satellite",
            "specs": {"dry_mass_kg": 360.0, "fuel_mass_kg": 0.0},
            "initial_state": {
                "coes": {
                    "a_km": 7000.0,
                    "ecc": 0.0,
                    "inc_deg": 45.0,
                    "raan_deg": 0.0,
                    "argp_deg": 0.0,
                    "true_anomaly_deg": 0.0,
                }
            },
            "flight_software": {
                "stack": "fsw.passive",
                "hardware_profile": "hardware.passive.v1",
                "task_period_s": 1.0,
            },
        }
    return {
        "schema_version": "oel.scenario.v1",
        "scenario_name": f"{candidate_id}_public_smoke",
        "scenario_description": "Public FSW authoring deterministic smoke; not qualification evidence.",
        "metadata": {"owner": "public_fsw_author", "evidence_role": "candidate_smoke_only"},
        "objects": objects,
        "simulator": {
            "duration_s": 20.0,
            "dt_s": 1.0,
            "dynamics": {
                "orbit": {"model": "two_body", "orbit_substep_s": 1.0},
                "attitude": {"enabled": template == "adcs", "attitude_substep_s": 0.1},
            },
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": "outputs/source_smoke",
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True},
            "review": {"enabled": True, "detail": "standard"},
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }


def _test_template(module_path: str, class_name: str, template: str) -> str:
    expected = "fsw.attitude_reference" if template == "adcs" else "fsw.rpo_reference"
    return dedent(
        f'''\
        from {module_path} import {class_name}


        def test_generated_complete_stack_implements_public_lifecycle() -> None:
            stack = {class_name}()
            assert stack.identity.stack_id == "{expected}"
            for method in ("boot", "step", "shutdown", "snapshot", "restore"):
                assert callable(getattr(stack, method))
        '''
    )


def _readme(name: str, manifest: Path, template: str) -> str:
    return dedent(
        f'''\
        # {name} public FSW candidate

        This editable `{template}` complete-stack candidate uses OEL's public typed
        flight-software boundary. The included checks and smoke scenario are not
        Controller Bench, qualification, certification, or operational evidence.

        ```bash
        python -m sim.fsw_authoring inspect {manifest}
        python -m sim.fsw_authoring validate {manifest} --trusted-import
        python -m sim.fsw_authoring test {manifest}
        python -m sim.fsw_authoring smoke {manifest}
        ```
        '''
    )


__all__ = ["CandidateScaffoldResult", "scaffold_candidate"]
