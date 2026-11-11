"""Entitled local spacecraft twin lifecycle. No remote extraction or simulation runs."""

from __future__ import annotations

import functools
import importlib.util
import shutil
from pathlib import Path

import yaml

from sim.pro_features import FEATURE_CUSTOMER_PACKAGES, require_pro_feature
from sim.security.output_writes import copy_output_file, current_output_writer, mkdir_output, write_output_text

from .compiler import ADAPTERS, assess
from .contracts import SCHEMA, contained, digest, file_digest, load, physical_output_path, write_json
from .viewer import write_viewer


def compiler_identity():
    # Resolve by fully qualified module name without importing each owner.
    # Several package ``__init__`` files intentionally export convenience
    # functions whose names shadow implementation modules (notably
    # ``spacecraft_twin.geometry``).  ``importlib.import_module`` would
    # overwrite that public callable with the submodule on first use, while
    # ``from . import geometry`` can hash the function instead of its source.
    module_names = (
        "sim.spacecraft_twin.compiler",
        "sim.spacecraft_twin.contracts",
        "sim.spacecraft_twin.geometry",
        "sim.spacecraft_twin.qualification",
        "sim.spacecraft_twin.runtime_binding",
        "sim.spacecraft_twin.schema",
        "sim.digital_twin.mass_properties",
        "sim.actuators.physical",
        "sim.dynamics.spacecraft_geometry",
        "sim.rust_spacecraft_backend",
        "sim.runtime.satellites.factory",
        "sim.runtime.satellites.flight_software_runtime",
        "sim.flight_software.catalog",
    )
    origins = []
    for name in module_names:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
            raise RuntimeError(f"Cannot resolve a source origin for compiler owner {name!r}.")
        origins.append((name, spec.origin))

    return digest(
        {
            name: file_digest(origin)
            for name, origin in origins
        }
    )


def entitled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        require_pro_feature(FEATURE_CUSTOMER_PACKAGES)
        return fn(*args, **kwargs)

    return wrapper


@entitled
def capabilities():
    from .schema import SCHEMA as authoring_schema

    return {
        "schema": "oel.twin_capabilities.v1",
        "twin_schema": SCHEMA,
        "authoring_schema": authoring_schema,
        "entitlement": FEATURE_CUSTOMER_PACKAGES,
        "operations": [
            "intake",
            "propose",
            "inspect",
            "geometry",
            "accept",
            "verify",
            "diff",
            "materialize",
            "migrate-v0",
        ],
        "adapters": ADAPTERS,
        "purposes": ["passive", "attitude", "environment"],
        "external_communication": False,
        "agent_role": "Extract source-backed facts using host tools; submit data-only proposals. OEL owns compilation and validation.",
        "frame_contract": "Right-handed body frame. local vector -> parent: R @ vector; position -> parent: R @ (position * unit_scale) + translation. Inertia is about COM in body axes.",
        "evidence_contract": "evidence maps field paths to [{source_id, locator, basis: extracted|derived|assumed, explanation}]. Components use IDs in paths; cite each parameter. Derived and assumed facts require explanations. Assumptions require explicit accepted=true and reason.",
        "limits": [
            "No arbitrary native CAD import; provide STL plus documented assembly transforms.",
            "Images and scanned PDFs require host visual extraction.",
            "Only advertised component adapters have runtime placement semantics.",
            "Fixed geometry configurations; no continuous articulation.",
            "Qualification is bounded numerical evidence, not flight qualification.",
        ],
    }


@entitled
def intake(source_dir, output_dir, **kwargs):
    from .intake import inventory

    return inventory(source_dir, output_dir, **kwargs)


def _copy_references(model, root, output):
    refs = [row["path"] for row in model.get("sources", [])]
    refs += [row["extraction"] for row in model.get("sources", []) if row.get("extraction")]
    geom = model.get("geometry", {})
    refs += [geom[k] for k in ("profile_path", "qualification_path") if geom.get(k)]
    for relative in set(refs):
        src, dst = contained(root, relative), contained(output, relative)
        if dst.name in {"twin.json", "assessment.json", "object.yaml", "review.html", "receipt.json"}:
            raise ValueError("Source path collides with a generated artifact name.")
        mkdir_output(dst.parent)
        copy_output_file(src, dst)


def _discard_output(path):
    """Discard one failed output without deleting outside the bound writer."""

    physical = physical_output_path(path)
    writer = current_output_writer()
    if writer is not None and physical == Path(writer.path):
        writer.remove_tree()
        return
    shutil.rmtree(physical)


def _artifact_digests(root):
    """Hash generated artifacts by logical path before any result projection."""

    physical_root = physical_output_path(root)
    return {
        path.relative_to(physical_root).as_posix(): file_digest(root / path.relative_to(physical_root))
        for path in physical_root.rglob("*")
        if path.is_file()
    }


@entitled
def propose(model_path, output_dir, *, purpose="attitude"):
    path, out = Path(model_path).resolve(), Path(output_dir).resolve()
    model = load(path)
    if (current_output_writer() is None and out.exists()) or out.is_relative_to(path.parent):
        raise ValueError("Proposal output must be a new directory outside the input package.")
    mkdir_output(out)
    try:
        _copy_references(model, path.parent, out)
        write_json(out / "twin.json", model)
        report = assess(model, out, purpose)
        write_json(out / "assessment.json", report)
        write_viewer(out / "review.html", model, report)
        return {
            **report,
            "manifest_path": str(out / "twin.json"),
            "review_path": str(out / "review.html"),
            "proposal_digest": digest(model),
        }
    except BaseException:
        _discard_output(out)
        raise


@entitled
def inspect(model_path, *, purpose="attitude"):
    path = Path(model_path).resolve()
    model = load(path)
    return {**assess(model, path.parent, purpose), "proposal_digest": digest(model), "manifest_path": str(path)}


@entitled
def geometry(mesh_path, output_dir, **kwargs):
    if "scale_to_m" not in kwargs or not str(kwargs.get("configuration", "")).strip():
        raise ValueError("Geometry requires an explicit scale_to_m and named configuration.")
    from .geometry import build_profile

    return build_profile(mesh_path, output_dir, **kwargs)


@entitled
def accept(model_path, output_dir, *, expected_digest, reviewer, purpose="attitude"):
    path, out = Path(model_path).resolve(), Path(output_dir).resolve()
    model = load(path)
    if digest(model) != expected_digest:
        raise ValueError("Proposal changed after review.")
    if not reviewer.strip():
        raise ValueError("Reviewer identity/reference is required.")
    report = assess(model, path.parent, purpose)
    if not report["ready"]:
        raise ValueError(f"Twin is not ready: {report['errors'] + report['missing']}")
    if (current_output_writer() is None and out.exists()) or out.is_relative_to(path.parent):
        raise ValueError("Accepted version requires a new directory outside the candidate.")
    mkdir_output(out)
    try:
        _copy_references(model, path.parent, out)
        write_json(out / "twin.json", model)
        report = assess(model, out, purpose)
        if not report["ready"]:
            raise ValueError("Copied evidence identity changed during acceptance.")
        write_json(out / "assessment.json", report)
        write_json(out / "hardware-parameters.json", report["compiled"]["hardware_parameters"])
        write_output_text(
            out / "object.yaml",
            yaml.safe_dump({"objects": {model["id"]: report["compiled"]["object"]}}, sort_keys=False),
        )
        write_viewer(out / "review.html", model, report)
        artifacts = _artifact_digests(out)
        receipt = {
            "schema": "oel.twin_acceptance.v1",
            "twin_id": model["id"],
            "version": model["version"],
            "twin_digest": expected_digest,
            "compiler_digest": compiler_identity(),
            "purpose": purpose,
            "reviewer": reviewer,
            "artifacts": artifacts,
            "claim": report["claim"],
        }
        write_json(out / "receipt.json", receipt)
        return {
            "status": "accepted",
            "receipt_path": str(out / "receipt.json"),
            "receipt_sha256": file_digest(out / "receipt.json"),
            "twin_digest": expected_digest,
        }
    except BaseException:
        _discard_output(out)
        raise


@entitled
def verify(receipt_path, *, expected_receipt_sha256=None):
    path = Path(receipt_path).resolve()
    if expected_receipt_sha256 and file_digest(path) != expected_receipt_sha256:
        raise ValueError("Acceptance receipt identity changed.")
    receipt = load(path)
    if receipt.get("schema") != "oel.twin_acceptance.v1":
        raise ValueError("Not a twin acceptance receipt.")
    if receipt.get("compiler_digest") != compiler_identity():
        raise ValueError("Twin compiler/runtime changed; recompile, review and accept a new version.")
    if not {"twin.json", "object.yaml", "assessment.json", "review.html"}.issubset(receipt.get("artifacts", {})):
        raise ValueError("Acceptance receipt omits required artifacts.")
    for relative, sha in receipt["artifacts"].items():
        if file_digest(contained(path.parent, relative)) != sha:
            raise ValueError(f"Accepted artifact changed: {relative}")
    model = load(path.parent / "twin.json")
    if digest(model) != receipt["twin_digest"]:
        raise ValueError("Accepted twin changed.")
    report = assess(model, path.parent, receipt["purpose"])
    if not report["ready"]:
        raise ValueError("Accepted twin is no longer ready.")
    return {
        "status": "verified",
        "receipt_sha256": file_digest(path),
        "twin_digest": digest(model),
        "purpose": receipt["purpose"],
        "compiler_digest": receipt["compiler_digest"],
    }


@entitled
def diff(left_path, right_path):
    left, right = load(left_path), load(right_path)
    changes = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), f"{path}/{key}")
        elif a != b:
            changes.append({"field": path, "before": a, "after": b})

    walk(left, right, "")
    return {
        "left_digest": digest(left),
        "right_digest": digest(right),
        "changes": changes,
        "requires_recompile": bool(changes),
        "requires_review": bool(changes),
    }


@entitled
def materialize(receipt_path, template_path, output_dir, *, expected_receipt_sha256):
    receipt = Path(receipt_path).resolve()
    identity = verify(receipt, expected_receipt_sha256=expected_receipt_sha256)
    model = load(receipt.parent / "twin.json")
    report = assess(model, receipt.parent, identity["purpose"])
    template = load(template_path)
    dynamics = template.get("simulator", {}).get("dynamics", {})
    if dynamics.get("orbit", {}).get("drag") or dynamics.get("orbit", {}).get("srp"):
        environment_report = assess(model, receipt.parent, "environment")
        if not environment_report["ready"]:
            raise ValueError("Requested drag/SRP scenario requires a twin qualified for environment studies.")
    template.setdefault("metadata", {})["spacecraft_twin"] = {
        "id": model["id"],
        "version": model["version"],
        "digest": identity["twin_digest"],
        "acceptance_receipt_sha256": identity["receipt_sha256"],
    }
    object_id = model["id"]
    existing = template.get("objects", {}).get(object_id)
    if not isinstance(existing, dict):
        raise ValueError("Scenario template must explicitly contain the twin object ID and initial state.")
    if "initial_state" not in existing:
        raise ValueError("Scenario template must provide an initial state.")
    if existing.get("specs"):
        raise ValueError("Remove template specs: twin compilation owns spacecraft specifications.")
    from .runtime_binding import bind, bind_sensor

    flight_software, hardware_binding = bind(existing, report["compiled"])
    knowledge, sensor_active = bind_sensor(existing, report["compiled"])
    hardware_binding["tracking_sensor_active"] = sensor_active
    if not sensor_active:
        for placement in report["compiled"]["placements"]:
            if (
                placement["kind"] == "tracking_sensor"
                and placement["id"] not in hardware_binding["inactive_components"]
            ):
                hardware_binding["inactive_components"].append(placement["id"])
    template["objects"][object_id] = {
        **existing,
        "kind": "satellite",
        "specs": report["compiled"]["object"]["specs"],
        "flight_software": flight_software,
        "knowledge": knowledge,
    }
    out = Path(output_dir).resolve()
    if current_output_writer() is None and out.exists():
        raise ValueError("Materialization requires a new output directory.")
    mkdir_output(out)
    scenario_path = out / "scenario.yaml"
    write_output_text(scenario_path, yaml.safe_dump(template, sort_keys=False))
    from sim.api import SimulationWorkspace

    physical_out = physical_output_path(out)
    workspace = SimulationWorkspace(
        workspace_root=physical_out,
        read_roots=(physical_out, receipt.parent),
        write_roots=(physical_out,),
    )
    try:
        validation = workspace.validate(workspace.load(physical_output_path(scenario_path)), import_plugins=False)
    except Exception as exc:
        validation = {"ok": False, "errors": [str(exc)]}
    provenance = {
        **identity,
        "schema": "oel.twin_scenario_binding.v1",
        "receipt_path": str(receipt),
        "template_sha256": file_digest(template_path),
        "scenario_sha256": file_digest(scenario_path),
        "scenario_path": str(scenario_path),
        "status": "requires_trusted_validation" if validation.get("ok") else "failed",
        "safe_validation": validation,
        "hardware_binding": hardware_binding,
        "next_step": "Run safe validation, then trusted validate-only, before simulation. Retain this binding with run evidence.",
    }
    write_json(out / "twin-binding.json", provenance)
    return provenance


@entitled
def migrate_v0(manifest_path, output_dir):
    from sim.digital_twin.package import SpacecraftTwinPackage

    path = Path(manifest_path).resolve()
    # v0 permits references outside its root; the migration surface does not.
    package = SpacecraftTwinPackage.load(path)
    for candidate in package._referenced_paths().values():
        if candidate is not None and not candidate.resolve().is_relative_to(path.parent):
            raise ValueError("v0 migration requires a self-contained source package.")
    specs = package.assembled_object()["specs"]
    mp = specs.get("mass_properties", {})
    model = {
        "schema": SCHEMA,
        "id": package.object_id,
        "version": package.version,
        "mass_kg": specs.get("mass_kg"),
        "center_of_mass_body_m": mp.get("center_of_mass_body_m"),
        "inertia_kg_m2": mp.get("inertia_kg_m2"),
        "frames": {},
        "components": [],
        "sources": [],
        "evidence": {},
        "assumptions": [],
        "conflicts": [{"field": "components", "reason": "", "resolution": ""}],
        "metadata": {
            "legacy_manifest_sha256": file_digest(path),
            "legacy_specs": specs,
            "migration_note": "Review legacy component conventions and add evidence before acceptance.",
        },
    }
    out = Path(output_dir)
    if current_output_writer() is None and out.exists():
        raise ValueError("Migration requires a new directory.")
    mkdir_output(out)
    write_json(out / "twin.json", model)
    return {"manifest_path": str(out / "twin.json"), "status": "review_required"}
