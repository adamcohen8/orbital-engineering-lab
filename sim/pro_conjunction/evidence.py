"""Atomic, content-bound evidence, source identity and authoritative replay."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from sim.pro_conjunction.contracts import digest
from sim.pro_features import FEATURE_SCALE_ANALYTICS, require_pro_feature
from sim.security.output_writes import staged_output
from sim.utils.io import read_regular_file_nofollow

MAX_BYTES = 32 * 1024 * 1024


def read_json(path):
    def unique(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON field: {key}")
            value[key] = child
        return value

    return json.loads(
        read_regular_file_nofollow(Path(path), max_bytes=MAX_BYTES).decode(),
        object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")),
    )


def runtime_identity():
    root = Path(__file__).resolve().parents[2]
    files = [
        *sorted((root / "sim/pro_conjunction").glob("*.py")),
        *sorted((root / "sim/dynamics/orbit").glob("*.py")),
        root / "sim/estimation/augmented_dynamics.py",
        root / "sim/estimation/covariance.py",
        root / "sim/analysis/conjunction_geometry.py",
        root / "sim/analysis/conjunction_probability.py",
    ]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numeric_backend": "python",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }


def execute(operation, inputs):
    from sim.pro_conjunction.assessment import assess, sensitivity
    from sim.pro_conjunction.trades import optimize

    if operation == "assess":
        return assess(inputs["problem"])
    if operation == "sensitivity":
        return sensitivity(inputs["problem"], inputs["scales"], inputs["execution_scales"])
    if operation == "optimize":
        return optimize(inputs["problem"], inputs["trade"])
    raise ValueError("unsupported evidence operation")


def render_report(result):
    lines = [
        "# Conjunction analysis evidence",
        "",
        f"Status: {result.get('status', 'completed')}",
        "",
        "Conditional engineering analysis. Input covariance realism is not empirically qualified.",
        "",
        "| Case | Secondary | TCA (s) | Miss (m) | 2D Pc | Nonlinear confidence interval |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    def assessment_rows(packet, prefix=""):
        for label in ("baseline", "candidate"):
            assessment = packet.get(label)
            if assessment is None:
                continue
            for row in assessment["encounters"]:
                ca, pc, mc = row["closest_approach"], row["probability_2d"], row["nonlinear_probability"]
                value = f"{pc['collision_probability']:.8g}" if pc else "withheld"
                interval = str(mc["confidence_interval"]) if mc else "not requested"
                secondary = row["secondary_id"].replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {prefix}{label} | {secondary} | {ca['time_s']:.8g} | {ca['miss_distance_km'] * 1000:.8g} | {value} | {interval} |"
                )

    if "baseline" in result:
        assessment_rows(result)
    for row in result.get("rows", []):
        assessment_rows(
            row["assessment"],
            f"cov={row['covariance_multiplier']}, execution={row['execution_covariance_multiplier']} ",
        )
    for i, row in enumerate(result.get("verification", [])):
        assessment_rows(row["assessment"], f"verification {i} ")
    lines += [
        "",
        "The retained JSON includes full propagated covariances, usage diagnostics, numerical and sampling",
        "uncertainty, constraints, failed attempts, and all declared secondary rescreens.",
        "",
        "No operational safety, calibrated collision-risk, global-optimum or maneuver-authorization claim.",
        "",
    ]
    return "\n".join(lines)


def write_bundle(destination, operation, inputs, result):
    require_pro_feature(FEATURE_SCALE_ANALYTICS)
    destination = Path(destination).absolute()
    payloads = {"inputs.json": inputs, "result.json": result}
    report = render_report(result)
    manifest = {
        "schema_version": "oel.pro_conjunction_bundle.v1",
        "operation": operation,
        "runtime": runtime_identity(),
        "artifacts": {name: digest(value) for name, value in payloads.items()},
    }
    manifest["artifacts"]["REPORT.md"] = hashlib.sha256(report.encode()).hexdigest()
    manifest["manifest_sha256"] = digest(manifest)
    ancestor = destination.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    with staged_output(destination, approved_roots=[ancestor]) as writer:
        for name, value in {**payloads, "manifest.json": manifest}.items():
            writer.staging.write_text(name, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        writer.staging.write_text("REPORT.md", report)
    return manifest


def inspect_bundle(directory):
    require_pro_feature(FEATURE_SCALE_ANALYTICS)
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    expected = manifest.pop("manifest_sha256")
    if manifest["schema_version"] != "oel.pro_conjunction_bundle.v1" or digest(manifest) != expected:
        raise ValueError("evidence manifest identity mismatch")
    if set(manifest["artifacts"]) != {"inputs.json", "result.json", "REPORT.md"}:
        raise ValueError("invalid evidence inventory")
    payloads = {name: read_json(directory / name) for name in ("inputs.json", "result.json")}
    if any(digest(value) != manifest["artifacts"][name] for name, value in payloads.items()):
        raise ValueError("evidence artifact identity mismatch")
    report = read_regular_file_nofollow(directory / "REPORT.md", max_bytes=MAX_BYTES)
    if hashlib.sha256(report).hexdigest() != manifest["artifacts"]["REPORT.md"]:
        raise ValueError("evidence report identity mismatch")
    return {"status": "identity_verified", "manifest": {**manifest, "manifest_sha256": expected}, **payloads}


def replay_bundle(directory):
    verified = inspect_bundle(directory)
    if verified["manifest"]["runtime"] != runtime_identity():
        raise ValueError("source/dependency identity drift; use an explicitly new study")
    result = execute(verified["manifest"]["operation"], verified["inputs.json"])
    if digest(result) != digest(verified["result.json"]):
        raise ValueError("authoritative replay mismatch")
    return {"status": "authoritative_replay_verified", "result_sha256": digest(result)}
