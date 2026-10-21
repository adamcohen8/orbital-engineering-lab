"""Cross-owner object IO regressions and real-engine continuation checks."""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from sim.data_io.api import convert, export_review, read_data
from sim.data_io.contracts import AttitudeHistory, DataIOError, Trajectory
from sim.data_io.materialize import materialize_scenario
from sim.data_io.objects import (
    ImportedObject,
    compose_object,
    create_scenario,
    describe_export,
    describe_import,
    export_object,
    import_object,
    materialize_continuation,
)
from sim.data_io.products import trajectory_from_state_product
from sim.data_io.time import parse_time
from sim.interchange.provenance import compute_product_id
from sim.tests.data_io_helpers import make_review

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "sim/interchange/examples"


def state_product(tmp_path, fault=None):
    original = EXAMPLES / "state_estimate_accepted_current.json"
    value = json.loads(original.read_text())
    for row in value["provenance"]["source_artifacts"]:
        row["path"] = str((original.parent / row["path"]).resolve())
    if fault == "rejected":
        value["quality"]["disposition"] = "rejected"
    if fault == "stale":
        value["freshness"]["integrity_status"] = "stale"
    if fault == "model":
        value["payload"]["model_assumptions"]["attitude"]["source"] = "analytic_ideal"
    value["product_id"] = compute_product_id(value)
    path = tmp_path / "product.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("fault", ["rejected", "stale", "model"])
def test_promotion_gates_survive_normalization_and_block_both_materializers(tmp_path, fault):
    source = state_product(tmp_path, fault)
    # Inspection and conversion remain read-only and are allowed for valid,
    # non-promotable products; neither action upgrades their disposition.
    trajectory = trajectory_from_state_product(source)
    converted = tmp_path / "converted.json"
    convert(source, converted, input_format="oel-state-product", output_format="trajectory-json")
    for value in (trajectory, read_data(converted, format="trajectory-json")):
        with pytest.raises(DataIOError, match="promotion|compatibility"):
            materialize_scenario(
                value,
                tmp_path / "blocked.yaml",
                scenario_name="blocked",
                output_dir=tmp_path / "run",
                duration_s=60,
                step_s=10,
            )
    imported = import_object(source, format="oel-state-product")
    with pytest.raises(DataIOError, match="promotion|compatibility"):
        create_scenario(
            imported,
            tmp_path / "shared.yaml",
            scenario_name="blocked",
            output_dir=tmp_path / "run",
            duration_s=60,
            step_s=10,
        )
    assert describe_import(source, format="oel-state-product")["status"] == "blocked"
    assert not (tmp_path / "shared.yaml").exists() and not (tmp_path / "blocked.yaml").exists()


def test_hash_bound_external_format_keeps_nonpromotable_product_context(tmp_path):
    source = state_product(tmp_path, "rejected")
    output = tmp_path / "rejected.opm"
    convert(source, output, input_format="oel-state-product", output_format="opm-kvn", options={"sample_index": 0})
    imported = import_object(output, format="opm-kvn")
    with pytest.raises(DataIOError, match="quality.rejected"):
        create_scenario(
            imported,
            tmp_path / "blocked.yaml",
            scenario_name="blocked",
            output_dir=tmp_path / "run",
            duration_s=60,
            step_s=10,
        )
    from sim.data_io.cli import main

    assert (
        main(
            [
                "materialize",
                str(output),
                str(tmp_path / "legacy-blocked.yaml"),
                "--format",
                "opm-kvn",
                "--scenario-name",
                "blocked",
                "--run-output",
                str(tmp_path / "run"),
                "--duration-s",
                "60",
                "--step-s",
                "10",
            ]
        )
        == 2
    )
    assert not (tmp_path / "legacy-blocked.yaml").exists()


def test_state_product_preserves_specs_and_force_assumptions(tmp_path):
    source = state_product(tmp_path)
    value = json.loads(source.read_text())
    value["payload"]["model_assumptions"]["orbit_force_model"]["j2"] = True
    value["product_id"] = compute_product_id(value)
    source.write_text(json.dumps(value))
    imported = import_object(source, format="oel-state-product")
    result = create_scenario(
        imported,
        tmp_path / "scenario.yaml",
        scenario_name="accepted",
        output_dir=tmp_path / "run",
        duration_s=60,
        step_s=10,
    )
    doc = yaml.safe_load((tmp_path / "scenario.yaml").read_text())
    assert doc["objects"]["example_satellite"]["specs"]["mass_kg"] == 500
    assert doc["simulator"]["dynamics"]["orbit"]["j2"] is True
    assert result["validation"]["ok"]
    assert not (tmp_path / "run").exists()


def test_opm_properties_and_current_review_mass_survive_export_reimport(tmp_path):
    run = make_review(tmp_path / "run")
    with sqlite3.connect(run / "review/run.sqlite") as conn:
        conn.execute("UPDATE object_state SET mass_kg=1000-time_s")
    export_review(run, tmp_path / "state.opm", format="opm-kvn", options={"sample_index": 1})
    native = read_data(tmp_path / "state.opm", format="opm-kvn")
    assert float(native.physical["MASS"]) == 970
    imported = import_object(tmp_path / "state.opm", format="opm-kvn")
    create_scenario(
        imported,
        tmp_path / "next.yaml",
        scenario_name="next",
        output_dir=tmp_path / "next-run",
        duration_s=60,
        step_s=10,
    )
    assert yaml.safe_load((tmp_path / "next.yaml").read_text())["objects"]["SAT"]["specs"]["mass_kg"] == 970
    original = import_object(EXAMPLES / "oel_earth_eme2000_utc_v3.opm", format="opm-kvn")
    entry = compose_object({"objects": {}, "simulator": {}}, original)["objects"]["2024-001A"]
    assert entry["specs"]["mass_kg"] == 1000 and entry["specs"]["drag_area_m2"] == 12 and entry["specs"]["cd"] == 2.2


@pytest.fixture
def orbital():
    return Trajectory("SAT", "EARTH", "EME2000", parse_time("2024-01-01T00:00:00"), [0], [[7000, 0, 0, 0, 7.5, 0]])


def test_composition_preserves_features_and_requires_explicit_replacement(orbital):
    original = {
        "objects": {
            "target": {
                "kind": "satellite",
                "role": "target",
                "specs": {"mass_kg": 250, "mass_properties": {"inertia_kg_m2": [[1, 0, 0], [0, 2, 0], [0, 0, 3]]}},
                "initial_state": {"coes": {"a_km": 7100}, "attitude_quat_bn": [1, 0, 0, 0]},
                "flight_software": {"stack": "fsw.passive"},
                "knowledge": {"targets": ["other"]},
            }
        },
        "simulator": {"initial_jd_utc": 2460310.5},
        "ground_stations": [{"id": "SITE"}],
    }
    before = deepcopy(original)
    imported = ImportedObject(orbital, "normalized")
    with pytest.raises(DataIOError, match="already exists"):
        compose_object(original, imported, object_id="target")
    result = compose_object(original, imported, object_id="target", replace=True)
    obj = result["objects"]["target"]
    assert obj["specs"]["mass_kg"] == 250 and "mass_properties" in obj["specs"]
    assert obj["flight_software"] == before["objects"]["target"]["flight_software"]
    assert obj["knowledge"] == before["objects"]["target"]["knowledge"]
    assert obj["initial_state"]["attitude_quat_bn"] == [1, 0, 0, 0] and "coes" not in obj["initial_state"]
    assert result["ground_stations"] == before["ground_stations"] and original == before


def test_builder_import_and_epoch_mismatch(orbital):
    from sim.scenarios import ScenarioBuilder

    imported = ImportedObject(orbital, "normalized")
    builder = ScenarioBuilder("rpo").duration(60, dt_s=10, initial_jd_utc=2460310.5)
    builder.import_object(imported, object_id="target").chaser_relative_ric([0.1, 0, 0, 0, 0, 0], mass_kg=100)
    assert set(builder.to_dict()["objects"]) == {"target", "chaser"}
    doc = builder.to_dict()
    doc["simulator"]["initial_jd_utc"] += 1
    with pytest.raises(DataIOError, match="epoch differs"):
        compose_object(doc, imported, object_id="third")


def test_attitude_attachment_is_separate_and_does_not_replace_orbit(orbital):
    history = AttitudeHistory("ATT", "EME2000", "SC_BODY_1", orbital.epoch, [0], [[1, 0, 0, 0]], [[0.01, 0.02, 0.03]])
    doc = compose_object({"objects": {}, "simulator": {}}, ImportedObject(orbital, "normalized"))
    result = compose_object(doc, ImportedObject(history, "normalized"), object_id="SAT")
    assert result["objects"]["SAT"]["initial_state"]["velocity_eci_km_s"] == [0, 7.5, 0]
    assert result["objects"]["SAT"]["initial_state"]["angular_rate_body_rad_s"] == [0.01, 0.02, 0.03]
    assert result["simulator"]["dynamics"]["attitude"]["enabled"] is True
    with pytest.raises(DataIOError, match="existing target"):
        compose_object({"objects": {}, "simulator": {}}, ImportedObject(history, "normalized"))


def test_object_source_hash_is_rechecked_before_composition(tmp_path, orbital):
    from sim.data_io.tabular import trajectory_document

    source = tmp_path / "source.json"
    source.write_text(json.dumps(trajectory_document(orbital)))
    imported = import_object(source, format="trajectory-json")
    source.write_text(source.read_text().replace("7000", "7001"))
    with pytest.raises(DataIOError, match="source/resource changed"):
        compose_object({"objects": {}, "simulator": {}}, imported)


def test_planetary_materializers_preserve_body_and_named_axes(tmp_path, orbital):
    mars = replace(orbital, object_id="probe", center="MARS", frame="SPICE:J2000", states=[[4000, 0, 0, 0, 3.27217, 0]])
    for name, call in [
        (
            "shared",
            lambda output: create_scenario(
                ImportedObject(mars, "normalized"),
                output,
                scenario_name="mars",
                output_dir=tmp_path / "run",
                duration_s=60,
                step_s=10,
            ),
        ),
        (
            "legacy",
            lambda output: materialize_scenario(
                mars, output, scenario_name="mars", output_dir=tmp_path / "run", duration_s=60, step_s=10
            ),
        ),
    ]:
        call(tmp_path / f"{name}.yaml")
        doc = yaml.safe_load((tmp_path / f"{name}.yaml").read_text())
        assert doc["simulator"]["central_body"] == "mars"
        assert doc["objects"]["probe"]["initial_state"]["position_inertial_km"] == [4000, 0, 0]
        assert "position_eci_km" not in doc["objects"]["probe"]["initial_state"]


def test_preflight_returns_requirements_and_corrective_errors(tmp_path):
    plan = describe_import(tmp_path / "unused", format="stk-ephemeris")
    assert plan["status"] == "input_required" and "object_id" in plan["missing_options"]
    with pytest.raises(DataIOError, match="--object-id"):
        read_data(tmp_path / "unused", format="stk-ephemeris")


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    from sim.execution import run_simulation_config_file
    from sim.security.config_paths import ConfigPathPolicy

    root = tmp_path_factory.mktemp("shared-object-real-run")
    imported = import_object(EXAMPLES / "oel_earth_eme2000_utc_v3.opm", format="opm-kvn")
    from sim.scenarios import ScenarioBuilder

    builder = ScenarioBuilder("object_io").duration(60, dt_s=10, initial_jd_utc=2460310.5)
    builder.import_object(imported, object_id="target").chaser_relative_ric([0.1, 0, 0, 0, 0, 0], mass_kg=100)
    builder.outputs(root / "run", stats={"save_full_log": False}).review()
    config = root / "scenario.yaml"
    builder.artifact().write(config)
    assert builder.artifact().validate()["ok"]
    run_simulation_config_file(config, path_policy=ConfigPathPolicy.default(config_path=config, write_roots=(root,)))
    return root


def test_shared_checkpoint_and_snapshot_routes_use_existing_owners(real_run, tmp_path):
    for fmt, kwargs in [
        ("satellite-checkpoint", {"object_id": "target"}),
        ("snapshot", {"object_ids": ["target", "chaser"], "final": True}),
    ]:
        product = tmp_path / f"{fmt}.json"
        export_object(real_run / "run", product, format=fmt, **kwargs)
        output = tmp_path / f"{fmt}.yaml"
        result = materialize_continuation(
            product,
            output,
            format=fmt,
            scenario_name="continued",
            output_dir=ROOT / "outputs" / f"object-io-{tmp_path.name}-{fmt}",
            duration_s=60,
            step_s=10,
        )
        assert result["status"] == "materialized"
        doc = yaml.safe_load(output.read_text())
        assert doc["objects"]["target"]["specs"]["mass_kg"] == 1000
        if fmt == "satellite-checkpoint":
            assert "checkpoint" in doc["objects"]["target"]["flight_software"]
        else:
            assert set(doc["objects"]) == {"target", "chaser"}


def test_shared_export_retains_selected_mass_and_preflight_lists_context(real_run, tmp_path):
    output = tmp_path / "target.opm"
    export_object(real_run / "run", output, format="opm-kvn", object_id="target", final=True)
    assert float(read_data(output, format="opm-kvn").physical["MASS"]) == 1000
    preflight = describe_export(real_run / "run")
    assert preflight["object_selection_required"]
    assert "satellite-checkpoint" in preflight["continuation_products"]
    assert {row["format"] for row in preflight["products"]} >= {"oem-kvn", "opm-kvn", "trajectory-json", "sp3", "czml"}


@pytest.fixture(scope="module")
def od_run(tmp_path_factory):
    from sim.tracking_od import TrackingOdProblem, assess_tdm_orbit_determination, read_tdm_kvn

    root = tmp_path_factory.mktemp("shared-object-od")
    problem = json.loads((ROOT / "examples/tracking_od/public_tdm_fit_holdout_problem.json").read_text())
    assess_tdm_orbit_determination(
        read_tdm_kvn(ROOT / "examples/tracking_od/public_reduced_geometric_azel_range.tdm"),
        TrackingOdProblem.from_mapping(problem),
        output_dir=root / "od",
    )
    return root / "od"


def test_completed_od_export_and_object_seed_never_rerun_estimation(od_run, tmp_path, monkeypatch):
    import sim.tracking_od

    def forbidden(*args, **kwargs):
        raise AssertionError("OD export must not estimate again")

    monkeypatch.setattr(sim.tracking_od, "assess_tdm_orbit_determination", forbidden)
    for root in (od_run, od_run / "estimator"):
        output = tmp_path / (root.name + ".opm")
        result = export_review(root, output, format="opm-kvn")
        assert result["state_semantics"] == "estimated"
        assert read_data(output, format="opm-kvn").covariance is not None
    imported = import_object(od_run, format="tracking-od")
    create_scenario(
        imported,
        tmp_path / "estimated.yaml",
        scenario_name="prediction",
        output_dir=tmp_path / "prediction",
        duration_s=60,
        step_s=10,
    )
    assert yaml.safe_load((tmp_path / "estimated.yaml").read_text())["simulator"]["initial_jd_utc"] == 2451545.0
    assert describe_export(od_run)["state_semantics"] == "estimated"


def test_completed_od_artifact_tampering_blocks_export(od_run, tmp_path):
    from sim.data_io.objects.od import read_tracking_od

    value = read_tracking_od(od_run)
    binding = deepcopy(value.metadata["tracking_od_binding"])
    binding["sources"][0]["sha256"] = "0" * 64
    from sim.data_io.objects.od import verify_tracking_od_binding

    with pytest.raises(DataIOError, match="changed after import"):
        verify_tracking_od_binding(binding)


def test_object_cli_uses_shared_owner_and_preserves_input(tmp_path, capsys):
    from sim.data_io.cli import main

    source = EXAMPLES / "oel_earth_eme2000_utc_v3.opm"
    assert main(["object", "describe-import", str(source), "--format", "opm-kvn"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert (
        main(
            [
                "object",
                "import",
                str(source),
                str(tmp_path / "object.yaml"),
                "--format",
                "opm-kvn",
                "--as",
                "target",
                "--scenario-name",
                "shared",
                "--run-output",
                str(tmp_path / "run"),
                "--duration-s",
                "60",
                "--step-s",
                "10",
            ]
        )
        == 0
    )
    assert yaml.safe_load((tmp_path / "object.yaml").read_text())["objects"]["target"]["specs"]["mass_kg"] == 1000


def test_model_mismatch_requires_explicit_override_and_records_both_models(tmp_path, orbital):
    source = state_product(tmp_path)
    value = json.loads(source.read_text())
    value["payload"]["model_assumptions"]["orbit_force_model"]["j2"] = False
    value["product_id"] = compute_product_id(value)
    source.write_text(json.dumps(value))
    imported = import_object(source, format="oel-state-product")
    doc = compose_object({"objects": {}, "simulator": {}}, ImportedObject(orbital, "normalized"))
    from sim.frame_time import epoch_julian_date

    doc["simulator"]["initial_jd_utc"] = epoch_julian_date(imported.epoch, "UTC")
    doc["simulator"]["dynamics"] = {"orbit": {"model": "two_body", "j2": True}}
    with pytest.raises(DataIOError, match="model assumptions differ"):
        compose_object(doc, imported)
    result = compose_object(doc, imported, allow_model_change=True)
    assert result["simulator"]["dynamics"]["orbit"]["j2"] is True
    disposition = result["metadata"]["object_imports"][-1]["disposition"]
    assert disposition["model_change_explicit"]
    assert disposition["source_model_context"]["dynamics"]["orbit"]["j2"] is False
    assert disposition["destination_model_context"]["dynamics"]["orbit"]["j2"] is True


def test_export_preflight_blocks_disabled_attitude_and_lossy_formats(real_run, od_run):
    plan = describe_export(real_run / "run", object_id="target")
    rows = {row["format"]: row for row in plan["products"]}
    assert rows["aem-kvn"]["status"] == "blocked"
    assert rows["trajectory-json"]["status"] == "ready"
    assert rows["sp3"]["status"] == "options_required"
    estimates = {row["format"]: row for row in describe_export(od_run)["products"]}
    assert estimates["spk"]["status"] == "blocked"
    assert estimates["stk-ephemeris"]["status"] == "blocked"
    assert estimates["opm-xml"]["status"] == "ready"


def test_preflight_checks_seed_frame_and_planetary_axes(tmp_path, orbital):
    from sim.data_io.tabular import trajectory_document

    source = tmp_path / "itrf.json"
    source.write_text(json.dumps(trajectory_document(replace(orbital, frame="ITRF"))))
    assert describe_import(source, format="trajectory-json")["status"] == "blocked"
    # The actual planetary review path is covered by the end-to-end acceptance run.
    # Exercise source compatibility independently of optional native backends.
    mars = replace(orbital, center="MARS", frame="SPICE:J2000")
    source = tmp_path / "mars.json"
    source.write_text(json.dumps(trajectory_document(mars)))
    assert describe_import(source, format="trajectory-json")["status"] == "ready"


def test_rejected_context_survives_multiple_external_conversions_and_sidecar_tampering(tmp_path):
    source = state_product(tmp_path, "rejected")
    opm, oem = tmp_path / "rejected.opm", tmp_path / "rejected.oem"
    convert(source, opm, input_format="oel-state-product", output_format="opm-kvn", options={"sample_index": 0})
    convert(opm, oem, input_format="opm-kvn", output_format="oem-kvn")
    assert describe_import(oem, format="oem-kvn")["status"] == "blocked"
    opm.write_text(opm.read_text() + "\n")
    with pytest.raises(DataIOError, match="receipt|identity|digest"):
        import_object(opm, format="opm-kvn")


def test_native_exports_reject_ignored_options_and_conflicting_selectors(real_run, od_run, tmp_path):
    for fmt in ("snapshot", "satellite-checkpoint", "state-product", "power-history"):
        with pytest.raises(DataIOError, match="do not accept"):
            export_object(real_run / "run", tmp_path / "out", format=fmt, options={"target_frame": "ITRF"})
    with pytest.raises(DataIOError, match="not both"):
        export_object(real_run / "run", tmp_path / "out", format="opm-kvn", final=True, sample_index=0)
    with pytest.raises(DataIOError, match="sample index zero"):
        export_review(od_run, tmp_path / "out", format="trajectory-json", options={"sample_index": 1})


def test_shared_sample_selection_applies_to_history_formats_and_preserves_review_index(real_run, tmp_path):
    from sim.data_io.api import as_trajectory

    output = tmp_path / "final.oem"
    result = export_object(real_run / "run", output, format="oem-kvn", object_id="target", final=True)
    value = as_trajectory(read_data(output, format="oem-kvn"))
    assert len(value.times_s) == 1
    assert result["source"]["sample_indices"] == [6]
    assert value.epoch == parse_time("2024-01-01T00:01:00")


def test_bound_product_cannot_silently_restore_a_different_state(tmp_path):
    imported = import_object(state_product(tmp_path), format="oel-state-product")
    states = imported.value.states.copy()
    states[0, 0] += 1
    changed = replace(imported, value=replace(imported.value, states=states))
    with pytest.raises(DataIOError, match="differs from its OEL product binding"):
        compose_object({"objects": {}, "simulator": {}}, changed)


def test_cli_composition_safely_validates_without_importing_plugins(tmp_path, capsys):
    from sim.data_io.cli import main
    from sim.scenarios import ScenarioArtifact, ScenarioBuilder

    source = EXAMPLES / "oel_earth_eme2000_utc_v3.opm"
    builder = ScenarioBuilder("destination").duration(60, dt_s=10, initial_jd_utc=2460310.5)
    builder.import_object(import_object(source, format="opm-kvn"), object_id="target")
    doc = builder.to_dict()
    doc["objects"]["target"]["flight_software"] = {
        "module": "sim.tests.missing_controller_module",
        "class_name": "MissingController",
        "hardware_profile": "hardware.passive.v1",
    }
    original = tmp_path / "destination.yaml"
    ScenarioArtifact.from_dict(doc).write(original)
    before = original.read_bytes()
    output = tmp_path / "composed.yaml"
    result = main(
        [
            "object",
            "import",
            str(source),
            str(output),
            "--format",
            "opm-kvn",
            "--into",
            str(original),
            "--as",
            "target",
            "--replace",
            "--scenario-name",
            "composed",
            "--run-output",
            str(tmp_path / "run"),
        ]
    )
    assert result == 0, capsys.readouterr()
    assert original.read_bytes() == before
    assert (
        yaml.safe_load(output.read_text())["objects"]["target"]["flight_software"]["module"]
        == "sim.tests.missing_controller_module"
    )
    assert not (tmp_path / "run").exists()


def test_missing_destination_model_context_requires_explicit_choice(orbital):
    destination = compose_object({"objects": {}, "simulator": {}}, ImportedObject(orbital, "normalized"))
    imported = ImportedObject(
        replace(orbital, metadata={"seed_simulator": {"environment": {"atmosphere": {"model": "exponential"}}}}),
        "normalized",
    )
    with pytest.raises(DataIOError, match="environment assumptions differ"):
        compose_object(destination, imported, object_id="new")
    result = compose_object(destination, imported, object_id="new", allow_model_change=True)
    assert "environment" not in result["simulator"]
    assert result["metadata"]["object_imports"][-1]["disposition"]["model_change_explicit"] is True
