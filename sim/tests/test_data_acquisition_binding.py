from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sim.data_acquisition.binding import bind_file, verify_bindings, verify_config
from sim.data_acquisition.cli import main
from sim.data_acquisition.contracts import AcquisitionError, canonical
from sim.tests.data_acquisition_helpers import GRAVITY, gp, service
from sim.tests.test_data_acquisition_environment import gravity_requirements


def selected_catalog(tmp_path):
    svc = service(tmp_path)
    source = tmp_path / "history.json"
    source.write_bytes(canonical([gp(day=9), gp(day=10), gp(day=11)]))
    value = svc.register(source, kind="catalog", format="gp-json")
    result = svc.select([value["artifact_id"]], object_ids=[123456789], at="2024-01-10T12:00:00Z")
    return svc, value, result


def study(tmp_path):
    svc, source, selected = selected_catalog(tmp_path)
    path = tmp_path / "study.yaml"
    result = svc.build_catalog_study(selected["artifact_id"], scenario_path=path, scenario_name="acquisition_test",
        output_dir=tmp_path / "run", start="2024-01-10T12:00:00Z", duration_s=10, step_s=5)
    return svc, source, selected, path, result


def test_selection_is_explicit_and_preserves_restrictions_and_parent_identity(tmp_path):
    svc, source, selected = selected_catalog(tmp_path)
    _, raw = svc.store.get(selected["artifact_id"])
    assert json.loads(raw)[0]["EPOCH"] == "2024-01-10T00:00:00"
    assert selected["parents"] == [{"artifact_id": source["artifact_id"], "sha256": source["sha256"]}]
    assert selected["terms"]["redistribution"] == "inherits-all-source-restrictions"
    assert next(r for r in svc.store.retention()["artifacts"] if r["artifact_id"] == source["artifact_id"])["references"] == [selected["artifact_id"]]


def test_selection_rejects_missing_old_and_ambiguous_elements(tmp_path):
    svc, source, _ = selected_catalog(tmp_path)
    for kwargs in ({"object_ids": [2]}, {"max_age_s": 1}, {"policy": "exact"}):
        arguments = dict(object_ids=[123456789], at="2024-01-10T12:00:00Z")
        arguments.update(kwargs)
        with pytest.raises(AcquisitionError):
            svc.select([source["artifact_id"]], **arguments)
    duplicate = tmp_path / "ambiguous.json"
    duplicate.write_bytes(canonical([gp(), gp(MEAN_ANOMALY=30.0, GP_ID=999)]))
    value = svc.register(duplicate, kind="catalog", format="gp-json")
    with pytest.raises(AcquisitionError) as error:
        svc.select([value["artifact_id"]], object_ids=[123456789], at="2024-01-10T12:00:00Z")
    assert error.value.code == "AMBIGUOUS_SELECTION"


def test_publication_cutoff_cannot_use_later_or_undated_information(tmp_path):
    svc, source, _ = selected_catalog(tmp_path)
    selected = svc.select([source["artifact_id"]], object_ids=[123456789], at="2024-01-10T12:00:00Z",
                          published_before="2024-01-10T00:00:00Z", max_age_s=2 * 86400)
    assert json.loads(svc.store.get(selected["artifact_id"])[1])[0]["EPOCH"].startswith("2024-01-09")


def test_catalog_materializer_requires_selection_and_binds_the_result(tmp_path):
    svc, source, selected, path, result = study(tmp_path)
    config = yaml.safe_load(path.read_text())
    bindings = verify_bindings(config)
    assert bindings[0]["artifact_id"] == selected["artifact_id"]
    assert config["objects"]["123456789"]["propagation_method"] == "general"
    assert json.loads(config["metadata"]["acquisition_source_receipts"][source["artifact_id"]]["receipt_json"])["sha256"] == source["sha256"]
    assert not result["execution_authorized"]
    with pytest.raises(AcquisitionError) as error:
        svc.build_catalog_study(source["artifact_id"], scenario_path=tmp_path / "blocked.yaml", scenario_name="blocked",
            output_dir=tmp_path / "run", start="2024-01-10T12:00:00Z", duration_s=10, step_s=5)
    assert error.value.code == "SELECTION_REQUIRED"


def test_changed_mean_elements_or_input_bytes_fail_before_consumption(tmp_path):
    _, _, _, path, _ = study(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["objects"]["123456789"]["initial_state"]["ogp_mean_elements"]["mean_anomaly_deg"] += 1
    with pytest.raises(AcquisitionError, match="mean elements changed"):
        verify_bindings(config)
    config = yaml.safe_load(path.read_text())
    input_path = Path(config["metadata"]["acquisition_inputs"][0]["path"])
    input_path.write_bytes(input_path.read_bytes() + b" ")
    with pytest.raises(AcquisitionError, match="identity changed"):
        verify_bindings(config)


def test_bound_catalog_epoch_and_source_receipts_cannot_drift(tmp_path):
    _, _, _, path, _ = study(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["simulator"]["initial_jd_utc"] += 1
    with pytest.raises(AcquisitionError, match="selection epoch"):
        verify_bindings(config)
    config = yaml.safe_load(path.read_text())
    record = next(iter(config["metadata"]["acquisition_source_receipts"].values()))
    record["receipt_json"] += " "
    with pytest.raises(AcquisitionError, match="source receipt changed"):
        verify_bindings(config)


def test_ordinary_validation_and_actual_run_retain_acquisition_provenance(tmp_path):
    from sim.config import load_simulation_yaml
    from sim.execution import run_simulation_config_file
    from sim.security import ConfigPathPolicy

    _, source, selected, path, _ = study(tmp_path)
    policy = ConfigPathPolicy.default(config_path=path, write_roots=(tmp_path,))
    config = load_simulation_yaml(path, path_policy=policy)
    assert verify_config(config)[0]["artifact_id"] == selected["artifact_id"]
    run_simulation_config_file(path, path_policy=policy)
    summary = json.loads((tmp_path / "run" / "master_run_summary.json").read_text())
    assert summary["samples"] == 3
    assert summary["acquisition_provenance"][0]["source_sha256"] == selected["sha256"]
    assert summary["acquisition_source_receipts"][source["artifact_id"]]["source_name"] == "history.json"


def test_direct_engine_entry_rechecks_acquired_file_after_validation(tmp_path):
    from sim.config import load_simulation_yaml
    from sim.single_run import _SingleRunEngine
    _, _, _, path, _ = study(tmp_path)
    config = load_simulation_yaml(path)
    input_path = Path(config.metadata["acquisition_inputs"][0]["path"])
    input_path.write_text("[]")
    with pytest.raises(AcquisitionError, match="identity changed"):
        _SingleRunEngine(config)


def test_bound_gravity_cannot_silently_change_degree_or_source(tmp_path):
    svc = service(tmp_path)
    source = tmp_path / "model.gfc"
    source.write_bytes(GRAVITY)
    artifact = svc.register(source, kind="gravity", format="icgem")
    destination = tmp_path / "inputs" / "gravity.gfc"
    svc.materialize(artifact["artifact_id"], destination, requirements=gravity_requirements())
    config = {"simulator": {"dynamics": {"orbit": {"spherical_harmonics": {"source": "icgem", "degree": 2, "order": 0}}}}}
    bound = bind_file(config, path=destination, field="simulator.dynamics.orbit.spherical_harmonics.coeff_path")
    assert verify_bindings(bound)
    bound["simulator"]["dynamics"]["orbit"]["spherical_harmonics"]["degree"] = 3
    with pytest.raises(AcquisitionError):
        verify_bindings(bound)


def test_acquired_gravity_eop_and_weather_run_through_existing_onp_engine(tmp_path):
    from datetime import datetime, timezone

    from sim.config import load_simulation_yaml
    from sim.dynamics.orbit.epoch import datetime_to_julian_date
    from sim.execution import run_simulation_config_file
    from sim.security import ConfigPathPolicy
    from sim.tests.data_acquisition_helpers import eop, weather
    from sim.tests.test_data_acquisition_environment import environmental_requirements

    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/automation_smoke.yaml").read_text())
    config["scenario_name"] = "acquired_environment_smoke"
    simulator = config["simulator"]
    simulator.update(duration_s=10, dt_s=5, initial_jd_utc=datetime_to_julian_date(datetime(2024, 1, 10, tzinfo=timezone.utc)))
    simulator["dynamics"]["attitude"]["enabled"] = False
    simulator["dynamics"]["orbit"].update(drag=True, spherical_harmonics={"enabled": True, "source": "icgem", "degree": 2, "order": 0})
    simulator["environment"]["atmosphere_model"] = "nrlmsise00"
    simulator["frames"] = {"model": "iau76_80_eop"}
    config["outputs"].update(output_dir=str(tmp_path / "run"), mode="save", stats={"save_full_log": False},
                            review={"enabled": True, "detail": "standard"})
    svc = service(tmp_path)
    for raw, kind, format, field, requirements in [
        (GRAVITY, "gravity", "icgem", "simulator.dynamics.orbit.spherical_harmonics.coeff_path", gravity_requirements()),
        (eop(), "eop", "celestrak-eop-csv", "simulator.frames.eop_path", environmental_requirements()),
        (weather(), "space-weather", "celestrak-weather-csv", "simulator.environment.nrlmsise00_sw_path", environmental_requirements("nrlmsise00")),
    ]:
        path = tmp_path / f"source-{kind}"
        path.write_bytes(raw)
        artifact = svc.register(path, kind=kind, format=format)
        target = tmp_path / "inputs" / kind
        svc.materialize(artifact["artifact_id"], target, requirements=requirements)
        config = bind_file(config, path=target, field=field)
    path = tmp_path / "onp.yaml"
    path.write_text(yaml.safe_dump(config))
    policy = ConfigPathPolicy.default(config_path=path, write_roots=(tmp_path,))
    load_simulation_yaml(path, path_policy=policy)
    run_simulation_config_file(path, path_policy=policy)
    summary = json.loads((tmp_path / "run/master_run_summary.json").read_text())
    assert summary["samples"] == 3 and len(summary["acquisition_provenance"]) == 3


def test_wheel_schemas_validate_returned_contracts(tmp_path):
    from importlib.resources import files

    import jsonschema

    from sim.data_acquisition import Request
    request_schema = json.loads(files("sim.data_acquisition").joinpath("schemas/request-v1.schema.json").read_text())
    receipt_schema = json.loads(files("sim.data_acquisition").joinpath("schemas/receipt-v1.schema.json").read_text())
    jsonschema.validate(Request(provider="gravity", product="egm96").to_dict(), request_schema)
    _, _, result = selected_catalog(tmp_path)
    jsonschema.validate(result, receipt_schema)


def test_binding_uses_existing_path_security(tmp_path):
    from sim.security import ConfigPathPolicy
    from sim.security.config_paths import ConfigPathSecurityError
    _, _, _, path, _ = study(tmp_path)
    config = yaml.safe_load(path.read_text())
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    policy = ConfigPathPolicy(workspace_root=restricted, read_roots=(restricted,))
    with pytest.raises(ConfigPathSecurityError):
        verify_bindings(config, path_policy=policy)


def test_cli_returns_structured_errors_without_source_data(tmp_path, capsys):
    path = tmp_path / "request.json"
    path.write_text('{"password":"do-not-echo"}')
    assert main(["plan", str(path)]) == 2
    output = capsys.readouterr().out
    assert "do-not-echo" not in output and json.loads(output)["error"] == "INVALID_REQUEST"
    assert main(["providers"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["mcp_network_tools"] is False


def test_cli_binds_an_incomplete_gravity_draft_then_safely_validates_it(tmp_path, capsys):
    svc = service(tmp_path)
    model = tmp_path / "source.gfc"
    model.write_bytes(GRAVITY)
    artifact = svc.register(model, kind="gravity", format="icgem")
    prepared = tmp_path / "inputs/model.gfc"
    svc.materialize(artifact["artifact_id"], prepared, requirements=gravity_requirements())
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/automation_smoke.yaml").read_text())
    config["outputs"]["output_dir"] = str(tmp_path / "run")
    config["simulator"]["dynamics"]["orbit"]["spherical_harmonics"] = {"enabled": True, "source": "icgem", "degree": 2, "order": 0}
    draft = tmp_path / "draft.yaml"
    draft.write_text(yaml.safe_dump(config))
    destination = tmp_path / "bound.yaml"
    arguments = ["bind", str(draft), "--input", str(prepared), "--field", "simulator.dynamics.orbit.spherical_harmonics.coeff_path", "--out", str(destination)]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["result"]["execution_authorized"] is False
    assert verify_bindings(yaml.safe_load(destination.read_text()))
    # Reuse the authoritative scenario parser's duplicate-key rejection.
    draft.write_text(draft.read_text() + "scenario_name: duplicate\n")
    destination.unlink()
    assert main(arguments) == 2 and not destination.exists()
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_credentials_cannot_be_passed_noninteractively(capsys):
    assert main(["connections", "add", "spacetrack", "--name", "test"]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "INTERACTIVE_SETUP_REQUIRED"
