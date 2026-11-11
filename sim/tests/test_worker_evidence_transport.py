"""Behavioral parity for bounded persistent-worker reporting evidence."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import yaml

from sim.config import scenario_config_from_dict
from sim.execution import object_workers
from sim.performance.suite import physics_payload_hash
from sim.single_run import _SingleRunEngine

_ORDINARY_WORKER_LOOP = object_workers._persistent_object_worker_loop


def _complete_evidence_worker_loop(connection, engines) -> None:
    # A top-level target runs this override inside spawned workers as well.
    object_workers._compact_flight_software_evidence = lambda *_: None
    _ORDINARY_WORKER_LOOP(connection, engines)


def test_evidence_delta_transport_preserves_complete_process_run(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(sys, "path", [str(root), *sys.path])
    config = yaml.safe_load((root / "configs" / "quickstart_5min.yaml").read_text())
    config["objects"]["observer"] = deepcopy(config["objects"]["target"])
    config["simulator"]["duration_s"] = 20.0
    config["simulator"]["resource_profile"] = "off"
    config["simulator"]["execution"] = {
        "policy": "parallel",
        "object_parallelism": {"enabled": True, "backend": "process_pool", "workers": 2, "min_objects": 3},
    }
    config["outputs"]["output_dir"] = str(tmp_path / "run")
    config["outputs"]["plots"] = {"enabled": False}
    config["outputs"]["animations"] = {"enabled": False}
    config["outputs"]["review"] = {"enabled": False}
    config["outputs"]["stats"] = {
        "print_summary": False, "save_json": False, "save_csv": False, "save_full_log": False,
    }
    cfg = scenario_config_from_dict(config)
    compacted = _SingleRunEngine(cfg).run()
    monkeypatch.setattr(object_workers, "_persistent_object_worker_loop", _complete_evidence_worker_loop)
    complete = _SingleRunEngine(cfg).run()
    assert physics_payload_hash(compacted) == physics_payload_hash(complete)
    assert compacted["flight_software_evidence_by_object"]
    for evidence in compacted["flight_software_evidence_by_object"].values():
        assert len(evidence["invocations"]) > 2
        assert len(evidence["realizations"]) > len(evidence["invocations"])
