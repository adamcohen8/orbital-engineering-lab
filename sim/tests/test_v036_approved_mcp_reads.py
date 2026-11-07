from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.policy import MCPPathPolicy
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from sim.config import load_simulation_yaml
from sim.config.scenario.presets import _resolve_agent_preset
from sim.dynamics.orbit.cr3bp import cr3bp_halo_seed_state_km_s, propagate_cr3bp_state
from sim.review.cr3bp import STATE_COLUMNS
from sim.security import ConfigPathPolicy, ConfigPathSecurityError, SecureReadRoot, SecureReadUnavailable
from sim.security import config_paths as config_paths_module

ROOT = Path(__file__).resolve().parents[2]
HANDLING = {"marking": "PUBLIC_TEST", "release_scope": "public"}
WRITE_APPROVAL = {"approval_id": "v036-read-write", "scope": "write"}
TDM_FIXTURE = ROOT / "examples" / "tracking_od" / "public_reduced_geometric_azel_range.tdm"


def _rename_root_after_open(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    replacement_files: dict[str, bytes] | None = None,
) -> tuple[Path, dict[str, bool]]:
    """Replace a visible root immediately after its descriptor is acquired."""

    original_open = SecureReadRoot.open
    moved = root.with_name(root.name + "-retained")
    state = {"swapped": False}

    def open_and_replace(cls: Any, path: str | Path, **kwargs: Any) -> SecureReadRoot:
        reader = original_open(path, **kwargs)
        if not state["swapped"] and Path(path).absolute() == root.absolute():
            root.rename(moved)
            root.mkdir()
            for relative, payload in (replacement_files or {}).items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            state["swapped"] = True
        return reader

    monkeypatch.setattr(SecureReadRoot, "open", classmethod(open_and_replace))
    return moved, state


def _minimal_yaml(output_dir: Path) -> str:
    return (
        "scenario_name: retained_yaml\n"
        "simulator:\n"
        "  duration_s: 1\n"
        "  dt_s: 1\n"
        "outputs:\n"
        f"  output_dir: {json.dumps(str(output_dir))}\n"
        "  mode: save\n"
    )


def test_local_yaml_reads_consume_retained_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    config_path = root / "scenario.yaml"
    config_path.write_text(_minimal_yaml(tmp_path / "output"), encoding="utf-8")
    _moved, state = _rename_root_after_open(
        monkeypatch,
        root,
        {"scenario.yaml": b"scenario_name: replacement\n[invalid\n"},
    )
    policy = ConfigPathPolicy.default(
        config_path=config_path,
        workspace_root=tmp_path,
        read_roots=(root,),
        write_roots=(tmp_path,),
    )

    loaded = load_simulation_yaml(config_path, path_policy=policy)

    assert state["swapped"] is True
    assert loaded.scenario_name == "retained_yaml"


def test_preset_reads_consume_retained_content_after_logical_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "preset-inputs"
    root.mkdir()
    config_path = root / "scenario.yaml"
    config_path.write_text(_minimal_yaml(tmp_path / "output"), encoding="utf-8")
    preset = root / "vehicle.yaml"
    preset.write_text("specs:\n  mass_kg: 120\n", encoding="utf-8")
    policy = ConfigPathPolicy.default(
        config_path=config_path,
        workspace_root=tmp_path,
        read_roots=(root,),
        write_roots=(tmp_path,),
    )
    # Presets use the exact read API after logical resolution; an old
    # authorize-then-reopen sequence would parse the replacement below.
    _moved, preset_state = _rename_root_after_open(
        monkeypatch,
        root,
        {"vehicle.yaml": b"specs: [invalid\n"},
    )
    resolved = _resolve_agent_preset(
        {"preset": "vehicle.yaml"},
        role="objects.target",
        base_dir=root,
        path_policy=policy,
    )
    assert preset_state["swapped"] is True
    assert resolved["specs"]["mass_kg"] == 120


def test_secure_policy_rejects_missing_posix_primitives_and_trusted_cli_fallback_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "scenario.yaml"
    source.write_text("scenario_name: bounded\n", encoding="utf-8")
    monkeypatch.setattr(config_paths_module, "secure_read_primitives_available", lambda: False)

    def reject_secure_reads() -> None:
        raise SecureReadUnavailable("test host has no descriptor primitives")

    monkeypatch.setattr(config_paths_module, "require_secure_read_primitives", reject_secure_reads)
    secure_policy = ConfigPathPolicy.default(
        config_path=source,
        workspace_root=tmp_path,
        read_roots=(tmp_path,),
        require_secure_reads=True,
    )
    with pytest.raises(SecureReadUnavailable):
        secure_policy.read_input_bytes(source, purpose="secure scenario")

    trusted_policy = ConfigPathPolicy.default(
        config_path=source,
        workspace_root=tmp_path,
        read_roots=(tmp_path,),
    )
    logical, payload = trusted_policy.read_input_bytes(
        source,
        purpose="trusted scenario",
        max_bytes=128,
    )
    assert logical == source
    assert payload == b"scenario_name: bounded\n"
    with pytest.raises(ValueError, match="bounded read limit"):
        trusted_policy.read_input_bytes(source, purpose="trusted scenario", max_bytes=4)

    original_open = Path.open
    mutated = {"done": False}

    def open_then_replace(path: Path, *args: Any, **kwargs: Any):
        stream = original_open(path, *args, **kwargs)
        if path == source and not mutated["done"]:
            mutated["done"] = True
            source.write_text("scenario_name: changed during read\n", encoding="utf-8")
        return stream

    monkeypatch.setattr(Path, "open", open_then_replace)
    with pytest.raises(ConfigPathSecurityError, match="changed during the compatibility read"):
        trusted_policy.read_input_bytes(source, purpose="trusted scenario", max_bytes=128)


def _public_handlers(root: Path, *, writes: Path | None = None) -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(
        read_roots=(root,),
        write_roots=(writes or root,),
        approval_policy=ExecutionApprovalPolicy(
            write_approval_ids=frozenset({"v036-read-write"}),
        ),
    )


def test_tdm_and_eop_handlers_consume_bytes_after_visible_root_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ccsds-inputs"
    root.mkdir()
    tdm = root / TDM_FIXTURE.name
    tdm.write_bytes(TDM_FIXTURE.read_bytes())
    expected_tdm_hash = hashlib.sha256(tdm.read_bytes()).hexdigest()
    handlers = _public_handlers(root, writes=tmp_path)
    _moved, tdm_state = _rename_root_after_open(
        monkeypatch,
        root,
        {tdm.name: b"replacement is not a TDM\n"},
    )
    inspected_tdm = handlers.inspect_ccsds(
        path=tdm,
        product_kind="tdm",
        handling=HANDLING,
    )
    assert tdm_state["swapped"] is True
    assert inspected_tdm["status"] == "completed"
    assert inspected_tdm["result"]["source_sha256"] == expected_tdm_hash

    root = tmp_path / "eop-inputs"
    root.mkdir()
    eop = root / "eop.csv"
    eop.write_text(
        "MJD,xp,yp,UT1-UTC,dX,dY,LOD,quality\n"
        "60310,0.136894,0.202185,0.0087572,0.283,-0.183,0.121,observed-final\n"
        "60311,0.136920,0.202210,0.0086900,0.280,-0.180,0.122,predicted\n",
        encoding="utf-8",
    )
    expected_eop_hash = hashlib.sha256(eop.read_bytes()).hexdigest()
    handlers = _public_handlers(root, writes=tmp_path)
    _moved, eop_state = _rename_root_after_open(
        monkeypatch,
        root,
        {"eop.csv": b"replacement,not,c04\n"},
    )
    inspected_eop = handlers.convert_frame_time(
        operation="inspect_eop",
        eop_path=eop,
        eop_format="c04_csv",
        as_of="2024-01-01T00:00:00Z",
        max_observed_age_days=45,
        handling=HANDLING,
    )
    assert eop_state["swapped"] is True
    assert inspected_eop["status"] == "completed"
    assert inspected_eop["result"]["eop_source"]["source_sha256"] == expected_eop_hash


def _completed_output(root: Path) -> Path:
    output = root / "report-run"
    review = output / "review"
    review.mkdir(parents=True)
    with sqlite3.connect(review / "run.sqlite") as conn:
        conn.execute(
            "CREATE TABLE run_metadata "
            "(scenario_name TEXT, duration_s REAL, dt_s REAL, samples INTEGER, oel_version TEXT, "
            "review_schema_version INTEGER)"
        )
        conn.execute(
            "INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?)",
            ("retained-report", 2.0, 1.0, 3, "test", 1),
        )
    (output / "master_run_summary.json").write_text(
        json.dumps({"scenario_name": "retained-report", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    return output


@contextlib.contextmanager
def _replace_after_snapshot(
    original: Any,
    source_root: Path,
    state: dict[str, bool],
) -> Iterator[tuple[Path, Path]]:
    with original as pair:
        if not state["swapped"]:
            moved = source_root.with_name(source_root.name + "-retained")
            source_root.rename(moved)
            source_root.mkdir()
            (source_root / "review").mkdir()
            (source_root / "review" / "run.sqlite").write_bytes(b"replacement sqlite")
            state["swapped"] = True
        yield pair


def test_report_and_cr3bp_handlers_keep_logical_identity_when_snapshot_root_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _completed_output(tmp_path)
    handlers = _public_handlers(tmp_path)
    original_snapshot = MCPPathPolicy.snapshot_read_files
    report_state = {"swapped": False}

    def report_snapshot(self: MCPPathPolicy, root: str | Path, relative_paths: Any, **kwargs: Any):
        context = original_snapshot(self, root, relative_paths, **kwargs)
        if Path(root).absolute() == output.absolute():
            return _replace_after_snapshot(context, output, report_state)
        return context

    monkeypatch.setattr(MCPPathPolicy, "snapshot_read_files", report_snapshot)
    prepared = handlers.prepare_report_packet(
        source_output_dir=output,
        packet_output_dir=tmp_path / "packet",
        packet_id="retained-report",
        query_names=["run_metadata"],
        max_rows=10,
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )
    assert report_state["swapped"] is True
    assert prepared["status"] == "completed"
    packet = json.loads(Path(prepared["result"]["packet_path"]).read_text(encoding="utf-8"))
    assert packet["source_output_dir"] == str(output)
    assert packet["source_evidence_identity_sha256"]

    # Rebuild an independent source root because the report race intentionally
    # replaces its visible directory.
    cr3bp_root = tmp_path / "cr3bp-run"
    review = cr3bp_root / "review"
    review.mkdir(parents=True)
    (cr3bp_root / "index.md").write_text("# CR3BP review\n", encoding="utf-8")
    with sqlite3.connect(review / "run.sqlite") as conn:
        conn.execute("CREATE TABLE object_state_frame (object_id TEXT, state_frame TEXT)")
        conn.execute("INSERT INTO object_state_frame VALUES ('vehicle', 'cr3bp_rotating')")
        conn.execute(
            "CREATE TABLE object_propagation "
            "(object_id TEXT, propagator_family TEXT, propagator_name TEXT)"
        )
        conn.execute("INSERT INTO object_propagation VALUES ('vehicle', 'CR3BP', 'earth_moon CR3BP')")
        conn.execute(
            "CREATE TABLE object_state (object_id TEXT, time_s REAL, "
            + ", ".join(column + " REAL" for column in STATE_COLUMNS)
            + ")"
        )
        state = cr3bp_halo_seed_state_km_s(family="l1_northern_large")
        for time_s in np.arange(0.0, 900.0, 300.0):
            conn.execute(
                "INSERT INTO object_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ["vehicle", float(time_s), *state],
            )
            state = propagate_cr3bp_state(state, 300.0, float(time_s))

    cr3bp_state = {"swapped": False}

    def cr3bp_snapshot(self: MCPPathPolicy, root: str | Path, relative_paths: Any, **kwargs: Any):
        context = original_snapshot(self, root, relative_paths, **kwargs)
        if Path(root).absolute() == cr3bp_root.absolute():
            return _replace_after_snapshot(context, cr3bp_root, cr3bp_state)
        return context

    monkeypatch.setattr(MCPPathPolicy, "snapshot_read_files", cr3bp_snapshot)
    planned = handlers.plan_cr3bp_render(
        output_dir=cr3bp_root,
        object_id="vehicle",
        kind="trajectory",
        axes="rotating",
        origin="barycenter",
        plane="xy",
        resolution=31,
        frames=6,
        fps=3,
        style="oel_dark",
        handling=HANDLING,
    )
    assert cr3bp_state["swapped"] is True
    assert planned["status"] == "completed"
    assert planned["result"]["output_dir"] == str(cr3bp_root)
    assert planned["result"]["evidence_identity"]["relative_path"] == "review/run.sqlite"
    assert planned["result"]["evidence_identity"]["sha256"]
