from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_cmd(py: str, script: Path, args: list[str]) -> list[str]:
    return [py, str(script)] + args


def _iter_flight_software_sections(root: dict) -> list[dict]:
    sections: list[dict] = []
    for key in ("rocket", "chaser", "target"):
        section = root.get(key)
        if isinstance(section, dict) and isinstance(section.get("flight_software"), dict):
            sections.append(section["flight_software"])
    objects = root.get("objects")
    if isinstance(objects, dict):
        for section in objects.values():
            if isinstance(section, dict) and isinstance(section.get("flight_software"), dict):
                sections.append(section["flight_software"])
    return sections


def _write_stack_config(
    *,
    config_path: Path,
    bind_host: str,
    sim_port: int,
    mock_port: int,
    allow_non_loopback: bool,
) -> Path:
    if allow_non_loopback or bind_host not in {"127.0.0.1", "localhost"}:
        raise ValueError("the maintained cFS v2 mock mission is loopback-only")
    root = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(root, dict):
        raise ValueError(f"Scenario YAML root must be a mapping: {config_path}")
    updated = 0
    for flight_software in _iter_flight_software_sections(root):
        if str(flight_software.get("class_name", "")) != "CfsV2UdpFlightSoftwareAdapter":
            continue
        params = dict(flight_software.get("params", {}) or {})
        params["bind_host"] = str(bind_host)
        params["bind_port"] = int(sim_port)
        params["cfs_host"] = str(bind_host)
        params["cfs_port"] = int(mock_port)
        flight_software["params"] = params
        updated += 1
    if updated <= 0:
        raise ValueError(f"No cFS v2 flight_software section found in {config_path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="oel_cfs_sil_stack_",
        dir=str(config_path.parent),
        delete=False,
        encoding="utf-8",
    ) as tf:
        yaml.safe_dump(root, tf, sort_keys=False)
        return Path(tf.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mock cFS endpoint + config-driven simulator SIL scenario.")
    parser.add_argument(
        "--mock-mode", choices=["hold", "prograde", "retrograde", "att_damp", "att_hold"], default="hold"
    )
    parser.add_argument("--accel-mag-km-s2", type=float, default=2e-6)
    parser.add_argument("--att-kp", type=float, default=0.10)
    parser.add_argument("--att-kd", type=float, default=0.05)
    parser.add_argument("--max-torque-nm", type=float, default=0.05)
    parser.add_argument("--desired-quat-bn", type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--bind-host", type=str, default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=50100)
    parser.add_argument("--mock-port", type=int, default=50101)
    parser.add_argument("--config", type=str, default="configs/cfs_sil_single_sat.yaml")
    parser.add_argument("--allow-non-loopback", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    mock_script = root / "integrations" / "cfs_sil" / "mock_cfs_v2_endpoint.py"
    sim_script = root / "run_simulation.py"
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    py = sys.executable
    sim_config_path = _write_stack_config(
        config_path=config_path,
        bind_host=args.bind_host,
        sim_port=int(args.sim_port),
        mock_port=int(args.mock_port),
        allow_non_loopback=bool(args.allow_non_loopback),
    )

    mock_cmd = _build_cmd(
        py,
        mock_script,
        [
            "--bind-host",
            args.bind_host,
            "--bind-port",
            str(args.mock_port),
            "--att-kp",
            str(args.att_kp),
            "--att-kd",
            str(args.att_kd),
            "--max-torque-nm",
            str(args.max_torque_nm),
        ],
    )
    sim_cmd = _build_cmd(
        py,
        sim_script,
        [
            "--config",
            str(sim_config_path),
        ],
    )

    print("Starting mock cFS endpoint...")
    mock_proc = subprocess.Popen(mock_cmd)
    time.sleep(0.4)

    print("Starting simulator loop...")
    sim_rc = 1
    try:
        sim_rc = subprocess.call(sim_cmd)
    finally:
        if mock_proc.poll() is None:
            print("Stopping mock cFS endpoint...")
            mock_proc.terminate()
            try:
                mock_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                mock_proc.send_signal(signal.SIGKILL)
                mock_proc.wait(timeout=2.0)
        try:
            sim_config_path.unlink()
        except OSError:
            pass

    return int(sim_rc)


if __name__ == "__main__":
    raise SystemExit(main())
