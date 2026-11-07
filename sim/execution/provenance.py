from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def runtime_implementation_digest() -> str:
    """Digest executable OEL source so checkpoints cannot survive code changes."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


_ENVIRONMENT_INPUT_FIELDS = (
    "de440_coeff_path", "de440_eop_path", "density_eop_path", "drag_eop_path",
    "harris_priester_coeff_path", "hp_coeff_path", "jacchia70_sw_path",
    "jb2006_ap_path", "jb2006_sol_path", "jb2008_dtc_path", "jb2008_sol_path",
    "msis86_sw_path", "msis_sw_path", "nrlmsise00_sw_path", "spherical_harmonics_eop_path",
)
_ORBIT_INPUT_FIELDS = ("drag_eop_path", "de440_coeff_path", "de440_eop_path")
_GEOMETRY_INPUT_FIELDS = ("geometry_profile_path", "area_profile_path", "attitude_area_profile_path")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _configured_input_paths(config_dict: dict[str, Any]) -> dict[str, str]:
    """List configured file inputs consumed by dynamics or object geometry."""

    inputs: dict[str, str] = {}

    def add(prefix: str, section: Any, fields: tuple[str, ...]) -> None:
        mapping = _mapping(section)
        for field in fields:
            raw = mapping.get(field)
            if raw not in (None, ""):
                inputs[f"{prefix}.{field}"] = str(raw)

    simulator = _mapping(config_dict.get("simulator"))
    environment = _mapping(simulator.get("environment"))
    orbit = _mapping(_mapping(simulator.get("dynamics")).get("orbit"))
    add("simulator.frames", simulator.get("frames"), ("eop_path",))
    add("simulator.environment", environment, _ENVIRONMENT_INPUT_FIELDS)
    add("simulator.environment.atmosphere_env", environment.get("atmosphere_env"), _ENVIRONMENT_INPUT_FIELDS)
    for index, raw in enumerate(environment.get("spice_kernels", []) or []):
        if raw not in (None, ""):
            inputs[f"simulator.environment.spice_kernels[{index}]"] = str(raw)
    add("simulator.dynamics.orbit", orbit, _ORBIT_INPUT_FIELDS)
    add("simulator.dynamics.orbit.ocean_tides", orbit.get("ocean_tides"), ("coeff_path",))
    add(
        "simulator.dynamics.orbit.spherical_harmonics",
        orbit.get("spherical_harmonics"),
        ("coeff_path", "source_path", "eop_path"),
    )
    for object_id, section in _mapping(config_dict.get("objects")).items():
        specs = _mapping(_mapping(section).get("specs"))
        prefix = f"objects.{object_id}.specs"
        add(prefix, specs, _GEOMETRY_INPUT_FIELDS)
        add(f"{prefix}.geometry", specs.get("geometry"), ("profile_path", "area_profile_path", "attitude_area_profile_path"))
        add(f"{prefix}.aero", specs.get("aero"), ("geometry_profile_path", "area_profile_path"))
    return inputs


def _file_sha256(path: str, size: int, mtime_ns: int, ctime_ns: int, inode: int) -> str:
    # Checkpoints bind bytes, even when two writes share coarse filesystem metadata.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_physics_input_digests(config_dict: dict) -> dict[str, str]:
    """Bind configured file-backed physics inputs to campaign checkpoints."""

    orbit = _mapping(_mapping(_mapping(config_dict.get("simulator")).get("dynamics")).get("orbit"))
    ocean = _mapping(orbit.get("ocean_tides"))
    if ocean.get("enabled", False) and not ocean.get("coeff_path"):
        raise ValueError("Enabled ocean tides require a coefficient path for checkpoint identity")
    digests: dict[str, str] = {}
    for field, raw_path in sorted(_configured_input_paths(config_dict).items()):
        path = Path(raw_path).expanduser()
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            if field == "simulator.dynamics.orbit.ocean_tides.coeff_path" and ocean.get("enabled", False):
                raise
            digests[field] = "missing"
            continue
        if not path.is_file():
            raise ValueError(f"Configured physics input is not a regular file: {field}")
        digests[field] = _file_sha256(
            str(path.resolve()), file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns, file_stat.st_ino
        )
    return digests
