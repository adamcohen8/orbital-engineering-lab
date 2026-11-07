from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sim.acceleration.settings import acceleration_settings_from_config
from sim.config import default_reference_object_id, iter_object_sections


class SimulationMemoryBudgetError(MemoryError):
    """Raised before a run allocates more history memory than the active budget allows."""


class ResourcePressureError(RuntimeError):
    """Raised when a guarded batch run should not start on the current machine."""


@dataclass(frozen=True)
class HistoryMemoryEstimate:
    samples: int
    active_objects: int
    knowledge_pairs: int
    array_bytes: int
    estimated_peak_bytes: int
    limit_bytes: int

    @property
    def estimated_peak_mb(self) -> float:
        return float(self.estimated_peak_bytes) / (1024.0 * 1024.0)

    @property
    def limit_mb(self) -> float:
        return float(self.limit_bytes) / (1024.0 * 1024.0)


DEFAULT_MAX_HISTORY_MEMORY_MB = 1024.0
_ENV_MAX_HISTORY_MEMORY_MB = "OEL_MAX_HISTORY_MEMORY_MB"

# NumPy history arrays use ``np.intp``/the platform pointer-sized integer for
# their leading dimension.  Keep the check in this dependency-light module so
# config preflight and the live single-run owner reject the same impossible
# duration/timestep pair before trying to construct an array.
_MAX_PLANNED_STEPS = sys.maxsize - 1


def _simulation_completion_tolerance_s(duration_s: float) -> float:
    """Return one local floating-point ULP without consuming a duration."""

    duration = float(duration_s)
    if not math.isfinite(duration):
        raise ValueError("simulation duration_s must be finite.")
    if duration <= 0.0:
        return 0.0
    # A scale-relative tolerance (for example, 1e-9 * duration) can consume
    # thousands of seconds from a long simulation.  The next representable
    # value at the requested epoch is the appropriate roundoff bound.
    return min(math.ulp(duration), 0.5 * duration)


def _planned_step_count(duration_s: float, dt_s: float) -> int:
    """Return the bounded number of intervals needed for a run.

    The check is deliberately performed before history sizing/allocation.  A
    finite duration divided by a subnormal timestep can overflow to ``inf``;
    converting that value to an array length otherwise raises an opaque
    ``OverflowError`` (or attempts an impossible allocation).
    """

    duration = float(duration_s)
    dt = float(dt_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("simulation duration_s must be positive and finite.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("simulation dt_s must be positive and finite.")

    ratio = duration / dt
    if not math.isfinite(ratio):
        raise ValueError(
            "simulation duration_s/dt_s ratio is not representable for history allocation; "
            "reduce duration_s or increase dt_s."
        )

    completion_tol_s = _simulation_completion_tolerance_s(duration)
    adjusted_ratio = max((duration - completion_tol_s) / dt, 0.0)
    if not math.isfinite(adjusted_ratio):
        raise ValueError(
            "simulation duration_s/dt_s ratio is not representable for history allocation; "
            "reduce duration_s or increase dt_s."
        )
    # A positive duration always needs at least one advancing interval, even
    # when duration/dt underflows to zero (for example, a subnormal duration
    # paired with an intentionally very large requested timestep).
    planned_steps = max(1, int(math.ceil(adjusted_ratio)))
    if planned_steps > _MAX_PLANNED_STEPS:
        raise ValueError(
            "simulation duration_s/dt_s ratio exceeds the platform's representable "
            "history length; reduce duration_s or increase dt_s."
        )
    return planned_steps


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    description: str
    max_parallel_workers: int | None = None
    force_serial: bool = False
    disable_plots: bool = False
    checkpoint_enabled: bool = True
    throttle_enabled: bool = True
    # Legacy config names are retained, but these floors apply to projected
    # post-start headroom rather than the raw preflight snapshot.
    min_available_memory_mb: float | None = None
    hard_min_available_memory_mb: float | None = None
    max_load_per_cpu: float | None = None
    pause_seconds: float = 10.0
    max_wait_s: float = 1800.0


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp_s: float
    cpu_count: int
    load_1m: float | None = None
    total_memory_mb: float | None = None
    available_memory_mb: float | None = None
    memory_pressure_free_percent: float | None = None
    platform: str = platform.system()

    @property
    def load_per_cpu(self) -> float | None:
        if self.load_1m is None or self.cpu_count <= 0:
            return None
        return float(self.load_1m) / float(self.cpu_count)


@dataclass(frozen=True)
class ResourceEstimate:
    profile: str
    study_type: str
    runs: int
    steps_per_run: int
    requested_workers: int
    effective_workers: int
    active_objects: int
    plots_enabled: bool
    checkpoint_enabled: bool
    estimated_history_mb_per_run: float
    estimated_parallel_history_mb: float
    estimated_incremental_memory_mb: float
    current_available_memory_mb: float | None
    projected_available_memory_mb: float | None
    memory_pressure_free_percent: float | None
    load_per_cpu: float | None
    acceleration_mode: str
    acceleration_backend: str
    risk: str
    notes: tuple[str, ...] = ()
    hierarchical_processes: int = 1
    object_workers_per_run: int = 0

    @property
    def action(self) -> str:
        if self.risk == "unsafe":
            return "refuse"
        if self.risk in {"moderate", "heavy"}:
            return "advisory"
        return "proceed"


DEFAULT_PARALLEL_WORKER_OVERHEAD_MB = 192.0
DEFAULT_PLOT_OVERHEAD_MB = 96.0
MACOS_MEMORY_PRESSURE_WARN_PERCENT = 10.0
MACOS_MEMORY_PRESSURE_CRITICAL_PERCENT = 5.0


RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    "off": ResourceProfile(
        name="off",
        description="Do not apply resource safety rewrites or runtime pressure checks.",
        checkpoint_enabled=False,
        throttle_enabled=False,
    ),
    "config": ResourceProfile(
        name="config",
        description="Use the config as written, but keep checkpointing and pressure checks enabled.",
    ),
    "laptop-safe": ResourceProfile(
        name="laptop-safe",
        description="One run at a time, plots disabled, checkpoint/resume enabled, and conservative pressure gates.",
        max_parallel_workers=1,
        force_serial=True,
        disable_plots=True,
        checkpoint_enabled=True,
        throttle_enabled=True,
        min_available_memory_mb=512.0,
        hard_min_available_memory_mb=256.0,
        max_load_per_cpu=1.25,
        pause_seconds=15.0,
    ),
    "standard": ResourceProfile(
        name="standard",
        description="A balanced local profile with checkpointing, throttling, and at most two workers.",
        max_parallel_workers=2,
        checkpoint_enabled=True,
        throttle_enabled=True,
        min_available_memory_mb=512.0,
        hard_min_available_memory_mb=256.0,
        max_load_per_cpu=1.75,
        pause_seconds=10.0,
    ),
    "aggressive": ResourceProfile(
        name="aggressive",
        description="Checkpoint/resume stays on, but worker count follows the config unless the system is already unsafe.",
        checkpoint_enabled=True,
        throttle_enabled=True,
        min_available_memory_mb=256.0,
        hard_min_available_memory_mb=128.0,
        max_load_per_cpu=3.0,
        pause_seconds=5.0,
    ),
}


def _positive_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    out = float(value)
    if out <= 0:
        raise ValueError("max history memory must be positive.")
    return out


def configured_history_memory_limit_mb(cfg: Any) -> float:
    external_limit = _positive_float_or_none(os.environ.get(_ENV_MAX_HISTORY_MEMORY_MB))
    if external_limit is None:
        external_limit = DEFAULT_MAX_HISTORY_MEMORY_MB

    resource_limits = dict(getattr(getattr(cfg, "outputs", None), "resource_limits", {}) or {})
    config_limit = _positive_float_or_none(resource_limits.get("max_history_memory_mb"))
    if config_limit is None:
        return float(external_limit)
    return float(min(external_limit, config_limit))


def bytes_from_mb(value_mb: float) -> int:
    return int(float(value_mb) * 1024.0 * 1024.0)


def format_bytes_mb(value: int) -> str:
    return f"{float(value) / (1024.0 * 1024.0):.2f} MB"


def enforce_history_memory_budget(estimate: HistoryMemoryEstimate) -> None:
    if estimate.estimated_peak_bytes <= estimate.limit_bytes:
        return
    raise SimulationMemoryBudgetError(
        "Estimated simulation history memory exceeds the active budget: "
        f"estimated_peak={format_bytes_mb(estimate.estimated_peak_bytes)}, "
        f"limit={format_bytes_mb(estimate.limit_bytes)}, "
        f"samples={estimate.samples}, active_objects={estimate.active_objects}, "
        f"knowledge_pairs={estimate.knowledge_pairs}. "
        f"Raise the caller-controlled cap with {_ENV_MAX_HISTORY_MEMORY_MB} or "
        "--max-history-memory-mb, or reduce duration/dt/object count."
    )


def estimate_history_memory_from_config(
    cfg: Any,
    *,
    samples: int | None = None,
) -> HistoryMemoryEstimate:
    """Estimate retained single-run history using the engine's allocation contract.

    This stays config-only so resource preflight does not instantiate dynamics,
    controllers, bridges, or other user-provided plugins.
    """

    if samples is None:
        duration_s = float(getattr(cfg.simulator, "duration_s", 0.0) or 0.0)
        dt_s = float(getattr(cfg.simulator, "dt_s", 1.0) or 1.0)
        sample_count = _planned_step_count(duration_s, dt_s) + 1
    else:
        sample_count = int(max(samples, 0))

    active = [(str(object_id), section) for object_id, section in iter_object_sections(cfg, enabled_only=True)]
    active_ids = {object_id for object_id, _section in active}
    active_objects = len(active)
    attitude_cfg = dict(dict(getattr(cfg.simulator, "dynamics", {}) or {}).get("attitude", {}) or {})
    attitude_enabled = bool(attitude_cfg.get("enabled", True))

    float_columns = 1  # t_s
    reference_id = default_reference_object_id(cfg, available_ids=active_ids)
    reference_section = next((section for object_id, section in active if object_id == reference_id), None)
    if reference_section is not None and bool(dict(getattr(reference_section, "reference_orbit", {}) or {}).get("enabled", False)):
        float_columns += 6

    knowledge_pairs = 0
    retained_python_bytes_per_sample = 0
    rocket_count = 0
    satellite_ids: list[str] = []
    controller_debug_enabled = bool(getattr(getattr(cfg.outputs, "stats", None), "controller_debug", False))

    for object_id, section in active:
        kind = str(getattr(section, "kind", "satellite") or "satellite").strip().lower()
        is_rocket = kind == "rocket"
        if is_rocket:
            rocket_count += 1
            belief_columns = 6
        else:
            satellite_ids.append(object_id)
            belief_columns = 13 if attitude_enabled else 6

        float_columns += 14  # truth
        float_columns += belief_columns
        float_columns += 3  # thrust
        float_columns += 3  # torque
        float_columns += 4  # desired attitude
        if is_rocket:
            float_columns += 1  # throttle history
        elif controller_debug_enabled:
            retained_python_bytes_per_sample += 4096

        if getattr(cfg, "ground_segment", {}).get("enabled", False):
            retained_python_bytes_per_sample += 8192 + 4096 * len(cfg.ground_stations)
        specs = dict(getattr(section, "specs", {}) or {})
        if any(dict(specs.get(name, {}) or {}).get("enabled", False) for name in ("thermal", "power")):
            retained_python_bytes_per_sample += 4096  # resource row dict, values, cumulative ledgers
        knowledge = dict(getattr(section, "knowledge", {}) or {})
        targets = list(knowledge.get("targets", []) or [])
        knowledge_pairs += len(targets)
        float_columns += 12 * len(targets)  # estimated state and raw measurement histories

        bridge = getattr(section, "bridge", None)
        if bridge is not None and bool(getattr(bridge, "enabled", False)):
            retained_python_bytes_per_sample += 512

    if rocket_count:
        float_columns += 20  # stage/q/mach plus rocket GNC/navigation metric histories

    reentry_raw = dict(dict(getattr(cfg.simulator, "dynamics", {}) or {}).get("reentry", {}) or {})
    if bool(reentry_raw.get("enabled", False)):
        configured = reentry_raw.get("object_ids", ())
        configured_ids = [configured] if isinstance(configured, str) else list(configured or ())
        configured_ids = [str(item).strip() for item in configured_ids if str(item).strip()]
        if configured_ids and not any(item in {"*", "all"} for item in configured_ids):
            reentry_count = sum(1 for object_id in configured_ids if object_id in active_ids)
        else:
            reentry_count = len(satellite_ids)
        # Keep this local to avoid importing the dynamics stack during ordinary config parsing.
        from sim.dynamics.reentry import REENTRY_METRIC_KEYS

        float_columns += reentry_count * len(REENTRY_METRIC_KEYS)

    itemsize = 8
    array_bytes = int(sample_count * float_columns * itemsize) + int(
        sample_count * retained_python_bytes_per_sample
    )
    estimated_peak_bytes = int(array_bytes * 2)
    return HistoryMemoryEstimate(
        samples=sample_count,
        active_objects=max(active_objects, 1),
        knowledge_pairs=knowledge_pairs,
        array_bytes=array_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        limit_bytes=bytes_from_mb(configured_history_memory_limit_mb(cfg)),
    )


def resource_profile(name: str | None) -> ResourceProfile:
    key = str(name or "config").strip().lower()
    if not key:
        key = "config"
    if key not in RESOURCE_PROFILES:
        available = ", ".join(sorted(RESOURCE_PROFILES))
        raise ValueError(f"Unknown resource profile {name!r}. Available: {available}")
    return RESOURCE_PROFILES[key]


def _dict_section(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if isinstance(value, dict):
        return dict(value)
    return {}


def apply_resource_profile_to_config_dict(root: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    """Return a config copy with conservative resource-profile rewrites applied."""
    profile = resource_profile(profile_name)
    out = deepcopy(dict(root))
    if profile.name == "off":
        simulator = _dict_section(out, "simulator")
        simulator["resource_profile"] = "off"
        out["simulator"] = simulator
        outputs = _dict_section(out, "outputs")
        resource_limits = _dict_section(outputs, "resource_limits")
        resource_limits["checkpoint_enabled"] = False
        resource_limits["throttle_enabled"] = False
        outputs["resource_limits"] = resource_limits
        mc_outputs = _dict_section(outputs, "monte_carlo")
        mc_outputs["checkpoint_enabled"] = False
        outputs["monte_carlo"] = mc_outputs
        out["outputs"] = outputs
        return out

    simulator = _dict_section(out, "simulator")
    simulator["resource_profile"] = profile.name
    simulator_execution = _dict_section(simulator, "execution")
    object_parallelism = _dict_section(simulator_execution, "object_parallelism")
    if profile.force_serial:
        if str(simulator_execution.get("policy", "configured") or "configured").strip().lower() != "parallel":
            simulator_execution["policy"] = "serial"
        object_parallelism["enabled"] = False
        object_parallelism["backend"] = "serial"
        object_parallelism["workers"] = 1
    elif profile.max_parallel_workers is not None and (
        object_parallelism.get("enabled", False)
        or str(simulator_execution.get("policy", "")).strip().lower() in {"auto", "parallel"}
    ):
        current = int(object_parallelism.get("workers", 0) or 0)
        if current <= 0:
            object_parallelism["workers"] = int(profile.max_parallel_workers)
        else:
            object_parallelism["workers"] = int(max(1, min(current, profile.max_parallel_workers)))
    if object_parallelism:
        simulator_execution["object_parallelism"] = object_parallelism
        simulator["execution"] = simulator_execution
    out["simulator"] = simulator

    outputs = _dict_section(out, "outputs")
    resource_limits = _dict_section(outputs, "resource_limits")
    resource_limits.setdefault("checkpoint_enabled", bool(profile.checkpoint_enabled))
    resource_limits.setdefault("throttle_enabled", bool(profile.throttle_enabled))
    if profile.min_available_memory_mb is not None:
        resource_limits.setdefault("min_available_memory_mb", float(profile.min_available_memory_mb))
    if profile.hard_min_available_memory_mb is not None:
        resource_limits.setdefault("hard_min_available_memory_mb", float(profile.hard_min_available_memory_mb))
    if profile.max_load_per_cpu is not None:
        resource_limits.setdefault("max_load_per_cpu", float(profile.max_load_per_cpu))
    resource_limits.setdefault("resource_pause_seconds", float(profile.pause_seconds))
    resource_limits.setdefault("resource_max_wait_s", float(profile.max_wait_s))
    outputs["resource_limits"] = resource_limits

    if profile.disable_plots:
        plots = _dict_section(outputs, "plots")
        plots["enabled"] = False
        outputs["plots"] = plots

    mc_outputs = _dict_section(outputs, "monte_carlo")
    mc_outputs.setdefault("checkpoint_enabled", bool(profile.checkpoint_enabled))
    outputs["monte_carlo"] = mc_outputs
    out["outputs"] = outputs

    mc = _dict_section(out, "monte_carlo")
    if mc.get("enabled", False):
        if profile.force_serial:
            mc["parallel_enabled"] = False
            mc["parallel_workers"] = 1
        elif profile.max_parallel_workers is not None:
            current = int(mc.get("parallel_workers", 0) or 0)
            if current <= 0:
                mc["parallel_workers"] = int(profile.max_parallel_workers)
            else:
                mc["parallel_workers"] = int(max(1, min(current, profile.max_parallel_workers)))
        out["monte_carlo"] = mc

    analysis = _dict_section(out, "analysis")
    execution = _dict_section(analysis, "execution")
    if analysis.get("enabled", False):
        if profile.force_serial:
            execution["parallel_enabled"] = False
            execution["parallel_workers"] = 1
        elif profile.max_parallel_workers is not None:
            current = int(execution.get("parallel_workers", 0) or 0)
            if current <= 0:
                execution["parallel_workers"] = int(profile.max_parallel_workers)
            else:
                execution["parallel_workers"] = int(max(1, min(current, profile.max_parallel_workers)))
        analysis["execution"] = execution
        out["analysis"] = analysis
    return out


def _read_proc_mem_available_mb() -> tuple[float | None, float | None]:
    meminfo = "/proc/meminfo"
    if not os.path.exists(meminfo):
        return None, None
    values: dict[str, float] = {}
    try:
        with open(meminfo, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    values[parts[0].rstrip(":")] = float(parts[1]) / 1024.0
    except OSError:
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _run_text(command: list[str], timeout_s: float = 1.0) -> str:
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
    except Exception:
        return ""
    return proc.stdout.strip()


def _read_macos_memory_mb() -> tuple[float | None, float | None]:
    if platform.system() != "Darwin":
        return None, None
    total_mb: float | None = None
    total_text = _run_text(["sysctl", "-n", "hw.memsize"])
    try:
        total_mb = float(total_text) / (1024.0 * 1024.0) if total_text else None
    except ValueError:
        total_mb = None

    vm_text = _run_text(["vm_stat"])
    if not vm_text:
        return total_mb, None
    parsed_total_mb, available_mb = _parse_macos_vm_stat_mb(vm_text)
    if total_mb is None:
        total_mb = parsed_total_mb
    return total_mb, available_mb


def _parse_macos_memory_pressure(text: str) -> tuple[float | None, float | None]:
    total_mb: float | None = None
    free_percent: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("The system has "):
            parts = stripped.split()
            try:
                total_bytes = int(parts[3])
            except (IndexError, ValueError):
                continue
            total_mb = float(total_bytes) / (1024.0 * 1024.0)
        elif stripped.startswith("System-wide memory free percentage:"):
            raw = stripped.partition(":")[2].strip().rstrip("%")
            try:
                free_percent = float(raw)
            except ValueError:
                continue
    return total_mb, free_percent


def _read_macos_memory_pressure() -> tuple[float | None, float | None]:
    if platform.system() != "Darwin":
        return None, None
    text = _run_text(["memory_pressure", "-Q"])
    if not text:
        return None, None
    return _parse_macos_memory_pressure(text)


def _parse_macos_vm_stat_mb(vm_text: str) -> tuple[float | None, float | None]:
    page_size = 4096.0
    values: dict[str, float] = {}
    for line in vm_text.splitlines():
        if "page size of" in line:
            words = [w for w in line.replace(".", "").split() if w.isdigit()]
            if words:
                page_size = float(words[-1])
            continue
        label, _, raw_value = line.partition(":")
        key = label.strip()
        value_text = raw_value.strip().rstrip(".").replace(",", "")
        try:
            values[key] = float(value_text)
        except ValueError:
            continue

    # macOS keeps useful cache in inactive/speculative pages. Counting only
    # "Pages free" makes ordinary desktops look exhausted and causes false
    # resource-preflight pressure for runs that only need modest memory.
    available_pages = sum(
        values.get(key, 0.0)
        for key in (
            "Pages free",
            "Pages speculative",
            "Pages inactive",
            "Pages purgeable",
        )
    )
    total_pages = sum(
        values.get(key, 0.0)
        for key in (
            "Pages free",
            "Pages active",
            "Pages inactive",
            "Pages speculative",
            "Pages wired down",
            "Pages occupied by compressor",
        )
    )
    available_mb = (available_pages * page_size) / (1024.0 * 1024.0) if available_pages > 0 else None
    total_mb = (total_pages * page_size) / (1024.0 * 1024.0) if total_pages > 0 else None
    return total_mb, available_mb


def _memory_bytes_to_mb(total_bytes: int, available_bytes: int) -> tuple[float, float]:
    return float(total_bytes) / (1024.0 * 1024.0), float(available_bytes) / (1024.0 * 1024.0)


def _read_windows_memory_mb() -> tuple[float | None, float | None]:
    if platform.system() != "Windows":
        return None, None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if not ok:
            return None, None
        return _memory_bytes_to_mb(int(status.ullTotalPhys), int(status.ullAvailPhys))
    except Exception:
        return None, None


def current_resource_snapshot() -> ResourceSnapshot:
    cpu_count = int(os.cpu_count() or 1)
    load_1m: float | None = None
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = None
    memory_pressure_free_percent = None
    total_mb, available_mb = _read_proc_mem_available_mb()
    if total_mb is None and available_mb is None:
        total_mb, available_mb = _read_macos_memory_mb()
        pressure_total_mb, memory_pressure_free_percent = _read_macos_memory_pressure()
        if pressure_total_mb is not None:
            total_mb = pressure_total_mb
        if total_mb is not None and memory_pressure_free_percent is not None:
            available_mb = total_mb * memory_pressure_free_percent / 100.0
    if total_mb is None and available_mb is None:
        total_mb, available_mb = _read_windows_memory_mb()
    return ResourceSnapshot(
        timestamp_s=time.monotonic(),
        cpu_count=cpu_count,
        load_1m=load_1m,
        total_memory_mb=total_mb,
        available_memory_mb=available_mb,
        memory_pressure_free_percent=memory_pressure_free_percent,
    )


def _resource_limits_dict(cfg: Any) -> dict[str, Any]:
    limits = dict(getattr(getattr(cfg, "outputs", None), "resource_limits", {}) or {})
    simulator_profile = getattr(getattr(cfg, "simulator", None), "resource_profile", None)
    if simulator_profile not in (None, ""):
        limits["resource_profile"] = str(simulator_profile)
    return limits


def checkpoint_enabled(cfg: Any, mc_out_cfg: dict[str, Any] | None = None) -> bool:
    mc_out = dict(mc_out_cfg or {})
    if "checkpoint_enabled" in mc_out:
        return bool(mc_out.get("checkpoint_enabled"))
    limits = _resource_limits_dict(cfg)
    if "checkpoint_enabled" in limits:
        return bool(limits.get("checkpoint_enabled"))
    return bool(resource_profile(limits.get("resource_profile", "config")).checkpoint_enabled)


class ResourceGovernor:
    def __init__(self, cfg: Any, *, emit: Any = None) -> None:
        limits = _resource_limits_dict(cfg)
        profile = resource_profile(limits.get("resource_profile", "config"))
        self.profile = profile
        self.enabled = bool(limits.get("throttle_enabled", profile.throttle_enabled))
        self.min_available_memory_mb = _optional_float(
            limits.get("min_available_memory_mb"), profile.min_available_memory_mb
        )
        self.hard_min_available_memory_mb = _optional_float(
            limits.get("hard_min_available_memory_mb"), profile.hard_min_available_memory_mb
        )
        self.max_load_per_cpu = _optional_float(limits.get("max_load_per_cpu"), profile.max_load_per_cpu)
        pause_seconds = limits.get("resource_pause_seconds", profile.pause_seconds)
        if pause_seconds in (None, ""):
            pause_seconds = profile.pause_seconds
        max_wait_s = limits.get("resource_max_wait_s", profile.max_wait_s)
        if max_wait_s in (None, ""):
            max_wait_s = profile.max_wait_s
        if isinstance(pause_seconds, bool) or isinstance(max_wait_s, bool):
            raise ValueError("resource wait settings must be numeric, not boolean.")
        self.pause_seconds = float(pause_seconds)
        self.max_wait_s = float(max_wait_s)
        if not math.isfinite(self.pause_seconds) or self.pause_seconds <= 0.0:
            raise ValueError("outputs.resource_limits.resource_pause_seconds must be positive and finite.")
        if not math.isfinite(self.max_wait_s) or self.max_wait_s < 0.0:
            raise ValueError("outputs.resource_limits.resource_max_wait_s must be nonnegative and finite.")
        self.emit = emit
        estimate = estimate_resource_requirements(cfg)
        self.estimated_incremental_memory_mb = float(estimate.estimated_incremental_memory_mb)
        self.wait_event_count = 0
        self.total_wait_s = 0.0

    def projected_available_memory_mb(self, snapshot: ResourceSnapshot) -> float | None:
        if snapshot.available_memory_mb is None:
            return None
        return float(snapshot.available_memory_mb) - self.estimated_incremental_memory_mb

    def pressure_reasons(self, snapshot: ResourceSnapshot, *, include_load: bool = True) -> list[str]:
        reasons: list[str] = []
        projected_mb = self.projected_available_memory_mb(snapshot)
        if self.min_available_memory_mb is not None and projected_mb is not None:
            if projected_mb < self.min_available_memory_mb:
                reasons.append(
                    f"projected post-start headroom {projected_mb:.0f} MB < {self.min_available_memory_mb:.0f} MB"
                )
        if (
            snapshot.memory_pressure_free_percent is not None
            and snapshot.memory_pressure_free_percent < MACOS_MEMORY_PRESSURE_WARN_PERCENT
        ):
            reasons.append(
                f"macOS memory-pressure free percentage {snapshot.memory_pressure_free_percent:.0f}% "
                f"< {MACOS_MEMORY_PRESSURE_WARN_PERCENT:.0f}%"
            )
        if include_load and self.max_load_per_cpu is not None and snapshot.load_per_cpu is not None:
            if snapshot.load_per_cpu > self.max_load_per_cpu:
                reasons.append(f"load/core {snapshot.load_per_cpu:.2f} > {self.max_load_per_cpu:.2f}")
        return reasons

    def assert_safe_to_start(self) -> ResourceSnapshot:
        snapshot = current_resource_snapshot()
        if not self.enabled:
            return snapshot
        if (
            (self.min_available_memory_mb is not None or self.hard_min_available_memory_mb is not None)
            and snapshot.available_memory_mb is None
            and snapshot.memory_pressure_free_percent is None
        ):
            raise ResourcePressureError(
                "Memory telemetry is unavailable, so the configured memory safety floor cannot be enforced."
            )
        if (
            snapshot.memory_pressure_free_percent is not None
            and snapshot.memory_pressure_free_percent < MACOS_MEMORY_PRESSURE_CRITICAL_PERCENT
        ):
            raise ResourcePressureError(
                "macOS reports critical memory pressure: "
                f"free={snapshot.memory_pressure_free_percent:.0f}%, "
                f"critical={MACOS_MEMORY_PRESSURE_CRITICAL_PERCENT:.0f}%."
            )
        projected_mb = self.projected_available_memory_mb(snapshot)
        if (
            self.hard_min_available_memory_mb is not None
            and projected_mb is not None
            and projected_mb < self.hard_min_available_memory_mb
        ):
            raise ResourcePressureError(
                "Projected post-start memory headroom is below the hard safety floor: "
                f"available={snapshot.available_memory_mb:.0f} MB, "
                f"estimated_increment={self.estimated_incremental_memory_mb:.1f} MB, "
                f"projected={projected_mb:.0f} MB, "
                f"floor={self.hard_min_available_memory_mb:.0f} MB."
            )
        return snapshot

    def wait_for_capacity(self, *, context: str = "batch run", include_load: bool = True) -> ResourceSnapshot:
        snapshot = self.assert_safe_to_start()
        if not self.enabled:
            return snapshot
        started = time.monotonic()
        warned = False
        while True:
            reasons = self.pressure_reasons(snapshot, include_load=include_load)
            if not reasons:
                if warned:
                    waited_s = max(time.monotonic() - started, 0.0)
                    self.wait_event_count += 1
                    self.total_wait_s += waited_s
                    if self.emit is not None:
                        self.emit(f"Resuming {context} after {waited_s:.1f}s of resource waiting.")
                return snapshot
            if time.monotonic() - started >= self.max_wait_s:
                raise ResourcePressureError(
                    f"Resource pressure remained high before {context}: " + "; ".join(reasons)
                )
            if self.emit is not None and not warned:
                self.emit(
                    f"Throttling {context}: " + "; ".join(reasons) + f"; waiting {self.pause_seconds:.0f}s."
                )
                warned = True
            time.sleep(max(float(self.pause_seconds), 0.1))
            snapshot = self.assert_safe_to_start()

    def telemetry(self) -> dict[str, Any]:
        return {
            "wait_event_count": int(self.wait_event_count),
            "total_wait_s": float(self.total_wait_s),
            "estimated_incremental_memory_mb": float(self.estimated_incremental_memory_mb),
        }


def _optional_float(value: Any, default: float | None) -> float | None:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ValueError("resource limit values must be numeric, not boolean.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("resource limit values must be finite.")
    if parsed < 0.0:
        raise ValueError("resource limit values must be nonnegative.")
    return parsed


def _active_object_count(cfg: Any) -> int:
    object_ids = {str(object_id) for object_id, _agent in iter_object_sections(cfg, enabled_only=True)}
    return max(len(object_ids), 1)


def _study_type(cfg: Any) -> str:
    analysis = getattr(cfg, "analysis", None)
    if bool(getattr(analysis, "enabled", False)):
        return str(getattr(analysis, "study_type", "monte_carlo") or "monte_carlo").strip().lower()
    if bool(getattr(getattr(cfg, "monte_carlo", None), "enabled", False)):
        return "monte_carlo"
    return "single_run"


def _sensitivity_run_count(cfg: Any) -> int:
    sensitivity = getattr(getattr(cfg, "analysis", None), "sensitivity", None)
    method = str(getattr(sensitivity, "method", "one_at_a_time") or "one_at_a_time").strip().lower()
    params = list(getattr(sensitivity, "parameters", []) or [])
    if method == "lhs":
        return max(int(getattr(sensitivity, "samples", 0) or 0), 0)
    if method == "two_parameter_grid" and len(params) == 2:
        return int(len(list(getattr(params[0], "values", []) or [])) * len(list(getattr(params[1], "values", []) or [])))
    return int(sum(len(list(getattr(param, "values", []) or [])) for param in params))


def _incremental_memory_mb(
    *,
    history_mb_per_run: float,
    effective_workers: int,
    plots_enabled: bool,
) -> float:
    workers = max(int(effective_workers), 1)
    worker_overhead_mb = float(max(workers - 1, 0)) * DEFAULT_PARALLEL_WORKER_OVERHEAD_MB
    plot_overhead_mb = DEFAULT_PLOT_OVERHEAD_MB if plots_enabled else 0.0
    return float(history_mb_per_run) * float(workers) + worker_overhead_mb + plot_overhead_mb


def _metric_gate_entries(gates: Any) -> list[dict[str, Any]]:
    if isinstance(gates, list):
        return [dict(item) for item in gates if isinstance(item, dict)]
    if not isinstance(gates, dict):
        return []
    entries: list[dict[str, Any]] = []
    for key in ("metric_gates", "metrics", "generic"):
        raw = gates.get(key)
        if isinstance(raw, list):
            entries.extend(dict(item) for item in raw if isinstance(item, dict))
    return entries


def _monte_carlo_retains_full_payloads(cfg: Any) -> bool:
    monte_carlo_outputs = dict(getattr(getattr(cfg, "outputs", None), "monte_carlo", {}) or {})
    for gate in _metric_gate_entries(monte_carlo_outputs.get("gates", {}) or {}):
        metric_path = str(gate.get("metric", "") or gate.get("path", "") or "").strip()
        if metric_path.startswith("payload."):
            return True
        if metric_path in {"derived.final_altitude_km_min", "derived.final_altitude_km_max"}:
            return True
    return False


def _retained_payload_run_count(cfg: Any, *, study_type: str, runs: int) -> int:
    if study_type == "monte_carlo":
        return max(int(runs), 0) if _monte_carlo_retains_full_payloads(cfg) else 0
    if study_type != "sensitivity":
        return 0
    retained_runs = max(int(runs), 0)
    baseline = getattr(getattr(cfg, "analysis", None), "baseline", None)
    baseline_mode = str(getattr(baseline, "mode", "none") or "none").strip().lower()
    if baseline_mode == "run" or (baseline_mode == "none" and bool(getattr(baseline, "enabled", False))):
        retained_runs += 1
    return retained_runs


def _raise_risk(current: str, candidate: str) -> str:
    order = {"safe": 0, "moderate": 1, "heavy": 2, "unsafe": 3}
    return candidate if order[candidate] > order[current] else current


def estimate_resource_requirements(cfg: Any) -> ResourceEstimate:
    study_type = _study_type(cfg)
    steps = _planned_step_count(
        float(getattr(cfg.simulator, "duration_s", 0.0)),
        float(getattr(cfg.simulator, "dt_s", 1.0)),
    )
    if study_type == "monte_carlo":
        runs = int(getattr(cfg.monte_carlo, "iterations", 1))
        parallel_enabled = bool(getattr(cfg.monte_carlo, "parallel_enabled", False))
        requested_workers = int(getattr(cfg.monte_carlo, "parallel_workers", 0) or 0) if parallel_enabled else 1
    elif study_type == "sensitivity":
        runs = max(_sensitivity_run_count(cfg), 1)
        execution = getattr(getattr(cfg, "analysis", None), "execution", None)
        parallel_enabled = bool(getattr(execution, "parallel_enabled", False))
        requested_workers = int(getattr(execution, "parallel_workers", 0) or 0) if parallel_enabled else 1
    else:
        runs = 1
        parallel_enabled = False
        requested_workers = 1
    if parallel_enabled and requested_workers <= 0:
        requested_workers = max(1, int(os.cpu_count() or 1) - 1)
    effective_workers = max(1, min(requested_workers, max(runs, 1)))
    limits = _resource_limits_dict(cfg)
    profile = resource_profile(limits.get("resource_profile", "config"))
    if profile.force_serial:
        effective_workers = 1
    elif profile.max_parallel_workers is not None:
        effective_workers = max(1, min(effective_workers, int(profile.max_parallel_workers)))
    active_objects = _active_object_count(cfg)
    hierarchical_processes = effective_workers
    object_workers_per_run = 0
    prepared_config_memory_mb = 0.0
    if study_type in {"monte_carlo", "sensitivity"} and runs > 0:
        try:
            from sim.execution.hierarchical import plan_hierarchical_execution
        except ModuleNotFoundError as exc:
            if exc.name != "sim.execution.hierarchical":
                raise
        else:
            config_root = cfg.to_dict() if callable(getattr(cfg, "to_dict", None)) else {}
            serialized_root_bytes = len(
                json.dumps(config_root, sort_keys=True, default=str).encode("utf-8")
            )
            # Estimate retained campaign configuration memory arithmetically.
            # Resource preflight must remain bounded and must not materialize
            # every deep-copied trial before it can refuse an oversized study.
            prepared_config_memory_mb = (
                serialized_root_bytes * max(runs, 1) / (1024.0 * 1024.0)
            )
            hierarchy = plan_hierarchical_execution(
                task_roots=(config_root,),
                task_count=max(runs, 1),
                requested_campaign_workers=requested_workers if parallel_enabled else 1,
                profile=profile,
            )
            effective_workers = int(hierarchy.campaign_workers)
            object_workers_per_run = int(hierarchy.object_workers_per_run)
            hierarchical_processes = effective_workers * (1 + object_workers_per_run)
    history_estimate = estimate_history_memory_from_config(cfg)
    estimated_history_mb_per_run = history_estimate.estimated_peak_mb
    estimated_parallel_history_mb = estimated_history_mb_per_run * hierarchical_processes
    plots_enabled = bool(getattr(getattr(cfg, "outputs", None), "plots", {}).get("enabled", True))
    estimated_incremental_memory_mb = _incremental_memory_mb(
        history_mb_per_run=estimated_history_mb_per_run,
        effective_workers=hierarchical_processes,
        plots_enabled=plots_enabled,
    )
    estimated_incremental_memory_mb += prepared_config_memory_mb
    retained_payload_runs = _retained_payload_run_count(cfg, study_type=study_type, runs=runs)
    if retained_payload_runs:
        estimated_incremental_memory_mb += estimated_history_mb_per_run * retained_payload_runs
    snapshot = current_resource_snapshot()
    projected_available_memory_mb = (
        None
        if snapshot.available_memory_mb is None
        else float(snapshot.available_memory_mb) - estimated_incremental_memory_mb
    )
    acceleration = acceleration_settings_from_config(cfg)
    min_available_mb = _optional_float(limits.get("min_available_memory_mb"), profile.min_available_memory_mb)
    hard_min_available_mb = _optional_float(
        limits.get("hard_min_available_memory_mb"), profile.hard_min_available_memory_mb
    )
    max_load_per_cpu = _optional_float(limits.get("max_load_per_cpu"), profile.max_load_per_cpu)
    notes: list[str] = []
    if prepared_config_memory_mb > 0.0:
        notes.append(f"prepared campaign configs retain approximately {prepared_config_memory_mb:.2f} MB")
    risk = "safe"
    if retained_payload_runs:
        notes.append(f"full run payloads retained for {retained_payload_runs} batch runs")
    if runs >= 5 or steps >= 10000:
        risk = _raise_risk(risk, "moderate")
        notes.append("long campaign envelope")
    if hierarchical_processes >= 3:
        risk = _raise_risk(risk, "heavy")
        notes.append("three or more concurrent workers")
    if object_workers_per_run:
        notes.append(
            f"hierarchical execution may launch {effective_workers} campaign workers plus "
            f"{object_workers_per_run} object workers per active run "
            f"({hierarchical_processes} worker processes total)"
        )
    if plots_enabled and runs > 1:
        notes.append("plots enabled for a batch workflow")
    if (
        snapshot.memory_pressure_free_percent is not None
        and snapshot.memory_pressure_free_percent < MACOS_MEMORY_PRESSURE_CRITICAL_PERCENT
    ):
        risk = "unsafe"
        notes.append("macOS reports critical memory pressure")
    elif (
        snapshot.memory_pressure_free_percent is not None
        and snapshot.memory_pressure_free_percent < MACOS_MEMORY_PRESSURE_WARN_PERCENT
    ):
        risk = _raise_risk(risk, "heavy")
        notes.append("macOS reports elevated memory pressure")
    if (
        (min_available_mb is not None or hard_min_available_mb is not None)
        and snapshot.available_memory_mb is None
        and snapshot.memory_pressure_free_percent is None
    ):
        risk = "unsafe"
        notes.append("memory telemetry is unavailable for the configured safety floor")
    if hard_min_available_mb is not None and projected_available_memory_mb is not None:
        if projected_available_memory_mb < hard_min_available_mb:
            risk = "unsafe"
            notes.append("projected post-start memory is below the hard safety floor")
    if min_available_mb is not None and projected_available_memory_mb is not None:
        if projected_available_memory_mb < min_available_mb and risk != "unsafe":
            risk = _raise_risk(risk, "heavy")
            notes.append("projected post-start memory is below the profile's preferred headroom")
    if (
        hierarchical_processes > 1
        and max_load_per_cpu is not None
        and snapshot.load_per_cpu is not None
        and snapshot.load_per_cpu > max_load_per_cpu
    ):
        risk = _raise_risk(risk, "heavy")
        notes.append("current system load is elevated for a parallel launch")
    return ResourceEstimate(
        profile=str(_resource_limits_dict(cfg).get("resource_profile", "config") or "config"),
        study_type=study_type,
        runs=runs,
        steps_per_run=steps,
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        active_objects=active_objects,
        plots_enabled=plots_enabled,
        checkpoint_enabled=checkpoint_enabled(cfg),
        estimated_history_mb_per_run=estimated_history_mb_per_run,
        estimated_parallel_history_mb=estimated_parallel_history_mb,
        estimated_incremental_memory_mb=estimated_incremental_memory_mb,
        current_available_memory_mb=snapshot.available_memory_mb,
        projected_available_memory_mb=projected_available_memory_mb,
        memory_pressure_free_percent=snapshot.memory_pressure_free_percent,
        load_per_cpu=snapshot.load_per_cpu,
        acceleration_mode=acceleration.requested_mode,
        acceleration_backend=acceleration.effective_backend,
        risk=risk,
        notes=tuple(notes),
        hierarchical_processes=hierarchical_processes,
        object_workers_per_run=object_workers_per_run,
    )
