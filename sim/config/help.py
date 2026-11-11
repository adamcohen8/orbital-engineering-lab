from __future__ import annotations

import importlib
import re
import textwrap
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from sim.config.profiles import profile_choices

DEFAULT_HELP_SCOPE = "all"


@dataclass(frozen=True)
class ConfigOption:
    value: str
    description: str


@dataclass(frozen=True)
class ConfigHelpEntry:
    path: str
    title: str
    description: str
    options: tuple[ConfigOption, ...] = ()
    option_loader: Callable[[], tuple[ConfigOption, ...]] | None = None
    aliases: tuple[str, ...] = ()
    example: str = ""
    notes: tuple[str, ...] = ()
    visibility: str = "public"

    def resolved_options(self) -> tuple[ConfigOption, ...]:
        if self.option_loader is None:
            return self.options
        return self.option_loader()

    def search_terms(self) -> tuple[str, ...]:
        return (self.path, self.title, *self.aliases)


def _option(value: str, description: str) -> ConfigOption:
    return ConfigOption(value=value, description=description)


def _scope_allows(entry: ConfigHelpEntry, scope: str) -> bool:
    normalized = str(scope or DEFAULT_HELP_SCOPE).strip().lower()
    if normalized == "all":
        return True
    if normalized == "pro":
        return entry.visibility in {"public", "pro"}
    return entry.visibility == "public"


def _figure_id_options() -> tuple[ConfigOption, ...]:
    from sim.master_outputs import AVAILABLE_FIGURE_IDS

    descriptions = {
        "run_dashboard": "Single-run summary dashboard.",
        "rendezvous_summary": "Rendezvous-focused relative motion summary.",
        "rendezvous_summary_curvilinear": "Curvilinear-RIC rendezvous summary with range/speed and delta-v.",
        "ground_track_multi": "Multi-object ground tracks.",
        "trajectory_ric_curv_2d_multi": "Multi-object curvilinear-RIC 2D relative trajectory.",
        "trajectory_ric_rect_2d_multi_target_burns": "Rectangular-RIC 2D trajectory with target-burn markers.",
        "trajectory_ric_curv_2d_multi_target_burns": "Curvilinear-RIC 2D trajectory with target-burn markers.",
        "relative_range": "Range between relevant objects over time.",
        "control_effort": "Control effort or delta-v usage over time.",
        "orbital_elements_summary": "Six-panel classical orbital element history.",
        "satellite_delta_v_remaining": "Remaining satellite delta-v budget over time.",
        "ground_station_access": "Ground-station access, elevation, and range.",
    }
    return tuple(
        _option(figure_id, descriptions.get(figure_id, "Generated single-run plot artifact."))
        for figure_id in AVAILABLE_FIGURE_IDS
    )


def _plot_preset_options() -> tuple[ConfigOption, ...]:
    from sim.master_outputs import PLOT_PRESETS

    descriptions = {
        "minimal": "Smallest useful plot bundle.",
        "orbit": "Orbit trajectory, ground track, and orbital-elements plots.",
        "rendezvous": "RPO-oriented relative trajectory, range, and control plots.",
        "attitude": "Attitude state, rates, error, and control summary.",
        "estimation": "Estimator, knowledge, and sensor-access diagnostics.",
        "access": "Ground-station access and ground-track plots.",
        "rocket": "Rocket ascent, GNC, insertion, and resource diagnostics.",
        "debug": "Every available figure ID.",
    }
    return tuple(_option(name, descriptions.get(name, ", ".join(ids))) for name, ids in sorted(PLOT_PRESETS.items()))


def _animation_type_options() -> tuple[ConfigOption, ...]:
    from sim.master_outputs import AVAILABLE_ANIMATION_TYPES

    return tuple(_option(name, "Animation artifact type.") for name in AVAILABLE_ANIMATION_TYPES)


def _profile_options() -> tuple[ConfigOption, ...]:
    descriptions = {
        "fast": "Lower-cost settings for interactive experiments.",
        "ops": "Default operational balance of cost and fidelity.",
        "high_fidelity": "Tighter integration settings for validation-style runs.",
    }
    return tuple(_option(name, descriptions.get(name, "Simulation fidelity profile.")) for name in profile_choices())


def _gnc_builtin_options(*categories: str) -> tuple[ConfigOption, ...]:
    """Load public built-in choices from the authoritative GNC catalog."""
    from sim.gnc import catalog_entries

    selected = set(categories)
    return tuple(
        _option(
            entry.builtin_id,
            f"{entry.display_name} [{entry.maturity}] — {entry.summary}",
        )
        for entry in catalog_entries(include_internal=False)
        if entry.category in selected
    )


def _orbit_control_builtin_options() -> tuple[ConfigOption, ...]:
    return _gnc_builtin_options("orbit_controller", "orbit_wrapper")


def _attitude_control_builtin_options() -> tuple[ConfigOption, ...]:
    return _gnc_builtin_options("attitude_controller")


def _mission_strategy_builtin_options() -> tuple[ConfigOption, ...]:
    return _gnc_builtin_options("mission_strategy")


def _mission_execution_builtin_options() -> tuple[ConfigOption, ...]:
    return _gnc_builtin_options("mission_execution")


def _flight_software_profile_options() -> tuple[ConfigOption, ...]:
    from sim.flight_software.profiles import use_case_profiles

    return tuple(
        _option(
            profile.profile_id,
            f"{profile.display_name} [{profile.maturity.value}/{profile.qualification_status}] — {profile.summary}",
        )
        for profile in use_case_profiles()
    )


def _discover_named_presets(module_name: str, type_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    preset_type = getattr(module, type_name)
    return sorted(name for name, value in vars(module).items() if name.isupper() and isinstance(value, preset_type))


def _satellite_preset_options() -> tuple[ConfigOption, ...]:
    return tuple(
        _option(name, "Reusable satellite hardware preset.")
        for name in _discover_named_presets("sim.presets.satellites", "SatellitePreset")
    )


def _rocket_preset_options() -> tuple[ConfigOption, ...]:
    return tuple(
        _option(name, "Reusable rocket stack preset.")
        for name in _discover_named_presets("sim.presets.rockets", "RocketStackPreset")
    )


CONFIG_HELP_ENTRIES: tuple[ConfigHelpEntry, ...] = (
    ConfigHelpEntry(
        path="simulator.environment.ephemeris_mode",
        title="Ephemeris Mode",
        description="Controls how Sun/Moon and other time-dependent ephemeris positions are resolved.",
        aliases=("ephemeris model", "emphemeris model", "sun moon model", "environment ephemeris"),
        options=(
            _option("analytic_enhanced", "Default low-cost analytic Sun/Moon model with improved approximations."),
            _option(
                "analytic_simple", "Older lightweight analytic Sun/Moon model; useful for repeatable legacy cases."
            ),
            _option("de440", "Use the configured OEL DE440 coefficient product."),
            _option("spice", "Use spiceypy and configured SPICE kernels, or env['spice_ephemeris_callable']."),
        ),
        example='simulator:\n  environment:\n    ephemeris_mode: "analytic_enhanced"',
        notes=("Aliases accepted by lower-level code include simple/enhanced and spiceypy/callable forms.",),
    ),
    ConfigHelpEntry(
        path="simulator.frames.model",
        title="Frame Model",
        description=(
            "Selects the scenario-level Earth-fixed frame policy used by frame-sensitive geometry, "
            "ground access, plotting, drag density rotation, and spherical harmonics defaults."
        ),
        aliases=(
            "frame model",
            "earth fixed frame",
            "eci ecef",
            "teme",
            "eop",
            "hpop frame",
            "itrf",
            "gcrf",
        ),
        options=(
            _option("simple_gmst", "Default GMST/simple Earth-rotation model for lightweight deterministic runs."),
            _option(
                "iau76_80_eop",
                "IAU-76/FK5 plus IAU-80 nutation and EOP-backed polar motion for HPOP-parity validation cases.",
            ),
        ),
        example=(
            "simulator:\n"
            "  frames:\n"
            "    model: iau76_80_eop\n"
            "    eop_path: validation/resources/hpop/eop19620101.txt"
        ),
        notes=(
            "The run payload and review store record frame_provenance for auditability.",
            "Legacy per-force frame_model/eop_path settings remain accepted but scenario-level frames are preferred.",
        ),
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.orbit.model",
        title="Orbit Base Model",
        description=(
            "Names the base orbit propagation model. Two-body is the default Earth-centered model; "
            "CR3BP is an opt-in rotating-frame model for cislunar teaching cases. Object-level "
            "central-body overrides such as Mars `mu` are not supported."
        ),
        aliases=(
            "orbit model",
            "dynamics model",
            "propagation model",
            "central body",
            "central-body",
            "primary body",
            "non earth orbit",
            "non-earth orbit",
            "mars orbit",
            "cislunar",
            "cr3bp",
        ),
        options=(
            _option(
                "two_body",
                "Central-body Keplerian gravity baseline. Add j2, drag, SRP, or spherical_harmonics for higher fidelity.",
            ),
            _option(
                "cr3bp",
                "Circular restricted three-body propagation in a configured rotating-frame system such as earth_moon.",
            ),
        ),
        example='simulator:\n  dynamics:\n    orbit:\n      model: "two_body"\n      j2: true\n      drag: false',
        notes=(
            "Use two_body for Earth-centered public-core propagation.",
            "Use model: cr3bp only for documented rotating-frame cislunar teaching cases.",
            "Do not set object-level central_body, primary_body, or mu_km3_s2 fields; validation rejects them.",
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.runtime_profile",
        title="Satellite Runtime Profile",
        description=(
            "Selects whether a satellite runs complete onboard flight software or only its configured deterministic dynamics."
        ),
        aliases=("trajectory only", "dynamics only", "no flight software", "passive object"),
        options=(
            _option("flight_software", "Default complete-stack boundary; omitted flight_software selects fsw.passive."),
            _option("trajectory_only", "Propagate configured dynamics without sensors, knowledge, FSW tasks, devices, or commands."),
        ),
        example=(
            "objects:\n"
            "  debris:\n"
            "    kind: satellite\n"
            "    runtime_profile: trajectory_only\n"
            "    initial_state:\n"
            "      default_circular_earth: true"
        ),
        notes=(
            "trajectory_only is explicit; omitting runtime_profile preserves the fsw.passive compatibility default.",
            "A trajectory-only object cannot declare flight_software or knowledge, but other objects may observe it.",
            "The configured ONP/OGP propagation family and physical force models remain authoritative.",
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.propagation_method",
        title="Propagation Method",
        description=(
            "Selects an object's propagation family. Special perturbations use OEL numerical dynamics; "
            "general perturbations currently support passive SGP4 propagation from TLEs."
        ),
        aliases=(
            "sgp4",
            "general perturbations",
            "special perturbations",
            "catalog propagation",
            "tle propagation",
            "propagation method",
        ),
        options=(
            _option("special", "Default OEL numerical propagation with configured force models and controls."),
            _option("general", "Passive catalog-style propagation; v1 supports general.model: sgp4 with TLE input."),
        ),
        example=(
            "objects:\n"
            "  catalog_object:\n"
            "    propagation_method: general\n"
            "    general:\n"
            "      model: sgp4\n"
            "    initial_state:\n"
            "      tle:\n"
            "        line1: \"1 ...\"\n"
            "        line2: \"2 ...\""
        ),
        notes=(
            "SGP4 objects are passive in v1: no orbit_control, attitude_control, thrust, mission objectives, or maneuvers.",
            "TLEs with orbital period >= 225 minutes route through the supported OGP-SDP4 deep-space/resonance path.",
            "The default general.output_frame is native TEME; set teme_as_eci only for the documented legacy ECI-compatible approximation.",
            "Use frame_transform: teme_to_eci_iau80 for the opt-in Vallado IAU-80 TEME-to-ECI reduction.",
            "Scenario-level simulator.frames can supply EOP files or manual EOP/time-scale corrections for Earth-fixed transforms.",
            "Use the existing special propagation path for controlled spacecraft and OEL force-model studies.",
        ),
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.orbit.integrator",
        title="Orbit Integrator",
        description="Numerical integrator used by the orbit propagator.",
        aliases=("integrator", "orbit integration", "propagator integrator"),
        options=(
            _option("rk4", "Fixed-step fourth-order Runge-Kutta; normal starting point."),
            _option("rkf78", "Adaptive RKF78 path for tighter propagation checks."),
            _option("dopri5", "Adaptive Dormand-Prince 5 path."),
            _option("adaptive", "Alias that currently routes to RKF78 behavior."),
        ),
        example="simulator:\n  dynamics:\n    orbit:\n      integrator: rk4",
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.rocket.atmosphere_model",
        title="Atmosphere Model",
        description="Density model used by drag/aerodynamic calculations when atmosphere-dependent forces are enabled.",
        aliases=("atmosphere", "atmospheric model", "density model", "drag atmosphere"),
        options=(
            _option("exponential", "Simple exponential atmosphere; cheapest and most robust."),
            _option("ussa1976", "U.S. Standard Atmosphere 1976 table-style model; default for rocket aero configs."),
            _option(
                "msis86",
                "Local MSIS-86 backend copied from MATLAB HPOP; supports direct env inputs or HPOP-style SW-All table input.",
            ),
            _option(
                "nrlmsise00",
                "Local NRLMSISE-00 backend copied from MATLAB HPOP; supports env-provided inputs or callable override.",
            ),
            _option(
                "jacchia70",
                "Local Jacchia-70 backend copied from MATLAB HPOP; supports direct env inputs and HPOP-style solar tables.",
            ),
            _option(
                "jb2006",
                "Local Jacchia-Bowman 2006 backend copied from MATLAB HPOP; supports HPOP-style SOL and geomagnetic inputs.",
            ),
            _option(
                "jb2008",
                "Local Jacchia-Bowman 2008 backend copied from MATLAB HPOP; supports HPOP-style SOL and DTC inputs.",
            ),
            _option(
                "harris_priester",
                "Local Harris-Priester backend using the bundled HPOP coefficient table.",
            ),
        ),
        example="simulator:\n  dynamics:\n    rocket:\n      atmosphere_model: ussa1976",
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.rocket.wind_enu_m_s",
        title="Rocket Wind Vector",
        description=(
            "Steady wind vector for rocket/ascent aerodynamics, expressed in local ENU axes "
            "as meters per second. Use this for simple crosswind stress cases; it is not a "
            "weather forecast, wind-shear profile, or time-varying atmospheric model."
        ),
        aliases=(
            "wind_enu_m_s",
            "wind",
            "crosswind",
            "rocket wind",
            "wind shear",
            "weather",
            "launch weather",
        ),
        options=(
            _option("east", "East component of steady wind in meters per second."),
            _option("north", "North component of steady wind in meters per second."),
            _option("up", "Up component of steady wind in meters per second."),
        ),
        example=(
            "simulator:\n"
            "  dynamics:\n"
            "    rocket:\n"
            "      atmosphere_model: ussa1976\n"
            "      wind_enu_m_s: [0.0, 5.0, 0.0]"
        ),
        notes=(
            "See configs/rocket_tvc_tracking_smoke.yaml for a maintained mild-crosswind example.",
            "For drag-sensitive orbital studies, use atmosphere_env and the documented atmosphere models instead.",
        ),
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.reentry",
        title="Re-Entry Diagnostics",
        description=(
            "Enables atmospheric re-entry diagnostics such as dynamic pressure, drag deceleration, "
            "g-load, heat rate, heat load, and termination thresholds. This is an engineering "
            "diagnostic path, not a breakup, ablation, plasma, or certification model."
        ),
        aliases=(
            "reentry",
            "re-entry",
            "deorbit",
            "deorbit lifetime",
            "orbital decay",
            "decay",
            "reentry breakup",
            "breakup",
            "ablation",
        ),
        options=(
            _option("enabled", "Turn re-entry diagnostics on for configured objects."),
            _option("begin_altitude_km", "Altitude threshold below which re-entry metrics become active."),
            _option("object_ids", "Object id or list of ids to track; omit to track all active objects."),
            _option("atmosphere_model", "Atmosphere model used for diagnostics, such as exponential or ussa1976."),
            _option("termination", "Optional min-altitude, max-g, heat-load, and dynamic-pressure stop limits."),
        ),
        example=(
            "simulator:\n"
            "  dynamics:\n"
            "    orbit:\n"
            "      drag: true\n"
            "    reentry:\n"
            "      enabled: true\n"
            "      begin_altitude_km: 300.0\n"
            "outputs:\n"
            "  plots:\n"
            "    preset: reentry"
        ),
        notes=(
            "Start with configs/reentry_smoke.yaml or examples/configs/public_reentry_interactive_demo.yaml.",
            "For deorbit lifetime studies, use the documented drag/re-entry workflow and state fidelity limits clearly.",
            "OEL does not model breakup debris, ablation, plasma, plume heating, or operational re-entry safety certification.",
        ),
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.orbit.srp_shadow_model",
        title="SRP Shadow Model",
        description="Eclipse/shadow model used when solar radiation pressure is enabled.",
        aliases=("shadow model", "eclipse model", "srp eclipse"),
        options=(
            _option("conical", "Conical Earth shadow approximation; normal fidelity-oriented choice."),
            _option("cylindrical", "Simpler cylindrical shadow approximation."),
            _option("none", "Disable SRP shadow gating."),
        ),
        example="simulator:\n  dynamics:\n    orbit:\n      srp: true\n      srp_shadow_model: conical",
    ),
    ConfigHelpEntry(
        path="outputs.mode",
        title="Output Mode",
        description="Controls whether plots and artifacts are shown interactively, saved, or both.",
        aliases=("output mode", "plot mode", "display mode"),
        options=(
            _option("interactive", "Show interactive windows where supported."),
            _option("save", "Write artifacts to outputs.output_dir without opening windows."),
            _option("both", "Save artifacts and show interactive windows."),
        ),
        example='outputs:\n  mode: "save"',
    ),
    ConfigHelpEntry(
        path="outputs.plots.preset",
        title="Plot Preset",
        description="Expands to a maintained bundle of figure IDs. You can combine presets with explicit figure_ids.",
        aliases=("plot preset", "plot presets", "figure preset", "outputs plots preset"),
        option_loader=_plot_preset_options,
        example='outputs:\n  plots:\n    preset: "rendezvous"',
    ),
    ConfigHelpEntry(
        path="outputs.plots.figure_ids",
        title="Figure IDs",
        description="Explicit single-run plot artifacts to request.",
        aliases=("figure ids", "figures", "plot ids", "available plots"),
        option_loader=_figure_id_options,
        example='outputs:\n  plots:\n    figure_ids: ["run_dashboard", "relative_range"]',
    ),
    ConfigHelpEntry(
        path="outputs.plots.style",
        title="Plot Style",
        description="Controls the visual identity applied to saved single-run plot artifacts.",
        aliases=("plot style", "plot theme", "artifact style", "artifact theme"),
        options=(
            _option("oel_dark", "Branded dark OEL artifact style for screen review and demos."),
            _option("oel_light", "Branded light OEL artifact style for reports and print-friendly review."),
            _option("matplotlib", "Use Matplotlib defaults without OEL artifact styling."),
        ),
        example='outputs:\n  plots:\n    style: "oel_dark"',
    ),
    ConfigHelpEntry(
        path="outputs.animations.types",
        title="Animation Types",
        description="Animation artifacts to request when outputs.animations.enabled is true.",
        aliases=("animation types", "animations", "animation ids"),
        option_loader=_animation_type_options,
        example='outputs:\n  animations:\n    enabled: true\n    types: ["ground_track_multi"]',
    ),
    ConfigHelpEntry(
        path="outputs.animations.style",
        title="Animation Style",
        description="Controls the visual identity applied to saved animation artifacts. Defaults to outputs.plots.style.",
        aliases=("animation style", "animation theme", "movie style", "movie theme"),
        options=(
            _option("oel_dark", "Branded dark OEL artifact style for screen review and demos."),
            _option("oel_light", "Branded light OEL artifact style for reports and print-friendly review."),
            _option("matplotlib", "Use Matplotlib defaults without OEL artifact styling."),
        ),
        example='outputs:\n  animations:\n    style: "oel_dark"',
    ),
    ConfigHelpEntry(
        path="outputs.review",
        title="Output Review Store",
        description=(
            "Enables the durable SQLite review store for completed single-run outputs. "
            "Use it with the sim.review CLI/API and custom review plotting tools."
        ),
        aliases=("review store", "output review", "sqlite review", "outputs review"),
        options=(
            _option("enabled", "Write review/run.sqlite and review/schema.json beside normal artifacts."),
            _option("detail", "compact, standard, or full. Initial tables focus on standard single-run review."),
            _option("strict", "Raise if review-store writing fails instead of preserving normal artifacts."),
        ),
        example='outputs:\n  review:\n    enabled: true\n    detail: "standard"',
    ),
    ConfigHelpEntry(
        path="controller_bench",
        title="Controller Bench Workflow",
        description=(
            "Defines repeatable controller benchmark suites run through the simulator's "
            "`--controller-bench <suite.yaml>` workflow. Use validate-only before executing "
            "a suite, especially when editing cases, controllers, or optimization settings."
        ),
        aliases=(
            "controller bench",
            "controller benchmark",
            "controller benchmarks",
            "bench suite",
            "benchmark suite",
            "controller-bench",
        ),
        options=(
            _option("suite_id", "Stable suite identifier reported in validation and outputs."),
            _option("description", "Human-readable summary of what the suite exercises."),
            _option("output_dir", "Workspace-relative directory for benchmark outputs."),
            _option("cases", "Benchmark cases, each referencing a scenario config and optional case settings."),
            _option("controllers", "Controller variants or plugin specs compared across cases."),
            _option("optimization", "Optional optimizer configuration, such as PSO tuning settings."),
        ),
        example=(
            "# Validate a maintained suite before running it:\n"
            ".venv/bin/python run_simulation.py --controller-bench "
            "configs/controller_bench_rendezvous.yaml --validate-only"
        ),
        notes=("See docs/controller-bench.md for the complete suite schema and workflow.",),
        visibility="pro",
    ),
    ConfigHelpEntry(
        path="ground_stations",
        title="Ground Stations",
        description=(
            "Defines passive ground sites used to compute geometric access, elevation, range, "
            "and access-window review evidence for active objects."
        ),
        aliases=(
            "ground access",
            "ground station access",
            "ground stations",
            "station access",
            "access windows",
        ),
        options=(
            _option("id", "Unique station identifier used in review tables and reports."),
            _option("lat_deg", "Geodetic station latitude in degrees."),
            _option("lon_deg", "Geodetic station longitude in degrees."),
            _option("alt_km", "Station altitude above the reference ellipsoid in kilometers."),
            _option("min_elevation_deg", "Minimum elevation angle required for access."),
            _option("max_range_km", "Optional maximum slant range for access."),
        ),
        example=(
            "ground_stations:\n"
            "  - id: colorado_springs\n"
            "    lat_deg: 38.803\n"
            "    lon_deg: -104.526\n"
            "    alt_km: 1.9\n"
            "    min_elevation_deg: 10.0"
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.knowledge.sensor_error",
        title="Knowledge Sensor Error",
        description=(
            "Defines measurement-error assumptions for object knowledge and estimation updates. "
            "This is not an optical/radar camera hardware model; modeled sensor hardware blocks "
            "such as knowledge.sensor are rejected."
        ),
        aliases=(
            "sensor error",
            "sensor_error",
            "measurement error",
            "measurement noise",
            "optical camera",
            "radar",
            "radar tracking",
            "optical sensor",
            "modeled sensor",
            "knowledge sensor",
        ),
        options=(
            _option("pos_sigma_km", "Position 1-sigma measurement error by axis, in kilometers."),
            _option("vel_sigma_km_s", "Velocity 1-sigma measurement error by axis, in kilometers per second."),
            _option("estimation.type", "Estimator choice, such as ekf or measured_state, under knowledge.estimation."),
            _option(
                "estimation.maneuver_detection",
                "Optional EKF innovation/NIS maneuver detector with rolling persistence gates.",
            ),
            _option("conditions", "Optional access constraints such as require_line_of_sight and max_range_km."),
        ),
        example=(
            "objects:\n"
            "  chaser:\n"
            "    knowledge:\n"
            "      targets: [target]\n"
            "      sensor_error:\n"
            "        pos_sigma_km: [0.01, 0.01, 0.01]\n"
            "        vel_sigma_km_s: [0.0001, 0.0001, 0.0001]\n"
            "      estimation:\n"
            "        type: ekf\n"
            "        maneuver_detection:\n"
            "          enabled: true\n"
            "          window_size: 5\n"
            "          detection_count: 3"
        ),
        notes=(
            "For geometric ground access, use ground_stations rather than a modeled RF or optical sensor.",
            "OEL public configs model knowledge error/estimation here, not camera aperture, limiting magnitude, or RF link budgets.",
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.kind",
        title="Object Kind",
        description="Declares the runtime object family for a configured object.",
        aliases=("agent kind", "object kind", "vehicle kind"),
        options=(
            _option("satellite", "Spacecraft/satellite object using orbital and optional attitude dynamics."),
            _option("rocket", "Launch/ascent vehicle using rocket-specific dynamics and guidance."),
        ),
        example="objects:\n  chaser:\n    kind: satellite",
    ),
    ConfigHelpEntry(
        path="objects.<id>.orbit_control.builtin",
        title="Built-In Orbit Controller",
        description=(
            "Selects a trusted OEL orbit controller by stable catalog ID. Constructor settings go in the sibling "
            "params mapping; use the GNC catalog CLI to inspect parameter schemas, assumptions, limits, and evidence."
        ),
        aliases=("orbit controller", "built in orbit control", "GNC orbit catalog", "RPO controller"),
        option_loader=_orbit_control_builtin_options,
        example=(
            "objects:\n"
            "  chaser:\n"
            "    orbit_control:\n"
            "      builtin: orbit.vbar_approach\n"
            "      params:\n"
            "        approach_speed_m_s: 0.1"
        ),
        notes=("Run `.venv/bin/python -m sim.gnc show orbit.vbar_approach` for the complete descriptor.",),
    ),
    ConfigHelpEntry(
        path="objects.<id>.attitude_control.builtin",
        title="Built-In Attitude Controller",
        description=(
            "Selects a trusted OEL attitude controller by stable catalog ID. The catalog identifies Reference, "
            "workbench, experimental, compatibility, and internal maturity honestly."
        ),
        aliases=("attitude controller", "built in attitude control", "pointing controller", "attitude catalog"),
        option_loader=_attitude_control_builtin_options,
        example=(
            "objects:\n"
            "  observer:\n"
            "    attitude_control:\n"
            "      builtin: attitude.nadir_pointing\n"
            "      params:\n"
            "        primary_axis_body: [1.0, 0.0, 0.0]"
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.mission_strategy.builtin",
        title="Built-In Mission Strategy",
        description=(
            "Selects a trusted OEL mission-intent producer by stable catalog ID. Mission strategies express intent; "
            "mission execution modules turn that intent into deterministic commands and gates."
        ),
        aliases=("mission strategy", "mission intent", "built in mission strategy", "mission catalog"),
        option_loader=_mission_strategy_builtin_options,
        example=(
            "objects:\n"
            "  chaser:\n"
            "    mission_strategy:\n"
            "      builtin: mission.stationkeep\n"
            "      params:\n"
            "        target_id: target"
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.mission_execution.builtin",
        title="Built-In Mission Execution",
        description=(
            "Selects a trusted OEL command-execution or safety-gating module by stable catalog ID. Commands remain "
            "inputs to OEL's deterministic dynamics and actuator paths; built-ins do not replace the physics engine."
        ),
        aliases=("mission execution", "command module", "burn execution", "safety gate", "execution catalog"),
        option_loader=_mission_execution_builtin_options,
        example=(
            "objects:\n"
            "  chaser:\n"
            "    mission_execution:\n"
            "      builtin: execution.integrated_command\n"
            "      params:\n"
            "        require_attitude_alignment: false"
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.flight_software.profile",
        title="Flight-Software Use-Case Profile",
        description=(
            "Selects a versioned use-case profile that resolves to a complete v2 stack, "
            "declared hardware family, task cadence, and coherent defaults. Profile maturity is independent "
            "from component and stack maturity."
        ),
        aliases=(
            "flight software profile",
            "fsw profile",
            "satellite software profile",
            "prebuilt flight software",
            "gnc use case",
        ),
        option_loader=_flight_software_profile_options,
        example=(
            "objects:\n"
            "  chaser:\n"
            "    flight_software:\n"
            "      profile: fsw.profile.rpo_far_field_rendezvous.v1\n"
            "      params:\n"
            "        reference_object_id: target"
        ),
        notes=(
            "Use `.venv/bin/python -m sim.flight_software show <profile-id>` for requirements and limits.",
            "Profile support is bounded to each exact version's documented simulation envelope and does not imply operational suitability.",
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.specs.mass_properties",
        title="Mass Properties",
        description=(
            "Optional rigid-body mass-property block for attitude dynamics and runtime inertia. "
            "When inertia is supplied for runtime use, reference it to the center of mass."
        ),
        aliases=(
            "mass properties",
            "inertia",
            "center of mass",
            "centre of mass",
            "inertia reference",
            "runtime inertia",
        ),
        options=(
            _option("mass_kg", "Object mass in kilograms when not supplied by another specs field."),
            _option("center_of_mass_body_m", "Center of mass in body-frame meters."),
            _option("inertia_kg_m2", "3x3 inertia matrix in kg m^2."),
            _option("inertia_reference_point", "Use `center_of_mass` for runtime inertia."),
        ),
        example=(
            "objects:\n"
            "  target:\n"
            "    specs:\n"
            "      mass_properties:\n"
            "        center_of_mass_body_m: [0.0, 0.0, 0.0]\n"
            "        inertia_reference_point: center_of_mass\n"
            "        inertia_kg_m2:\n"
            "          - [12.0, 0.0, 0.0]\n"
            "          - [0.0, 10.0, 0.0]\n"
            "          - [0.0, 0.0, 8.0]"
        ),
    ),
    ConfigHelpEntry(
        path="objects.<id>.specs.preset_satellite",
        title="Satellite Preset",
        description="Named Python satellite hardware preset used by compatibility config paths.",
        aliases=("satellite preset", "preset satellite", "spacecraft preset"),
        option_loader=_satellite_preset_options,
        example="objects:\n  target:\n    specs:\n      preset_satellite: BASIC_SATELLITE",
    ),
    ConfigHelpEntry(
        path="objects.<id>.specs.preset_stack",
        title="Rocket Stack Preset",
        description="Named Python rocket stack preset used by rocket configs.",
        aliases=("rocket preset", "stack preset", "preset stack"),
        option_loader=_rocket_preset_options,
        example="rocket:\n  specs:\n    preset_stack: BASIC_TWO_STAGE_STACK",
    ),
    ConfigHelpEntry(
        path="profile",
        title="Simulation Profile",
        description="Shared fidelity profile used by Python builders and profile-aware workflows.",
        aliases=("fidelity profile", "simulation profile"),
        option_loader=_profile_options,
        example='profile: "ops"',
    ),
)


def _load_pro_config_help_entries() -> tuple[ConfigHelpEntry, ...]:
    try:
        from sim.config.pro_help import PRO_CONFIG_HELP_ENTRIES
    except Exception:
        return ()
    return tuple(PRO_CONFIG_HELP_ENTRIES)


CONFIG_HELP_ENTRIES = (*CONFIG_HELP_ENTRIES, *_load_pro_config_help_entries())


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _score(query: str, entry: ConfigHelpEntry) -> float:
    q = _normalize(query)
    if not q:
        return 0.0
    best = 0.0
    q_tokens = set(q.split())
    domain_tokens = {"sensitivity", "covariance", "monte", "carlo", "analysis"}
    for term_raw in entry.search_terms():
        term = _normalize(term_raw)
        if not term:
            continue
        term_tokens = set(term.split())
        if q_tokens & domain_tokens and not (q_tokens & domain_tokens & term_tokens):
            continue
        if q == term:
            return 1.0
        if q in term or term in q:
            best = max(best, 0.92)
        overlap = len(q_tokens & term_tokens) / max(1, len(q_tokens | term_tokens))
        best = max(best, 0.55 * overlap + 0.45 * SequenceMatcher(None, q, term).ratio())
    return best


def find_config_help(
    query: str,
    *,
    limit: int = 5,
    scope: str = DEFAULT_HELP_SCOPE,
) -> list[tuple[float, ConfigHelpEntry]]:
    """Return matching config help entries ranked by fuzzy relevance."""
    entries = [entry for entry in CONFIG_HELP_ENTRIES if _scope_allows(entry, scope)]
    ranked = sorted(((_score(query, entry), entry) for entry in entries), key=lambda item: item[0], reverse=True)
    return [(score, entry) for score, entry in ranked[:limit] if score >= 0.35]


def list_config_help_entries(*, scope: str = DEFAULT_HELP_SCOPE) -> list[ConfigHelpEntry]:
    return [entry for entry in CONFIG_HELP_ENTRIES if _scope_allows(entry, scope)]


def load_config_help_context(path: str | Path) -> dict[str, Any]:
    """Load YAML as plain data for contextual help without resolving plugins or presets."""
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to inspect config files. Install with `pip install pyyaml`.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config YAML root must be a mapping/object: {config_path}")
    return dict(raw)


def _is_placeholder_token(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")


def _resolve_config_values(value: Any, tokens: list[str], actual_path: str = "") -> list[tuple[str, Any]]:
    if not tokens:
        return [(actual_path, value)]

    token = tokens[0]
    rest = tokens[1:]
    out: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        if _is_placeholder_token(token):
            for key, child in value.items():
                child_path = f"{actual_path}.{key}" if actual_path else str(key)
                out.extend(_resolve_config_values(child, rest, child_path))
        elif token in value:
            child_path = f"{actual_path}.{token}" if actual_path else token
            out.extend(_resolve_config_values(value[token], rest, child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = f"{actual_path}[{idx}]" if actual_path else f"[{idx}]"
            out.extend(_resolve_config_values(child, tokens, child_path))
    return out


def _format_config_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    try:
        import yaml  # type: ignore

        rendered = yaml.safe_dump(value, sort_keys=False, default_flow_style=True).strip()
        return rendered.rstrip("\n")
    except Exception:
        return repr(value)


def _context_label(config_path: str | Path | None) -> str:
    if config_path is None:
        return "Current Config"
    return f"Current Config ({config_path})"


def _entry_config_values(config_data: dict[str, Any], entry: ConfigHelpEntry) -> list[tuple[str, Any]]:
    values = _resolve_config_values(config_data, entry.path.split("."))
    if values or not entry.path.startswith("objects.<id>."):
        return values
    compatibility_path = entry.path.removeprefix("objects.")
    return _resolve_config_values(config_data, compatibility_path.split("."))


def format_config_help(
    query: str,
    *,
    max_options: int = 80,
    config_data: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    scope: str = DEFAULT_HELP_SCOPE,
) -> str:
    matches = find_config_help(query, limit=5, scope=scope)
    if not matches:
        return f"No config help match for {query!r}.\n\nRun `python config_help.py --list` to see known help topics."

    score, entry = matches[0]
    lines = [
        f"{entry.title}",
        f"Field: {entry.path}",
        "",
        *textwrap.wrap(entry.description, width=88),
        "",
    ]
    if config_data is not None:
        lines.append(_context_label(config_path))
        values = _entry_config_values(config_data, entry)
        if values:
            for actual_path, value in values:
                lines.append(f"  {actual_path}: {_format_config_value(value)}")
        else:
            lines.append(f"  {entry.path}: not set in this config")
        lines.append("")

    lines.append("Options:")
    options = entry.resolved_options()
    if not options:
        lines.append("  (No fixed option list; this field accepts user-supplied values.)")
    else:
        width = min(max(len(opt.value) for opt in options), 34)
        for opt in options[:max_options]:
            wrapped = textwrap.wrap(opt.description, width=max(40, 84 - width))
            lines.append(f"  {opt.value:<{width}}  {wrapped[0] if wrapped else ''}")
            for continuation in wrapped[1:]:
                lines.append(f"  {'':<{width}}  {continuation}")
        if len(options) > max_options:
            lines.append(
                f"  ... {len(options) - max_options} more options hidden; refine the query for a narrower list."
            )
    if entry.notes:
        lines.extend(["", "Notes:"])
        for note in entry.notes:
            for idx, wrapped in enumerate(textwrap.wrap(note, width=84)):
                prefix = "  - " if idx == 0 else "    "
                lines.append(f"{prefix}{wrapped}")
    if entry.example:
        lines.extend(["", "Example:", textwrap.indent(entry.example, "  ")])
    if score < 0.75 and len(matches) > 1:
        lines.extend(["", "Other possible matches:"])
        for _, other in matches[1:4]:
            lines.append(f"  - {other.path} ({other.title})")
    return "\n".join(lines)


def format_config_help_list(*, scope: str = DEFAULT_HELP_SCOPE) -> str:
    lines = ["Known config help topics:"]
    entries = list_config_help_entries(scope=scope)
    if not entries:
        return "Known config help topics: none"
    width = max(len(entry.path) for entry in entries)
    for entry in sorted(entries, key=lambda item: item.path):
        lines.append(f"  {entry.path:<{width}}  {entry.title}")
    return "\n".join(lines)
