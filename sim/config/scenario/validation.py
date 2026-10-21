from __future__ import annotations

from sim.config.scenario.models import (
    SimulationScenarioConfig,
)
from sim.config.scenario.primitives import (
    _parse_bool,
)
from sim.numeric_backend import normalize_numeric_backend

__all__ = [
    '_validate_physics_runtime_settings',
    '_validate_object_references',
    '_validate_orbital_analysis_references',
]


def _validate_orbital_analysis_references(cfg: SimulationScenarioConfig) -> None:
    section = cfg.outputs.orbital_analysis
    if not bool(section.enabled):
        return
    if cfg.simulator.initial_jd_utc is None:
        raise ValueError("outputs.orbital_analysis requires simulator.initial_jd_utc.")
    if not section.coverage and not section.directed_links:
        raise ValueError("outputs.orbital_analysis.enabled requires coverage or directed_links entries.")
    objects = dict(cfg.objects or {})
    ground_stations = {
        str(station.id): station
        for station in list(cfg.ground_stations or [])
    }

    def require_object(object_id: str, path: str) -> object:
        item = objects.get(object_id)
        if item is None:
            raise ValueError(f"{path} references unknown object {object_id!r}.")
        if not bool(getattr(item, "enabled", False)):
            raise ValueError(f"{path} references disabled object {object_id!r}.")
        return item

    attitude_enabled = bool(
        dict(dict(cfg.simulator.dynamics or {}).get("attitude", {}) or {}).get("enabled", True)
    )
    propagation_method = str(
        dict(dict(cfg.simulator.dynamics or {}).get("orbit", {}) or {}).get(
            "propagation_method", "special"
        )
        or "special"
    ).strip().lower()

    def require_achieved_attitude(item: object, path: str) -> None:
        trajectory_only = str(getattr(item, "runtime_profile", "") or "").strip().lower() == "trajectory_only"
        if propagation_method == "general":
            raise ValueError(
                f"{path} requires achieved attitude, but general OGP propagation retains only static "
                "initial attitude. Use special ONP propagation or an attitude-independent product."
            )
        if not attitude_enabled or trajectory_only:
            raise ValueError(f"{path} requires achieved attitude from enabled non-trajectory-only dynamics.")

    for index, item in enumerate(section.coverage):
        path = f"outputs.orbital_analysis.coverage[{index}].source_object_id"
        source = require_object(str(item["source_object_id"]), path)
        require_achieved_attitude(source, path)
    for index, item in enumerate(section.directed_links):
        for endpoint in ("tx", "rx"):
            object_id = str(item.get(f"{endpoint}_object_id") or "").strip()
            station_id = str(item.get(f"{endpoint}_ground_station_id") or "").strip()
            endpoint_path = f"outputs.orbital_analysis.directed_links[{index}].{endpoint}"
            terminal = dict(item[f"{endpoint}_terminal"] or {})
            pattern = dict(terminal.get("pattern", {}) or {})
            if object_id:
                endpoint_object = require_object(object_id, f"{endpoint_path}_object_id")
                if str(pattern.get("kind", "constant") or "constant").lower() != "constant":
                    require_achieved_attitude(endpoint_object, f"{endpoint_path}_object_id")
            elif station_id:
                station = ground_stations.get(station_id)
                if station is None:
                    raise ValueError(
                        f"{endpoint_path}_ground_station_id references unknown ground station {station_id!r}."
                    )
                if not bool(station.enabled):
                    raise ValueError(
                        f"{endpoint_path}_ground_station_id references disabled ground station {station_id!r}."
                    )

def _require_rust_symbols(path: str, symbols: tuple[str, ...]) -> None:
    try:
        import oel_rust_orbit as native
    except ImportError as exc:
        raise ValueError(f"{path} requires the optional oel_rust_orbit wheel in this Python environment.") from exc
    missing = [name for name in symbols if not callable(getattr(native, name, None))]
    if missing:
        raise ValueError(f"{path} requires an updated oel_rust_orbit wheel; missing kernels: {', '.join(missing)}.")


def _validate_physics_runtime_settings(cfg: SimulationScenarioConfig) -> None:
    if cfg.simulator.additional_gravity is not None:
        from .additional_gravity import validate_additional_gravity
        validate_additional_gravity(cfg)
    if cfg.simulator.central_body != "earth":
        from .planetary import validate_planetary_scenario
        validate_planetary_scenario(cfg)
        return
    for object_id, obj in cfg.objects.items():
        if "position_inertial_km" in obj.initial_state or "reference_axes" in obj.initial_state:
            raise ValueError(f"Earth objects.{object_id} must use existing Earth state fields; neutral states require non-Earth central_body.")
    dynamics = dict(cfg.simulator.dynamics or {})
    execution = dict(cfg.simulator.execution or {})
    parallelism = dict(execution.get("object_parallelism", {}) or {})
    if (parallelism.get("enabled", False) and parallelism.get("backend") == "thread_pool"
            and parallelism.get("thread_scope", "auto") == "native_history"):
        path = "simulator.execution.object_parallelism.backend=thread_pool"
        _require_rust_symbols(path, ("propagate_passive_sampled_eci_bytes",))
        import oel_rust_orbit

        if not getattr(oel_rust_orbit, "PASSIVE_HISTORY_GIL_RELEASED", False):
            raise ValueError(f"{path} requires a Rust wheel with GIL-releasing sampled histories.")
    native_requirements = {
        "attitude": ("attitude_propagate_exponential_map", "attitude_builtin_disturbance_torque"),
        "rocket": ("rocket_aero_state", "rocket_propellant_step"),
        "reentry": ("reentry_metrics",),
    }
    for section, symbols in native_requirements.items():
        backend = normalize_numeric_backend(
            dict(dynamics.get(section, {}) or {}).get("numeric_backend", "rust"),
            field_name=f"simulator.dynamics.{section}.numeric_backend",
        )
        if backend == "rust":
            _require_rust_symbols(f"simulator.dynamics.{section}.numeric_backend=rust", symbols)
    collisions_backend = normalize_numeric_backend(
        dict(cfg.simulator.collisions or {}).get("numeric_backend", "rust"),
        field_name="simulator.collisions.numeric_backend",
    )
    if collisions_backend == "rust":
        _require_rust_symbols("simulator.collisions.numeric_backend=rust", ("collision_chord_geometry", "collision_elastic_impact"))
    covariance_backend = normalize_numeric_backend(
        getattr(cfg.analysis.covariance, "numeric_backend", "rust"),
        field_name="analysis.covariance.numeric_backend",
    )
    if cfg.analysis.enabled and cfg.analysis.covariance.enabled and covariance_backend == "rust":
        _require_rust_symbols("analysis.covariance.numeric_backend=rust", ("covariance_propagate_history",))
    for object_id, obj in cfg.objects.items():
        knowledge = dict(getattr(obj, "knowledge", {}) or {})
        estimation = dict(knowledge.get("estimation", {}) or {})
        estimation_ekf = dict(estimation.get("ekf", knowledge.get("ekf", {})) or {})
        tracking_backend = normalize_numeric_backend(
            estimation.get(
                "numeric_backend",
                estimation_ekf.get("numeric_backend", knowledge.get("numeric_backend", "rust")),
            ),
            field_name=f"objects.{object_id}.knowledge.numeric_backend",
        )
        if tracking_backend == "rust":
            _require_rust_symbols(
                f"objects.{object_id}.knowledge.numeric_backend=rust",
                ("tracking_measurement", "tracking_measurement_and_jacobian"),
            )
    orbit = dict((cfg.simulator.dynamics or {}).get("orbit", {}) or {})
    model = str(orbit.get("model", "two_body") or "two_body").strip().lower()
    propagation_method = str(orbit.get("propagation_method", "special") or "special").strip().lower()
    integrator = str(orbit.get("integrator", "rk4") or "rk4").strip().lower()
    system_forces = list(getattr(cfg.simulator, "system_force_models", []) or [])
    if any(
        obj.enabled
        and normalize_numeric_backend(
            dict(obj.general or {}).get("numeric_backend", "rust"),
            field_name=f"objects.{object_id}.general.numeric_backend",
        )
        == "rust"
        for object_id, obj in cfg.objects.items()
    ):
        try:
            import oel_rust_orbit
        except ImportError as exc:
            raise ValueError(
                "objects.*.general.numeric_backend=rust requires the optional oel_rust_orbit wheel."
            ) from exc
        if not hasattr(oel_rust_orbit, "OGPContext"):
            raise ValueError("objects.*.general.numeric_backend=rust requires an OGP-enabled oel_rust_orbit wheel.")
    orbit_backend = normalize_numeric_backend(
        orbit.get("numeric_backend", "rust"),
        field_name="simulator.dynamics.orbit.numeric_backend",
    )
    if orbit_backend == "rust" and propagation_method != "general":
        path = "simulator.dynamics.orbit.numeric_backend=rust"
        if model not in {"two_body", "cr3bp"} or propagation_method != "special":
            raise ValueError(f"simulator.dynamics.orbit.model with {path} requires two-body ECI ONP or rotating CR3BP special propagation.")
        if integrator not in {"rk4", "rkf78", "adaptive", "dopri5"}:
            raise ValueError(f"simulator.dynamics.orbit.integrator with {path} supports only RK4, RKF78, and DOPRI5.")
        from importlib.util import find_spec

        if find_spec("oel_rust_orbit") is None:
            raise ValueError(f"{path} requires the optional oel_rust_orbit wheel in this Python environment.")
        if model == "cr3bp":
            import oel_rust_orbit

            if not hasattr(oel_rust_orbit, "cr3bp_propagate"):
                raise ValueError(f"{path} with model=cr3bp requires a CR3BP-enabled oel_rust_orbit wheel (0.6.0 or newer).")
    collisions = dict(getattr(cfg.simulator, "collisions", {}) or {})
    if collisions.get("enabled", False):
        active = {oid: obj for oid, obj in cfg.objects.items() if obj.enabled}
        radii = dict(collisions.get("radii_m", {}) or {})
        execution = dict(cfg.simulator.execution or {})
        attitude = dict((cfg.simulator.dynamics or {}).get("attitude", {}) or {})
        if len(active) != 2 or set(radii) != set(active) or any(
            obj.kind != "satellite" or obj.runtime_profile != "trajectory_only"
            for obj in active.values()
        ):
            raise ValueError("simulator.collisions requires exactly two enabled trajectory_only satellites with radii_m entries.")
        if (model != "two_body"
                or propagation_method != "special"
                or any(obj.propagation_method not in (None, "", "special") for obj in active.values())):
            raise ValueError("simulator.collisions requires Earth-centered ECI ONP propagation.")
        if bool(attitude.get("enabled", True)):
            raise ValueError("simulator.collisions requires attitude.enabled=false in this first slice.")
        if system_forces or any(obj.force_models for obj in active.values()):
            raise ValueError("simulator.collisions cannot be combined with system or object force plugins.")
        if execution.get("policy", "configured") in {"auto", "parallel"} or bool(
            dict(execution.get("object_parallelism", {}) or {}).get("enabled", False)
        ):
            raise ValueError("simulator.collisions requires serial object execution.")
        if any(
            (obj.bridge is not None and obj.bridge.enabled)
            or bool(dict(obj.reference_orbit or {}).get("enabled", False))
            or any(bool(dict(obj.specs.get(name, {}) or {}).get("enabled", False)) for name in ("thermal", "power"))
            for obj in active.values()
        ):
            raise ValueError("simulator.collisions does not support bridges, reference orbits, or spacecraft resources.")
    if system_forces:
        attitude = dict((cfg.simulator.dynamics or {}).get("attitude", {}) or {})
        execution = dict(getattr(cfg.simulator, "execution", {}) or {})
        active = {oid: obj for oid, obj in cfg.objects.items() if obj.enabled}
        if len(active) != 2 or any(
            obj.kind != "satellite" or obj.runtime_profile != "trajectory_only"
            for obj in active.values()
        ):
            raise ValueError("simulator.system_force_models requires exactly two trajectory_only satellites.")
        if (model != "two_body"
                or propagation_method != "special"
                or integrator != "rk4"
                or any(obj.propagation_method not in (None, "", "special") for obj in active.values())):
            raise ValueError("simulator.system_force_models requires two-body ECI ONP with RK4.")
        if cfg.simulator.initial_jd_utc is None:
            raise ValueError("simulator.system_force_models requires simulator.initial_jd_utc.")
        if bool(attitude.get("enabled", True)):
            raise ValueError("simulator.system_force_models requires attitude.enabled=false.")
        if execution.get("policy", "configured") in {"auto", "parallel"} or bool(
            dict(execution.get("object_parallelism", {}) or {}).get("enabled", False)
        ):
            raise ValueError("simulator.system_force_models requires serial object execution.")
        if any(obj.force_models for obj in active.values()):
            raise ValueError("simulator.system_force_models cannot be combined with object force_models in this first slice.")
        if any(
            (obj.bridge is not None and obj.bridge.enabled)
            or bool(dict(obj.reference_orbit or {}).get("enabled", False))
            for obj in active.values()
        ):
            raise ValueError("simulator.system_force_models does not yet support bridges or reference orbits.")
        if any(
            bool(dict(obj.specs.get(name, {}) or {}).get("enabled", False))
            for obj in active.values() for name in ("thermal", "power")
        ):
            raise ValueError("simulator.system_force_models does not yet support spacecraft resources.")
        perturbations = ("j2", "j3", "j4", "drag", "lift", "srp", "third_body_sun", "third_body_moon")
        if any(bool(orbit.get(name, False)) for name in perturbations) or any(
            bool(dict(orbit.get(name, {}) or {}).get("enabled", False))
            for name in ("spherical_harmonics", "schwarzschild", "earth_radiation", "ocean_tides", "solid_earth_tides")
        ):
            raise ValueError("simulator.system_force_models currently requires unperturbed two-body ONP.")
    reentry = dict((cfg.simulator.dynamics or {}).get("reentry", {}) or {})
    if propagation_method not in {"special", "general"}:
        raise ValueError("simulator.dynamics.orbit.propagation_method must be one of: special, general.")
    if integrator not in {"rk4", "rkf78", "dopri5", "adaptive"}:
        raise ValueError("simulator.dynamics.orbit.integrator must be one of: adaptive, dopri5, rk4, rkf78.")

    env = dict(cfg.simulator.environment or {})
    if _parse_bool(reentry.get("enabled", False), "simulator.dynamics.reentry.enabled"):
        if not _parse_bool(orbit.get("drag", False), "simulator.dynamics.orbit.drag"):
            raise ValueError(
                "simulator.dynamics.reentry.enabled requires simulator.dynamics.orbit.drag=true "
                "so reported atmospheric loads and heating use a trajectory propagated with drag."
            )
        reentry_atmosphere = str(reentry.get("atmosphere_model", "") or "").strip().lower().replace("-", "_")
        environment_atmosphere = str(env.get("atmosphere_model", "") or "").strip().lower().replace("-", "_")
        atmosphere_aliases = {
            "hp": "harris_priester",
            "hpop_harris_priester": "harris_priester",
            "hpop_msis86": "msis86",
            "hpop_jacchia70": "jacchia70",
        }
        reentry_atmosphere = atmosphere_aliases.get(reentry_atmosphere, reentry_atmosphere)
        environment_atmosphere = atmosphere_aliases.get(environment_atmosphere, environment_atmosphere)
        if (
            env.get("density_kg_m3") is None
            and reentry_atmosphere
            and environment_atmosphere
            and reentry_atmosphere != environment_atmosphere
        ):
            raise ValueError(
                "simulator.dynamics.reentry.atmosphere_model must match "
                "simulator.environment.atmosphere_model. Configure the shared atmospheric model under "
                "simulator.environment; the reentry field is a compatibility alias only."
            )
    if _parse_bool(orbit.get("drag", False), "simulator.dynamics.orbit.drag") or _parse_bool(
        orbit.get("lift", False), "simulator.dynamics.orbit.lift"
    ):
        if env.get("atmosphere_model") in (None, "") and env.get("density_kg_m3") is None:
            raise ValueError(
                "simulator.dynamics.orbit drag/lift requires simulator.environment.atmosphere_model "
                "or density_kg_m3; atmosphere selection is explicit."
            )
    ephemeris_mode = env.get("ephemeris_mode")
    if ephemeris_mode not in (None, ""):
        mode = str(ephemeris_mode).strip().lower()
        if mode not in {
            "analytic_enhanced",
            "enhanced",
            "analytic_simple",
            "simple",
            "de440",
            "hpop_de440",
            "de440_hpop",
            "spice",
            "spiceypy",
        }:
            raise ValueError(
                "simulator.environment.ephemeris_mode must be one of: analytic_enhanced, analytic_simple, "
                "de440, hpop_de440, de440_hpop, spice, spiceypy."
            )

    for oid, obj in cfg.objects.items():
        if any(dict(obj.specs.get(name, {}) or {}).get("enabled", False) for name in ("thermal", "power")):
            propagation = obj.propagation_method or orbit.get("propagation_method", "special")
            if model != "two_body" or propagation == "general":
                raise ValueError(f"objects.{oid}: spacecraft resources require Earth-centered ONP dynamics")
    if model not in {"two_body", "cr3bp"}:
        raise ValueError("simulator.dynamics.orbit.model must be one of: cr3bp, two_body.")
    for oid, obj in cfg.objects.items():
        if not obj.enabled or not obj.force_models:
            continue
        path = f"objects.{oid}.force_models"
        if obj.kind != "satellite" or model != "two_body" or (obj.propagation_method or propagation_method) != "special":
            raise ValueError(f"{path} requires ECI ONP special propagation for a satellite.")
        if cfg.simulator.initial_jd_utc is None:
            raise ValueError(f"{path} requires simulator.initial_jd_utc.")
    if bool(dict(orbit.get("schwarzschild", {}) or {}).get("enabled", False)):
        if model != "two_body" or propagation_method != "special":
            raise ValueError("schwarzschild requires Earth-centered special ONP propagation.")
        for oid, obj in cfg.objects.items():
            if obj.enabled and (obj.propagation_method or propagation_method) == "general":
                raise ValueError(f"objects.{oid}: schwarzschild cannot be applied to general OGP propagation.")
    radiation = dict(orbit.get("earth_radiation", {}) or {})
    if radiation.get("enabled", False):
        if model != "two_body" or propagation_method != "special":
            raise ValueError("earth_radiation requires Earth-centered special ONP propagation.")
        if cfg.simulator.initial_jd_utc is None:
            raise ValueError("earth_radiation requires simulator.initial_jd_utc.")
        if not env.get("ephemeris_mode"):
            raise ValueError("earth_radiation requires an explicit environment.ephemeris_mode.")
        for oid, obj in cfg.objects.items():
            if obj.enabled and (obj.propagation_method or propagation_method) == "general":
                raise ValueError(f"objects.{oid}: earth_radiation cannot be applied to general OGP propagation.")
    tides = dict(orbit.get("ocean_tides", {}) or {})
    if tides.get("enabled", False):
        if model != "two_body" or propagation_method != "special":
            raise ValueError("ocean_tides requires Earth-centered special ONP propagation.")
        for oid, obj in cfg.objects.items():
            if obj.enabled and (obj.propagation_method or propagation_method) == "general":
                raise ValueError(f"objects.{oid}: ocean_tides cannot be applied to general OGP propagation.")
        from sim.dynamics.orbit.frames import frame_context_from_mapping
        frame = frame_context_from_mapping(dict(cfg.simulator.frames), jd_utc_start=cfg.simulator.initial_jd_utc)
        if frame.jd_utc_start is None or not frame.eop_rotation_available:
            raise ValueError("ocean_tides requires an absolute epoch and EOP-backed simulator.frames.")
        if not frame.eop_path and any(getattr(frame, k) is None for k in ("dut1_s", "dat_s", "xp_arcsec", "yp_arcsec")):
            raise ValueError("ocean_tides requires EOP data or explicit dut1_s, dat_s, xp_arcsec, yp_arcsec.")
    tides = dict(orbit.get("solid_earth_tides", {}) or {})
    if tides.get("enabled", False):
        if model != "two_body" or propagation_method != "special":
            raise ValueError("solid_earth_tides requires Earth-centered special ONP propagation.")
        for oid, obj in cfg.objects.items():
            if obj.enabled and (obj.propagation_method or propagation_method) == "general":
                raise ValueError(f"objects.{oid}: solid_earth_tides cannot be applied to general OGP propagation.")
        from sim.dynamics.orbit.frames import frame_context_from_mapping
        frame = frame_context_from_mapping(dict(cfg.simulator.frames), jd_utc_start=cfg.simulator.initial_jd_utc)
        if frame.jd_utc_start is None or not frame.eop_rotation_available:
            raise ValueError("solid_earth_tides requires an absolute epoch and EOP-backed simulator.frames.")
        if not frame.eop_path and any(getattr(frame, k) is None for k in ("dut1_s", "dat_s", "xp_arcsec", "yp_arcsec")):
            raise ValueError("solid_earth_tides requires EOP data or explicit dut1_s, dat_s, xp_arcsec, yp_arcsec.")
        if not env.get("ephemeris_mode"):
            raise ValueError("solid_earth_tides requires an explicit environment.ephemeris_mode.")
    if model == "cr3bp":
        unsupported = []
        for key in ("j2", "j3", "j4", "drag", "srp", "third_body_sun", "third_body_moon", "lift"):
            if _parse_bool(orbit.get(key, False), f"simulator.dynamics.orbit.{key}"):
                unsupported.append(key)
        if unsupported:
            raise ValueError(
                "simulator.dynamics.orbit.model=cr3bp does not support two-body perturbation flags: "
                + ", ".join(sorted(unsupported))
                + "."
            )

    sh = dict(orbit.get("spherical_harmonics", {}) or {})
    if _parse_bool(sh.get("enabled", False), "simulator.dynamics.orbit.spherical_harmonics.enabled"):
        degree = int(sh.get("degree", 0) or 0)
        source = str(sh.get("source", sh.get("model", "")) or "").strip().lower()
        terms = list(sh.get("terms", []) or [])
        has_terms = bool(terms)
        has_path = sh.get("coeff_path") not in (None, "") or sh.get("source_path") not in (None, "")
        if has_terms and (has_path or source):
            raise ValueError(
                "simulator.dynamics.orbit.spherical_harmonics must use either inline terms or a coefficient source, "
                "not both."
            )
        if not has_terms and degree < 2:
            raise ValueError(
                "File-backed simulator.dynamics.orbit.spherical_harmonics requires degree >= 2; "
                "inline terms infer degree and order when omitted."
            )
        supported_sources = {"hpop", "hpop_ggm03", "ggm03", "icgem", "gfc", "egm96"}
        if source and source not in supported_sources:
            choices = ", ".join(sorted(supported_sources))
            raise ValueError(
                f"simulator.dynamics.orbit.spherical_harmonics.source must be one of: {choices}."
            )
        if not has_terms and not source and not has_path:
            raise ValueError(
                "simulator.dynamics.orbit.spherical_harmonics.enabled requires inline terms or a supported "
                "coefficient source/path; degree and order alone do not define a gravity field."
            )
        if source in {"icgem", "gfc"} and not has_path:
            raise ValueError(
                "ICGEM spherical harmonics require spherical_harmonics.coeff_path or source_path."
            )
def _validate_object_references(cfg: SimulationScenarioConfig) -> None:
    objects = dict(cfg.objects or {})
    if not objects:
        objects = {
            "rocket": cfg.rocket,
            "chaser": cfg.chaser,
            "target": cfg.target,
        }
    relative_forms = {
        "relative_to_target_ric",
        "relative_ric_rect",
        "relative_ric_curv",
        "relative_to_target_cislunar",
        "relative_cislunar",
    }
    enabled_rockets = [
        object_id
        for object_id, section in objects.items()
        if bool(getattr(section, "enabled", False)) and str(getattr(section, "kind", "")).strip().lower() == "rocket"
    ]
    relative_dependencies: dict[str, str] = {}
    for object_id, section in objects.items():
        initial_state = dict(getattr(section, "initial_state", {}) or {})
        if initial_state.get("source") in {"rocket_deployment", "rocket_insertion"} and not enabled_rockets:
            raise ValueError(
                f"objects.{object_id}.initial_state.source requires at least one enabled rocket object."
            )
        reference_id = str(initial_state.get("relative_to", "") or "").strip()
        selected_forms = relative_forms.intersection(initial_state)
        if not reference_id and selected_forms:
            target_defaulted = (
                bool(selected_forms.intersection({"relative_to_target_ric", "relative_to_target_cislunar"}))
                or str(object_id) == "chaser"
            )
            if target_defaulted and "target" in objects:
                reference_id = "target"
            else:
                path = f"objects.{object_id}.initial_state.relative_to"
                raise ValueError(f"{path} is required for relative initial-state form '{sorted(selected_forms)[0]}'.")
        if not reference_id:
            continue
        path = f"objects.{object_id}.initial_state.relative_to"
        if reference_id == str(object_id):
            raise ValueError(f"{path} cannot reference the same object.")
        reference = objects.get(reference_id)
        if reference is None:
            raise ValueError(f"{path} references unknown object '{reference_id}'.")
        if not bool(getattr(reference, "enabled", False)):
            raise ValueError(f"{path} references disabled object '{reference_id}'.")
        if selected_forms:
            relative_dependencies[str(object_id)] = reference_id

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(object_id: str, chain: list[str]) -> None:
        if object_id in visited:
            return
        if object_id in visiting:
            start = chain.index(object_id)
            cycle = chain[start:] + [object_id]
            raise ValueError("Relative initial-state reference cycle: " + " -> ".join(cycle))
        visiting.add(object_id)
        reference_id = relative_dependencies.get(object_id)
        if reference_id in relative_dependencies:
            visit(str(reference_id), [*chain, str(reference_id)])
        visiting.remove(object_id)
        visited.add(object_id)

    for object_id in sorted(relative_dependencies):
        visit(object_id, [object_id])
