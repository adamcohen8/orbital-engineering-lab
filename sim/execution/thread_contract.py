"""Eligibility contract for isolated in-process object-step workers.

The process object executor gets a safety boundary from the operating system:
each object snapshot is unpickled in a different process.  A thread executor
does not have that boundary.  It must therefore clone every object-owned
mutable component and reject an unrecognised callback or provider unless the
component declares the contract below.

``OEL_THREAD_SAFE = True`` is a declaration by the implementation owner.  It
means that the component is safe to call concurrently *after the executor has
given each object an independent state instance*.  The marker is intentionally
read from the class or callable definition's own namespace; an inherited
marker, instance field, or scenario string cannot opt an object into this
contract.

The registry in this module is deliberately made from exact registered OEL
types and callable identities.  A module prefix such as ``sim.*`` is not an
approval mechanism: an arbitrary class placed in a ``sim`` module still has
to carry the explicit marker.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import is_dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any

THREAD_SAFE_MARKER = "OEL_THREAD_SAFE"
THREAD_CONTRACT_VERSION = 1

_MISSING = object()


def _definition_marker(value: Any, name: str = THREAD_SAFE_MARKER) -> bool:
    """Return a class/definition marker without invoking user properties."""

    receiver = getattr(value, "__self__", None)
    if receiver is not None and not inspect.isclass(receiver):
        try:
            if vars(type(receiver)).get(name, _MISSING) is True:
                return True
        except (AttributeError, TypeError):
            pass
    target = getattr(value, "__func__", value) if inspect.ismethod(value) else value
    try:
        owner = target if inspect.isfunction(target) or inspect.isbuiltin(target) else type(target)
        # An implementation marker inherited from a base class would silently
        # approve a subclass
        # that adds a mutable callback or native provider of its own.  The
        # declaration therefore has to be present in this definition's own
        # namespace; ``vars`` is also side-effect free for classes/functions.
        return vars(owner).get(name, _MISSING) is True
    except (AttributeError, TypeError):
        return False


def is_declared_thread_safe(value: Any) -> bool:
    """Whether *value* has a definition-level thread-safety declaration."""

    return _definition_marker(value)


def _exact_type(value: Any, types: tuple[type[Any], ...]) -> bool:
    return type(value) in types


def _safe_import_type(module_name: str, class_name: str) -> type[Any] | None:
    try:
        value = getattr(import_module(module_name), class_name)
    except (ImportError, AttributeError, TypeError):
        return None
    return value if inspect.isclass(value) else None


@lru_cache(maxsize=1)
def _registered_builtin_types() -> tuple[type[Any], ...]:
    """Return exact types owned by OEL's registered runtime surfaces.

    This list is intentionally assembled from the authoritative GNC and FSW
    registries plus concrete runtime classes.  It must not become a broad
    ``module.startswith('sim.')`` check: private or user-injected classes can
    live below the same namespace.
    """

    candidates: list[type[Any]] = []

    def add(value: Any) -> None:
        if inspect.isclass(value) and value not in candidates:
            candidates.append(value)

    # Core dynamics and object-owned numerical state.
    for module_name, class_names in (
        (
            "sim.dynamics.model",
            ("OrbitalAttitudeDynamics",),
        ),
        (
            "sim.dynamics.attitude.disturbances",
            ("DisturbanceTorqueModel", "DisturbanceTorqueConfig"),
        ),
        (
            "sim.dynamics.orbit.propagator",
            ("OrbitPropagator",),
        ),
        (
            "sim.dynamics.orbit.custom_force",
            ("CustomForceModel",),
        ),
        (
            "sim.spacecraft_resources.model",
            ("SpacecraftResources",),
        ),
        (
            "sim.dynamics.spacecraft_geometry",
            ("GeometryAreaProfile", "RectangularPrismGeometry"),
        ),
        ("sim.dynamics.orbit.frames", ("FrameContext",)),
    ):
        for class_name in class_names:
            add(_safe_import_type(module_name, class_name))

    # Built-in sensors, estimation and knowledge.  Private helper classes are
    # listed by exact type because they are constructed by the OEL knowledge
    # factory, not discovered by trusting a module path.
    for module_name, class_names in (
        (
            "sim.sensors.models",
            ("OwnStateSensor", "RelativeSensor"),
        ),
        ("sim.sensors.access", ("AccessModel",)),
        ("sim.sensors.composite", ("CompositeSensorModel",)),
        ("sim.sensors.joint_state", ("JointStateSensor",)),
        ("sim.sensors.noisy_own_state", ("NoisyOwnStateSensor",)),
        (
            "sim.knowledge.object_tracking",
            ("ObjectKnowledgeBase", "ObjectDetectionGate", "_OtherObjectStateSensor", "_Track"),
        ),
        ("sim.estimation.aoi", ("AoITrackingEstimator",)),
        ("sim.estimation.attitude_ekf", ("AttitudeEKFEstimator",)),
        ("sim.estimation.joint_ekf", ("JointStateEKFEstimator",)),
        ("sim.estimation.joint_state", ("JointStateEstimator",)),
        ("sim.estimation.orbit_ekf", ("OrbitEKFEstimator",)),
        ("sim.estimation.orbit_ukf", ("OrbitUKFEstimator",)),
        ("sim.estimation.relative_hcw_ekf", ("HCWRelativeEKFEstimator", "SSJ2RelativeEKFEstimator")),
        ("sim.estimation.relative_th_ekf", ("THRelativeEKFEstimator", "YARelativeEKFEstimator")),
        ("sim.estimation.maneuver_detection", ("EKFManeuverDetector",)),
    ):
        for class_name in class_names:
            add(_safe_import_type(module_name, class_name))

    # Complete typed FSW runtime and concrete reference stacks.  Custom stack
    # implementations still require OEL_THREAD_SAFE on their class.
    for module_name, class_names in (
        (
            "sim.runtime.satellites.flight_software_runtime",
            ("SatelliteFlightSoftwareRuntime",),
        ),
        (
            "sim.flight_software.reference_stacks",
            (
                "ReferenceStackBase",
                "PassiveFlightSoftwareStack",
                "AttitudeReferenceFlightSoftwareStack",
                "OrbitReferenceFlightSoftwareStack",
                "RpoReferenceFlightSoftwareStack",
                "LowThrustReferenceFlightSoftwareStack",
            ),
        ),
        ("sim.flight_software.game_stacks", ("GamePilotReferenceFlightSoftwareStack",)),
        (
            "sim.flight_software.rust_game_stacks",
            ("RustGamePilotFlightSoftwareStack", "RustRpoFlightSoftwareStack"),
        ),
        ("sim.flight_software.delivery", ("InputDeliveryQueue",)),
        ("sim.actuators.command_bus", ("ActuatorCommandBus",)),
        (
            "sim.gnc.attitude_v2",
            ("QuaternionTorqueController", "SmallAngleLqrTorqueController", "AttitudeAllocator"),
        ),
        ("sim.gnc.orbit_v2", ("TranslationController", "TranslationAllocator")),
        ("sim.gnc.navigation_v2", ("OrbitNavigator",)),
        (
            "sim.actuators.physical",
            (
                "IdealWrenchHardware",
                "ReactionWheelHardware",
                "MagnetorquerHardware",
                "CmgHardware",
                "ContinuousEngineHardware",
                "RcsThrusterHardware",
                "AerodynamicEffectorHardware",
            ),
        ),
    ):
        for class_name in class_names:
            add(_safe_import_type(module_name, class_name))

    # Rocket simulation and guidance are stateful but object-owned.  Exact
    # classes are safe after the worker clones the complete runtime.
    for module_name, class_names in (
        (
            "sim.rocket.engine",
            ("RocketAscentSimulator",),
        ),
        (
            "sim.rocket.models",
            ("RocketState", "RocketSimConfig", "RocketVehicleConfig"),
        ),
        (
            "sim.rocket.guidance",
            (
                "OpenLoopPitchProgramGuidance",
                "ClosedLoopInsertionGuidance",
                "HoldAttitudeGuidance",
                "TVCSteeringGuidance",
                "MaxQThrottleLimiterGuidance",
                "OrbitInsertionCutoffGuidance",
            ),
        ),
    ):
        for class_name in class_names:
            add(_safe_import_type(module_name, class_name))

    # The GNC catalog is the authority for built-in controllers, strategies,
    # execution modules and legacy mission modules.  Exact identity is still
    # required below, so a user class in a matching module is not enough.
    try:
        from sim.gnc.catalog import catalog_entries

        for entry in catalog_entries(include_internal=True):
            add(_safe_import_type(entry.module, entry.class_name))
    except (ImportError, AttributeError, TypeError):
        pass

    return tuple(candidates)


@lru_cache(maxsize=1)
def _registered_builtin_callables() -> tuple[Callable[..., Any], ...]:
    """Return exact built-in force callables used by ONP and rockets."""

    names = (
        "j2_plugin",
        "j3_plugin",
        "j4_plugin",
        "drag_plugin",
        "lift_plugin",
        "srp_plugin",
        "third_body_moon_plugin",
        "third_body_sun_plugin",
        "third_body_planets_plugin",
        "spherical_harmonics_plugin",
    )
    try:
        module = import_module("sim.dynamics.orbit.propagator")
    except ImportError:
        return ()
    return tuple(
        value for name in names
        if callable(value := getattr(module, name, None))
    )


def _is_registered_builtin(value: Any) -> bool:
    if type(value) in _registered_builtin_types():
        return True
    if callable(value) and any(value is item for item in _registered_builtin_callables()):
        return True
    return False


def _format_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _component_incompatibility(value: Any, path: str) -> str | None:
    """Return one fail-closed reason for an object-owned component."""

    if value is None or isinstance(value, (str, bytes, int, float, bool, complex)):
        return None
    receiver = getattr(value, "__self__", None)
    if (
        _is_registered_builtin(value)
        or is_declared_thread_safe(value)
        or (receiver is not None and _is_registered_builtin(receiver))
    ):
        return None
    if callable(value):
        return (
            f"{path} uses unregistered callback {_format_type(value)}; define "
            f"{THREAD_SAFE_MARKER}=True on its function/class and provide isolated per-object state"
        )
    return (
        f"{path} uses unregistered component {_format_type(value)}; define "
        f"{THREAD_SAFE_MARKER}=True on its class and provide isolated per-object state"
    )


def _iter_callable_values(value: Any, path: str) -> Iterable[tuple[str, Any]]:
    """Yield callbacks in a bounded config/environment container."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_callable_values(item, f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            yield from _iter_callable_values(item, f"{path}[{index}]")
    elif callable(value):
        yield path, value


def _add_component_reason(reasons: list[str], value: Any, path: str) -> None:
    reason = _component_incompatibility(value, path)
    if reason is not None:
        reasons.append(reason)


def _add_callback_reason(
    reasons: list[str],
    callback: Any,
    path: str,
    *,
    owner_only: bool = False,
) -> None:
    """Check a callback and, when present, the receiver it closes over.

    A function marker only describes the function body.  A bound method also
    carries ``__self__``; that receiver must either be one of the isolated
    registered objects or carry its own declaration.  Environment callbacks
    can be invoked from more than one object and therefore use the stricter
    owner-only path even when a provider advertises the marker.
    """

    receiver = getattr(callback, "__self__", None)
    if receiver is not None and not inspect.isclass(receiver):
        _add_component_reason(reasons, receiver, f"{path}.__self__")
    if owner_only:
        reasons.append(
            f"{path} is an owner-only live callback/provider; full thread workers require "
            "an isolated per-object provider"
        )
        return
    _add_component_reason(reasons, callback, path)


def _named_type(value: Any, module_name: str, class_name: str) -> bool:
    """Match a type name only after exact registry admission has succeeded."""

    cls = type(value)
    return cls.__module__ == module_name and cls.__name__ == class_name


def _value_at(value: Any, name: str, default: Any = _MISSING) -> Any:
    """Read an instance field without evaluating a user-defined property."""

    try:
        namespace = vars(value)
    except (AttributeError, TypeError):
        namespace = None
    if namespace is not None and name in namespace:
        return namespace[name]
    # Registered OEL types use ordinary attributes and a few slot-backed
    # values.  Only reach this branch after exact type admission.
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError):
        return default


def _iter_container_values(value: Any, path: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield f"{path}[{key!r}]", item
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            yield f"{path}[{index}]", item
    else:
        yield path, value


def _nested_component_fields(value: Any) -> tuple[str, ...]:
    """Return component-bearing fields for a registered built-in object.

    This is intentionally a small semantic map rather than a recursive walk
    over every dataclass field.  Numeric configs, arrays, RNGs, and telemetry
    state are ordinary object state; callback-bearing fields and nested
    strategy objects are the ownership boundary that needs an admission
    check.
    """

    key = (type(value).__module__, type(value).__name__)
    return {
        ("sim.sensors.models", "OwnStateSensor"): ("access_model",),
        ("sim.sensors.models", "RelativeSensor"): ("access_model",),
        ("sim.sensors.composite", "CompositeSensorModel"): ("sensors",),
        ("sim.knowledge.object_tracking", "ObjectDetectionGate"): ("access",),
        ("sim.knowledge.object_tracking", "_OtherObjectStateSensor"): ("_detection_gate",),
        ("sim.knowledge.object_tracking", "_Track"): ("sensor", "estimator", "maneuver_detector"),
        ("sim.estimation.aoi", "AoITrackingEstimator"): ("base_estimator",),
        ("sim.estimation.joint_state", "JointStateEstimator"): (
            "orbit_estimator",
            "attitude_estimator",
        ),
        ("sim.estimation.joint_ekf", "JointStateEKFEstimator"): (
            "orbit_estimator",
            "attitude_estimator",
        ),
        ("sim.dynamics.orbit.custom_force", "CustomForceModel"): ("acceleration",),
        ("sim.rocket.guidance", "TVCSteeringGuidance"): ("base_guidance",),
        ("sim.rocket.guidance", "MaxQThrottleLimiterGuidance"): ("base_guidance",),
        ("sim.rocket.guidance", "OrbitInsertionCutoffGuidance"): ("base_guidance",),
        ("sim.control.attitude.switching", "DetumbleThenSlewController"): ("nominal", "detumble"),
        ("sim.control.attitude.cmg_steering", "CMGSteeringController"): ("base_controller",),
        ("sim.control.orbit.electric_propulsion", "ElectricPropulsionController"): ("base_controller",),
        ("sim.control.orbit.gimbaled_thruster", "GimbaledThrusterController"): ("base_controller",),
        ("sim.control.orbit.rcs_allocator", "RCSAllocationAwareController"): ("base_controller",),
        ("sim.control.orbit.lqr_curv_variant", "HCWCurvInputRectOutputController"): ("base_lqr",),
        ("sim.control.orbit.reference_rpo", "RICFlyaroundController"): ("_delegate",),
    }.get(key, ())


def _inspect_nested_component(
    reasons: list[str],
    value: Any,
    path: str,
    seen: set[int] | None = None,
) -> None:
    """Inspect a registered object's explicitly component-bearing children."""

    _add_component_reason(reasons, value, path)
    if value is None or isinstance(value, (str, bytes, int, float, bool, complex)):
        return
    if is_declared_thread_safe(value) or not _is_registered_builtin(value):
        return
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    for field_name in _nested_component_fields(value):
        child = _value_at(value, field_name)
        if child is _MISSING or child is None:
            continue
        child_path = f"{path}.{field_name}"
        for item_path, item in _iter_container_values(child, child_path):
            if _named_type(value, "sim.dynamics.orbit.custom_force", "CustomForceModel"):
                _add_callback_reason(reasons, item, item_path)
            else:
                _inspect_nested_component(reasons, item, item_path, seen)


_CONFIG_COMPONENT_NAMES = {
    "attitude_controller",
    "orbit_controller",
    "translation_controller",
    "attitude_allocator",
    "translation_allocator",
    "controller",
    "controllers",
    "sensor",
    "sensors",
    "estimator",
    "estimators",
    "guidance",
    "guidance_law",
    "strategy",
    "strategies",
    "execution",
    "execution_module",
    "delegate",
    "provider",
    "providers",
    "callback",
    "callbacks",
    "hook",
    "hooks",
}


def _is_component_config_field(name: str) -> bool:
    key = str(name).strip().lower()
    return key in _CONFIG_COMPONENT_NAMES or any(
        key.endswith(f"_{suffix}")
        for suffix in (
            "controller",
            "allocator",
            "sensor",
            "estimator",
            "guidance",
            "strategy",
            "execution",
            "provider",
            "callback",
            "hook",
            "delegate",
        )
    )


def _iter_object_fields(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        yield from ((str(key), item) for key, item in value.items())
        return
    try:
        namespace = vars(value)
    except (AttributeError, TypeError):
        return
    yield from namespace.items()


def _inspect_stack_config(
    reasons: list[str],
    config: Any,
    path: str,
    seen: set[int] | None = None,
) -> None:
    """Inspect component hooks in a typed FSW config without walking numbers."""

    if config is None:
        return
    seen = set() if seen is None else seen
    identity = id(config)
    if identity in seen:
        return
    seen.add(identity)
    for name, value in _iter_object_fields(config):
        if not _is_component_config_field(name) or value is None:
            continue
        child_path = f"{path}.{name}"
        key = str(name).lower()
        if "provider" in key or "callback" in key or "hook" in key:
            _inspect_provider_value(reasons, value, child_path)
            continue
        # Allocator/control/executive configs are numeric dataclasses.  Recurse
        # only into their own semantically named fields; a component with a
        # controller/sensor name is checked as a component and cannot hide
        # behind a dataclass wrapper.
        if is_dataclass(value) and type(value).__name__.endswith("Config"):
            _inspect_stack_config(reasons, value, child_path, seen)
        else:
            for item_path, item in _iter_container_values(value, child_path):
                _inspect_nested_component(reasons, item, item_path, seen)


def _is_provider_key(name: Any) -> bool:
    key = str(name).strip().lower()
    return (
        "provider" in key
        or key.endswith("_callback")
        or key.endswith("_callable")
        or "hook" in key
    )


def _is_dependency_key(name: Any) -> bool:
    key = str(name).strip().lower()
    return key in {
        "frame_context",
        "eop_context",
        "ephemeris_context",
        "atmosphere_context",
        "weather_context",
        "native_context",
        "provider_context",
    } or key.endswith("_context") or key.endswith("_handle")


def _inspect_dependency_value(reasons: list[str], value: Any, path: str) -> None:
    """Admit only registered/declared dependency contexts in the environment."""

    if value is None or isinstance(value, (str, bytes, int, float, bool, complex)):
        return
    if _is_registered_builtin(value) or is_declared_thread_safe(value):
        _inspect_nested_component(reasons, value, path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}[{key!r}]"
            if callable(item):
                _add_callback_reason(reasons, item, item_path, owner_only=True)
            elif not isinstance(item, (str, bytes, int, float, bool, complex, type(None))):
                _inspect_dependency_value(reasons, item, item_path)
        return
    _add_component_reason(reasons, value, path)


def _inspect_provider_value(reasons: list[str], value: Any, path: str) -> None:
    """Reject provider pointers/callbacks that refer to owner-side state."""

    for item_path, item in _iter_container_values(value, path):
        if item is None:
            continue
        if callable(item):
            _add_callback_reason(reasons, item, item_path, owner_only=True)
        elif isinstance(item, str):
            reasons.append(
                f"{item_path} is an owner-only provider pointer; full thread workers require "
                "an isolated per-object provider"
            )
        elif not isinstance(item, (int, float, bool, bytes, complex)):
            _add_component_reason(reasons, item, item_path)


def _inspect_environment(reasons: list[str], environment: Any, path: str) -> None:
    if not isinstance(environment, Mapping):
        _add_component_reason(reasons, environment, path)
        return

    mode = str(environment.get("ephemeris_mode", "") or "").strip().lower()
    if mode in {"spice", "spiceypy"}:
        reasons.append(
            f"{path}.ephemeris_mode={mode!r} uses SPICE process-global kernel state and is owner-only"
        )
    kernels = environment.get("spice_kernels")
    if kernels:
        reasons.append(
            f"{path}.spice_kernels configures SPICE process-global kernel state and is owner-only"
        )

    for key, value in environment.items():
        key_path = f"{path}[{key!r}]"
        if _is_provider_key(key):
            _inspect_provider_value(reasons, value, key_path)
        elif _is_dependency_key(key):
            _inspect_dependency_value(reasons, value, key_path)
        elif isinstance(value, Mapping):
            # Extensions are an escape hatch for providers.  Scalar metadata
            # is harmless, but a non-empty provider/module pointer is not an
            # isolated object-owned dependency.
            if str(key).strip().lower() == "extensions":
                for ext_key, ext_value in value.items():
                    ext_path = f"{key_path}[{ext_key!r}]"
                    if _is_provider_key(ext_key):
                        _inspect_provider_value(reasons, ext_value, ext_path)
                    else:
                        for callback_path, callback in _iter_callable_values(ext_value, ext_path):
                            _add_callback_reason(reasons, callback, callback_path, owner_only=True)
            else:
                for callback_path, callback in _iter_callable_values(value, key_path):
                    _add_callback_reason(reasons, callback, callback_path, owner_only=True)
        elif callable(value):
            _add_callback_reason(reasons, value, key_path, owner_only=True)


def _inspect_dynamics(reasons: list[str], dynamics: Any, path: str) -> None:
    if dynamics is None:
        return
    _add_component_reason(reasons, dynamics, path)
    if not (_is_registered_builtin(dynamics) or is_declared_thread_safe(dynamics)):
        return
    propagator = getattr(dynamics, "orbit_propagator", None)
    if propagator is not None:
        propagator_path = f"{path}.orbit_propagator"
        _inspect_nested_component(reasons, propagator, propagator_path)
        for index, plugin in enumerate(getattr(propagator, "plugins", ()) or ()):
            plugin_path = f"{propagator_path}.plugins[{index}]"
            _inspect_nested_component(reasons, plugin, plugin_path)
        for name in (
            "_rust_force_context_cache",
            "_rust_environment_context_cache",
            "_rust_mechanical_context_cache",
            "_rust_prepared_frame_cache",
            "_rust_forecast_zonal_context",
            "_rust_stage_context_cache",
            "_rust_atmosphere_frame_cache",
        ):
            native_context = _value_at(propagator, name)
            if native_context is not _MISSING and native_context is not None:
                # The force-plan adapter initializes frame scratch lazily.
                # An empty built-in dictionary contains no native provider
                # and is isolated by the worker's pickle round-trip.
                if type(native_context) is dict and not native_context:
                    continue
                _add_component_reason(
                    reasons,
                    native_context,
                    f"{propagator_path}.{name}",
                )
    for name in ("disturbance_model", "resource_model", "geometry_area_profile"):
        component = _value_at(dynamics, name, None)
        if component is not None:
            component_path = f"{path}.{name}"
            _inspect_nested_component(reasons, component, component_path)
            if _named_type(component, "sim.dynamics.attitude.disturbances", "DisturbanceTorqueModel"):
                native_context = _value_at(component, "_compiled_native_context_cache", None)
                if native_context is not None:
                    _add_component_reason(
                        reasons,
                        native_context,
                        f"{component_path}._compiled_native_context_cache",
                    )
                config = _value_at(component, "config", None)
                geometry = _value_at(config, "geometry_area_profile", None) if config is not None else None
                if geometry is not None:
                    _inspect_nested_component(
                        reasons,
                        geometry,
                        f"{component_path}.config.geometry_area_profile",
                    )


def _inspect_flight_software(reasons: list[str], runtime: Any, path: str) -> None:
    if runtime is None:
        return
    _add_component_reason(reasons, runtime, path)
    if not (_is_registered_builtin(runtime) or is_declared_thread_safe(runtime)):
        return
    stack = _value_at(runtime, "stack", None)
    _inspect_nested_component(reasons, stack, f"{path}.stack")
    if stack is not None and (_is_registered_builtin(stack) or is_declared_thread_safe(stack)):
        _inspect_stack_config(reasons, _value_at(stack, "config", None), f"{path}.stack.config")
    _inspect_nested_component(reasons, _value_at(runtime, "command_bus", None), f"{path}.command_bus")
    for actuator_id, hardware in dict(_value_at(runtime, "hardware", {}) or {}).items():
        _inspect_nested_component(reasons, hardware, f"{path}.hardware[{actuator_id!r}]")
    publishers = tuple(_value_at(runtime, "_input_publishers", ()) or ())
    for index, publisher in enumerate(publishers):
        _add_callback_reason(
            reasons,
            publisher,
            f"{path}._input_publishers[{index}]",
            owner_only=True,
        )
    _inspect_nested_component(reasons, _value_at(runtime, "_target_access", None), f"{path}._target_access")
    profile_params = _value_at(runtime, "profile_params", None)
    if isinstance(profile_params, Mapping):
        for key, value in profile_params.items():
            item_path = f"{path}.profile_params[{key!r}]"
            if _is_provider_key(key):
                _inspect_provider_value(reasons, value, item_path)
            else:
                for callback_path, callback in _iter_callable_values(value, item_path):
                    _add_callback_reason(reasons, callback, callback_path)


def _inspect_knowledge(reasons: list[str], knowledge: Any, path: str) -> None:
    if knowledge is None:
        return
    _add_component_reason(reasons, knowledge, path)
    if not (_is_registered_builtin(knowledge) or is_declared_thread_safe(knowledge)):
        return
    for target_id, track in dict(getattr(knowledge, "_tracks", {}) or {}).items():
        track_path = f"{path}._tracks[{target_id!r}]"
        _inspect_nested_component(reasons, track, track_path)


def _inspect_agent(reasons: list[str], agent: Any, path: str) -> None:
    if agent is None:
        return
    for name in (
        "sensor",
        "estimator",
        "orbit_controller",
        "attitude_controller",
        "mission_strategy",
        "mission_execution",
        "actuator",
        "rocket_guidance",
        "rocket_sim",
    ):
        _inspect_nested_component(reasons, getattr(agent, name, None), f"{path}.{name}")
    for index, module in enumerate(getattr(agent, "mission_modules", ()) or ()):
        _inspect_nested_component(reasons, module, f"{path}.mission_modules[{index}]")
    _inspect_dynamics(reasons, getattr(agent, "dynamics", None), f"{path}.dynamics")
    _inspect_knowledge(reasons, getattr(agent, "knowledge_base", None), f"{path}.knowledge_base")
    _inspect_flight_software(
        reasons,
        getattr(agent, "flight_software_runtime", None),
        f"{path}.flight_software_runtime",
    )
    bridge = getattr(agent, "bridge", None)
    if bridge is not None:
        _inspect_nested_component(reasons, bridge, f"{path}.bridge")


def thread_worker_incompatibilities(engine: Any, active_ids: Iterable[str]) -> list[str]:
    """Return deterministic reasons a full object-step thread pool is unsafe.

    The function is deliberately read-only.  It does not import arbitrary
    scenario modules, execute callbacks, or mutate the engine.  A caller may
    use a non-empty result to select the existing process executor (or serial
    execution) and should report the returned reasons in its execution plan.
    """

    reasons: list[str] = []
    ids = [str(value) for value in active_ids]
    agents = getattr(engine, "agents", {}) or {}

    # These are the same lifecycle boundaries that make the current process
    # object executor ineligible, plus coupled paths that cannot be split into
    # independent object steps.
    if getattr(engine, "general_propagation", None):
        reasons.append("general propagation providers require the process/general-propagation path")
    simulator = getattr(getattr(engine, "cfg", None), "simulator", None)
    configured_system_forces = getattr(simulator, "system_force_models", ()) or ()
    if getattr(engine, "system_force_stepper", None) is not None or configured_system_forces:
        reasons.append("system force stepping couples objects and requires its coordinated executor")
    configured_collisions = getattr(simulator, "collisions", {}) or {}
    if getattr(engine, "collision_stepper", None) is not None or bool(
        isinstance(configured_collisions, Mapping) and configured_collisions.get("enabled", False)
    ):
        reasons.append("collision stepping couples objects and requires its coordinated executor")
    if any(
        str(getattr(agent, "deploy_source", "") or "") in {"rocket_deployment", "rocket_insertion"}
        for agent in agents.values()
    ):
        reasons.append("dynamic rocket deployment changes object-worker membership")

    # The environment is shared by every object step.  Even a marked callback
    # can retain a live owner, native handle, or process-global provider, so
    # callbacks/providers at this boundary are owner-only.  SPICE is called
    # out separately because SpiceyPy's kernel pool is process-global even
    # when the config contains only plain paths and strings.
    _inspect_environment(
        reasons,
        getattr(engine, "base_environment", {}),
        "engine.base_environment",
    )

    for aid in ids:
        agent = agents.get(aid)
        if agent is None:
            reasons.append(f"active object {aid!r} is missing from engine.agents")
            continue
        _inspect_agent(reasons, agent, f"object {aid!r}")

    # System force models may be stored in a coordinated stepper even when a
    # caller constructed it lazily.  Inspect their callbacks if present.
    system_force_stepper = getattr(engine, "system_force_stepper", None)
    for index, entry in enumerate(getattr(system_force_stepper, "models", ()) or ()):
        callback = entry[1] if isinstance(entry, tuple) and len(entry) > 1 else entry
        _add_callback_reason(reasons, callback, f"engine.system_force_stepper.models[{index}]")

    # Preserve order while removing duplicate reasons from nested references.
    return list(dict.fromkeys(reasons))


__all__ = [
    "THREAD_CONTRACT_VERSION",
    "THREAD_SAFE_MARKER",
    "is_declared_thread_safe",
    "thread_worker_incompatibilities",
]
