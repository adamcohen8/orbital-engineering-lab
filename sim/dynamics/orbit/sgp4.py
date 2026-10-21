from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from sim.acceleration.settings import acceleration_settings_from_mode
from sim.core.models import StateTruth
from sim.dynamics.orbit.frames import teme_to_eci_vallado_iau80
from sim.dynamics.orbit.tle import TLEElements, ogp_mean_elements_from_mapping, parse_tle_lines
from sim.numeric_backend import normalize_numeric_backend

try:  # pragma: no cover - availability is environment-dependent.
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range

SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN = 225.0
SGP4_DEEP_SPACE_MEAN_MOTION_THRESHOLD_REV_DAY = 1440.0 / SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN


@dataclass(frozen=True)
class SGP4State:
    position_teme_km: np.ndarray
    velocity_teme_km_s: np.ndarray
    error: str | None = None


@dataclass(frozen=True)
class SGP4EphemerisState:
    """Configured-frame state returned by ``configured_state_at()``.

    The neutral position/velocity names prevent native TEME products from being
    represented by fields whose names claim ECI coordinates. The ECI aliases
    remain available for explicitly ECI output to preserve that direct API.
    """

    position_km: np.ndarray
    velocity_km_s: np.ndarray
    frame: str
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    mass_kg: float
    t_s: float

    @property
    def position_eci_km(self) -> np.ndarray:
        if self.frame != "eci":
            raise AttributeError("position_eci_km is unavailable for a non-ECI ephemeris state.")
        return self.position_km

    @property
    def velocity_eci_km_s(self) -> np.ndarray:
        if self.frame != "eci":
            raise AttributeError("velocity_eci_km_s is unavailable for a non-ECI ephemeris state.")
        return self.velocity_km_s


@dataclass(frozen=True)
class SGP4BatchResult:
    """Array-shaped SGP4 output contract for reference and accelerated backends."""

    backend: str
    tsince_min: np.ndarray
    position_teme_km: np.ndarray
    velocity_teme_km_s: np.ndarray
    errors: np.ndarray
    execution_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def success(self) -> np.ndarray:
        return self.errors == ""

    @property
    def object_count(self) -> int:
        return int(self.position_teme_km.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.position_teme_km.shape[1])


@dataclass(frozen=True)
class SGP4PropagationMetadata:
    propagation_method: str
    propagator_family: str
    propagator_name: str
    general_model: str
    native_frame: str
    output_frame: str
    state_history_frame: str
    frame_transform: str
    propagation_backend: str
    requested_backend: str
    backend_fallback_reason: str
    numerical_equivalence: str
    tle_epoch_jd_utc: float
    tle_age_start_days: float
    tle_age_end_days: float
    ogp_regime: str
    orbital_period_min: float
    max_tle_age_days_warning: float | None
    tle_age_warning: bool


@dataclass(frozen=True)
class SGP4EphemerisProvider:
    elements: TLEElements
    mass_kg: float
    start_jd_utc: float
    duration_s: float
    output_frame: str = "teme"
    frame_transform: str = "native"
    attitude_quat_bn: np.ndarray | None = None
    angular_rate_body_rad_s: np.ndarray | None = None
    max_tle_age_days_warning: float | None = None
    acceleration_mode: str = "off"
    numeric_backend: str = "rust"
    _sdp4_context: object | None = field(default=None, init=False, repr=False, compare=False)
    _rust_context: object | None = field(default=None, init=False, repr=False, compare=False)
    _sgp4_numeric_elements: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _propagation_backend: str = field(default="python_scalar", init=False, repr=False, compare=False)
    _requested_backend: str = field(default="python_scalar", init=False, repr=False, compare=False)
    _backend_fallback_reason: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in ("mass_kg", "start_jd_utc", "duration_s"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if float(self.mass_kg) <= 0.0:
            raise ValueError("mass_kg must be positive.")
        if float(self.duration_s) < 0.0:
            raise ValueError("duration_s must be nonnegative.")
        frame = str(self.output_frame or "teme").strip().lower()
        transform = str(self.frame_transform or "").strip().lower()
        object.__setattr__(self, "output_frame", frame)
        object.__setattr__(self, "frame_transform", transform)
        if frame == "teme" and not transform:
            object.__setattr__(self, "frame_transform", "native")
        if frame == "eci" and not transform:
            object.__setattr__(self, "frame_transform", "teme_to_eci_iau80")
        backend = normalize_numeric_backend(self.numeric_backend, error_message="OGP numeric_backend must be python or rust.")
        object.__setattr__(self, "numeric_backend", backend)
        if backend == "rust":
            from sim.rust_ogp_backend import RustOGPContext

            object.__setattr__(self, "_rust_context", RustOGPContext(self.elements))
            object.__setattr__(self, "_propagation_backend", "rust_scalar")
            object.__setattr__(self, "_requested_backend", "rust_scalar")
        elif sgp4_orbital_period_min(self.elements) >= SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN:
            from sim.dynamics.orbit.sdp4 import sdp4_initialize

            object.__setattr__(self, "_sdp4_context", sdp4_initialize(self.elements))
            object.__setattr__(self, "_propagation_backend", "sdp4_context")
            object.__setattr__(self, "_requested_backend", "sdp4_context")
        else:
            acceleration = acceleration_settings_from_mode(
                self.acceleration_mode,
                allow_env_override=False,
            )
            object.__setattr__(self, "acceleration_mode", acceleration.requested_mode)
            requested_backend = "numba_scalar" if acceleration.requested_mode != "off" else "python_scalar"
            object.__setattr__(self, "_requested_backend", requested_backend)
            if acceleration.enabled and njit is not None:
                numeric_elements = _sgp4_numeric_element_array([self.elements])[0]
                numeric_elements.setflags(write=False)
                object.__setattr__(self, "_sgp4_numeric_elements", numeric_elements)
                object.__setattr__(self, "_propagation_backend", "numba_scalar")
            else:
                object.__setattr__(self, "_propagation_backend", "python_scalar")
                if requested_backend == "numba_scalar":
                    object.__setattr__(self, "_backend_fallback_reason", acceleration.reason or "Numba unavailable")
        threshold = self.max_tle_age_days_warning
        if threshold is not None:
            threshold = float(threshold)
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError("max_tle_age_days_warning must be a nonnegative finite number.")
            object.__setattr__(self, "max_tle_age_days_warning", threshold)
            start_age = abs(float(self.start_jd_utc) - float(self.elements.epoch_jd_utc))
            end_age = abs(
                float(self.start_jd_utc)
                + float(self.duration_s) / 86400.0
                - float(self.elements.epoch_jd_utc)
            )
            if max(start_age, end_age) > threshold:
                warnings.warn(
                    f"TLE age reaches {max(start_age, end_age):.6g} days, exceeding "
                    f"max_tle_age_days_warning={threshold:.6g}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @classmethod
    def from_tle_block(
        cls,
        tle_block: dict,
        *,
        mass_kg: float,
        start_jd_utc: float | None,
        duration_s: float,
        output_frame: str = "teme",
        frame_transform: str | None = None,
        attitude_quat_bn: np.ndarray | None = None,
        angular_rate_body_rad_s: np.ndarray | None = None,
        max_tle_age_days_warning: float | None = None,
        acceleration_mode: str = "off",
        numeric_backend: str = "rust",
    ) -> SGP4EphemerisProvider:
        block = dict(tle_block or {})
        lines = block.get("lines")
        if isinstance(lines, (list, tuple)) and len(lines) >= 2:
            line1 = str(lines[0])
            line2 = str(lines[1])
        else:
            line1 = str(block.get("line1", "") or "")
            line2 = str(block.get("line2", "") or "")
        elements = parse_tle_lines(line1, line2, require_checksum=bool(block.get("require_checksum", True)))
        if float(elements.mean_motion_rev_per_day) <= 0.0:
            raise ValueError("OGP mean motion must be positive.")
        if float(elements.eccentricity) < 0.0 or float(elements.eccentricity) >= 1.0:
            raise ValueError("OGP eccentricity must be in [0, 1).")
        resolved_start = float(elements.epoch_jd_utc if start_jd_utc is None else start_jd_utc)
        resolved_output_frame = str(output_frame or "teme").strip().lower()
        resolved_frame_transform = str(frame_transform or "").strip().lower()
        if not resolved_frame_transform:
            resolved_frame_transform = "native" if resolved_output_frame == "teme" else "teme_to_eci_iau80"
        return cls(
            elements=elements,
            mass_kg=float(mass_kg),
            start_jd_utc=resolved_start,
            duration_s=float(duration_s),
            output_frame=resolved_output_frame,
            frame_transform=resolved_frame_transform,
            attitude_quat_bn=attitude_quat_bn,
            angular_rate_body_rad_s=angular_rate_body_rad_s,
            max_tle_age_days_warning=max_tle_age_days_warning,
            acceleration_mode=acceleration_mode,
            numeric_backend=numeric_backend,
        )

    @classmethod
    def from_mean_elements(
        cls,
        mean_elements: dict,
        **kwargs,
    ) -> SGP4EphemerisProvider:
        """Construct an OGP provider from native fitted elements, not TLE text."""

        elements = ogp_mean_elements_from_mapping(mean_elements)
        resolved_start = kwargs.pop("start_jd_utc", None)
        return cls(
            elements=elements,
            start_jd_utc=float(elements.epoch_jd_utc if resolved_start is None else resolved_start),
            **kwargs,
        )

    def state_at(self, t_s: float) -> StateTruth:
        """Return the historical configured-frame ``StateTruth`` contract.

        Native TEME output predates the ECI-specific field names on
        :class:`StateTruth`.  That behavior remains for API compatibility, but
        callers selecting a non-ECI product frame are directed to
        :meth:`configured_state_at`, whose fields and frame metadata cannot
        mislabel TEME as ECI.
        """

        if self.output_frame != "eci":
            warnings.warn(
                "SGP4EphemerisProvider.state_at() preserves the legacy StateTruth "
                "contract for non-ECI output; use configured_state_at() for an "
                "explicit, frame-neutral product state.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._truth_state_at(
            t_s,
            output_frame=self.output_frame,
            frame_transform=self.frame_transform,
        )

    def configured_state_at(self, t_s: float) -> SGP4EphemerisState:
        """Return the configured product state with explicit frame metadata."""

        t_s = self._validated_time(t_s)
        pos, vel = self._position_velocity_at(
            t_s,
            output_frame=self.output_frame,
            frame_transform=self.frame_transform,
        )
        return SGP4EphemerisState(
            position_km=pos,
            velocity_km_s=vel,
            frame=str(self.output_frame),
            attitude_quat_bn=self._attitude_quat(),
            angular_rate_body_rad_s=self._angular_rate_body(),
            mass_kg=float(self.mass_kg),
            t_s=t_s,
        )

    def canonical_state_at(self, t_s: float) -> StateTruth:
        """Return the engine's canonical ECI truth state regardless of product frame."""

        return self._truth_state_at(
            t_s,
            output_frame="eci",
            frame_transform="teme_to_eci_iau80",
        )

    def canonical_states_at(self, times_s: np.ndarray | list[float]) -> np.ndarray:
        """Evaluate exact requested ECI epochs in one Rust OGP batch.

        This numerical preparation does not advance any engine event, sensor,
        or output boundary. Arbitrary query ordering retains the context's
        resonance-reset contract. Other backends retain the scalar owner.
        """

        times = np.asarray(times_s, dtype=float).reshape(-1)
        for time in times:
            self._validated_time(float(time))
        if times.size == 0:
            return np.empty((0, 6), dtype=float)
        if self._rust_context is None:
            return np.asarray([
                np.concatenate(self._position_velocity_at(float(time), output_frame="eci", frame_transform="teme_to_eci_iau80"))
                for time in times
            ])
        offsets = (float(self.start_jd_utc) - float(self.elements.epoch_jd_utc)) * 1440.0 + times / 60.0
        positions, velocities, errors = self._rust_context.propagate_many(offsets)
        for error in errors:
            if error:
                raise ValueError(str(error))
        from sim.dynamics.orbit.frames import teme_to_eci_matrix_vallado_iau80

        rotations = np.asarray([
            teme_to_eci_matrix_vallado_iau80(float(self.start_jd_utc) + float(time) / 86400.0)
            for time in times
        ])
        positions_eci = (rotations @ positions[..., None])[..., 0]
        velocities_eci = (rotations @ velocities[..., None])[..., 0]
        return np.column_stack((positions_eci, velocities_eci))

    def _truth_state_at(self, t_s: float, *, output_frame: str, frame_transform: str) -> StateTruth:
        t_s = self._validated_time(t_s)
        pos, vel = self._position_velocity_at(
            t_s,
            output_frame=output_frame,
            frame_transform=frame_transform,
        )
        return StateTruth(
            position_eci_km=pos,
            velocity_eci_km_s=vel,
            attitude_quat_bn=self._attitude_quat(),
            angular_rate_body_rad_s=self._angular_rate_body(),
            mass_kg=float(self.mass_kg),
            t_s=t_s,
        )

    def _position_velocity_at(
        self,
        t_s: float,
        *,
        output_frame: str,
        frame_transform: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        jd_utc = float(self.start_jd_utc) + t_s / 86400.0
        tsince_min = (float(self.start_jd_utc) - float(self.elements.epoch_jd_utc)) * 1440.0 + t_s / 60.0
        if self._rust_context is not None:
            position, velocity = self._rust_context.propagate(tsince_min)
            native = SGP4State(position, velocity)
        elif self._sdp4_context is not None:
            from sim.dynamics.orbit.sdp4 import sdp4_propagate_teme_from_context

            native = sdp4_propagate_teme_from_context(self._sdp4_context, tsince_min)
        elif self._propagation_backend == "numba_scalar" and self._sgp4_numeric_elements is not None:
            position = np.empty(3, dtype=float)
            velocity = np.empty(3, dtype=float)
            try:
                error_code = _sgp4_numba_state(
                    *self._sgp4_numeric_elements,
                    float(tsince_min),
                    position,
                    velocity,
                )
            except Exception as exc:  # pragma: no cover - backend/runtime dependent.
                object.__setattr__(self, "_propagation_backend", "python_scalar")
                object.__setattr__(
                    self,
                    "_backend_fallback_reason",
                    f"{type(exc).__name__}: {exc}",
                )
                native = sgp4_propagate_teme(self.elements, tsince_min)
            else:
                if int(error_code) == 0:
                    native = SGP4State(position, velocity)
                else:
                    # Preserve the scalar reference's detailed error contract;
                    # accelerated error codes are only a fast success-path gate.
                    native = sgp4_propagate_teme(self.elements, tsince_min)
        else:
            native = sgp4_propagate_teme(self.elements, tsince_min)
        if native.error:
            raise ValueError(native.error)
        pos, vel = transform_teme_to_output_frame(
            native.position_teme_km,
            native.velocity_teme_km_s,
            jd_utc=jd_utc,
            output_frame=output_frame,
            frame_transform=frame_transform,
        )
        return np.array(pos, dtype=float), np.array(vel, dtype=float)

    def _validated_time(self, t_s: float) -> float:
        value = float(t_s)
        if not math.isfinite(value):
            raise ValueError("t_s must be finite.")
        if value < 0.0 or value > float(self.duration_s) + 1.0e-12:
            raise ValueError(f"t_s must be within [0, {float(self.duration_s):g}] seconds.")
        return value

    def _attitude_quat(self) -> np.ndarray:
        return (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if self.attitude_quat_bn is None
            else np.array(self.attitude_quat_bn, dtype=float)
        )

    def _angular_rate_body(self) -> np.ndarray:
        return (
            np.zeros(3, dtype=float)
            if self.angular_rate_body_rad_s is None
            else np.array(self.angular_rate_body_rad_s, dtype=float)
        )

    def metadata(self) -> SGP4PropagationMetadata:
        start_age = float(self.start_jd_utc) - float(self.elements.epoch_jd_utc)
        end_age = start_age + float(self.duration_s) / 86400.0
        orbital_period_min = sgp4_orbital_period_min(self.elements)
        propagator_name = (
            "OGP-SDP4"
            if orbital_period_min >= SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN
            else "OGP-SGP4"
        )
        return SGP4PropagationMetadata(
            propagation_method="general",
            propagator_family="OGP",
            propagator_name=propagator_name,
            general_model="sgp4",
            native_frame="teme",
            output_frame=self.output_frame,
            state_history_frame="eci",
            frame_transform=self.frame_transform,
            propagation_backend=self._propagation_backend,
            requested_backend=self._requested_backend,
            backend_fallback_reason=self._backend_fallback_reason,
            numerical_equivalence=(
                "rounding_level"
                if self._propagation_backend == "numba_scalar"
                else "cross_implementation"
                if self._propagation_backend == "rust_scalar"
                else "bitwise_reference"
            ),
            tle_epoch_jd_utc=float(self.elements.epoch_jd_utc),
            tle_age_start_days=float(start_age),
            tle_age_end_days=float(end_age),
            ogp_regime="deep_space" if propagator_name == "OGP-SDP4" else "near_earth",
            orbital_period_min=float(orbital_period_min),
            max_tle_age_days_warning=self.max_tle_age_days_warning,
            tle_age_warning=bool(
                self.max_tle_age_days_warning is not None
                and max(abs(start_age), abs(end_age)) > float(self.max_tle_age_days_warning)
            ),
        )


def sgp4_orbital_period_min(elements: TLEElements) -> float:
    """Return the SGP4-corrected (un-Kozai) orbital period in minutes."""

    mean_motion = float(elements.mean_motion_rev_per_day)
    if mean_motion <= 0.0:
        return math.inf
    eccentricity = float(elements.eccentricity)
    if not 0.0 <= eccentricity < 1.0:
        return math.nan
    xno = mean_motion * 2.0 * math.pi / 1440.0
    xke = math.sqrt((3600.0 * 398600.8) / (6378.135**3))
    a1 = (xke / xno) ** (2.0 / 3.0)
    theta2 = math.cos(math.radians(float(elements.inclination_deg))) ** 2
    beta2 = 1.0 - eccentricity * eccentricity
    beta = math.sqrt(beta2)
    ck2 = 1.0826158e-3 / 2.0
    x3thm1 = 3.0 * theta2 - 1.0
    del1 = 1.5 * ck2 * x3thm1 / ((a1 * a1) * beta * beta2)
    ao = a1 * (1.0 - del1 * (1.0 / 3.0 + del1 * (1.0 + 134.0 / 81.0 * del1)))
    delo = 1.5 * ck2 * x3thm1 / ((ao * ao) * beta * beta2)
    no_unkozai = xno / (1.0 + delo)
    return 2.0 * math.pi / no_unkozai


def sgp4_unsupported_reason(elements: TLEElements) -> str | None:
    period_min = sgp4_orbital_period_min(elements)
    if period_min >= SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN:
        return (
            "OEL OGP-SGP4 supports near-Earth SGP4 only; OGP-SDP4/deep-space SDP4/resonance TLEs "
            f"with orbital period >= {SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN:.0f} min are not supported "
            f"(period={period_min:.3f} min, mean_motion={float(elements.mean_motion_rev_per_day):.8f} rev/day)."
        )
    return None


def _actan(y: float, x: float) -> float:
    return float(math.atan2(y, x) % (2.0 * math.pi))


def _fmod2p(x: float) -> float:
    return float(x % (2.0 * math.pi))


def _sgp4_satdata(elements: TLEElements) -> dict[str, float]:
    minutes_per_day = 1440.0
    return {
        "xmo": math.radians(float(elements.mean_anomaly_deg)),
        "xnodeo": math.radians(float(elements.raan_deg)),
        "omegao": math.radians(float(elements.argp_deg)),
        "xincl": math.radians(float(elements.inclination_deg)),
        "eo": float(elements.eccentricity),
        "xno": float(elements.mean_motion_rev_per_day) * 2.0 * math.pi / minutes_per_day,
        "xndt2o": float(elements.mean_motion_derivative_rev_per_day2)
        * 2.0
        * math.pi
        / (minutes_per_day * minutes_per_day),
        "xndd6o": float(elements.mean_motion_second_derivative_rev_per_day3)
        * 2.0
        * math.pi
        / (minutes_per_day * minutes_per_day * minutes_per_day),
        "bstar": float(elements.bstar),
    }


def sgp4_propagate_teme(elements: TLEElements, tsince_min: float) -> SGP4State:
    try:
        tsince = float(tsince_min)
    except (TypeError, ValueError):
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 time offset must be finite.")
    if not math.isfinite(tsince):
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 time offset must be finite.")
    satdata = _sgp4_satdata(elements)
    ae = 1.0
    tothrd = 2.0 / 3.0
    xj3 = -2.53881e-6
    e6a = 1.0e-6
    xkmper = 6378.135
    ge = 398600.8
    ck2 = 1.0826158e-3 / 2.0
    ck4 = -3.0 * -1.65597e-6 / 8.0

    unsupported = sgp4_unsupported_reason(elements)
    if unsupported:
        return SGP4State(np.zeros(3), np.zeros(3), unsupported)
    if satdata["xno"] <= 0.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 mean motion must be positive.")
    if satdata["eo"] < 0.0 or satdata["eo"] >= 1.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 eccentricity must be in [0, 1).")

    s = ae + 78.0 / xkmper
    qo = ae + 120.0 / xkmper
    xke = math.sqrt((3600.0 * ge) / (xkmper**3))
    qoms2t = (qo - s) ** 4
    a1 = (xke / satdata["xno"]) ** tothrd
    cosio = math.cos(satdata["xincl"])
    theta2 = cosio * cosio
    x3thm1 = 3.0 * theta2 - 1.0
    eosq = satdata["eo"] * satdata["eo"]
    betao2 = 1.0 - eosq
    betao = math.sqrt(betao2)
    del1 = 1.5 * ck2 * x3thm1 / ((a1 * a1) * betao * betao2)
    ao = a1 * (1.0 - del1 * (1.0 / 3.0 + del1 * (1.0 + 134.0 / 81.0 * del1)))
    delo = 1.5 * ck2 * x3thm1 / ((ao * ao) * betao * betao2)
    xnodp = satdata["xno"] / (1.0 + delo)
    aodp = ao / (1.0 - delo)

    isimp = (aodp * (1.0 - satdata["eo"]) / ae) < (220.0 / xkmper + ae)
    s4 = s
    qoms24 = qoms2t
    perige = (aodp * (1.0 - satdata["eo"]) - ae) * xkmper
    if perige < 156.0:
        s4 = perige - 78.0
        if perige <= 98.0:
            s4 = 20.0
        qoms24 = ((120.0 - s4) * ae / xkmper) ** 4
        s4 = s4 / xkmper + ae

    pinvsq = 1.0 / ((aodp * aodp) * (betao2 * betao2))
    tsi = 1.0 / (aodp - s4)
    eta = aodp * satdata["eo"] * tsi
    etasq = eta * eta
    eeta = satdata["eo"] * eta
    psisq = abs(1.0 - etasq)
    coef = qoms24 * (tsi**4)
    coef1 = coef / (psisq**3.5)
    c2 = coef1 * xnodp * (
        aodp * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
        + 0.75 * ck2 * tsi / psisq * x3thm1 * (8.0 + 3.0 * etasq * (8.0 + etasq))
    )
    c1 = satdata["bstar"] * c2
    sinio = math.sin(satdata["xincl"])
    a3ovk2 = -xj3 / ck2 * (ae**3)
    c3 = 0.0 if abs(satdata["eo"]) < 1.0e-12 else coef * tsi * a3ovk2 * xnodp * ae * sinio / satdata["eo"]
    x1mth2 = 1.0 - theta2
    c4 = 2.0 * xnodp * coef1 * aodp * betao2 * (
        eta * (2.0 + 0.5 * etasq)
        + satdata["eo"] * (0.5 + 2.0 * etasq)
        - 2.0
        * ck2
        * tsi
        / (aodp * psisq)
        * (
            -3.0 * x3thm1 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
            + 0.75
            * x1mth2
            * (2.0 * etasq - eeta * (1.0 + etasq))
            * math.cos(2.0 * satdata["omegao"])
        )
    )
    c5 = 2.0 * coef1 * aodp * betao2 * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
    theta4 = theta2 * theta2
    temp1 = 3.0 * ck2 * pinvsq * xnodp
    temp2 = temp1 * ck2 * pinvsq
    temp3 = 1.25 * ck4 * pinvsq * pinvsq * xnodp
    xmdot = xnodp + 0.5 * temp1 * betao * x3thm1 + 0.0625 * temp2 * betao * (13.0 - 78.0 * theta2 + 137.0 * theta4)
    x1m5th = 1.0 - 5.0 * theta2
    omgdot = -0.5 * temp1 * x1m5th + 0.0625 * temp2 * (7.0 - 114.0 * theta2 + 395.0 * theta4) + temp3 * (3.0 - 36.0 * theta2 + 49.0 * theta4)
    xhdot1 = -temp1 * cosio
    xnodot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * theta2) + 2.0 * temp3 * (3.0 - 7.0 * theta2)) * cosio
    omgcof = satdata["bstar"] * c3 * math.cos(satdata["omegao"])
    xmcof = 0.0 if abs(eeta) < 1.0e-12 else -(2.0 / 3.0) * coef * satdata["bstar"] * ae / eeta
    xnodcf = 3.5 * betao2 * xhdot1 * c1
    t2cof = 1.5 * c1
    xlcof_denominator = 1.0 + cosio
    if abs(xlcof_denominator) < 1.5e-12:
        xlcof_denominator = 1.5e-12
    xlcof = 0.125 * a3ovk2 * sinio * (3.0 + 5.0 * cosio) / xlcof_denominator
    aycof = 0.25 * a3ovk2 * sinio
    delmo = (1.0 + eta * math.cos(satdata["xmo"])) ** 3
    sinmo = math.sin(satdata["xmo"])
    x7thm1 = 7.0 * theta2 - 1.0
    d2 = d3 = d4 = t3cof = t4cof = t5cof = 0.0
    if not isimp:
        c1sq = c1 * c1
        d2 = 4.0 * aodp * tsi * c1sq
        temp = d2 * tsi * c1 / 3.0
        d3 = (17.0 * aodp + s4) * temp
        d4 = 0.5 * temp * aodp * tsi * (221.0 * aodp + 31.0 * s4) * c1
        t3cof = d2 + 2.0 * c1sq
        t4cof = 0.25 * (3.0 * d3 + c1 * (12.0 * d2 + 10.0 * c1sq))
        t5cof = 0.2 * (3.0 * d4 + 12.0 * c1 * d3 + 6.0 * d2 * d2 + 15.0 * c1sq * (2.0 * d2 + c1sq))

    xmdf = satdata["xmo"] + xmdot * tsince
    omgadf = satdata["omegao"] + omgdot * tsince
    xnoddf = satdata["xnodeo"] + xnodot * tsince
    omega = omgadf
    xmp = xmdf
    tsq = tsince * tsince
    xnode = xnoddf + xnodcf * tsq
    tempa = 1.0 - c1 * tsince
    tempe = satdata["bstar"] * c4 * tsince
    templ = t2cof * tsq
    if not isimp:
        delomg = omgcof * tsince
        delm = xmcof * ((1.0 + eta * math.cos(xmdf)) ** 3 - delmo)
        temp = delomg + delm
        xmp = xmdf + temp
        omega = omgadf - temp
        tcube = tsq * tsince
        tfour = tsince * tcube
        tempa = tempa - d2 * tsq - d3 * tcube - d4 * tfour
        tempe = tempe + satdata["bstar"] * c5 * (math.sin(xmp) - sinmo)
        templ = templ + t3cof * tcube + tfour * (t4cof + tsince * t5cof)

    a = aodp * (tempa * tempa)
    e = satdata["eo"] - tempe
    if a <= 0.0 or e < 0.0 or e >= 1.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 propagated orbit became invalid.")
    xl = xmp + omega + xnode + xnodp * templ
    beta = math.sqrt(1.0 - e * e)
    xn = xke / (a**1.5)
    axn = e * math.cos(omega)
    temp = 1.0 / (a * beta * beta)
    xll = temp * xlcof * axn
    aynl = temp * aycof
    xlt = xl + xll
    ayn = e * math.sin(omega) + aynl

    capu = _fmod2p(xlt - xnode)
    epw = capu
    sinepw = cosepw = temp3 = temp4 = temp5 = temp6 = 0.0
    for _ in range(10):
        sinepw = math.sin(epw)
        cosepw = math.cos(epw)
        temp3 = axn * sinepw
        temp4 = ayn * cosepw
        temp5 = axn * cosepw
        temp6 = ayn * sinepw
        next_epw = (capu - temp4 + temp3 - epw) / (1.0 - temp5 - temp6) + epw
        prev_epw = epw
        epw = next_epw
        if abs(next_epw - prev_epw) <= e6a:
            break
    ecose = temp5 + temp6
    esine = temp3 - temp4
    elsq = axn * axn + ayn * ayn
    temp = 1.0 - elsq
    pl = a * temp
    if pl <= 0.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 semi-latus rectum became invalid.")
    r = a * (1.0 - ecose)
    temp1 = 1.0 / r
    rdot = xke * math.sqrt(a) * esine * temp1
    rfdot = xke * math.sqrt(pl) * temp1
    temp2 = a * temp1
    betal = math.sqrt(temp)
    temp3 = 1.0 / (1.0 + betal)
    cosu = temp2 * (cosepw - axn + ayn * esine * temp3)
    sinu = temp2 * (sinepw - ayn - axn * esine * temp3)
    u = _actan(sinu, cosu)
    sin2u = 2.0 * sinu * cosu
    cos2u = 2.0 * cosu * cosu - 1.0
    temp = 1.0 / pl
    temp1 = ck2 * temp
    temp2 = temp1 * temp
    rk = r * (1.0 - 1.5 * temp2 * betal * x3thm1) + 0.5 * temp1 * x1mth2 * cos2u
    uk = u - 0.25 * temp2 * x7thm1 * sin2u
    xnodek = xnode + 1.5 * temp2 * cosio * sin2u
    xinck = satdata["xincl"] + 1.5 * temp2 * cosio * sinio * cos2u
    rdotk = rdot - xn * temp1 * x1mth2 * sin2u
    rfdotk = rfdot + xn * temp1 * (x1mth2 * cos2u + 1.5 * x3thm1)

    mv = np.array([-math.sin(xnodek) * math.cos(xinck), math.cos(xnodek) * math.cos(xinck), math.sin(xinck)])
    nv = np.array([math.cos(xnodek), math.sin(xnodek), 0.0])
    uv = mv * math.sin(uk) + nv * math.cos(uk)
    vv = mv * math.cos(uk) - nv * math.sin(uk)
    pos_er = rk * uv
    vel_er_per_min = rdotk * uv + rfdotk * vv
    return SGP4State(
        position_teme_km=np.array(pos_er * xkmper, dtype=float),
        velocity_teme_km_s=np.array(vel_er_per_min * xkmper / 60.0, dtype=float),
    )


def sgp4_propagate_teme_batch_reference(
    elements: list[TLEElements] | tuple[TLEElements, ...],
    tsince_min: np.ndarray | list[float] | tuple[float, ...],
) -> SGP4BatchResult:
    """Propagate many near-Earth TLEs on a common array-shaped contract.

    This is the correctness reference backend for future vectorized SGP4
    implementations. It intentionally calls the scalar SGP4 implementation for
    every object/time sample while enforcing the batched input/output shape.

    `tsince_min` may be either:

    - shape `(sample_count,)`, shared by every object, or
    - shape `(object_count, sample_count)`, giving each object its own offsets.
    """

    element_list = list(elements)
    object_count = len(element_list)
    if object_count <= 0:
        raise ValueError("Batched SGP4 propagation requires at least one element set.")
    time_grid = _coerce_sgp4_batch_times(tsince_min, object_count=object_count)
    sample_count = int(time_grid.shape[1])
    positions = np.zeros((object_count, sample_count, 3), dtype=float)
    velocities = np.zeros((object_count, sample_count, 3), dtype=float)
    errors = np.full((object_count, sample_count), "", dtype=object)
    for object_index, element in enumerate(element_list):
        for sample_index, offset_min in enumerate(time_grid[object_index]):
            state = sgp4_propagate_teme(element, float(offset_min))
            positions[object_index, sample_index, :] = np.asarray(state.position_teme_km, dtype=float)
            velocities[object_index, sample_index, :] = np.asarray(state.velocity_teme_km_s, dtype=float)
            errors[object_index, sample_index] = "" if state.error is None else str(state.error)
    return SGP4BatchResult(
        backend="scalar_reference",
        tsince_min=time_grid,
        position_teme_km=positions,
        velocity_teme_km_s=velocities,
        errors=errors,
    )


def sgp4_propagate_teme_batch_numba(
    elements: list[TLEElements] | tuple[TLEElements, ...],
    tsince_min: np.ndarray | list[float] | tuple[float, ...],
) -> SGP4BatchResult:
    """Propagate many near-Earth TLEs with the compiled Numba CPU backend."""

    if njit is None:
        raise RuntimeError("Numba is not available; install the optional acceleration dependencies.")
    element_list = list(elements)
    object_count = len(element_list)
    if object_count <= 0:
        raise ValueError("Batched SGP4 propagation requires at least one element set.")
    time_grid = _coerce_sgp4_batch_times(tsince_min, object_count=object_count)
    numeric_elements = _sgp4_numeric_element_array(element_list)
    sample_count = int(time_grid.shape[1])
    positions = np.zeros((object_count, sample_count, 3), dtype=np.float64)
    velocities = np.zeros((object_count, sample_count, 3), dtype=np.float64)
    error_codes = np.zeros((object_count, sample_count), dtype=np.int64)
    _sgp4_batch_numba_kernel(numeric_elements, time_grid, positions, velocities, error_codes)
    errors = np.full((object_count, sample_count), "", dtype=object)
    for code, message in _SGP4_NUMBA_ERROR_MESSAGES.items():
        if code:
            errors[error_codes == code] = message
    return SGP4BatchResult(
        backend="numba_cpu",
        tsince_min=time_grid,
        position_teme_km=positions,
        velocity_teme_km_s=velocities,
        errors=errors,
    )


def _coerce_sgp4_batch_times(
    tsince_min: np.ndarray | list[float] | tuple[float, ...],
    *,
    object_count: int,
) -> np.ndarray:
    times = np.asarray(tsince_min, dtype=float)
    if times.ndim == 0:
        times = times.reshape(1)
    if times.ndim == 1:
        if times.size <= 0:
            raise ValueError("Batched SGP4 propagation requires at least one time sample.")
        if not np.all(np.isfinite(times)):
            raise ValueError("Batched SGP4 time offsets must be finite.")
        return np.broadcast_to(times.reshape(1, -1), (object_count, times.size)).copy()
    if times.ndim == 2:
        if times.shape[0] != object_count:
            raise ValueError(
                "Batched SGP4 per-object time grid must have shape "
                f"(object_count, sample_count); got {times.shape} for {object_count} objects."
            )
        if times.shape[1] <= 0:
            raise ValueError("Batched SGP4 propagation requires at least one time sample.")
        if not np.all(np.isfinite(times)):
            raise ValueError("Batched SGP4 time offsets must be finite.")
        return np.array(times, dtype=float, copy=True)
    raise ValueError("Batched SGP4 time offsets must be a 1-D or 2-D array.")


def _sgp4_numeric_element_array(elements: list[TLEElements]) -> np.ndarray:
    rows = np.zeros((len(elements), 9), dtype=np.float64)
    for index, item in enumerate(elements):
        satdata = _sgp4_satdata(item)
        rows[index] = [
            satdata["xmo"],
            satdata["xnodeo"],
            satdata["omegao"],
            satdata["xincl"],
            satdata["eo"],
            satdata["xno"],
            satdata["xndt2o"],
            satdata["xndd6o"],
            satdata["bstar"],
        ]
    return rows


_SGP4_NUMBA_ERROR_MESSAGES = {
    1: (
        "OEL OGP-SGP4 supports near-Earth SGP4 only; OGP-SDP4/deep-space SDP4/resonance TLEs "
        f"with orbital period >= {SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN:.0f} min are not supported."
    ),
    2: "SGP4 mean motion must be positive.",
    3: "SGP4 eccentricity must be in [0, 1).",
    4: "SGP4 propagated orbit became invalid.",
    5: "SGP4 semi-latus rectum became invalid.",
}


if njit is not None:

    @njit(cache=True)
    def _sgp4_numba_state(
        xmo: float,
        xnodeo: float,
        omegao: float,
        xincl: float,
        eo: float,
        xno: float,
        xndt2o: float,
        xndd6o: float,
        bstar: float,
        tsince: float,
        pos_out: np.ndarray,
        vel_out: np.ndarray,
    ) -> int:
        ae = 1.0
        tothrd = 2.0 / 3.0
        xj3 = -2.53881e-6
        e6a = 1.0e-6
        xkmper = 6378.135
        ge = 398600.8
        ck2 = 1.0826158e-3 / 2.0
        ck4 = -3.0 * -1.65597e-6 / 8.0
        if xno <= 0.0:
            return 2
        if eo < 0.0 or eo >= 1.0:
            return 3

        s = ae + 78.0 / xkmper
        qo = ae + 120.0 / xkmper
        xke = math.sqrt((3600.0 * ge) / (xkmper**3))
        qoms2t = (qo - s) ** 4
        a1 = (xke / xno) ** tothrd
        cosio = math.cos(xincl)
        theta2 = cosio * cosio
        x3thm1 = 3.0 * theta2 - 1.0
        eosq = eo * eo
        betao2 = 1.0 - eosq
        betao = math.sqrt(betao2)
        del1 = 1.5 * ck2 * x3thm1 / ((a1 * a1) * betao * betao2)
        ao = a1 * (1.0 - del1 * (1.0 / 3.0 + del1 * (1.0 + 134.0 / 81.0 * del1)))
        delo = 1.5 * ck2 * x3thm1 / ((ao * ao) * betao * betao2)
        xnodp = xno / (1.0 + delo)
        period_min = (2.0 * math.pi) / xnodp
        if period_min >= SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN:
            return 1
        aodp = ao / (1.0 - delo)

        isimp = (aodp * (1.0 - eo) / ae) < (220.0 / xkmper + ae)
        s4 = s
        qoms24 = qoms2t
        perige = (aodp * (1.0 - eo) - ae) * xkmper
        if perige < 156.0:
            s4 = perige - 78.0
            if perige <= 98.0:
                s4 = 20.0
            qoms24 = ((120.0 - s4) * ae / xkmper) ** 4
            s4 = s4 / xkmper + ae

        pinvsq = 1.0 / ((aodp * aodp) * (betao2 * betao2))
        tsi = 1.0 / (aodp - s4)
        eta = aodp * eo * tsi
        etasq = eta * eta
        eeta = eo * eta
        psisq = abs(1.0 - etasq)
        coef = qoms24 * (tsi**4)
        coef1 = coef / (psisq**3.5)
        c2 = coef1 * xnodp * (
            aodp * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
            + 0.75 * ck2 * tsi / psisq * x3thm1 * (8.0 + 3.0 * etasq * (8.0 + etasq))
        )
        c1 = bstar * c2
        sinio = math.sin(xincl)
        a3ovk2 = -xj3 / ck2 * (ae**3)
        c3 = 0.0 if abs(eo) < 1.0e-12 else coef * tsi * a3ovk2 * xnodp * ae * sinio / eo
        x1mth2 = 1.0 - theta2
        c4 = 2.0 * xnodp * coef1 * aodp * betao2 * (
            eta * (2.0 + 0.5 * etasq)
            + eo * (0.5 + 2.0 * etasq)
            - 2.0
            * ck2
            * tsi
            / (aodp * psisq)
            * (
                -3.0 * x3thm1 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                + 0.75 * x1mth2 * (2.0 * etasq - eeta * (1.0 + etasq)) * math.cos(2.0 * omegao)
            )
        )
        c5 = 2.0 * coef1 * aodp * betao2 * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
        theta4 = theta2 * theta2
        temp1 = 3.0 * ck2 * pinvsq * xnodp
        temp2 = temp1 * ck2 * pinvsq
        temp3 = 1.25 * ck4 * pinvsq * pinvsq * xnodp
        xmdot = xnodp + 0.5 * temp1 * betao * x3thm1 + 0.0625 * temp2 * betao * (
            13.0 - 78.0 * theta2 + 137.0 * theta4
        )
        x1m5th = 1.0 - 5.0 * theta2
        omgdot = (
            -0.5 * temp1 * x1m5th
            + 0.0625 * temp2 * (7.0 - 114.0 * theta2 + 395.0 * theta4)
            + temp3 * (3.0 - 36.0 * theta2 + 49.0 * theta4)
        )
        xhdot1 = -temp1 * cosio
        xnodot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * theta2) + 2.0 * temp3 * (3.0 - 7.0 * theta2)) * cosio
        omgcof = bstar * c3 * math.cos(omegao)
        xmcof = 0.0 if abs(eeta) < 1.0e-12 else -(2.0 / 3.0) * coef * bstar * ae / eeta
        xnodcf = 3.5 * betao2 * xhdot1 * c1
        t2cof = 1.5 * c1
        xlcof_denominator = 1.0 + cosio
        if abs(xlcof_denominator) < 1.5e-12:
            xlcof_denominator = 1.5e-12
        xlcof = 0.125 * a3ovk2 * sinio * (3.0 + 5.0 * cosio) / xlcof_denominator
        aycof = 0.25 * a3ovk2 * sinio
        delmo = (1.0 + eta * math.cos(xmo)) ** 3
        sinmo = math.sin(xmo)
        x7thm1 = 7.0 * theta2 - 1.0
        d2 = 0.0
        d3 = 0.0
        d4 = 0.0
        t3cof = 0.0
        t4cof = 0.0
        t5cof = 0.0
        if not isimp:
            c1sq = c1 * c1
            d2 = 4.0 * aodp * tsi * c1sq
            temp = d2 * tsi * c1 / 3.0
            d3 = (17.0 * aodp + s4) * temp
            d4 = 0.5 * temp * aodp * tsi * (221.0 * aodp + 31.0 * s4) * c1
            t3cof = d2 + 2.0 * c1sq
            t4cof = 0.25 * (3.0 * d3 + c1 * (12.0 * d2 + 10.0 * c1sq))
            t5cof = 0.2 * (3.0 * d4 + 12.0 * c1 * d3 + 6.0 * d2 * d2 + 15.0 * c1sq * (2.0 * d2 + c1sq))

        xmdf = xmo + xmdot * tsince
        omgadf = omegao + omgdot * tsince
        xnoddf = xnodeo + xnodot * tsince
        omega = omgadf
        xmp = xmdf
        tsq = tsince * tsince
        xnode = xnoddf + xnodcf * tsq
        tempa = 1.0 - c1 * tsince
        tempe = bstar * c4 * tsince
        templ = t2cof * tsq
        if not isimp:
            delomg = omgcof * tsince
            delm = xmcof * ((1.0 + eta * math.cos(xmdf)) ** 3 - delmo)
            temp = delomg + delm
            xmp = xmdf + temp
            omega = omgadf - temp
            tcube = tsq * tsince
            tfour = tsince * tcube
            tempa = tempa - d2 * tsq - d3 * tcube - d4 * tfour
            tempe = tempe + bstar * c5 * (math.sin(xmp) - sinmo)
            templ = templ + t3cof * tcube + tfour * (t4cof + tsince * t5cof)

        a = aodp * (tempa * tempa)
        e = eo - tempe
        if a <= 0.0 or e < 0.0 or e >= 1.0:
            return 4
        xl = xmp + omega + xnode + xnodp * templ
        beta = math.sqrt(1.0 - e * e)
        xn = xke / (a**1.5)
        axn = e * math.cos(omega)
        temp = 1.0 / (a * beta * beta)
        xll = temp * xlcof * axn
        aynl = temp * aycof
        xlt = xl + xll
        ayn = e * math.sin(omega) + aynl

        capu = (xlt - xnode) % (2.0 * math.pi)
        epw = capu
        sinepw = 0.0
        cosepw = 0.0
        temp3 = 0.0
        temp4 = 0.0
        temp5 = 0.0
        temp6 = 0.0
        for _ in range(10):
            sinepw = math.sin(epw)
            cosepw = math.cos(epw)
            temp3 = axn * sinepw
            temp4 = ayn * cosepw
            temp5 = axn * cosepw
            temp6 = ayn * sinepw
            next_epw = (capu - temp4 + temp3 - epw) / (1.0 - temp5 - temp6) + epw
            prev_epw = epw
            epw = next_epw
            if abs(next_epw - prev_epw) <= e6a:
                break
        ecose = temp5 + temp6
        esine = temp3 - temp4
        elsq = axn * axn + ayn * ayn
        temp = 1.0 - elsq
        pl = a * temp
        if pl <= 0.0:
            return 5
        r = a * (1.0 - ecose)
        temp1 = 1.0 / r
        rdot = xke * math.sqrt(a) * esine * temp1
        rfdot = xke * math.sqrt(pl) * temp1
        temp2 = a * temp1
        betal = math.sqrt(temp)
        temp3 = 1.0 / (1.0 + betal)
        cosu = temp2 * (cosepw - axn + ayn * esine * temp3)
        sinu = temp2 * (sinepw - ayn - axn * esine * temp3)
        u = math.atan2(sinu, cosu) % (2.0 * math.pi)
        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0
        temp = 1.0 / pl
        temp1 = ck2 * temp
        temp2 = temp1 * temp
        rk = r * (1.0 - 1.5 * temp2 * betal * x3thm1) + 0.5 * temp1 * x1mth2 * cos2u
        uk = u - 0.25 * temp2 * x7thm1 * sin2u
        xnodek = xnode + 1.5 * temp2 * cosio * sin2u
        xinck = xincl + 1.5 * temp2 * cosio * sinio * cos2u
        rdotk = rdot - xn * temp1 * x1mth2 * sin2u
        rfdotk = rfdot + xn * temp1 * (x1mth2 * cos2u + 1.5 * x3thm1)

        sin_xnodek = math.sin(xnodek)
        cos_xnodek = math.cos(xnodek)
        cos_xinck = math.cos(xinck)
        sin_xinck = math.sin(xinck)
        sin_uk = math.sin(uk)
        cos_uk = math.cos(uk)
        mv0 = -sin_xnodek * cos_xinck
        mv1 = cos_xnodek * cos_xinck
        mv2 = sin_xinck
        nv0 = cos_xnodek
        nv1 = sin_xnodek
        nv2 = 0.0
        uv0 = mv0 * sin_uk + nv0 * cos_uk
        uv1 = mv1 * sin_uk + nv1 * cos_uk
        uv2 = mv2 * sin_uk + nv2 * cos_uk
        vv0 = mv0 * cos_uk - nv0 * sin_uk
        vv1 = mv1 * cos_uk - nv1 * sin_uk
        vv2 = mv2 * cos_uk - nv2 * sin_uk
        pos_out[0] = rk * uv0 * xkmper
        pos_out[1] = rk * uv1 * xkmper
        pos_out[2] = rk * uv2 * xkmper
        vel_out[0] = (rdotk * uv0 + rfdotk * vv0) * xkmper / 60.0
        vel_out[1] = (rdotk * uv1 + rfdotk * vv1) * xkmper / 60.0
        vel_out[2] = (rdotk * uv2 + rfdotk * vv2) * xkmper / 60.0
        return 0

    @njit(cache=True, parallel=True)
    def _sgp4_batch_numba_kernel(
        numeric_elements: np.ndarray,
        time_grid: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        error_codes: np.ndarray,
    ) -> None:
        object_count = numeric_elements.shape[0]
        sample_count = time_grid.shape[1]
        for object_index in prange(object_count):
            xmo = numeric_elements[object_index, 0]
            xnodeo = numeric_elements[object_index, 1]
            omegao = numeric_elements[object_index, 2]
            xincl = numeric_elements[object_index, 3]
            eo = numeric_elements[object_index, 4]
            xno = numeric_elements[object_index, 5]
            xndt2o = numeric_elements[object_index, 6]
            xndd6o = numeric_elements[object_index, 7]
            bstar = numeric_elements[object_index, 8]
            for sample_index in range(sample_count):
                code = _sgp4_numba_state(
                    xmo,
                    xnodeo,
                    omegao,
                    xincl,
                    eo,
                    xno,
                    xndt2o,
                    xndd6o,
                    bstar,
                    time_grid[object_index, sample_index],
                    positions[object_index, sample_index],
                    velocities[object_index, sample_index],
                )
                error_codes[object_index, sample_index] = code

else:

    def _sgp4_batch_numba_kernel(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Numba is not available; install the optional acceleration dependencies.")


def transform_teme_to_output_frame(
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    *,
    jd_utc: float,
    output_frame: str = "teme",
    frame_transform: str = "native",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the v1 OEL output-frame view of a propagated TEME state.

    v1 supports native TEME output, the legacy transparent TEME-as-ECI
    approximation, and an explicit Vallado IAU-80 TEME-to-ECI reduction.
    """
    frame = str(output_frame or "teme").strip().lower()
    transform = str(frame_transform or ("native" if frame == "teme" else "teme_to_eci_iau80")).strip().lower()
    if frame == "teme":
        if transform not in {"native", "none", "identity", "teme"}:
            raise ValueError("SGP4 output_frame='teme' requires frame_transform='native'.")
        _ = float(jd_utc)
        return np.array(position_teme_km, dtype=float), np.array(velocity_teme_km_s, dtype=float)
    if frame != "eci":
        raise ValueError("SGP4 v1 only supports output_frame='eci' or output_frame='teme'.")
    if transform == "teme_to_eci_iau80":
        return teme_to_eci_vallado_iau80(position_teme_km, velocity_teme_km_s, jd_utc=float(jd_utc))
    if transform != "teme_as_eci":
        raise ValueError(
            "SGP4 output_frame='eci' requires frame_transform='teme_as_eci' or "
            "'teme_to_eci_iau80' for v1."
        )
    _ = float(jd_utc)
    return np.array(position_teme_km, dtype=float), np.array(velocity_teme_km_s, dtype=float)
