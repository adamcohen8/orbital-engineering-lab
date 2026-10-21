from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

import numpy as np

from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.core.models import StateTruth
from sim.dynamics.attitude.rigid_body import _add_guardrail_counts
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.eclipse import srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S, srp_pressure_n_m2
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, RectangularPrismGeometry
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.quaternion import quaternion_to_dcm_bn

# Earth dipole field parameter in T*m^3 for a simple centered dipole model.
# The IGRF g10 coefficient is negative, so the equivalent dipole points toward
# geographic south (negative ECI z in the fallback model).
EARTH_MAGNETIC_DIPOLE_T_M3 = -7.94e15
_EARTH_MAGNETIC_DIPOLE_ECI_T_M3 = np.array([0.0, 0.0, EARTH_MAGNETIC_DIPOLE_T_M3])
_FACET_MODE_NONE = 0
_FACET_MODE_SCALAR = 1
_FACET_MODE_FACETS = 2
_PROPAGATE_ATTITUDE_BUILTIN_DISTURBANCES_KERNEL = None
_DISTURBANCE_KERNEL_LOCK = RLock()


def _load_builtin_disturbances_kernel():
    global _PROPAGATE_ATTITUDE_BUILTIN_DISTURBANCES_KERNEL
    kernel = _PROPAGATE_ATTITUDE_BUILTIN_DISTURBANCES_KERNEL
    if kernel is None:
        with _DISTURBANCE_KERNEL_LOCK:
            kernel = _PROPAGATE_ATTITUDE_BUILTIN_DISTURBANCES_KERNEL
            if kernel is None:
                from sim.acceleration.kernels.attitude import propagate_attitude_builtin_disturbances_kernel

                kernel = propagate_attitude_builtin_disturbances_kernel
                _PROPAGATE_ATTITUDE_BUILTIN_DISTURBANCES_KERNEL = kernel
    return kernel


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cross product of two three-vectors without generic axis setup."""
    out = np.empty(3, dtype=float)
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
    return out


def _norm3(vector: np.ndarray) -> float:
    """Return the Euclidean norm of a three-vector without generic linalg dispatch."""
    return float(np.sqrt(np.dot(vector, vector)))


@dataclass(frozen=True)
class DisturbanceTorqueConfig:
    use_gravity_gradient: bool = True
    use_magnetic: bool = True
    use_drag: bool = True
    use_srp: bool = True
    magnetic_dipole_body_a_m2: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 0.0]))
    drag_area_m2: float = 1.5
    drag_cd: float = 2.2
    drag_cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.02, -0.01]))
    drag_facet_normal_body: np.ndarray | None = None
    drag_facets: tuple[dict, ...] | None = None
    srp_area_m2: float = 1.0
    srp_cr: float = 1.3
    srp_cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.array([-0.02, 0.03, 0.01]))
    sun_dir_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    srp_facet_normal_body: np.ndarray | None = None
    srp_facets: tuple[dict, ...] | None = None
    use_rectangular_prism_faces: bool = False
    rectangular_prism_dims_m: tuple[float, float, float] | None = None
    geometry_area_profile: GeometryAreaProfile | None = None
    center_of_mass_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    numeric_backend: str = "rust"


@dataclass(frozen=True)
class DisturbanceTorqueModel:
    mu_km3_s2: float
    inertia_kg_m2: np.ndarray
    config: DisturbanceTorqueConfig = field(default_factory=DisturbanceTorqueConfig)
    _compiled_plan_supported: bool = field(init=False, repr=False, compare=False)
    _compiled_has_disturbances: bool = field(init=False, repr=False, compare=False)
    _compiled_enabled: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_drag_mode: int = field(init=False, repr=False, compare=False)
    _compiled_drag_facet_normals: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_drag_facet_areas: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_drag_facet_cd: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_drag_facet_cp_offsets: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_drag_facet_signature: tuple | None = field(init=False, repr=False, compare=False)
    _compiled_srp_mode: int = field(init=False, repr=False, compare=False)
    _compiled_srp_facet_normals: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_srp_facet_areas: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_srp_facet_cp_offsets: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_srp_facet_signature: tuple | None = field(init=False, repr=False, compare=False)
    _compiled_inertia: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_magnetic_dipole: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_drag_cp_offset: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_srp_cp_offset: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_zero3: np.ndarray = field(init=False, repr=False, compare=False)
    _compiled_native_context_cache: tuple | None = field(default=None, init=False, repr=False, compare=False)
    _rect_prism_geometry_cache: RectangularPrismGeometry | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        config = self.config
        normalize_numeric_backend(config.numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
        supported = config.geometry_area_profile is None and not config.use_rectangular_prism_faces
        enabled = np.asarray(
            [config.use_gravity_gradient, config.use_magnetic, config.use_drag, config.use_srp],
            dtype=np.int64,
        )

        drag_normals, drag_areas, drag_cp_offsets, drag_cd = _empty_drag_facets()
        drag_facet_signature = None
        drag_mode = _FACET_MODE_SCALAR if config.use_drag else _FACET_MODE_NONE
        if config.use_drag and config.drag_facets:
            try:
                drag_normals, drag_areas, drag_cp_offsets, drag_cd = _stage_drag_facets(
                    config.drag_facets,
                    default_cd=float(config.drag_cd),
                )
                drag_facet_signature = _drag_facet_signature(config.drag_facets, default_cd=float(config.drag_cd))
                drag_mode = _FACET_MODE_FACETS
            except (KeyError, TypeError, ValueError):
                supported = False
        elif config.use_drag and config.drag_facet_normal_body is not None:
            try:
                drag_normals = np.asarray(config.drag_facet_normal_body, dtype=float).reshape(1, 3)
                drag_areas = np.asarray([config.drag_area_m2], dtype=float)
                drag_cp_offsets = np.asarray(config.drag_cp_offset_body_m, dtype=float).reshape(1, 3)
                drag_cd = np.asarray([config.drag_cd], dtype=float)
                drag_mode = _FACET_MODE_FACETS
            except (TypeError, ValueError):
                supported = False

        srp_normals, srp_areas, srp_cp_offsets = _empty_srp_facets()
        srp_facet_signature = None
        srp_mode = _FACET_MODE_SCALAR if config.use_srp else _FACET_MODE_NONE
        if config.use_srp and config.srp_facets:
            try:
                srp_normals, srp_areas, srp_cp_offsets = _stage_srp_facets(config.srp_facets)
                srp_facet_signature = _srp_facet_signature(config.srp_facets)
                srp_mode = _FACET_MODE_FACETS
            except (KeyError, TypeError, ValueError):
                supported = False
        elif config.use_srp and config.srp_facet_normal_body is not None:
            try:
                srp_normals = np.asarray(config.srp_facet_normal_body, dtype=float).reshape(1, 3)
                srp_areas = np.asarray([config.srp_area_m2], dtype=float)
                srp_cp_offsets = np.asarray(config.srp_cp_offset_body_m, dtype=float).reshape(1, 3)
                srp_mode = _FACET_MODE_FACETS
            except (TypeError, ValueError):
                supported = False

        object.__setattr__(self, "_compiled_plan_supported", bool(supported))
        object.__setattr__(self, "_compiled_has_disturbances", bool(np.any(enabled)))
        object.__setattr__(self, "_compiled_enabled", enabled)
        object.__setattr__(self, "_compiled_drag_mode", int(drag_mode))
        object.__setattr__(self, "_compiled_drag_facet_normals", drag_normals)
        object.__setattr__(self, "_compiled_drag_facet_areas", drag_areas)
        object.__setattr__(self, "_compiled_drag_facet_cd", drag_cd)
        object.__setattr__(self, "_compiled_drag_facet_cp_offsets", drag_cp_offsets)
        object.__setattr__(self, "_compiled_drag_facet_signature", drag_facet_signature)
        object.__setattr__(self, "_compiled_srp_mode", int(srp_mode))
        object.__setattr__(self, "_compiled_srp_facet_normals", srp_normals)
        object.__setattr__(self, "_compiled_srp_facet_areas", srp_areas)
        object.__setattr__(self, "_compiled_srp_facet_cp_offsets", srp_cp_offsets)
        object.__setattr__(self, "_compiled_srp_facet_signature", srp_facet_signature)
        object.__setattr__(self, "_compiled_inertia", np.asarray(self.inertia_kg_m2, dtype=float).reshape(3, 3))
        object.__setattr__(
            self,
            "_compiled_magnetic_dipole",
            np.asarray(config.magnetic_dipole_body_a_m2, dtype=float).reshape(3),
        )
        object.__setattr__(
            self,
            "_compiled_drag_cp_offset",
            np.asarray(config.drag_cp_offset_body_m, dtype=float).reshape(3),
        )
        object.__setattr__(
            self,
            "_compiled_srp_cp_offset",
            np.asarray(config.srp_cp_offset_body_m, dtype=float).reshape(3),
        )
        object.__setattr__(self, "_compiled_zero3", np.zeros(3, dtype=float))

    def try_propagate_compiled(
        self,
        *,
        quat_bn: np.ndarray,
        omega_body_rad_s: np.ndarray,
        command_torque_body_nm: np.ndarray,
        position_eci_km: np.ndarray,
        t_s: float,
        env: dict,
        substeps_s: np.ndarray,
        acceleration_mode: str,
        numeric_backend: str | None = None,
        acceleration_enabled: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Use the numeric built-in torque plan, or return ``None`` for the Python fallback."""

        config = self.config
        selected_backend = normalize_numeric_backend(
            config.numeric_backend if numeric_backend is None else numeric_backend,
            error_message="numeric_backend must be 'python' or 'rust'.",
        )
        if (
            type(self) is not DisturbanceTorqueModel
            or not self._compiled_plan_supported
            or not self._compiled_has_disturbances
            or not (
                True
                if selected_backend == "rust"
                else (
                    acceleration_enabled_from_mode(acceleration_mode)
                    if acceleration_enabled is None
                    else acceleration_enabled
                )
            )
        ):
            return None

        if not self._refresh_mutable_facet_staging():
            return None

        config = self.config
        density = 0.0
        drag_v_rel = self._compiled_zero3
        drag_v_norm = 0.0
        if config.use_drag:
            if "density_kg_m3" not in env or "drag_v_rel_eci_m_s" not in env:
                return None
            density = float(env["density_kg_m3"])
            drag_v_rel = np.asarray(env["drag_v_rel_eci_m_s"], dtype=float).reshape(3)
            drag_v_norm = (
                float(env["drag_v_rel_norm_m_s"])
                if "drag_v_rel_norm_m_s" in env
                else _norm3(drag_v_rel)
            )

        magnetic_field_provided = "magnetic_field_eci_t" in env
        magnetic_field = (
            np.asarray(env["magnetic_field_eci_t"], dtype=float).reshape(3)
            if magnetic_field_provided
            else self._compiled_zero3
        )

        sun_dir = self._compiled_zero3
        pressure_scaled = 0.0
        if config.use_srp:
            raw_sun_dir = np.asarray(
                env.get("sun_dir_eci_unit", env.get("sun_dir_eci", config.sun_dir_eci)),
                dtype=float,
            ).reshape(3)
            sun_norm = _norm3(raw_sun_dir)
            if sun_norm > 0.0:
                sun_dir = raw_sun_dir if "sun_dir_eci_unit" in env else raw_sun_dir / sun_norm
                if "srp_shadow_factor" in env:
                    shadow = float(env["srp_shadow_factor"])
                else:
                    shadow = float(srp_shadow_factor(r_sc_eci_km=position_eci_km, t_s=t_s, env=env))
                if shadow > 0.0:
                    pressure_scaled = (
                        srp_pressure_n_m2(env)
                        * float(env.get("srp_distance_scale", 1.0))
                        * float(config.srp_cr)
                        * shadow
                    )

        if selected_backend == "rust":
            from sim.rust_attitude_backend import propagate_builtin_disturbances

            prepared_context = self._native_disturbance_context()
            rust_result = propagate_builtin_disturbances(
                quat_bn=np.asarray(quat_bn, dtype=float).reshape(4),
                omega_body_rad_s=np.asarray(omega_body_rad_s, dtype=float).reshape(3),
                inertia_kg_m2=self._compiled_inertia,
                command_torque_body_nm=np.asarray(command_torque_body_nm, dtype=float).reshape(3),
                substeps_s=np.asarray(substeps_s, dtype=float).reshape(-1),
                position_eci_km=np.asarray(position_eci_km, dtype=float).reshape(3),
                mu_km3_s2=float(self.mu_km3_s2),
                enabled=self._compiled_enabled,
                magnetic_dipole_body_a_m2=self._compiled_magnetic_dipole,
                magnetic_field_eci_t=np.asarray(magnetic_field, dtype=float).reshape(3),
                magnetic_field_provided=bool(magnetic_field_provided),
                density_kg_m3=float(density),
                drag_v_rel_eci_m_s=np.asarray(drag_v_rel, dtype=float).reshape(3),
                drag_v_rel_norm_m_s=float(drag_v_norm),
                drag_mode=int(self._compiled_drag_mode),
                drag_area_m2=float(config.drag_area_m2),
                drag_cd=float(config.drag_cd),
                drag_cp_offset_body_m=self._compiled_drag_cp_offset,
                drag_facet_normals_body=self._compiled_drag_facet_normals,
                drag_facet_areas_m2=self._compiled_drag_facet_areas,
                drag_facet_cd=self._compiled_drag_facet_cd,
                drag_facet_cp_offsets_body_m=self._compiled_drag_facet_cp_offsets,
                sun_dir_eci_unit=np.asarray(sun_dir, dtype=float).reshape(3),
                srp_pressure_scaled_n_m2=float(pressure_scaled),
                srp_mode=int(self._compiled_srp_mode),
                srp_area_m2=float(config.srp_area_m2),
                srp_cp_offset_body_m=self._compiled_srp_cp_offset,
                srp_facet_normals_body=self._compiled_srp_facet_normals,
                srp_facet_areas_m2=self._compiled_srp_facet_areas,
                srp_facet_cp_offsets_body_m=self._compiled_srp_facet_cp_offsets,
                prepared_context=prepared_context,
            )
            _add_guardrail_counts(rust_result[2])
            return rust_result[0], rust_result[1]

        kernel = _load_builtin_disturbances_kernel()
        q_next, omega_next, counts = kernel(
            np.asarray(quat_bn, dtype=float).reshape(4),
            np.asarray(omega_body_rad_s, dtype=float).reshape(3),
            self._compiled_inertia,
            np.asarray(command_torque_body_nm, dtype=float).reshape(3),
            np.asarray(substeps_s, dtype=float).reshape(-1),
            np.asarray(position_eci_km, dtype=float).reshape(3),
            float(self.mu_km3_s2),
            self._compiled_enabled,
            self._compiled_magnetic_dipole,
            magnetic_field,
            bool(magnetic_field_provided),
            float(density),
            drag_v_rel,
            float(drag_v_norm),
            int(self._compiled_drag_mode),
            float(config.drag_area_m2),
            float(config.drag_cd),
            self._compiled_drag_cp_offset,
            self._compiled_drag_facet_normals,
            self._compiled_drag_facet_areas,
            self._compiled_drag_facet_cd,
            self._compiled_drag_facet_cp_offsets,
            sun_dir,
            float(pressure_scaled),
            int(self._compiled_srp_mode),
            float(config.srp_area_m2),
            self._compiled_srp_cp_offset,
            self._compiled_srp_facet_normals,
            self._compiled_srp_facet_areas,
            self._compiled_srp_facet_cp_offsets,
        )
        _add_guardrail_counts(counts)
        return q_next, omega_next

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compiled_native_context_cache"] = None
        return state

    def _native_disturbance_context(self):
        from sim.rust_attitude_backend import prepare_builtin_disturbance_context

        config = self.config
        arrays = (self._compiled_inertia, self._compiled_enabled, self._compiled_magnetic_dipole,
                  self._compiled_drag_cp_offset, self._compiled_drag_facet_normals,
                  self._compiled_drag_facet_areas, self._compiled_drag_facet_cd,
                  self._compiled_drag_facet_cp_offsets, self._compiled_srp_cp_offset,
                  self._compiled_srp_facet_normals, self._compiled_srp_facet_areas,
                  self._compiled_srp_facet_cp_offsets)
        signature = (self.mu_km3_s2, self._compiled_drag_mode, config.drag_area_m2, config.drag_cd,
                     self._compiled_srp_mode, config.srp_area_m2, *(value.tobytes() for value in arrays))
        cached = self._compiled_native_context_cache
        if cached is not None and cached[0] == signature:
            return cached[1]
        context = prepare_builtin_disturbance_context(
            inertia_kg_m2=self._compiled_inertia, mu_km3_s2=self.mu_km3_s2,
            enabled=self._compiled_enabled, magnetic_dipole_body_a_m2=self._compiled_magnetic_dipole,
            drag_mode=self._compiled_drag_mode, drag_area_m2=config.drag_area_m2, drag_cd=config.drag_cd,
            drag_cp_offset_body_m=self._compiled_drag_cp_offset,
            drag_facet_normals_body=self._compiled_drag_facet_normals,
            drag_facet_areas_m2=self._compiled_drag_facet_areas, drag_facet_cd=self._compiled_drag_facet_cd,
            drag_facet_cp_offsets_body_m=self._compiled_drag_facet_cp_offsets,
            srp_mode=self._compiled_srp_mode, srp_area_m2=config.srp_area_m2,
            srp_cp_offset_body_m=self._compiled_srp_cp_offset,
            srp_facet_normals_body=self._compiled_srp_facet_normals,
            srp_facet_areas_m2=self._compiled_srp_facet_areas,
            srp_facet_cp_offsets_body_m=self._compiled_srp_facet_cp_offsets,
        )
        object.__setattr__(self, "_compiled_native_context_cache", (signature, context))
        return context

    def _refresh_mutable_facet_staging(self) -> bool:
        """Refresh copied facet values only when a public nested mapping changed."""

        config = self.config
        if self._compiled_drag_facet_signature is not None:
            try:
                signature = _drag_facet_signature(config.drag_facets, default_cd=float(config.drag_cd))
                if signature != self._compiled_drag_facet_signature:
                    normals, areas, cp_offsets, drag_cd = _stage_drag_facets(
                        config.drag_facets,
                        default_cd=float(config.drag_cd),
                    )
                    object.__setattr__(self, "_compiled_drag_facet_normals", normals)
                    object.__setattr__(self, "_compiled_drag_facet_areas", areas)
                    object.__setattr__(self, "_compiled_drag_facet_cp_offsets", cp_offsets)
                    object.__setattr__(self, "_compiled_drag_facet_cd", drag_cd)
                    object.__setattr__(self, "_compiled_drag_facet_signature", signature)
            except (KeyError, TypeError, ValueError):
                return False

        if self._compiled_srp_facet_signature is not None:
            try:
                signature = _srp_facet_signature(config.srp_facets)
                if signature != self._compiled_srp_facet_signature:
                    normals, areas, cp_offsets = _stage_srp_facets(config.srp_facets)
                    object.__setattr__(self, "_compiled_srp_facet_normals", normals)
                    object.__setattr__(self, "_compiled_srp_facet_areas", areas)
                    object.__setattr__(self, "_compiled_srp_facet_cp_offsets", cp_offsets)
                    object.__setattr__(self, "_compiled_srp_facet_signature", signature)
            except (KeyError, TypeError, ValueError):
                return False
        return True

    def total_torque_body_nm(self, state: StateTruth, env: dict | None = None) -> np.ndarray:
        env = env or {}
        tau = np.zeros(3)
        c_bn = None

        if self.config.use_gravity_gradient:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._gravity_gradient_torque(state, c_bn)
        if self.config.use_magnetic:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._magnetic_torque(state, c_bn, env)
        if self.config.use_drag:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._drag_torque(state, env, c_bn)
        if self.config.use_srp:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._srp_torque(state, env, c_bn)

        return tau

    def _gravity_gradient_torque(self, state: StateTruth, c_bn: np.ndarray) -> np.ndarray:
        r_i_m = state.position_eci_km * 1e3
        r_norm_m = _norm3(r_i_m)
        if r_norm_m == 0.0:
            return np.zeros(3)
        r_hat_b = c_bn @ (r_i_m / r_norm_m)
        mu_m3_s2 = self.mu_km3_s2 * 1e9
        return 3.0 * mu_m3_s2 / (r_norm_m**3) * _cross3(r_hat_b, self.inertia_kg_m2 @ r_hat_b)

    def _magnetic_torque(self, state: StateTruth, c_bn: np.ndarray, env: dict) -> np.ndarray:
        if "magnetic_field_eci_t" in env:
            b_eci = np.asarray(env["magnetic_field_eci_t"], dtype=float).reshape(3)
        else:
            r_i_m = state.position_eci_km * 1e3
            r_norm_m = _norm3(r_i_m)
            if r_norm_m == 0.0:
                return np.zeros(3)

            r_hat = r_i_m / r_norm_m
            b_eci = (
                3.0 * r_hat * np.dot(_EARTH_MAGNETIC_DIPOLE_ECI_T_M3, r_hat)
                - _EARTH_MAGNETIC_DIPOLE_ECI_T_M3
            ) / (r_norm_m**3)
        b_body = c_bn @ b_eci
        return _cross3(self.config.magnetic_dipole_body_a_m2, b_body)

    def _drag_torque(self, state: StateTruth, env: dict, c_bn: np.ndarray) -> np.ndarray:
        if "density_kg_m3" in env:
            rho = float(env["density_kg_m3"])
        else:
            rho = density_from_model(
                str(env.get("atmosphere_model", "exponential")).lower(),
                state.position_eci_km,
                state.t_s,
                env=env,
            )
        if "drag_v_rel_eci_m_s" in env:
            v_rel_eci_m_s = np.asarray(env["drag_v_rel_eci_m_s"], dtype=float)
            v_norm = (
                float(env["drag_v_rel_norm_m_s"])
                if "drag_v_rel_norm_m_s" in env
                else _norm3(v_rel_eci_m_s)
            )
        else:
            omega_raw = env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
            from sim.aero.core import atmosphere_relative_velocity_eci_km_s

            v_rel_eci_km_s = atmosphere_relative_velocity_eci_km_s(
                state.position_eci_km,
                state.velocity_eci_km_s,
                t_s=float(state.t_s),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                frame_model=str(env.get("drag_frame_model", "inertial_z")),
                jd_utc_start=env.get("jd_utc_start"),
                eop_path=env.get("drag_eop_path"),
                dut1_s=env.get("dut1_s"),
                xp_arcsec=env.get("xp_arcsec"),
                yp_arcsec=env.get("yp_arcsec"),
                dat_s=env.get("dat_s"),
                tt_minus_utc_s=env.get("tt_minus_utc_s"),
                ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
            )
            v_rel_eci_m_s = v_rel_eci_km_s * 1e3
            v_norm = _norm3(v_rel_eci_m_s)
        if v_norm == 0.0 or rho <= 0.0:
            return np.zeros(3)

        v_rel_body = c_bn @ v_rel_eci_m_s
        if self.config.geometry_area_profile is not None:
            q_dyn = 0.5 * rho * (v_norm**2) * self.config.drag_cd
            return self.config.geometry_area_profile.pressure_torque_sum_body_nm(
                -v_rel_body,
                q_dyn,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )
        if self._rect_prism_geometry is not None and self.config.use_rectangular_prism_faces:
            q_dyn = 0.5 * rho * (v_norm**2) * self.config.drag_cd
            return self._rect_prism_geometry.face_torque_sum_body_nm(
                -v_rel_body,
                q_dyn,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )

        if self.config.drag_facets:
            torque = np.zeros(3)
            for facet in self.config.drag_facets:
                torque += self._drag_facet_torque_body_nm(facet, rho, v_norm, v_rel_body, self.config.drag_cd)
            return torque

        if self.config.drag_facet_normal_body is not None:
            facet = {
                "area_m2": self.config.drag_area_m2,
                "drag_cd": self.config.drag_cd,
                "normal_body": self.config.drag_facet_normal_body,
                "cp_offset_body_m": self.config.drag_cp_offset_body_m,
            }
            return self._drag_facet_torque_body_nm(facet, rho, v_norm, v_rel_body, self.config.drag_cd)

        f_drag_mag = 0.5 * rho * (v_norm**2) * self.config.drag_cd * self.config.drag_area_m2
        f_drag_body = -f_drag_mag * (v_rel_body / v_norm)
        return _cross3(self.config.drag_cp_offset_body_m, f_drag_body)

    def _srp_torque(self, state: StateTruth, env: dict, c_bn: np.ndarray) -> np.ndarray:
        sun_dir_eci = np.asarray(
            env.get("sun_dir_eci_unit", env.get("sun_dir_eci", self.config.sun_dir_eci)), dtype=float
        )
        n = _norm3(sun_dir_eci)
        if n == 0.0:
            return np.zeros(3)
        if "sun_dir_eci_unit" not in env:
            sun_dir_eci = sun_dir_eci / n
        if "srp_shadow_factor" in env:
            shadow = float(env["srp_shadow_factor"])
        else:
            shadow = float(srp_shadow_factor(r_sc_eci_km=state.position_eci_km, t_s=state.t_s, env=env))
        if shadow <= 0.0:
            return np.zeros(3)

        sun_dir_body = c_bn @ sun_dir_eci
        distance_scale = float(env.get("srp_distance_scale", 1.0))
        p_srp_base = srp_pressure_n_m2(env) * distance_scale * self.config.srp_cr * shadow
        if self.config.geometry_area_profile is not None:
            return self.config.geometry_area_profile.pressure_torque_sum_body_nm(
                -sun_dir_body,
                p_srp_base,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )
        if self._rect_prism_geometry is not None and self.config.use_rectangular_prism_faces:
            return self._rect_prism_geometry.face_torque_sum_body_nm(
                -sun_dir_body,
                p_srp_base,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )

        if self.config.srp_facets:
            torque = np.zeros(3)
            for facet in self.config.srp_facets:
                torque += self._srp_facet_torque_body_nm(facet, p_srp_base, sun_dir_body)
            return torque

        if self.config.srp_facet_normal_body is not None:
            facet = {
                "area_m2": self.config.srp_area_m2,
                "normal_body": self.config.srp_facet_normal_body,
                "cp_offset_body_m": self.config.srp_cp_offset_body_m,
            }
            return self._srp_facet_torque_body_nm(facet, p_srp_base, sun_dir_body)

        force_mag = p_srp_base * self.config.srp_area_m2
        f_srp_body = -force_mag * sun_dir_body
        return _cross3(self.config.srp_cp_offset_body_m, f_srp_body)

    @staticmethod
    def _srp_facet_torque_body_nm(facet: dict, pressure_n_m2: float, sun_dir_body: np.ndarray) -> np.ndarray:
        normal_body = np.asarray(facet["normal_body"], dtype=float).reshape(3)
        normal_norm = _norm3(normal_body)
        if normal_norm <= 0.0:
            return np.zeros(3)
        normal_body = normal_body / normal_norm
        incident_dir_body = -sun_dir_body
        illum = max(0.0, -float(np.dot(normal_body, incident_dir_body)))
        if illum <= 0.0:
            return np.zeros(3)
        area_m2 = float(facet["area_m2"])
        cp_offset_body_m = np.asarray(facet["cp_offset_body_m"], dtype=float).reshape(3)
        force_body = pressure_n_m2 * area_m2 * illum * incident_dir_body
        return _cross3(cp_offset_body_m, force_body)

    @staticmethod
    def _drag_facet_torque_body_nm(
        facet: dict, density_kg_m3: float, v_norm_m_s: float, v_rel_body_m_s: np.ndarray, default_cd: float
    ) -> np.ndarray:
        normal_body = np.asarray(facet["normal_body"], dtype=float).reshape(3)
        normal_norm = _norm3(normal_body)
        if normal_norm <= 0.0 or v_norm_m_s <= 0.0:
            return np.zeros(3)
        normal_body = normal_body / normal_norm
        v_hat_body = np.asarray(v_rel_body_m_s, dtype=float).reshape(3) / float(v_norm_m_s)
        projected_area_m2 = float(facet["area_m2"]) * max(0.0, float(np.dot(normal_body, v_hat_body)))
        if projected_area_m2 <= 0.0:
            return np.zeros(3)
        drag_cd = float(facet.get("drag_cd", facet.get("cd", default_cd)))
        cp_offset_body_m = np.asarray(facet["cp_offset_body_m"], dtype=float).reshape(3)
        force_body = -0.5 * float(density_kg_m3) * projected_area_m2 * drag_cd * float(v_norm_m_s) ** 2 * v_hat_body
        return _cross3(cp_offset_body_m, force_body)

    @property
    def _rect_prism_geometry(self) -> RectangularPrismGeometry | None:
        dims = self.config.rectangular_prism_dims_m
        if dims is None:
            return None
        cached = self._rect_prism_geometry_cache
        if cached is None:
            cached = RectangularPrismGeometry(lx_m=float(dims[0]), ly_m=float(dims[1]), lz_m=float(dims[2]))
            object.__setattr__(self, "_rect_prism_geometry_cache", cached)
        return cached


def _empty_drag_facets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return np.empty((0, 3)), np.empty(0), np.empty((0, 3)), np.empty(0)


def _empty_srp_facets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.empty((0, 3)), np.empty(0), np.empty((0, 3))


def _stage_drag_facets(
    facets: tuple[dict, ...],
    *,
    default_cd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray([facet["normal_body"] for facet in facets], dtype=float).reshape(-1, 3)
    areas = np.asarray([facet["area_m2"] for facet in facets], dtype=float)
    cp_offsets = np.asarray([facet["cp_offset_body_m"] for facet in facets], dtype=float).reshape(-1, 3)
    drag_cd = np.asarray(
        [facet.get("drag_cd", facet.get("cd", default_cd)) for facet in facets],
        dtype=float,
    )
    return normals, areas, cp_offsets, drag_cd


def _stage_srp_facets(facets: tuple[dict, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray([facet["normal_body"] for facet in facets], dtype=float).reshape(-1, 3)
    areas = np.asarray([facet["area_m2"] for facet in facets], dtype=float)
    cp_offsets = np.asarray([facet["cp_offset_body_m"] for facet in facets], dtype=float).reshape(-1, 3)
    return normals, areas, cp_offsets


def _drag_facet_signature(facets: tuple[dict, ...] | None, *, default_cd: float) -> tuple:
    if not facets:
        raise ValueError("The staged drag facet plan requires at least one facet.")
    return tuple(
        (
            float(facet["area_m2"]),
            float(facet.get("drag_cd", facet.get("cd", default_cd))),
            tuple(float(value) for value in np.asarray(facet["normal_body"], dtype=float).reshape(3)),
            tuple(float(value) for value in np.asarray(facet["cp_offset_body_m"], dtype=float).reshape(3)),
        )
        for facet in facets
    )


def _srp_facet_signature(facets: tuple[dict, ...] | None) -> tuple:
    if not facets:
        raise ValueError("The staged SRP facet plan requires at least one facet.")
    return tuple(
        (
            float(facet["area_m2"]),
            tuple(float(value) for value in np.asarray(facet["normal_body"], dtype=float).reshape(3)),
            tuple(float(value) for value in np.asarray(facet["cp_offset_body_m"], dtype=float).reshape(3)),
        )
        for facet in facets
    )
