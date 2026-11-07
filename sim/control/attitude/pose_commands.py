from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.models import StateTruth
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping, transform_position
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn

_DEFAULT_PANEL_NORMAL_BODY = np.array([0.0, 0.0, 1.0])
_DEFAULT_BORESIGHT_BODY = np.array([1.0, 0.0, 0.0])


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.array(v, dtype=float).reshape(3)
    if not np.all(np.isfinite(x)):
        raise ValueError("Vector must contain finite values.")
    n = float(np.linalg.norm(x))
    if n <= eps:
        raise ValueError("Vector magnitude is zero; cannot normalize.")
    return x / n


def _orthogonal_basis_from_primary(primary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e1 = _unit(primary)
    basis = np.eye(3)
    ref_idx = int(np.argmin(np.abs(e1)))
    ref = basis[:, ref_idx]
    e2 = ref - np.dot(ref, e1) * e1
    n2 = float(np.linalg.norm(e2))
    if n2 <= 1e-12:
        ref = basis[:, (ref_idx + 1) % 3]
        e2 = ref - np.dot(ref, e1) * e1
        n2 = float(np.linalg.norm(e2))
        if n2 <= 1e-12:
            raise ValueError("Failed to build orthogonal basis from primary axis.")
    e2 = e2 / n2
    e3 = np.cross(e1, e2)
    e3 = e3 / max(float(np.linalg.norm(e3)), 1e-12)
    return e2, e3


def _attitude_quat_align_primary(
    truth: StateTruth,
    primary_axis_body: np.ndarray,
    target_axis_eci: np.ndarray,
    secondary_axis_body: np.ndarray | None = None,
    secondary_axis_eci_hint: np.ndarray | None = None,
) -> np.ndarray:
    b1 = _unit(primary_axis_body)
    b2, b3 = _orthogonal_basis_from_primary(b1) if secondary_axis_body is None else (None, None)
    if secondary_axis_body is not None:
        sb = _unit(secondary_axis_body)
        b2 = sb - np.dot(sb, b1) * b1
        n2 = float(np.linalg.norm(b2))
        if n2 <= 1e-12:
            b2, b3 = _orthogonal_basis_from_primary(b1)
        else:
            b2 = b2 / n2
            b3 = np.cross(b1, b2)
            b3 = b3 / max(float(np.linalg.norm(b3)), 1e-12)
    assert b2 is not None and b3 is not None
    b_mat = np.column_stack((b1, b2, b3))

    i1 = _unit(target_axis_eci)
    if secondary_axis_eci_hint is None:
        c_nb_cur = quaternion_to_dcm_bn(truth.attitude_quat_bn).T
        i2_hint = c_nb_cur @ b2
    else:
        i2_hint = _unit(secondary_axis_eci_hint)
    i2 = i2_hint - np.dot(i2_hint, i1) * i1
    n2i = float(np.linalg.norm(i2))
    if n2i <= 1e-12:
        i2, i3 = _orthogonal_basis_from_primary(i1)
    else:
        i2 = i2 / n2i
        i3 = np.cross(i1, i2)
        i3 = i3 / max(float(np.linalg.norm(i3)), 1e-12)
        i2 = np.cross(i3, i1)
        i2 = i2 / max(float(np.linalg.norm(i2)), 1e-12)
    i_mat = np.column_stack((i1, i2, i3))

    c_nb_des = i_mat @ b_mat.T
    c_bn_des = c_nb_des.T
    return dcm_to_quaternion_bn(c_bn_des)


@dataclass(frozen=True)
class PoseCommandGenerator:
    @staticmethod
    def sun_track(
        truth: StateTruth,
        sun_dir_eci: np.ndarray,
        panel_normal_body: np.ndarray | None = None,
        panel_tangent_body: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return q_bn that points the panel normal toward the Sun direction.
        """
        return _attitude_quat_align_primary(
            truth=truth,
            primary_axis_body=np.array(
                _DEFAULT_PANEL_NORMAL_BODY if panel_normal_body is None else panel_normal_body,
                dtype=float,
            ),
            target_axis_eci=np.array(sun_dir_eci, dtype=float),
            secondary_axis_body=None if panel_tangent_body is None else np.array(panel_tangent_body, dtype=float),
            secondary_axis_eci_hint=None,
        )

    @staticmethod
    def spotlight_latlon(
        truth: StateTruth,
        latitude_deg: float,
        longitude_deg: float,
        altitude_km: float = 0.0,
        boresight_body: np.ndarray | None = None,
        up_body: np.ndarray | None = None,
        frame_context: FrameContext | None = None,
    ) -> np.ndarray:
        """
        Return q_bn that points a boresight to an Earth-fixed lat/lon target.
        """
        lat = np.deg2rad(float(latitude_deg))
        lon = np.deg2rad(float(longitude_deg))
        r_km = EARTH_RADIUS_KM + float(altitude_km)
        r_ecef = np.array(
            [
                r_km * np.cos(lat) * np.cos(lon),
                r_km * np.cos(lat) * np.sin(lon),
                r_km * np.sin(lat),
            ],
            dtype=float,
        )
        frame_ctx = frame_context or frame_context_from_mapping({}, source="attitude_pose")
        r_target_eci = transform_position(r_ecef, "ecef", "eci", t_s=truth.t_s, context=frame_ctx)
        los_eci = r_target_eci - np.array(truth.position_eci_km, dtype=float)
        return _attitude_quat_align_primary(
            truth=truth,
            primary_axis_body=np.array(
                _DEFAULT_BORESIGHT_BODY if boresight_body is None else boresight_body, dtype=float
            ),
            target_axis_eci=los_eci,
            secondary_axis_body=None if up_body is None else np.array(up_body, dtype=float),
            secondary_axis_eci_hint=None,
        )

    @staticmethod
    def spotlight_ric_direction(
        truth: StateTruth,
        ric_direction: np.ndarray,
        boresight_body: np.ndarray | None = None,
        up_body: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return q_bn that points boresight along a specified RIC direction.
        """
        d_ric = _unit(np.array(ric_direction, dtype=float))
        c_ir = ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s)
        d_eci = c_ir @ d_ric
        return _attitude_quat_align_primary(
            truth=truth,
            primary_axis_body=np.array(
                _DEFAULT_BORESIGHT_BODY if boresight_body is None else boresight_body, dtype=float
            ),
            target_axis_eci=d_eci,
            secondary_axis_body=None if up_body is None else np.array(up_body, dtype=float),
            secondary_axis_eci_hint=None,
        )
