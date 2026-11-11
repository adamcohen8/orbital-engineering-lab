from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from sim.actuators.attitude import ReactionWheelLimits
from sim.control.attitude import (
    ReactionWheelPDController,
    ReactionWheelPIDController,
    RICFramePDController,
    RICFramePIDController,
)
from sim.core.interfaces import Controller
from sim.core.models import StateBelief
from sim.optimization.base import ParameterBound
from sim.optimization.pso import ParticleSwarmOptimizer, PSOConfig
from sim.utils.quaternion import dcm_to_quaternion_bn


def _rot_x(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])


def _rot_y(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])


def _rot_z(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])


def _quat_error_deg(q_target: np.ndarray, q_current: np.ndarray) -> float:
    qt = np.array(q_target, dtype=float)
    qc = np.array(q_current, dtype=float)
    qt /= max(np.linalg.norm(qt), 1e-12)
    qc /= max(np.linalg.norm(qc), 1e-12)
    qt_conj = np.array([qt[0], -qt[1], -qt[2], -qt[3]])
    qe = np.array(
        [
            qt_conj[0] * qc[0] - qt_conj[1] * qc[1] - qt_conj[2] * qc[2] - qt_conj[3] * qc[3],
            qt_conj[0] * qc[1] + qt_conj[1] * qc[0] + qt_conj[2] * qc[3] - qt_conj[3] * qc[2],
            qt_conj[0] * qc[2] - qt_conj[1] * qc[3] + qt_conj[2] * qc[0] + qt_conj[3] * qc[1],
            qt_conj[0] * qc[3] + qt_conj[1] * qc[2] - qt_conj[2] * qc[1] + qt_conj[3] * qc[0],
        ]
    )
    if qe[0] < 0.0:
        qe *= -1.0
    return float(np.rad2deg(2.0 * np.arccos(np.clip(qe[0], -1.0, 1.0))))


@dataclass(frozen=True)
class AttitudeTuneCase:
    name: str
    duration_s: float = 600.0
    dt_s: float = 1.0
    attitude_dt_s: float = 0.01
    wheel_scale: float = 1.0
    orbit_radius_km: float = 6778.0
    phase_rad: float = 0.0
    init_eci_euler_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_rate_body_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    desired_eci_euler_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    desired_rate_body_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    desired_ric_euler_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    desired_ric_rate_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    enable_disturbances: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.duration_s)) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if (
            not np.isfinite(float(self.dt_s))
            or not np.isfinite(float(self.attitude_dt_s))
            or self.dt_s <= 0.0
            or self.attitude_dt_s <= 0.0
        ):
            raise ValueError("dt_s and attitude_dt_s must be positive.")
        steps = float(self.duration_s) / float(self.dt_s)
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1.0e-12):
            raise ValueError("duration_s must be an integer multiple of dt_s.")
        if self.wheel_scale <= 0.0:
            raise ValueError("wheel_scale must be positive.")


@dataclass(frozen=True)
class TuneCaseResult:
    name: str
    final_error_deg: float
    mean_error_deg: float
    max_error_deg: float
    mean_torque_nm: float
    mean_rate_norm_rad_s: float


ControllerAlgorithm = Literal["pd", "pid", "ric_pd", "ric_pid"]


def default_parameter_bounds(algorithm: ControllerAlgorithm) -> list[ParameterBound]:
    if algorithm in ("pd", "ric_pd"):
        return [
            ParameterBound("kp_x", 0.01, 2.5),
            ParameterBound("kp_y", 0.01, 2.5),
            ParameterBound("kp_z", 0.01, 2.5),
            ParameterBound("kd_x", 0.05, 8.0),
            ParameterBound("kd_y", 0.05, 8.0),
            ParameterBound("kd_z", 0.05, 8.0),
        ]
    if algorithm in ("pid", "ric_pid"):
        return [
            ParameterBound("kp_x", 0.01, 2.5),
            ParameterBound("kp_y", 0.01, 2.5),
            ParameterBound("kp_z", 0.01, 2.5),
            ParameterBound("kd_x", 0.05, 8.0),
            ParameterBound("kd_y", 0.05, 8.0),
            ParameterBound("kd_z", 0.05, 8.0),
            ParameterBound("ki_x", 0.0, 0.5),
            ParameterBound("ki_y", 0.0, 0.5),
            ParameterBound("ki_z", 0.0, 0.5),
        ]
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def preset_tuning_cases(name: str = "attitude_ric_nominal") -> list[AttitudeTuneCase]:
    if name == "attitude_ric_nominal":
        return [
            AttitudeTuneCase(
                name="ric_nominal_1",
                init_eci_euler_rad=np.deg2rad(np.array([20.0, -15.0, 25.0])),
                init_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
                desired_ric_euler_rad=np.zeros(3),
                desired_ric_rate_rad_s=np.zeros(3),
                enable_disturbances=True,
            ),
            AttitudeTuneCase(
                name="ric_nominal_2",
                init_eci_euler_rad=np.deg2rad(np.array([-30.0, 10.0, -20.0])),
                init_rate_body_rad_s=np.array([0.02, -0.01, 0.015]),
                desired_ric_euler_rad=np.zeros(3),
                desired_ric_rate_rad_s=np.zeros(3),
                enable_disturbances=True,
            ),
            AttitudeTuneCase(
                name="ric_nominal_3",
                init_eci_euler_rad=np.deg2rad(np.array([45.0, -25.0, 10.0])),
                init_rate_body_rad_s=np.array([-0.03, 0.015, -0.02]),
                desired_ric_euler_rad=np.zeros(3),
                desired_ric_rate_rad_s=np.zeros(3),
                enable_disturbances=True,
            ),
        ]
    if name == "attitude_eci_nominal":
        return [
            AttitudeTuneCase(
                name="eci_nominal_1",
                init_eci_euler_rad=np.deg2rad(np.array([20.0, -15.0, 25.0])),
                init_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
                desired_eci_euler_rad=np.zeros(3),
                desired_rate_body_rad_s=np.zeros(3),
                enable_disturbances=True,
            ),
            AttitudeTuneCase(
                name="eci_nominal_2",
                init_eci_euler_rad=np.deg2rad(np.array([-35.0, 20.0, -30.0])),
                init_rate_body_rad_s=np.array([0.025, -0.01, 0.02]),
                desired_eci_euler_rad=np.zeros(3),
                desired_rate_body_rad_s=np.zeros(3),
                enable_disturbances=True,
            ),
        ]
    raise ValueError(f"Unknown preset case set: {name}")


def default_case_cost(result: TuneCaseResult) -> float:
    # Lower is better. Emphasize tracking error, then rate settling and control effort.
    return (
        0.5 * result.final_error_deg
        + 0.35 * result.mean_error_deg
        + 20.0 * result.mean_rate_norm_rad_s
        + 5.0 * result.mean_torque_nm
    )


@dataclass(frozen=True)
class GainTuningResult:
    algorithm: ControllerAlgorithm
    best_parameters: dict[str, float]
    aggregate_cost: float
    per_case_results: list[TuneCaseResult]
    history_best_cost: list[float]


def _parameters_from_vector(bounds: list[ParameterBound], x: np.ndarray) -> dict[str, float]:
    if x.size != len(bounds):
        raise ValueError("Parameter vector length does not match bounds.")
    return {b.name: float(v) for b, v in zip(bounds, x)}


def _make_controller(
    algorithm: ControllerAlgorithm,
    params: dict[str, float],
    case: AttitudeTuneCase,
) -> Controller:
    from sim.presets.attitude_control import BASIC_REACTION_WHEEL_TRIAD

    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float) * float(
        case.wheel_scale
    )
    kp = np.array([params["kp_x"], params["kp_y"], params["kp_z"]], dtype=float)
    kd = np.array([params["kd_x"], params["kd_y"], params["kd_z"]], dtype=float)

    if algorithm == "pd":
        q_des = dcm_to_quaternion_bn(
            _rot_z(case.desired_eci_euler_rad[2])
            @ _rot_y(case.desired_eci_euler_rad[1])
            @ _rot_x(case.desired_eci_euler_rad[0])
        )
        return ReactionWheelPDController(
            wheel_axes_body=wheel_axes,
            wheel_torque_limits_nm=wheel_limits,
            desired_attitude_quat_bn=q_des,
            desired_rate_body_rad_s=np.array(case.desired_rate_body_rad_s, dtype=float),
            kp=kp,
            kd=kd,
        )

    if algorithm == "pid":
        q_des = dcm_to_quaternion_bn(
            _rot_z(case.desired_eci_euler_rad[2])
            @ _rot_y(case.desired_eci_euler_rad[1])
            @ _rot_x(case.desired_eci_euler_rad[0])
        )
        ki = np.array([params["ki_x"], params["ki_y"], params["ki_z"]], dtype=float)
        return ReactionWheelPIDController(
            wheel_axes_body=wheel_axes,
            wheel_torque_limits_nm=wheel_limits,
            desired_attitude_quat_bn=q_des,
            desired_rate_body_rad_s=np.array(case.desired_rate_body_rad_s, dtype=float),
            kp=kp,
            kd=kd,
            ki=ki,
            integral_limit=np.array([0.4, 0.4, 0.4], dtype=float),
        )

    if algorithm == "ric_pd":
        inner = ReactionWheelPDController(
            wheel_axes_body=wheel_axes,
            wheel_torque_limits_nm=wheel_limits,
            kp=kp,
            kd=kd,
        )
        ctrl = RICFramePDController(pd=inner)
        ctrl.set_desired_ric_state(
            yaw_r_rad=float(case.desired_ric_euler_rad[0]),
            roll_i_rad=float(case.desired_ric_euler_rad[1]),
            pitch_c_rad=float(case.desired_ric_euler_rad[2]),
            w_ric_rad_s=np.array(case.desired_ric_rate_rad_s, dtype=float),
        )
        return ctrl

    if algorithm == "ric_pid":
        ki = np.array([params["ki_x"], params["ki_y"], params["ki_z"]], dtype=float)
        inner = ReactionWheelPIDController(
            wheel_axes_body=wheel_axes,
            wheel_torque_limits_nm=wheel_limits,
            kp=kp,
            kd=kd,
            ki=ki,
            integral_limit=np.array([0.4, 0.4, 0.4], dtype=float),
        )
        ctrl = RICFramePIDController(pid=inner)
        ctrl.set_desired_ric_state(
            yaw_r_rad=float(case.desired_ric_euler_rad[0]),
            roll_i_rad=float(case.desired_ric_euler_rad[1]),
            pitch_c_rad=float(case.desired_ric_euler_rad[2]),
            w_ric_rad_s=np.array(case.desired_ric_rate_rad_s, dtype=float),
        )
        return ctrl

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _run_case(algorithm: ControllerAlgorithm, params: dict[str, float], case: AttitudeTuneCase) -> TuneCaseResult:
    from sim.presets.satellites import BASIC_SATELLITE
    from sim.presets.simulation import build_sim_object_from_presets

    init_c_bn = (
        _rot_z(case.init_eci_euler_rad[2]) @ _rot_y(case.init_eci_euler_rad[1]) @ _rot_x(case.init_eci_euler_rad[0])
    )
    q0 = dcm_to_quaternion_bn(init_c_bn)
    ctrl = _make_controller(algorithm, params, case)
    sat = build_sim_object_from_presets(
        object_id=f"tune_{case.name}",
        dt_s=case.dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=case.enable_disturbances,
        enable_attitude_knowledge=True,
        attitude_quat_bn=q0,
        angular_rate_body_rad_s=np.array(case.init_rate_body_rad_s, dtype=float),
        orbit_radius_km=case.orbit_radius_km,
        phase_rad=case.phase_rad,
        controller=ctrl,
        attitude_substep_s=case.attitude_dt_s,
    )

    if hasattr(sat.actuator, "attitude") and hasattr(sat.actuator.attitude, "reaction_wheels"):
        rw = sat.actuator.attitude.reaction_wheels
        if rw is not None:
            sat.actuator.attitude.reaction_wheels = ReactionWheelLimits(
                max_torque_nm=rw.max_torque_nm * float(case.wheel_scale),
                max_momentum_nms=rw.max_momentum_nms * float(case.wheel_scale),
            )

    q_target_eci = None
    if algorithm in ("pd", "pid"):
        q_target_eci = dcm_to_quaternion_bn(
            _rot_z(case.desired_eci_euler_rad[2])
            @ _rot_y(case.desired_eci_euler_rad[1])
            @ _rot_x(case.desired_eci_euler_rad[0])
        )

    n = int(np.ceil(case.duration_s / case.dt_s))
    err = np.zeros(n + 1)
    tau_norm = np.zeros(n + 1)
    w_norm = np.zeros(n + 1)

    if q_target_eci is not None:
        err[0] = _quat_error_deg(q_target_eci, sat.truth.attitude_quat_bn)

    for k in range(n):
        t_now = sat.truth.t_s
        meas = sat.sensor.measure(sat.truth, env={}, t_s=t_now + case.dt_s)
        sat.belief = sat.estimator.update(sat.belief, meas, t_s=t_now + case.dt_s)
        belief_att = StateBelief(
            state=np.hstack(
                (
                    sat.truth.position_eci_km,
                    sat.truth.velocity_eci_km_s,
                    sat.truth.attitude_quat_bn,
                    sat.truth.angular_rate_body_rad_s,
                )
            ),
            covariance=np.eye(13),
            last_update_t_s=t_now,
        )
        cmd = ctrl.act(belief_att, t_s=t_now, budget_ms=1.0)
        applied = sat.actuator.apply(cmd, sat.limits, case.dt_s)
        sat.truth = sat.dynamics.step(sat.truth, applied, env={}, dt_s=case.dt_s)
        tau_norm[k + 1] = float(np.linalg.norm(applied.torque_body_nm))
        w_norm[k + 1] = float(np.linalg.norm(sat.truth.angular_rate_body_rad_s))

        if "attitude_error_deg" in cmd.mode_flags:
            err[k + 1] = float(cmd.mode_flags["attitude_error_deg"])
        elif q_target_eci is not None:
            err[k + 1] = _quat_error_deg(q_target_eci, sat.truth.attitude_quat_bn)
        else:
            err[k + 1] = err[k]

    return TuneCaseResult(
        name=case.name,
        final_error_deg=float(err[-1]),
        mean_error_deg=float(np.mean(err)),
        max_error_deg=float(np.max(err)),
        mean_torque_nm=float(np.mean(tau_norm)),
        mean_rate_norm_rad_s=float(np.mean(w_norm)),
    )


def tune_controller_gains(
    algorithm: ControllerAlgorithm,
    bounds: list[ParameterBound] | None = None,
    test_cases: list[AttitudeTuneCase] | None = None,
    preset_case_set: str | None = "attitude_ric_nominal",
    case_cost_fn: Callable[[TuneCaseResult], float] | None = None,
    optimizer: str = "pso",
    pso_config: PSOConfig | None = None,
    seed: int = 0,
) -> GainTuningResult:
    if test_cases is None:
        if preset_case_set is None:
            raise ValueError("Provide test_cases or preset_case_set.")
        test_cases = preset_tuning_cases(preset_case_set)
    if not test_cases:
        raise ValueError("At least one test case is required.")

    bounds = default_parameter_bounds(algorithm) if bounds is None else bounds
    case_cost_fn = default_case_cost if case_cost_fn is None else case_cost_fn

    if optimizer != "pso":
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Currently supported: ['pso'].")
    pso = ParticleSwarmOptimizer(pso_config or PSOConfig())

    def objective(x: np.ndarray) -> float:
        params = _parameters_from_vector(bounds, x)
        case_costs = []
        for tc in test_cases:
            r = _run_case(algorithm=algorithm, params=params, case=tc)
            case_costs.append(float(case_cost_fn(r)))
        return float(np.mean(case_costs))

    opt_result = pso.optimize(objective=objective, bounds=bounds, seed=seed)
    best_params = _parameters_from_vector(bounds, opt_result.best_x)
    best_case_results = [_run_case(algorithm=algorithm, params=best_params, case=tc) for tc in test_cases]

    return GainTuningResult(
        algorithm=algorithm,
        best_parameters=best_params,
        aggregate_cost=float(np.mean([case_cost_fn(r) for r in best_case_results])),
        per_case_results=best_case_results,
        history_best_cost=opt_result.history_best_cost,
    )
