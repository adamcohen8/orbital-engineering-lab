"""Shared batch/filter tracking predictions and correction receipts.

SLR is passive two-way laser ranging. Ground range-rate is explicitly reduced
geometric LOS rate; raw Doppler counts and carrier phase are not accepted.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.linalg import block_diag

from sim.estimation.covariance import validate_covariance
from sim.pro_tracking_od.core import _angular_difference, _measurement_vectors
from sim.slr_tracking import C_LIGHT_KM_S, predict_slr_two_way_time_of_flight_s


class TrackingMeasurementModel:
    def __init__(self, dynamics, *, frame_context=None, frame_model="simple", eop_path=None):
        self.dynamics, self.frame_context = dynamics, frame_context
        self.frame_model, self.eop_path = frame_model, eop_path

    def _clock_state(self, row, state):
        offset = sum(
            state[i]
            for i, p in enumerate(self.dynamics.parameters, 6)
            if p.kind == "clock_offset" and p.station_id == row.get("station_id")
        )
        changed = copy.deepcopy(row)
        if offset:
            from sim.estimation.sequential_od import _propagate

            ctx, env = self.dynamics.force_inputs(state)
            shifted = (
                _propagate(
                    state[:6], row["time_s"], abs(offset), self.dynamics.propagator, ctx, env, self.dynamics.step_s
                )
                if offset > 0
                else self.dynamics.propagator.propagate(state[:6], offset, row["time_s"], np.zeros(3), env, ctx)
            )
            state = state.copy()
            state[:6] = shifted
            changed["frame_epoch_jd_utc"] = row["time_jd_utc"]
            changed["clock_offset_s"] = offset
            changed["time_s"] += offset
            changed["time_jd_utc"] += offset / 86400.0
        return changed, state

    def vectors(self, source, state):
        row, state = self._clock_state(source, np.asarray(state, dtype=float))
        kind = row["kind"]
        audit = {"measurement_id": row["measurement_id"], "kind": kind}
        if kind == "pv":
            observed = np.asarray(row["vector"], dtype=float)
            if observed.shape not in {(3,), (6,)}:
                raise ValueError("PV rows require three position or six position/velocity components")
            predicted = state[: len(observed)].copy()
            covariance = np.asarray(row["covariance"], dtype=float)
            angular = set()
        elif kind in {"ground", "optical"}:
            if row.get("measurement_semantics") != "reduced_geometric":
                raise ValueError("ground/optical rows must explicitly declare reduced_geometric semantics")
            model = "ground_topocentric" if kind == "ground" else "optical_ra_dec"
            observed, predicted, covariance = _measurement_vectors(model, row, state[:6], self.frame_context)
            angular = (
                {0} if kind == "optical" else {i for i, name in enumerate(row["components"]) if name == "azimuth_deg"}
            )
        elif kind == "slr":
            if not np.isfinite(row["sigma_range_km"]) or row["sigma_range_km"] <= 0:
                raise ValueError("SLR range sigma must be finite and positive")
            geometry, correction = self.slr_prediction(row, state)
            observed = np.array([row["observed_time_of_flight_s"] * C_LIGHT_KM_S / 2])
            predicted = np.array([geometry["predicted_two_way_range_km"] + correction["one_way_equivalent_km"]])
            covariance = np.array([[float(row["sigma_range_km"]) ** 2]])
            angular = set()
            audit.update(correction)
        else:
            raise ValueError("unsupported tracking observable; raw radiometric data require a separate contract")
        for i, parameter in enumerate(self.dynamics.parameters, 6):
            if parameter.kind == "range_bias" and parameter.station_id == row.get("station_id"):
                if kind == "slr":
                    predicted[0] += state[i]
                elif kind == "ground" and "range_km" in row["components"]:
                    predicted[row["components"].index("range_km")] += state[i]
        if not np.all(np.isfinite(np.r_[observed, predicted])):
            raise ValueError("tracking vectors must be finite")
        validate_covariance(covariance, len(observed), name="measurement covariance")
        np.linalg.cholesky(covariance)  # Measurement noise must be strictly PD.
        return observed, predicted, covariance, angular, audit

    def slr_prediction(self, row, state):
        from sim.pro_slr_od.media import two_way_optical_media_correction

        required = {"troposphere", "center_of_mass", "target_signature", "station_eccentricity", "instrument"}
        corrections = row.get("corrections", {})
        if set(corrections) != required:
            raise ValueError("SLR requires explicit correction disposition for every modeled effect")
        for name, entry in corrections.items():
            if entry.get("status") not in {"applied", "model", "unmodeled"}:
                raise ValueError(f"invalid SLR correction status: {name}")
            digest = entry.get("source_sha256", "")
            if entry["status"] != "unmodeled" and (
                len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("SLR corrections require an authoritative source receipt")
        applied = bool(row["correction_flags"].get("tropospheric_refraction_applied", False))
        if applied != (corrections["troposphere"]["status"] == "applied"):
            raise ValueError("SLR CRD troposphere flag disagrees with correction disposition")
        com_applied = bool(row["correction_flags"].get("center_of_mass_applied", False))
        if com_applied and corrections["center_of_mass"]["status"] != "applied":
            raise ValueError("SLR CRD center-of-mass correction would be applied twice")
        station = np.asarray(row["station_metadata"]["ecef_km"], dtype=float)
        epoch = float(row["time_jd_utc"])
        metadata = row["station_metadata"]
        if "velocity_ecef_km_year" in metadata:
            station = station + np.asarray(metadata["velocity_ecef_km_year"]) * (
                (epoch - float(metadata["reference_epoch_jd_utc"])) / 365.25
            )
        if corrections["station_eccentricity"]["status"] == "model":
            station = station + np.asarray(corrections["station_eccentricity"]["ecef_offset_km"], dtype=float)
        ctx, env = self.dynamics.force_inputs(state)

        def at_offset(offset):
            # A light-time leg is short, signed and uses the same ONP owner.
            prop = copy.copy(self.dynamics.propagator)
            prop._rkf78_h_next = prop._rkf78_last_t_s = None
            return prop.propagate(state[:6].copy(), float(offset), float(row["time_s"]), np.zeros(3), env, ctx)

        geometry = predict_slr_two_way_time_of_flight_s(
            epoch_jd_utc=row.get("frame_epoch_jd_utc", epoch),
            epoch_offset_s=row.get("clock_offset_s", 0.0),
            epoch_event=row["epoch_event"],
            station_ecef_km=station,
            state_at_jd_utc=lambda jd: at_offset((jd - epoch) * 86400),
            state_at_offset_s=at_offset,
            frame_model=self.frame_model,
            eop_path=self.eop_path,
        )
        correction = 0.0
        if corrections["troposphere"]["status"] == "model":
            media_row = {**row, "station_metadata": {**metadata, "ecef_km": station.tolist()}}
            correction += two_way_optical_media_correction(
                media_row,
                geometry,
                lambda jd: at_offset((jd - epoch) * 86400),
                frame_model=self.frame_model,
                eop_path=self.eop_path,
            )["one_way_equivalent_km"]
        for name in ("center_of_mass", "target_signature", "instrument"):
            if corrections[name]["status"] == "model":
                # Explicit signed one-way-equivalent path correction, never an inferred calibration.
                value = float(corrections[name]["one_way_equivalent_km"])
                if not np.isfinite(value):
                    raise ValueError("SLR path corrections must be finite")
                correction += value
        return geometry, {
            "one_way_equivalent_km": correction,
            "corrections": copy.deepcopy(corrections),
            "unmodeled_effects": [k for k, v in corrections.items() if v["status"] == "unmodeled"],
        }

    def __call__(self, row, state, *, with_jacobian=True):
        if "epoch_rows" in row:
            values = [self(item, state, with_jacobian=with_jacobian) for item in row["epoch_rows"]]
            covariance = row.get("epoch_covariance")
            if covariance is None and row.get("independent_measurements") is not True:
                raise ValueError("simultaneous rows require cross covariance or explicit independence")
            full_covariance = (
                np.asarray(covariance) if covariance is not None else block_diag(*[v["covariance"] for v in values])
            )
            validate_covariance(full_covariance, sum(len(v["observed"]) for v in values), name="epoch covariance")
            np.linalg.cholesky(full_covariance)
            return {
                "observed": np.concatenate([v["observed"] for v in values]),
                "predicted": np.concatenate([v["predicted"] for v in values]),
                "innovation": np.concatenate([v["innovation"] for v in values]),
                "jacobian": np.vstack([v["jacobian"] for v in values]),
                "covariance": full_covariance,
            }
        observed, predicted, covariance, angular, _audit = self.vectors(row, state)
        jacobian = np.zeros((len(observed), len(state)))
        for i, step in enumerate(self.dynamics.steps if with_jacobian else []):
            plus, minus = state.copy(), state.copy()
            plus[i] += step
            minus[i] -= step
            high, low = self.vectors(row, plus)[1], self.vectors(row, minus)[1]
            jacobian[:, i] = _angular_difference(high, low, angular) / (2 * step)
        return {
            "observed": observed,
            "predicted": predicted,
            "jacobian": jacobian,
            "covariance": covariance,
            "innovation": _angular_difference(observed, predicted, angular),
        }

    def batch_residual(self, rows, state_at_epoch):
        """Shared predictions and full within-epoch whitening for batch least squares."""
        return np.concatenate(
            [
                np.linalg.solve(np.linalg.cholesky(value["covariance"]), value["innovation"])
                for row in rows
                for value in [self(row, state_at_epoch(float(row["time_s"])), with_jacobian=False)]
            ]
        )
