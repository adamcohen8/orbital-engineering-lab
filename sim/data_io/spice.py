"""Optional NAIF SPK/CK interchange and explicit center/frame/time transforms."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from .common import read_bytes, source_receipt, temporary_path
from .contracts import AttitudeHistory, DataIOError, Trajectory
from .time import at, format_time

_LOCK = threading.RLock()


@contextmanager
def kernel_session(kernels):
    try:
        import spiceypy as sp
    except ImportError as exc:
        raise DataIOError("SPICE interchange requires the optional 'spiceypy' IO dependency.") from exc
    if not kernels:
        raise DataIOError("Explicit local kernels, including time kernels, are required.")
    paths = [Path(k).expanduser().resolve(strict=True) for k in kernels]
    for path in paths:
        with path.open("rb") as stream:
            prefix = stream.read(32).lstrip().upper()
        if path.suffix.lower() in {".tm", ".mk"} or prefix.startswith(b"KPL/MK"):
            raise DataIOError("List kernels explicitly; meta-kernel path discovery is outside the IO contract.")
    receipts = [source_receipt(p) for p in paths]
    with _LOCK:
        previous = {str(Path(sp.kdata(i, "ALL")[0]).resolve()) for i in range(sp.ktotal("ALL"))}
        if previous - {str(path) for path in paths}:
            raise DataIOError(
                "Unlisted kernels are already loaded in this process; include every active kernel explicitly or use an isolated process."
            )
        loaded = []
        try:
            for p in paths:
                if str(p) not in previous:
                    sp.furnsh(str(p))
                    loaded.append(str(p))
            yield sp, receipts
            if [source_receipt(p) for p in paths] != receipts:
                raise DataIOError("Kernel inputs changed during conversion.")
        finally:
            for p in reversed(loaded):
                sp.unload(p)


def _frame(frame):
    if frame in {"EME2000", "OEL/ECI/J2000"}:
        return "J2000"
    if frame.startswith("SPICE:") and frame[6:]:
        return frame[6:]
    raise DataIOError("SPICE requires EME2000 or an explicit SPICE:<frame> name; GCRF/ITRF are not relabeled.")


def _et(sp, epoch):
    return float(sp.str2et(format_time(epoch)))


def encode_spk(t: Trajectory, *, target_id: int, center_id: int, kernels, degree=3) -> tuple[bytes, dict]:
    if t.covariances is not None:
        raise DataIOError("SPK has no covariance field; use a separate covariance product.")
    if degree != 3 or len(t.times_s) < 2:
        raise DataIOError("This SPK type-13 profile requires degree 3 and at least two PV samples.")
    if t.metadata.get("discontinuity_sample_indices"):
        raise DataIOError(
            "Separate SPK segments are required at trajectory discontinuities; this writer accepts one continuous arc."
        )
    if target_id == center_id:
        raise DataIOError("SPK target and center must differ.")
    with kernel_session(kernels) as (sp, receipts), temporary_path(".bsp") as directory:
        actual_center = str(sp.bodc2n(int(center_id))).upper()
        if t.center.upper() not in {actual_center, str(center_id)}:
            raise DataIOError("Explicit SPK center ID disagrees with trajectory center.")
        epochs = np.asarray([_et(sp, at(t.epoch, time)) for time in t.times_s])
        path = str(Path(directory) / "trajectory.bsp")
        handle = sp.spkopn(path, "OEL saved-evidence trajectory", 0)
        try:
            sp.spkw13(
                handle,
                int(target_id),
                int(center_id),
                _frame(t.frame),
                epochs[0],
                epochs[-1],
                "OEL bounded PV Hermite",
                degree,
                len(epochs),
                t.states.copy(),
                epochs,
            )
        except BaseException:
            sp.dafcls(handle)
            raise
        else:
            sp.spkcls(handle)
        # Read through NAIF rather than trusting writer return status.
        sp.furnsh(path)
        try:
            readback = np.asarray(
                [sp.spkez(int(target_id), et, _frame(t.frame), "NONE", int(center_id))[0] for et in epochs]
            )
        finally:
            sp.unload(path)
        if not np.allclose(readback, t.states, atol=1e-8, rtol=1e-13):
            raise DataIOError("SPK independent API sample read-back failed.")
        return read_bytes(path), {
            "kernel_inputs": receipts,
            "spk_type": 13,
            "degree": degree,
            "target_id": target_id,
            "center_id": center_id,
            "sample_readback": True,
        }


def read_spk(path, *, target_id: int, center_id: int, frame: str, epochs, object_id: str, kernels) -> Trajectory:
    if not epochs:
        raise DataIOError("SPK import requires explicit sample epochs; interpolation policy belongs to NAIF.")
    with kernel_session([*kernels, path]) as (sp, receipts):
        states = [
            sp.spkez(int(target_id), _et(sp, epoch), _frame(frame), "NONE", int(center_id))[0] for epoch in epochs
        ]
        center = sp.bodc2n(int(center_id))
    return Trajectory(
        object_id,
        center,
        frame,
        epochs[0],
        [e.tai_seconds - epochs[0].tai_seconds for e in epochs],
        states,
        metadata={"kernel_inputs": receipts, "aberration_correction": "NONE", "source_format": "spk"},
    )


def transform_spice(t: Trajectory, *, target_center: str, target_frame: str, kernels) -> Trajectory:
    states, covs = [], []
    with kernel_session(kernels) as (sp, receipts):
        source_frame, output_frame = _frame(t.frame), _frame(target_frame)
        for time, state in zip(t.times_s, t.states):
            et = _et(sp, at(t.epoch, time))
            matrix = sp.sxform(source_frame, output_frame, et)
            center_delta = sp.spkezr(t.center, et, output_frame, "NONE", target_center)[0]
            states.append(matrix @ state + center_delta)
        if t.covariances is not None:
            for time, covariance in zip(t.covariance_times_s, t.covariances):
                matrix = sp.sxform(source_frame, output_frame, _et(sp, at(t.epoch, time)))
                covs.append(matrix @ covariance @ matrix.T)
    return replace(
        t,
        center=target_center,
        frame=target_frame,
        states=states,
        covariances=covs or None,
        metadata={
            **t.metadata,
            "spice_transform": {
                "kernel_inputs": receipts,
                "aberration_correction": "NONE",
                "origin_translation_covariance": "deterministic ephemeris; center uncertainty not added",
            },
        },
    )


def encode_ck(h: AttitudeHistory, *, instrument_id: int, clock_id: int, kernels) -> tuple[bytes, dict]:
    if len(h.times_s) < 2:
        raise DataIOError("CK type 3 requires at least two attitude samples.")
    with kernel_session(kernels) as (sp, receipts), temporary_path(".bc") as directory:
        ticks = np.asarray([sp.sce2c(int(clock_id), _et(sp, at(h.epoch, time))) for time in h.times_s])
        if np.any(np.diff(ticks) <= 0):
            raise DataIOError("Spacecraft-clock ticks must increase strictly.")
        # OEL's passive reference-to-body convention is the transpose of NAIF
        # q2m's Hamilton matrix. Convert via the authoritative OEL DCM owner.
        from sim.utils.quaternion import quaternion_to_dcm_bn

        matrices = np.asarray([quaternion_to_dcm_bn(q) for q in h.quaternions])
        quaternions = np.asarray([sp.m2q(matrix) for matrix in matrices])
        path = str(Path(directory) / "attitude.bc")
        handle = sp.ckopn(path, "OEL saved-evidence attitude", 0)
        try:
            # NAIF CK rates are expressed in the segment reference frame;
            # OEL rates are expressed in body coordinates (CKW03 contract).
            rates = (
                np.zeros((len(ticks), 3)) if h.rates_rad_s is None else np.einsum("nji,nj->ni", matrices, h.rates_rad_s)
            )
            sp.ckw03(
                handle,
                ticks[0],
                ticks[-1],
                int(instrument_id),
                _frame(h.reference_frame),
                h.rates_rad_s is not None,
                "OEL bounded attitude",
                len(ticks),
                ticks,
                quaternions,
                rates,
                1,
                ticks[:1],
            )
        except BaseException:
            sp.dafcls(handle)
            raise
        else:
            sp.ckcls(handle)
        sp.furnsh(path)
        try:
            for index, (tick, q) in enumerate(zip(ticks, h.quaternions)):
                matrix, returned_tick = sp.ckgp(int(instrument_id), tick, 0.0, _frame(h.reference_frame))
                if returned_tick != tick or not np.allclose(matrix, quaternion_to_dcm_bn(q), atol=1e-12):
                    raise DataIOError("CK orientation read-back failed.")
                if h.rates_rad_s is not None:
                    _, rate, _ = sp.ckgpav(int(instrument_id), tick, 0.0, _frame(h.reference_frame))
                    if not np.allclose(matrix @ rate, h.rates_rad_s[index], atol=1e-12):
                        raise DataIOError("CK angular-rate frame read-back failed.")
        finally:
            sp.unload(path)
        return read_bytes(path), {
            "kernel_inputs": receipts,
            "ck_type": 3,
            "instrument_id": instrument_id,
            "clock_id": clock_id,
            "angular_rates": h.rates_rad_s is not None,
            "encoded_rate_frame": "segment reference frame",
            "sample_readback": True,
        }


def read_ck(
    path, *, instrument_id: int, clock_id: int, frame: str, epochs, object_id: str, body_frame: str, kernels
) -> AttitudeHistory:
    quaternions, rates = [], []
    with kernel_session([*kernels, path]) as (sp, receipts):
        for epoch in epochs:
            tick = sp.sce2c(int(clock_id), _et(sp, epoch))
            matrix, rate, returned_tick = sp.ckgpav(int(instrument_id), tick, 0.0, _frame(frame))
            if returned_tick != tick:
                raise DataIOError("CK returned a different epoch.")
            q = np.asarray(sp.m2q(matrix))
            q[1:] *= -1
            quaternions.append(q)
            rates.append(matrix @ rate)
    return AttitudeHistory(
        object_id,
        frame,
        body_frame,
        epochs[0],
        [e.tai_seconds - epochs[0].tai_seconds for e in epochs],
        quaternions,
        rates,
        {"kernel_inputs": receipts},
    )


def parse_spice_epoch(value: str, *, scale: str, kernels):
    from sim.frame_time import Epoch, TimeScale

    if scale not in {"TDB", "TCB"}:
        raise DataIOError("Use canonical OEL time conversion for UTC, TAI, TT, UT1, and GPS.")
    if scale == "TCB":
        raise DataIOError("TCB conversion requires a separately qualified relativistic time model.")
    with kernel_session(kernels) as (sp, _):
        et = sp.str2et(value.removesuffix("Z").replace("T", " ") + " TDB")
        seconds = float(sp.unitim(et, "ET", "TAI")) + 946728000.0
    return Epoch(seconds, TimeScale.TAI, value)
