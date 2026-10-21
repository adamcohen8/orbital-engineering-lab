"""Actual NAIF binary writer/reader checks; optional dependency, no network."""

from __future__ import annotations

import numpy as np
import pytest

from sim.data_io.contracts import AttitudeHistory, DataIOError, Trajectory
from sim.data_io.spice import (
    encode_ck,
    encode_spk,
    parse_spice_epoch,
    read_ck,
    read_spk,
    transform_spice,
)
from sim.data_io.time import at, parse_time

sp = pytest.importorskip("spiceypy")


@pytest.fixture
def lsk(tmp_path):
    # A synthetic, explicitly bounded test time kernel. Production callers
    # provide their own dated NAIF/IERS kernel with its recorded hash.
    path = tmp_path / "synthetic-time.tls"
    path.write_text("""KPL/LSK
\\begindata
DELTET/DELTA_T_A = 32.184
DELTET/K = 1.657D-3
DELTET/EB = 1.671D-2
DELTET/M = (6.239996D0 1.99096871D-7)
DELTET/DELTA_AT = (37, @2017-JAN-1)
\\begintext
Synthetic test kernel for post-2017 epochs only.
""")
    return path


@pytest.fixture
def clock(tmp_path):
    path = tmp_path / "synthetic-clock.tsc"
    path.write_text("""KPL/SCLK
\\begindata
SCLK_KERNEL_ID = ( @2024-01-01/00:00:00 )
SCLK_DATA_TYPE_1000 = ( 1 )
SCLK01_TIME_SYSTEM_1000 = ( 1 )
SCLK01_N_FIELDS_1000 = ( 2 )
SCLK01_MODULI_1000 = ( 4294967296 1000 )
SCLK01_OFFSETS_1000 = ( 0 0 )
SCLK01_OUTPUT_DELIM_1000 = ( 1 )
SCLK_PARTITION_START_1000 = ( 0.0 )
SCLK_PARTITION_END_1000 = ( 4.294967295999D12 )
SCLK01_COEFFICIENTS_1000 = ( 0.0 0.0 1.0 )
\\begintext
Synthetic test spacecraft clock; no flight clock claim.
""")
    return path


def test_spk_type13_samples_center_and_coverage_cross_read_via_naif(tmp_path, lsk):
    epoch = parse_time("2024-01-01T00:00:00")
    trajectory = Trajectory(
        "TEST",
        "EARTH",
        "EME2000",
        epoch,
        [0, 30, 60],
        [[7000, 0, 0, 0, 7.5, 0], [7000, 225, 0, 0, 7.5, 0], [7000, 450, 0, 0, 7.5, 0]],
    )
    before = sp.ktotal("ALL")
    raw, receipt = encode_spk(trajectory, target_id=-1000, center_id=399, kernels=[lsk])
    assert sp.ktotal("ALL") == before
    path = tmp_path / "orbit.bsp"
    path.write_bytes(raw)
    result = read_spk(
        path,
        target_id=-1000,
        center_id=399,
        frame="EME2000",
        object_id="TEST",
        epochs=[at(epoch, value) for value in [0, 15, 60]],
        kernels=[lsk],
    )
    np.testing.assert_allclose(result.states[1], [7000, 112.5, 0, 0, 7.5, 0], atol=1e-8)
    assert receipt["sample_readback"] and result.center == "EARTH"
    assert sp.ktotal("ALL") == before
    with pytest.raises(sp.utils.exceptions.SpiceSPKINSUFFDATA):
        read_spk(
            path,
            target_id=-1000,
            center_id=399,
            frame="EME2000",
            object_id="TEST",
            epochs=[at(epoch, 61)],
            kernels=[lsk],
        )
    assert sp.ktotal("ALL") == before


def test_spice_center_translation_and_covariance_rotation_use_declared_kernels(tmp_path, lsk):
    epoch = parse_time("2024-01-01T00:00:00")
    earth = Trajectory("SOURCE", "EARTH", "EME2000", epoch, [0, 60], [[100, 0, 0, 0, 0, 0], [100, 0, 0, 0, 0, 0]])
    raw, _ = encode_spk(earth, target_id=-1000, center_id=399, kernels=[lsk])
    path = tmp_path / "center.bsp"
    path.write_bytes(raw)
    target = Trajectory("OTHER", "EARTH", "EME2000", epoch, [0, 60], [[101, 0, 0, 0, 0, 0], [101, 0, 0, 0, 0, 0]])
    result = transform_spice(target, target_center="-1000", target_frame="EME2000", kernels=[lsk, path])
    np.testing.assert_allclose(result.states[:, 0], 1)


def test_ck_orientation_and_rates_roundtrip_through_naif(tmp_path, lsk, clock):
    epoch = parse_time("2024-01-01T00:00:00")
    angle = np.array([0, 0.1, 0.2])
    history = AttitudeHistory(
        "TEST",
        "EME2000",
        "SC_BODY_1",
        epoch,
        [0, 30, 60],
        np.column_stack((np.cos(angle / 2), np.zeros((3, 2)), np.sin(angle / 2))),
        np.tile([0.001, 0.002, 0.1 / 30], (3, 1)),
    )
    raw, receipt = encode_ck(history, instrument_id=-1000000, clock_id=-1000, kernels=[lsk, clock])
    path = tmp_path / "attitude.bc"
    path.write_bytes(raw)
    # Inspect NAIF's raw rate in its reference coordinates independently of
    # the IO reader. Non-aligned components expose a body/reference mix-up.
    from sim.data_io.spice import kernel_session
    from sim.data_io.time import format_time
    from sim.utils.quaternion import quaternion_to_dcm_bn

    with kernel_session([lsk, clock, path]):
        tick = sp.sce2c(-1000, sp.str2et(format_time(at(epoch, 60))))
        _, reference_rate, _ = sp.ckgpav(-1000000, tick, 0, "J2000")
        np.testing.assert_allclose(
            reference_rate, quaternion_to_dcm_bn(history.quaternions[-1]).T @ history.rates_rad_s[-1], atol=1e-14
        )
    result = read_ck(
        path,
        instrument_id=-1000000,
        clock_id=-1000,
        frame="EME2000",
        object_id="TEST",
        body_frame="SC_BODY_1",
        epochs=[at(epoch, value) for value in history.times_s],
        kernels=[lsk, clock],
    )
    assert receipt["sample_readback"]
    np.testing.assert_allclose(np.abs(np.sum(result.quaternions * history.quaternions, axis=1)), 1, atol=1e-14)
    np.testing.assert_allclose(result.rates_rad_s, history.rates_rad_s, atol=1e-14)


def test_tdb_and_gps_epochs_remain_distinct_time_scales(lsk):
    epoch = parse_spice_epoch("2024-01-01T00:00:00", scale="TDB", kernels=[lsk])
    tt = parse_time("2024-01-01T00:00:00", "TT")
    assert abs(epoch.tai_seconds - tt.tai_seconds) < 0.002
    gps = parse_time("2024-01-01T00:00:00", "GPS")
    utc = parse_time("2024-01-01T00:00:00", "UTC")
    assert utc.tai_seconds - gps.tai_seconds == 18


def test_kernel_sessions_reject_unrecorded_ambient_or_hidden_meta_kernels(tmp_path, lsk, clock):
    from sim.data_io.spice import kernel_session

    sp.furnsh(str(clock))
    try:
        with pytest.raises(DataIOError, match="Unlisted kernels"):
            with kernel_session([lsk]):
                pytest.fail("Unrecorded kernel state must not influence conversion.")
        assert sp.ktotal("ALL") == 1
    finally:
        sp.unload(str(clock))
    hidden = tmp_path / "misleading.bsp"
    hidden.write_text("KPL/MK\n\\begindata\nKERNELS_TO_LOAD = ('unrecorded.bsp')\n")
    with pytest.raises(DataIOError, match="meta-kernel"):
        with kernel_session([hidden]):
            pytest.fail("Meta-kernels are rejected regardless of filename suffix.")
