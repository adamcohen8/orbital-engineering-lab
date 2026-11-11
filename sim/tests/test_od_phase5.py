from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.initial_orbit import (
    IODHypothesis,
    _angle_partition_metrics,
    _propagate_state,
    evaluate_position_triplet_iod,
    run_angles_only_iod,
    solve_gauss_angles_only_iod,
    solve_gibbs_iod,
    solve_herrick_gibbs_iod,
)
from sim.optical_tracking import adapt_optical_ra_dec_observations, los_to_ra_dec

_JD0 = 2460000.5
_TRUTH_STATE = np.array([7000.0, 0.0, 0.0, 0.0, 7.1, 2.55], dtype=float)


def _truth_state(time_s: float) -> np.ndarray:
    return _propagate_state(_TRUTH_STATE, time_s, mu_km3_s2=EARTH_MU_KM3_S2)


def _observer(time_s: float) -> np.ndarray:
    latitude = np.deg2rad(20.0)
    longitude = np.deg2rad(-155.0) + 7.2921150e-5 * time_s
    radius = 6378.137
    return radius * np.array(
        [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]
    )


def make_optical_packet(*, cadence_s: float = 120.0, count: int = 8) -> dict:
    rows = []
    for index in range(count):
        time_s = cadence_s * index
        observer = _observer(time_s)
        ra_deg, dec_deg = los_to_ra_dec(_truth_state(time_s)[:3] - observer)
        rows.append(
            {
                "measurement_id": f"optical:{index:03d}",
                "observer_id": "synthetic_station",
                "time_s": time_s,
                "observer_eci_km": observer.tolist(),
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "ra_sigma_arcsec": 2.0,
                "dec_sigma_arcsec": 2.0,
                "bias_metadata": {"status": "none_injected"},
                "quality_flags": {"synthetic": True},
            }
        )
    return adapt_optical_ra_dec_observations(rows, object_id="phase5_target", initial_jd_utc=_JD0)


def test_phase5_position_triplet_methods_recover_two_body_truth() -> None:
    moderate_times = [-300.0, 0.0, 300.0]
    moderate_positions = [_truth_state(time_s)[:3] for time_s in moderate_times]
    gibbs = solve_gibbs_iod(moderate_positions, epoch_time_s=0.0)
    assert np.linalg.norm(np.asarray(gibbs.state_eci_km_s) - _TRUTH_STATE) < 2.0e-4

    short_times = [-20.0, 0.0, 20.0]
    short_positions = [_truth_state(time_s)[:3] for time_s in short_times]
    herrick = solve_herrick_gibbs_iod(short_positions, short_times)
    assert np.linalg.norm(np.asarray(herrick.state_eci_km_s) - _TRUTH_STATE) < 2.0e-4

    candidates = evaluate_position_triplet_iod(moderate_positions, moderate_times)
    assert candidates["status"] == "candidates_generated"
    assert {item["method"] for item in candidates["hypotheses"]} >= {"gibbs", "lambert_position"}


def test_phase5_optical_contract_preserves_required_metadata() -> None:
    packet = make_optical_packet(count=4)
    row = packet["observations"][0]
    assert packet["measurement_contract"] == "oel_optical_ra_dec_v1"
    assert row["observer_metadata"]["source"] == "explicit_observer_eci_km"
    assert np.asarray(row["angular_covariance_deg2"]) == pytest.approx(
        np.diag([(2.0 / 3600.0) ** 2, (2.0 / 3600.0) ** 2])
    )
    assert row["bias_metadata"]["status"] == "none_injected"


def test_phase5_ra_dec_covariance_is_whitened_in_its_declared_coordinate_basis() -> None:
    dec_deg = 80.0
    predicted_ra_deg = 1.0
    direction = np.array(
        [
            np.cos(np.deg2rad(dec_deg)) * np.cos(np.deg2rad(predicted_ra_deg)),
            np.cos(np.deg2rad(dec_deg)) * np.sin(np.deg2rad(predicted_ra_deg)),
            np.sin(np.deg2rad(dec_deg)),
        ]
    )
    hypothesis = IODHypothesis(
        hypothesis_id="high-dec",
        method="test",
        epoch_time_s=0.0,
        state_eci_km_s=tuple(np.hstack((7000.0 * direction, np.zeros(3)))),
        source_observation_ids=("obs",),
        geometry={},
        provenance={},
    )
    metrics = _angle_partition_metrics(
        hypothesis,
        [
            {
                "measurement_id": "obs",
                "time_s": 0.0,
                "observer_eci_km": [0.0, 0.0, 0.0],
                "ra_deg": 0.0,
                "dec_deg": dec_deg,
                "angular_covariance_deg2": np.diag([0.1**2, 0.1**2]),
            }
        ],
    )

    # Statistical normalization uses the 1 deg RA-coordinate residual.  The
    # separately reported sky-plane magnitude uses dRA*cos(dec).
    assert metrics["normalized_rms"] == pytest.approx(np.sqrt((1.0 / 0.1) ** 2 / 2.0))
    assert metrics["angular_rms_arcsec"] == pytest.approx(3600.0 * np.cos(np.deg2rad(dec_deg)))


def test_phase5_gauss_generates_physical_short_tracklet_seed() -> None:
    packet = make_optical_packet(cadence_s=60.0, count=3)
    hypotheses = solve_gauss_angles_only_iod(packet["observations"])
    assert len(hypotheses) >= 1
    seed = np.asarray(hypotheses[0].state_eci_km_s)
    assert np.linalg.norm(seed - _truth_state(60.0)) < 5.0
    assert min(hypotheses[0].geometry["slant_ranges_km"]) > 0.0


def test_phase5_short_arc_returns_insufficient_without_state(tmp_path: Path) -> None:
    report = run_angles_only_iod(
        make_optical_packet(cadence_s=5.0, count=4),
        output_dir=tmp_path,
        fit_observation_count=3,
    )
    assert report["status"] == "insufficiently_constrained"
    assert report["selected_hypothesis"] is None
    assert report["quality_gates"]["false_precision_prevented"] is True


def test_phase5_angles_only_refines_and_writes_review_evidence(tmp_path: Path) -> None:
    report = run_angles_only_iod(
        make_optical_packet(),
        output_dir=tmp_path,
        fit_observation_count=6,
        radius_grid_km=(6800.0, 7000.0, 7200.0, 8000.0),
        max_hypotheses_to_refine=4,
        max_refinement_nfev=32,
    )
    assert report["status"] in {"unique_solution", "ambiguous"}
    assert report["hypothesis_count"] > 0
    assert report["quality_gates"]["withheld_observations_evaluated"] is True
    if report["status"] == "unique_solution":
        selected = report["selected_hypothesis"]
        assert selected["holdout_metrics"]["normalized_rms"] < selected["seed_metrics"]["holdout"]["normalized_rms"]
        assert (tmp_path / "iod_mission_input_packet.json").is_file()
    with sqlite3.connect(tmp_path / "review/run.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM iod_hypotheses").fetchone()[0] == report["hypothesis_count"]
    assert json.loads((tmp_path / "iod_report.json").read_text(encoding="utf-8"))["status"] == report["status"]


def test_optical_adapter_rejects_frame_context_from_another_epoch() -> None:
    from sim.dynamics.orbit.frames import FrameContext

    rows = make_optical_packet(count=3)["observations"]
    with pytest.raises(ValueError, match="frame context epoch"):
        adapt_optical_ra_dec_observations(
            rows, object_id="target", initial_jd_utc=_JD0,
            frame_context=FrameContext(jd_utc_start=_JD0 + 1.0),
        )


def test_optical_packet_validation_preserves_retained_frame_provenance_without_file_access(monkeypatch) -> None:
    from sim.dynamics.orbit.frames import FrameContext
    from sim.optical_tracking import validate_optical_angle_packet

    packet = adapt_optical_ra_dec_observations(
        make_optical_packet(count=3)["observations"], object_id="target", initial_jd_utc=_JD0,
        frame_context=FrameContext(model="iau76_80_eop", jd_utc_start=_JD0,
                                   xp_arcsec=0.1, yp_arcsec=0.2, dut1_s=0.3, source="retained-reference"),
    )
    packet["frame_context"]["eop_path"] = "/missing/retained-eop.csv"
    packet["frame_context"]["eop_table_sha256"] = "a" * 64
    def unexpected_read(*args, **kwargs):
        raise AssertionError("normalized optical validation must not reopen retained EOP files")
    monkeypatch.setattr("sim.dynamics.orbit.frames._load_eop_table", unexpected_read)
    restored = validate_optical_angle_packet(packet)
    assert restored["frame_context"] == packet["frame_context"]
    assert restored["observations"] == packet["observations"]


@pytest.mark.parametrize("field,value", [("frame", "TEME"), ("time_system", "TAI")])
@pytest.mark.parametrize("level", ["packet", "row"])
def test_optical_packet_rejects_incompatible_frame_or_time_system(field, value, level) -> None:
    from sim.optical_tracking import validate_optical_angle_packet

    packet = make_optical_packet(count=3)
    target = packet if level == "packet" else packet["observations"][0]
    target[field] = value
    with pytest.raises(ValueError, match="frame|time_system"):
        validate_optical_angle_packet(packet)
