import json
from pathlib import Path

import pytest

from sim.dynamics.orbit import reference_accuracy as accuracy


@pytest.mark.parametrize(("case", "epochs"), [("vanguard1", 13), ("cbers2", 25)])
def test_archived_cases_score_backends_separately(case, epochs, tmp_path):
    path = Path(accuracy.__file__).with_name("data") / f"{case}_teme_reference.json"
    result = accuracy.compare_backends(path, tmp_path, position_limit_m=1, velocity_limit_mm_s=1)
    for backend, row in result["backends"].items():
        assert row["epochs"] == epochs
        assert row["passed"] == (row["compared_epochs"] == epochs and
                                 row["max_position_error_m"] <= 1 and row["max_velocity_error_mm_s"] <= 1)
        assert (tmp_path / backend / "ogp_reference_errors.csv").is_file()
    assert result["backends"]["rust"]["passed"]
    # Allow the Python implementation to improve beyond its original performance.
    assert result["backends"]["python"]["max_position_error_m"] < (8 if case == "vanguard1" else 1)
    assert json.loads((tmp_path / "backend_accuracy.json").read_text()) == result


def test_failed_backend_cannot_pass_partial_reference(monkeypatch, tmp_path):
    from dataclasses import replace

    original = accuracy.ogp_propagate_teme
    def fail_one(elements, t, *, backend="rust"):
        state = original(elements, t, backend=backend)
        return replace(state, error="test error") if backend == "python" and t == 0 else state
    monkeypatch.setattr(accuracy, "ogp_propagate_teme", fail_one)
    path = Path(accuracy.__file__).with_name("data") / "cbers2_teme_reference.json"
    result = accuracy.compare_backends(path, tmp_path, position_limit_m=1e9, velocity_limit_mm_s=1e9)
    failed = result["backends"]["python"]
    assert failed["compared_epochs"] == 24 and not failed["passed"]
    assert result["backends"]["rust"]["passed"]
