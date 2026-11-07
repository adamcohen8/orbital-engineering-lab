"""v0.36 OGP full-wheel/OGP-only selector regressions."""

from types import SimpleNamespace

import pytest

import sim.rust_ogp_backend as backend


def _install_importer(monkeypatch, *, full, ogp_only, calls):
    def importer(name):
        calls.append(name)
        if name == "oel_rust_orbit":
            if isinstance(full, BaseException):
                raise full
            return full
        if name == "oel_rust_ogp":
            if isinstance(ogp_only, BaseException):
                raise ogp_only
            return ogp_only
        raise ImportError(name)

    monkeypatch.setattr(backend, "import_module", importer)


def test_selector_prefers_full_wheel_when_ogp_context_is_available(monkeypatch):
    full = SimpleNamespace(OGPContext=object)
    standalone = object()
    calls = []
    _install_importer(monkeypatch, full=full, ogp_only=standalone, calls=calls)
    monkeypatch.setattr("sim.rust_orbit_backend._extension", lambda: full)

    assert backend._extension() is full
    assert calls == ["oel_rust_orbit"]


def test_selector_falls_back_when_full_loader_reports_compatibility_failure(monkeypatch):
    standalone = SimpleNamespace(OGPContext=object)
    calls = []
    _install_importer(monkeypatch, full=object(), ogp_only=standalone, calls=calls)

    def unavailable():
        raise RuntimeError(
            "Rust orbit backend requested but oel_rust_orbit is unavailable; test compatibility failure"
        )

    monkeypatch.setattr("sim.rust_orbit_backend._extension", unavailable)

    assert backend._extension() is standalone
    assert calls == ["oel_rust_orbit", "oel_rust_ogp"]


def test_selector_falls_back_when_full_wheel_lacks_ogp_context(monkeypatch):
    full = SimpleNamespace(ogp_propagate_batch_bytes=lambda *_args: None)
    standalone = SimpleNamespace(OGPContext=object)
    calls = []
    _install_importer(monkeypatch, full=full, ogp_only=standalone, calls=calls)
    monkeypatch.setattr("sim.rust_orbit_backend._extension", lambda: full)

    assert backend._extension() is standalone
    assert calls == ["oel_rust_orbit", "oel_rust_ogp"]


def test_selector_reports_both_full_and_ogp_only_failures(monkeypatch):
    calls = []
    _install_importer(
        monkeypatch,
        full=object(),
        ogp_only=ImportError("OGP-only wheel missing"),
        calls=calls,
    )

    def unavailable():
        raise RuntimeError("Rust orbit backend requested but oel_rust_orbit is unavailable; full wheel incompatible")

    monkeypatch.setattr("sim.rust_orbit_backend._extension", unavailable)

    with pytest.raises(
        RuntimeError,
        match=(
            "Full-wheel compatibility failure: Rust orbit backend requested.*"
            "OGP-only wheel import failure: OGP-only wheel missing"
        ),
    ):
        backend._extension()
    assert calls == ["oel_rust_orbit", "oel_rust_ogp"]


def test_selector_does_not_mask_noncompatibility_errors(monkeypatch):
    calls = []
    _install_importer(monkeypatch, full=object(), ogp_only=object(), calls=calls)

    def algorithm_failure():
        raise RuntimeError("native propagation algorithm failure")

    monkeypatch.setattr("sim.rust_orbit_backend._extension", algorithm_failure)

    with pytest.raises(RuntimeError, match="native propagation algorithm failure"):
        backend._extension()
    assert calls == ["oel_rust_orbit"]
