from __future__ import annotations

import io
import urllib.error
from email.message import Message

import pytest

from sim.data_acquisition.contracts import AcquisitionError, Request
from sim.data_acquisition.providers import route
from sim.data_acquisition.transport import CuratedRedirects, Transport, _retry_after
from sim.tests.data_acquisition_helpers import FakeNetwork, gp, history, service


def configured(tmp_path, payloads):
    network = FakeNetwork(payloads)
    svc = service(tmp_path, network)
    svc.connections.add("personal", "synthetic@example.invalid", "fake-password")
    return svc, network


def test_history_is_reused_offline_and_only_missing_intervals_are_queried(tmp_path):
    svc, network = configured(tmp_path, [[gp(day=10), gp(day=11)], [gp(day=12)]])
    first = svc.fetch(history())
    assert len(network.calls) == 2
    assert svc.fetch(history(), offline=True)["network_used"] is False
    assert len(network.calls) == 2
    plan = svc.plan(history(stop="2024-01-13T00:00:00Z"))
    assert plan["missing_requests"][0]["start"] == "2024-01-12T00:00:00Z"
    second = svc.fetch(history(stop="2024-01-13T00:00:00Z"))
    assert first["artifacts"][0] in second["artifacts"] and len(network.calls) == 4


def test_environment_cache_reuse_requires_requested_coverage(tmp_path):
    from sim.tests.data_acquisition_helpers import eop

    network = FakeNetwork([eop()])
    svc = service(tmp_path, network)
    first = {
        "provider": "celestrak",
        "product": "eop",
        "start": "2024-01-10T00:00:00Z",
        "stop": "2024-01-11T00:00:00Z",
    }
    svc.fetch(first)
    later = {**first, "start": "2024-01-20T00:00:00Z", "stop": "2024-01-21T00:00:00Z"}
    plan = svc.plan(later)
    assert plan["cached_artifacts"] == []
    assert plan["missing_requests"] == [plan["request"]]
    with pytest.raises(AcquisitionError) as error:
        svc.fetch(later, offline=True)
    assert error.value.code == "OFFLINE"


def test_environment_cache_reuses_verified_stored_inspection(tmp_path, monkeypatch):
    from sim.data_acquisition import service as service_module
    from sim.tests.data_acquisition_helpers import eop

    svc = service(tmp_path, FakeNetwork([eop()]))
    request = {
        "provider": "celestrak",
        "product": "eop",
        "start": "2024-01-10T00:00:00Z",
        "stop": "2024-01-11T00:00:00Z",
    }
    first = svc.fetch(request)
    monkeypatch.setattr(
        service_module,
        "inspect_bytes",
        lambda *_args, **_kwargs: pytest.fail("cache hit reparsed retained bytes"),
    )
    plan = svc.plan(request)
    assert plan["cached_artifacts"] == first["artifacts"]
    assert plan["missing_requests"] == []


def test_environment_fetch_rejects_snapshot_without_requested_coverage(tmp_path):
    from sim.tests.data_acquisition_helpers import eop

    svc = service(tmp_path, FakeNetwork([eop()]))
    request = {
        "provider": "celestrak",
        "product": "eop",
        "start": "2024-01-20T00:00:00Z",
        "stop": "2024-01-21T00:00:00Z",
    }
    with pytest.raises(AcquisitionError) as error:
        svc.fetch(request)
    assert error.value.code == "COVERAGE_MISSING"
    assert svc.store.receipts() == []


def test_history_queries_batch_objects_and_keep_publication_cutoff(tmp_path):
    svc, network = configured(tmp_path, [[gp(1), gp(2)]])
    value = svc.fetch(history(object_ids=[1, 2], published_before="2024-01-12T00:00:00Z"))
    url = network.calls[-1][0]
    assert "/NORAD_CAT_ID/1,2/" in url and "CREATION_DATE" in url
    assert svc.inspect(value["artifacts"][0])["request"]["published_before"] == "2024-01-12T00:00:00Z"
    with pytest.raises(AcquisitionError) as error:
        svc.plan(history(object_ids=[1, 2]))
    assert error.value.code == "HISTORY_PUBLICATION_CONFLICT"


def test_publication_cutoff_uses_a_local_filtered_view_of_retained_history(tmp_path):
    svc, network = configured(tmp_path, [[gp(day=10), gp(day=11)]])
    original = svc.fetch(history())
    cutoff = history(published_before="2024-01-11T00:00:00Z")
    assert not svc.plan(cutoff)["missing_requests"]
    result = svc.fetch(cutoff, offline=True)
    assert not result["network_used"] and len(network.calls) == 2
    value, raw = svc.store.get(result["artifacts"][0])
    from sim.data_acquisition.contracts import strict_json
    assert [r["GP_ID"] for r in strict_json(raw)] == [10]
    assert value["derived_query_view"] and value["parents"][0]["artifact_id"] == original["artifacts"][0]
    again = svc.fetch(cutoff, offline=True)
    assert again["artifacts"] == result["artifacts"] and len(network.calls) == 2


def test_cache_is_scoped_to_actual_account_not_alias(tmp_path):
    svc, network = configured(tmp_path, [[gp()]])
    svc.fetch(history())
    svc.connections.add("alias", "synthetic@example.invalid", "fake-password")
    assert not svc.plan(history(connection="alias"))["missing_requests"]
    svc.connections.add("other", "different@example.invalid", "fake-password")
    assert svc.plan(history(connection="other"))["missing_requests"]


def test_empty_complete_history_is_retained_without_claiming_orbit_coverage(tmp_path):
    svc, network = configured(tmp_path, [[]])
    value = svc.fetch(history())
    receipt = svc.inspect(value["artifacts"][0])
    assert receipt["inspection"]["record_count"] == 0
    assert receipt["inspection"]["compatible"] is False
    assert not svc.plan(history())["missing_requests"]


@pytest.mark.parametrize("rows", [
    [gp(2)], [gp(day=9)], [gp(), gp()], {"error": "login failed"},
    [gp(day=11), gp(day=10)], [gp(CREATION_DATE="2024-01-15T00:00:00")],
])
def test_unexpected_provider_rows_are_not_accepted(tmp_path, rows):
    svc, _ = configured(tmp_path, [rows])
    with pytest.raises(AcquisitionError):
        svc.fetch(history(published_before="2024-01-12T00:00:00Z"))
    assert svc.store.receipts() == []


@pytest.mark.parametrize("payload", [b"<html>password expires</html>", b"", b"not json"])
def test_html_empty_and_malformed_responses_do_not_become_datasets(tmp_path, payload):
    svc, _ = configured(tmp_path, [payload])
    with pytest.raises(AcquisitionError):
        svc.fetch(history())
    assert svc.store.receipts() == []


def test_authentication_failure_does_not_retry_or_return_credentials(tmp_path):
    svc, network = configured(tmp_path, [])
    network.authenticates = False
    with pytest.raises(AcquisitionError) as error:
        svc.fetch(history())
    assert len(network.calls) == 1 and "fake-password" not in str(error.value.result())


def test_explicit_failed_login_is_rejected_even_with_a_cookie(tmp_path):
    from types import SimpleNamespace

    from sim.data_acquisition.transport import Response
    svc = service(tmp_path)
    svc.connections.add("personal", "synthetic@example.invalid", "fake-password")
    transport = SimpleNamespace(jar=[SimpleNamespace(domain="www.space-track.org", is_expired=lambda: False)],
        request=lambda *a, **k: Response(b'{"Login":"Failed"}', "https://www.space-track.org/ajaxauth/login", "application/json"))
    with pytest.raises(AcquisitionError) as error:
        svc._login(transport, "personal")
    assert error.value.code == "AUTHENTICATION_FAILED"


def test_max_records_detects_truncation_before_committing(tmp_path):
    svc, _ = configured(tmp_path, [[gp(), gp(day=11)]])
    with pytest.raises(AcquisitionError) as error:
        svc.fetch(history(max_records=1))
    assert error.value.code == "TRUNCATED_RESPONSE"
    assert not svc.store.receipts()


def test_offline_and_sealed_misses_never_authenticate(tmp_path, monkeypatch):
    svc, network = configured(tmp_path, [])
    for sealed in (False, True):
        monkeypatch.setenv("OEL_SEALED_MODE", str(int(sealed)))
        with pytest.raises(AcquisitionError) as error:
            svc.fetch(history(), offline=not sealed)
        assert error.value.code == "OFFLINE"
    assert not network.calls


def test_corruption_does_not_silently_redownload_historical_data(tmp_path):
    svc, network = configured(tmp_path, [[gp()]])
    result = svc.fetch(history())
    value = svc.inspect(result["artifacts"][0])
    (svc.store.root / value["stored_path"]).write_bytes(b"corrupt")
    with pytest.raises(AcquisitionError):
        svc.fetch(history())
    assert len(network.calls) == 2


def test_budgets_include_login_and_cached_aggregate_records(tmp_path):
    svc, network = configured(tmp_path, [[gp(), gp(day=11)]])
    with pytest.raises(AcquisitionError) as error:
        svc.plan(history(max_requests=1))
    assert error.value.code == "REQUEST_LIMIT"
    svc.fetch(history())
    with pytest.raises(AcquisitionError) as error:
        svc.fetch(history(max_records=1))
    assert error.value.code == "RECORD_LIMIT" and len(network.calls) == 2


def test_gp_route_uses_modern_class_and_supports_nine_digit_ids():
    url, _, _ = route(Request.from_dict({"provider": "spacetrack", "product": "gp", "connection": "one", "object_ids": [123456789]}))
    assert "/class/gp/" in url and "123456789" in url and "/format/json" in url


class HTTPResponse(io.BytesIO):
    def __init__(self, data, **headers):
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        for key, value in headers.items():
            self.headers[key.replace("_", "-")] = str(value)

    def geturl(self):
        return "https://celestrak.org/SpaceData/EOP-All.csv"


def test_transport_enforces_streamed_and_declared_size_limits():
    from types import SimpleNamespace
    for headers in ({}, {"Content_Length": 1000}):
        transport = Transport({"celestrak.org"})
        transport.opener = SimpleNamespace(open=lambda *a, headers=headers, **k: HTTPResponse(b"12345", **headers))
        with pytest.raises(AcquisitionError) as error:
            transport.request("https://celestrak.org/x", max_bytes=4)
        assert error.value.code == "SIZE_LIMIT"


def test_transport_rejects_incomplete_and_compressed_responses():
    from types import SimpleNamespace
    for headers, expected in [({"Content_Length": 10}, "INCOMPLETE_RESPONSE"), ({"Content_Encoding": "gzip"}, "ENCODING_UNSUPPORTED")]:
        transport = Transport({"celestrak.org"})
        transport.opener = SimpleNamespace(open=lambda *a, headers=headers, **k: HTTPResponse(b"123", **headers))
        with pytest.raises(AcquisitionError) as error:
            transport.request("https://celestrak.org/x", max_bytes=100)
        assert error.value.code == expected


def test_transport_preserves_retry_after_without_sleeping_or_echoing_response():
    from types import SimpleNamespace
    pauses = []
    transport = Transport({"celestrak.org"}, on_backoff=pauses.append, clock=lambda: 1000)
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://celestrak.org/secret", 429, "sensitive text", {"Retry-After": "120"}, None)
    transport.opener = SimpleNamespace(open=fail)
    with pytest.raises(AcquisitionError) as error:
        transport.request("https://celestrak.org/x", max_bytes=100)
    assert error.value.retry_after_s == 120 and pauses == [1120]
    assert "sensitive" not in str(error.value)
    assert _retry_after("nan", 0) == 3600


@pytest.mark.parametrize("url", ["http://celestrak.org/x", "https://evil.invalid/x", "https://user:secret@celestrak.org/x", "https://celestrak.org:444/x"])
def test_transport_rejects_noncurated_endpoints(url):
    with pytest.raises(AcquisitionError):
        Transport({"celestrak.org"}).request(url, max_bytes=100)


def test_redirects_cannot_forward_login_or_escape_provider():
    import urllib.request
    redirect = CuratedRedirects({"celestrak.org"})
    with pytest.raises(AcquisitionError):
        redirect.redirect_request(urllib.request.Request("https://celestrak.org/x"), None, 302, "", {}, "https://evil.invalid")
    with pytest.raises(AcquisitionError):
        redirect.redirect_request(urllib.request.Request("https://celestrak.org/x", data=b"secret"), None, 302, "", {}, "https://celestrak.org/y")
