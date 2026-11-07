from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

from integrations.oel_mcp.frontier_projection import frontier_safe_value


def test_authorized_root_projection_preserves_legacy_digest() -> None:
    value = "prefix /authorized/run/artifact.json"
    projected = frontier_safe_value(value, authorized_roots=(Path("/authorized"),))

    assert projected == "oel-local-ref:f6dfe70c685776cb"


def test_unmatched_absolute_local_paths_are_opaque_across_platform_forms() -> None:
    values = (
        "/private/customer/run/artifact.json",
        r"C:\Users\customer\run\artifact.json",
        r"\\server\share\customer\artifact.json",
    )

    projected = [frontier_safe_value(value, authorized_roots=(Path("/authorized"),)) for value in values]

    assert all(value.startswith("oel-local-ref:") for value in projected)
    assert all("customer" not in value for value in projected)


def test_embedded_json_projects_nested_posix_windows_and_unc_paths() -> None:
    value = json.dumps(
        {
            "posix": "/private/customer/posix.json",
            "windows": r"C:\Users\customer\windows.json",
            "unc": r"\\server\share\customer\unc.json",
            "url": "https://example.test/review/1",
            "rate": "1.0e-6 km/s",
        }
    )

    projected = frontier_safe_value(value, authorized_roots=(Path("/authorized"),))
    decoded = json.loads(projected)

    assert all(decoded[key].startswith("oel-local-ref:") for key in ("posix", "windows", "unc"))
    assert all("customer" not in decoded[key] for key in ("posix", "windows", "unc"))
    assert decoded["url"] == "https://example.test/review/1"
    assert decoded["rate"] == "1.0e-6 km/s"


def test_urls_scientific_text_and_path_free_json_remain_unchanged() -> None:
    values = (
        "https://example.test/private/customer.json",
        "file:///private/customer/review.json",
        "1.0e-6 km/s",
        "range_km / elapsed_s",
        '{"url":"https://example.test/review/1","value":1.0e-6}',
    )

    assert [frontier_safe_value(value, authorized_roots=(Path("/authorized"),)) for value in values] == list(values)


def test_pathlike_values_are_projected_before_json_safety() -> None:
    posix = frontier_safe_value(Path("/private/customer/run"), authorized_roots=())
    windows = frontier_safe_value(PureWindowsPath(r"C:\Users\customer\run"), authorized_roots=())
    relative = frontier_safe_value(Path("outputs/run"), authorized_roots=())

    assert posix.startswith("oel-local-ref:")
    assert windows.startswith("oel-local-ref:")
    assert relative == "outputs/run"


def test_nested_containers_keep_existing_projection_shapes_and_bytes() -> None:
    value = {
        "tuple": ("/private/customer/file", b"secret"),
        "ordinary": "orbit / time",
    }

    projected = frontier_safe_value(value, authorized_roots=())

    assert isinstance(projected["tuple"], list)
    assert projected["tuple"][0].startswith("oel-local-ref:")
    assert projected["tuple"][1] == b"secret"
    assert projected["ordinary"] == "orbit / time"


def test_authorized_root_in_json_retains_whole_string_legacy_behavior() -> None:
    value = '{"path":"/authorized/run/artifact.json","kind":"review"}'

    projected = frontier_safe_value(value, authorized_roots=(Path("/authorized"),))

    assert projected == "oel-local-ref:6a05115fe54a4d5c"
