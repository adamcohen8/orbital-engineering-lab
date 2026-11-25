"""Compose same-epoch access windows from existing OEL link and sensor evidence."""

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite

from sim.analysis.healpix import healpix_npix

SCHEMA = "oel.access-chains.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_PATHS = 256


@dataclass(frozen=True)
class Link:
    source: tuple[str, str]
    destination: tuple[str, str]
    rate_bps: float
    rx_terminal: str
    tx_terminal: str

    @classmethod
    def from_scenario(cls, row):
        def endpoint(prefix):
            object_id = row.get(f"{prefix}_object_id")
            if object_id:
                return "spacecraft", str(object_id).strip()
            return "fixed_wgs84_site", str(row[f"{prefix}_ground_station_id"]).strip()

        return cls(
            endpoint("tx"), endpoint("rx"), float(row["data_rate_bps"]),
            str(row["rx_terminal"]["terminal_id"]).strip(),
            str(row["tx_terminal"]["terminal_id"]).strip(),
        )


@dataclass(frozen=True)
class PathSpec:
    path_id: str
    links: tuple[str, ...]


@dataclass(frozen=True)
class SensorGate:
    coverage_analysis_id: str
    cell_index: int | None = None
    latitude_deg: float | None = None
    longitude_deg: float | None = None


@dataclass(frozen=True)
class Chain:
    analysis_id: str
    service_rate_bps: float
    simultaneous_relays: tuple[str, ...]
    paths: tuple[PathSpec, ...]
    sensor_gate: SensorGate | None
    include_timeline: bool


def _fields(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = sorted(set(value) - allowed, key=str)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}.")
    return value


def _id(value, label):
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a path-safe identifier of up to 128 characters.")
    return value


def _number(value, label, minimum, maximum=float("inf")):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    if not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be finite and within [{minimum}, {maximum}].")
    return float(value)


def _ids(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of IDs.")
    ids = tuple(_id(item, label) for item in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate IDs.")
    return ids


def _discover(spec, catalog, relays, rate):
    _fields(spec, {"source_object_id", "destination_ground_station_ids", "link_analysis_ids", "max_hops"}, "discover")
    source = ("spacecraft", _id(spec.get("source_object_id"), "discover.source_object_id"))
    sites = _ids(spec.get("destination_ground_station_ids"), "discover.destination_ground_station_ids")
    hops = spec.get("max_hops")
    if not sites or type(hops) is not int or not 1 <= hops <= 16:
        raise ValueError("Discovery requires at least one destination and max_hops from 1 to 16.")
    edges = _ids(spec.get("link_analysis_ids", sorted(catalog)), "discover.link_analysis_ids")
    outgoing = defaultdict(list)
    for link_id in sorted(edges):
        if link_id not in catalog:
            raise ValueError(f"Discovery references unknown directed-link analysis {link_id!r}.")
        link = catalog[link_id]
        if link.rate_bps >= rate:
            outgoing[link.source].append(link_id)

    paths = []
    # Bound traversal even if a dense graph has few paths to the requested sites.
    visits = 0

    def visit(node, legs, seen):
        nonlocal visits
        visits += 1
        if visits > 100_000:
            raise ValueError("Discovery exceeded its search budget. Narrow the links or max_hops.")
        if node[0] == "fixed_wgs84_site" and node[1] in sites:
            if len(paths) == MAX_PATHS:
                raise ValueError(f"Discovery exceeds {MAX_PATHS} paths. Narrow the links or max_hops.")
            paths.append(PathSpec(f"path_{len(paths) + 1:03d}", legs))
        elif len(legs) < hops and (node == source or node[0] == "spacecraft" and node[1] in relays):
            for link_id in outgoing[node]:
                link = catalog[link_id]
                if link.destination in seen:
                    continue
                visit(link.destination, (*legs, link_id), seen | {link.destination})

    visit(source, (), {source})
    return tuple(paths)


def parse_chain(raw, catalog, coverage):
    """Resolve a scenario or review request against declared link and sensor identities."""
    _fields(raw, {"analysis_id", "service_rate_bps", "simultaneous_relays", "paths",
                  "discover", "sensor_gate", "include_timeline"}, "Access chain")
    analysis_id = _id(raw.get("analysis_id"), "analysis_id")
    rate = _number(raw.get("service_rate_bps"), "service_rate_bps", 0.0)
    if rate == 0.0:
        raise ValueError("service_rate_bps must be positive.")
    relays = _ids(raw.get("simultaneous_relays", []), "simultaneous_relays")
    include_timeline = raw.get("include_timeline", True)
    if not isinstance(include_timeline, bool):
        raise ValueError("include_timeline must be a boolean.")
    if ("paths" in raw) == ("discover" in raw):
        raise ValueError("Declare exactly one of paths or discover.")
    if "discover" in raw:
        paths = _discover(raw["discover"], catalog, relays, rate)
    else:
        items = raw["paths"]
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_PATHS:
            raise ValueError(f"paths must contain between 1 and {MAX_PATHS} entries.")
        paths = []
        for item in items:
            _fields(item, {"path_id", "links"}, "Path")
            path_id = _id(item.get("path_id"), "path_id")
            legs = _ids(item.get("links"), f"{path_id}.links")
            if not 1 <= len(legs) <= 16:
                raise ValueError(f"{path_id} must contain from 1 to 16 links.")
            paths.append(PathSpec(path_id, legs))
        paths = tuple(paths)
    if not paths:
        raise ValueError("Discovery found no compatible paths to the requested destinations.")
    if len({path.path_id for path in paths}) != len(paths) or len({path.links for path in paths}) != len(paths):
        raise ValueError("Path IDs and link sequences must be unique.")
    sources = set()
    for path in paths:
        try:
            legs = [catalog[link_id] for link_id in path.links]
        except KeyError as exc:
            raise ValueError(f"{path.path_id} references unknown directed-link analysis {exc.args[0]!r}.") from exc
        sources.add(legs[0].source)
        if any(leg.rate_bps < rate for leg in legs):
            raise ValueError(f"{path.path_id} contains a leg below service_rate_bps.")
        for left, right in zip(legs, legs[1:]):
            if left.destination != right.source:
                raise ValueError(f"{path.path_id} contains disconnected or reversed links.")
            if right.source[0] != "spacecraft" or right.source[1] not in relays:
                raise ValueError(f"{path.path_id} requires a simultaneous relay declaration for {right.source[1]!r}.")
        nodes = [legs[0].source, *(leg.destination for leg in legs)]
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"{path.path_id} revisits an endpoint.")
    if len(sources) != 1:
        raise ValueError("All alternate paths must originate at the same endpoint.")

    gate = None
    if "sensor_gate" in raw:
        item = _fields(raw["sensor_gate"], {"coverage_analysis_id", "cell_index", "latitude_deg", "longitude_deg"},
                       "sensor_gate")
        coverage_id = _id(item.get("coverage_analysis_id"), "coverage_analysis_id")
        if coverage_id not in coverage:
            raise ValueError(f"Unknown sensor coverage analysis {coverage_id!r}.")
        source, order = coverage[coverage_id]
        if sources != {("spacecraft", source)}:
            raise ValueError("The sensor gate must belong to the spacecraft at the start of every path.")
        if "cell_index" in item:
            cell = item["cell_index"]
            if {"latitude_deg", "longitude_deg"} & item.keys() or type(cell) is not int or not 0 <= cell < healpix_npix(order):
                raise ValueError("sensor_gate must declare one valid cell_index or an exact latitude and longitude.")
            gate = SensorGate(coverage_id, cell_index=cell)
        else:
            gate = SensorGate(
                coverage_id,
                latitude_deg=_number(item.get("latitude_deg"), "latitude_deg", -90.0, 90.0),
                longitude_deg=_number(item.get("longitude_deg"), "longitude_deg", -180.0, 180.0),
            )
    return Chain(analysis_id, rate, relays, paths, gate, include_timeline)


def _boundary(row, kind, evidence_id):
    if kind == "start":
        disposition, reason = "acquisition_disposition", "acquisition_reason"
    else:
        disposition, reason = "loss_disposition", "loss_reason"
    return {
        "analysis_id": evidence_id,
        "interval_index": row["interval_index"],
        "censored": bool(row[f"{kind}_censored"]),
        "disposition": row[disposition],
        "reason": row.get(reason),
    }


def _intersect(windows):
    cursors = [0] * len(windows)
    while all(index < len(rows) for index, (_, rows) in zip(cursors, windows, strict=True)):
        active = [(key, rows[index]) for index, (key, rows) in zip(cursors, windows, strict=True)]
        start = max(row["start_s"] for _, row in active)
        end = min(row["end_s"] for _, row in active)
        if start < end:
            yield {
                "start_s": start, "end_s": end,
                "start_evidence": [_boundary(row, "start", key) for key, row in active if row["start_s"] == start],
                "end_evidence": [_boundary(row, "end", key) for key, row in active if row["end_s"] == end],
            }
        for index, (_, row) in enumerate(active):
            if row["end_s"] == end:
                cursors[index] += 1


def _segments(paths, start, end):
    opening, closing = defaultdict(list), defaultdict(list)
    for path in paths:
        for row in path["windows"]:
            opening[row["start_s"]].append(path["path_id"])
            closing[row["end_s"]].append(path["path_id"])
    boundaries = sorted({start, end, *opening, *closing})
    active = set()
    for left, right in zip(boundaries, boundaries[1:]):
        active.difference_update(closing[left])
        active.update(opening[left])
        yield {"start_s": left, "end_s": right, "path_ids": sorted(active)}


def _validate_windows(rows, start, end):
    previous = start
    for row in rows:
        if not previous <= row["start_s"] <= row["end_s"] <= end:
            raise ValueError("Input windows must be ordered, disjoint, and inside the link horizon.")
        previous = row["end_s"]


def evaluate_chain(chain, catalog, link_rows, *, sensor=None, exclude_ground_stations=()):
    """Evaluate bounded same-run evidence, preserving both interval and sample semantics."""
    excluded = set(exclude_ground_stations)
    link_ids = sorted({link_id for path in chain.paths for link_id in path.links})
    sites = {
        endpoint[1] for link_id in link_ids
        for endpoint in (catalog[link_id].source, catalog[link_id].destination)
        if endpoint[0] == "fixed_wgs84_site"
    }
    if excluded - sites:
        raise ValueError(f"Unknown ground stations in exclusion: {sorted(excluded - sites)}.")
    if bool(chain.sensor_gate) != (sensor is not None):
        raise ValueError("Sensor evidence must match the configured sensor gate.")
    times = [row["time_s"] for row in link_rows[link_ids[0]]["samples"]]
    if not times or any(not isfinite(time) for time in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Access chains require finite, strictly increasing link sample epochs.")
    start, end = times[0], times[-1]
    for key in link_ids:
        rows = link_rows[key]
        if [row["time_s"] for row in rows["samples"]] != times:
            raise ValueError("Access chains require the same retained epochs for every link.")
        _validate_windows(rows["windows"], start, end)
    if sensor is not None:
        if sensor["times_s"] != times or len(sensor["available"]) != len(times):
            raise ValueError("Sensor evidence must use the same retained epochs as the links.")
        _validate_windows(sensor["windows"], start, end)

    paths = []
    baseline_paths = []
    ranking_by_index = {}
    for path in sorted(chain.paths, key=lambda item: item.path_id):
        used = {endpoint[1] for key in path.links for endpoint in (catalog[key].source, catalog[key].destination)
                if endpoint[0] == "fixed_wgs84_site"}
        is_excluded = bool(used & excluded)
        windows = [(key, link_rows[key]["windows"]) for key in path.links]
        if sensor is not None:
            windows.append((f"sensor:{chain.sensor_gate.coverage_analysis_id}", sensor["windows"]))
        feasible_windows = list(_intersect(windows))
        baseline_paths.append({"path_id": path.path_id, "windows": feasible_windows})
        paths.append({
            **asdict(path), "excluded_ground_stations": sorted(used & excluded),
            "endpoint_sequence": [catalog[path.links[0]].source, *(catalog[key].destination for key in path.links)],
            "windows": [] if is_excluded else feasible_windows,
        })
        if is_excluded:
            continue
        for index, samples in enumerate(zip(*(link_rows[key]["samples"] for key in path.links), strict=True)):
            if not all(row["available"] for row in samples) or sensor is not None and not sensor["available"][index]:
                continue
            margins = [row["margin_db"] for row in samples]
            if any(value is None or not isfinite(value) for value in margins):
                raise ValueError("Available link samples must contain a finite RF margin.")
            weakest = min(margins)
            winner = ranking_by_index.get(index)
            if winner is None or weakest > winner["weakest_leg_margin_db"]:
                ranking_by_index[index] = {
                    "time_s": times[index], "path_id": path.path_id, "weakest_leg_margin_db": weakest,
                }
    segments = list(_segments(paths, start, end))
    service_windows = []
    for row in segments:
        if not row["path_ids"]:
            continue
        if service_windows and service_windows[-1]["end_s"] == row["start_s"]:
            service_windows[-1]["end_s"] = row["end_s"]
        else:
            service_windows.append({key: row[key] for key in ("start_s", "end_s")})
    duration = sum(row["end_s"] - row["start_s"] for row in service_windows)
    disabled_paths = {path["path_id"] for path in paths if path["excluded_ground_stations"]}
    lost_service = [
        row for row in _segments(baseline_paths, start, end)
        if row["path_ids"] and disabled_paths.issuperset(row["path_ids"])
    ] if disabled_paths else []
    return {
        "schema_version": SCHEMA,
        "analysis_id": chain.analysis_id,
        "service_rate_bps": chain.service_rate_bps,
        "simultaneous_relays": chain.simultaneous_relays,
        "sensor_gate": None if sensor is None else sensor["provenance"],
        "excluded_ground_stations": sorted(excluded),
        "summary": {
            "horizon_start_s": start, "horizon_end_s": end,
            "available_duration_s": duration,
            "redundant_duration_s": sum(row["end_s"] - row["start_s"] for row in segments if len(row["path_ids"]) > 1),
            "lost_service_duration_s": sum(row["end_s"] - row["start_s"] for row in lost_service),
            "available_fraction": duration / (end - start) if end > start else None,
            "path_count": len(paths),
        },
        "paths": paths, "segments": segments, "service_windows": service_windows,
        "lost_service_segments": lost_service,
        "ranked_samples": [ranking_by_index[index] for index in sorted(ranking_by_index)],
        "link_evidence": {
            key: {
                "input_evidence_sha256": link_rows[key]["input_evidence_sha256"],
                "semantic_sha256": link_rows[key]["semantic_sha256"],
                "tx_terminal_id": catalog[key].tx_terminal,
                "rx_terminal_id": catalog[key].rx_terminal,
                "data_rate_bps": catalog[key].rate_bps,
            }
            for key in link_ids
        },
    }
