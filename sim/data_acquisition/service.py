"""One orchestration owner for CLI/Python acquisition; never runs a simulation."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from sim.installation.paths import InstallationPaths
from sim.security.sealed_mode import sealed_mode_enabled

from .compatibility import check, consumer_bytes, coverage_contains
from .contracts import AcquisitionError, Request, canonical, digest, epoch, stamp, strict_json
from .credentials import Connections
from .files import headroom, publish, publish_json, read_snapshot
from .inspection import inspect_bytes, provider_epoch, require_compatible
from .providers import ATTRIBUTION, GRAVITY, HOSTS, route, validate_spacetrack_rows
from .store import Store
from .transport import Transport


class AcquisitionService:
    def __init__(self, *, data_root=None, config_root=None, secrets=None, transport_factory=Transport, clock=time.time):
        paths = InstallationPaths.default()
        self.store = Store(Path(data_root) if data_root is not None else paths.data_root / "acquired-data")
        self.connections = Connections(Path(config_root) if config_root is not None else paths.config_root / "data-connections", secrets)
        self.transport_factory = transport_factory
        self.clock = clock

    def _scope(self, request):
        return self.connections.describe(request.connection)["scope"] if request.provider == "spacetrack" else "public-" + request.provider

    def _transport(self, provider, scope, *, max_requests=16):
        attempted = 0

        def reserve(product, interval):
            nonlocal attempted
            if attempted >= max_requests:
                raise AcquisitionError("REQUEST_LIMIT", "Acquisition reached its total network request budget, including login and retries.")
            attempted += 1
            self.store.reserve(scope, product, now=self.clock(), retrieval_interval=interval)

        return self.transport_factory(HOSTS[provider],
            reserve=reserve,
            on_backoff=lambda until: self.store.backoff(scope, until), clock=self.clock)

    def _history(self, request, scope):
        receipts = self.store.receipts(scope) if (self.store.root / "index.sqlite").exists() else []
        histories = [r for r in receipts if r.get("provider") == "spacetrack" and r.get("request", {}).get("product") == "gp_history" and r.get("complete_response")]
        def contains_publications(receipt):
            cutoff = receipt["request"].get("published_before")
            return cutoff is None or (request.published_before is not None and epoch(cutoff) >= epoch(request.published_before))
        matches = [r for r in histories if contains_publications(r)]
        groups, reused = {}, set()
        for oid in request.object_ids:
            intervals = []
            for receipt in matches:
                source = receipt["request"]
                if oid in source["object_ids"]:
                    lo, hi = max(epoch(request.start), epoch(source["start"])), min(epoch(request.stop), epoch(source["stop"]))
                    if lo < hi:
                        intervals.append((lo, hi, receipt))
            cursor, end = epoch(request.start), epoch(request.stop)
            for lo, hi, receipt in sorted(intervals, key=lambda item: (item[0], -item[1].timestamp(), item[2]["request"].get("published_before") != request.published_before)):
                if hi <= cursor:
                    continue
                if lo > cursor:
                    groups.setdefault((stamp(cursor), stamp(lo)), []).append(oid)
                cursor = max(cursor, hi)
                reused.add(receipt["artifact_id"])
            if cursor < end:
                groups.setdefault((stamp(cursor), stamp(end)), []).append(oid)
        missing = [replace(request, start=lo, stop=hi, object_ids=tuple(ids)) for (lo, hi), ids in sorted(groups.items())]
        for part in missing:
            for prior in histories:
                old = prior["request"]
                if set(part.object_ids) & set(old["object_ids"]) and epoch(part.start) < epoch(old["stop"]) and epoch(old["start"]) < epoch(part.stop):
                    raise AcquisitionError("HISTORY_PUBLICATION_CONFLICT", "Overlapping history is already retained under a narrower publication cutoff; use compatible retained data or an explicit provider archive rather than downloading the history again.")
        if len(missing) > request.max_requests:
            raise AcquisitionError("BOUNDS_EXCEEDED", "Missing history exceeds max_requests; split the request or register provider archive files.")
        return missing, sorted(reused)

    def _history_views(self, ids, request, scope):
        result = []
        for artifact in ids:
            value, raw = self.store.get(artifact)
            source = value["request"]
            if source.get("published_before") == request.published_before:
                result.append(artifact)
                continue
            selected = []
            for row in strict_json(raw):
                if not row.get("CREATION_DATE"):
                    raise AcquisitionError("PUBLICATION_MISSING", "Retained history lacks publication timestamps needed for this cutoff.")
                if int(row["NORAD_CAT_ID"]) in request.object_ids and epoch(request.start) <= provider_epoch(row["EPOCH"]) < epoch(request.stop) and provider_epoch(row["CREATION_DATE"]) < epoch(request.published_before):
                    selected.append(row)
            fragment = replace(request, start=max(request.start, source["start"], key=epoch), stop=min(request.stop, source["stop"], key=epoch),
                               object_ids=tuple(sorted(set(request.object_ids) & set(source["object_ids"]))))
            payload = canonical(selected)
            metadata = {"provider": "spacetrack", "product": "gp_history", "kind": "catalog", "format": "gp-json",
                        "request": fragment.to_dict(), "inspection": inspect_bytes(payload, kind="catalog", format="gp-json", provider="spacetrack"),
                        "parents": [{"artifact_id": artifact, "sha256": value["sha256"]}], "source_url": value.get("source_url"),
                        "complete_response": True, "derived_query_view": True, "terms": value["terms"],
                        "integrity": "derived-from-hashed-retained-source", "source_retrieved_at": value["retrieved_at"]}
            metadata["source_receipts"] = dict(value.get("source_receipts", {}))
            metadata["source_receipts"][artifact] = {k: v for k, v in value.items() if k != "source_receipts"}
            view = self.store.put(payload, metadata, scope=scope, request_key=fragment.key(scope))
            self.store.pin(artifact, view["artifact_id"])
            result.append(view["artifact_id"])
        return result

    def _cached(self, request, scope):
        if not (self.store.root / "index.sqlite").exists():
            return []
        receipts = self.store.lookup(request.key(scope))
        if not receipts and request.provider in {"gravity", "celestrak"}:
            receipts = [r for r in reversed(self.store.receipts(scope)) if r.get("product") == request.product]
        if request.provider == "celestrak":
            # Environment feeds are shared snapshots, so a request with a
            # different interval can fall through to the product-level cache
            # above.  Do not treat a recent snapshot as a hit unless its
            # parsed, compatible coverage contains the requested interval.
            covered = []
            for receipt in receipts:
                inspection = receipt.get("inspection")
                if not isinstance(inspection, dict):
                    value, raw = self.store.get(receipt["artifact_id"])
                    inspection = inspect_bytes(
                        raw,
                        kind=value["kind"],
                        format=value["format"],
                        provider=value["provider"],
                    )
                if not inspection.get("parsed") or not inspection.get("compatible"):
                    continue
                if request.start is not None and not coverage_contains(
                    str(receipt.get("kind", "")),
                    inspection.get("coverage"),
                    request.start,
                    request.stop,
                ):
                    continue
                covered.append(receipt)
            receipts = covered
        lifetime = 3600 if request.product == "gp" else 86400 if request.product == "satcat" else 86400
        if request.provider == "gravity":
            return receipts[:1]
        return [r for r in receipts[:1] if self.clock() - epoch(r["retrieved_at"]).timestamp() < lifetime]

    def plan(self, request):
        request = Request.from_dict(request) if isinstance(request, dict) else request.validated()
        url, kind, format = route(request)
        scope = self._scope(request)
        if request.product == "gp_history":
            missing, reused = self._history(request, scope)
        else:
            cached = self._cached(request, scope)
            missing, reused = ([] if cached else [request]), [r["artifact_id"] for r in cached]
        minimum_calls = len(missing) + (1 if missing and request.provider == "spacetrack" else 0)
        if minimum_calls > request.max_requests:
            raise AcquisitionError("REQUEST_LIMIT", "The missing queries and login exceed max_requests.")
        return {"schema": "oel.data-acquisition-plan.v1", "request": request.to_dict(), "kind": kind, "format": format,
                "source_url": url, "cached_artifacts": reused, "missing_requests": [r.to_dict() for r in missing],
                "maximum_network_requests": min(request.max_requests, len(missing) * 2 + (1 if missing and request.provider == "spacetrack" else 0)),
                "maximum_download_bytes": request.max_bytes, "expected_bytes": GRAVITY[request.product]["bytes"] if request.provider == "gravity" else None,
                "network_used": False, "execution_authorized": False, "amount": 0,
                "network_permitted": not sealed_mode_enabled()}

    def _login(self, transport, connection):
        value = self.connections.describe(connection)
        credentials = self.connections.secrets.get(value["secret_ref"])
        response = transport.request("https://www.space-track.org/ajaxauth/login", fields=credentials, max_bytes=64 * 1024, product="login")
        try:
            status = strict_json(response.body)
        except AcquisitionError:
            raise AcquisitionError("AUTHENTICATION_FAILED", "Space-Track returned an unexpected login response.") from None
        failed = isinstance(status, dict) and (str(status.get("Login", "")).casefold() == "failed" or bool(status.get("error")))
        # Do not depend on an undocumented cookie name. An explicit failed login
        # is rejected even if the server issued an anonymous session cookie.
        session = any(cookie.domain.lstrip(".") in {"www.space-track.org", "space-track.org"} and not cookie.is_expired() for cookie in transport.jar)
        if failed or not session:
            raise AcquisitionError("AUTHENTICATION_FAILED", "Space-Track did not establish an authenticated session.")
        return response

    def test_connection(self, name, *, offline=False):
        if offline or sealed_mode_enabled():
            raise AcquisitionError("OFFLINE", "Connection testing requires an explicit online operation outside sealed mode.")
        value = self.connections.describe(name)
        with self.store.operation(value["scope"]):
            transport = self._transport("spacetrack", value["scope"])
            try:
                self._login(transport, name)
                return {"name": name, "authenticated": True, "network_used": True, "product_access": "not_tested"}
            finally:
                transport.clear()

    def fetch(self, request, *, offline=False):
        request = Request.from_dict(request) if isinstance(request, dict) else request.validated()
        scope = self._scope(request)
        with self.store.operation(scope):
            plan = self.plan(request)
            ids = list(plan["cached_artifacts"])
            if not plan["missing_requests"]:
                if request.product == "gp_history":
                    ids = self._history_views(ids, request, scope)
                self._result_bounds(ids, request)
                return {"artifacts": ids, "network_used": False, "request": request.to_dict(), "complete": True}
            if offline or sealed_mode_enabled():
                raise AcquisitionError("OFFLINE", "Required data is not retained; network acquisition is disabled.")
            headroom(self.store.root, request.max_bytes * 2)
            transport = self._transport(request.provider, scope, max_requests=request.max_requests)
            try:
                if request.provider == "spacetrack":
                    self._login(transport, request.connection)
                downloaded = 0
                for raw_request in plan["missing_requests"]:
                    part = Request.from_dict(raw_request)
                    url, kind, format = route(part)
                    interval = 3600 if part.product == "gp" else 86400 if part.product == "satcat" or part.provider == "celestrak" else 0
                    response = transport.request(url, max_bytes=request.max_bytes - downloaded, product=part.product, retrieval_interval=interval)
                    downloaded += len(response.body)
                    if not response.body or response.content_type == "text/html" or response.body.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
                        raise AcquisitionError("INVALID_RESPONSE", "Provider returned an empty document or HTML instead of the requested data.")
                    if part.provider == "spacetrack":
                        validate_spacetrack_rows(strict_json(response.body), part)
                    if part.provider == "gravity":
                        spec = GRAVITY[part.product]
                        if len(response.body) != spec["bytes"] or digest(response.body) != spec["sha256"]:
                            raise AcquisitionError("INTEGRITY_ERROR", "Gravity source differs from its curated size/SHA-256; no artifact was accepted.")
                    inspection = inspect_bytes(response.body, kind=kind, format=format, provider=part.provider)
                    if not inspection["parsed"]:
                        raise AcquisitionError("INVALID_DATA", "Provider response failed its declared format inspection.")
                    if (
                        part.provider == "celestrak"
                        and part.start is not None
                        and not coverage_contains(kind, inspection.get("coverage"), part.start, part.stop)
                    ):
                        raise AcquisitionError(
                            "COVERAGE_MISSING",
                            "Provider response does not cover every requested sample, including stop.",
                        )
                    metadata = {"provider": part.provider, "product": part.product, "kind": kind, "format": format,
                                "request": part.to_dict(), "inspection": inspection, "source_url": response.url,
                                "etag": response.etag, "last_modified": response.last_modified,
                                "terms": ATTRIBUTION[part.provider], "complete_response": True,
                                "integrity": "curated-sha256" if part.provider == "gravity" else "locally-hashed-https-response"}
                    value = self.store.put(response.body, metadata, scope=scope, request_key=part.key(scope))
                    ids.append(value["artifact_id"])
                if request.product == "gp_history":
                    ids = self._history_views(ids, request, scope)
                self._result_bounds(ids, request)
                return {"artifacts": ids, "network_used": True, "request": request.to_dict(), "complete": True}
            finally:
                transport.clear()

    def _result_bounds(self, ids, request):
        values = [self.store.get(i)[0] for i in ids]
        if sum(v["bytes"] for v in values) > request.max_bytes:
            raise AcquisitionError("SIZE_LIMIT", "Retained source artifacts exceed this request's aggregate byte budget; raise max_bytes or narrow the selection.")
        if request.provider == "spacetrack":
            records = set()
            for item in ids:
                _, raw = self.store.get(item)
                for row in strict_json(raw):
                    if int(row["NORAD_CAT_ID"]) not in request.object_ids:
                        continue
                    if request.product == "gp_history" and not epoch(request.start) <= provider_epoch(row["EPOCH"]) < epoch(request.stop):
                        continue
                    records.add(canonical(row))
            if len(records) > request.max_records:
                raise AcquisitionError("RECORD_LIMIT", "Combined retained records exceed max_records.")

    def register(self, path, *, kind, format, terms=None, max_bytes=64 * 1024 * 1024):
        from .contracts import MAX_BYTES
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
            raise AcquisitionError("INVALID_REQUEST", "max_bytes is outside the supported range.")
        if not isinstance(terms, (dict, type(None))) or (terms and (set(terms) - {"source", "license", "redistribution", "attribution"} or any(not isinstance(v, str) or len(v) > 4096 for v in terms.values()))):
            raise AcquisitionError("INVALID_REQUEST", "Terms must use documented bounded text fields.")
        path = Path(path).expanduser().absolute()
        headroom(self.store.root, min(path.stat().st_size, max_bytes) * 2)
        raw = read_snapshot(path, max_bytes)
        inspection = inspect_bytes(raw, kind=kind, format=format)
        metadata = {"provider": "local", "kind": kind, "format": format, "source_name": path.name,
                    "inspection": inspection, "terms": terms or {"redistribution": "unknown"},
                    "integrity": "locally-hashed-snapshot", "request": None}
        return self.store.put(raw, metadata, scope="local")

    def inspect(self, artifact):
        value, raw = self.store.get(artifact)
        inspection = inspect_bytes(raw, kind=value["kind"], format=value["format"], provider=value["provider"])
        return dict(value, inspection=inspection)

    def check(self, artifact, requirements):
        value, raw = self.store.get(artifact)
        return dict(check(raw, value, requirements), artifact_id=artifact, source_sha256=value["sha256"])

    def materialize(self, artifact, destination, *, requirements=None):
        value, raw = self.store.get(artifact)
        if requirements is None:
            output, format = raw, value["format"]
        else:
            output, format = consumer_bytes(raw, value, requirements)
        destination = Path(destination).expanduser().absolute()
        adjacent = destination.with_name(destination.name + ".acquisition.json")
        if destination.exists() or adjacent.exists():
            raise AcquisitionError("DESTINATION_EXISTS", "Select a new destination for the input and receipt.")
        result = {"schema": "oel.acquisition-materialization.v1", "artifact_id": artifact,
                  "source_sha256": value["sha256"], "sha256": digest(output), "format": format,
                  "requirements": requirements, "source_receipt": value, "execution_authorized": False}
        publish(destination, output)
        try:
            publish_json(adjacent, result)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        self.store.pin(artifact, destination)
        return dict(result, path=str(destination), receipt_path=str(adjacent))

    def select(self, artifacts, *, object_ids, at, policy="latest-before", max_age_s=86400, published_before=None):
        import math

        if not isinstance(artifacts, (list, tuple)) or not 1 <= len(artifacts) <= 64:
            raise AcquisitionError("INVALID_SELECTION", "Select 1..64 retained source artifact IDs.")
        if policy not in {"latest-before", "exact"} or type(max_age_s) not in (int, float) or not 0 <= max_age_s <= 180 * 86400:
            raise AcquisitionError("INVALID_SELECTION", "Select exact or latest-before with bounded max_age_s.")
        if not math.isfinite(max_age_s):
            raise AcquisitionError("INVALID_SELECTION", "max_age_s must be finite.")
        if not isinstance(object_ids, (list, tuple)) or not object_ids or len(set(object_ids)) != len(object_ids) or any(type(i) is not int or not 1 <= i <= 999_999_999 for i in object_ids):
            raise AcquisitionError("INVALID_SELECTION", "Select distinct positive integer catalog IDs.")
        when, cutoff = epoch(at), epoch(published_before) if published_before else None
        rows = {oid: [] for oid in object_ids}
        parents = []
        scope = None
        for artifact in artifacts:
            value, raw = self.store.get(artifact)
            if value["kind"] != "catalog" or value["format"] != "gp-json":
                raise AcquisitionError("INVALID_SELECTION", "Record selection requires GP JSON sources.")
            require_compatible(inspect_bytes(raw, kind=value["kind"], format=value["format"], provider=value["provider"]))
            if scope is not None and value["scope"] != scope:
                raise AcquisitionError("INVALID_SELECTION", "Selection cannot combine different connection scopes.")
            scope = value["scope"]
            parents.append({"artifact_id": artifact, "sha256": value["sha256"]})
            for row in strict_json(raw):
                oid = int(row["NORAD_CAT_ID"])
                date = provider_epoch(row["EPOCH"])
                if oid in rows and date <= when and (when - date).total_seconds() <= max_age_s:
                    if cutoff and (not row.get("CREATION_DATE") or provider_epoch(row["CREATION_DATE"]) >= cutoff):
                        continue
                    if policy == "latest-before" or date == when:
                        if row not in rows[oid]:
                            rows[oid].append(row)
        selected = []
        for oid in object_ids:
            candidates = sorted(rows[oid], key=lambda r: provider_epoch(r["EPOCH"]), reverse=True)
            if not candidates:
                raise AcquisitionError("SELECTION_MISSING", "No element set satisfies the object, epoch, age and publication selection.")
            if len(candidates) > 1 and provider_epoch(candidates[0]["EPOCH"]) == provider_epoch(candidates[1]["EPOCH"]):
                raise AcquisitionError("AMBIGUOUS_SELECTION", "Multiple different element sets share the selected epoch; narrow the source artifacts.")
            # Explicit canonical context on the derived product; original provider bytes remain unchanged.
            selected.append({"CENTER_NAME": "EARTH", "REF_FRAME": "TEME", "TIME_SYSTEM": "UTC", "MEAN_ELEMENT_THEORY": "SGP4", **candidates[0]})
        raw = canonical(selected)
        source_terms, source_receipts = [], {}
        for parent in parents:
            source = self.store.get(parent["artifact_id"])[0]
            source_terms.append(source.get("terms", {}))
            source_receipts.update(source.get("source_receipts", {}))
            source_receipts[parent["artifact_id"]] = {k: v for k, v in source.items() if k != "source_receipts"}
        result = self.store.put(raw, {"provider": "local", "kind": "catalog", "format": "gp-json", "parents": parents,
            "source_receipts": source_receipts,
            "selection": {"policy": policy, "at": stamp(when), "max_age_s": max_age_s, "published_before": published_before},
            "inspection": inspect_bytes(raw, kind="catalog", format="gp-json"), "terms": {"redistribution": "inherits-all-source-restrictions", "sources": source_terms}}, scope=scope or "local")
        for parent in parents:
            self.store.pin(parent["artifact_id"], result["artifact_id"])
        return result

    def build_catalog_study(self, artifact, *, scenario_path, scenario_name, output_dir, start, duration_s, step_s):
        from sim.data_io.catalog import read_catalog
        from sim.data_io.materialize import materialize_scenario
        from sim.data_io.time import parse_time

        value, _ = self.store.get(artifact)
        if not value.get("selection") or value["kind"] != "catalog":
            raise AcquisitionError("SELECTION_REQUIRED", "Select exact catalog records before building a study.")
        if epoch(start) != epoch(value["selection"]["at"]):
            raise AcquisitionError("INVALID_SELECTION", "Study start must match the recorded selection time; select again for a different initial epoch.")
        destination = Path(scenario_path).expanduser().absolute()
        if destination.exists():
            raise AcquisitionError("DESTINATION_EXISTS", "Select a new scenario path.")
        source = destination.parent / "inputs" / f"{artifact}.json"
        if source.exists():
            if digest(read_snapshot(source, value["bytes"])) != value["sha256"]:
                raise AcquisitionError("INTEGRITY_ERROR", "Previously materialized input changed.")
        else:
            self.materialize(artifact, source)
        result = materialize_scenario(read_catalog(source, format="gp-json", provider_profile="strict"), destination,
            scenario_name=scenario_name, output_dir=output_dir, duration_s=duration_s, step_s=step_s,
            start_epoch=parse_time(start), source_artifacts=(source,))
        self.store.pin(artifact, destination)
        return {"scenario_path": str(destination), "data_io_receipt": result, "execution_authorized": False}
