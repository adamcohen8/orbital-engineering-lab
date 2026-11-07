"""Curated provider routes. No arbitrary URL, upload, purchase or execution API."""
from __future__ import annotations

from urllib.parse import quote

from sim.dynamics.orbit.gravity_assets import REAL_MODEL_DOWNLOADS

from .contracts import AcquisitionError, Request, epoch
from .inspection import provider_epoch

GRAVITY = {name.lower(): {"url": spec.urls[0], "sha256": spec.sha256, "bytes": spec.size_bytes}
           for name, spec in REAL_MODEL_DOWNLOADS.items()}

ENVIRONMENT = {
    "eop": ("https://celestrak.org/SpaceData/EOP-All.csv", "eop", "celestrak-eop-csv"),
    "space_weather": ("https://celestrak.org/SpaceData/SW-All.csv", "space-weather", "celestrak-weather-csv"),
}
HOSTS = {
    "spacetrack": {"www.space-track.org"},
    "gravity": {"icgem.gfz.de", "icgem.gfz-potsdam.de"},
    "celestrak": {"celestrak.org"},
}
ATTRIBUTION = {
    "spacetrack": {"source": "Space-Track.org", "terms_url": "https://www.space-track.org/documentation#user-agreement", "redistribution": "restricted-unless-authorized"},
    "gravity": {"source": "ICGEM / original model authors", "terms_url": "https://icgem.gfz.de/tom_longtime", "redistribution": "review-model-specific-terms"},
    "celestrak": {"source": "CelesTrak and cited primary data providers", "terms_url": "https://celestrak.org/spacedata/", "redistribution": "review-source-specific-terms"},
}


def describe_providers():
    return {"schema": "oel.data-providers.v1", "providers": [
        {"id": "spacetrack", "products": ["gp", "gp_history", "satcat"], "authentication": "local-os-credential-store", "read_only": True},
        {"id": "gravity", "products": list(GRAVITY), "authentication": "none", "models": GRAVITY, "static_earth_only": True},
        {"id": "celestrak", "products": list(ENVIRONMENT), "authentication": "none"},
        {"id": "local", "operation": "register", "authentication": "filesystem-permissions"}],
        "mcp_network_tools": False, "purchases": False}


def route(request: Request):
    if request.provider == "gravity":
        spec = GRAVITY[request.product]
        if request.max_bytes < spec["bytes"]:
            raise AcquisitionError("SIZE_LIMIT", f"This complete coefficient file requires max_bytes >= {spec['bytes']}.")
        return spec["url"], "gravity", "icgem"
    if request.provider == "celestrak":
        return ENVIRONMENT[request.product]
    fields = [("class", request.product), ("NORAD_CAT_ID", ",".join(map(str, request.object_ids)))]
    if request.product == "gp_history":
        def provider_time(value):
            return epoch(value).replace(tzinfo=None).isoformat(sep=" ")

        fields.append(("EPOCH", provider_time(request.start) + "--" + provider_time(request.stop)))
        if request.published_before:
            fields.append(("CREATION_DATE", "<" + provider_time(request.published_before)))
    fields.extend([("orderby", "NORAD_CAT_ID asc,EPOCH asc" if request.product != "satcat" else "NORAD_CAT_ID asc"),
                   ("limit", str(request.max_records + 1)), ("format", "json")])
    url = "https://www.space-track.org/basicspacedata/query/" + "/".join(quote(v, safe=",") for pair in fields for v in pair)
    return url, "catalog-metadata" if request.product == "satcat" else "catalog", "satcat-json" if request.product == "satcat" else "gp-json"


def validate_spacetrack_rows(rows, request):
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise AcquisitionError("INVALID_RESPONSE", "Expected a JSON array of provider records.")
    if len(rows) > request.max_records:
        raise AcquisitionError("TRUNCATED_RESPONSE", "Provider record limit reached; narrow this request or use the bulk archive route.")
    seen = set()
    previous = None
    for row in rows:
        try:
            identity = row["NORAD_CAT_ID"]
            if isinstance(identity, bool) or str(identity) != str(int(identity)) or int(identity) not in request.object_ids:
                raise ValueError("object mismatch")
            if request.product != "satcat":
                date = provider_epoch(row["EPOCH"])
                if request.product == "gp_history" and not epoch(request.start) <= date <= epoch(request.stop):
                    raise ValueError("epoch mismatch")
                if request.published_before and provider_epoch(row["CREATION_DATE"]) >= epoch(request.published_before):
                    raise ValueError("publication cutoff")
                key = (int(identity), date, str(row.get("GP_ID", "")))
            else:
                key = (int(identity),)
            if key in seen or (previous is not None and key[:2] < previous[:2]):
                raise ValueError("duplicate/order")
            previous = key
            seen.add(key)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise AcquisitionError("INVALID_RESPONSE", "Provider returned unexpected object, epoch, cutoff, ordering or duplicate records.") from exc
    return rows
