"""Compatibility bridge from the existing gravity API to acquisition transport."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

from .contracts import MAX_BYTES, digest
from .files import publish, read_snapshot
from .providers import ATTRIBUTION, HOSTS
from .transport import Transport


def download_model_file(model, outpath, spec, verify):
    """Keep the historical pinned API, including local fixtures and final naming."""
    if spec is None:
        raise RuntimeError(f"No download URL configured for gravity model '{model}'.")
    last_error = None
    outpath = Path(outpath).parent.resolve() / Path(outpath).name
    for url in spec.urls:
        try:
            parsed = urlsplit(url)
            if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
                # Internal compatibility fixture seam, not an agent URL-fetch surface.
                raw = read_snapshot(Path(unquote(parsed.path)), MAX_BYTES)
            else:
                transport = Transport(HOSTS["gravity"])
                raw = transport.request(url, max_bytes=spec.size_bytes or MAX_BYTES).body
            if len(raw) < spec.min_size_bytes or (spec.size_bytes is not None and len(raw) != spec.size_bytes):
                raise RuntimeError(f"Downloaded gravity model '{model}' has unexpected size.")
            if digest(raw) != spec.sha256.lower():
                raise RuntimeError(f"Gravity model '{model}' integrity check failed.")
            # Verification occurs before publication; preserve callers' existing verifier seam.
            publish(Path(outpath), raw, replace=True)
            verify(model=model, path=Path(outpath))
            if parsed.scheme != "file":
                from sim.installation.paths import InstallationPaths

                from .inspection import inspect_bytes
                from .store import Store

                Store(InstallationPaths.default().data_root / "acquired-data").put(raw, {
                    "provider": "gravity", "product": model.lower(), "kind": "gravity", "format": "icgem",
                    "source_url": url, "terms": ATTRIBUTION["gravity"], "integrity": "curated-sha256",
                    "inspection": inspect_bytes(raw, kind="gravity", format="icgem"), "legacy_gravity_api": True,
                }, scope="public-gravity")
            return
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
    if last_error:
        raise RuntimeError(f"Failed downloading gravity model '{model}' to {outpath}: {last_error}") from last_error
    raise RuntimeError(f"No download URL configured for gravity model '{model}'.")
