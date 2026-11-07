"""Curated HTTPS transport, bounded reads and secret-safe errors."""
from __future__ import annotations

import http.cookiejar
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

from .contracts import AcquisitionError


@dataclass(frozen=True)
class Response:
    body: bytes
    url: str
    content_type: str
    etag: str | None = None
    last_modified: str | None = None


class CuratedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts, reserve=None):
        self.hosts = hosts
        self.reserve = reserve

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if req.get_method() != "GET" or parsed.scheme != "https" or parsed.hostname not in self.hosts or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise AcquisitionError("REDIRECT_BLOCKED", "Provider redirected outside its approved HTTPS endpoint.")
        if self.reserve:
            self.reserve("redirect", 0)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Transport:
    def __init__(self, hosts, *, reserve=None, on_backoff=None, clock=time.time):
        self.hosts = frozenset(hosts)
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(CuratedRedirects(self.hosts, reserve), urllib.request.HTTPCookieProcessor(self.jar))
        self.reserve = reserve
        self.on_backoff = on_backoff
        self.clock = clock

    def clear(self):
        self.jar.clear()

    def request(self, url, *, max_bytes, fields=None, product="request", retrieval_interval=0):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in self.hosts or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise AcquisitionError("ENDPOINT_BLOCKED", "Only the provider's curated HTTPS endpoints are permitted.")
        encoded = None if fields is None else urllib.parse.urlencode(fields).encode()
        for attempt in range(2):
            if self.reserve:
                self.reserve(product, retrieval_interval if attempt == 0 else 0)
            request = urllib.request.Request(url, data=encoded, headers={"User-Agent": "OEL-data-acquisition/1", "Accept-Encoding": "identity"})
            try:
                with self.opener.open(request, timeout=30) as response:
                    if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
                        raise AcquisitionError("ENCODING_UNSUPPORTED", "Compressed transfer encoding is outside this bounded transport profile.")
                    length = response.headers.get("Content-Length")
                    if length is not None:
                        try:
                            declared = int(length)
                        except ValueError as exc:
                            raise AcquisitionError("INVALID_RESPONSE", "Invalid provider Content-Length.") from exc
                        if not 0 <= declared <= max_bytes:
                            raise AcquisitionError("SIZE_LIMIT", "Provider response exceeds max_bytes.")
                    chunks, total = [], 0
                    started = time.monotonic()
                    while True:
                        chunk = response.read(min(1024 * 1024, max_bytes - total + 1))
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise AcquisitionError("SIZE_LIMIT", "Provider response exceeds max_bytes.")
                        if time.monotonic() - started > 120:
                            raise AcquisitionError("NETWORK_TIMEOUT", "Provider transfer exceeded its bounded duration.")
                        chunks.append(chunk)
                    if length is not None and total != declared:
                        raise AcquisitionError("INCOMPLETE_RESPONSE", "Provider response was shorter than declared.")
                    return Response(b"".join(chunks), response.geturl(), response.headers.get_content_type(), response.headers.get("ETag"), response.headers.get("Last-Modified"))
            except urllib.error.HTTPError as exc:
                exc.close()
                if exc.code in (401, 403):
                    raise AcquisitionError("AUTHENTICATION_FAILED", "Provider denied access; check the configured account and product permissions.") from None
                if exc.code == 429:
                    wait = _retry_after(exc.headers.get("Retry-After"), self.clock())
                    if self.on_backoff:
                        self.on_backoff(self.clock() + wait)
                    raise AcquisitionError("RATE_LIMIT", "Provider requested a pause before another request.", retry_after_s=wait) from None
                if exc.code in (502, 503, 504) and fields is None and attempt == 0:
                    continue
                raise AcquisitionError("PROVIDER_ERROR", f"Provider returned HTTP {exc.code}.") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                raise AcquisitionError("NETWORK_ERROR", "Provider connection failed or timed out; retained data is unchanged.") from None
        raise AcquisitionError("PROVIDER_ERROR", "Provider request failed.")


def _retry_after(value, now):
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - now
        except (ValueError, TypeError, OverflowError):
            seconds = 3600
    import math
    return max(1, seconds) if math.isfinite(seconds) else 3600
