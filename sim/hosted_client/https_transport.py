"""Fixed-origin TLS transport with bounded, repeatable byte transfers."""

import base64
import hashlib
import http.client
import json
import math
import ssl
from pathlib import Path
from urllib.parse import urlsplit

from .transport import HostedServiceError

CHUNK_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024


def validate_https_transport(transport):
    url = str(transport["url"])
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}
            or any(c.isspace() for c in url)):
        raise ValueError("Hosted service requires an exact HTTPS origin without credentials or path")
    if not math.isfinite(float(transport["timeout_s"])) or not 0 < float(transport["timeout_s"]) <= 120:
        raise ValueError("HTTPS timeout must be positive and at most 120 seconds")
    if parsed.port is not None and not 0 < parsed.port <= 65535:
        raise ValueError("Invalid HTTPS port")
    ca_file = transport.get("ca_file")
    if ca_file and hashlib.sha256(Path(ca_file).read_bytes()).hexdigest() != transport.get("ca_sha256"):
        raise ValueError("Configured service trust certificate changed")
    return parsed


class HttpsJsonTransport:
    def __init__(self, profile):
        self.options = dict(profile["transport"])
        self.origin = validate_https_transport(self.options)

    def _post(self, path, request, *, account_key=None):
        context = ssl.create_default_context(cafile=self.options.get("ca_file"))
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        connection = http.client.HTTPSConnection(self.origin.hostname, self.origin.port,
            timeout=float(self.options["timeout_s"]), context=context)
        body = json.dumps(request, allow_nan=False).encode()
        if len(body) > MAX_RESPONSE_BYTES:
            raise HostedServiceError("Hosted request exceeds the network size bound")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if account_key is not None:
            if not isinstance(account_key, str) or any(c.isspace() for c in account_key):
                raise ValueError("Invalid account credential")
            headers["Authorization"] = "Bearer " + account_key
        try:
            connection.request("POST", path, body, headers)
            response = connection.getresponse()
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if (response.status != 200 or len(payload) > MAX_RESPONSE_BYTES or
                    response.getheader("Content-Type", "").split(";")[0] != "application/json"):
                raise HostedServiceError("Hosted request rejected or response exceeds its bound")
            document = json.loads(payload)
            if not isinstance(document, dict) or document.get("outcome") != "ok":
                raise HostedServiceError("Hosted service rejected the request")
            return document["result"]
        except (OSError, http.client.HTTPException, ValueError, KeyError):
            raise HostedServiceError("Hosted connection failed; reconcile the retained request identity") from None
        finally:
            connection.close()

    def exchange(self, account_key, scopes):
        return self._post("/v1/session", {"scopes": list(scopes)}, account_key=account_key)

    def _call(self, action, arguments):
        return self._post("/v1/control", {"schema": "oel.hosted_control_request.v1",
            "action": action, "arguments": dict(arguments)})

    def call(self, action, arguments):
        if action == "stage_input":
            from sim.hosted_client.upload_source import read_scenario
            payload = read_scenario(arguments["source_path"])
            digest = hashlib.sha256(payload).hexdigest()
            if digest != arguments["expected_sha256"]:
                raise ValueError("Selected upload changed before transfer")
            base = {"token": arguments["token"]}
            upload = self._call("begin_upload", {**base, "metadata": {
                "input_id": arguments["input_id"], "kind": arguments["kind"],
                "expected_sha256": digest, "bytes": len(payload)}})
            for index, offset in enumerate(range(0, len(payload), CHUNK_BYTES)):
                chunk = payload[offset:offset + CHUNK_BYTES]
                receipt = self._call("upload_chunk", {**base, "upload_id": upload["upload_id"], "index": index,
                    "content_base64": base64.b64encode(chunk).decode()})
                if (receipt.get("upload_id") != upload["upload_id"] or receipt.get("index") != index or
                        receipt.get("content_sha256") != hashlib.sha256(chunk).hexdigest()):
                    raise ValueError("Upload chunk receipt changed")
            return self._call("finish_upload", {**base, "upload_id": upload["upload_id"]})
        if action == "read_result_file":
            payload = bytearray()
            total, file_sha = None, None
            while total is None or len(payload) < total:
                response = self._call("read_result_chunk", {**arguments, "offset": len(payload), "length": CHUNK_BYTES})
                chunk = base64.b64decode(response["content_base64"], validate=True)
                size = response["file_bytes"]
                if (type(size) is not int or not 0 <= size <= MAX_FILE_BYTES or
                        response["artifact_id"] != arguments["artifact_id"] or
                        response["relative_path"] != arguments["relative_path"] or
                        response["offset"] != len(payload) or response["bytes"] != len(chunk) or
                        hashlib.sha256(chunk).hexdigest() != response["content_sha256"] or
                        len(chunk) != min(CHUNK_BYTES, size - len(payload))):
                    raise ValueError("Result chunk failed identity or size verification")
                if total is not None and (size != total or response["file_sha256"] != file_sha):
                    raise ValueError("Result changed during download")
                total, file_sha = size, response["file_sha256"]
                payload.extend(chunk)
            if hashlib.sha256(payload).hexdigest() != file_sha:
                raise ValueError("Result failed complete-content verification")
            return {"artifact_id": arguments["artifact_id"], "relative_path": arguments["relative_path"],
                    "bytes": total, "content_sha256": file_sha, "content_base64": base64.b64encode(payload).decode()}
        return self._call(action, arguments)
