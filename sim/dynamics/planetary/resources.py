"""Content-bound local resources. No network or executable providers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(__file__).with_name("data")


@dataclass(frozen=True)
class ResourceReceipt:
    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def read(
        cls, path: str | Path, expected_sha256: str | None = None
    ) -> tuple[ResourceReceipt, bytes]:
        p = Path(path).resolve()
        if p.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f"Planetary declarative resource exceeds 32 MiB: {p}")
        with p.open("rb") as stream:
            content = stream.read(32 * 1024 * 1024 + 1)
        if len(content) > 32 * 1024 * 1024:
            raise ValueError(f"Planetary declarative resource exceeds 32 MiB: {p}")
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Planetary resource SHA256 mismatch: {p}")
        return cls(str(p), digest, len(content)), content

    def verify(self) -> None:
        self.read(self.path, self.sha256)


def read_json(path: str | Path, expected_sha256: str | None = None):
    receipt, data = ResourceReceipt.read(path, expected_sha256)

    def invalid_constant(value):
        raise ValueError(f"Nonfinite planetary JSON value: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate planetary JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(
        data, parse_constant=invalid_constant, object_pairs_hook=unique_pairs
    )
    json.dumps(result, allow_nan=False)
    return receipt, result


def semantic_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
