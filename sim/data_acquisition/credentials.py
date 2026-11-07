"""Local secret references; no credentials in receipts or CLI arguments."""
from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from pathlib import Path

from .contracts import AcquisitionError, alias, canonical, strict_json
from .files import private_dir, publish, publish_json, read_snapshot

SERVICE = "OEL data acquisition"


class SystemSecrets:
    def __init__(self):
        try:
            import keyring

            backend = keyring.get_keyring()
            module = type(backend).__module__
            # Explicitly reject plaintext, fail, null, and arbitrary third-party backends.
            if not module.startswith(("keyring.backends.macOS", "keyring.backends.Windows", "keyring.backends.SecretService")):
                raise RuntimeError("unsupported backend")
            self.backend = backend
        except Exception as exc:
            raise AcquisitionError("CREDENTIAL_BACKEND", "Install OEL's acquisition extra and configure macOS Keychain, Windows Credential Manager, or Linux Secret Service.") from exc

    def get(self, reference):
        try:
            value = self.backend.get_password(SERVICE, reference)
            if value is None:
                raise ValueError("missing")
            result = strict_json(value)
            if set(result) != {"identity", "password"} or any(not isinstance(v, str) or not v for v in result.values()):
                raise ValueError("invalid")
            return result
        except Exception as exc:
            raise AcquisitionError("CREDENTIAL_UNAVAILABLE", "Unlock the credential store or configure this connection again.") from exc

    def set(self, reference, value):
        try:
            self.backend.set_password(SERVICE, reference, canonical(value).decode())
        except Exception as exc:
            raise AcquisitionError("CREDENTIAL_BACKEND", "Unable to store the connection in the OS credential manager.") from exc

    def delete(self, reference):
        try:
            self.backend.delete_password(SERVICE, reference)
        except Exception as exc:
            raise AcquisitionError("CREDENTIAL_BACKEND", "Unable to remove the OS credential; unlock the store and retry.") from exc


class Connections:
    def __init__(self, root: Path, secrets=None):
        self.root = root
        self._secrets = secrets

    @property
    def secrets(self):
        if self._secrets is None:
            self._secrets = SystemSecrets()
        return self._secrets

    def path(self, name):
        return self.root / f"{alias(name)}.json"

    def describe(self, name):
        path = self.path(name)
        if not path.exists():
            raise AcquisitionError("CONNECTION_MISSING", "Configure this local connection first.")
        value = strict_json(read_snapshot(path, 8192))
        if not isinstance(value, dict) or set(value) != {"schema", "name", "provider", "scope", "secret_ref"} or value.get("schema") != "oel.data-connection.v1" or value.get("provider") != "spacetrack" or value.get("name") != name:
            raise AcquisitionError("CONNECTION_INVALID", "Connection metadata is invalid; configure it again.")
        try:
            uuid.UUID(hex=value["scope"])
            uuid.UUID(hex=value["secret_ref"])
        except (ValueError, TypeError, AttributeError) as exc:
            raise AcquisitionError("CONNECTION_INVALID", "Invalid connection references.") from exc
        return value

    def status(self, name):
        value = self.describe(name)
        return {"name": name, "provider": value["provider"], "configured": True,
                "authentication": "not_tested", "network_used": False}

    def add(self, name, identity, password):
        path = self.path(name)
        if path.exists():
            raise AcquisitionError("CONNECTION_EXISTS", "Remove the existing connection before replacing it.")
        if not isinstance(identity, str) or not identity.strip() or not isinstance(password, str) or not password:
            raise AcquisitionError("INVALID_CREDENTIAL", "Identity and password must be nonempty.")
        reference = uuid.uuid4().hex
        private_dir(self.root)
        os.chmod(self.root, 0o700)
        salt_path = self.root / ".scope-salt"
        if not salt_path.exists():
            try:
                publish(salt_path, os.urandom(32))
            except AcquisitionError as exc:
                if exc.code != "DESTINATION_EXISTS":
                    raise
        salt = read_snapshot(salt_path, 32)
        if len(salt) != 32:
            raise AcquisitionError("CONNECTION_INVALID", "Account-scope salt changed; restore the original connection configuration.")
        # Local keyed pseudonym: stable across aliases, but receipts cannot be
        # used to guess the account email from a public unsalted hash.
        scope = hmac.new(salt, identity.strip().casefold().encode(), hashlib.sha256).hexdigest()[:32]
        value = {"schema": "oel.data-connection.v1", "name": name, "provider": "spacetrack",
                 "scope": scope, "secret_ref": reference}
        self.secrets.set(reference, {"identity": identity.strip(), "password": password})
        try:
            publish_json(path, value)
        except Exception:
            self.secrets.delete(reference)
            raise
        return self.status(name)

    def remove(self, name):
        value = self.describe(name)
        self.secrets.delete(value["secret_ref"])
        self.path(name).unlink()
        return {"name": name, "removed": True, "datasets_preserved": True}
