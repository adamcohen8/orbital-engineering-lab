"""Platform-native path selection for OEL-managed installations."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstallationPaths:
    data_root: Path
    config_root: Path

    @classmethod
    def default(
        cls,
        *,
        home: str | Path | None = None,
        environ: dict[str, str] | None = None,
        system: str | None = None,
    ) -> InstallationPaths:
        env = dict(os.environ if environ is None else environ)
        home_path = Path(home).expanduser() if home is not None else Path.home()
        override_data = env.get("OEL_DATA_HOME", "").strip()
        override_config = env.get("OEL_CONFIG_HOME", "").strip()
        host = system or platform.system()
        if host == "Windows":
            local_app_data = str(env.get("LOCALAPPDATA", "") or "").strip()
            app_data = str(env.get("APPDATA", "") or "").strip()
            # Explicit OEL_* overrides are complete managed roots, just like
            # the CLI --data-root/--config-root options.  Only platform
            # defaults receive the conventional Windows /OEL suffix.
            data = (
                Path(override_data)
                if override_data
                else Path(local_app_data or home_path / "AppData" / "Local") / "OEL"
            )
            config = (
                Path(override_config)
                if override_config
                else Path(app_data or home_path / "AppData" / "Roaming") / "OEL"
            )
        else:
            xdg_data = Path(str(env.get("XDG_DATA_HOME", "") or "").strip() or home_path / ".local" / "share")
            xdg_config = Path(str(env.get("XDG_CONFIG_HOME", "") or "").strip() or home_path / ".config")
            data = Path(override_data) if override_data else xdg_data / "oel"
            config = Path(override_config) if override_config else xdg_config / "oel"
        return cls(data.expanduser().resolve(), config.expanduser().resolve())

    @property
    def versions(self) -> Path:
        return self.data_root / "versions"

    @property
    def cache(self) -> Path:
        return self.data_root / "cache"

    @property
    def state(self) -> Path:
        return self.data_root / "state"

    @property
    def launcher(self) -> Path:
        return self.data_root / "launcher"

    @property
    def current_state(self) -> Path:
        return self.state / "current.json"

    @property
    def installations_state(self) -> Path:
        return self.state / "installations.json"

    @property
    def workspaces_state(self) -> Path:
        return self.state / "workspaces.json"

    @property
    def transaction_lock(self) -> Path:
        return self.state / "update.lock"

    @property
    def trusted_release_keys(self) -> Path:
        return self.config_root / "trusted-release-keys.json"

    @property
    def channel_config(self) -> Path:
        return self.config_root / "update-channels.json"

    @property
    def channel_state(self) -> Path:
        return self.state / "channels.json"

    def version_root(self, version: str) -> Path:
        return self.versions / version

    def ensure(self) -> None:
        for path in (self.versions, self.cache, self.state, self.launcher, self.config_root):
            path.mkdir(parents=True, exist_ok=True)
