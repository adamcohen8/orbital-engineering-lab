from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sim.platform_compat import canonicalize_platform_compatibility_root
from sim.security.input_reads import (
    DEFAULT_MAX_READ_BYTES,
    SecureReadRoot,
    require_secure_read_primitives,
    secure_read_primitives_available,
)


class ConfigPathSecurityError(ValueError):
    """Raised when an untrusted config path escapes the allowed filesystem roots."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _sealed_mode_env() -> bool:
    return _truthy_env("OEL_SEALED_MODE")


def _as_resolved_roots(paths: Iterable[str | Path | None]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw in paths:
        if raw in (None, ""):
            continue
        root = Path(raw).expanduser().resolve()
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _default_output_root(workspace: Path, repo: Path) -> Path:
    if _is_relative_to(workspace, repo):
        return repo / "outputs"
    return workspace / "outputs"


@dataclass(frozen=True)
class ConfigPathPolicy:
    """Filesystem trust policy for paths supplied by scenario configs.

    The config file cannot mark itself trusted. Callers opt in by constructing a
    policy with explicit roots or by setting the corresponding CLI/API flags.
    """

    config_path: Path | None = None
    workspace_root: Path = field(default_factory=_repo_root)
    read_roots: tuple[Path, ...] = field(default_factory=tuple)
    write_roots: tuple[Path, ...] = field(default_factory=tuple)
    allow_external_config_paths: bool = False
    allow_external_ai_prompt_files: bool = False
    restrict_ai_prompt_to_read_roots: bool = False
    require_secure_reads: bool = False

    @classmethod
    def default(
        cls,
        *,
        config_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
        read_roots: Iterable[str | Path] = (),
        write_roots: Iterable[str | Path] = (),
        allow_external_config_paths: bool = False,
        allow_external_ai_prompt_files: bool = False,
        allow_config_dir_writes: bool = True,
        require_secure_reads: bool = False,
    ) -> ConfigPathPolicy:
        repo = _repo_root()
        cfg_path = None if config_path is None else Path(config_path).expanduser().resolve()
        cfg_dir = None if cfg_path is None else cfg_path.parent
        workspace = Path(workspace_root).expanduser().resolve() if workspace_root is not None else repo
        default_read_roots = [workspace, repo]
        default_write_roots = [_default_output_root(workspace, repo)]
        if cfg_dir is not None:
            default_read_roots.append(cfg_dir)
            if allow_config_dir_writes and not _is_relative_to(cfg_dir, repo):
                default_write_roots.append(cfg_dir)
        env_allows_external_paths = _truthy_env("OEL_ALLOW_EXTERNAL_CONFIG_PATHS")
        env_allows_external_ai_prompts = _truthy_env("OEL_ALLOW_EXTERNAL_AI_PROMPT_FILES")
        cli_allows_external_paths = _truthy_env("OEL_CLI_ALLOW_EXTERNAL_CONFIG_PATHS")
        cli_allows_external_ai_prompts = _truthy_env("OEL_CLI_ALLOW_EXTERNAL_AI_PROMPT_FILES")
        if _sealed_mode_env():
            if env_allows_external_paths and not (allow_external_config_paths or cli_allows_external_paths):
                raise ConfigPathSecurityError(
                    "OEL_SEALED_MODE blocks OEL_ALLOW_EXTERNAL_CONFIG_PATHS. "
                    "Pass allow_external_config_paths=True only for an explicitly trusted sealed-mode run."
                )
            if env_allows_external_ai_prompts and not (
                allow_external_ai_prompt_files or cli_allows_external_ai_prompts
            ):
                raise ConfigPathSecurityError(
                    "OEL_SEALED_MODE blocks OEL_ALLOW_EXTERNAL_AI_PROMPT_FILES. "
                    "Pass allow_external_ai_prompt_files=True only for an explicitly trusted sealed-mode run."
                )
        return cls(
            config_path=cfg_path,
            workspace_root=workspace,
            read_roots=_as_resolved_roots([*default_read_roots, *read_roots]),
            write_roots=_as_resolved_roots([*default_write_roots, *write_roots]),
            allow_external_config_paths=bool(
                allow_external_config_paths or env_allows_external_paths
            ),
            allow_external_ai_prompt_files=bool(
                allow_external_ai_prompt_files or env_allows_external_ai_prompts
            ),
            require_secure_reads=bool(require_secure_reads),
        )

    @classmethod
    def trusted(
        cls,
        *,
        config_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
        read_roots: Iterable[str | Path] = (),
        write_roots: Iterable[str | Path] = (),
        allow_external_ai_prompt_files: bool = False,
    ) -> ConfigPathPolicy:
        return cls.default(
            config_path=config_path,
            workspace_root=workspace_root,
            read_roots=read_roots,
            write_roots=write_roots,
            allow_external_config_paths=True,
            allow_external_ai_prompt_files=allow_external_ai_prompt_files,
        )

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent if self.config_path is not None else self.workspace_root

    def resolve_input_file(
        self,
        path_text: str | Path,
        *,
        purpose: str,
        base_dir: str | Path | None = None,
        must_exist: bool = True,
    ) -> Path:
        path = self._resolve_path(path_text, base_dir=base_dir or self.config_dir, purpose=purpose)
        if must_exist and not path.is_file():
            raise FileNotFoundError(f"{purpose} file does not exist: {path}")
        self._ensure_allowed(path, roots=self.read_roots, purpose=purpose, access="read")
        return path

    def read_input_bytes(
        self,
        path_text: str | Path,
        *,
        purpose: str,
        base_dir: str | Path | None = None,
        max_bytes: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> tuple[Path, bytes]:
        """Read a config input from the authorized inode, preserving provenance.

        The returned path is a logical source label.  Consumers must parse the
        returned bytes and must not reopen that path after authorization.
        """

        logical = self._resolve_lexical_path(
            path_text,
            base_dir=base_dir or self.config_dir,
            purpose=purpose,
        )
        self._ensure_allowed(logical, roots=self.read_roots, purpose=purpose, access="read")
        roots = self.read_roots
        if self.allow_external_config_paths and not any(_is_relative_to(logical, root) for root in roots):
            roots = (*roots, logical.parent)
        return logical, self._read_bytes_from_authorized_path(
            logical,
            roots=roots,
            purpose=purpose,
            max_bytes=max_bytes,
        )

    def authorized_input_file(
        self,
        path_text: str | Path,
        *,
        purpose: str,
        base_dir: str | Path | None = None,
    ) -> Path:
        """Check one config input through a retained parent descriptor."""

        logical = self._resolve_lexical_path(
            path_text,
            base_dir=base_dir or self.config_dir,
            purpose=purpose,
        )
        self._ensure_allowed(logical, roots=self.read_roots, purpose=purpose, access="read")
        roots = self.read_roots
        if self.allow_external_config_paths and not any(_is_relative_to(logical, root) for root in roots):
            roots = (*roots, logical.parent)
        if secure_read_primitives_available():
            with SecureReadRoot.open(logical.parent, approved_roots=roots) as reader:
                reader.stat(logical.name)
        elif self.require_secure_reads:
            require_secure_read_primitives()
        elif not logical.is_file():
            raise FileNotFoundError(f"{purpose} file does not exist: {logical}")
        return logical

    def read_input_text(
        self,
        path_text: str | Path,
        *,
        purpose: str,
        base_dir: str | Path | None = None,
        encoding: str = "utf-8",
        max_bytes: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> tuple[Path, str]:
        logical, payload = self.read_input_bytes(
            path_text,
            purpose=purpose,
            base_dir=base_dir,
            max_bytes=max_bytes,
        )
        try:
            return logical, payload.decode(encoding)
        except UnicodeDecodeError as exc:
            raise ConfigPathSecurityError(f"{purpose} is not valid {encoding} text: {logical}") from exc

    def read_ai_prompt_text(
        self,
        path_text: str | Path,
        *,
        purpose: str,
        base_dir: str | Path | None = None,
        encoding: str = "utf-8",
        max_bytes: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> tuple[Path, str]:
        """Read an AI prompt through the prompt-specific trust boundary."""

        logical = self._resolve_lexical_path(
            path_text,
            base_dir=base_dir or self.config_dir,
            purpose=purpose,
        )
        if self.allow_external_ai_prompt_files:
            roots = self.read_roots
            if not any(_is_relative_to(logical, root) for root in roots):
                roots = (*roots, logical.parent)
        else:
            roots = self.read_roots if self.restrict_ai_prompt_to_read_roots else _as_resolved_roots([self.config_dir])
            self._ensure_allowed(
                logical,
                roots=roots,
                purpose=purpose,
                access="read",
                hint="Pass allow_external_ai_prompt_files=True or set OEL_ALLOW_EXTERNAL_AI_PROMPT_FILES=1 for trusted prompt files.",
                honor_external_config_paths=False,
            )
        payload = self._read_bytes_from_authorized_path(
            logical,
            roots=roots,
            purpose=purpose,
            max_bytes=max_bytes,
        )
        try:
            return logical, payload.decode(encoding)
        except UnicodeDecodeError as exc:
            raise ConfigPathSecurityError(f"{purpose} is not valid {encoding} text: {logical}") from exc

    def _read_bytes_from_authorized_path(
        self,
        logical: Path,
        *,
        roots: tuple[Path, ...],
        purpose: str,
        max_bytes: int | None,
    ) -> bytes:
        """Read a bounded input through the secure or trusted local route.

        The normal local policy selects descriptor-bound reads whenever the
        host provides them.  A trusted CLI caller can still load ordinary
        local configs on hosts without the POSIX primitives: that compatibility
        route bounds the read and compares the target's metadata before and
        after consumption.  It detects ordinary local replacement, but does
        not claim hostile same-account process protection.  MCP policies set
        ``require_secure_reads`` and therefore fail closed instead of using
        this fallback.
        """

        if secure_read_primitives_available():
            with SecureReadRoot.open(logical.parent, approved_roots=roots) as reader:
                return reader.read_bytes(logical.name, max_bytes=max_bytes)
        if self.require_secure_reads:
            require_secure_read_primitives()
        return self._read_path_compatibility_bytes(
            logical,
            purpose=purpose,
            max_bytes=max_bytes,
        )

    @staticmethod
    def _read_path_compatibility_bytes(
        logical: Path,
        *,
        purpose: str,
        max_bytes: int | None,
    ) -> bytes:
        if max_bytes is not None and int(max_bytes) < 0:
            raise ValueError(f"{purpose} read limit must be non-negative.")
        try:
            before = os.stat(logical)
        except OSError as exc:
            raise FileNotFoundError(f"{purpose} file does not exist: {logical}") from exc
        if not stat.S_ISREG(before.st_mode):
            raise ConfigPathSecurityError(f"{purpose} is not a regular file: {logical}")
        if max_bytes is not None and int(before.st_size) > int(max_bytes):
            raise ValueError(f"{purpose} exceeds the bounded read limit: {logical}")
        try:
            with logical.open("rb") as stream:
                payload = stream.read(None if max_bytes is None else int(max_bytes) + 1)
        except OSError as exc:
            raise ConfigPathSecurityError(f"{purpose} could not be read: {logical}") from exc
        try:
            after = os.stat(logical)
        except OSError as exc:
            raise ConfigPathSecurityError(
                f"{purpose} changed or disappeared during the compatibility read: {logical}"
            ) from exc
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_mode),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
            int(before.st_nlink),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
            int(after.st_nlink),
        )
        if before_identity != after_identity:
            raise ConfigPathSecurityError(
                f"{purpose} changed during the compatibility read: {logical}"
            )
        if max_bytes is not None and len(payload) > int(max_bytes):
            raise ValueError(f"{purpose} exceeds the bounded read limit: {logical}")
        return payload

    def resolve_ai_prompt_file(
        self,
        path_text: str | Path,
        *,
        purpose: str,
        base_dir: str | Path | None = None,
        must_exist: bool = True,
    ) -> Path:
        path = self._resolve_path(path_text, base_dir=base_dir or self.config_dir, purpose=purpose)
        if must_exist and not path.is_file():
            raise FileNotFoundError(f"{purpose} file does not exist: {path}")
        if self.allow_external_ai_prompt_files:
            return path
        roots = self.read_roots if self.restrict_ai_prompt_to_read_roots else _as_resolved_roots([self.config_dir])
        self._ensure_allowed(
            path,
            roots=roots,
            purpose=purpose,
            access="read",
            hint="Pass allow_external_ai_prompt_files=True or set OEL_ALLOW_EXTERNAL_AI_PROMPT_FILES=1 for trusted prompt files.",
            honor_external_config_paths=False,
        )
        return path

    def resolve_output_dir(
        self,
        path_text: str | Path,
        *,
        purpose: str = "outputs.output_dir",
        base_dir: str | Path | None = None,
    ) -> Path:
        path = self._resolve_path(path_text, base_dir=base_dir or self.workspace_root, purpose=purpose)
        self._ensure_allowed(path, roots=self.write_roots, purpose=purpose, access="write")
        return path

    def resolve_output_file(
        self,
        path_text: str | Path,
        *,
        purpose: str,
        base_dir: str | Path | None = None,
    ) -> Path:
        path = self._resolve_path(path_text, base_dir=base_dir or self.workspace_root, purpose=purpose)
        self._ensure_allowed(path, roots=self.write_roots, purpose=purpose, access="write")
        return path

    def _resolve_path(self, path_text: str | Path, *, base_dir: str | Path, purpose: str) -> Path:
        raw = str(path_text or "").strip()
        if not raw:
            raise ValueError(f"{purpose} path must be non-empty.")
        if raw.startswith("~") and not self.allow_external_config_paths:
            raise ConfigPathSecurityError(
                f"{purpose} uses '~', which is blocked for untrusted configs: {raw}. "
                "Pass allow_external_config_paths=True or set OEL_ALLOW_EXTERNAL_CONFIG_PATHS=1 for trusted configs."
            )
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        return path.resolve()

    def _resolve_lexical_path(self, path_text: str | Path, *, base_dir: str | Path, purpose: str) -> Path:
        raw = str(path_text or "").strip()
        if not raw:
            raise ValueError(f"{purpose} path must be non-empty.")
        if raw.startswith("~") and not self.allow_external_config_paths:
            raise ConfigPathSecurityError(
                f"{purpose} uses '~', which is blocked for untrusted configs: {raw}. "
                "Pass allow_external_config_paths=True or set OEL_ALLOW_EXTERNAL_CONFIG_PATHS=1 for trusted configs."
            )
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        return canonicalize_platform_compatibility_root(Path(os.path.abspath(path)))

    def _ensure_allowed(
        self,
        path: Path,
        *,
        roots: tuple[Path, ...],
        purpose: str,
        access: str,
        hint: str | None = None,
        honor_external_config_paths: bool = True,
    ) -> None:
        if honor_external_config_paths and self.allow_external_config_paths:
            return
        resolved = path.resolve()
        if any(_is_relative_to(resolved, root) for root in roots):
            return
        roots_text = ", ".join(str(root) for root in roots) or "(none)"
        extra = f" {hint}" if hint else " Pass allow_external_config_paths=True or set OEL_ALLOW_EXTERNAL_CONFIG_PATHS=1 for trusted configs."
        raise ConfigPathSecurityError(
            f"{purpose} cannot {access} outside allowed config roots: {resolved}. "
            f"Allowed roots: {roots_text}.{extra}"
        )
