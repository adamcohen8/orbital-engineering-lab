from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

OEL_DISTRIBUTION_NAME = "orbital-engineering-lab"


@dataclass(frozen=True)
class ProjectVersionStatus:
    source_version: str | None
    installed_version: str | None
    ok: bool
    required: bool
    detail: str


def inspect_project_version(
    *,
    source_root: str | Path | None = None,
    distribution_name: str = OEL_DISTRIBUTION_NAME,
) -> ProjectVersionStatus:
    """Compare source-tree and active-environment OEL versions."""
    source_version = source_project_version(source_root=source_root)
    installed_version = installed_project_version(distribution_name=distribution_name)

    if source_version and installed_version:
        if source_version == installed_version:
            return ProjectVersionStatus(
                source_version=source_version,
                installed_version=installed_version,
                ok=True,
                required=True,
                detail=f"{source_version} (source and installed metadata agree)",
            )
        return ProjectVersionStatus(
            source_version=source_version,
            installed_version=installed_version,
            ok=False,
            required=True,
            detail=(
                f"source {source_version}; installed {installed_version}. "
                "Recreate the virtual environment or run `python -m pip install -e .`."
            ),
        )

    if source_version:
        return ProjectVersionStatus(
            source_version=source_version,
            installed_version=None,
            ok=False,
            required=False,
            detail=(
                f"source {source_version}; package metadata is not installed. "
                "Run `python -m pip install -e .` for a managed OEL environment."
            ),
        )

    if installed_version:
        return ProjectVersionStatus(
            source_version=None,
            installed_version=installed_version,
            ok=True,
            required=True,
            detail=f"{installed_version} (installed package)",
        )

    return ProjectVersionStatus(
        source_version=None,
        installed_version=None,
        ok=False,
        required=True,
        detail="OEL source version and installed package metadata were both unavailable.",
    )


def source_project_version(*, source_root: str | Path | None = None) -> str | None:
    root = Path(source_root).expanduser() if source_root is not None else Path(__file__).resolve().parents[1]
    pyproject = root if root.name == "pyproject.toml" else root / "pyproject.toml"
    try:
        lines = pyproject.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    in_project = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version") and "=" in line:
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def installed_project_version(*, distribution_name: str = OEL_DISTRIBUTION_NAME) -> str | None:
    # A source checkout can contain stale legacy ``.egg-info`` alongside the
    # active environment's modern ``.dist-info``. importlib.metadata.version()
    # returns the first match on sys.path, which can make a healthy editable
    # install look stale. Prefer installed dist-info while retaining egg-info
    # as the compatibility fallback when it is the only metadata available.
    try:
        distributions = list(metadata.distributions(name=distribution_name))
    except (metadata.PackageNotFoundError, TypeError, KeyError, AttributeError, ValueError):
        distributions = []
    if distributions:
        # ``importlib.metadata`` yields distributions in discovery order,
        # which follows the active import path.  Keep that order among modern
        # ``.dist-info`` entries: sorting paths can select an older inherited
        # environment over the active checkout merely because its directory
        # name sorts first.  A legacy ``.egg-info`` remains the fallback when
        # no modern metadata is present.
        selected = next(
            (
                distribution
                for distribution in distributions
                if str(getattr(distribution, "_path", "")).endswith(".dist-info")
            ),
            distributions[0],
        )
        value = str(selected.version or "").strip()
        return value or None
    try:
        value = str(metadata.version(distribution_name) or "").strip()
    except (metadata.PackageNotFoundError, TypeError, KeyError, AttributeError, ValueError):
        return None
    return value or None
