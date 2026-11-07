from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cycler import cycler

OEL_DARK_PALETTE = {
    "background": "#0B1020",
    "panel": "#111827",
    "panel_alt": "#172033",
    "grid": "#374151",
    "text": "#E5E7EB",
    "muted_text": "#CBD5E1",
    "edge": "#6B7280",
    "target": "#FBBF24",
    "chaser": "#F97316",
    "blue": "#38BDF8",
    "green": "#34D399",
    "violet": "#A78BFA",
    "rose": "#FB7185",
    "warning": "#F43F5E",
    "coast": "#94A3B8",
    "safety": "#22D3EE",
}

OEL_LIGHT_PALETTE = {
    "background": "#F8FAFC",
    "panel": "#FFFFFF",
    "panel_alt": "#EEF2F7",
    "grid": "#CBD5E1",
    "text": "#111827",
    "muted_text": "#475569",
    "edge": "#64748B",
    "target": "#B7791F",
    "chaser": "#C2410C",
    "blue": "#0369A1",
    "green": "#047857",
    "violet": "#6D28D9",
    "rose": "#BE123C",
    "warning": "#DC2626",
    "coast": "#64748B",
    "safety": "#0891B2",
}

OEL_ROLE_COLORS = {
    "target": OEL_DARK_PALETTE["target"],
    "chaser": OEL_DARK_PALETTE["chaser"],
    "actual": OEL_DARK_PALETTE["blue"],
    "desired": OEL_DARK_PALETTE["green"],
    "coast": OEL_DARK_PALETTE["coast"],
    "burn": OEL_DARK_PALETTE["violet"],
    "warning": OEL_DARK_PALETTE["warning"],
    "safety_zone": OEL_DARK_PALETTE["safety"],
}

OEL_ROLE_COLORS_LIGHT = {
    "target": OEL_LIGHT_PALETTE["target"],
    "chaser": OEL_LIGHT_PALETTE["chaser"],
    "actual": OEL_LIGHT_PALETTE["blue"],
    "desired": OEL_LIGHT_PALETTE["green"],
    "coast": OEL_LIGHT_PALETTE["coast"],
    "burn": OEL_LIGHT_PALETTE["violet"],
    "warning": OEL_LIGHT_PALETTE["warning"],
    "safety_zone": OEL_LIGHT_PALETTE["safety"],
}


OEL_STYLE_DARK: dict[str, Any] = {
    "figure.facecolor": OEL_DARK_PALETTE["background"],
    "savefig.facecolor": OEL_DARK_PALETTE["background"],
    "savefig.edgecolor": OEL_DARK_PALETTE["background"],
    "axes.facecolor": OEL_DARK_PALETTE["panel"],
    "axes.edgecolor": OEL_DARK_PALETTE["edge"],
    "axes.labelcolor": OEL_DARK_PALETTE["text"],
    "axes.titlecolor": OEL_DARK_PALETTE["text"],
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(
        color=[
            OEL_DARK_PALETTE["blue"],
            OEL_DARK_PALETTE["target"],
            OEL_DARK_PALETTE["chaser"],
            OEL_DARK_PALETTE["green"],
            OEL_DARK_PALETTE["violet"],
            OEL_DARK_PALETTE["rose"],
            OEL_DARK_PALETTE["coast"],
        ]
    ),
    "xtick.color": OEL_DARK_PALETTE["muted_text"],
    "ytick.color": OEL_DARK_PALETTE["muted_text"],
    "grid.color": OEL_DARK_PALETTE["grid"],
    "grid.alpha": 0.35,
    "text.color": OEL_DARK_PALETTE["text"],
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 12.0,
    "axes.titlepad": 10.0,
    "axes.labelsize": 10.0,
    "legend.facecolor": OEL_DARK_PALETTE["panel_alt"],
    "legend.edgecolor": OEL_DARK_PALETTE["edge"],
    "legend.labelcolor": OEL_DARK_PALETTE["text"],
    "lines.linewidth": 2.0,
    "patch.edgecolor": OEL_DARK_PALETTE["edge"],
}

OEL_STYLE_LIGHT: dict[str, Any] = {
    "figure.facecolor": OEL_LIGHT_PALETTE["background"],
    "savefig.facecolor": OEL_LIGHT_PALETTE["background"],
    "savefig.edgecolor": OEL_LIGHT_PALETTE["background"],
    "axes.facecolor": OEL_LIGHT_PALETTE["panel"],
    "axes.edgecolor": OEL_LIGHT_PALETTE["edge"],
    "axes.labelcolor": OEL_LIGHT_PALETTE["text"],
    "axes.titlecolor": OEL_LIGHT_PALETTE["text"],
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(
        color=[
            OEL_LIGHT_PALETTE["blue"],
            OEL_LIGHT_PALETTE["target"],
            OEL_LIGHT_PALETTE["chaser"],
            OEL_LIGHT_PALETTE["green"],
            OEL_LIGHT_PALETTE["violet"],
            OEL_LIGHT_PALETTE["rose"],
            OEL_LIGHT_PALETTE["coast"],
        ]
    ),
    "xtick.color": OEL_LIGHT_PALETTE["muted_text"],
    "ytick.color": OEL_LIGHT_PALETTE["muted_text"],
    "grid.color": OEL_LIGHT_PALETTE["grid"],
    "grid.alpha": 0.45,
    "text.color": OEL_LIGHT_PALETTE["text"],
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 12.0,
    "axes.titlepad": 10.0,
    "axes.labelsize": 10.0,
    "legend.facecolor": OEL_LIGHT_PALETTE["panel"],
    "legend.edgecolor": OEL_LIGHT_PALETTE["edge"],
    "legend.labelcolor": OEL_LIGHT_PALETTE["text"],
    "lines.linewidth": 2.0,
    "patch.edgecolor": OEL_LIGHT_PALETTE["edge"],
}

OEL_STYLES = {
    "oel_dark": OEL_STYLE_DARK,
    "oel_light": OEL_STYLE_LIGHT,
}

MATPLOTLIB_STYLE_NAMES = {"", "none", "default", "matplotlib"}


@dataclass(frozen=True)
class OELArtifactMetadata:
    scenario_name: str = ""
    generated_utc: str = ""
    version: str = ""
    artifact_id: str = ""


_CURRENT_METADATA: ContextVar[OELArtifactMetadata | None] = ContextVar("oel_plot_metadata", default=None)
_CURRENT_STYLE_NAME: ContextVar[str] = ContextVar("oel_plot_style_name", default="oel_dark")


def get_oel_version() -> str:
    pyproject_version = _version_from_source_pyproject()
    if pyproject_version:
        return pyproject_version
    try:
        value = str(version("orbital-engineering-lab") or "").strip()
        if value and value.lower() != "none":
            return value
    except (PackageNotFoundError, TypeError, KeyError, AttributeError, ValueError):
        pass
    return "unknown"


def _version_from_source_pyproject() -> str | None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        in_project = False
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                in_project = stripped == "[project]"
                continue
            if in_project and stripped.startswith("version"):
                value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                return value or None
    except OSError:
        return None
    return None


def utc_stamp() -> str:
    return os.environ.get("OEL_GENERATED_UTC", "").strip()


def current_artifact_metadata() -> OELArtifactMetadata | None:
    return _CURRENT_METADATA.get()


def current_style_name() -> str:
    return _CURRENT_STYLE_NAME.get()


def role_color(role: str, *, style_name: str | None = None) -> str:
    style_key = str(style_name or current_style_name() or "oel_dark").strip().lower()
    colors = OEL_ROLE_COLORS_LIGHT if style_key == "oel_light" else OEL_ROLE_COLORS
    return colors.get(str(role or "").strip().lower(), colors["actual"])


def artifact_metadata(
    *,
    scenario_name: str = "",
    generated_utc: str | None = None,
    version_text: str | None = None,
    artifact_id: str = "",
) -> OELArtifactMetadata:
    return OELArtifactMetadata(
        scenario_name=str(scenario_name or "").strip(),
        generated_utc=str(generated_utc or utc_stamp()).strip(),
        version=str(version_text or get_oel_version()).strip(),
        artifact_id=str(artifact_id or "").strip(),
    )


def style_name_from_config(plots_cfg: dict[str, Any] | None) -> str:
    cfg = dict(plots_cfg or {})
    return str(cfg.get("style", cfg.get("theme", "oel_dark")) or "oel_dark").strip().lower()


def resolve_oel_style(style_name: str | None) -> dict[str, Any] | None:
    key = str(style_name or "").strip().lower()
    if key in MATPLOTLIB_STYLE_NAMES:
        return None
    if key not in OEL_STYLES:
        valid = ", ".join(sorted([*OEL_STYLES.keys(), "matplotlib"]))
        raise ValueError(f"Unknown plot style '{style_name}'. Valid styles: {valid}")
    return OEL_STYLES[key]


@contextmanager
def oel_plot_context(
    *,
    style_name: str | None = "oel_dark",
    metadata: OELArtifactMetadata | None = None,
) -> Iterator[None]:
    import matplotlib.pyplot as plt

    token = _CURRENT_METADATA.set(metadata)
    style_token = _CURRENT_STYLE_NAME.set(str(style_name or "matplotlib").strip().lower())
    style = resolve_oel_style(style_name)
    try:
        if style is None:
            yield
        else:
            with plt.rc_context(style):
                yield
    finally:
        _CURRENT_STYLE_NAME.reset(style_token)
        _CURRENT_METADATA.reset(token)


def _footer_text(metadata: OELArtifactMetadata, artifact_id: str = "") -> str:
    parts = ["Orbital Engineering Lab"]
    if metadata.version:
        parts.append(f"v{metadata.version}")
    scenario = metadata.scenario_name
    if scenario:
        parts.append(f"scenario: {scenario}")
    artifact = artifact_id or metadata.artifact_id
    if artifact:
        parts.append(f"artifact: {artifact}")
    if metadata.generated_utc:
        parts.append(f"generated: {metadata.generated_utc}")
    return " · ".join(parts)


def add_artifact_footer(
    fig: Any,
    *,
    metadata: OELArtifactMetadata | None = None,
    artifact_id: str = "",
) -> None:
    meta = metadata or current_artifact_metadata()
    if meta is None:
        return
    if not callable(getattr(fig, "text", None)):
        return
    text = _footer_text(meta, artifact_id=artifact_id)
    footer_color = "#94A3B8"
    existing = [
        item
        for item in getattr(fig, "texts", [])
        if str(getattr(item, "get_gid", lambda: "")() or "") == "oel_artifact_footer"
    ]
    for item in existing:
        item.remove()
    fig.text(
        0.995,
        0.005,
        text,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=footer_color,
        alpha=0.9,
        gid="oel_artifact_footer",
    )


def apply_oel_style_to_figure(fig: Any, *, style_name: str | None = None) -> None:
    style_key = str(style_name or current_style_name() or "").strip().lower()
    if style_key in MATPLOTLIB_STYLE_NAMES:
        return
    palette = OEL_DARK_PALETTE if style_key == "oel_dark" else OEL_LIGHT_PALETTE
    patch = getattr(fig, "patch", None)
    if patch is not None and hasattr(patch, "set_facecolor"):
        patch.set_facecolor(palette["background"])
    get_axes = getattr(fig, "get_axes", None)
    if not callable(get_axes):
        return
    for ax in get_axes():
        if hasattr(ax, "set_facecolor"):
            ax.set_facecolor(palette["panel"])
        if hasattr(ax, "tick_params"):
            ax.tick_params(colors=palette["muted_text"])
        xaxis = getattr(ax, "xaxis", None)
        yaxis = getattr(ax, "yaxis", None)
        if xaxis is not None and hasattr(xaxis, "label"):
            xaxis.label.set_color(palette["text"])
        if yaxis is not None and hasattr(yaxis, "label"):
            yaxis.label.set_color(palette["text"])
        zaxis = getattr(ax, "zaxis", None)
        if zaxis is not None and hasattr(zaxis, "label"):
            zaxis.label.set_color(palette["text"])
        title = getattr(ax, "title", None)
        if title is not None and hasattr(title, "set_color"):
            title.set_color(palette["text"])
        spines = getattr(ax, "spines", None)
        if spines is not None and hasattr(spines, "values"):
            for spine in spines.values():
                if hasattr(spine, "set_color"):
                    spine.set_color(palette["edge"])
        if hasattr(ax, "grid"):
            ax.grid(True, color=palette["grid"], alpha=0.35 if style_key == "oel_dark" else 0.45)
        legend = ax.get_legend() if hasattr(ax, "get_legend") else None
        if legend is not None:
            legend.get_frame().set_facecolor(palette["panel_alt"])
            legend.get_frame().set_edgecolor(palette["edge"])
            for text in legend.get_texts():
                text.set_color(palette["text"])
    for legend in getattr(fig, "legends", []) or []:
        legend.get_frame().set_facecolor(palette["panel_alt"])
        legend.get_frame().set_edgecolor(palette["edge"])
        for text in legend.get_texts():
            text.set_color(palette["text"])
    for text in getattr(fig, "texts", []) or []:
        if str(getattr(text, "get_gid", lambda: "")() or "") == "oel_artifact_footer":
            continue
        if hasattr(text, "set_color"):
            text.set_color(palette["text"])


@contextmanager
def _cache_builtin_text_layouts(fig: Any) -> Iterator[None]:
    """Reuse identical built-in text layouts within one Agg render only."""
    if not hasattr(fig, "axes"):
        yield
        return
    from matplotlib.axis import Axis, XAxis, YAxis
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.text import Text

    patched = []
    tick_factories = []
    missing = object()

    def patch_text(artist):
        if type(artist) is not Text or artist.get_wrap():
            return
        previous = artist.__dict__.get("_get_layout", missing)
        if previous is not missing:
            return
        original = artist._get_layout
        if getattr(original, "__func__", None) is not Text._get_layout:
            return
        cache: dict[tuple[Any, ...], Any] = {}

        def layout(renderer, *, text=artist, compute=original, values=cache):
            if type(renderer) is not RendererAgg or text.get_wrap():
                return compute(renderer)
            props = text.get_fontproperties()
            key = (
                renderer, float(text.get_figure().dpi), text.get_text(),
                tuple(props.get_family()), props.get_style(), props.get_variant(),
                props.get_weight(), props.get_stretch(), props.get_size_in_points(),
                props.get_file(), props.get_math_fontfamily(), text.get_usetex(),
                text.get_parse_math(), text._linespacing, text._get_multialignment(),
                text.get_rotation(), text.get_rotation_mode(),
                text.get_horizontalalignment(), text.get_verticalalignment(),
            )
            result = values.get(key)
            if result is None:
                result = compute(renderer)
                if len(values) >= 16:
                    values.clear()
                values[key] = result
            return result

        artist._get_layout = layout
        patched.append((artist, previous))

    # findobj traverses Axis.get_children(), which eagerly constructs ticks.
    # Visit existing labels and hook ordinary built-in tick creation instead.
    # Unrecognized nested/custom artists retain their ordinary text method.
    try:
        for artist in getattr(fig, "texts", ()):
            patch_text(artist)
        axes = list(getattr(fig, "axes", ()))
        legends = list(getattr(fig, "legends", ()))
        for ax in axes:
            axes.extend(getattr(ax, "child_axes", ()))
            for artist in (*getattr(ax, "texts", ()), getattr(ax, "title", None),
                           getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
                patch_text(artist)
            legend = ax.get_legend()
            if legend is not None:
                legends.append(legend)
            for name in ("xaxis", "yaxis", "zaxis"):
                axis = getattr(ax, name, None)
                if axis is None:
                    continue
                patch_text(axis.label)
                patch_text(axis.offsetText)
                for ticks in (axis.__dict__.get("majorTicks", ()), axis.__dict__.get("minorTicks", ())):
                    for tick in ticks:
                        patch_text(tick.label1)
                        patch_text(tick.label2)
                factory = axis._get_tick
                if type(axis) not in (XAxis, YAxis) or (
                    getattr(factory, "__func__", None) is not Axis._get_tick
                    or "_get_tick" in axis.__dict__
                ):
                    continue

                def make_tick(*args, create=factory, **kwargs):
                    tick = create(*args, **kwargs)
                    patch_text(tick.label1)
                    patch_text(tick.label2)
                    return tick

                axis._get_tick = make_tick
                tick_factories.append(axis)
        for legend in legends:
            for artist in (*legend.get_texts(), legend.get_title()):
                patch_text(artist)
        yield
    finally:
        for axis in tick_factories:
            axis.__dict__.pop("_get_tick", None)
        for artist, previous in patched:
            if previous is missing:
                artist.__dict__.pop("_get_layout", None)
            else:
                artist._get_layout = previous


def tight_layout_oel_figure(fig: Any, *args: Any, **kwargs: Any) -> None:
    """Run the same layout with bounded reuse of repeated built-in text metrics."""
    with _cache_builtin_text_layouts(fig):
        return fig.tight_layout(*args, **kwargs)


def save_oel_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int,
    metadata: OELArtifactMetadata | None = None,
    artifact_id: str = "",
    style_name: str | None = None,
    **savefig_kwargs: Any,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    apply_oel_style_to_figure(fig, style_name=style_name)
    add_artifact_footer(fig, metadata=metadata, artifact_id=artifact_id)
    if str(savefig_kwargs.get("format", p.suffix.lstrip("."))).lower() == "png":
        # Compression is lossless at every level. Preserve the rendered pixels,
        # DPI and metadata while avoiding the default level-six CPU cost.
        pil_kwargs = dict(savefig_kwargs.get("pil_kwargs") or {})
        pil_kwargs.setdefault("compress_level", 1)
        savefig_kwargs["pil_kwargs"] = pil_kwargs
    # tight_layout() leaves Matplotlib's no-op placeholder engine installed.
    # print_figure treats any non-None engine as requiring a preliminary draw,
    # even though this placeholder cannot change layout. Temporarily remove
    # exactly that inert engine; active/custom engines still own their draws.
    import matplotlib
    try:
        from matplotlib.layout_engine import PlaceHolderLayoutEngine
    except ImportError:
        PlaceHolderLayoutEngine = None

    get_layout_engine = getattr(fig, "get_layout_engine", None)
    layout = get_layout_engine() if callable(get_layout_engine) else None
    remove_placeholder = (
        PlaceHolderLayoutEngine is not None and type(layout) is PlaceHolderLayoutEngine
        and not matplotlib.rcParams["figure.autolayout"]
        and not matplotlib.rcParams["figure.constrained_layout.use"]
    )
    if remove_placeholder:
        fig.set_layout_engine(None)
    try:
        with _cache_builtin_text_layouts(fig):
            fig.savefig(p, dpi=int(dpi), **savefig_kwargs)
    finally:
        if remove_placeholder:
            fig.set_layout_engine(layout)


def prepare_oel_animation_figure(
    fig: Any,
    *,
    metadata: OELArtifactMetadata | None = None,
    artifact_id: str = "",
    style_name: str | None = None,
) -> None:
    apply_oel_style_to_figure(fig, style_name=style_name)
    add_artifact_footer(fig, metadata=metadata, artifact_id=artifact_id)


def save_oel_animation(
    animation_obj: Any,
    fig: Any,
    path: str | Path,
    *,
    fps: float,
    metadata: OELArtifactMetadata | None = None,
    artifact_id: str = "",
    style_name: str | None = None,
    **save_kwargs: Any,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prepare_oel_animation_figure(
        fig,
        metadata=metadata,
        artifact_id=artifact_id or p.stem,
        style_name=style_name,
    )
    animation_obj.save(str(p), fps=max(float(fps), 1.0), **save_kwargs)


def show_save_close_oel(
    fig: Any,
    *,
    mode: str = "save",
    out_path: str | Path | None = None,
    dpi: int = 150,
    artifact_id: str = "",
    metadata: OELArtifactMetadata | None = None,
    style_name: str | None = None,
    plt_module: Any | None = None,
    close: bool | None = None,
    show_block: bool = False,
    **savefig_kwargs: Any,
) -> str | None:
    path: Path | None = None
    mode_norm = str(mode or "save").strip().lower()
    should_save = mode_norm in {"save", "both"}
    should_show = mode_norm in {"interactive", "both"}
    if should_save:
        if out_path is None:
            raise ValueError("out_path is required when saving a plot artifact.")
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_oel_figure(
            fig,
            path,
            dpi=int(dpi),
            metadata=metadata,
            artifact_id=artifact_id or path.stem,
            style_name=style_name,
            **savefig_kwargs,
        )
    elif metadata is not None or style_name is not None:
        apply_oel_style_to_figure(fig, style_name=style_name)
        add_artifact_footer(fig, metadata=metadata, artifact_id=artifact_id)

    if should_show:
        if plt_module is not None and callable(getattr(plt_module, "show", None)):
            plt_module.show(block=show_block)
        elif callable(getattr(fig, "show", None)):
            fig.show()

    should_close = (not should_show) if close is None else bool(close)
    if should_close and plt_module is not None and callable(getattr(plt_module, "close", None)):
        plt_module.close(fig)
    return None if path is None else str(path)
