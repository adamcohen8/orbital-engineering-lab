"""Actionable format requirements shared by dispatch and preflight."""

from .contracts import DataIOError

READ_OPTIONS = {
    "stk-ephemeris": ("object_id",),
    "stk-attitude": ("object_id",),
    "sp3": ("frame",),
    "sentinel-eof": ("object_id", "frame"),
    "spk": ("epochs_utc", "target_id", "center_id", "frame", "object_id", "kernels"),
    "ck": ("epochs_utc", "instrument_id", "clock_id", "frame", "object_id", "body_frame", "kernels"),
}
WRITE_OPTIONS = {
    "opm-kvn": ("sample_index",),
    "opm-xml": ("sample_index",),
    "sp3": ("satellite_id",),
    "spk": ("target_id", "center_id", "kernels"),
    "ck": ("instrument_id", "clock_id", "kernels"),
}


def require_read_options(format, options):
    missing = [key for key in READ_OPTIONS.get(format, ()) if key not in options or options[key] is None]
    if missing:
        flags = ["--" + key.replace("_", "-") for key in missing]
        raise DataIOError(
            f"{format} input requires {', '.join(flags)}; advanced kernel/time selections belong in --options JSON."
        )
