"""Route full continuation products to their established handoff owners."""

from sim.data_io.contracts import DataIOError


def materialize_continuation(source, output, *, format, scenario_name, output_dir, duration_s, step_s):
    common = {
        "scenario_name": scenario_name,
        "scenario_path": output,
        "output_dir": output_dir,
        "duration_s": duration_s,
        "dt_s": step_s,
    }
    if format == "satellite-checkpoint":
        from sim.interchange.satellite_checkpoints import materialize_satellite_checkpoint

        return materialize_satellite_checkpoint(source, **common)
    if format == "snapshot":
        from sim.interchange.snapshots import materialize_snapshot_onp

        return materialize_snapshot_onp(source, **common)
    raise DataIOError("Select satellite-checkpoint or snapshot for complete continuation.")
