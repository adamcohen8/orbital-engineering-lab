# Custom Analysis

Orbital Engineering Lab writes JSON, CSV, Markdown, and image artifacts that can
be used outside the built-in plotting system. If you want a plot that the
public core does not provide, the intended path is:

1. enable the right saved data in YAML,
2. run the scenario,
3. open the output directory `index.md`, and
4. load the saved artifacts with Python, NumPy, Matplotlib, or another analysis
   tool.

Built-in plots are useful defaults. Saved artifacts are the escape hatch for
custom engineering questions.

When `outputs.review.enabled: true`, prefer the queryable review store before
loading the full run log. Confirm `review/run.sqlite` and its schema with
`result.evidence_manifest()`, open it with `result.review()`, or use
`python -m sim.review <output-dir> --query <SELECT-or-WITH-query>`. Discover
columns from `review/schema.json` or a small `SELECT * ... LIMIT 1` query. The
review API is read-only. Fall back to `master_run_log.json` only when the
required signal was not recorded in review.

## Choose The Right Artifact

For a single run:

- `index.md`: start-here output guide and artifact inventory.
- `master_run_summary.json`: stable summary metrics and artifact maps.
- `master_run_log.json`: full time histories for custom plots.
- `review/run.sqlite` and `review/schema.json`: queryable, provenance-bound
  evidence when review output was enabled.

The v1 local edition also emits bounded Monte Carlo, sensitivity,
controller-benchmark, and reporting artifacts. Keep custom analysis inside the
declared campaign/resource limits; Hosted orchestration and provider-driven
reports remain private.

## Save Full Single-Run Data

To make arbitrary time-history plots from a single run, enable the full log:

```yaml
outputs:
  output_dir: "outputs/my_run"
  mode: "save"
  stats:
    enabled: true
    save_json: true
    save_full_log: true
```

After the run, open:

```text
outputs/my_run/index.md
outputs/my_run/master_run_log.json
```

`master_run_summary.json` is enough for summary plots. `master_run_log.json` is
the better source for time-series plots because it contains histories such as
`time_s`, `truth_by_object`, `belief_by_object`, `applied_thrust_by_object`,
`applied_torque_by_object`, `knowledge_by_observer`,
`ground_station_access`, and controller debug data when available.

## Single-Run Example: Altitude Over Time

The truth state convention starts with ECI position and velocity. The first
three truth columns are position in kilometers.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.style import artifact_metadata, save_oel_figure

EARTH_RADIUS_KM = 6378.137

outdir = Path("outputs/plotting_rendezvous_demo")
payload = json.loads((outdir / "master_run_log.json").read_text())
metadata = artifact_metadata(scenario_name=payload.get("summary", {}).get("scenario_name", "custom_analysis"))

time_s = np.asarray(payload["time_s"], dtype=float)
target_truth = np.asarray(payload["truth_by_object"]["target"], dtype=float)
target_r_eci_km = target_truth[:, 0:3]
target_altitude_km = np.linalg.norm(target_r_eci_km, axis=1) - EARTH_RADIUS_KM

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(time_s / 60.0, target_altitude_km)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Altitude (km)")
ax.set_title("Target Altitude")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_oel_figure(
    fig,
    outdir / "custom_target_altitude.png",
    dpi=160,
    metadata=metadata,
    artifact_id="custom_target_altitude",
    style_name="oel_dark",
)
```

If `master_run_log.json` is missing, set `outputs.stats.save_full_log: true`
and rerun the scenario.

## Single-Run Example: Two-Object Range

For two-object scenarios, relative range can be computed directly from the
truth histories. This example uses the `chaser` and `target` IDs from the
public rendezvous config; replace them with your own object IDs for other
scenarios.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.style import artifact_metadata, save_oel_figure

outdir = Path("outputs/examples/public_rendezvous_closed_loop")
payload = json.loads((outdir / "master_run_log.json").read_text())
metadata = artifact_metadata(scenario_name=payload.get("summary", {}).get("scenario_name", "custom_analysis"))

time_s = np.asarray(payload["time_s"], dtype=float)
target = np.asarray(payload["truth_by_object"]["target"], dtype=float)
chaser = np.asarray(payload["truth_by_object"]["chaser"], dtype=float)

range_km = np.linalg.norm(chaser[:, 0:3] - target[:, 0:3], axis=1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(time_s, range_km)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Range (km)")
ax.set_title("Object-to-Object Range")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_oel_figure(
    fig,
    outdir / "custom_chaser_target_range.png",
    dpi=160,
    metadata=metadata,
    artifact_id="custom_chaser_target_range",
    style_name="oel_dark",
)
```

This is useful when a built-in figure is close to what you need but not exactly
right for a review or paper.

## Single-Run Example: Ground-Station Access

When a scenario defines `ground_stations`, the built-in `ground_station_access`
figure plots access, elevation, and range for configured station/object pairs.
The full run log also contains the raw histories. Use the custom-analysis path
when you need a station-specific, publication-specific, or heavily customized
variant:

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.style import artifact_metadata, save_oel_figure

outdir = Path("outputs/my_ground_station_run")
payload = json.loads((outdir / "master_run_log.json").read_text())
metadata = artifact_metadata(scenario_name=payload.get("summary", {}).get("scenario_name", "custom_analysis"))

time_s = np.asarray(payload["time_s"], dtype=float)
access_payload = payload["ground_station_access"]["colorado_springs"]["targets"]["target"]

access = np.asarray(access_payload["access"], dtype=bool)
elevation_deg = np.asarray(access_payload["elevation_deg"], dtype=float)
range_km = np.asarray(access_payload["range_km"], dtype=float)

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].step(time_s / 60.0, access.astype(int), where="post")
axes[0].set_ylabel("Access")
axes[0].set_yticks([0, 1])

axes[1].plot(time_s / 60.0, elevation_deg)
axes[1].set_ylabel("Elevation (deg)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time_s / 60.0, range_km)
axes[2].set_xlabel("Time (min)")
axes[2].set_ylabel("Range (km)")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
save_oel_figure(
    fig,
    outdir / "custom_ground_station_access.png",
    dpi=160,
    metadata=metadata,
    artifact_id="custom_ground_station_access",
    style_name="oel_dark",
)
```

For quick review without plotting, inspect `ground_station_access_summary` in
`master_run_summary.json` or `master_run_log.json`. It includes first/last
access time, access duration, minimum range, and maximum elevation.

## Practical Guidance

- Open `index.md` first. It lists the artifacts actually written for that run.
- Use `master_run_log.json` for arbitrary single-run time histories.
- Use summary JSON for stable top-level metrics.
- Prefer artifact maps in JSON over hard-coded filenames when writing reusable
  scripts.
- Save custom plots back into the same output directory when they belong to the
  same analysis record.

## Limitations

Saved artifacts reflect what the run was configured to record. If a signal was
not logged, a custom plot cannot reconstruct it after the fact. In that case,
enable the relevant output option, add the metric to the analysis payload, or
extend the simulator logging before rerunning.
