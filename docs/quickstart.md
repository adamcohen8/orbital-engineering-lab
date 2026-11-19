# Quickstart

This guide takes you from a fresh checkout to a completed headless simulation
run.

## Install

OEL supports Python 3.10 through 3.14. Python 3.14 is recommended for a new
environment. See [Installing OEL On Windows, macOS, And
Linux](installation.md) for interpreter selection, constrained profiles, and
troubleshooting.

Windows PowerShell:

```powershell
git clone https://github.com/adamcohen8/orbital-engineering-lab.git
Set-Location orbital-engineering-lab
py --list
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe run_simulation.py --doctor
.\.venv\Scripts\python.exe run_simulation.py --quickstart
```

macOS or Linux:

```bash
git clone https://github.com/adamcohen8/orbital-engineering-lab.git
cd orbital-engineering-lab
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ".[dev]"
.venv/bin/python run_simulation.py --doctor
.venv/bin/python run_simulation.py --quickstart
```

To use another supported minor, change `3.14` consistently and select its
matching `constraints/py3XX.txt` file for constrained installs. Windows users
can run `py --list` (or `py -0p` with an older launcher) to see installed
interpreters.

The remaining commands assume the environment is activated:

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

POSIX:

```bash
source .venv/bin/activate
```

## Check Your Environment

```bash
python run_simulation.py --doctor
```

Doctor treats the core plotting stack as required and reports optional trainer,
acceleration, validation, development, and ML capabilities separately. If the
environment is incomplete, copy the OS-appropriate recovery commands printed
at the end of the report.

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules; loading an untrusted scenario can run
untrusted Python code.

If you only want to inspect a scenario from an unknown source, start with:

```bash
python run_simulation.py --config <path> --safe-validate
```

That mode avoids importing configured plugin modules. It does not make the
scenario safe to execute.

For shared classroom, government, or enterprise review, validate with the
restricted profile before execution:

```bash
python run_simulation.py --config <path> --sealed-mode --validate-only
```

Sealed mode blocks arbitrary plugin modules, hosted/custom AI endpoints,
non-loopback cFS/SIL networking, and high-detail output retention unless the
caller explicitly opts into the specific exception.

## Validate The Five-Minute Scenario

```bash
python run_simulation.py --quickstart --validate-only
```

Validation loads the YAML, checks timing and plugin pointers, and confirms the
scenario is structurally ready to run.

## Run The Scenario

```bash
python run_simulation.py --quickstart
```

The quickstart scenario is intentionally small and headless. It propagates a
two-satellite relative-motion hold with public controllers, sensing, and EKF
knowledge updates, then writes summary artifacts under
`outputs/quickstart_5min/`. Open `outputs/quickstart_5min/index.md` first for
the run summary, review order, and artifact inventory.

Both spacecraft maneuver, so the reported total delta-v includes both the
chaser and the target. The target uses an ideal actuator that applies force
without consuming propellant.

Plots are disabled in this first path to keep the first run fast, headless, and
focused on the generated summary artifacts. The quickstart does write
`review/run.sqlite`, so you can inspect the completed run with the review CLI:

```bash
python -m sim.review outputs/quickstart_5min --saved-query run_metadata
```

To open the output folder automatically after the run:

```bash
python run_simulation.py --quickstart --open-output
```

## Run The Flagship Review Scenario

After the quickstart succeeds, run the polished 10 km RIC_PD rendezvous
workflow:

```bash
python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
```

This scenario is longer than the quickstart. It exercises a tuned RIC_PD orbit
controller, attitude dynamics, quaternion attitude control through the configured
ideal-wrench abstraction, and thrust-alignment
gating, then writes review artifacts under `outputs/flagship_ric_pd_10km/`.
Open `outputs/flagship_ric_pd_10km/index.md` first.

For the scenario-specific review path, see
[Flagship RIC_PD 10 km Scenario And Validation](validation-ric-pd-10km.md).

## Get Config Field Help

When editing scenario YAML, use `config_help.py` to list valid values for a
field or topic:

```bash
python config_help.py "ephemeris model"
python config_help.py "plot preset"
python config_help.py --list
```

Add `--config` to inspect the value currently set in a scenario file without
loading scenario plugins or running the simulation:

```bash
python config_help.py "ephemeris model" --config configs/ric_pd_10km_experiment.yaml
```

The helper accepts fuzzy queries, so near-misses such as `"emphemeris model"`
still resolve to the ephemeris-mode field.

## Use The API

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary)
```

If your scenario defines `ground_stations`, the same result exposes passive
access histories:

```python
for station_id, station_payload in result.ground_station_access.items():
    for object_id, access in station_payload["targets"].items():
        print(station_id, object_id, access["access"])
```

Access is computed from line of sight, minimum elevation, and optional maximum
range. See [Scenario YAML](scenario-yaml.md) for the ground-station fields.
For more examples, see [Python API](python-api.md).

The flagship scenario also has a companion analysis script:

```bash
python examples/python/flagship_analysis.py
```

It writes custom review metrics under
`outputs/flagship_ric_pd_10km/custom_analysis/`.

## Next Scenarios

See [Examples Matrix](examples-matrix.md) for the maintained public examples,
what each one demonstrates, and which output artifacts to inspect first.

Two additional headless examples cover spacecraft resources and ground
knowledge:

- Run `python run_simulation.py --config configs/spacecraft_resources_demo.yaml`
  for the opt-in ONP thermal and battery model. See
  [Spacecraft Resources](models/spacecraft-resources.md) for its assumptions
  and output fields.
- Run `python run_simulation.py --config configs/ground_segment_demo.yaml`
  for delayed station tracking and ideal resource telemetry. See
  [Ground Segment](models/ground-segment.md) for priors, packet timing, and
  freshness semantics.

Run the compact rendezvous example:

```bash
python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml
```

The broader plotting example writes artifacts such as:

```text
outputs/examples/public_rendezvous_closed_loop/rendezvous_summary.png
outputs/examples/public_rendezvous_closed_loop/ground_track_multi.png
outputs/examples/public_rendezvous_closed_loop/sensor_access.png
```

See [Plotting](plotting.md) for the maintained plot presets, figure IDs, and
map-backed ground-track option.

Atmospheric re-entry diagnostics have a short laptop-safe smoke config and a
long-running ten-day study. The long study reports CLI progress, saves review
evidence, and avoids a redundant full JSON history; it can still take substantial
time depending on the machine:

```bash
python run_simulation.py --config configs/reentry_smoke.yaml
python run_simulation.py --config examples/configs/public_reentry_interactive_demo.yaml
```

## Optional Profiles

```bash
python -m pip install ".[ml]"
python -m pip install ".[full]"
```

The base package already installs NumPy and Matplotlib for simulation and
plotting support. The recommended public onboarding path remains
CLI/YAML/Python API, and the scripted review path remains
`python -m sim.review`. The ML profile enables the bundled
Gymnasium-style environments.

## Gravity Coefficient Files

The public core supports spherical-harmonic gravity from inline YAML terms or
from coefficient files you provide. HPOP/GGM03 validation data is not bundled in
the public distribution, so scenarios that set `source: "hpop_ggm03"` should also
set `coeff_path` to a local coefficient file.

`source: "egm96"` is the exception: it can resolve the managed digest-pinned
EGM96 model without `coeff_path`. Set `allow_download` explicitly; downloads
remain unavailable in sealed mode. `source: "icgem"` still requires a local
coefficient file.

For a bounded passive two-satellite question with explicit assumptions and
review queries, continue with [Your first relative-motion study](first-relative-study.md).
