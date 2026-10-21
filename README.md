<!-- Source template: tools/export_public.py copies this file to the public export root as README.md. Root-relative links below are intentional. -->
# Orbital Engineering Lab


Orbital Engineering Lab is an open-core Python/YAML simulator for spacecraft
rendezvous, proximity operations, and mission-analysis prototyping. Define a
scenario, run a deterministic simulation, and inspect review-ready Markdown,
JSON, CSV, SQLite, and plot artifacts.

Use OEL when you want to prototype closed-loop spacecraft behavior, not just
propagate a trajectory. A single scenario can combine orbital mechanics, attitude
dynamics, sensors, estimators, controllers, actuators, mission logic,
ground-station access, and output artifacts.

OEL is intended for research, education, prototyping, pre-flight engineering
analysis, and software-in-the-loop experimentation. It is not flight-qualified
software or an operational decision system.

![Flagship run dashboard](docs/assets/plots/run_dashboard.png)

## Install And Run

OEL supports Python 3.10 through 3.14. Python 3.14 is recommended. Official
public releases provide signed managed installers that keep immutable OEL
engines separate from user-owned projects and workspaces.

Inspect the small installer before running it.

macOS or Linux:

```bash
curl --proto '=https' --tlsv1.2 -fsSLo /tmp/oel-install.sh \
  https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/install.sh
less /tmp/oel-install.sh
sh /tmp/oel-install.sh
```

Windows PowerShell:

```powershell
Invoke-WebRequest https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/install.ps1 -OutFile $env:TEMP\oel-install.ps1
Get-Content $env:TEMP\oel-install.ps1
& $env:TEMP\oel-install.ps1
```

The bootstrap verifies the signed release metadata and exact artifact digest
before installing OEL. It does not edit or overwrite a workspace. Create a
project for your scenarios, flight-software source, configuration, and outputs,
then run the managed quickstart:

```text
oel update status --full
oel doctor
oel workspace init my-oel-workspace
oel --workspace my-oel-workspace sim --quickstart --validate-only
oel --workspace my-oel-workspace sim --quickstart
```

Expected result: the run writes summary artifacts under
`my-oel-workspace/outputs/quickstart_5min/`. Open `index.md` there first. See
[Installing OEL](docs/installation.md), [Updating OEL](docs/updating.md), and
[OEL Workspaces](docs/workspaces.md) for installation, updates, explicit
workspace adoption, rollback, offline installation, and troubleshooting.

## Source Installation For Contributors

The public checkout includes the Rust implementations and build inputs for both
required native packages. Follow [building native source](docs/native-source-build.md)
before installing OEL from source. Install the Rust toolchain before running
the native build commands below.

Cloning the repository remains the contributor workflow and manual fallback.
It is not required for the managed installation above. See
[Installing OEL](docs/installation.md) for other supported Python minors and
their matching constraint files.

Windows PowerShell:

```powershell
git clone https://github.com/adamcohen8/orbital-engineering-lab.git
Set-Location orbital-engineering-lab
py --list
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "maturin>=1.10,<2"
.\.venv\Scripts\python.exe -m maturin build --release --locked --manifest-path rust/oel-orbit/Cargo.toml --interpreter .\.venv\Scripts\python.exe --out dist/native
.\.venv\Scripts\python.exe -m maturin build --release --locked --manifest-path rust/oel-game/Cargo.toml --interpreter .\.venv\Scripts\python.exe --out dist/native
.\.venv\Scripts\python.exe -m pip install --no-index --find-links dist/native "oel-rust-orbit>=0.18.3,<0.19" "oel-rust-game>=0.5,<0.6"
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
.venv/bin/python -m pip install "maturin>=1.10,<2"
.venv/bin/python -m maturin build --release --locked --manifest-path rust/oel-orbit/Cargo.toml --interpreter .venv/bin/python --out dist/native
.venv/bin/python -m maturin build --release --locked --manifest-path rust/oel-game/Cargo.toml --interpreter .venv/bin/python --out dist/native
.venv/bin/python -m pip install --no-index --find-links dist/native "oel-rust-orbit>=0.18.3,<0.19" "oel-rust-game>=0.5,<0.6"
.venv/bin/python -m pip install ".[dev]"
.venv/bin/python run_simulation.py --doctor
.venv/bin/python run_simulation.py --quickstart
```

The source-checkout commands below use `python` after activation. In
PowerShell, run `.\.venv\Scripts\Activate.ps1`; in Bash or Zsh, run
`source .venv/bin/activate`. Activation is optional when using the explicit
interpreter paths above.

## Fast Proof Path For A Source Checkout

Check the environment again when troubleshooting:

```bash
python run_simulation.py --doctor
```

Doctor runs before the scientific runtime imports, reports the exact supported
Python range and detected install profile, and prints OS-appropriate recovery
commands for missing or incompatible dependencies.

Run the fast headless quickstart:

```bash
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
```

Expected result: the source-checkout run writes summary artifacts under
`outputs/quickstart_5min/`. Open `outputs/quickstart_5min/index.md` first.

Then run the flagship 10 km RIC_PD RPO review scenario:

```bash
python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
```

Expected result: OEL writes a dashboard, rendezvous summary, control-effort,
relative-range, trajectory, attitude, quaternion-error, and thrust-alignment
plots under `outputs/flagship_ric_pd_10km/`. Start with
`outputs/flagship_ric_pd_10km/index.md`, then compare against the checked-in
[Plot Gallery](docs/plot-gallery.md).

For the guided first-run walkthrough, see [Quickstart](docs/quickstart.md).

## What You Can Do

- Propagate spacecraft from classical orbital elements or TLE-initialized
  states.
- Run closed-loop chaser/target rendezvous and relative-motion scenarios.
- Exercise public orbit and attitude controllers, estimators, sensors, mission
  modules, and actuator models.
- Scaffold, inspect, validate, component-test, and deterministically smoke-test
  a custom public ADCS or RPO complete-stack flight-software candidate.
- Inspect passive ground-station access using line of sight, elevation, and
  range histories.
- Simulate headless ground tracking, packet delay, telemetry freshness, and
  service outages; see [Ground Segment](docs/models/ground-segment.md).
- Enable one-node thermal balance and bounded battery state in supported ONP
  satellite runs; see [Spacecraft Resources](docs/models/spacecraft-resources.md).
- Evaluate whole-Earth coverage and directed free-space links from deterministic
  ONP, completed-review, or ECI OGP histories.
- Explore perturbation models, atmosphere, SRP, third bodies, spherical
  harmonics, re-entry diagnostics, and threshold termination criteria.
- Import, inspect and convert local orbital data, compose scenario objects, and
  export completed review evidence through [Modular Data IO](docs/data-io.md).
- Opt into experimental [full-state EKF innovation recovery](docs/innovation-recovery.md)
  for HCW, SS-J2, TH, YA, or absolute ECI knowledge estimation.
- Generate single-run dashboards, plots, summaries, review stores, and artifact
  indexes.
- Use a Python API or scenario YAML as the durable scenario interface.
- Launch the Pygame RPO trainer for hands-on relative-motion practice.
- Use checked-in OEL Agent instructions and task cards with AI coding
  assistants.

The primary public surfaces are the CLI, scenario YAML, Python API, review
query CLI/API, modular data IO, custom review plotting API, and the RPO trainer.
Public single-scenario runs step objects deterministically and serially.

## Choose A Workflow

| Goal | Start here |
| --- | --- |
| First successful run | `python run_simulation.py --quickstart` |
| Flagship RPO artifact review | `python run_simulation.py --config configs/ric_pd_10km_experiment.yaml` |
| Approximate TLE-initialized OEL propagation | `python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml` |
| Passive OGP-SGP4 general-perturbations propagation | `python run_simulation.py --config examples/configs/public_sgp4_passive_propagation.yaml` |
| Geometric ground-station access from a TLE-initialized OEL run | `python run_simulation.py --config examples/configs/public_ground_station_access_from_tle.yaml` |
| Simulate spacecraft thermal and battery resources | `python run_simulation.py --config configs/spacecraft_resources_demo.yaml` |
| Simulate delayed ground tracking and telemetry | `python run_simulation.py --config configs/ground_segment_demo.yaml` |
| Whole-Earth coverage and directed free-space link analysis | `python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml` |
| Closed-loop public rendezvous | `python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml` |
| Mission-recovery evidence case | `python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml` |
| Mission-reconstitution trade space | `python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml` |
| Event-driven impulsive trajectory target | `python -m sim.trajectory_design solve examples/trajectory_targeting/hohmann_apoapsis.json --output outputs/trajectory_targeting/hohmann_evidence.json` |
| Attitude hold with disturbance torque | `python run_simulation.py --config examples/configs/public_attitude_hold_disturbance.yaml` |
| Import local files, compose objects, or export saved evidence | `oel io formats` then [Modular Data IO](docs/data-io.md) |
| Configure opt-in full-state EKF recovery | [Full-state innovation recovery](docs/innovation-recovery.md) |
| Author a public FSW stack | `oel fsw init my_adcs --template adcs` then [Public FSW Authoring](docs/fsw-authoring.md) |
| Re-entry diagnostics | `python run_simulation.py --config configs/reentry_smoke.yaml` |
| RPO trainer game | `python -m pip install ".[game]"` then `python run_game.py` |
| AI-agent workflows | [Capability Routing And Golden Paths](docs/agent-capability-routing.md) |

Most TLE examples use TLE lines to initialize an ECI state, then OEL
numerically integrates the configured force model. Treat only scenarios with
`propagation_method: general` and `general.model: sgp4` as OGP
general-perturbations runs. OGP-SDP4/deep-space propagation is available for
passive deep-space/resonance TLEs.

## Use OEL With AI Coding Agents

The public repo includes OEL Agents: instructions, examples, task cards, and
tests for AI coding assistants such as Codex, Cursor, Claude Code, Gemini CLI,
and Grok Build.

The agent loop is:

```text
natural-language request -> scenario YAML -> validation -> deterministic run
-> review-store query or artifact inspection -> evidence-backed answer
```

For the shortest reliable path, point your agent at:

- [AGENTS.md](AGENTS.md)
- [agents/public/AGENTS.md](agents/public/AGENTS.md)
- [OEL Agents](docs/oel-agents.md)
- [Agent Capability Routing And Golden Paths](docs/agent-capability-routing.md)

Runs that enable `outputs.review.enabled: true` write
`review/run.sqlite`. Use the SELECT-only review CLI/API and cite the query or
saved query that supports a conclusion:

```bash
python -m sim.review outputs/my_run --saved-query run_metadata
python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s FROM run_metadata"
```

If an agent finds public-safe workflow friction, use
[Agent Feedback Loop](docs/agent-feedback-loop.md). Agents must show the draft
and ask before submitting feedback upstream.

## RPO Trainer

![OEL RPO Trainer start screen](sim/game/assets/OEL_RPO_Trainer.png)

The bundled Pygame trainer turns RPO intuition into a playable progression:
tutorial, coast-relative motion, V-bar and R-bar approaches, close rendezvous,
passively safe inspection, eccentric-orbit lessons, defensive-target tracking,
evasive-target survival, cislunar rendezvous, arcade pursuit, and sandbox.

Install the game extra and launch the level selector:

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: `
  -c constraints/py314.txt ".[game]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe run_simulation.py --doctor
.\.venv\Scripts\python.exe run_game.py
```

macOS or Linux:

```bash
.venv/bin/python -m pip install --only-binary=:all: \
  -c constraints/py314.txt ".[game]"
.venv/bin/python -m pip check
.venv/bin/python run_simulation.py --doctor
.venv/bin/python run_game.py
```

These commands reuse the `.venv` created in the installation section and do
not require activation. If you selected Python 3.10, 3.11, 3.12, or 3.13,
replace `py314.txt` with the matching constraints file. Doctor should report
the trainer capability as available before launch. On Windows, if `py` is
missing, install a supported CPython from python.org with the Python launcher
and reopen PowerShell. If the environment was created with another Python
minor or on another operating system, recreate `.venv` rather than reusing it.

Use Up/Down or W/S to choose a level, Left/Right to change assists, Enter or
Space to launch, and Escape to return to the selector. Training runs can also
write debriefs and recordings under `outputs/`; see
[RPO Trainer](docs/rpo-trainer.md) for controls, debriefs, recording, and
classroom-use notes.

### RPO Duel Beta And Browser Preview

The public export also includes the experimental browser-native
[RPO Duel Beta](web/rpo-duel-prototype/README.md) and a standalone trainer web preview. Duel is a
multiplayer game surface, not a browser port of the deterministic OEL physics
engine. Follow its local/deployment guide and treat the preview as an interface
demonstration rather than simulation evidence.

During supported RPO levels, `O` and `P` swap the RI or RC plot into an
orbit-plane view. In `Bonus Level - Cislunar Rendezvous`, that swapped view is
Moon-centered and shows the target NRHO with the chaser's current position.

## Trust, Limits, And Safety

Orbital Engineering Lab is an independent personal-capacity software project. It
is not an official product, program, or endorsement of the Department of
Defense, Department of the Air Force, United States Space Force, or any other
U.S. Government organization.

The public repository should not contain nonpublic government information,
government-provided code, controlled operational scenarios, classified
information, or customer-controlled technical data. Users are responsible for
their own validation, security review, export-control review, mission
qualification, and compliance obligations before using the software in any
sensitive, commercial, government, or operational context.

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules; loading an untrusted scenario can run
untrusted Python code. For untrusted configs, start with safe validation and the
security docs.

For vulnerability reporting, supply-chain evidence, data handling, export/CUI
boundaries, and incident response, start with:

- [Security Policy](SECURITY.md)
- [Supply Chain And Procurement Baseline](docs/security/supply-chain.md)
- [Data Handling And Boundary Statement](docs/security/data-handling.md)
- [Security Incident Process](docs/security/incident-response.md)

## Public Core And Pro Layer

The public core is useful on its own: deterministic single-run simulation,
public controllers and mission primitives, review artifacts, Python/YAML
interfaces, examples, docs, the bounded two-body Lambert orbit-transfer
planner, the transparent event-driven single-shooting targeter with
authoritative ONP repropagation, deterministic coverage and directed-link
analysis, the Public FSW Authoring Kit, and the RPO trainer.

The Pro layer adds workflow acceleration for teams that need repeatable
analysis at scale: controller benchmarks, general optimization and gain tuning,
Monte Carlo and sensitivity campaigns, curated validation packs, AI-assisted
reports, custom GNC workbench scaffolding, and program-specific integrations.
The private FSWDK adds Controller Bench, tuning, qualification, baseline
promotion, packaged evidence, external-process candidates, and cFS/SIL.

See [Public Core And Pro Boundary](docs/public-vs-pro.md) for the product and
repository boundary. Public examples do not require hosted AI accounts or API
keys.

## Documentation

- [Documentation Index](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Scenario YAML](docs/scenario-yaml.md)
- [CCSDS OPM And OMM Interoperability](docs/ccsds-odm.md)
- [Modular Data IO And Review Exports](docs/data-io.md)
- [Full-State EKF Innovation Recovery](docs/innovation-recovery.md)
- [Event-Driven Trajectory Targeting](docs/trajectory-targeting.md)
- [Coverage And Link Scenario Analysis](docs/coverage-link-scenario-analysis.md)
- [Python API](docs/python-api.md)
- [Product Inventory](docs/product-inventory.md)
- [Public FSW Authoring](docs/fsw-authoring.md)
- [Flagship RIC_PD 10 km Scenario And Validation](docs/validation-ric-pd-10km.md)
- [Review Store](docs/review-store.md)
- [Examples Matrix](docs/examples-matrix.md)
- [Known Limitations](docs/known-limitations.md)

## Source Install Profiles

```bash
python -m pip install .
python -m pip install ".[dev]"
python -m pip install ".[game]"
python -m pip install -c constraints/py314.txt ".[cross-platform]"
python -m pip install ".[ml]"
python -m pip install ".[full]"
```

These profiles apply to source checkouts; managed release installation uses the
signed artifact workflow at the top of this README. For scripted output
inspection, prefer
`python -m sim.review`. The cross-platform profile is the aggregate
dev/game/acceleration/OEL-native-validation profile. ML and `full` remain
separately qualified and do not make a universal cross-platform promise.

## Bug Reports

For non-security bugs, open a GitHub Issue and use the Bug Report template.
Include the OEL version or commit, operating system, Python version, install
profile, exact command, minimal scenario/config or reproduction steps, expected
behavior, actual behavior, and relevant traceback or artifact path.

For simulation-model concerns, include the physical claim, scenario, reference
source or tolerance, and generated evidence. Do not post secrets, API keys,
customer data, CUI, export-controlled data, classified information, or private
generated report packets in public issues.

## Project Layout

- `run_simulation.py` CLI entrypoint for validation and simulation runs
- `run_game.py` Pygame RPO trainer launcher
- `sim/api.py` public programmatic API
- `sim/config/` config schema, fidelity profiles, and plugin validation
- `sim/dynamics/` orbit, attitude, environment, and propagation logic
- `sim/control/` orbit and attitude controllers
- `sim/actuators/` public actuator models
- `sim/sensors/`, `sim/estimation/`, `sim/knowledge/`, and `sim/mission/`
  closed-loop behavior components
- `sim/review/` review-store query API and saved queries
- `examples/` curated runnable configs
- `agents/` public agent instructions, examples, and task cards
- `docs/` user-facing documentation

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).

## Hosted availability

**Closed alpha: access is not publicly available.** Public OEL remains free to
run locally without a Hosted account. The bundled client and package validator
do not grant service access. Only invited operators with a configured profile
can request alpha offers; the alpha uses simulated settlement, not real payments.
