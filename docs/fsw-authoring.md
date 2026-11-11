# Public Flight Software Authoring

The OEL Public FSW Authoring Kit is the dependency-clean path for creating and
exercising a custom Python, C, C++, or Rust complete-stack flight-software
candidate against OEL's typed simulation boundary. It provides editable ADCS and RPO starters,
safe manifest inspection, explicit trusted-import validation, component tests,
one deterministic serial smoke run, and content-bound receipts.

It is an authoring and integration workflow. It is not Controller Bench,
tuning, qualification, certification, hardware readiness, or operational
approval.

## What Is Public

The public kit owns these operations:

| Operation | Effect | Evidence |
| --- | --- | --- |
| `describe` and `doctor` | Read-only | Supported templates, contracts, boundaries, and environment checks |
| `init` | Writes a new candidate | Candidate manifest, editable stack, component test, smoke scenario, scaffold receipt |
| `inspect` | Read-only; does not import candidate code | Normalized identity, hashes, paths, handling metadata |
| `plan` | Read-only | Content-bound work order, effects, resource posture, and approval requirements |
| `validate` | Imports only with explicit trust | Schema, path, direct-import truth-firewall, lifecycle, plugin, and serial-smoke checks |
| `build` | Executes trusted native build scripts | Offline CMake/Cargo build receipt and executable identity |
| `test` | Executes the declared component suite | Bounded pytest, CTest, or Cargo result tied to the exact candidate identity |
| `smoke` | Executes one deterministic serial scenario | Run manifest, ordinary OEL artifacts, and review store |
| `verify-receipt` | Read-only | Current candidate and artifact identity check |

Python `python_stack` and native `native_stack` candidates are accepted. Candidate paths must remain inside
one authorized workspace, symbolic-link inputs are refused, outputs must be new
or empty, component tests are time-bounded, and the workflow does not use the
network or hosted AI.

The local FSWDK is the workflow superset for Controller Bench, comparison
campaigns, tuning, qualification and baseline promotion, packaged review
evidence, cFS/SIL, and program-specific integration. Qualification inputs and
generated release-governance packets may remain private; running a local
workflow does not establish flight qualification or operational approval.

## First Candidate

Activate OEL as described in [Installation](installation.md), then work from an
OEL workspace or the source checkout:

```bash
oel fsw describe
oel fsw doctor
oel fsw init my_adcs --template adcs
oel fsw inspect fsw_candidates/my_adcs/candidate.yaml
oel fsw plan fsw_candidates/my_adcs/candidate.yaml validate
oel fsw validate fsw_candidates/my_adcs/candidate.yaml --trusted-import
oel fsw test fsw_candidates/my_adcs/candidate.yaml
oel fsw smoke fsw_candidates/my_adcs/candidate.yaml
```

Use `--template rpo` for the public RPO starter. The source-tree equivalent is
`python -m sim.fsw_authoring`; the catalog façade also delegates through
`python -m sim.flight_software author`.

`inspect` is the correct first operation for unfamiliar candidate material. It
parses and hashes the candidate without importing or executing its Python
source. `validate --trusted-import`, `test`, and `smoke` do import candidate
code. Only use them for source the operator has reviewed and trusts. Static
direct-import checks enforce the normal truth boundary, but they are not a
sandbox and do not make hostile Python safe.

## Candidate Contract

Python `oel.fsw_authoring.candidate.v1` and native
`oel.fsw_authoring.candidate.v2` manifests declare:

- a stable candidate identity and revision;
- one Python module/class, or native source plus a content-bound executable;
- the public `oel.fsw.boundary.v1` onboard contract;
- a compatible public hardware profile and task period;
- one component-test directory and one deterministic smoke scenario;
- public handling metadata and a bounded intended use.

The candidate hash binds the normalized manifest, candidate source, component
suite, smoke scenario, and contract version. Validation IDs are recomputed
before tests or smoke execution, so a supplied stale ID fails rather than being
silently reused.

Candidate source should consume typed flight-software inputs and compose public
`sim.flight_software` and `sim.gnc` interfaces. It must not import
simulator-owned truth, propagation, sensor, actuator, or runtime internals.
OEL remains authoritative for deterministic physics, execution, review data,
and generated artifacts.

## Evidence And Claims

A passing component suite proves only that the declared tests passed for the
recorded candidate. A passing smoke proves only that the exact candidate ran in
the exact saved OEL scenario and model configuration. Inspect the run's
`index.md`, summary JSON, and `review/run.sqlite` before interpreting behavior.

Receipts deliberately repeat these non-claims:

- results apply only to the exact content-bound inputs;
- the public kit did not perform comparison, tuning, or qualification;
- receipts are not flight qualification, certification, hardware readiness,
  or operational approval.

## Agent And MCP Use

The supported `public_local` MCP profile exposes the same bounded lifecycle as
`oel.fsw.*.v1` tools. Inspection and planning are read-only. Scaffolding and
receipt writes require configured write approval. Candidate import requires a
separate source-trust approval. Component tests and smoke runs require execute
approval. The tools are absent from restricted frontier profiles and never
expose hidden truth or private orchestration.

Agents should plan before an effectful operation, show the operator the source
trust and effect boundary, use a fresh output directory, and summarize saved
OEL evidence rather than inferring qualification from a successful command.

## C, C++, and Rust candidates

Native source authoring uses the same bounded inspect/build/validate/test/smoke
workflow. For example, from the authorized workspace:

```bash
oel fsw doctor
oel fsw init my_native_adcs --language c --template adcs
oel fsw inspect fsw_candidates/my_native_adcs/candidate.yaml
oel fsw plan fsw_candidates/my_native_adcs/candidate.yaml build
oel fsw build fsw_candidates/my_native_adcs/candidate.yaml --trusted-build
oel fsw validate fsw_candidates/my_native_adcs/candidate.yaml
oel fsw test fsw_candidates/my_native_adcs/candidate.yaml
oel fsw smoke fsw_candidates/my_native_adcs/candidate.yaml
```

Use `--language cpp` or `--language rust`, and `--template rpo` for the
corresponding editable starter. All native builds require CMake and a C
compiler; C++ also needs a C++ compiler, Rust also needs Cargo and rustc. The
Rust SDK shares the C runtime. Dependencies are optional for Python-only users;
`doctor` reports the detected executables without installing anything.

Builds execute reviewed build scripts and record compiler/source/binary identity.
Build directories and logs remain beside their receipt; remove them only when
no retained evidence needs the executable. Edits require rebuild/revalidation.
Native validation is non-executing; native tests and smoke execute trusted code.
The generic local stdio adapter, cFS/SIL bridge, and FSWDK verification
workflows are public local code. Hosted infrastructure, restricted program
inputs, and generated qualification/governance evidence retain their private
boundary.

Read the [native stdio contract](contracts/fsw-native-stdio-contract.md) for wire
records, checkpoint declarations, timing, failure behavior, SDK coverage, and
platform qualification limits. Native support currently has macOS arm64
implementation evidence; Windows/Linux require native acceptance runs.

Public-local MCP scaffolding accepts the optional `language` field. Native
builds use `oel.fsw.build_candidate.v1`, with independent source-trust and execute
approvals. Existing Python commands and candidate-v1 manifests retain their
behavior. The native build tool exposes fixed local toolchain operations,
without a general-purpose shell or remote deployment facility.
