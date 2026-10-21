# Compatibility And Install Profiles

This page is the authoritative OEL compatibility and dependency-profile
contract. Package metadata accepts CPython 3.10 through 3.14 and rejects 3.15
or newer until the new-Python admission gate has passed.

For source installation, interpreter selection, and explicit Windows
PowerShell and macOS/Linux commands, use
[Installing OEL](installation.md). General commands on this page use `python`
after the virtual environment has been activated.

## Support Terms

- **Functional compatibility** means installation, doctor, validation,
  deterministic execution, review queries, plotting, trainer smoke tests, and
  documented artifact checks pass for the applicable profile.
- **Security-supported baseline** means the interpreter and selected dependency
  profile still receive upstream security maintenance and pass OEL's
  dependency-audit gate.
- **Qualified integration** means only the recorded operating system,
  architecture, Python, dependency, and external-tool combination is
  supported.

Python 3.10 is a functional compatibility target. When upstream security
maintenance ends, it becomes a legacy functional tier and is no longer a
security-supported procurement baseline. Python lifecycle status is governed by
the [CPython version table](https://devguide.python.org/versions/), not by OEL
package metadata alone.

Pillow 13.0.0 requires Python 3.11 or newer. Python 3.11-3.14 core and media
profiles use Pillow 13; Python 3.10 retains Pillow 12.3.0 for functional compatibility.
That legacy pin has known advisories with no upstream 3.10 backport, so 3.10
dependency security qualification (including core/media/full) is blocked.
The doctor warns and recommends Python 3.11+; there is no 3.10 managed bundle or clean dependency-audit claim.
See the [supply-chain security findings](security/supply-chain.md#pillow-security-baseline).

The doctor records the following upstream security-maintenance end months for
the currently admitted minors:

| Python | OEL functional status | Recorded upstream security maintenance |
| --- | --- | --- |
| 3.10 | Supported | Through 2026-10 |
| 3.11 | Supported | Through 2027-10 |
| 3.12 | Supported | Through 2028-10 |
| 3.13 | Supported | Through 2029-10 |
| 3.14 | Supported | Through 2030-10 |

After a recorded end month, `--doctor` labels that interpreter a functional
legacy tier without silently changing the package's bounded functional range.
Release dependency-audit evidence remains required even while the interpreter
itself is receiving upstream security fixes.

## Compatibility Matrix

The exact declared host admission matrix is Windows 11 or Server 2022 x64,
Ubuntu 22.04 or 24.04 x64, and macOS 14+ on arm64 or x64. Maintained evidence
rows are Windows Server 2022 x64, Ubuntu 22.04 x64, and macOS 15 arm64/Intel.
These are distinct from package admission and from separately qualified
external integrations.

| Profile | Python | Windows | Linux | macOS |
| --- | --- | --- | --- | --- |
| Core CLI/YAML/API/review | 3.10-3.14 | Supported matrix target | Supported matrix target | Supported matrix target |
| Dev/test | 3.10-3.14 | Supported matrix target | Supported matrix target | Supported matrix target |
| Trainer | 3.10 functional only; 3.11-3.14 patched dependency baseline | Supported matrix target | Supported matrix target | Supported matrix target |
| Numba acceleration | 3.10-3.14 | Where approved constraints resolve | Where approved constraints resolve | Where approved constraints resolve |
| OEL-native validation | 3.10-3.14 | Supported matrix target | Supported matrix target | Supported matrix target |
| ML | Qualified combinations only | No universal claim | Qualified lanes only | Qualified lanes only |
| MATLAB/cFS/Orekit/GPU | Qualified combinations only | No universal claim | Qualified lanes only | Qualified lanes only |

“Supported matrix target” identifies the declared compatibility program. A
release claim for an individual row requires retained compatibility evidence
from that row; package metadata and resolver success alone are not runtime or
physics evidence.

### Historical v0.23.1 Release Qualification

This retained historical packet is not qualification evidence for the current
source checkout. The `v0.23.1` release source completed clean local installation and runtime
smoke checks on macOS 15 arm64 with each supported Python minor, 3.10 through
3.14. Boundary-minor full regression and acceleration/validation checks, the
blocking Python 3.11 full suite, the authoritative compatibility acceptance
packet, private merge check, and generated-public release rehearsal also
passed on that host.

This local evidence does not establish Windows, Linux, or Intel macOS runtime
support by extrapolation. Publish or procurement material should name the exact
retained packets for every row it claims. Controlled local environments are the
preferred evidence source.

`.github/workflows/compatibility.yml` is an advisory escape hatch for the two
environments unavailable to the maintainer locally: Windows x64 and Intel
macOS. It never runs automatically; explicit manual dispatch can select Windows
or Intel macOS and one Python minor. Each run performs only wheel installation
plus the platform smoke. It has no pull-request trigger, schedule, matrix, full
regression, authoritative release acceptance, or dependency audit. The latter
gates remain local and release-blocking through
`tools/release_public.py --candidate`.

The smoke collector distinguishes three evidence classes:

- `local-diagnostic`, the default for local investigation and not a support
  claim;
- `github-hosted-automation`, accepted only when complete GitHub Actions and
  runner provenance is present; and
- `controlled-windows-11-desktop`, accepted only with the documented host and
  manual desktop attestations.

Audit JSON is accepted only when it covers every installed third-party
distribution, matches every audited version, and contains no unresolved
vulnerability records. The locally installed first-party
`orbital-engineering-lab` distribution may be absent because `pip-audit` does
not query the local project itself. An audit file's existence alone is not a
passing security gate.

An untested macOS release needs retained controlled-machine evidence before it
can be claimed for a release.

## Install Profiles

Use the constraint file matching the interpreter minor for reproducible
compatibility work. For Python 3.11:

```bash
python -m pip install -c constraints/py311.txt ".[cross-platform]"
```

| Profile | Contents | Qualification |
| --- | --- | --- |
| core (`.`) | CLI, YAML/API runtime, NumPy/SciPy, plotting, review store | Cross-platform core |
| `.[dev]` | Core plus pytest and Ruff | Cross-platform development/test |
| `.[game]` | Core plus trainer graphics/media dependencies | Cross-platform trainer |
| `.[accel]` | Core plus Numba/llvmlite | Version-specific constraints required |
| `.[validation]` | Core plus the OEL-native SGP4 reference dependency | Cross-platform validation |
| `.[cross-platform]` | Union of dev, game, OEL-native validation, and acceleration where binary wheels are published | Aggregate cross-platform acceptance profile |
| `.[ml]` | Gymnasium, Torch, and ML support dependencies | Separately qualified combinations only |
| `.[full]` | Union of optional local capabilities, including ML | Convenience profile; not a universal support claim |

Python 3.10 through 3.13 use the upstream `pygame` distribution. Python 3.14
uses `pygame-ce`, which provides the same `pygame` import surface and publishes
the required Python 3.14 wheels.

Numba and llvmlite do not currently publish Intel macOS wheels for the approved
versions. The aggregate `cross-platform` profile therefore omits Numba on Intel
macOS; deterministic serial execution remains available. The dedicated
`.[accel]` and convenience `.[full]` profiles retain Numba as an explicit,
separately qualified capability and are not universal Intel macOS profiles.

## SGP4 Dependency Transition Gate

The approved `sgp4` dependency is compared against every checked-in
`python-sgp4 2.23` native-TEME fixture before the approved version changes.
The compatibility gate permits no more than `1e-8 km` absolute error in any
position component and `2e-12 km/s` in any velocity component. These limits
cover the observed last-bit differences between supported package builds while
remaining far below OEL's meter and millimeter-per-second reference-validation
thresholds.

The OGP validation owner maintains this gate. Its source evidence is the
near-Earth fixture under `validation/data/sgp4_reference/` and every
near-Earth/deep-space fixture under `validation/data/ogp_reference/`.

## Constraints And Evidence

`constraints/py310.txt` through `constraints/py314.txt` select the approved
direct and transitive versions for the cross-platform profile. They are not
hash-locked, OS-specific release lockfiles. Each CI or release row must retain:

- the pip installation report;
- `python-freeze.txt`;
- `pip-check.txt`;
- `wheel-inventory.json`, including source URLs and wheel tags;
- the CycloneDX SBOM;
- the dependency-audit result; and
- the SHA-256 digest of the applied constraints file.

Core installation must resolve binary wheels without a compiler. ML and
external integrations remain outside the universal core claim and may require
platform-specific overlays.

## Operating-System Portability Contract

OEL's universal runtime uses `pathlib` paths, argument-list subprocess calls,
spawn-serializable module-level worker targets, and a shared platform helper
for native folder opening and host resource telemetry. Windows folder opening
uses the shell API directly rather than command-string quoting. Review-store
connections use encoded file URIs, so ordinary workspace names containing
spaces (including `Orbital Engineering Lab`) are supported.

CI forces Matplotlib's non-interactive `Agg` backend. Trainer acceptance uses
Pygame's dummy video and audio drivers when no desktop display is available.
These headless settings exercise rendering initialization; they do not claim
that a CI runner represents the performance or input behavior of a physical
desktop.

Process-pool object stepping uses Windows' `spawn` context and requires
module-level worker targets plus serializable engine snapshots and step
messages. If process transport is unavailable under the automatic execution
policy, OEL falls back to deterministic serial stepping before applying the
failed worker step. An explicitly requested process-pool backend still reports
the transport failure instead of silently changing execution policy.

MATLAB, cFS, Orekit, and GPU workflows are OS-qualified integrations, not
universal core capabilities. Their support requires evidence for the recorded
host OS, architecture, Python version, dependency set, and external-tool
version. The core compatibility matrix must not be read as a claim that those
external tools install or execute on every supported OEL host.

## Doctor And Recovery

Run doctor with the interpreter you intend to use:

```bash
python run_simulation.py --doctor
```

Doctor executes its interpreter, operating-system, architecture, package
metadata, and dependency-range checks before importing the scientific runtime.
It reports the active executable, resolved package versions, `pip check`
result, detected install profile, and exact availability of the core, trainer,
acceleration, validation, development, and separately qualified ML
capabilities. Quickstart validation runs afterward in an isolated child
process, so a broken NumPy, SciPy, YAML, or plotting installation cannot prevent
the recovery report from being printed.

An interpreter outside `>=3.10,<3.15` fails with that exact bounded range.
Doctor recommends Python 3.14 for recovery rather than emitting an unbounded
“Python 3.10+” instruction. Recovery commands use `py` and
`.\.venv\Scripts\python.exe` on Windows, and `python3.x` plus
`python` on Linux and macOS. The recommended aggregate repair install
is `.[cross-platform]` through the constraint file matching the selected Python
minor.

## Python And NumPy Admission Gates

OEL's approved dependency graphs use NumPy 2. A supported Python minor is
admitted only after the following checks pass with that minor's constraint
file:

1. install the `cross-platform` profile from binary wheels and run
   `python -m pip check`;
2. import the public runtime, run `--doctor`, and collect the complete test
   suite;
3. pass the fast, regression, and compiled marker lanes;
4. pass the NumPy 2 migration lint rule (`NPY201`) and the dtype, scalar
   promotion, and platform-integer compatibility contracts;
5. compare deterministic scenario evidence with the approved baseline; and
6. pass the applicable OGP/SGP4 validation and plotting artifact checks.

The OEL wheel is pure Python (`py3-none-any`), so OEL itself has no compiled
NumPy ABI boundary. Numba, SciPy, Matplotlib, Pillow, pygame, and SGP4 remain
third-party binary boundaries and must resolve from wheels for the exact
operating-system, architecture, and Python row being claimed.

Run the source admission checks from a constrained development environment:

```bash
python -m ruff check .
python -m pytest -q sim/tests/test_compatibility_metadata.py \
  sim/tests/test_numpy2_compatibility.py
python -m pytest -q -m fast
python -m pytest -q -m \
  "regression and not fast and not compiled and not slow and not external"
OEL_ACCELERATION=auto python -m pytest -q -m compiled
```

Python 3.14 is an independent admission target. Successful Python 3.13
execution or a dependency resolver result is not evidence that Python 3.14 is
supported.

Deterministic evidence from the same approved dependency graph is expected to
match across supported Python minors. When a numerical dependency graph
changes, compare physics columns numerically at the owning validation
tolerances rather than comparing SQLite files or JSON bytes. Record the
largest observed delta and rerun the owning external or checked-in reference
suite before accepting the new graph.

OEL's own release version is retained as provenance but normalized out of the
physics comparison, and it is not treated as a third-party dependency-graph
change. For a changed third-party graph, an exact mismatch closes only when all
named owner suites are present and passing in the candidate acceptance packet.

Release maintainers retain one authoritative compatibility packet that
validates every acceptance scenario before execution and records normalized
scenario evidence, read-only review results, model and frame provenance,
artifact inventories, and the applicable model-owned validation-suite results.
A quickstart-only packet is runtime and installation evidence, not physics
validation.

## Controlled Windows 11 Desktop Gate

Windows Server 2022 hosted evidence is not Windows 11 desktop evidence. Before
a release claims Windows 11 x64 desktop support, repeat the compatibility smoke
on a controlled Windows 11 x64 laptop and retain a separate
`controlled-windows-11-desktop` packet.

From PowerShell, create a clean virtual environment with the target Python
minor, install the matching constrained profile from wheels with a pip report,
and run `pip-audit` to JSON:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: `
  -c constraints/py314.txt ".[cross-platform]" `
  --report outputs/desktop-acceptance/pip-install-report.json
.\.venv\Scripts\python.exe -m pip install "pip-audit>=2.9,<3"
.\.venv\Scripts\python.exe -m pip_audit --format json `
  --output outputs/desktop-acceptance/pip-audit.json
```

Then run:

```powershell
.\.venv\Scripts\python.exe tools/run_compatibility_smoke.py `
  --constraints constraints/py314.txt `
  --install-report outputs/desktop-acceptance/pip-install-report.json `
  --audit-result outputs/desktop-acceptance/pip-audit.json `
  --output-dir outputs/desktop-acceptance `
  --acceptance-class controlled-windows-11-desktop `
  --expected-system Windows `
  --expected-machine AMD64 `
  --expected-acceleration available `
  --desktop-attestation desktop-attestation.json
```

The attestation is deliberately separate from automation. An operator must run
`.\.venv\Scripts\python.exe run_simulation.py --quickstart --open-output`,
confirm that Explorer opens the folder whose path may contain spaces, launch
`.\.venv\Scripts\python.exe run_game.py`, and confirm native display rendering
and keyboard input. The JSON file must contain these booleans:

```json
{
  "native_folder_open_verified": true,
  "trainer_window_verified": true,
  "keyboard_input_verified": true,
  "display_rendering_verified": true
}
```

The collector rejects this evidence class on a host that does not identify
itself as Windows 11 x64, and it rejects missing or false attestations. The
hosted and controlled-desktop packets must not be substituted for one another.
