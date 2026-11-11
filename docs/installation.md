# Installing OEL On Windows, macOS, And Linux

This is the authoritative managed- and source-installation guide for Orbital
Engineering Lab. OEL supports CPython 3.10 through 3.14 on its declared Windows,
macOS, and Linux compatibility targets. Python 3.14 is recommended.

Python 3.10 retains legacy functional compatibility with Pillow 12.3.0, whose
known advisories, together with the legacy filelock used by ML/full profiles,
block dependency security qualification, including core, media
and full profiles. Use Python 3.11 or newer with Pillow 13 for the current
dependency security baseline.
No Python 3.10 managed bundle or clean-audit claim is provided. See
[Compatibility](compatibility.md) and [Supply chain](security/supply-chain.md).

## Managed Installation

The v1.0.0 managed bundle is qualified for macOS arm64 with CPython 3.11.
The signed bootstrap selects that installed Python minor, downloads the
versioned GitHub bundle, verifies its manifest and every dependency wheel,
and installs without consulting a package index. Install CPython 3.11 first
if it is unavailable. The engine's broader Python compatibility and the
Windows/Linux native diagnostic results do not qualify this managed bundle
for those other targets. Linux, Windows, and Intel Mac users should follow
[Developer Source Installation](#developer-source-installation) and
[Building public native source](native-source-build.md).

For an existing v0.36.0 managed installation, the signed-inventory trust
boundary is already established and `oel update install latest` can perform the
v1.0.0 update. For a v0.31.0 through v0.35.0 installation, rerun the official
installer below with the same data and config roots before updating. Use
`--profile <profile>` to retain a nondefault dependency profile. The installer
verifies the signed source/runtime inventories and publishes the trusted
launcher. Earlier updaters predate this provisioning contract. Workspace files
and pins remain unchanged. Older engines remain on disk for inspection;
releases without signed inventories cannot serve as authenticated rollback
targets under the new launcher.

After this migration, the v1.0 managed launcher supports the ordinary
`oel update install latest` and explicit activation workflow.

An official public release publishes a small `install.sh`, `install.ps1`,
signed `release-manifest.json`, source artifact, and offline bundle as immutable
assets on the public OEL GitHub release. The stable convenience URL below
resolves to the installer asset on the latest promoted public release. Never
pipe an uninspected URL from a third party. OEL v1.0 has one local edition.
Install profiles select dependencies; they do not activate features.

POSIX:

```bash
curl --proto '=https' --tlsv1.2 -fsSLo /tmp/oel-install.sh \
  https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/install.sh
less /tmp/oel-install.sh
sh /tmp/oel-install.sh
```

After verifying the release host and script through the documented channel,
the equivalent convenience form is:

```bash
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/install.sh | sh
```

The download-inspect-execute form remains preferred for first use because it
makes the small trust bootstrap visible before execution.

Windows PowerShell:

```powershell
Invoke-WebRequest https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/install.ps1 -OutFile $env:TEMP\oel-install.ps1
Get-Content $env:TEMP\oel-install.ps1
& $env:TEMP\oel-install.ps1
```

The bootstrap requires an existing supported CPython. It verifies the embedded
bootstrap digest, signed release manifest, artifact size and SHA-256, and safe
archive shape before importing release code. It installs immutable versions
side by side under platform-native application data and writes a stable
launcher. The stable launcher runs an external stdlib-only verifier before it
imports the selected managed version; source/runtime inventories reject
pre-existing Python bytecode caches, and provisioning uses `pip --no-compile`.
Official runtime provisioning passes each signed wheel's authenticated SHA-256
to `pip --require-hashes`, independently of source constraints. CPython 3.14's
optional `𝜋thon` alias is bound to the interpreter provider in the signed
runtime inventory and verified along with the ordinary interpreter aliases.
The rendered installer also records the official signed-channel URL
beside OEL's trusted release keys, so future update checks do not require users
to copy URLs or key paths. The channel URL locates signed metadata; it is not a
replacement trust root. The bootstrap does not edit a workspace.

The standard installer includes RPO Trainer and adds a per-user launcher: `RPO Trainer.app` in `~/Applications` on macOS, `RPO Trainer.exe` with a Start Menu shortcut on Windows, or an application-menu entry on Linux. Double-click it to open the Trainer. The launcher follows the active managed version after updates or rollback. Existing supported Python is still required. Set `OEL_INSTALL_PROFILE=core` before running the shell installer (PowerShell: `$env:OEL_INSTALL_PROFILE="core"`) for a command-line-only installation. The Python bootstrap also accepts `--profile core`.

You can also launch the Trainer with `oel trainer`. Saved Trainer progress remains in your user account. Updates inherit the active dependency profile unless you explicitly choose another one. macOS and Windows desktop launch logs are saved under the managed data directory at `logs/trainer.log`.

To remove desktop integration, delete `~/Applications/RPO Trainer.app` on macOS, the RPO Trainer Start Menu shortcut on Windows, or `$XDG_DATA_HOME/applications/oel-rpo-trainer.desktop` (normally `~/.local/share/applications/oel-rpo-trainer.desktop`) on Linux. This does not remove OEL or saved progress. Rerunning the installer recreates the shortcut.

After installation:

On macOS or Linux, the launcher is installed at `~/.local/bin/oel`. Reopen the
shell or ensure that directory is on `PATH` before using the short `oel`
command, for example `export PATH="$HOME/.local/bin:$PATH"`. Until then, invoke
`~/.local/bin/oel` explicitly.

```text
oel update status --full
oel update check
oel doctor
oel workspace init path/to/my-oel-workspace
oel --workspace path/to/my-oel-workspace sim --quickstart --validate-only
oel --workspace path/to/my-oel-workspace sim --quickstart
oel --workspace path/to/my-oel-workspace review outputs/quickstart_5min --saved-query run_metadata
```

### Declared Host Admission Matrix

| Host | Architecture | Doctor admission | Maintained evidence row |
| --- | --- | --- | --- |
| Windows 11 or Server 2022 | x64 | Windows 11/Server 2022 x64 | Server 2022 x64 diagnostic |
| Ubuntu 22.04 or 24.04 | x64 | Ubuntu 22.04/24.04 x64 | Ubuntu 22.04 x64 |
| macOS 14 or newer | arm64 or x64 | macOS 14+ arm64/x64 | macOS 15 arm64 and Intel diagnostics |

On other Linux x64/arm64 distributions, doctor warns about missing host
qualification and attempts functional runtime checks. Passing those checks does
not qualify that host or admit a signed bundle built for another host. Other
unsupported OS/architecture combinations remain failures.

Admission and package metadata are not release evidence. A claim for a row
requires its retained packet; external integrations remain separately
qualified.

See [Updating OEL](updating.md), [OEL Workspaces](workspaces.md), and
[Offline Installation](offline-installation.md). Managed installation and
workspace adoption are separate operations by design.

## Developer Source Installation

Source installations require native wheels built from the bundled Rust crates.
The platform blocks below create the environment, build both wheels, and install
OEL with the matching constraints. See
[Building public native source](native-source-build.md) for native tests and
offline build guidance.
The signed managed bundle already contains its qualified native wheels.

### Get The Source

Clone the public repository, or start in the root of an existing OEL checkout:

```text
https://github.com/adamcohen8/orbital-engineering-lab.git
```

The checkout directory may contain spaces. Run the commands below from the
directory containing `pyproject.toml` and `run_simulation.py`.

The Python distribution is named `orbital-engineering-lab`. Existing checkout
URLs and `oel` command names remain usable during the repository transition.

Source installation requires Rust/Cargo 1.85 or newer and Maturin. Both native
wheels must be built locally before installing OEL; they are not on PyPI.

## Windows PowerShell

List the Python installations known to the Windows Python launcher:

```powershell
py --list
```

Older launcher versions use `py -0p` for the same inventory. Select a supported
minor, then create, install, diagnose, and run OEL with that same interpreter:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "maturin>=1.10,<2"
.\.venv\Scripts\python.exe -m maturin build --release --locked --manifest-path rust/oel-orbit/Cargo.toml --interpreter .\.venv\Scripts\python.exe --out dist/native
.\.venv\Scripts\python.exe -m maturin build --release --locked --manifest-path rust/oel-game/Cargo.toml --interpreter .\.venv\Scripts\python.exe --out dist/native
.\.venv\Scripts\python.exe -m pip install -c constraints\py314.txt --find-links dist/native ".[dev]"
.\.venv\Scripts\python.exe run_simulation.py --doctor
.\.venv\Scripts\python.exe run_simulation.py --quickstart
```

These commands do not require virtual-environment activation. This avoids
PowerShell execution-policy problems and guarantees that installation and
execution use the same interpreter.

## macOS Or Linux (POSIX Shell)

Select a supported interpreter installed on the host:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "maturin>=1.10,<2"
.venv/bin/python -m maturin build --release --locked --manifest-path rust/oel-orbit/Cargo.toml --interpreter .venv/bin/python --out dist/native
.venv/bin/python -m maturin build --release --locked --manifest-path rust/oel-game/Cargo.toml --interpreter .venv/bin/python --out dist/native
.venv/bin/python -m pip install -c constraints/py314.txt --find-links dist/native ".[dev]"
.venv/bin/python run_simulation.py --doctor
.venv/bin/python run_simulation.py --quickstart
```

Use these commands from Bash, Zsh, or another POSIX-compatible shell. Do not use
them unchanged in PowerShell; the virtual-environment interpreter path is
different.

## Choose Another Supported Python Minor

Replace `3.14` consistently with `3.10`, `3.11`, `3.12`, or `3.13`. On Windows,
for example:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python --version
```

OEL maintains one approved constraints file per supported minor:

| Python | Constraints file |
| --- | --- |
| 3.10 | `constraints/py310.txt` |
| 3.11 | `constraints/py311.txt` |
| 3.12 | `constraints/py312.txt` |
| 3.13 | `constraints/py313.txt` |
| 3.14 | `constraints/py314.txt` |

Use the matching file for the first OEL install and subsequent profile changes.
When choosing another Python minor, replace `py314.txt` in every install command
with its matching file. The commands below reuse the native wheels built above;
rebuild them if you change host or interpreter compatibility.

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: `
  -c constraints/py314.txt --find-links dist/native ".[cross-platform]"
.\.venv\Scripts\python.exe -m pip check
```

POSIX:

```bash
.venv/bin/python -m pip install --only-binary=:all: \
  -c constraints/py314.txt --find-links dist/native ".[cross-platform]"
.venv/bin/python -m pip check
```

Do not use a constraints file for a different Python minor. Constraints are
approved reproducibility inputs, not universal lockfiles for every optional
external integration.

## Portable Command Convention

Onboarding, installation, classroom, and troubleshooting material shows
explicit PowerShell and POSIX commands. Other OEL documentation uses `python`
after the virtual environment has been activated, or links back to this guide.

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

Activate it in a POSIX shell:

```bash
source .venv/bin/activate
python --version
```

Activation is optional. If local policy blocks `Activate.ps1`, keep using
`.\.venv\Scripts\python.exe` directly. After activation, the following commands
are portable across PowerShell, macOS, and Linux:

```text
python run_simulation.py --doctor
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
python -m sim.review outputs/quickstart_5min --saved-query run_metadata
```

When copying a command from a general OEL document, first activate the
environment or replace its leading `python` with the explicit interpreter path
for the current platform.

## Native numeric runtime

Version 0.36.0 requires matching `oel-rust-orbit` >=0.18.3,<0.19 and `oel-rust-game`
0.5.x wheels for core installation and runtime-built flight-software stacks. Source
installations must first install wheels built for their host from the native
crates; signed offline bundles retain the exact qualified wheel inventory.
The public repository includes the source and build inputs for both native
packages; see [building native source](native-source-build.md).
Rust is the default numeric engine. Explicit `numeric_backend: python` and
Trainer `--backend python` retain reference execution.

## Install Profiles

Choose only the profile required by the workflow:

| Command | Purpose |
| --- | --- |
| `python -m pip install .` | Core CLI, YAML/API runtime, plotting, and review store |
| `python -m pip install ".[acquisition]"` | Experimental local data connections with an OS credential backend; public downloads and local registration also work without the extra |
| `python -m pip install ".[dev]"` | Core plus tests and Ruff |
| `python -m pip install ".[game]"` | RPO trainer, native Rust backend, and media dependencies |
| `python -m pip install ".[accel]"` | Separately qualified Numba acceleration |
| `python -m pip install ".[validation]"` | OEL-native validation dependencies |
| `python -m pip install ".[cross-platform]"` | Aggregate compatibility-acceptance profile |
| `python -m pip install ".[ml]"` | Separately qualified ML dependencies |
| `python -m pip install ".[full]"` | Convenience union; not a universal support claim |

See [Compatibility And Install Profiles](compatibility.md) before using
acceleration, ML, `full`, or an external integration as support evidence.

### CPU-only Linux ML environments

For a Linux host without an NVIDIA GPU, install the official CPU build of the
approved Torch version before adding `.[ml]` or `.[full]`. The default PyPI
Torch build can pull CUDA packages that are unnecessary on a CPU-only host;
some of those packages can also fail platform checks on Linux ARM64.

From the source root, after building the native wheels, use:

```bash
.venv/bin/python -m pip install "torch==2.13.0+cpu" \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -c constraints/py312.txt \
  --find-links dist/native ".[full]"
.venv/bin/python -m pip check
```

This example uses Python 3.12; select the matching constraints file for another
supported minor. Keep the CPU index limited to the Torch installation command.
The installed CPU build satisfies OEL's bounded Torch requirement. It does not
qualify CUDA, model inference, or an additional operating-system/architecture
target. See the [official PyTorch wheel index](https://download.pytorch.org/whl/cpu/torch/).

## Classroom Or Restricted Environment Check

Only execute scenario YAML from a trusted source. For an unfamiliar file, use
safe validation first.

PowerShell:

```powershell
.\.venv\Scripts\python.exe run_simulation.py --config <path> --safe-validate
.\.venv\Scripts\python.exe run_simulation.py --config <path> --sealed-mode --validate-only
```

POSIX:

```bash
.venv/bin/python run_simulation.py --config <path> --safe-validate
.venv/bin/python run_simulation.py --config <path> --sealed-mode --validate-only
```

Safe validation is an inspection boundary, not permission to execute an
untrusted config. Sealed mode restricts plugins, external paths, hosted AI,
networked integrations, and high-detail outputs unless explicitly allowed.

## Troubleshooting A Failed Installation

Run the commands for the platform where the failure occurred.

PowerShell:

```powershell
py --list
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe run_simulation.py --doctor
```

POSIX:

```bash
.venv/bin/python --version
.venv/bin/python -m pip --version
.venv/bin/python -m pip check
.venv/bin/python run_simulation.py --doctor
```

Common recovery rules:

- If `py` is not recognized on Windows, install a supported CPython from
  python.org with the Python launcher, reopen PowerShell, and run `py --list`.
- If `python3.14` is not found on macOS or Linux, install a supported Python
  minor and use that minor consistently.
- If `.venv` was created by another interpreter, remove it through your normal
  file-management workflow and create a new one; do not reuse it across Python
  minors or operating systems.
- If activation is blocked, use the explicit interpreter path. Activation is
  never required for OEL.
- If binary dependency installation fails, confirm the OS, architecture,
  Python minor, matching constraints file, and install profile shown by
  `--doctor`.
- Do not post secrets, customer inputs, controlled data, or private report
  packets in a public bug report.

For the guided simulation walkthrough, continue to [Quickstart](quickstart.md).
For support boundaries and evidence requirements, read
[Compatibility And Install Profiles](compatibility.md).

### Updating from RPO Trainer

Official managed public installations check their configured release channel in the
background when the level selector opens. The start screen does not check or show
updates. A newer signed release displays an update notice in the selector footer.
Click it or press Ctrl+U (also Cmd+U on macOS) to install and relaunch Trainer.
The check alone never downloads or installs a release.

Installation preserves the active dependency profile and uses signed-channel,
signed-manifest, and artifact verification before activation. Levels cannot launch
while installation is in progress. If installation fails, the current Trainer stays
open and offers retry; details are recorded in the managed cache's
`trainer-update-error.log`. Saved progress and settings are preserved. If relaunch
fails, the selector asks you to close and reopen Trainer.

This integration is for official managed public desktop installations. Source
checkouts, developer installations, and the web preview do not
automatically check or install updates from the game.
