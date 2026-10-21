# Contributing

Thanks for taking a look at Orbital Engineering Lab.

## How Contributions Reach A Release

Public issues and pull requests are welcome. For a substantial feature or API
change, open an issue first so maintainers can discuss scope, compatibility,
and ongoing work before you invest in an implementation. Small fixes and
documentation improvements can be submitted directly.

OEL's public repository is generated from a separate development repository.
Maintainers review public contributions, integrate accepted work there, and
validate it before a later public export. A contribution may be adapted,
combined with broader work, or replaced by another implementation rather than
merged directly from its original pull request. Submission does not guarantee
acceptance or a release date. Contributors do not need access to the development
repository; public discussion and release references remain here.

When work lands through a replacement implementation, maintainers will link
the original issue or pull request and explain its disposition. Accepted
contributions receive appropriate public credit in release notes and, where
applicable, commit authorship or coauthor attribution. Original authorship and
required notices are preserved for incorporated material. An independently
written implementation inspired by a proposal is credited as such.

## Licensing And Submission Rights

Unless separately agreed in writing, contributions intentionally submitted for
inclusion in OEL are licensed under the [Apache License 2.0](LICENSE.txt),
consistent with its contribution provision. Contributors retain copyright in
their contributions; submission does not assign copyright to the maintainer.

OEL includes public and private/Pro development. Apache-2.0 permits commercial
reuse, including use in proprietary products, subject to its license and notice
requirements. Contributions do not by themselves create an entitlement to
payment, royalties, equity, employment, or a maintainer role. Any different
arrangement must be agreed separately in writing.

Only submit material you have the right to license. Obtain any necessary
employer or other rights-holder permission. Identify incorporated third-party
code, data, assets, and their sources and licenses; retain required notices.
Do not submit confidential or restricted material. If terms are unclear or
differ from the repository license, discuss them before submission.

## Developer Certificate Of Origin

For new contributions submitted after this policy is published, each
contribution commit must include a sign-off certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
Read the certificate before signing. A sign-off records your certification of
submission rights; it does not assign copyright or replace the project license.

Use your name and an email address you are willing to publish in commit history:

```bash
git commit -s
```

This adds a trailer in this form:

```text
Signed-off-by: Your Name <your-public-email@example.com>
```

A GitHub-provided no-reply address associated with your account is acceptable.
Sign-offs are public and retained in repository history. Sign only your own
certification; maintainers must not invent or add another person's sign-off.
Earlier submissions are reviewed under their original terms; this policy does
not imply that an existing contribution was signed off. Maintainers review
submission rights, third-party notices, and sign-offs before accepting work.

## Development Setup

Use Python 3.10 through 3.14. Python 3.14 is recommended for a new environment;
use the matching `constraints/py3XX.txt` file when reproducing a qualified
dependency graph. See [Installing OEL](docs/installation.md) for the complete
cross-platform command convention.

Windows PowerShell:

```powershell
py --list
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe run_simulation.py --doctor
```

macOS or Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[dev]"
python run_simulation.py --doctor
```

PowerShell activation is optional. Run `.\.venv\Scripts\Activate.ps1` when
local policy permits it; otherwise keep using the explicit
`.\.venv\Scripts\python.exe` path. Commands below assume an activated
environment and therefore use `python`.

## Reporting Bugs

For non-security bugs, open a GitHub Issue in the public repository and use the
Bug Report template. Maintainers monitor repository issue notifications.

Good bug reports include:

- the OEL version, release, branch, or commit SHA,
- operating system and Python version,
- install profile, such as core, dev, gui, game, ml, or full,
- the exact command that failed,
- the smallest scenario YAML, code snippet, or game/trainer step that
  reproduces the issue,
- expected behavior and actual behavior,
- relevant traceback, log excerpt, or output artifact path,
- whether the issue is repeatable from a clean checkout.

For simulation-model or validation concerns, also include the physical claim
being questioned, the scenario/config used, any external reference or expected
tolerance, and the generated artifacts that support the concern.

Do not post secrets, API keys, customer data, CUI, export-controlled data,
classified information, or private generated report packets in public issues.
For suspected vulnerabilities or sensitive data exposure, follow `SECURITY.md`
instead of opening a public issue.

## Checks

Run the focused public checks before opening a pull request:

```bash
python -m ruff check <changed-python-files>
python -m ruff format --check <changed-python-files>
python -m pytest -q sim/tests/test_scenario_yaml_config.py sim/tests/test_utils_io.py sim/tests/test_api.py sim/tests/test_master_simulator.py
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

Ruff is being rolled in incrementally. Use the changed-file commands for normal
PRs until the existing project baseline has been cleaned enough for
`python -m ruff check .` and `python -m ruff format --check .` to become hard
repo-wide gates.

To format intentionally:

```bash
python -m ruff format .
```

To preview local generated artifacts and caches that can be cleaned:

```bash
python tools/clean_local_artifacts.py
```

Apply the cleanup only after reviewing the dry run:

```bash
python tools/clean_local_artifacts.py --apply
```

## Contribution Scope

Good public-core contributions include:

- simulator correctness fixes
- scenario YAML usability improvements
- reference controllers and examples
- documentation and onboarding improvements
- small validation and smoke-test coverage

Product workflows such as controller benchmarking, optimization, Monte Carlo
campaign orchestration, sensitivity studies, and campaign reporting are kept in
the pro layer.

## Style

- Keep changes scoped and testable.
- Prefer existing module patterns over new abstractions.
- Add or update tests when behavior changes.
- Avoid committing generated outputs, local data, or large ephemeris files.
