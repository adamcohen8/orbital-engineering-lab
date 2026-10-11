# Supply Chain And Procurement Baseline

This page is the OEL baseline for software supply-chain review. It is intended
to give evaluators a clear starting packet, not to replace a customer's own
security, legal, export-control, or mission-assurance process.

## Supported Versions

- Public releases: security fixes target the current public release line,
  `v0.33.1` once published. During release preparation, `v0.33.0` remains
  the published support target.
- Private/Pro releases: security fixes target the active customer-supported
  release line or pilot branch named in the agreement.
- Declared Python compatibility range: Python 3.10 through 3.14. Functional and
  security support are separate; see `docs/compatibility.md` and retain
  acceptance evidence for the exact interpreter used in an evaluation.
- Operating-system target: Windows, Linux, and macOS for the cross-platform
  profiles. A release claim for a specific OS/architecture/Python row requires
  its retained compatibility evidence; package metadata alone is not evidence.

Commands below use `python` after activating the environment. See
[Installing OEL](../installation.md) for explicit Windows PowerShell and
macOS/Linux paths and for selecting the constraints file matching the
interpreter.

## SBOM

Generate a CycloneDX JSON software bill of materials for the installed Python
environment:

```bash
python tools/generate_python_sbom.py --output outputs/supply_chain/sbom.cdx.json
```

The SBOM records the installed Python distributions, versions, and PyPI package
URLs visible in the current interpreter environment. Generate it after
installing the exact profile under review. The aggregate `.[cross-platform]`
profile excludes ML and external integrations. `.[full]` is audited on its
named reference environment and is not a universal cross-platform promise.

## Dependency Audit

Run the dependency audit after installing the evaluated profile:

```bash
python -m pip install -U pip-audit
python -m pip_audit --format json --output outputs/supply_chain/pip-audit.json
```

The regular release gate runs the complete local workflow with:

```bash
python tools/run_supply_chain_gate.py --output-dir outputs/supply_chain --install-full
```

That command creates a disposable virtual environment, installs the constrained
full profile there, captures the pip install report and `pip check`, writes the
wheel inventory, SBOM, and freeze from that same isolated interpreter, runs an
unsuppressed audit, and records artifact hashes in `supply-chain-gate.json`.
The environment is removed after evidence generation so unrelated packages in
the developer or release checkout environment cannot enter the release audit.
No pull-request or scheduled GitHub workflow repeats this audit. It is a local,
release-authoritative gate. When a controlled Linux environment needs the
disk-bounded CPU-only Torch source, add `--torch-cpu-index`; the selected wheel
URL and hash remain captured in the install report and wheel inventory.

After the exact source candidate and supply-chain gate pass, build signed
installable artifacts from the authorized source root. Official signing is run
from the private release workspace, which owns the complete public-boundary
checker; `--source-root` points at the generated public export:

```bash
python tools/build_installable_release.py \
  --source-root <generated-public-export> \
  --output-dir <release-artifact-directory> \
  --edition public --channel stable \
  --private-key <offline-release-private-key.json> \
  --public-keys <trusted-release-keys.json> \
  --supply-chain-evidence outputs/supply_chain \
  --wheelhouse <exact-platform-reviewed-wheelhouse> \
  --platform <Windows-Linux-or-Darwin> --architecture <machine> \
  --base-url https://github.com/adamcohen8/orbital-engineering-lab/releases/download/v0.30.1 \
  --channel-url https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/public-stable.json
```

The `v0.31.0` base URL above is the last published public release at this
snapshot. Substitute the exact tag for a later release candidate; keep the
channel URL pointed at the stable metadata endpoint.

For an authorized Pro artifact, run the builder with an explicitly authorized
Pro source root and Pro edition/package policy; never combine a private source
root with `--edition public`.

Signed builds fail closed without a passing, version-matched
`supply-chain-gate.json` and unchanged referenced artifacts. The release
manifest binds their verified hashes through the public-safe attestation. They
also require an RS256 signing key of
at least 2048 bits whose exact public key is active in the bundled trust
registry. Every wheel is checked as a structurally valid wheel archive, and the
builder creates a clean runtime, performs a full-profile install with
`--no-index` from the copied wheelhouse, and imports the installed package
before signing the release. Public artifacts must be built from the generated
public export; the builder refuses a public artifact from the private source
root.

Distributable bundles do not copy the retained supply-chain gate or its raw
artifacts verbatim. Private release evidence can contain absolute workspace
paths, exact commands, and source-control provenance. The bundle instead
contains `release-evidence/public-supply-chain-attestation.json`, a
deterministic list of the verified source-evidence names, sizes, and SHA-256
digests. The complete evidence remains in the private release packet; the
published attestation binds that packet without disclosing its local content.

Treat a known vulnerability as a release finding until it is upgraded, removed,
documented as not applicable, or accepted by the evaluator in writing.

The published `v0.31.0` full-profile release candidate required the supported
PyTorch 2.13 release line and an unsuppressed passing audit with no implicit
exceptions. The active `pyproject.toml` still constrains the full profile to
that PyTorch line; each later candidate must be gated against its own dependency
contract and resolved environment. Do not add `--ignore-vuln` to release or
compatibility workflows. If an advisory is not applicable, document the
evidence and evaluator approval separately while retaining the unsuppressed
machine-readable audit result.

## Reproducible Dependency Workflow

OEL source dependency ranges live in `pyproject.toml`. Approved Python-minor
graphs live under `constraints/`; `requirements/` remains a compatibility shim.
The example below targets the blocking Python 3.11 baseline. Install it only
from an activated Python 3.11 environment; otherwise substitute the constraints
file matching the active supported minor. Retain the pip report:

```bash
python -m pip install \
  -c constraints/py311.txt \
  ".[cross-platform]" \
  --report outputs/supply_chain/pip-install-report.json
python -m pip check > outputs/supply_chain/pip-check.txt
python tools/generate_dependency_evidence.py \
  --install-report outputs/supply_chain/pip-install-report.json \
  --constraints constraints/py311.txt \
  --output outputs/supply_chain/wheel-inventory.json
python -m pip freeze --all > outputs/supply_chain/python-freeze.txt
```

The wheel inventory records the constraint digest, package/version, wheel tag,
source URL, and archive hash where pip supplies it. Constraint files are not
hash-locked, OS-specific release locks. For higher-assurance installs, use the
freeze, pip report, and wheel inventory as inputs to a reviewed, platform-specific,
hash-checked release lock. A release evidence packet applies only to its exact
recorded environment; resolving the source ranges again produces a new candidate.
To reproduce an archived candidate, install from its reviewed frozen record before
running the retained validation commands.

The recommended buyer-side gate is:

- install from a clean checkout through the matching constraint set,
- retain the pip installation report and wheel inventory,
- run and retain `pip check`,
- generate `python-freeze.txt`,
- generate the SBOM,
- run `pip-audit`,
- archive those artifacts with the validation evidence matrix and release
  report.

## GitHub Actions Pinning

Every checked-in third-party `uses:` reference must be pinned to a full
40-character commit SHA. The repository audit rejects semantic tags. Maintain
the upstream tag only as a review comment and:

- record the upstream action repository, tag, and resolved SHA in the release
  evidence packet,
- review and refresh pinned SHAs on a planned cadence or when GitHub issues a
  security advisory,
- validate the manual advisory workflow and local release gate after a pin
  update; do not treat hosted diagnostics as release acceptance.

## Release Evidence Packet

A procurement-ready packet should include:

- `outputs/supply_chain/sbom.cdx.json`,
- `outputs/supply_chain/pip-audit.json`,
- `outputs/supply_chain/pip-install-report.json`,
- `outputs/supply_chain/build-install-report.json`, which binds the declared
  `pyproject.toml` build-system wheels needed for a zero-network source install,
- `outputs/supply_chain/pip-check.txt`,
- `outputs/supply_chain/python-freeze.txt`,
- `outputs/supply_chain/wheel-inventory.json`,
- public/private export integrity check output,
- validation evidence matrix and harness reports when available,
- release checklist result and version/commit provenance.

The build and full-profile resolver reports use `--force-reinstall` inside the
disposable audit environment. This prevents packages installed for bootstrap or
audit tooling from being mistaken for already-satisfied OEL dependencies and
omitted from the signed wheel inventory, while replacing stale bootstrap copies
with the qualified versions that are actually audited.

## Non-Claims

The SBOM and audit evidence do not make OEL flight-qualified, FedRAMP
authorized, export-control classified, or approved for classified/CUI handling.
They are review artifacts that help an evaluator decide what additional
controls are required for their environment.
