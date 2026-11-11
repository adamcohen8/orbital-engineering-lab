# Data Handling And Boundary Statement

OEL is local-first simulation software. By default, scenario inputs are read
from local files and run artifacts are written to local `outputs/` folders.
The public core does not require telemetry, hosted services, or network access
to run documented simulation examples.

## Local Data Written By OEL

Depending on configuration, OEL may write:

- JSON summaries and run logs,
- CSV histories,
- Markdown indexes and reports,
- PNG plots and optional animation/video artifacts,
- SQLite review stores under `outputs/<scenario>/review/`,
- validation and release evidence artifacts.

These outputs can contain scenario parameters, object states, controller
commands, estimator histories, sensor measurements, and user-authored metadata.
Treat output folders as sensitive whenever the input scenario or run context is
sensitive.

## Network And AI Boundaries

The public core examples run locally. Private AI-assisted report workflows
can send selected run summaries, packets, prompts, or report context to a
hosted provider only when the user explicitly configures that provider and
creates the report. Local Ollama-style workflows can be used where available
for local experimentation.

For restricted environments, use sealed mode:

```bash
.venv/bin/python run_simulation.py --config <path> --sealed-mode --validate-only
```

Sealed mode blocks arbitrary plugin imports, hosted/custom AI endpoints, all
cFS/SIL socket networking including loopback UDP, and high-detail output
retention unless the caller explicitly opts into the specific exception. A
separate sealed-policy isolated-test opt-in and any adapter-level network opt-in
are both required; loopback is not implicitly allowed.

## Public Repository Boundary

Do not put any of the following into the public repository, public issues,
public PRs, public examples, public docs, generated public exports, or shared
demo artifacts:

- classified information,
- CUI,
- export-controlled technical data,
- nonpublic government information,
- customer data,
- proprietary mission data,
- real operational scenarios, TLEs, CONOPS, sensor capabilities, or threat
  models that are not already approved for public release,
- API keys, access tokens, private keys, credentials, or internal endpoints.

Use synthetic, public-release-approved examples for public docs and demos. When
in doubt, keep the material private and request the appropriate legal,
security, export-control, customer, or government review.

## Export And CUI Statement

OEL does not provide export-control classification, CUI marking authority, or
authorization to disclose controlled information. Users are responsible for
their own export-control, CUI, classification, customer, and contractual
obligations. Public OEL examples are intended to be public-safe synthetic
technical examples, not operational mission data.

## Retention And Deletion

OEL does not automatically upload or retain outputs outside the local checkout.
Users are responsible for deleting local output folders, temporary public
exports, logs, validation evidence, and notebooks when they are no longer
needed or when policy requires retention limits.
