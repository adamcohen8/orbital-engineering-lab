# Local data acquisition

The experimental `oel data` workflow acquires bounded datasets, preserves exact
source bytes and provenance, and prepares local inputs for existing OEL
consumers. `python -m sim.data_acquisition` is the source-checkout equivalent.
It never runs a simulation, fits an orbit, buys data, uploads files, or grants
permission to disclose a dataset. Ordinary offline IO remains `oel io`.

## Install and discover

Follow [installation](installation.md) for the supported environment. Public
downloads, registration, inspection and offline use need no credential backend.
For account setup in a source environment, install the optional profile:

```bash
python -m pip install '.[acquisition]'
oel data providers
oel data schema request
```

The extra supplies `keyring>=25,<26`. Credential persistence permits macOS
Keychain, Windows Credential Manager, or Linux Secret Service only. A missing,
locked or unsupported backend fails with an actionable error; there is no
plaintext fallback. Linux headless sessions may require unlocking Secret
Service. Install dependencies into a managed installation only through its
supported signed update/profile workflow; the release manifest must advertise
the acquisition profile. This source feature is not evidence of a published
signed acquisition bundle or cross-platform qualification.

## Scope

| Provider | Products | Interface |
| --- | --- | --- |
| Space-Track | `gp`, bounded `gp_history`, `satcat` | User's own account, HTTPS API, GP/SATCAT JSON |
| Gravity | `egm96`, `egm2008` | Curated static Earth `.gfc` files, pinned size and SHA-256 |
| CelesTrak | `eop`, `space_weather` | Documented complete CSV feeds |
| Local | Explicitly selected files | Immutable snapshot, format inspection, supplied terms |

Registration supports `gravity/icgem`, `catalog/gp-json`, `catalog/gp-csv`,
`catalog/tle`, `catalog-metadata/satcat-json`, `eop/celestrak-eop-csv`,
`eop/finals2000a`, `eop/c04_csv`, and `space-weather/celestrak-weather-csv`.
Use `opaque/opaque` for another licensed file. Opaque or incompatible inputs
remain stored evidence; registration never makes them executable. Archive
extraction, arbitrary URLs, commercial service connectors, SPICE/CDDIS
acquisition, scheduled refresh and remote acquisition are outside this profile.

## Connect locally

Run setup yourself in a local interactive terminal:

```bash
oel data connections add spacetrack --name personal
oel data connections status personal
oel data connections test personal
```

Setup prompts for identity and password without echo and puts them in the OS
credential manager. Configuration contains a secret reference and an opaque
account scope, not the password. Status is offline and does not access the
credential manager. Testing is an explicit network operation; it establishes a
session but does not prove access to every data product. Cookies are in-memory
only and cleared when the acquisition finishes. Multiple aliases for the same
account share OEL's request/retrieval accounting.

Agents use a connection alias, never a password. CLI setup rejects noninteractive
stdin; no password argument or secret environment variable is provided. The
Python integration accepts credentials for an application-owned local setup
flow and must keep those values out of agent/model messages. An agent with
unrestricted OS privileges is outside this credential abstraction's isolation
boundary.

```bash
oel data connections remove personal
```

Removal deletes the secret reference and credential. It preserves acquired
evidence and account query history; recreating an alias cannot reset limits.

## Plan and acquire

Create a request JSON using `oel.data-acquisition.v1`. This example uses a
synthetic catalog identifier: replace it with the objects and interval you
are authorized to retrieve.

```json
{
  "schema": "oel.data-acquisition.v1",
  "provider": "spacetrack",
  "product": "gp_history",
  "connection": "personal",
  "object_ids": [123456789],
  "start": "2024-01-10T00:00:00Z",
  "stop": "2024-01-12T00:00:00Z",
  "published_before": "2024-01-12T00:00:00Z",
  "max_bytes": 67108864,
  "max_records": 10000,
  "max_requests": 8
}
```

```bash
oel data plan request.json
oel data fetch request.json
oel data --offline fetch request.json
oel data inspect ARTIFACT_ID
```

Planning uses local metadata only: no authentication, discovery requests, or
external calls. It reports retained sources, missing intervals, request/byte
bounds and zero OEL purchase amount. Maximum network requests includes login,
redirects and retries. Requests require absolute timezone-qualified epochs;
ambiguous relative dates are rejected. Historical selection intervals are
`[start, stop)`. An inclusive provider boundary can remain in the untouched raw
response; downstream selection is explicit.

History is limited to 32 selected objects, 180 days, 50,000 records and 16 total
network requests, with smaller defaults. The byte ceiling is 512 MiB. OEL asks
for one extra record to detect truncation; it does not silently accept a limited
page. Oversized work receives an actionable bulk-archive/narrow-request error.
Downloaded history is reused by account, object, interval and publication
cutoff. Only missing intervals are requested. A complete empty response is
retained query evidence, not an orbital state. Corrupt historical artifacts
fail closed instead of being silently redownloaded. Successful portions of an
interrupted multi-request acquisition remain available to the next plan.
An earlier publication cutoff can be prepared as a local, hash-bound filtered
view of retained history without another download. If only a narrower cutoff
was originally acquired, widening it over the same interval is blocked; use an
explicit provider archive workflow for the missing publication revisions.
For flexible retrospective studies, acquire full epoch-bounded history once
and apply publication cutoffs during local selection.

Current GP snapshots are reused for one hour; SATCAT and environment feeds for
one day. Immutable curated gravity products are reused indefinitely. Request
and byte limits also apply when using retained products. A narrower request may
reuse a larger source artifact, so its byte budget must accommodate that file.
The current feed is a snapshot retrieved now, not proof of what was available
at a historical decision time.

Space-Track's [API guidelines](https://www.space-track.org/documentation)
require batching and retained history. OEL conservatively accounts for at most
20 attempts/minute and 180/hour, GP retrieval once/hour and SATCAT once/day.
HTTP 429 backoff is persisted. Other programs using the same account are not
visible to this ledger. An additional login is counted for every separate
acquisition; one session serves the bounded queries within it. Long waits are
returned as `retry_after_s`, not hidden behind an unbounded sleeping process.

## Gravity and licensed files

For curated EGM96, use `provider: gravity`, `product: egm96`. For EGM2008 use
`product: egm2008` and `max_bytes: 300000000`; the full source is 252,140,870
bytes. The pinned EGM2008 SHA-256 is
`ab5b524da073e63b5bdceb7ca47a0de07a26dd44a1c5798f39fc98dc80af70fd`.
The source is [ICGEM](https://icgem.gfz.de/tom_longtime); model authors and
source-specific terms remain authoritative. Pins identify accepted bytes,
not the scientific accuracy of the model. Degree/order truncation remains an
explicit scenario setting; the original full source is retained.
The exact pinned legacy EGM96 file omits a normalization header; its explicit
source profile supplies fully normalized coefficients and records that supplied
context in inspection. Unfamiliar files never inherit this default.

Register an already licensed coefficient file:

```bash
oel data register /path/to/model.gfc --kind gravity --format icgem --terms terms.json
```

`terms.json` accepts bounded `source`, `license`, `redistribution` and
`attribution` strings. These are supplied metadata, not an automated legal
decision. A future authenticated commercial downloader must use the provider's
permitted interface; login access alone does not establish reuse rights.

Gravity inspection reports GM, radius, normalization, tide convention, model
identity and coefficient availability. Missing metadata, unsupported conventions,
nonfinite/duplicate terms and non-Earth products remain unqualified. Temporal
terms are explicitly blocked instead of silently dropped. A geoid grid is not
a harmonic coefficient file. A requirements file must explicitly select
`consumer: earth-gravity`, `degree`, `order`, `normalization` and `tide_system`.

The existing `source: egm96` API remains compatible: it keeps verified legacy
cache lookup and sealed-mode opt-ins, while new downloads use the common
bounded transport and record an acquisition receipt. Explicit `coeff_path`
continues to use the existing physics owner.

## Environmental inputs

Use `provider: celestrak` with `product: eop` or `space_weather`. Acquisition
preserves the full CSV and its source fields. The runtime requirements include:

```json
{
  "consumer": "nrlmsise00",
  "start": "2024-01-10T00:00:00Z",
  "stop": "2024-01-10T12:00:00Z",
  "allow_predictions": false
}
```

Here `stop` is the final simulation sample and must itself be covered. EOP
uses `consumer: runtime-frames` or `frame-time`; weather uses `nrlmsise00` or
`msis86`. Predictions require an explicit choice. Historical data is judged by
requested coverage, not rejected simply because it is old.

EOP CSV conversion preserves separate equinox and celestial-pole corrections,
arcsecond/milliarcsecond distinctions, LOD and source DAT. Runtime conversion
writes the existing EOP-All layout; observed/predicted labels remain in the
source and receipt. Other local EOP formats can serve the canonical frame-time
API but are not silently converted to runtime EOP-All without explicit DAT.
See [EOP documentation](https://celestrak.org/spacedata/EOP-format.php).

Weather preparation checks contiguous coverage plus three prior days and the
required flux/geomagnetic fields. The derived numeric table preserves the
existing MSIS consumer's previous-day observed flux and adjusted centered
81-day flux selection. It does not redesign that consumer or infer missing
indices. Unused missing numeric fields remain NaN in the derived table and are
not fabricated. Quality flags remain in the original CSV/receipt. JB2008's
additional indices are not manufactured from this feed. See
[weather definitions](https://celestrak.org/spacedata/SpaceWx-format.php).

## Prepare and bind a study

```bash
oel data check ARTIFACT_ID --requirements requirements.json
oel data materialize ARTIFACT_ID --destination inputs/weather.txt --requirements requirements.json
oel data bind scenario.yaml --input inputs/weather.txt --field simulator.environment.nrlmsise00_sw_path --out bound.yaml
python run_simulation.py --config bound.yaml --safe-validate
python run_simulation.py --config bound.yaml --validate-only
```

The matching gravity field is
`simulator.dynamics.orbit.spherical_harmonics.coeff_path`; configure
`source: icgem` and matching degree/order first. Runtime EOP uses
`simulator.frames.eop_path`. Binding does not enable drag, perturbations or any
other physics. It checks source identity and the declared consumer/time bounds.
Select the matching `simulator.environment.atmosphere_model` for weather or
`simulator.frames.model: iau76_80_eop` for runtime EOP before binding.
Use the ordinary trusted run workflow only after validation.

Raw `materialize` without requirements only copies evidence. It does not make
an opaque or incompatible product ready for simulation. Every materialized file
has an adjacent `.acquisition.json` receipt; destinations are never overwritten.

For orbital histories, select one exact record per object before composition:

```bash
oel data select ARTIFACT_ID --object-ids 123456789 --at 2024-01-10T12:00:00Z --policy latest-before --max-age-s 86400
oel data build-catalog-study SELECTED_ID --scenario-path study.yaml --scenario-name catalog_study --output-dir outputs/catalog_study --start 2024-01-10T12:00:00Z --duration-s 600 --step-s 10
python run_simulation.py --config study.yaml --validate-only
```

`exact` is also supported. Equal-epoch conflicting records require narrower
sources. `--published-before` filters source publication timestamps; missing
timestamps cannot satisfy that cutoff. Selection preserves all parent
restrictions and hashes. The existing `oel io` catalog materializer also binds
adjacent acquired-selection receipts automatically. Mean elements select OGP,
not an implicit ONP osculating-state interpretation.

Bindings live in ordinary scenario `metadata.acquisition_inputs`. Loading,
direct engine entry and completed reporting recheck identities. File reads
during loading use OEL's existing path policy; metadata cannot grant itself
access to another directory. The run summary retains `acquisition_provenance`
and the effective config retains the full receipts. Deliberate modification or
removal of user-owned receipts is not prevented cryptographically. These
bindings detect accidental drift; they are not signatures or execution authority.
Original source queries and retrieval receipts are retained once in
`acquisition_source_receipts`, including when a catalog selection was derived
from several historical downloads. The completed summary preserves that lineage
without requiring an agent to open the account's acquisition index.

## Storage, offline use and agent boundary

The managed data root contains `acquired-data/`: content-addressed source
objects, an indexed receipt/query ledger, and cross-process locks. The config
root contains `data-connections/`. Honor existing `OEL_DATA_HOME` and
`OEL_CONFIG_HOME`; provider data never belongs in source exports. Source bytes
are hashed on consumption. Registration detects source mutation while copying.
Downloads are size/time bounded, use curated HTTPS endpoints, reject compressed
transfer encodings/unsafe redirects, and publish atomically. At least 5 GiB free
disk space is preserved before large allocations. No automatic eviction runs.

```bash
oel data retention
oel data --offline fetch request.json
```

Retention reports artifact sizes and materialization/selection references.
Keep evidence and account query history when archiving data; missing historical
bytes require restoration rather than an implicit retrieval-policy reset.
Sealed mode blocks new acquisition and connection tests even if a request file
asks for data. Cached files and local registration remain usable. The legacy
gravity API keeps its existing separately explicit sealed exception.

Agent results contain status, coverage and provenance, not raw response bodies
or credentials. Downloaded text is untrusted data, never agent instructions.
The account's data and derived-result disclosure conditions still apply when an
agent sends a summary to a hosted model. OEL does not classify data or grant
redistribution permission. The CLI/Python routes are the supported acquisition
surface; existing MCP tools retain their no-external-communication contract.

## Python and verification

```python
from sim.data_acquisition import AcquisitionService, Request

service = AcquisitionService()
request = Request(provider="gravity", product="egm96")
plan = service.plan(request)                 # local only
result = service.fetch(request, offline=True)  # reuse only; raises on a miss
```

The wheel includes request/receipt schemas. Offline synthetic tests cover
authentication failure, redaction, quotas, concurrency, caching and historical
gaps, corruption, convention/coverage failures, local file mutation, selection,
scenario validation and actual deterministic OGP execution. Run:

```bash
python -m pytest sim/tests/test_data_acquisition*.py
```

Live checks are opt-in and separate from offline tests. Space-Track requires a
configured account; automated tests never use one. Actual OS credential-store
round trips and Windows/Linux execution require their own recorded evidence.
Public download/format checks do not establish operational orbit accuracy,
standards completeness, commercial readiness or release qualification.
