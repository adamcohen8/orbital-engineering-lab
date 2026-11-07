# Acquisition examples

These JSON files are request/requirement inputs for the experimental local
`oel data` CLI; they do not execute a simulation. Read
[the acquisition guide](../../docs/data-acquisition.md) and
[installation](../../docs/installation.md) first.

Inspect without network access:

```bash
oel data providers
oel data plan examples/data_acquisition/egm96-request.json
```

Explicit online retrieval:

```bash
oel data fetch examples/data_acquisition/egm96-request.json
oel data check ARTIFACT_ID --requirements examples/data_acquisition/gravity-requirements.json
```

The history example uses a synthetic catalog ID. Configure your connection
locally and replace that ID/interval with the authorized study selection before
fetching. No example contains credentials or a provider response. The weather
requirements illustrate a historical interval; they do not assert what data
was available at that historical decision time. EGM2008 downloads its full
252 MB source; check disk headroom before requesting it.
