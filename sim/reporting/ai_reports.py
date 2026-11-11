"""Orbital Engineering Lab Pro AI reports are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Provider-driven reports are unavailable in public OEL. "
        "Use the local MCP report preparation and audit tools. "
    )


build_ai_report_packet = _unavailable
build_ai_report_request = _unavailable
estimate_ai_report_cost = _unavailable
load_ai_report_payload_from_outputs = _unavailable
def write_ai_report_artifacts(*, cfg, config_path, outdir, payload, payload_kind, **options):
    if cfg.outputs.ai_report.get("enabled", False):
        _unavailable()
    return payload


write_ai_report_estimate_artifacts = _unavailable
