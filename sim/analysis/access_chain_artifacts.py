"""Portable evidence and OEL-styled review timeline for access chains."""

import json
from html import escape
from pathlib import Path
from urllib.parse import quote

from sim.plotting.quality import STRICT_AGENT_PLOT_QUALITY, apply_plot_quality_policy
from sim.plotting.style import artifact_metadata, oel_plot_context, role_color, save_oel_figure
from sim.runtime_environment import configure_headless_runtime


def write_access_chain_artifacts(result, output_dir, *, include_timeline=True, scenario_name=""):
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    evidence = destination / "access_chain.json"
    artifacts = {"evidence_json": str(evidence)}
    if include_timeline:
        plot = destination / "access_chain_timeline.png"
        _write_timeline(result, plot, scenario_name)
        timeline = destination / "access_chain_review.html"
        _write_timeline_review(result, timeline, plot.name)
        artifacts["timeline_review_html"] = str(timeline)
        artifacts["timeline_plot_png"] = str(plot)
        artifacts["timeline_plot_quality_json"] = str(plot.with_suffix(".quality.json"))
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return artifacts


def _write_timeline(result, path, scenario_name):
    configure_headless_runtime()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    artifact_id = "access_chain_timeline"
    metadata = artifact_metadata(scenario_name=scenario_name, artifact_id=artifact_id)
    paths = result["paths"]
    # Paginate visually into columns to keep each route label legible.
    rows_per_column = 28
    columns = (len(paths) + rows_per_column - 1) // rows_per_column
    summary = result["summary"]
    with oel_plot_context(style_name="oel_light", metadata=metadata):
        figure, axes = plt.subplots(
            1, columns, squeeze=False,
            figsize=(8 * columns, 2.2 + 0.34 * min(len(paths), rows_per_column)),
        )
        try:
            for column, axis in enumerate(axes[0]):
                routes = paths[column * rows_per_column:(column + 1) * rows_per_column]
                labels = []
                for item in routes:
                    name = item["path_id"]
                    label = name if len(name) <= 38 else name[:35] + "…"
                    if item["excluded_ground_stations"]:
                        label += " (excluded)"
                    labels.append(label)
                labels.append("Any route available")
                for index, item in enumerate(routes):
                    ranges = [(row["start_s"], row["end_s"] - row["start_s"]) for row in item["windows"]]
                    if ranges:
                        axis.broken_barh(ranges, (index - 0.27, 0.54), facecolors=role_color("actual"))
                service = [
                    (row["start_s"], row["end_s"] - row["start_s"])
                    for row in result["segments"] if row["path_ids"]
                ]
                if service:
                    axis.broken_barh(service, (len(routes) - 0.27, 0.54), facecolors=role_color("reference"))
                axis._oel_categorical_y = True
                axis.set_yticks(range(len(labels)), labels)
                axis.get_yticklabels()[-1].set_fontweight("bold")
                axis.set_ylim(len(labels) - 0.5, -0.5)
                start, end = summary["horizon_start_s"], summary["horizon_end_s"]
                axis.set_xlim(start, end if end > start else start + 1.0)
                axis.set_xlabel("Analysis time (s)")
                axis.grid(False, axis="y")
                axis.grid(axis="x", alpha=0.25)
            figure.suptitle(f"Access chain availability · {result['analysis_id']}")
            figure.tight_layout(rect=(0.0, 0.065, 1.0, 0.98))
            quality = apply_plot_quality_policy(figure, policy=STRICT_AGENT_PLOT_QUALITY)
            save_oel_figure(
                figure, path, dpi=150, metadata=metadata, artifact_id=artifact_id,
                style_name="oel_light", bbox_inches="tight",
            )
        finally:
            plt.close(figure)
    path.with_suffix(".quality.json").write_text(
        json.dumps(quality.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _write_timeline_review(result, destination, plot_name):
    def endpoint(time, evidence):
        items = "".join(
            "<li>" + escape(
                f"{item['analysis_id']} · {item['disposition']} · {item['reason'] or 'reason unavailable'}"
            ) + "</li>"
            for item in evidence
        )
        return f"<details><summary>{time:.9g} s</summary><ul>{items}</ul></details>"

    rows = []
    for route in result["paths"]:
        route_name = escape(route["path_id"])
        for row in route["windows"]:
            start = endpoint(row["start_s"], row["start_evidence"])
            end = endpoint(row["end_s"], row["end_evidence"])
            rows.append(f"<tr><th scope='row'>{route_name}</th><td>{start}</td><td>{end}</td></tr>")
    if not rows:
        rows.append("<tr><td colspan='3'>No feasible route windows in this horizon.</td></tr>")
    title = escape(result["analysis_id"])
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · OEL access chains</title>
<style>
body {{ font: 16px system-ui, sans-serif; color: #172840; background: #fafbfd; margin: 0; }}
main {{ max-width: 74rem; margin: auto; padding: 2rem 1.25rem; }}
img {{ max-width: 100%; height: auto; }}
table {{ width: 100%; border-collapse: collapse; text-align: left; }}
th, td {{ padding: .6rem .7rem; border-bottom: 1px solid #cad5e1; vertical-align: top; }}
summary {{ cursor: pointer; color: #194a8c; }}
a {{ color: #194a8c; }}
</style>
<main>
<h1>{title}</h1>
<img src="{quote(plot_name)}" alt="Access windows by route and combined service availability">
<h2>Window boundaries</h2>
<table><thead><tr><th scope="col">Route</th><th scope="col">Acquisition</th><th scope="col">Loss</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p><a href="access_chain.json">Download access-chain evidence</a></p>
</main></html>
'''
    destination.write_text(document, encoding="utf-8")
