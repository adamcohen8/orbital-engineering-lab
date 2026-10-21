"""Public IO orchestration: format dispatch and receipts, never physics owners."""

from __future__ import annotations

from dataclasses import asdict

from .common import json_bytes, publish, read_text, source_receipt, strict_json
from .contracts import AttitudeHistory, DataIOError, Trajectory

OPTION_KEYS = {
    "context",
    "column_mapping",
    "provider_profile",
    "object_id",
    "body_frame",
    "frame",
    "derive_velocity",
    "epochs_utc",
    "target_id",
    "center_id",
    "instrument_id",
    "clock_id",
    "kernels",
    "segment_index",
    "sample_index",
    "version",
    "target_center",
    "spice_target_frame",
    "target_frame",
    "eop_path",
    "maneuvers",
    "satellite_id",
    "time_system",
    "reference_trajectory",
    "reference_format",
}


def _options(value):
    if value is not None and not isinstance(value, dict):
        raise DataIOError("Adapter options must be a mapping.")
    result = dict(value or {})
    if set(result) - OPTION_KEYS:
        raise DataIOError(f"Unknown adapter options: {sorted(set(result) - OPTION_KEYS)}.")
    for key in ("segment_index", "sample_index", "target_id", "center_id", "instrument_id", "clock_id"):
        if key in result and (isinstance(result[key], bool) or not isinstance(result[key], int)):
            raise DataIOError(f"{key} must be an integer.")
    if "derive_velocity" in result and not isinstance(result["derive_velocity"], bool):
        raise DataIOError("derive_velocity must be a boolean.")
    if "kernels" in result:
        kernels = result["kernels"]
        if (
            not isinstance(kernels, (list, tuple))
            or not 1 <= len(kernels) <= 16
            or any(not isinstance(k, str) for k in kernels)
        ):
            raise DataIOError("List 1..16 explicit kernel paths.")
    return result


def read_data(path, *, format: str, options=None):
    options = _options(options)
    from .requirements import require_read_options

    require_read_options(format, options)
    if format == "maneuver-plan-json":
        value = strict_json(read_text(path))
        if (
            not isinstance(value, dict)
            or value.get("schema") != "oel.maneuver-plan.v1"
            or value.get("execution_authorized") is not False
        ):
            raise DataIOError("Invalid non-executing maneuver-plan contract.")
        return value
    if format == "normalized-json":
        from .rinex import RinexDocument, encode_rinex, parse_rinex
        from .tracking import TrackingDocument, encode_tracking, parse_tracking

        document = strict_json(read_text(path))
        if not isinstance(document, dict):
            raise DataIOError("Normalized JSON requires an object document.")
        if document.get("schema") == "oel.native-tracking.v1":
            value = TrackingDocument(
                document["header"], tuple(document["segments"]), tuple(document.get("comments", []))
            )
            return parse_tracking(encode_tracking(value, format="tdm-kvn").decode())
        if document.get("schema") == "oel.rinex.v1":
            value = RinexDocument(
                document["kind"],
                document["version"],
                tuple(document["header"]),
                {k: tuple(v) for k, v in document["observation_types"].items()},
                tuple(document["records"]),
            )
            return parse_rinex(encode_rinex(value).decode())
        raise DataIOError("Unsupported normalized JSON document schema.")
    if format.startswith("trajectory-"):
        from .tabular import read_trajectory_table

        return read_trajectory_table(
            path, format=format, context=options.get("context"), column_mapping=options.get("column_mapping")
        )
    if format in {"gp-json", "gp-csv", "tle"}:
        from .catalog import read_catalog

        return read_catalog(path, format=format, provider_profile=options.get("provider_profile", "celestrak-gp"))
    if format in {"oem-kvn", "oem-xml", "opm-kvn", "opm-xml", "omm-kvn", "omm-xml", "cdm-kvn", "cdm-xml"}:
        from .navigation import read_navigation

        return read_navigation(path, format=format)
    if format in {"tdm-kvn", "tdm-xml"}:
        from .tracking import read_tracking

        return read_tracking(path, format=format)
    if format in {"aem-kvn", "apm-kvn"}:
        from .attitude import parse_adm

        return parse_adm(read_text(path), kind=format[:3])
    if format == "stk-attitude":
        from .attitude import read_stk_attitude

        return read_stk_attitude(
            path, object_id=options["object_id"], body_frame=options.get("body_frame", "SC_BODY_1")
        )
    if format == "stk-ephemeris":
        from .stk import read_stk

        return read_stk(path, object_id=options["object_id"])
    if format == "sp3":
        from .precise import read_sp3

        return read_sp3(
            path,
            object_id=options.get("object_id"),
            frame=options["frame"],
            derive_velocity=options.get("derive_velocity", False),
        )
    if format == "sentinel-eof":
        from .precise import read_sentinel_eof

        return read_sentinel_eof(path, object_id=options["object_id"], frame=options["frame"])
    if format in {"rinex-observation", "rinex-navigation"}:
        from .rinex import read_rinex

        value = read_rinex(path)
        if value.kind != format.removeprefix("rinex-"):
            raise DataIOError("RINEX file type disagrees with requested profile.")
        return value
    if format == "oel-state-product":
        from .products import trajectory_from_state_product

        return trajectory_from_state_product(path)
    if format in {"spk", "ck"}:
        from .spice import read_ck, read_spk
        from .time import parse_time

        epochs = [parse_time(value) for value in options["epochs_utc"]]
        if format == "spk":
            return read_spk(
                path,
                target_id=options["target_id"],
                center_id=options["center_id"],
                frame=options["frame"],
                object_id=options["object_id"],
                epochs=epochs,
                kernels=options["kernels"],
            )
        return read_ck(
            path,
            instrument_id=options["instrument_id"],
            clock_id=options["clock_id"],
            frame=options["frame"],
            object_id=options["object_id"],
            body_frame=options["body_frame"],
            epochs=epochs,
            kernels=options["kernels"],
        )
    raise DataIOError(f"Unsupported readable format: {format}.")


def as_trajectory(value, *, options=None) -> Trajectory:
    if isinstance(value, Trajectory):
        return value
    from sim.interchange.ccsds_oem import OemMessage

    if isinstance(value, OemMessage):
        from .navigation import trajectory_from_oem

        return trajectory_from_oem(value, segment_index=(options or {}).get("segment_index"))
    from sim.interchange.ccsds_odm import OpmMessage

    if isinstance(value, OpmMessage):
        from .time import parse_time

        if value.metadata.ref_frame != "EME2000" or value.metadata.center_name != "EARTH":
            raise DataIOError("OPM trajectory materialization currently requires Earth/EME2000.")
        covariance = None if value.covariance is None else [value.covariance.matrix]
        if value.covariance is not None and value.covariance.ref_frame not in {None, value.metadata.ref_frame}:
            raise DataIOError("OPM covariance frame requires an explicit transform.")
        return Trajectory(
            value.metadata.object_id,
            value.metadata.center_name,
            value.metadata.ref_frame,
            parse_time(value.state["EPOCH"], value.metadata.time_system),
            [0],
            [[float(value.state[key]) for key in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")]],
            value.metadata.object_name,
            [0] if covariance else None,
            covariance,
            {
                "maneuvers_preserved_not_executed": list(value.maneuvers),
                "physical": dict(value.physical),
                "keplerian": dict(value.keplerian),
                "user_defined": dict(value.user_defined),
            },
        )
    raise DataIOError("Input is not a Cartesian trajectory; mean elements are not relabeled as osculating states.")


def encode_data(value, *, format: str, options=None) -> tuple[bytes, dict]:
    options = _options(options)
    if isinstance(value, dict) and value.get("schema") == "oel.maneuver-plan.v1":
        if format == "maneuver-plan-json":
            return json_bytes(value), {"execution_authorized": False}
        from .products import opm_from_maneuver_plan

        reference = as_trajectory(
            read_data(options["reference_trajectory"], format=options["reference_format"], options=options)
        )
        value = opm_from_maneuver_plan(value, reference, sample_index=options["sample_index"])
    if isinstance(value, list):
        from .catalog import encode_catalog

        return encode_catalog(value, format=format), {}
    from .rinex import RinexDocument, encode_rinex

    if isinstance(value, RinexDocument):
        if format == "normalized-json":
            return json_bytes({"schema": "oel.rinex.v1", **asdict(value)}), {}
        if format != "rinex-" + value.kind:
            raise DataIOError(
                "RINEX observation/navigation conversion requires a separate computation, not relabeling."
            )
        return encode_rinex(value), {}
    from .tracking import TrackingDocument, encode_tracking

    if isinstance(value, TrackingDocument):
        if format == "normalized-json":
            return json_bytes({"schema": "oel.native-tracking.v1", **asdict(value)}), {
                "measurement_semantics": "native, unreduced"
            }
        return encode_tracking(value, format=format), {"measurement_semantics": "native, unreduced"}
    if isinstance(value, AttitudeHistory):
        from .attitude import encode_adm, encode_stk_attitude

        if format in {"aem-kvn", "apm-kvn"}:
            if format == "apm-kvn" and options.get("sample_index") is not None:
                from dataclasses import replace

                index = options["sample_index"]
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(value.times_s):
                    raise DataIOError("Invalid attitude sample index.")
                value = replace(
                    value,
                    times_s=value.times_s[index : index + 1],
                    quaternions=value.quaternions[index : index + 1],
                    rates_rad_s=None if value.rates_rad_s is None else value.rates_rad_s[index : index + 1],
                )
            return encode_adm(value, kind=format[:3]), {}
        if format == "stk-attitude":
            return encode_stk_attitude(value), {}
        if format == "ck":
            from .spice import encode_ck

            return encode_ck(
                value, instrument_id=options["instrument_id"], clock_id=options["clock_id"], kernels=options["kernels"]
            )
        raise DataIOError("Unsupported attitude output format.")
    if not isinstance(value, Trajectory):
        if format == "maneuver-plan-json":
            from .products import maneuver_plan_from_opm

            return json_bytes(maneuver_plan_from_opm(value)), {"execution_authorized": False}
        from sim.interchange.ccsds_cdm import CdmMessage
        from sim.interchange.ccsds_odm import OmmMessage, OpmMessage
        from sim.interchange.ccsds_oem import OemMessage

        native_family = {OemMessage: "oem", OpmMessage: "opm", OmmMessage: "omm", CdmMessage: "cdm"}.get(type(value))
        transform_requested = any(options.get(key) for key in ("target_frame", "target_center", "spice_target_frame"))
        if native_family and format.startswith(native_family + "-") and not transform_requested:
            from .navigation import encode_navigation

            return encode_navigation(value, format=format, version=options.get("version", "3.0")), {}
        trajectory = as_trajectory(value, options=options)
        data, extra = encode_data(trajectory, format=format, options=options)
        return data, {"source_annotations": trajectory.metadata, **extra}
    if options.get("target_center") or options.get("spice_target_frame"):
        from .spice import transform_spice

        value = transform_spice(
            value,
            target_center=options.get("target_center", value.center),
            target_frame=options.get("spice_target_frame", value.frame),
            kernels=options["kernels"],
        )
    elif options.get("target_frame") and value.frame != options["target_frame"]:
        from .trajectory import transform_trajectory

        value = transform_trajectory(value, options["target_frame"], eop_series=_eop(options))
    if format.startswith("trajectory-"):
        from .tabular import encode_trajectory_table

        data, context = encode_trajectory_table(value, format=format)
        return data, {"trajectory_context": context}
    if format in {"oem-kvn", "oem-xml"}:
        from .navigation import encode_navigation, oem_from_trajectory

        return encode_navigation(oem_from_trajectory(value), format=format, version=options.get("version", "3.0")), {}
    if format in {"opm-kvn", "opm-xml"}:
        from .navigation import encode_navigation
        from .products import opm_from_trajectory

        message = opm_from_trajectory(
            value, sample_index=options["sample_index"], maneuvers=options.get("maneuvers", ())
        )
        return encode_navigation(message, format=format), {"sample_index": options["sample_index"]}
    if format == "stk-ephemeris":
        from .stk import encode_stk

        return encode_stk(value), {}
    if format == "sp3":
        from .precise import encode_sp3

        return encode_sp3(
            value, satellite_id=options["satellite_id"], time_system=options.get("time_system", "GPS")
        ), {}
    if format == "spk":
        from .spice import encode_spk

        return encode_spk(
            value, target_id=options["target_id"], center_id=options["center_id"], kernels=options["kernels"]
        )
    if format in {"czml", "geojson", "kml"}:
        from .visualization import encode_visualization

        return encode_visualization(value, format=format)
    raise DataIOError(
        f"No supported conversion from Cartesian trajectory to {format}; fitting is a separate review operation."
    )


def _eop(options):
    if options.get("eop_path"):
        from sim.frame_time import load_iers_eop

        return load_iers_eop(options["eop_path"])
    return None


def _option_sources(options):
    return [
        *options.get("kernels", []),
        *[options[key] for key in ("eop_path", "reference_trajectory") if options.get(key)],
    ]


def convert(source, output, *, input_format: str, output_format: str, options=None):
    from pathlib import Path

    from .annotations import inherited_annotations

    options = _options(options)
    sources = [source, *_option_sources(options)]
    if Path(str(source) + ".receipt.json").is_file():
        sources.append(str(source) + ".receipt.json")
    identities = [source_receipt(p) for p in sources]
    value = read_data(source, format=input_format, options=options)
    annotations = inherited_annotations(source)
    if input_format in {"omm-kvn", "omm-xml"} and output_format in {"tle", "gp-json", "gp-csv"}:
        from .catalog import read_catalog

        value = read_catalog(source, format=input_format)
    data, extra = encode_data(value, format=output_format, options=options)
    if [source_receipt(p) for p in sources] != identities:
        raise DataIOError("Input artifacts changed during conversion.")
    return publish(
        output,
        data,
        receipt={
            "operation": "convert",
            "input_format": input_format,
            "input_metadata": {**annotations, **value.metadata}
            if isinstance(value, (Trajectory, AttitudeHistory))
            else annotations,
            "format": output_format,
            "sources": identities,
            "options": options,
            **extra,
        },
        sources=tuple(sources),
    )


def export_review(
    completed_run,
    output,
    *,
    format: str,
    object_id=None,
    start_s=None,
    stop_s=None,
    sample_index=None,
    final=False,
    options=None,
):
    from .objects.od import locate_tracking_od

    if locate_tracking_od(completed_run) is not None:
        if start_s is not None or stop_s is not None or final or sample_index not in {None, 0}:
            raise DataIOError("An OD result is one estimated state epoch; it is not a simulated history interval.")
        from .objects.exports import export_estimate

        return export_estimate(completed_run, output, format=format, options=options, object_id=object_id)
    from .review import read_review

    options = _options(options)
    attitude = format in {"aem-kvn", "apm-kvn", "stk-attitude", "ck"}
    selection = read_review(
        completed_run, object_id=object_id, start_s=start_s, stop_s=stop_s, include_attitude=attitude
    )
    if final or sample_index is not None:
        if final and sample_index is not None:
            raise DataIOError("Select final or sample_index, not both.")
        from .review import select_review_sample

        selection = select_review_sample(selection, sample_index=sample_index, final=final)
        options["sample_index"] = 0
    identities = [source_receipt(p) for p in _option_sources(options)]
    data, extra = encode_data(selection.attitude if attitude else selection.trajectory, format=format, options=options)
    if [source_receipt(p) for p in _option_sources(options)] != identities:
        raise DataIOError("Conversion resources changed during export.")
    return publish(
        output,
        data,
        receipt={
            "operation": "export-review",
            "format": format,
            "source": selection.receipt,
            "input_metadata": (selection.attitude if attitude else selection.trajectory).metadata,
            "resources": identities,
            "options": options,
            **extra,
        },
        sources=tuple(_option_sources(options)),
    )
