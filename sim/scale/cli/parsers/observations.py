from __future__ import annotations

import argparse

from sim.scale.measurements import DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM


def register_observations_parsers(sub: argparse._SubParsersAction) -> None:
    ingest_ground_measurements_parser = sub.add_parser(
        "ingest-ground-measurements",
        help="Ingest scenario-generated ground-station measurement rows into an OEL Scale store.",
    )
    ingest_ground_measurements_parser.add_argument(
        "payload_json", help="JSON payload containing ground_station_measurements."
    )
    ingest_ground_measurements_parser.add_argument(
        "--store", required=True, help="Operational catalog SQLite store path."
    )
    ingest_ground_measurements_parser.add_argument(
        "--source-label", default="", help="Source label for the ingest batch."
    )
    ingest_ground_measurements_parser.add_argument(
        "--jd-utc-start",
        type=float,
        help="Fallback start JD when rows include time_s but not jd_utc.",
    )
    ingest_ground_measurements_parser.add_argument(
        "--json", action="store_true", help="Print the ingest payload as JSON."
    )

    export_ground_packet_parser = sub.add_parser(
        "export-ground-measurement-packet",
        help="Export stored az/el/range ground measurements as a position-only OD observation packet.",
    )
    export_ground_packet_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    export_ground_packet_parser.add_argument("--station-id", required=True, help="Ground station ID to export.")
    export_ground_packet_parser.add_argument("--object-id", required=True, help="Object ID to export.")
    export_ground_packet_parser.add_argument("--out", required=True, help="Output observation_packet.json path.")
    export_ground_packet_parser.add_argument(
        "--source-label", default="", help="Source label for the observation packet."
    )
    export_ground_packet_parser.add_argument("--json", action="store_true", help="Print the export payload as JSON.")

    fit_ground_od_parser = sub.add_parser(
        "fit-ground-measurement-od",
        help="Run dynamics OD from stored az/el/range ground-station measurements.",
    )
    fit_ground_od_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    fit_ground_od_parser.add_argument("--station-id", required=True, help="Ground station ID to use.")
    fit_ground_od_parser.add_argument("--object-id", required=True, help="Object ID to fit.")
    fit_ground_od_parser.add_argument(
        "--output-dir", required=True, help="Directory for OD packet, reports, and manifest."
    )
    fit_ground_od_parser.add_argument("--fit-duration-s", type=float)
    fit_ground_od_parser.add_argument("--holdout-duration-s", type=float)
    fit_ground_od_parser.add_argument("--dt-s", type=float)
    fit_ground_od_parser.add_argument("--estimate", default="state")
    fit_ground_od_parser.add_argument("--dynamics-model", default="two_body")
    fit_ground_od_parser.add_argument("--j2", action="store_true")
    fit_ground_od_parser.add_argument("--drag", action="store_true")
    fit_ground_od_parser.add_argument("--srp", action="store_true")
    fit_ground_od_parser.add_argument("--atmosphere-model", default="")
    fit_ground_od_parser.add_argument("--max-nfev", type=int, default=24)
    fit_ground_od_parser.add_argument("--detect-maneuvers", action="store_true")
    fit_ground_od_parser.add_argument("--json", action="store_true", help="Print the OD manifest as JSON.")

    fit_ground_sensor_od_parser = sub.add_parser(
        "fit-ground-sensor-od",
        help="Fit initial Cartesian state directly against stored az/el/range ground-station measurements.",
    )
    fit_ground_sensor_od_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    fit_ground_sensor_od_parser.add_argument("--object-id", required=True, help="Object ID to fit.")
    fit_ground_sensor_od_parser.add_argument(
        "--station-id", action="append", default=[], help="Optional station ID; repeatable."
    )
    fit_ground_sensor_od_parser.add_argument(
        "--output-dir", required=True, help="Directory for native sensor OD artifacts."
    )
    fit_ground_sensor_prior = fit_ground_sensor_od_parser.add_mutually_exclusive_group()
    fit_ground_sensor_prior.add_argument(
        "--initial-state-eci-km-s",
        nargs=6,
        type=float,
        metavar=("X_KM", "Y_KM", "Z_KM", "VX_KM_S", "VY_KM_S", "VZ_KM_S"),
        help="Prior initial Cartesian state at the first measurement epoch.",
    )
    fit_ground_sensor_prior.add_argument(
        "--initial-state-packet",
        help="Mission input packet containing a prior state for --object-id.",
    )
    fit_ground_sensor_od_parser.add_argument("--fit-duration-s", type=float)
    fit_ground_sensor_od_parser.add_argument("--holdout-duration-s", type=float)
    fit_ground_sensor_od_parser.add_argument("--dt-s", type=float)
    fit_ground_sensor_od_parser.add_argument("--dynamics-model", default="two_body")
    fit_ground_sensor_od_parser.add_argument("--j2", action="store_true")
    fit_ground_sensor_od_parser.add_argument("--j3", action="store_true")
    fit_ground_sensor_od_parser.add_argument("--j4", action="store_true")
    fit_ground_sensor_od_parser.add_argument("--drag", action="store_true")
    fit_ground_sensor_od_parser.add_argument("--srp", action="store_true")
    fit_ground_sensor_od_parser.add_argument("--third-body-sun", action="store_true")
    fit_ground_sensor_od_parser.add_argument("--third-body-moon", action="store_true")
    fit_ground_sensor_od_parser.add_argument("--atmosphere-model", default="")
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-degree", type=int)
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-order", type=int)
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-source", default="")
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-coeff-path", default="")
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-frame-model", default="")
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-eop-path", default="")
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-reference-radius-km", type=float)
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-fd-step-km", type=float)
    fit_ground_sensor_od_parser.add_argument("--spherical-harmonics-normalized", action="store_true")
    fit_ground_sensor_od_parser.add_argument(
        "--spherical-harmonics-term-json",
        action="append",
        default=[],
        help='Inline spherical harmonic term JSON; repeatable. Example: \'{"n":2,"m":0,"c_nm":-4.8e-4}\'.',
    )
    fit_ground_sensor_od_parser.add_argument("--max-nfev", type=int, default=24)
    fit_ground_sensor_od_parser.add_argument(
        "--systematic-error-model-json",
        help="JSON file containing the version-1 ground systematic-error model.",
    )
    fit_ground_sensor_od_parser.add_argument(
        "--exclude-station-id",
        action="append",
        default=[],
        help="Exclude a station from the solve and record the disposition; repeatable.",
    )
    fit_ground_sensor_od_parser.add_argument(
        "--holdout-station-id",
        action="append",
        default=[],
        help="Reserve a station for cross-station holdout evidence; repeatable.",
    )
    fit_ground_sensor_od_parser.add_argument(
        "--robust-loss",
        choices=("linear", "huber", "soft_l1", "cauchy"),
        default="linear",
    )
    fit_ground_sensor_od_parser.add_argument("--robust-f-scale", type=float, default=1.0)
    fit_ground_sensor_od_parser.add_argument("--sigma-clip-threshold", type=float)
    fit_ground_sensor_od_parser.add_argument(
        "--json", action="store_true", help="Print the native sensor OD manifest as JSON."
    )

    assess_ground_parser = sub.add_parser(
        "assess-ground-measurement-quality",
        help="Assess residual quality for a completed ground-measurement OD manifest.",
    )
    assess_ground_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    assess_ground_parser.add_argument("--manifest", required=True, help="ground_measurement_od_manifest.json path.")
    assess_ground_parser.add_argument("--warning-sigma", type=float, default=3.0)
    assess_ground_parser.add_argument("--reject-sigma", type=float, default=6.0)
    assess_ground_parser.add_argument("--warning-error-m", type=float)
    assess_ground_parser.add_argument("--reject-error-m", type=float)
    assess_ground_parser.add_argument("--json", action="store_true", help="Print the assessment payload as JSON.")

    maneuver_parser = sub.add_parser(
        "detect-maneuvers", help="Write advisory maneuver flags from successive OD products."
    )
    maneuver_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    maneuver_parser.add_argument(
        "--object-id", action="append", default=[], help="Optional object ID to check; repeatable."
    )
    maneuver_parser.add_argument("--mean-motion-threshold", type=float, help="Mean-motion delta threshold in rev/day.")
    maneuver_parser.add_argument("--eccentricity-threshold", type=float, help="Eccentricity delta threshold.")
    maneuver_parser.add_argument(
        "--inclination-threshold-deg", type=float, help="Inclination delta threshold in degrees."
    )
    maneuver_parser.add_argument("--raan-threshold-deg", type=float, help="RAAN delta threshold in degrees.")
    maneuver_parser.add_argument(
        "--argp-threshold-deg", type=float, help="Argument-of-perigee delta threshold in degrees."
    )
    maneuver_parser.add_argument("--json", action="store_true", help="Print the maneuver detection payload as JSON.")

    sgp4_od_parser = sub.add_parser(
        "fit-sgp4-od",
        aliases=["fit-ogp-od"],
        help="Fit stored TEME PV: fit-ogp-od supports SGP4/SDP4; fit-sgp4-od is near-Earth only.",
    )
    sgp4_od_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    sgp4_od_parser.add_argument(
        "--object-id", action="append", default=[], help="Optional object ID to fit; repeatable."
    )
    sgp4_od_parser.add_argument(
        "--min-observations", type=int, default=2, help="Minimum TEME PV observations per object."
    )
    sgp4_od_parser.add_argument("--position-sigma-km", type=float, default=0.001, help="Position sigma for whitening.")
    sgp4_od_parser.add_argument(
        "--velocity-sigma-km-s", type=float, default=1.0e-6, help="Velocity sigma for whitening."
    )
    sgp4_od_parser.add_argument(
        "--fit-until-jd-utc", type=float, help="Optional fit-window cutoff; later samples are holdout."
    )
    sgp4_od_parser.add_argument("--max-nfev", type=int, default=100, help="Maximum least-squares function evaluations.")
    sgp4_od_parser.add_argument("--estimate-bstar", action="store_true", help="OGP only: request data-identifiable B* fitting.")
    sgp4_od_parser.add_argument("--bstar-prior-sigma", type=float, help="OGP only: positive B* prior sigma.")
    sgp4_od_parser.add_argument("--json", action="store_true", help="Print the OD payload as JSON.")

    freeze = sub.add_parser("freeze-ogp-od", help="Freeze one exact stored OGP fit; never refit or promote it.")
    freeze.add_argument("--store", required=True)
    freeze.add_argument("--od-solution-id", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--position-rms-m-max", type=float, default=2.0)
    freeze.add_argument("--minimum-observations", type=int, default=6)
    freeze.add_argument("--json", action="store_true")
    assess = sub.add_parser("assess-frozen-ogp-od", help="Read-only assessment of a frozen fit on exact later observations.")
    assess.add_argument("--store", required=True)
    assess.add_argument("--frozen-solution", required=True)
    assess.add_argument("--observation-id", action="append", required=True)
    assess.add_argument("--output-dir", required=True)
    assess.add_argument("--json", action="store_true")

    sgp4_ground_sensor_parser = sub.add_parser(
        "fit-ground-sgp4-sensor-od",
        aliases=["fit-ground-ogp-sensor-od"],
        help="Fit station measurements: fit-ground-ogp-sensor-od supports SGP4/SDP4; the SGP4 command is near-Earth only.",
    )
    sgp4_ground_sensor_parser.add_argument("--store", required=True, help="Operational catalog SQLite store path.")
    sgp4_ground_sensor_parser.add_argument("--object-id", required=True, help="Object ID to fit.")
    sgp4_ground_sensor_parser.add_argument(
        "--station-id", action="append", default=[], help="Optional station ID; repeatable."
    )
    sgp4_ground_sensor_parser.add_argument(
        "--prior-tle", required=True, help="Prior TLE file; first TLE is used as template."
    )
    sgp4_ground_sensor_parser.add_argument(
        "--output-dir", required=True, help="Directory for SGP4 sensor OD artifacts."
    )
    sgp4_ground_sensor_parser.add_argument("--fit-duration-s", type=float)
    sgp4_ground_sensor_parser.add_argument("--holdout-duration-s", type=float)
    sgp4_ground_sensor_parser.add_argument(
        "--frame-transform",
        default=DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
        choices=("teme_as_eci", "teme_to_eci_iau80"),
        help="Frame transform used before station measurement prediction; the legacy TEME-as-ECI approximation is opt-in.",
    )
    sgp4_ground_sensor_parser.add_argument("--max-nfev", type=int, default=100)
    sgp4_ground_sensor_parser.add_argument(
        "--json", action="store_true", help="Print the SGP4 sensor OD manifest as JSON."
    )

    synthetic_sgp4_sensor_parser = sub.add_parser(
        "run-synthetic-sgp4-sensor-od",
        aliases=["run-synthetic-ogp-sensor-od"],
        help="Simulate noisy station measurements and fit mean elements; the OGP command supports SGP4/SDP4.",
    )
    synthetic_sgp4_sensor_parser.add_argument("--prior-tle", required=True, help="Prior TLE file; first TLE is used.")
    synthetic_sgp4_sensor_parser.add_argument("--output-dir", required=True, help="Workflow artifact output directory.")
    synthetic_sgp4_sensor_parser.add_argument("--store", help="Optional Scale store path; defaults under --output-dir.")
    synthetic_sgp4_sensor_parser.add_argument("--object-id", default="synthetic_sgp4_target")
    synthetic_sgp4_sensor_parser.add_argument("--mean-anomaly-offset-deg", type=float, default=0.2)
    synthetic_sgp4_sensor_parser.add_argument("--mean-motion-scale", type=float, default=1.00002)
    synthetic_sgp4_sensor_parser.add_argument("--duration-s", type=float, default=720.0)
    synthetic_sgp4_sensor_parser.add_argument("--step-s", type=float, default=120.0)
    synthetic_sgp4_sensor_parser.add_argument("--station-spacing-s", type=float)
    synthetic_sgp4_sensor_parser.add_argument("--station-longitude-offset-deg", type=float, default=2.0)
    synthetic_sgp4_sensor_parser.add_argument("--angle-sigma-deg", type=float, default=0.02)
    synthetic_sgp4_sensor_parser.add_argument("--range-sigma-km", type=float, default=0.01)
    synthetic_sgp4_sensor_parser.add_argument("--range-rate-sigma-km-s", type=float, default=1.0e-5)
    synthetic_sgp4_sensor_parser.add_argument("--fit-duration-s", type=float)
    synthetic_sgp4_sensor_parser.add_argument("--holdout-duration-s", type=float)
    synthetic_sgp4_sensor_parser.add_argument(
        "--frame-transform",
        default=DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
        choices=("teme_as_eci", "teme_to_eci_iau80"),
    )
    synthetic_sgp4_sensor_parser.add_argument("--max-nfev", type=int, default=160)
    synthetic_sgp4_sensor_parser.add_argument("--json", action="store_true", help="Print the workflow payload as JSON.")

    synthetic_ground_sensor_parser = sub.add_parser(
        "run-synthetic-ground-sensor-od",
        help="Simulate noisy special-perturbation ground-station measurements and fit native sensor OD.",
    )
    synthetic_ground_sensor_parser.add_argument(
        "--output-dir", required=True, help="Workflow artifact output directory."
    )
    synthetic_ground_sensor_parser.add_argument(
        "--store", help="Optional Scale store path; defaults under --output-dir."
    )
    synthetic_ground_sensor_parser.add_argument("--object-id", default="synthetic_native_target")
    synthetic_ground_sensor_parser.add_argument(
        "--initial-state-eci-km-s",
        nargs=6,
        type=float,
        metavar=("X_KM", "Y_KM", "Z_KM", "VX_KM_S", "VY_KM_S", "VZ_KM_S"),
        default=[7000.0, 0.0, 0.0, 0.0, 7.546053290107541, 0.0],
    )
    synthetic_ground_sensor_parser.add_argument(
        "--prior-position-offset-m",
        nargs=3,
        type=float,
        metavar=("DX_M", "DY_M", "DZ_M"),
        default=[1000.0, 0.0, 0.0],
    )
    synthetic_ground_sensor_parser.add_argument(
        "--prior-velocity-offset-mm-s",
        nargs=3,
        type=float,
        metavar=("DVX_MM_S", "DVY_MM_S", "DVZ_MM_S"),
        default=[100.0, 0.0, 0.0],
    )
    synthetic_ground_sensor_parser.add_argument("--duration-s", type=float, default=3600.0)
    synthetic_ground_sensor_parser.add_argument("--step-s", type=float, default=120.0)
    synthetic_ground_sensor_parser.add_argument("--station-spacing-s", type=float)
    synthetic_ground_sensor_parser.add_argument("--station-longitude-offset-deg", type=float, default=2.0)
    synthetic_ground_sensor_parser.add_argument("--angle-sigma-deg", type=float, default=0.02)
    synthetic_ground_sensor_parser.add_argument("--range-sigma-km", type=float, default=0.01)
    synthetic_ground_sensor_parser.add_argument("--range-rate-sigma-km-s", type=float, default=1.0e-5)
    synthetic_ground_sensor_parser.add_argument("--fit-duration-s", type=float)
    synthetic_ground_sensor_parser.add_argument("--holdout-duration-s", type=float)
    synthetic_ground_sensor_parser.add_argument("--dynamics-model", default="two_body")
    synthetic_ground_sensor_parser.add_argument("--j2", action="store_true")
    synthetic_ground_sensor_parser.add_argument("--j3", action="store_true")
    synthetic_ground_sensor_parser.add_argument("--j4", action="store_true")
    synthetic_ground_sensor_parser.add_argument("--drag", action="store_true")
    synthetic_ground_sensor_parser.add_argument("--srp", action="store_true")
    synthetic_ground_sensor_parser.add_argument("--third-body-sun", action="store_true")
    synthetic_ground_sensor_parser.add_argument("--third-body-moon", action="store_true")
    synthetic_ground_sensor_parser.add_argument("--atmosphere-model", default="")
    synthetic_ground_sensor_parser.add_argument("--max-nfev", type=int, default=48)
    synthetic_ground_sensor_parser.add_argument(
        "--json", action="store_true", help="Print the workflow payload as JSON."
    )
