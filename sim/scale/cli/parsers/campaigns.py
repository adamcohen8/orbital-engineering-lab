from __future__ import annotations

import argparse


def register_campaigns_parsers(sub: argparse._SubParsersAction) -> None:
    sensitivity_parser = sub.add_parser(
        "materialize-sensitivity", help="Materialize OGP-SGP4 sensitivity campaign trials."
    )
    sensitivity_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    sensitivity_parser.add_argument("--json", action="store_true", help="Print the sensitivity payload as JSON.")

    sensitivity_serial_parser = sub.add_parser(
        "run-sensitivity-serial", help="Run the serial OGP-SGP4 sensitivity trial contract."
    )
    sensitivity_serial_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    sensitivity_serial_parser.add_argument(
        "--limit", type=int, default=0, help="Optional maximum number of trials to process."
    )
    sensitivity_serial_parser.add_argument(
        "--propagate-sgp4",
        action="store_true",
        help="Propagate fitted synthetic SGP4 trial products and summarize separation.",
    )
    sensitivity_serial_parser.add_argument(
        "--duration-s", type=float, help="Override propagation duration for this run."
    )
    sensitivity_serial_parser.add_argument(
        "--step-s", type=float, help="Override propagation sample step for this run."
    )
    sensitivity_serial_parser.add_argument(
        "--propagation-summary-backend",
        default="scalar_trial",
        choices=["scalar_trial", "numba_chunked"],
        help="Backend used to create per-trial propagation summary metrics.",
    )
    sensitivity_serial_parser.add_argument(
        "--chunk-duration-s",
        type=float,
        help="Maximum time span per chunk for numba_chunked propagation summaries.",
    )
    sensitivity_serial_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run trials whose latest result for this run is blocked or error.",
    )
    sensitivity_serial_parser.add_argument("--json", action="store_true", help="Print the serial run payload as JSON.")

    sensitivity_parallel_parser = sub.add_parser(
        "run-sensitivity-parallel-cpu",
        help="Run the OGP-SGP4 sensitivity trial contract with local CPU workers.",
    )
    sensitivity_parallel_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    sensitivity_parallel_parser.add_argument(
        "--limit", type=int, default=0, help="Optional maximum number of trials to process."
    )
    sensitivity_parallel_parser.add_argument(
        "--workers", type=int, default=0, help="Number of local campaign worker slots."
    )
    sensitivity_parallel_parser.add_argument(
        "--backend",
        choices=("parallel_cpu", "process_pool", "thread_pool"),
        default="parallel_cpu",
        help="Execution transport; process_pool/parallel_cpu is the default and thread_pool is explicit opt-in.",
    )
    sensitivity_parallel_parser.add_argument(
        "--propagate-sgp4",
        action="store_true",
        help="Propagate fitted synthetic SGP4 trial products and summarize separation.",
    )
    sensitivity_parallel_parser.add_argument(
        "--duration-s", type=float, help="Override propagation duration for this run."
    )
    sensitivity_parallel_parser.add_argument(
        "--step-s", type=float, help="Override propagation sample step for this run."
    )
    sensitivity_parallel_parser.add_argument(
        "--propagation-summary-backend",
        default="scalar_trial",
        choices=["scalar_trial", "numba_chunked"],
        help="Backend used to create per-trial propagation summary metrics.",
    )
    sensitivity_parallel_parser.add_argument(
        "--chunk-duration-s",
        type=float,
        help="Maximum time span per chunk for numba_chunked propagation summaries.",
    )
    sensitivity_parallel_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run trials whose latest result for this run is blocked or error.",
    )
    sensitivity_parallel_parser.add_argument(
        "--json", action="store_true", help="Print the parallel run payload as JSON."
    )

    monte_carlo_parser = sub.add_parser(
        "materialize-monte-carlo", help="Materialize OGP-SGP4 Monte Carlo campaign trials."
    )
    monte_carlo_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    monte_carlo_parser.add_argument("--json", action="store_true", help="Print the Monte Carlo payload as JSON.")

    mc_serial_parser = sub.add_parser(
        "run-monte-carlo-serial", help="Run the serial OGP-SGP4 Monte Carlo trial contract."
    )
    mc_serial_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    mc_serial_parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of trials to process.")
    mc_serial_parser.add_argument(
        "--propagate-sgp4",
        action="store_true",
        help="Propagate fitted synthetic SGP4 trial products and summarize separation.",
    )
    mc_serial_parser.add_argument("--duration-s", type=float, help="Override propagation duration for this serial run.")
    mc_serial_parser.add_argument("--step-s", type=float, help="Override propagation sample step for this serial run.")
    mc_serial_parser.add_argument(
        "--propagation-summary-backend",
        default="scalar_trial",
        choices=["scalar_trial", "numba_chunked"],
        help="Backend used to create per-trial propagation summary metrics.",
    )
    mc_serial_parser.add_argument(
        "--chunk-duration-s",
        type=float,
        help="Maximum time span per chunk for numba_chunked propagation summaries.",
    )
    mc_serial_parser.add_argument(
        "--sgp4-batch-backend",
        default="",
        choices=["", "scalar_reference", "numba"],
        help="Optional batched SGP4 parity backend to run after per-trial propagation.",
    )
    mc_serial_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run trials whose latest result for this run is blocked or error.",
    )
    mc_serial_parser.add_argument("--json", action="store_true", help="Print the serial run payload as JSON.")

    mc_parallel_parser = sub.add_parser(
        "run-monte-carlo-parallel-cpu",
        help="Run the OGP-SGP4 Monte Carlo trial contract with local CPU workers.",
    )
    mc_parallel_parser.add_argument("config", help="OEL Scale YAML config with mode: run.")
    mc_parallel_parser.add_argument(
        "--limit", type=int, default=0, help="Optional maximum number of trials to process."
    )
    mc_parallel_parser.add_argument("--workers", type=int, default=0, help="Number of local campaign worker slots.")
    mc_parallel_parser.add_argument(
        "--backend",
        choices=("parallel_cpu", "process_pool", "thread_pool"),
        default="parallel_cpu",
        help="Execution transport; process_pool/parallel_cpu is the default and thread_pool is explicit opt-in.",
    )
    mc_parallel_parser.add_argument(
        "--propagate-sgp4",
        action="store_true",
        help="Propagate fitted synthetic SGP4 trial products and summarize separation.",
    )
    mc_parallel_parser.add_argument("--duration-s", type=float, help="Override propagation duration for this run.")
    mc_parallel_parser.add_argument("--step-s", type=float, help="Override propagation sample step for this run.")
    mc_parallel_parser.add_argument(
        "--propagation-summary-backend",
        default="scalar_trial",
        choices=["scalar_trial", "numba_chunked"],
        help="Backend used to create per-trial propagation summary metrics.",
    )
    mc_parallel_parser.add_argument(
        "--chunk-duration-s",
        type=float,
        help="Maximum time span per chunk for numba_chunked propagation summaries.",
    )
    mc_parallel_parser.add_argument(
        "--sgp4-batch-backend",
        default="",
        choices=["", "scalar_reference", "numba"],
        help="Optional batched SGP4 parity backend to run after per-trial propagation.",
    )
    mc_parallel_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run trials whose latest result for this run is blocked or error.",
    )
    mc_parallel_parser.add_argument("--json", action="store_true", help="Print the parallel run payload as JSON.")
