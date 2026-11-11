# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *

from sim.execution.parameter_paths import validate_parameter_path_claims, validate_parameter_path_exists

import math
import operator
from statistics import NormalDist


def sample_monte_carlo_variation(variation: Any, rng: np.random.Generator) -> Any:
    mode = variation.mode.lower()
    if mode == "choice":
        if not variation.options:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=choice requires options.")
        if variation.weights:
            weights = np.asarray(variation.weights, dtype=float)
            scaled = weights / weights.max()
            index = int(rng.choice(len(variation.options), p=scaled / scaled.sum()))
        else:
            index = int(rng.integers(0, len(variation.options)))
        return deepcopy(variation.options[index])
    if mode == "uniform":
        if variation.low is None or variation.high is None:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=uniform requires low/high.")
        return float(rng.uniform(variation.low, variation.high))
    if mode == "normal":
        if variation.mean is None or variation.std is None:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=normal requires mean/std.")
        return float(rng.normal(variation.mean, variation.std))
    if mode == "truncated_normal":
        low_z = (float(variation.low) - float(variation.mean)) / float(variation.std)
        high_z = (float(variation.high) - float(variation.mean)) / float(variation.std)
        standard = NormalDist()
        unit = float(rng.random())
        if low_z >= 0.0:
            # Work in the survival function for upper tails; cdf(10) and
            # cdf(11) both round to one in ordinary double precision.
            lower_tail = 0.5 * math.erfc(low_z / math.sqrt(2.0))
            upper_tail = 0.5 * math.erfc(high_z / math.sqrt(2.0))
            interval = lower_tail - upper_tail
            if interval <= 0.0 or not math.isfinite(interval):
                raise ValueError(
                    f"Variation '{variation.parameter_path}' has no representable probability in its bounds."
                )
            tail_probability = lower_tail - unit * interval
            z = -standard.inv_cdf(max(tail_probability, math.nextafter(0.0, 1.0)))
        elif high_z <= 0.0:
            # Reflect the negative tail so Phi(-10) and Phi(-11) do not
            # underflow to zero in the ordinary CDF calculation.
            lower_tail = 0.5 * math.erfc((-high_z) / math.sqrt(2.0))
            upper_tail = 0.5 * math.erfc((-low_z) / math.sqrt(2.0))
            interval = lower_tail - upper_tail
            if interval <= 0.0 or not math.isfinite(interval):
                raise ValueError(
                    f"Variation '{variation.parameter_path}' has no representable probability in its bounds."
                )
            tail_probability = lower_tail - unit * interval
            z = standard.inv_cdf(max(tail_probability, math.nextafter(0.0, 1.0)))
        else:
            lower = standard.cdf(low_z)
            upper = standard.cdf(high_z)
            interval = upper - lower
            if interval <= 0.0 or not math.isfinite(interval):
                raise ValueError(
                    f"Variation '{variation.parameter_path}' has no representable probability in its bounds."
                )
            probability = lower + interval * unit
            z = standard.inv_cdf(min(max(probability, math.nextafter(0.0, 1.0)), math.nextafter(1.0, 0.0)))
        if not math.isfinite(z):
            raise ValueError(f"Variation '{variation.parameter_path}' has no representable probability in its bounds.")
        return min(max(float(variation.mean) + float(variation.std) * z, variation.low), variation.high)
    raise ValueError(f"Unsupported variation mode '{variation.mode}'.")


def _draw_bundle_option(bundle: Any, rng: np.random.Generator) -> Any:
    weights = np.asarray([option.weight for option in bundle.options], dtype=float)
    scaled = weights / weights.max()
    return bundle.options[int(rng.choice(len(bundle.options), p=scaled / scaled.sum()))]


def _parameter_value(root: dict[str, Any], path: str) -> Any:
    value: Any = root
    for token in path.split("."):
        if "[" in token and token.endswith("]"):
            key, index_text = token[:-1].split("[", 1)
            if key:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(f"Monte Carlo constraint path '{path}' does not exist.")
                value = value[key]
            try:
                value = value[int(index_text)]
            except (TypeError, ValueError, IndexError, KeyError) as exc:
                raise ValueError(f"Monte Carlo constraint path '{path}' does not exist.") from exc
        else:
            if not isinstance(value, dict) or token not in value:
                raise ValueError(f"Monte Carlo constraint path '{path}' does not exist.")
            value = value[token]
    return value


_COMPARATORS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt,
    ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
}


def _constraints_hold(root: dict[str, Any], constraints: list[Any]) -> bool:
    for constraint in constraints:
        left = _parameter_value(root, constraint.left_path)
        right = _parameter_value(root, constraint.right_path) if constraint.right_path is not None else constraint.right_value
        if constraint.op in {"<", "<=", ">", ">="}:
            if isinstance(left, bool) or isinstance(right, bool):
                raise ValueError(f"Monte Carlo constraint '{constraint.left_path}' needs numeric operands.")
            try:
                left, right = float(left), float(right)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Monte Carlo constraint '{constraint.left_path}' needs numeric operands.") from exc
            if not math.isfinite(left) or not math.isfinite(right):
                return False
        if not _COMPARATORS[constraint.op](left, right):
            return False
    return True


def _candidate_config(
    root: dict[str, Any], variations: list[Any], bundles: list[Any],
    correlated: list[tuple[Any, np.ndarray]], rng: np.random.Generator,
    *, fixed_bundle: tuple[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config = deepcopy(root)
    sampled: dict[str, Any] = {}
    labels: dict[str, str] = {}
    for variation in variations:
        value = sample_monte_carlo_variation(variation, rng)
        set_parameter_path_value(config, variation.parameter_path, value)
        sampled[variation.parameter_path] = value
    for group, transform in correlated:
        draws = np.asarray(group.means) + np.asarray(group.stds) * (transform @ rng.normal(size=len(group.parameter_paths)))
        for path, value in zip(group.parameter_paths, draws, strict=True):
            scalar = float(value)
            set_parameter_path_value(config, path, scalar)
            sampled[path] = scalar
    for bundle in bundles:
        option = fixed_bundle[1] if fixed_bundle is not None and bundle.name == fixed_bundle[0] else _draw_bundle_option(bundle, rng)
        labels[bundle.name] = option.label
        for path, value in option.values.items():
            set_parameter_path_value(config, path, deepcopy(value), allow_input_file=True)
            sampled[path] = value
    return config, sampled, labels


def _finalize_prepared_run(
    *, config: dict[str, Any], sampled: dict[str, Any], labels: dict[str, str],
    iteration: int, seed: int, outdir: Path, attempts: int,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .campaign_retention import configure_iteration

    if "metadata.seed" not in sampled:
        config.setdefault("metadata", {})["seed"] = seed
    if str(config.get("outputs", {}).get("mode", "interactive")) == "interactive":
        config.setdefault("outputs", {})["mode"] = "save"
    config.setdefault("outputs", {})["output_dir"] = str(outdir / f"mc_run_{iteration:04d}")
    configure_iteration(config)
    return {
        "iteration": iteration,
        "sampled_parameters": sampled,
        "bundle_options": labels,
        "comparison": comparison,
        "sampling_attempts": attempts,
        "config_dict": config,
        "config_hash": _config_fingerprint(config),
        "seed": int(config.get("metadata", {}).get("seed", seed)),
    }


def prepare_monte_carlo_runs(
    *, cfg: SimulationScenarioConfig, root: dict[str, Any], outdir: Path,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.monte_carlo.base_seed))
    variations = list(cfg.monte_carlo.variations)
    bundles = list(cfg.monte_carlo.bundles)
    constraints = list(cfg.monte_carlo.constraints)
    claims: list[tuple[str, str, bool]] = []
    for index, variation in enumerate(variations):
        claims.append((f"variation[{index}]", variation.parameter_path, False))
    for bundle in bundles:
        for path in bundle.options[0].values:
            claims.append((f"bundle[{bundle.name}]", path, True))
    for group in cfg.monte_carlo.correlated_normals:
        for path in group.parameter_paths:
            claims.append((f"correlated_normal[{group.name}]", path, False))
    validate_parameter_path_claims(root, claims)
    for constraint in constraints:
        validate_parameter_path_exists(root, constraint.left_path)
        if constraint.right_path is not None:
            validate_parameter_path_exists(root, constraint.right_path)
    correlated = []
    for group in cfg.monte_carlo.correlated_normals:
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(group.correlation, dtype=float))
        transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        correlated.append((group, transform))
    max_attempts = int(cfg.monte_carlo.max_attempts)
    comparison = cfg.monte_carlo.comparison
    prepared: list[dict[str, Any]] = []

    if comparison is not None and comparison.paired:
        compared_bundle = next(bundle for bundle in bundles if bundle.name == comparison.bundle)
        remaining_bundles = [bundle for bundle in bundles if bundle.name != compared_bundle.name]
        for replicate in range(comparison.runs_per_option):
            for _attempt in range(1, max_attempts + 1):
                shared_config, shared_sampled, shared_labels = _candidate_config(
                    root, variations, remaining_bundles, correlated, rng,
                )
                candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, str]]] = []
                for option in compared_bundle.options:
                    config = deepcopy(shared_config)
                    sampled = dict(shared_sampled)
                    labels = dict(shared_labels)
                    labels[compared_bundle.name] = option.label
                    for path, value in option.values.items():
                        set_parameter_path_value(config, path, deepcopy(value), allow_input_file=True)
                        sampled[path] = value
                    candidates.append((config, sampled, labels))
                if all(_constraints_hold(config, constraints) for config, _, _ in candidates):
                    break
            else:
                raise ValueError(f"Monte Carlo paired replicate {replicate} found no valid shared draw in {max_attempts} attempts.")
            seed = int(cfg.monte_carlo.base_seed) + replicate
            for option_index, (config, sampled, labels) in enumerate(candidates):
                iteration = replicate * len(candidates) + option_index
                prepared.append(_finalize_prepared_run(
                    config=config, sampled=sampled, labels=labels, iteration=iteration,
                    seed=seed, outdir=outdir, attempts=_attempt,
                    comparison={"bundle": compared_bundle.name, "option": compared_bundle.options[option_index].label,
                                "replicate": replicate, "paired": True},
                ))
        return prepared

    compared_bundle = next((bundle for bundle in bundles if comparison is not None and bundle.name == comparison.bundle), None)
    for iteration in range(int(cfg.monte_carlo.iterations)):
        replicate = iteration // len(compared_bundle.options) if compared_bundle is not None else None
        fixed = (
            (compared_bundle.name, compared_bundle.options[iteration % len(compared_bundle.options)])
            if compared_bundle is not None else None
        )
        for _attempt in range(1, max_attempts + 1):
            config, sampled, labels = _candidate_config(root, variations, bundles, correlated, rng, fixed_bundle=fixed)
            if _constraints_hold(config, constraints):
                break
        else:
            raise ValueError(f"Monte Carlo iteration {iteration} found no valid draw in {max_attempts} attempts.")
        comparison_info = None
        if compared_bundle is not None:
            comparison_info = {"bundle": compared_bundle.name, "option": labels[compared_bundle.name],
                               "replicate": replicate, "paired": False}
        prepared.append(_finalize_prepared_run(
            config=config, sampled=sampled, labels=labels, iteration=iteration,
            seed=int(cfg.monte_carlo.base_seed) + iteration, outdir=outdir,
            attempts=_attempt, comparison=comparison_info,
        ))
    return prepared


__all__ = [name for name in globals() if not name.startswith("__")]
