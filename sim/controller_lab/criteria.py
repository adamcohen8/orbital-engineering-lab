# ruff: noqa: F401,F403,F405,I001
from .bench_common import *
from .config import *

def _evaluate_pass_criteria(
    metrics: dict[str, Any], criteria: tuple[ControllerBenchPassCriterion, ...]
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for criterion in criteria:
        lhs = metrics.get(criterion.metric)
        rhs = criterion.value
        ok = False
        if criterion.op == "<=":
            ok = lhs is not None and float(lhs) <= float(rhs)
        elif criterion.op == "<":
            ok = lhs is not None and float(lhs) < float(rhs)
        elif criterion.op == ">=":
            ok = lhs is not None and float(lhs) >= float(rhs)
        elif criterion.op == ">":
            ok = lhs is not None and float(lhs) > float(rhs)
        elif criterion.op == "==":
            ok = lhs == rhs
        elif criterion.op == "!=":
            ok = lhs != rhs
        else:
            raise ValueError(f"Unsupported pass-criterion op: {criterion.op}")
        if not ok:
            failures.append(f"{criterion.metric} {criterion.op} {criterion.value}")
    return (len(failures) == 0, failures)


def _safe_metric_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _evaluate_single_pass_criterion(
    value: Any, criterion: ControllerBenchPassCriterion
) -> tuple[bool, str, float | None]:
    value_f = _safe_metric_float(value)
    threshold_f = _safe_metric_float(criterion.value)
    if value is None:
        return False, "missing_metric", None
    if value_f is not None and not math.isfinite(value_f):
        return False, "invalid_metric", None
    if criterion.op in {"<=", "<", ">=", ">"}:
        if value_f is None or threshold_f is None:
            return False, "non_numeric_metric", None
        if criterion.op == "<=":
            ok = value_f <= threshold_f
            margin = threshold_f - value_f
        elif criterion.op == "<":
            ok = value_f < threshold_f
            margin = threshold_f - value_f
        elif criterion.op == ">=":
            ok = value_f >= threshold_f
            margin = value_f - threshold_f
        else:
            ok = value_f > threshold_f
            margin = value_f - threshold_f
        return bool(ok), "passed" if ok else "threshold_violation", float(margin)
    if criterion.op == "==":
        ok = value == criterion.value
        margin = 0.0 if ok else None
        return bool(ok), "passed" if ok else "equality_violation", margin
    if criterion.op == "!=":
        ok = value != criterion.value
        margin = 0.0 if ok else None
        return bool(ok), "passed" if ok else "equality_violation", margin
    raise ValueError(f"Unsupported pass-criterion op: {criterion.op}")


def _criterion_failure_records(
    metrics: dict[str, Any],
    criteria: tuple[ControllerBenchPassCriterion, ...],
    *,
    objective_name: str = "",
    objective_kind: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for criterion in criteria:
        value = metrics.get(criterion.metric)
        ok, reason, margin = _evaluate_single_pass_criterion(value, criterion)
        if ok:
            continue
        out.append(
            {
                "metric": criterion.metric,
                "op": criterion.op,
                "threshold": criterion.value,
                "value": value,
                "reason": reason,
                "margin": margin,
                "objective": objective_name,
                "objective_kind": objective_kind,
            }
        )
    return out

__all__ = [name for name in globals() if not name.startswith("__")]
