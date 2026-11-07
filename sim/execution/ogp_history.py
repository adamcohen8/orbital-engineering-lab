"""Public OGP-only native work inside one scenario's deterministic timeline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from sim.execution.passive_history import _PreparedHistory
from sim.rust_ogp_backend import propagate_batch_with_telemetry


class OGPHistoryPreparation:
    """Batch bounded object rows; scenario state and all callbacks stay on owner."""

    def __init__(self, engine: Any, *, workers: int) -> None:
        self.engine = engine
        self.workers = workers

    def prepare(self, inputs: list[Any]) -> None:
        cache = self.engine.satellite_stepper.passive_history
        # 16 rows x 257 epochs clears the native 4096-sample crossover. Cap
        # temporary allocation independently of catalog size and CPU count.
        group_size = min(64, max(16, self.workers * 4))
        pending: list[Any] = []
        for item in inputs:
            if item.object_id not in self.engine.general_propagation:
                continue
            definition = cache._definition(item.agent, item.initial_truth)
            if definition is None or cache._matching_entry(
                item.object_id, definition, item.initial_truth, item.t_s, item.t_next,
            ) is not None:
                continue
            intervals, times = cache.ogp_schedule(definition, item.initial_truth, item.t_s, item.t_next)
            # Different clocks/cadences must never share the wrong row shape.
            if pending and len(times) != len(pending[0][3]):
                self._group(pending)
                pending = []
            pending.append((item, definition, intervals, times))
            if len(pending) >= group_size:
                self._group(pending)
                pending = []
        if pending:
            self._group(pending)

    def _group(self, group: list[Any]) -> None:
        started = perf_counter()
        providers = [self.engine.general_propagation[item.object_id] for item, *_ in group]
        offsets = np.asarray([
            (float(provider.start_jd_utc) - float(provider.elements.epoch_jd_utc)) * 1440.0
            + np.asarray(times) / 60.0
            for provider, (_, _, _, times) in zip(providers, group)
        ])
        try:
            positions, velocities, errors, telemetry = propagate_batch_with_telemetry(
                [provider.elements for provider in providers], offsets, workers=self.workers,
            )
        except (ValueError, ArithmeticError, RuntimeError):
            # Speculative future failures belong to the ordinary current step.
            return
        plan = self.engine.object_execution_plan
        used = max(int(plan["selected_workers"]), int(telemetry["workers_used"]))
        plan.update(
            selected_workers=used, selected_backend="native_threads" if used > 1 else "serial",
            reason="OGP native history preparation; deterministic owner stepping",
            ogp_batches=int(plan.get("ogp_batches", 0)) + 1, last_ogp_batch=telemetry,
        )
        cache = self.engine.satellite_stepper.passive_history
        for row, (provider, (item, definition, intervals, times)) in enumerate(zip(providers, group)):
            if np.any(errors[row] != ""):
                continue
            try:
                states = provider.canonical_states_from_teme(times, positions[row], velocities[row])
            except (ValueError, ArithmeticError, RuntimeError):
                continue
            cache.entries[item.object_id] = _PreparedHistory(definition, intervals, times, states)
            cache.preparation_count += 1
        self.engine.runtime_profiler.record_stage("ogp_history_prepare", perf_counter() - started)
