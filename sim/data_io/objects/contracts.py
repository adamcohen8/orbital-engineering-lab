"""An inspected object input; no scenario, executable plugin, or live object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim.data_io.catalog import CatalogRecord
from sim.data_io.contracts import AttitudeHistory, DataIOError, Trajectory
from sim.data_io.time import at, parse_time


@dataclass(frozen=True)
class ImportedObject:
    value: Trajectory | AttitudeHistory | CatalogRecord
    format: str
    sample_index: int = 0
    sources: tuple[dict[str, Any], ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.value, (Trajectory, AttitudeHistory, CatalogRecord)):
            raise DataIOError("An object input requires an orbital state, native mean elements, or attitude.")
        index = self.sample_index
        count = 1 if isinstance(self.value, CatalogRecord) else len(self.value.times_s)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            raise DataIOError("Select one valid integer object sample index.")

    @property
    def object_id(self):
        return self.value.object_id

    @property
    def epoch(self):
        if isinstance(self.value, CatalogRecord):
            return parse_time(str(self.value.fields["EPOCH"]))
        return at(self.value.epoch, self.value.times_s[self.sample_index])

    def verify_sources(self):
        from sim.data_io.common import source_receipt

        for identity in self.sources:
            if source_receipt(identity["path"]) != identity:
                raise DataIOError("An imported object source/resource changed; inspect and import it again.")
