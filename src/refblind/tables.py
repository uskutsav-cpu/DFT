"""Versioned observable-level study tables, with separate labels and features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .features import FEATURES, Stage, validate_feature_names
from .provenance import digest, read_json, write_json
from .schema import ReferenceQuality
from .validation import finite, identifier, nonnegative, probability


@dataclass(frozen=True)
class StudyRow:
    id: str
    group_id: str
    mlip: float | None
    dft: float | None
    reference: float | None
    mlip_features: dict[str, float | None]
    dft_features: dict[str, float | None]
    costs: dict[str, float]
    reference_quality: ReferenceQuality | str = ReferenceQuality.UNREVIEWED
    reference_uncertainty: float | None = None
    high_level_value: float | None = None
    high_level_supported: bool = True
    decision_group: str | None = None
    operations: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.id, "row id")
        identifier(self.group_id, "group id")
        object.__setattr__(self, "reference_quality", ReferenceQuality(self.reference_quality))
        for name in ("mlip", "dft", "reference", "high_level_value"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
        if self.reference_uncertainty is not None:
            nonnegative(self.reference_uncertainty, "reference_uncertainty")
        for features, stage in ((self.mlip_features, Stage.MLIP), (self.dft_features, Stage.DFT)):
            if features:
                validate_feature_names(list(features), stage)
            for name, value in features.items():
                if value is not None:
                    finite(value, f"feature {name}")
        if self.mlip_features.get("cheap_screen") is not None:
            probability(self.mlip_features["cheap_screen"], "cheap_screen")
        if set(self.mlip_features) & set(self.dft_features):
            raise ValueError("duplicate feature names across stages")
        if any(FEATURES[n].stage != Stage.DFT for n in self.dft_features):
            raise ValueError("DFT-stage map must contain DFT-stage features only")
        required = {"mlip", "dft", "high_level", "screen", "diagnostics"}
        if set(self.costs) != required:
            raise ValueError(f"costs must explicitly contain {sorted(required)}")
        for name, value in self.costs.items():
            nonnegative(value, f"cost {name}")
        for stage, operations in self.operations.items():
            if stage not in required or not operations:
                raise ValueError("invalid operation group")
            for value in operations.values():
                nonnegative(value, "operation cost")
            if abs(sum(operations.values()) - self.costs[stage]) > 1e-8:
                raise ValueError("per-operation and per-stage costs disagree")
        if not isinstance(self.high_level_supported, bool):
            raise ValueError("high_level_supported must be boolean")
        digest(self.metadata)

    @property
    def all_features(self) -> dict:
        return {**self.mlip_features, **self.dft_features}

    def stage_operations(self, stage: str) -> dict[str, float]:
        return self.operations.get(stage, {f"{self.id}/{stage}": self.costs[stage]})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StudyTable:
    rows: list[StudyRow]
    metadata: dict
    energy_unit: str = "kcal/mol"
    cost_unit: str = "relative"
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.schema_version != "1.0" or self.energy_unit != "kcal/mol":
            raise ValueError("study tables require schema 1.0 and kcal/mol observables")
        if not self.rows or len({r.id for r in self.rows}) != len(self.rows):
            raise ValueError("study rows must be nonempty with unique IDs")
        if not self.cost_unit:
            raise ValueError("cost_unit is required")
        seen, operations = {}, {}
        for row in self.rows:
            if row.decision_group:
                previous = seen.setdefault(row.decision_group, row.group_id)
                if previous != row.group_id:
                    raise ValueError("decision group crosses a split group")
            for group in row.operations.values():
                for key, value in group.items():
                    if key in operations and operations[key] != value:
                        raise ValueError("inconsistent shared job cost")
                    operations[key] = value
        digest(self.metadata)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "energy_unit": self.energy_unit,
            "cost_unit": self.cost_unit,
            "metadata": self.metadata,
            "rows": [row.to_dict() for row in self.rows],
        }

    @property
    def fingerprint(self) -> str:
        return digest(self.to_dict())

    def save(self, path: str | Path):
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> StudyTable:
        value = read_json(path)
        value["rows"] = [StudyRow(**row) for row in value["rows"]]
        return cls(**value)


def target_label(
    predicted: float, reference: float, tolerance: float, reference_uncertainty: float | None = None
) -> int | None:
    """Conservative label: None if the reference interval straddles tolerance.

    An omitted uncertainty yields a nominal label, NOT an assertion of exactness.
    """
    error = abs(finite(predicted) - finite(reference))
    tolerance = nonnegative(tolerance)
    u = 0.0 if reference_uncertainty is None else nonnegative(reference_uncertainty)
    if error - u > tolerance:
        return 1
    if error + u <= tolerance:
        return 0
    return None
