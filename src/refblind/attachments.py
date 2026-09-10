"""Explicitly attach costed, provenance-bearing auxiliary features to a table.

No feature values are inferred from labels. This imports independently computed
features (for example a separate FOD run) and preserves an audit trail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .features import FEATURES, Stage, validate_feature_names
from .provenance import digest
from .tables import StudyTable
from .validation import identifier, nonnegative


@dataclass(frozen=True)
class FeatureAttachment:
    row_id: str
    stage: str
    features: dict[str, float | None]
    operations: dict[str, float]
    provenance: dict

    def __post_init__(self):
        identifier(self.row_id, "row_id")
        if self.stage not in {"screen", "diagnostics"}:
            raise ValueError("auxiliary features must be screen or diagnostics stage")
        stage = Stage.MLIP if self.stage == "screen" else Stage.DFT
        validate_feature_names(list(self.features), stage)
        if self.stage == "diagnostics" and any(
            FEATURES[n].stage != Stage.DFT for n in self.features
        ):
            raise ValueError("diagnostic attachments must contain DFT-stage features")
        if not self.operations:
            raise ValueError(
                "explicit computation operations/costs are required, even if zero cost"
            )
        for name, cost in self.operations.items():
            if not isinstance(name, str) or not name:
                raise ValueError("operation identity is required")
            nonnegative(cost, "operation cost")
        if not self.provenance.get("method") or not self.provenance.get("source_sha256"):
            raise ValueError("feature provenance requires method and source_sha256")
        source_hash = self.provenance["source_sha256"]
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(c not in "0123456789abcdef" for c in source_hash)
        ):
            raise ValueError("source_sha256 must be a SHA-256 digest")
        digest(asdict(self))


def attach_features(table: StudyTable, attachments: list[FeatureAttachment]) -> StudyTable:
    rows = {row.id: row for row in table.rows}
    audit = []
    seen = set()
    for item in attachments:
        if item.row_id not in rows:
            raise ValueError(f"unknown attachment row: {item.row_id}")
        row = rows[item.row_id]
        features = dict(row.mlip_features if item.stage == "screen" else row.dft_features)
        for name, value in item.features.items():
            key = (item.row_id, name)
            if key in seen or features.get(name) is not None:
                raise ValueError(f"refusing to overwrite feature {name} for {row.id}")
            seen.add(key)
            features[name] = value
        operations = {k: dict(v) for k, v in row.operations.items()}
        existing = operations.get(item.stage, {})
        if not existing and row.costs[item.stage] != 0:
            existing = row.stage_operations(item.stage)
        for key, value in item.operations.items():
            if key in existing and existing[key] != value:
                raise ValueError("inconsistent shared auxiliary operation cost")
            existing[key] = value
        operations[item.stage] = existing
        costs = dict(row.costs)
        costs[item.stage] = sum(existing.values())
        kwargs = {"mlip_features" if item.stage == "screen" else "dft_features": features}
        rows[row.id] = replace(row, operations=operations, costs=costs, **kwargs)
        audit.append(asdict(item))
    return StudyTable(
        [rows[row.id] for row in table.rows],
        {
            **table.metadata,
            "parent_table_hash": table.fingerprint,
            "feature_attachments": table.metadata.get("feature_attachments", []) + audit,
        },
        cost_unit=table.cost_unit,
    )
