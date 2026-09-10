"""Reproducible grouped partitions and connected-component grouping helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .provenance import digest
from .validation import integer, probability


@dataclass(frozen=True)
class SplitPlan:
    assignments: dict[str, str]
    groups: dict[str, str]
    seed: int
    fractions: dict[str, float]

    def ids(self, partition: str) -> list[str]:
        return sorted(k for k, v in self.assignments.items() if v == partition)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return digest(self.to_dict())

    def validate(self, row_ids: list[str], groups: list[str]) -> None:
        if len(row_ids) != len(groups) or len(set(row_ids)) != len(row_ids):
            raise ValueError("row IDs and groups must align; row IDs must be unique")
        if set(self.assignments) != set(row_ids) or self.groups != dict(zip(row_ids, groups)):
            raise ValueError("split plan does not match the current data")
        per_group = defaultdict(set)
        for row, group in zip(row_ids, groups):
            partition = self.assignments[row]
            if partition not in self.fractions:
                raise ValueError("unknown partition")
            per_group[group].add(partition)
        if any(len(v) != 1 for v in per_group.values()):
            raise ValueError("group leakage across partitions")
        if any(not self.ids(p) for p in self.fractions):
            raise ValueError("empty partition")


def grouped_split(
    row_ids: list[str],
    groups: list[str],
    *,
    seed: int = 42,
    fractions: dict[str, float] | None = None,
) -> SplitPlan:
    integer(seed, "seed", 0)
    if not row_ids or len(row_ids) != len(groups) or len(set(row_ids)) != len(row_ids):
        raise ValueError("nonempty unique row IDs and aligned groups required")
    if any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("every row requires a non-empty group id")
    fractions = fractions or {"train": 0.5, "calibration": 0.2, "validation": 0.15, "test": 0.15}
    for value in fractions.values():
        probability(value, "split fraction")
        if value == 0:
            raise ValueError("each requested partition needs a positive fraction")
    if abs(sum(fractions.values()) - 1) > 1e-10:
        raise ValueError("fractions must sum to one")
    unique = sorted(set(groups))
    if len(unique) < len(fractions):
        raise ValueError("too few independent groups for the requested partitions")
    shuffled = np.random.default_rng(seed).permutation(unique).tolist()
    # Reserve one group per partition, distribute the remainder by largest remainder.
    remaining = len(unique) - len(fractions)
    desired = np.array(list(fractions.values())) * remaining
    counts = np.floor(desired).astype(int) + 1
    for i in np.argsort(-(desired - np.floor(desired)), kind="stable")[
        : len(unique) - counts.sum()
    ]:
        counts[i] += 1
    group_partition, cursor = {}, 0
    for partition, count in zip(fractions, counts):
        for group in shuffled[cursor : cursor + count]:
            group_partition[group] = partition
        cursor += count
    assignments = {r: group_partition[g] for r, g in zip(row_ids, groups)}
    plan = SplitPlan(assignments, dict(zip(row_ids, groups)), seed, fractions)
    plan.validate(row_ids, groups)
    return plan


def connected_groups(row_entities: dict[str, list[str]]) -> dict[str, str]:
    """Group rows sharing caller-selected entities, e.g. a parent reaction or TS.

    Avoid grouping on ubiquitous reagents unless that conservatism is intended.
    """
    parent = {row: row for row in row_entities}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for row, entities in row_entities.items():
        for entity in entities:
            if entity in owners:
                a, b = find(row), find(owners[entity])
                parent[max(a, b)] = min(a, b)
            else:
                owners[entity] = row
    return {row: find(row) for row in row_entities}
