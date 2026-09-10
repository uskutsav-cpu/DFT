"""Stage-aware features and training-only preprocessing.

A feature's availability is an explicit contract, not inferred from its name.
The whitelist is a guardrail, not a security boundary against deliberately
mislabelled data. Chemical provenance still requires human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .validation import finite, nonnegative


class Stage(IntEnum):
    MLIP = 0
    DFT = 1
    LABEL_ONLY = 2


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    stage: Stage
    description: str
    cost_included: bool = True


FEATURES = {
    "ensemble_std": FeatureSpec("ensemble_std", Stage.MLIP, "Observable-level MLIP disagreement"),
    "force_disagreement": FeatureSpec(
        "force_disagreement", Stage.MLIP, "Maximum force committee spread"
    ),
    "ood_distance": FeatureSpec("ood_distance", Stage.MLIP, "Train-fitted representation distance"),
    "cheap_screen": FeatureSpec(
        "cheap_screen", Stage.MLIP, "Separately computed and costed prescreen"
    ),
    "homo_lumo_gap_ev": FeatureSpec(
        "homo_lumo_gap_ev", Stage.DFT, "Minimum spin-channel orbital gap"
    ),
    "spin_contamination": FeatureSpec(
        "spin_contamination", Stage.DFT, "Absolute S2 minus target S(S+1)"
    ),
    "scf_cycles": FeatureSpec(
        "scf_cycles", Stage.DFT, "SCF iteration count, not a calibrated risk"
    ),
    "fod": FeatureSpec("fod", Stage.DFT, "N_FOD from a separate, costed FOD calculation"),
    "method_disagreement": FeatureSpec(
        "method_disagreement", Stage.DFT, "Separate costed DFA comparison"
    ),
    "t1_diagnostic": FeatureSpec(
        "t1_diagnostic", Stage.LABEL_ONLY, "Correlated-method label audit only"
    ),
    "reference_error": FeatureSpec("reference_error", Stage.LABEL_ONLY, "Evaluation target only"),
    "high_level_energy": FeatureSpec(
        "high_level_energy", Stage.LABEL_ONLY, "Evaluation target only"
    ),
}


def validate_feature_names(names: list[str], available_stage: Stage) -> None:
    if not names or len(set(names)) != len(names):
        raise ValueError("feature names must be non-empty and unique")
    for name in names:
        if name not in FEATURES:
            raise ValueError(
                f"unknown feature {name!r}; explicitly audit/register its availability"
            )
        feature = FEATURES[name]
        if feature.stage == Stage.LABEL_ONLY or feature.stage > available_stage:
            raise ValueError(f"feature {name!r} is unavailable before {available_stage.name}")
        if not feature.cost_included:
            raise ValueError(f"feature {name!r} has unaccounted computation cost")


def spin_contamination(s2: float, multiplicity: int) -> float:
    from .validation import integer

    integer(multiplicity, "multiplicity", 1)
    s = (multiplicity - 1) / 2
    return abs(nonnegative(s2, "S2") - s * (s + 1))


def orbital_gap_ev(energies_ev, occupations, *, occupancy_threshold: float = 1e-6) -> float:
    from .validation import positive, vector

    positive(occupancy_threshold, "occupancy_threshold")
    energies = vector(energies_ev, "orbital energies")
    occupation = vector(occupations, "occupations")
    if energies.shape != occupation.shape or (occupation < 0).any():
        raise ValueError("orbital energies/occupations must align and occupations be nonnegative")
    filled = energies[occupation > occupancy_threshold]
    empty = energies[occupation <= occupancy_threshold]
    if filled.size == 0 or empty.size == 0:
        raise ValueError("both occupied and virtual orbitals are needed")
    return float(empty.min() - filled.max())


class FeaturePreprocessor:
    """Train-only median imputation + z-scaling + explicit missing indicators."""

    def __init__(self, names: list[str], stage: Stage):
        validate_feature_names(names, stage)
        self.names = list(names)
        self.stage = Stage(stage)
        self.medians = None
        self.means = None
        self.scales = None
        self.all_missing = None

    def _raw(self, rows: list[dict]) -> np.ndarray:
        if not rows:
            raise ValueError("no feature rows")
        values = np.empty((len(rows), len(self.names)))
        for i, row in enumerate(rows):
            for j, name in enumerate(self.names):
                value = row.get(name)
                values[i, j] = np.nan if value is None else finite(value, name)
        return values

    def fit(self, rows: list[dict]) -> FeaturePreprocessor:
        raw = self._raw(rows)
        self.all_missing = np.isnan(raw).all(axis=0)
        self.medians = np.array(
            [0.0 if self.all_missing[j] else np.nanmedian(raw[:, j]) for j in range(raw.shape[1])]
        )
        imputed = np.where(np.isnan(raw), self.medians, raw)
        self.means = imputed.mean(axis=0)
        self.scales = imputed.std(axis=0)
        self.scales[self.scales < 1e-12] = 1.0
        return self

    def transform(self, rows: list[dict]) -> np.ndarray:
        if self.medians is None:
            raise ValueError("preprocessor is not fitted")
        raw = self._raw(rows)
        missing = np.isnan(raw)
        missing[:, self.all_missing] = True
        imputed = np.where(missing, self.medians, raw)
        # A feature never observed in training cannot acquire influence at test time.
        imputed[:, self.all_missing] = self.means[self.all_missing]
        return np.column_stack(((imputed - self.means) / self.scales, missing.astype(float)))

    def to_dict(self) -> dict:
        if self.medians is None:
            raise ValueError("preprocessor is not fitted")
        return {
            "names": self.names,
            "stage": int(self.stage),
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "all_missing": self.all_missing.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> FeaturePreprocessor:
        obj = cls(value["names"], Stage(value["stage"]))
        n = len(obj.names)
        for name in ("medians", "means", "scales"):
            array = np.asarray(value[name], dtype=float)
            if array.shape != (n,) or not np.isfinite(array).all():
                raise ValueError(f"malformed preprocessor {name}")
            setattr(obj, name, array)
        if (obj.scales <= 0).any():
            raise ValueError("invalid preprocessing scales")
        obj.all_missing = np.asarray(value["all_missing"], dtype=bool)
        if obj.all_missing.shape != (n,):
            raise ValueError("malformed all_missing mask")
        return obj
