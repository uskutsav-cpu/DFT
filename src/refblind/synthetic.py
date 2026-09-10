"""Deterministic software-only data. NEVER mix this with empirical benchmarks."""

from __future__ import annotations

import numpy as np

from .tables import StudyRow, StudyTable
from .validation import integer


def synthetic_table(*, groups: int = 60, rows_per_group: int = 4, seed: int = 42) -> StudyTable:
    integer(groups, "groups", 4)
    integer(rows_per_group, "rows_per_group", 1)
    integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(groups):
        hidden_reference_difficulty = rng.uniform()
        hidden_surrogate_difficulty = rng.uniform()
        base_barrier = rng.uniform(5, 30)
        for item in range(rows_per_group):
            difficulty = np.clip(hidden_surrogate_difficulty + rng.normal(0, 0.1), 0, 1)
            reference_difficulty = np.clip(hidden_reference_difficulty + rng.normal(0, 0.1), 0, 1)
            reference = base_barrier + rng.normal(0, 2)
            reference_error = (
                rng.choice([-1, 1]) * (2 + 6 * reference_difficulty)
                if reference_difficulty > 0.6
                else rng.normal(0, 0.35)
            )
            surrogate_error = rng.normal(0, 0.08 + 1.8 * difficulty**3)
            dft = reference + reference_error
            mlip = dft + surrogate_error
            uq = max(0.01, 0.08 + 1.8 * difficulty**3 + rng.normal(0, 0.03))
            screen = float(np.clip(reference_difficulty + rng.normal(0, 0.2), 0, 1))
            features1 = {
                "ensemble_std": float(uq),
                "ood_distance": float(difficulty * 3),
                "cheap_screen": screen,
            }
            features2 = {
                "homo_lumo_gap_ev": float(
                    max(0.01, 5 * (1 - reference_difficulty) + rng.normal(0, 0.4))
                ),
                "spin_contamination": float(max(0, reference_difficulty**2 + rng.normal(0, 0.05))),
                "scf_cycles": float(round(10 + 45 * reference_difficulty)),
                "fod": float(max(0, 2 * reference_difficulty + rng.normal(0, 0.1))),
            }
            row_id = f"SYNTHETIC:{group:03d}:{item:02d}"
            rows.append(
                StudyRow(
                    row_id,
                    f"SYNTHETIC-GROUP:{group:03d}",
                    float(mlip),
                    float(dft),
                    float(reference),
                    features1,
                    features2,
                    {
                        "mlip": 0.03,
                        "dft": 1.0,
                        "high_level": 20.0,
                        "screen": 0.05,
                        "diagnostics": 0.3,
                    },
                    reference_quality="synthetic",
                    reference_uncertainty=0.0,
                    high_level_value=float(reference + rng.normal(0, 0.05)),
                    decision_group=f"SYNTHETIC-DECISION:{group:03d}",
                    metadata={
                        "synthetic": True,
                        "generator": "designed statistical software fixture",
                    },
                )
            )
    return StudyTable(
        rows,
        {
            "synthetic": True,
            "generator_seed": seed,
            "not_for_publication": "artificial signal is designed into these data",
            "cost_semantics": "arbitrary synthetic relative costs, not timings or prices",
        },
    )
