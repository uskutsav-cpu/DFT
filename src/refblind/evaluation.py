from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np

from .validation import nonnegative, vector


def _relation(a: float, b: float, tie_tol: float) -> int:
    delta = float(a) - float(b)
    if abs(delta) <= tie_tol:
        return 0
    return 1 if delta > 0 else -1


def ordering_accuracy(
    candidate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    *,
    tie_tol: float = 1e-8,
) -> float:
    """Fraction of pairwise order relations preserved by ``candidate``.

    Ties count as correct only when both candidate and reference are tied within
    ``tie_tol``. A one-element sequence has accuracy 1 by convention.
    """

    nonnegative(tie_tol, "tie_tol")

    cand = vector(candidate, "candidate")
    ref = vector(reference, "reference")
    if cand.size != ref.size:
        raise ValueError("candidate and reference must have equal length")
    if cand.size == 0:
        raise ValueError("at least one value is required")
    if not np.isfinite(cand).all() or not np.isfinite(ref).all():
        raise ValueError("ordering inputs must be finite")

    pairs = list(combinations(range(cand.size), 2))
    if not pairs:
        return 1.0

    preserved = sum(
        _relation(cand[i], cand[j], tie_tol) == _relation(ref[i], ref[j], tie_tol) for i, j in pairs
    )
    return preserved / len(pairs)


def ordering_preserved(
    candidate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    *,
    tie_tol: float = 1e-8,
) -> bool:
    """Return True only when all pairwise order relations are preserved."""

    return ordering_accuracy(candidate, reference, tie_tol=tie_tol) == 1.0


def mean_cost(decision_costs: Iterable[float]) -> float:
    """Mean non-negative relative compute cost for a set of decisions."""

    values = np.asarray(list(decision_costs), dtype=float)
    if values.size == 0:
        raise ValueError("at least one cost is required")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("costs must be finite and non-negative")
    return float(values.mean())
