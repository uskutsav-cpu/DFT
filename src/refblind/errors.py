from __future__ import annotations

from dataclasses import dataclass

from .validation import finite, nonnegative


@dataclass(frozen=True)
class ErrorDecomposition:
    """Signed error decomposition for one scalar observable."""

    surrogate_error: float
    reference_error: float
    total_error: float


def decompose_error(mlip: float, dft: float, high_level: float) -> ErrorDecomposition:
    """Return signed MLIP→DFT, DFT→HL, and MLIP→HL errors.

    The identity ``total_error = surrogate_error + reference_error`` is exact
    up to floating-point arithmetic.
    """

    mlip, dft, high_level = finite(mlip), finite(dft), finite(high_level)
    surrogate = mlip - dft
    reference = float(dft) - float(high_level)
    total = float(mlip) - float(high_level)
    return ErrorDecomposition(surrogate, reference, total)


def reference_blind_failure(
    decomposition: ErrorDecomposition,
    *,
    surrogate_abs_max: float,
    reference_abs_min: float,
    decision_changed: bool = False,
) -> bool:
    """Flag a reference-blind failure under an explicit operational rule.

    A case is reference-blind when the surrogate looks close to its DFT teacher
    while the DFT reference is chemically consequential. Consequence may be
    established either by a reference-error magnitude threshold or, preferably,
    by an externally computed chemistry-level decision change.

    Thresholds must be non-negative and should be frozen before final testing.
    """

    nonnegative(surrogate_abs_max, "surrogate_abs_max")
    nonnegative(reference_abs_min, "reference_abs_min")
    for value in (
        decomposition.surrogate_error,
        decomposition.reference_error,
        decomposition.total_error,
    ):
        finite(value, "error")
    if not isinstance(decision_changed, bool):
        raise ValueError("decision_changed must be boolean")

    low_surrogate_error = abs(decomposition.surrogate_error) <= surrogate_abs_max
    consequential_reference_error = abs(decomposition.reference_error) >= reference_abs_min or bool(
        decision_changed
    )
    return low_surrogate_error and consequential_reference_error
