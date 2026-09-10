from __future__ import annotations

from collections.abc import Mapping


def weighted_risk(features: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Return a transparent weighted mean of normalized diagnostic risk scores.

    This is deliberately simple and suitable as a baseline. Feature construction
    should occur without access to higher-level evaluation labels at inference time.
    """

    if set(features) != set(weights):
        raise ValueError("features and weights must have identical keys")
    if not features:
        raise ValueError("at least one feature is required")

    numerator = 0.0
    denominator = 0.0
    for key, value in features.items():
        value = float(value)
        weight = float(weights[key])
        if not 0 <= value <= 1:
            raise ValueError(f"feature {key!r} must lie in [0, 1]")
        if weight < 0:
            raise ValueError("weights must be non-negative")
        numerator += value * weight
        denominator += weight

    if denominator == 0:
        raise ValueError("at least one weight must be positive")
    return numerator / denominator
