"""Numerically checked metrics, including explicit tie and undefined semantics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .validation import integer, nonnegative, probability, vector


def binary_arrays(labels, scores, *, probabilities: bool = False):
    y = vector(labels, "labels")
    s = vector(scores, "scores")
    if y.shape != s.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("binary labels and scores must have equal length")
    if probabilities and ((s < 0).any() or (s > 1).any()):
        raise ValueError("probabilities must lie in [0, 1]")
    return y.astype(int), s


def auroc(labels, scores) -> float | None:
    y, s = binary_arrays(labels, scores)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(s, kind="stable")
    sorted_scores = s[order]
    ranks = np.empty(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(labels, scores) -> float | None:
    """Non-interpolated AP; all equal-score observations enter together."""
    y, s = binary_arrays(labels, scores)
    npos = int(y.sum())
    if not npos:
        return None
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    ends = np.r_[np.flatnonzero(np.diff(s)) + 1, len(y)]
    recalls = tp[ends - 1] / npos
    precisions = tp[ends - 1] / ends
    return float(np.sum(np.diff(np.r_[0, recalls]) * precisions))


def brier_score(labels, scores) -> float:
    y, s = binary_arrays(labels, scores, probabilities=True)
    return float(np.mean((s - y) ** 2))


def calibration_bins(labels, scores, *, bins: int = 10) -> dict:
    integer(bins, "bins", 1)
    y, s = binary_arrays(labels, scores, probabilities=True)
    assignments = np.minimum((s * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for i in range(bins):
        mask = assignments == i
        count = int(mask.sum())
        prediction = float(s[mask].mean()) if count else None
        frequency = float(y[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(prediction - frequency)
        rows.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": count,
                "mean_probability": prediction,
                "event_fraction": frequency,
            }
        )
    return {"ece": ece, "bins": rows}


def classification_report(labels, probabilities, *, threshold: float = 0.5) -> dict:
    probability(threshold, "threshold")
    y, p = binary_arrays(labels, probabilities, probabilities=True)
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": auroc(y, p),
        "average_precision": average_precision(y, p),
        "brier": brier_score(y, p),
        "ece": calibration_bins(y, p)["ece"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def regression_report(predicted, reference) -> dict:
    p, r = vector(predicted, "predicted"), vector(reference, "reference")
    if p.shape != r.shape:
        raise ValueError("predicted/reference shape mismatch")
    error = p - r
    return {
        "n": len(p),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "max_abs": float(np.abs(error).max()),
    }


def risk_coverage(losses, uncertainty) -> list[dict]:
    losses, uncertainty = vector(losses, "losses"), vector(uncertainty, "uncertainty")
    if losses.shape != uncertainty.shape or (losses < 0).any():
        raise ValueError("non-negative aligned losses required")
    order = np.argsort(uncertainty, kind="stable")
    losses, uncertainty = losses[order], uncertainty[order]
    ends = np.r_[np.flatnonzero(np.diff(uncertainty)) + 1, len(losses)]
    totals = np.cumsum(losses)
    return [
        {
            "coverage": float(end / len(losses)),
            "risk": float(totals[end - 1] / end),
            "threshold": float(uncertainty[end - 1]),
            "accepted": int(end),
        }
        for end in ends
    ]


def cluster_bootstrap(
    values,
    groups,
    *,
    statistic: Callable = np.mean,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    values = vector(values, "values")
    if len(groups) != len(values):
        raise ValueError("group/value length mismatch")
    integer(repetitions, "repetitions", 2)
    probability(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("bootstrap requires at least two independent clusters")
    indices = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        draw = np.concatenate([indices[g] for g in chosen])
        result = float(statistic(values[draw]))
        if not np.isfinite(result):
            raise ValueError("bootstrap statistic returned a non-finite value")
        sampled.append(result)
    tail = (1 - confidence) / 2
    low, high = np.quantile(sampled, [tail, 1 - tail])
    return {
        "estimate": float(statistic(values)),
        "lower": float(low),
        "upper": float(high),
        "confidence": confidence,
        "groups": len(unique),
        "repetitions": repetitions,
        "estimand": "row-weighted statistic; clusters resampled with replacement",
    }


def pareto_frontier(costs, risks) -> list[int]:
    costs, risks = vector(costs, "costs"), vector(risks, "risks")
    if costs.shape != risks.shape or (costs < 0).any() or (risks < 0).any():
        raise ValueError("aligned nonnegative costs and risks required")
    kept = []
    for i in range(len(costs)):
        dominates = (
            (costs <= costs[i]) & (risks <= risks[i]) & ((costs < costs[i]) | (risks < risks[i]))
        )
        if not dominates.any():
            kept.append(i)
    return kept


def robust_ordering(
    candidate_gap: float,
    reference_gap: float,
    *,
    reference_uncertainty: float = 0.0,
    tie_tolerance: float = 0.0,
) -> bool | None:
    """Compare a *paired* decision; None when the reference is unresolved."""
    from .validation import finite

    c = finite(candidate_gap)
    r = finite(reference_gap)
    margin = nonnegative(reference_uncertainty) + nonnegative(tie_tolerance)
    if abs(r) <= margin:
        return None
    return (c > tie_tolerance and r > margin) or (c < -tie_tolerance and r < -margin)
