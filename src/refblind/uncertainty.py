"""Ensemble summaries, train-fitted OOD distance and group split conformal bounds."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .validation import matrix, positive, probability, vector


def ensemble_summary(predictions) -> dict:
    """Rows are observations; columns are distinct, teacher-compatible models."""
    values = matrix(predictions, "ensemble predictions")
    if values.shape[1] < 2:
        raise ValueError("at least two ensemble members required")
    return {
        "mean": values.mean(axis=1),
        "std": values.std(axis=1, ddof=1),
        "range": np.ptp(values, axis=1),
    }


class MahalanobisOOD:
    def __init__(self, regularization: float = 0.1):
        self.regularization = positive(regularization)
        self.mean = None
        self.scale = None
        self.precision = None

    def fit(self, training_embeddings) -> MahalanobisOOD:
        X = matrix(training_embeddings)
        if len(X) < 2:
            raise ValueError("at least two training embeddings needed")
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        Z = (X - self.mean) / self.scale
        covariance = Z.T @ Z / (len(Z) - 1)
        self.precision = np.linalg.inv(covariance + self.regularization * np.eye(X.shape[1]))
        return self

    def score(self, embeddings) -> np.ndarray:
        if self.mean is None:
            raise ValueError("OOD model is not fitted")
        X = matrix(embeddings)
        if X.shape[1] != len(self.mean):
            raise ValueError("embedding dimension mismatch")
        Z = (X - self.mean) / self.scale
        return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", Z, self.precision, Z), 0))


class GroupConformal:
    """Calibrate max absolute residual per group, not independent geometry rows.

    Coverage interpretations require exchangeable future groups and the same
    group-score construction. This is not a guarantee under chemistry shift.
    A too-small calibration set returns an explicitly unbounded interval.
    """

    def __init__(self, alpha: float = 0.1):
        probability(alpha)
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly inside (0,1)")
        self.alpha = alpha
        self.quantile = None
        self.n_groups = 0
        self.fitted = False

    def fit(self, residuals, groups: list[str]) -> GroupConformal:
        residuals = vector(residuals)
        if len(groups) != len(residuals):
            raise ValueError("residual/group length mismatch")
        maxima = defaultdict(float)
        for residual, group in zip(residuals, groups):
            maxima[group] = max(maxima[group], abs(float(residual)))
        self.n_groups = len(maxima)
        rank = math.ceil((self.n_groups + 1) * (1 - self.alpha))
        self.quantile = None if rank > self.n_groups else sorted(maxima.values())[rank - 1]
        self.fitted = True
        return self

    def interval(self, predictions) -> dict:
        if not self.fitted:
            raise ValueError("conformal model is not fitted")
        values = vector(predictions)
        if self.quantile is None:
            return {
                "bounded": False,
                "lower": None,
                "upper": None,
                "reason": "insufficient calibration groups for requested coverage",
            }
        return {
            "bounded": True,
            "lower": (values - self.quantile).tolist(),
            "upper": (values + self.quantile).tolist(),
        }
