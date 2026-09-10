"""Small-data baselines using NumPy; explicit serialization, no pickle loading."""

from __future__ import annotations

import numpy as np

from .features import FeaturePreprocessor, Stage
from .metrics import binary_arrays
from .validation import integer, matrix, positive


def sigmoid(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive_mask = x >= 0
    out[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
    ex = np.exp(x[~positive_mask])
    out[~positive_mask] = ex / (1 + ex)
    return out


class LogisticRegression:
    """L2 regularized Newton/IRLS solver with line search and stable objectives."""

    def __init__(self, l2: float = 1.0, max_iter: int = 100, tolerance: float = 1e-8):
        self.l2 = positive(l2, "l2")
        self.max_iter = integer(max_iter, "max_iter", 1)
        self.tolerance = positive(tolerance, "tolerance")
        self.coef = None
        self.n_features = None
        self.converged = False
        self.iterations = 0
        self.constant_probability = None

    def fit(self, X, y) -> LogisticRegression:
        X = matrix(X, "X")
        y, _ = binary_arrays(y, np.zeros(X.shape[0]))
        self.n_features = X.shape[1]
        self.constant_probability = None
        self.converged = False
        self.iterations = 0
        if len(np.unique(y)) == 1:
            # Explicitly report a single-class training set; no manufactured discrimination.
            self.constant_probability = float((y.sum() + 0.5) / (len(y) + 1))
            self.coef = np.zeros(self.n_features + 1)
            self.converged = True
            return self
        A = np.column_stack((np.ones(len(X)), X))
        beta = np.zeros(A.shape[1])
        prevalence = y.mean()
        beta[0] = np.log(prevalence / (1 - prevalence))
        penalty = np.eye(A.shape[1]) * self.l2
        penalty[0, 0] = 0.0

        def objective(b):
            z = A @ b
            return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * b @ penalty @ b

        for iteration in range(self.max_iter):
            p = sigmoid(A @ beta)
            gradient = A.T @ (p - y) + penalty @ beta
            hessian = (A.T * (p * (1 - p))) @ A + penalty
            step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-10, gradient)
            current = objective(beta)
            rate = 1.0
            for _ in range(30):
                updated = beta - rate * step
                if objective(updated) <= current:
                    break
                rate *= 0.5
            else:
                break
            delta = np.max(np.abs(updated - beta))
            beta = updated
            self.iterations = iteration + 1
            if delta < self.tolerance:
                self.converged = True
                break
        self.coef = beta
        if not np.isfinite(beta).all():
            raise ValueError("logistic optimization produced non-finite coefficients")
        return self

    def decision_function(self, X) -> np.ndarray:
        if self.coef is None:
            raise ValueError("model is not fitted")
        X = matrix(X, "X")
        if X.shape[1] != self.n_features:
            raise ValueError("feature count mismatch")
        if self.constant_probability is not None:
            p = self.constant_probability
            return np.full(len(X), np.log(p / (1 - p)))
        return self.coef[0] + X @ self.coef[1:]

    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def to_dict(self) -> dict:
        if self.coef is None:
            raise ValueError("model is not fitted")
        return {
            "l2": self.l2,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "coef": self.coef.tolist(),
            "n_features": self.n_features,
            "converged": self.converged,
            "iterations": self.iterations,
            "constant_probability": self.constant_probability,
        }

    @classmethod
    def from_dict(cls, value: dict) -> LogisticRegression:
        obj = cls(value["l2"], value["max_iter"], value["tolerance"])
        obj.n_features = integer(value["n_features"], "n_features", 1)
        obj.coef = np.asarray(value["coef"], dtype=float)
        if obj.coef.shape != (obj.n_features + 1,) or not np.isfinite(obj.coef).all():
            raise ValueError("malformed logistic coefficients")
        obj.converged = bool(value["converged"])
        obj.iterations = integer(value["iterations"], "iterations", 0)
        obj.constant_probability = value.get("constant_probability")
        if obj.constant_probability is not None:
            from .validation import probability

            probability(obj.constant_probability)
            if not 0 < obj.constant_probability < 1:
                raise ValueError("constant probability must be strictly inside (0,1)")
        return obj


class RiskModel:
    """Training-only preprocessing/model fit, followed by independent Platt fit."""

    def __init__(self, names: list[str], stage: Stage, *, l2: float = 1.0):
        self.preprocessor = FeaturePreprocessor(names, stage)
        self.model = LogisticRegression(l2=l2)
        self.calibrator = None
        self.training_ids: set[str] = set()
        self.calibration_ids: set[str] = set()

    def fit(self, rows: list[dict], labels, ids: list[str]) -> RiskModel:
        if len(rows) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("training IDs must be aligned and unique")
        self.training_ids = set(ids)
        self.calibration_ids = set()
        self.calibrator = None
        self.preprocessor.fit(rows)
        self.model.fit(self.preprocessor.transform(rows), labels)
        return self

    def calibrate(self, rows: list[dict], labels, ids: list[str]) -> RiskModel:
        if len(rows) != len(ids) or len(set(ids)) != len(ids) or set(ids) & self.training_ids:
            raise ValueError("calibration must use disjoint, unique IDs")
        logits = self.model.decision_function(self.preprocessor.transform(rows)).reshape(-1, 1)
        self.calibrator = LogisticRegression(l2=1.0).fit(logits, labels)
        self.calibration_ids = set(ids)
        return self

    def predict(self, rows: list[dict]) -> np.ndarray:
        X = self.preprocessor.transform(rows)
        if self.calibrator is None:
            return self.model.predict_proba(X)
        return self.calibrator.predict_proba(self.model.decision_function(X).reshape(-1, 1))

    def to_dict(self) -> dict:
        return {
            "preprocessor": self.preprocessor.to_dict(),
            "model": self.model.to_dict(),
            "calibrator": self.calibrator.to_dict() if self.calibrator is not None else None,
            "training_ids": sorted(self.training_ids),
            "calibration_ids": sorted(self.calibration_ids),
        }

    @classmethod
    def from_dict(cls, value: dict) -> RiskModel:
        prep = FeaturePreprocessor.from_dict(value["preprocessor"])
        obj = cls(prep.names, prep.stage)
        obj.preprocessor = prep
        obj.model = LogisticRegression.from_dict(value["model"])
        if obj.model.n_features != 2 * len(prep.names):
            raise ValueError("serialized feature and model dimensions mismatch")
        if value["calibrator"] is not None:
            obj.calibrator = LogisticRegression.from_dict(value["calibrator"])
            if obj.calibrator.n_features != 1:
                raise ValueError("calibrator must take one logit")
        obj.training_ids = set(value["training_ids"])
        obj.calibration_ids = set(value["calibration_ids"])
        if obj.training_ids & obj.calibration_ids:
            raise ValueError("serialized train/calibration overlap")
        return obj
