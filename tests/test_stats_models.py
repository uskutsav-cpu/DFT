from dataclasses import replace

import numpy as np
import pytest

from refblind.features import (
    FeaturePreprocessor,
    Stage,
    orbital_gap_ev,
    spin_contamination,
    validate_feature_names,
)
from refblind.learning import LogisticRegression, RiskModel, sigmoid
from refblind.metrics import (
    auroc,
    average_precision,
    brier_score,
    calibration_bins,
    cluster_bootstrap,
    pareto_frontier,
    risk_coverage,
    robust_ordering,
)
from refblind.splits import connected_groups, grouped_split
from refblind.tables import target_label
from refblind.uncertainty import GroupConformal, MahalanobisOOD, ensemble_summary


@pytest.mark.parametrize(
    "name",
    [
        "fod",
        "homo_lumo_gap_ev",
        "scf_cycles",
        "t1_diagnostic",
        "reference_error",
        "high_level_energy",
        "unknown",
    ],
)
def test_stage_one_leakage_rejected(name):
    with pytest.raises(ValueError):
        validate_feature_names([name], Stage.MLIP)


@pytest.mark.parametrize("name", ["t1_diagnostic", "reference_error", "high_level_energy"])
def test_label_leakage_rejected_everywhere(name):
    for stage in Stage:
        with pytest.raises(ValueError):
            validate_feature_names([name], stage)


def test_preprocessor_train_only_missing():
    p = FeaturePreprocessor(["ensemble_std", "cheap_screen"], Stage.MLIP)
    p.fit([{"ensemble_std": 1.0}, {"ensemble_std": 3.0}, {"ensemble_std": None}])
    before = p.to_dict()
    x = p.transform([{"ensemble_std": 1000.0, "cheap_screen": 1000.0}])
    assert p.to_dict() == before
    assert x[0, 1] == 0
    assert p.medians[0] == 2
    assert np.array_equal(FeaturePreprocessor.from_dict(before).transform([{}]), p.transform([{}]))
    with pytest.raises(ValueError):
        p.transform([{"ensemble_std": float("inf")}])
    with pytest.raises(ValueError):
        FeaturePreprocessor(["ensemble_std"], Stage.MLIP).transform([{}])


def test_diagnostics_math():
    assert spin_contamination(0.8, 2) == pytest.approx(0.05)
    assert orbital_gap_ev([-10.0, -3.0, 2.0, 4.0], [2.0, 2.0, 0.0, 0.0]) == 5.0
    with pytest.raises(ValueError):
        orbital_gap_ev([1.0], [1.0])
    with pytest.raises(ValueError):
        spin_contamination(-1, 2)


@pytest.mark.parametrize("seed", range(12))
def test_metrics_against_sklearn(seed):
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    y = np.r_[0, 1, rng.integers(0, 2, 98)]
    p = np.round(rng.random(100), 1)
    assert auroc(y, p) == pytest.approx(sk.roc_auc_score(y, p))
    assert average_precision(y, p) == pytest.approx(sk.average_precision_score(y, p))
    assert brier_score(y, p) == pytest.approx(sk.brier_score_loss(y, p))
    order = rng.permutation(100)
    assert auroc(y[order], p[order]) == pytest.approx(auroc(y, p))
    assert average_precision(y[order], p[order]) == pytest.approx(average_precision(y, p))


def test_ties_undefined_and_bins():
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert average_precision([0, 1], [0.5, 0.5]) == 0.5
    assert auroc([1, 1], [0.2, 0.8]) is None
    assert average_precision([0, 0], [0.2, 0.8]) is None
    bins = calibration_bins([0, 1], [0.0, 1.0])
    assert bins["ece"] == 0 and sum(b["count"] for b in bins["bins"]) == 2
    assert risk_coverage([0.0, 1.0], [0.5, 0.5]) == [
        {"coverage": 1.0, "risk": 0.5, "threshold": 0.5, "accepted": 2}
    ]
    assert pareto_frontier([1, 2, 3], [3, 2, 3]) == [0, 1]
    assert robust_ordering(1.0, 0.1, reference_uncertainty=0.2) is None
    assert robust_ordering(1.0, -2.0) is False
    assert robust_ordering(-1.0, -2.0) is True


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), 2])
def test_probability_metrics_reject(bad):
    with pytest.raises(ValueError):
        brier_score([1], [bad])


def test_group_bootstrap_deterministic():
    a = cluster_bootstrap([0, 0, 1, 1], ["a", "a", "b", "b"], repetitions=50)
    assert a == cluster_bootstrap([0, 0, 1, 1], ["a", "a", "b", "b"], repetitions=50)
    assert a["estimate"] == 0.5 and a["lower"] <= 0.5 <= a["upper"]
    with pytest.raises(ValueError):
        cluster_bootstrap([1, 2], ["a", "a"])


def test_group_split_no_leak():
    ids = [f"r{i}" for i in range(40)]
    groups = [f"g{i // 2}" for i in range(40)]
    plan = grouped_split(ids, groups)
    rev = grouped_split(ids[::-1], groups[::-1])
    assert plan.assignments == rev.assignments
    plan.validate(ids, groups)
    for i in range(0, 40, 2):
        assert plan.assignments[ids[i]] == plan.assignments[ids[i + 1]]
    bad = dict(plan.assignments)
    bad[ids[0]] = "test" if bad[ids[1]] != "test" else "train"
    with pytest.raises(ValueError):
        replace(plan, assignments=bad).validate(ids, groups)
    with pytest.raises(ValueError):
        grouped_split(["a", "b"], ["a", "b"])


def test_connected_groups():
    groups = connected_groups(
        {"a": ["ts1"], "b": ["ts1", "scaffold"], "c": ["scaffold"], "d": ["other"]}
    )
    assert groups["a"] == groups["b"] == groups["c"] and groups["d"] != groups["a"]


def test_logistic_fit_roundtrip():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    p = model.predict_proba(X)
    assert ((p >= 0.5) == y).mean() > 0.95
    assert model.converged
    assert np.allclose(LogisticRegression.from_dict(model.to_dict()).predict_proba(X), p)
    assert np.isfinite(sigmoid(np.array([-1e300, 1e300]))).all()
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((2, 2)))


def test_single_class_model_and_refit():
    model = LogisticRegression().fit([[0.0], [1.0]], [1, 1])
    assert np.allclose(model.predict_proba([[999.0], [-999.0]]), 2.5 / 3)
    model.fit([[0.0], [1.0]], [0, 1])
    assert model.constant_probability is None


def test_risk_model_calibration_disjoint():
    rows = [{"ensemble_std": float(i)} for i in range(12)]
    labels = [0] * 6 + [1] * 6
    m = RiskModel(["ensemble_std"], Stage.MLIP).fit(rows, labels, [f"t{i}" for i in range(12)])
    with pytest.raises(ValueError):
        m.calibrate(rows, labels, [f"t{i}" for i in range(12)])
    m.calibrate(rows, labels, [f"c{i}" for i in range(12)])
    n = RiskModel.from_dict(m.to_dict())
    assert np.allclose(n.predict(rows), m.predict(rows))
    tampered = m.to_dict()
    tampered["calibration_ids"] = ["t1"]
    with pytest.raises(ValueError):
        RiskModel.from_dict(tampered)


def test_conformal_groups_and_unbounded():
    c = GroupConformal(alpha=0.5).fit([1.0, 10.0, 2.0, 3.0], ["a", "a", "b", "c"])
    assert c.n_groups == 3 and c.quantile == 3
    assert c.interval([5.0])["lower"] == [2.0]
    assert GroupConformal(0.1).fit([1.0], ["a"]).interval([0.0])["bounded"] is False


def test_ood_training_fit_and_committee():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    m = MahalanobisOOD().fit(X)
    assert m.score([[100.0, 100.0]])[0] > m.score([[1.0, 1.0]])[0]
    assert np.allclose(ensemble_summary([[1, 3], [4, 4]])["std"], [np.sqrt(2), 0])
    with pytest.raises(ValueError):
        ensemble_summary([[1], [2]])


@pytest.mark.parametrize(
    "args,expected",
    [
        ((0.0, 0.0, 1.0, None), 0),
        ((4.0, 0.0, 3.0, 0.5), 1),
        ((3.0, 0.0, 3.0, 0.5), None),
        ((2.0, 0.0, 3.0, 0.5), 0),
    ],
)
def test_uncertain_reference_labels(args, expected):
    assert target_label(*args) == expected
