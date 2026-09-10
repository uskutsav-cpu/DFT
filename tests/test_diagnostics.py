import pytest

from refblind.diagnostics import weighted_risk


def test_weighted_risk():
    value = weighted_risk({"a": 0.2, "b": 0.8}, {"a": 1.0, "b": 3.0})
    assert value == pytest.approx(0.65)


def test_weighted_risk_rejects_mismatched_keys():
    with pytest.raises(ValueError):
        weighted_risk({"a": 0.2}, {"b": 1.0})
