import pytest

from refblind.errors import decompose_error, reference_blind_failure


def test_error_decomposition_identity():
    result = decompose_error(10.1, 10.0, 6.0)
    assert result.surrogate_error == pytest.approx(0.1)
    assert result.reference_error == pytest.approx(4.0)
    assert result.total_error == pytest.approx(4.1)
    assert result.total_error == pytest.approx(result.surrogate_error + result.reference_error)


def test_reference_blind_by_magnitude():
    result = decompose_error(10.1, 10.0, 6.0)
    assert reference_blind_failure(result, surrogate_abs_max=0.5, reference_abs_min=2.0)


def test_reference_blind_by_decision_change():
    result = decompose_error(1.01, 1.0, 0.95)
    assert reference_blind_failure(
        result,
        surrogate_abs_max=0.1,
        reference_abs_min=1.0,
        decision_changed=True,
    )


def test_large_surrogate_error_is_not_reference_blind():
    result = decompose_error(5.0, 1.0, 0.0)
    assert not reference_blind_failure(
        result, surrogate_abs_max=0.5, reference_abs_min=0.5, decision_changed=True
    )


def test_negative_threshold_rejected():
    result = decompose_error(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        reference_blind_failure(result, surrogate_abs_max=-1, reference_abs_min=1)
