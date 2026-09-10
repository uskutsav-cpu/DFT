import pytest

from refblind.evaluation import ordering_accuracy, ordering_preserved


def test_perfect_ordering():
    assert ordering_preserved([1, 2, 3], [10, 20, 30])
    assert ordering_accuracy([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_reversed_ordering_fails():
    assert not ordering_preserved([3, 2, 1], [1, 2, 3])
    assert ordering_accuracy([3, 2, 1], [1, 2, 3]) == pytest.approx(0.0)


def test_tie_semantics():
    assert ordering_preserved([1.0, 1.0 + 1e-9], [2.0, 2.0], tie_tol=1e-8)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        ordering_accuracy([1, 2], [1])
