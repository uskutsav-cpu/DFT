import pytest

from refblind.escalation import EscalationDecision, EscalationPolicy


def test_accept_low_risk_case():
    policy = EscalationPolicy()
    assert (
        policy.decide(mlip_uncertainty=0.1, electronic_risk=0.1) == EscalationDecision.ACCEPT_MLIP
    )


def test_mlip_uncertainty_can_trigger_dft():
    policy = EscalationPolicy(cost_penalty=0.0)
    assert policy.decide(mlip_uncertainty=0.8, electronic_risk=0.1) == EscalationDecision.RUN_DFT


def test_reference_risk_takes_priority():
    policy = EscalationPolicy(cost_penalty=0.0)
    assert (
        policy.decide(mlip_uncertainty=0.9, electronic_risk=0.9)
        == EscalationDecision.RUN_HIGH_LEVEL
    )


def test_invalid_risk_rejected():
    policy = EscalationPolicy()
    with pytest.raises(ValueError):
        policy.decide(mlip_uncertainty=1.1, electronic_risk=0.2)
