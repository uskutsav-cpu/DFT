"""Synthetic smoke demo. Values below are illustrative, not research results."""

from refblind import EscalationPolicy, decompose_error, reference_blind_failure


def main() -> None:
    example = decompose_error(mlip=10.1, dft=10.0, high_level=6.0)
    flag = reference_blind_failure(
        example,
        surrogate_abs_max=0.5,
        reference_abs_min=2.0,
        decision_changed=False,
    )
    policy = EscalationPolicy()
    decision = policy.decide(mlip_uncertainty=0.15, electronic_risk=0.80)

    print("Synthetic example only")
    print(example)
    print("reference_blind_failure:", flag)
    print("escalation_decision:", decision.value)


if __name__ == "__main__":
    main()
