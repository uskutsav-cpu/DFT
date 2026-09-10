from dataclasses import replace

import pytest

from refblind.policy import Action, CostLedger, SequentialPolicy
from refblind.replay import replay_row, summarize_replay
from refblind.tables import StudyTable


def test_two_stages():
    p = SequentialPolicy()
    assert p.before_dft(mlip_risk=0.1).action == Action.ACCEPT_MLIP
    assert p.before_dft(mlip_risk=0.1, screen_risk=0.9).action == Action.RUN_DFT
    assert p.before_dft(mlip_risk=None).action == Action.RUN_DFT
    assert p.after_dft(reference_risk=None, dft_succeeded=True).action == Action.RUN_HIGH_LEVEL
    assert p.after_dft(reference_risk=None, dft_succeeded=False).action == Action.ABSTAIN
    assert (
        p.after_dft(reference_risk=0.9, dft_succeeded=True, high_level_supported=False).action
        == Action.ABSTAIN
    )


def test_cost_ledger_deduplicates_and_atomic_budget():
    ledger = CostLedger(budget=3.0)
    assert ledger.buy({"a": 2.0}) and ledger.spent == 2
    assert ledger.buy({"a": 2.0}) and ledger.spent == 2
    assert not ledger.buy({"b": 1.0, "c": 1.0}) and ledger.spent == 2
    with pytest.raises(ValueError):
        ledger.buy({"a": 1.0})
    assert ledger.buy({"b": 1.0}) and ledger.spent == 3


def test_dft_not_read_before_paid(row):
    def forbidden():
        raise AssertionError("DFT stage accessed before escalation")

    a = replay_row(row, SequentialPolicy(), mlip_risk=0.0, reference_risk=forbidden, tolerance=1.0)
    assert a.action == "accept_mlip" and a.cost == 1 and not a.correct


def test_reference_features_paid_once(row):
    ledger = CostLedger()

    def later():
        assert ledger.spent == 10  # mlip1+dft2+screen3+diagnostics4
        return 0.9

    a = replay_row(
        row,
        SequentialPolicy(),
        mlip_risk=1.0,
        reference_risk=later,
        tolerance=1.0,
        ledger=ledger,
        use_reference_screen=True,
    )
    assert a.cost == 15 and a.correct
    ledger = CostLedger()
    b = replay_row(
        row,
        SequentialPolicy(),
        mlip_risk=1.0,
        reference_risk=lambda: 0.9,
        tolerance=1.0,
        ledger=ledger,
        use_screen=True,
        use_reference_screen=True,
    )
    assert b.cost == 15  # prescreen not charged twice


def test_missing_highlevel_is_not_oracle(row):
    a = replay_row(
        replace(row, high_level_value=None),
        SequentialPolicy(),
        mlip_risk=1.0,
        reference_risk=lambda: 1.0,
        tolerance=1.0,
    )
    assert a.abstained and a.prediction is None and a.ran_high_level and not a.correct
    assert "label" in a.reason


@pytest.mark.parametrize("budget", [0.0, 0.9, 1.0, 2.9, 3.0, 4.0, 9.0])
def test_budget_never_falls_back_to_mlip(row, budget):
    a = replay_row(
        row,
        SequentialPolicy(),
        mlip_risk=1.0,
        reference_risk=lambda: 1.0,
        tolerance=1.0,
        ledger=CostLedger(budget=budget),
        use_reference_screen=True,
    )
    assert a.abstained and a.cost <= budget


def test_failed_dft_abstains(row):
    a = replay_row(
        replace(row, dft=None),
        SequentialPolicy(),
        mlip_risk=1.0,
        reference_risk=lambda: 0.0,
        tolerance=1.0,
    )
    assert a.abstained and a.cost == 3 and not a.ran_high_level


@pytest.mark.parametrize(
    "mode,cost", [("always_mlip", 1), ("always_dft", 2), ("always_high_level", 5)]
)
def test_baseline_costs(row, mode, cost):
    a = replay_row(
        row, SequentialPolicy(), mlip_risk=0.0, reference_risk=lambda: 0.0, tolerance=1.0, mode=mode
    )
    assert a.cost == cost


def test_shared_job_charges(row):
    ops = {k: {"shared/" + k: v} for k, v in row.costs.items()}
    a = replace(row, operations=ops)
    b = replace(a, id="b")
    ledger = CostLedger()
    results = [
        replay_row(
            r,
            SequentialPolicy(),
            mlip_risk=0.0,
            reference_risk=lambda: 0.0,
            tolerance=1.0,
            ledger=ledger,
        )
        for r in [a, b]
    ]
    assert [r.cost for r in results] == [1.0, 0.0]
    assert summarize_replay(results, [a, b])["total_cost"] == 1


def test_table_stage_and_duplicate_guards(row):
    with pytest.raises(ValueError):
        replace(row, mlip_features={"fod": 1.0})
    with pytest.raises(ValueError):
        replace(row, dft_features={"t1_diagnostic": 1.0})
    with pytest.raises(ValueError):
        StudyTable([row, row], {})
    a = replace(row, decision_group="decision")
    with pytest.raises(ValueError):
        StudyTable([a, replace(a, id="b", group_id="other")], {})
