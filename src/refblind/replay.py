"""Offline policy replay using observed candidate outputs, not oracle HL labels.

Reference values are consulted only for scoring. RUN_HIGH_LEVEL with no observed
high-level output abstains; it is not automatically assigned the target label.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from .metrics import robust_ordering
from .policy import Action, CostLedger, SequentialPolicy
from .tables import StudyRow
from .validation import nonnegative


@dataclass(frozen=True)
class ReplayResult:
    row_id: str
    group_id: str
    action: str
    prediction: float | None
    cost: float
    ran_dft: bool
    ran_high_level: bool
    reason: str
    absolute_error: float | None
    correct: bool
    abstained: bool

    def to_dict(self):
        return asdict(self)


def replay_row(
    row: StudyRow,
    policy: SequentialPolicy,
    *,
    mlip_risk: float | None,
    reference_risk: Callable[[], float | None],
    tolerance: float,
    ledger: CostLedger | None = None,
    use_screen: bool = False,
    use_diagnostics: bool = True,
    use_reference_screen: bool = False,
    mode: str = "sequential",
) -> ReplayResult:
    nonnegative(tolerance, "tolerance")
    ledger = ledger or CostLedger()
    start = ledger.spent
    ran_dft = ran_hl = False

    def finish(action, prediction, reason):
        error = (
            None if prediction is None or row.reference is None else abs(prediction - row.reference)
        )
        uncertainty = row.reference_uncertainty or 0.0
        correct = error is not None and error + uncertainty <= tolerance
        return ReplayResult(
            row.id,
            row.group_id,
            action.value,
            prediction,
            ledger.spent - start,
            ran_dft,
            ran_hl,
            reason,
            error,
            bool(correct),
            prediction is None,
        )

    if mode not in {"sequential", "always_mlip", "always_dft", "always_high_level", "uq_only"}:
        raise ValueError("unknown replay mode")
    if mode != "always_high_level":
        # always_dft does not pay for MLIP calculations it does not use.
        if mode != "always_dft":
            if not ledger.buy(row.stage_operations("mlip")):
                return finish(Action.ABSTAIN, None, "MLIP budget exhausted")
            if row.mlip is None:
                return finish(Action.ABSTAIN, None, "MLIP failed")
        if mode == "always_mlip":
            return finish(Action.ACCEPT_MLIP, row.mlip, "always-MLIP baseline")
        if use_screen:
            if not ledger.buy(row.stage_operations("screen")):
                return finish(Action.ABSTAIN, None, "prescreen budget exhausted")
        decision = policy.before_dft(
            mlip_risk=mlip_risk,
            screen_risk=row.mlip_features.get("cheap_screen") if use_screen else None,
        )
        if mode == "always_dft":
            decision_action = Action.RUN_DFT
        else:
            decision_action = decision.action
        if decision_action == Action.ACCEPT_MLIP:
            return finish(Action.ACCEPT_MLIP, row.mlip, decision.reason)
        if not ledger.buy(row.stage_operations("dft")):
            return finish(Action.ABSTAIN, None, "DFT budget exhausted; no silent MLIP fallback")
        ran_dft = True
        if row.dft is None:
            return finish(Action.ABSTAIN, None, "DFT failed")
        if mode in {"always_dft", "uq_only"}:
            return finish(Action.ACCEPT_DFT, row.dft, "baseline accepts DFT")
        if use_diagnostics and not ledger.buy(row.stage_operations("diagnostics")):
            return finish(Action.ABSTAIN, None, "diagnostic budget exhausted")
        if use_reference_screen and not ledger.buy(row.stage_operations("screen")):
            return finish(Action.ABSTAIN, None, "reference-stage screen budget exhausted")
        # Critical: callback is evaluated only after all its feature costs are paid.
        decision = policy.after_dft(
            reference_risk=reference_risk(),
            dft_succeeded=True,
            high_level_supported=row.high_level_supported,
        )
        if decision.action == Action.ACCEPT_DFT:
            return finish(Action.ACCEPT_DFT, row.dft, decision.reason)
        if decision.action == Action.ABSTAIN:
            return finish(Action.ABSTAIN, None, decision.reason)
    if not row.high_level_supported:
        return finish(Action.ABSTAIN, None, "appropriate high-level method unavailable")
    if not ledger.buy(row.stage_operations("high_level")):
        return finish(Action.ABSTAIN, None, "high-level budget exhausted")
    ran_hl = True
    if row.high_level_value is None:
        return finish(
            Action.ABSTAIN, None, "high-level action has no observed output; not replaced by label"
        )
    return finish(Action.ACCEPT_HIGH_LEVEL, row.high_level_value, "observed high-level output")


def summarize_replay(
    results: list[ReplayResult],
    rows: list[StudyRow],
    *,
    cost_weight: float = 0.05,
    cost_scale: float = 20.0,
) -> dict:
    from .validation import positive

    nonnegative(cost_weight, "cost_weight")
    positive(cost_scale, "cost_scale")
    if (
        not results
        or len(results) != len(rows)
        or [r.row_id for r in results] != [r.id for r in rows]
    ):
        raise ValueError("replay results must align with input rows")
    coverage = np.mean([not r.abstained for r in results])
    errors = [r.absolute_error for r in results if r.absolute_error is not None]
    costs = np.array([r.cost for r in results])
    correct = np.array([r.correct for r in results])
    loss = 1 - float(correct.mean())  # abstentions count as unresolved, not correct.
    groups = {}
    for row, result in zip(rows, results):
        if row.decision_group:
            groups.setdefault(row.decision_group, []).append((row, result))
    pair_correct, pair_unresolved = [], 0
    from itertools import combinations

    for members in groups.values():
        for (a, ra), (b, rb) in combinations(members, 2):
            if (
                ra.prediction is None
                or rb.prediction is None
                or a.reference is None
                or b.reference is None
            ):
                pair_unresolved += 1
                continue
            relation = robust_ordering(
                ra.prediction - rb.prediction,
                a.reference - b.reference,
                reference_uncertainty=(a.reference_uncertainty or 0)
                + (b.reference_uncertainty or 0),
            )
            if relation is None:
                pair_unresolved += 1
            else:
                pair_correct.append(relation)
    return {
        "n": len(results),
        "coverage": float(coverage),
        "accuracy_all_cases": float(correct.mean()),
        "accuracy_accepted": float(correct.sum() / (coverage * len(results))) if coverage else None,
        "mae_accepted": float(np.mean(errors)) if errors else None,
        "total_cost": float(costs.sum()),
        "mean_cost": float(costs.mean()),
        "dft_call_fraction": float(np.mean([r.ran_dft for r in results])),
        "high_level_call_fraction": float(np.mean([r.ran_high_level for r in results])),
        "objective": loss + cost_weight * float(costs.mean()) / cost_scale,
        "pairwise_decisions_resolved": len(pair_correct),
        "pairwise_decisions_unresolved": pair_unresolved,
        "pairwise_ordering_accuracy": float(np.mean(pair_correct)) if pair_correct else None,
        "abstention_semantics": "unresolved cases count as failure in accuracy_all_cases/objective",
    }
