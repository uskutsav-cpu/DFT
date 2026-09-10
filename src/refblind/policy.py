"""Sequential fidelity decisions: no DFT features before purchasing DFT.

Missing evidence and exhausted compute budgets never silently imply low risk.
This is a research policy, not a physical correctness certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .validation import nonnegative, probability


class Action(str, Enum):
    ACCEPT_MLIP = "accept_mlip"
    RUN_DFT = "run_dft"
    ACCEPT_DFT = "accept_dft"
    RUN_HIGH_LEVEL = "run_high_level"
    ACCEPT_HIGH_LEVEL = "accept_high_level"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


@dataclass(frozen=True)
class SequentialPolicy:
    mlip_risk_threshold: float = 0.2
    reference_risk_threshold: float = 0.2
    screen_risk_threshold: float = 0.5
    screen_all: bool = False

    def __post_init__(self):
        probability(self.mlip_risk_threshold, "mlip_risk_threshold")
        probability(self.reference_risk_threshold, "reference_risk_threshold")
        probability(self.screen_risk_threshold, "screen_risk_threshold")
        if not isinstance(self.screen_all, bool):
            raise ValueError("screen_all must be boolean")

    def before_dft(self, *, mlip_risk: float | None, screen_risk: float | None = None) -> Decision:
        if mlip_risk is not None:
            probability(mlip_risk, "mlip_risk")
        if screen_risk is not None:
            probability(screen_risk, "screen_risk")
        if self.screen_all:
            return Decision(Action.RUN_DFT, "screen-all baseline: purchase reference diagnostics")
        if mlip_risk is None:
            return Decision(Action.RUN_DFT, "missing MLIP-stage risk; do not assume reliable")
        if screen_risk is not None and screen_risk >= self.screen_risk_threshold:
            return Decision(Action.RUN_DFT, "independent low-fidelity reference prescreen fired")
        if mlip_risk >= self.mlip_risk_threshold:
            return Decision(Action.RUN_DFT, "MLIP-stage risk exceeds frozen threshold")
        return Decision(Action.ACCEPT_MLIP, "below configured MLIP-stage risk threshold")

    def after_dft(
        self,
        *,
        reference_risk: float | None,
        dft_succeeded: bool,
        high_level_supported: bool = True,
    ) -> Decision:
        if reference_risk is not None:
            probability(reference_risk, "reference_risk")
        if not dft_succeeded:
            return Decision(
                Action.ABSTAIN, "DFT failed: repair the electronic state/convergence first"
            )
        if reference_risk is None or reference_risk >= self.reference_risk_threshold:
            if not high_level_supported:
                return Decision(Action.ABSTAIN, "no appropriate supported high-level method")
            return Decision(Action.RUN_HIGH_LEVEL, "unknown or elevated reference-method risk")
        return Decision(Action.ACCEPT_DFT, "below configured reference-risk threshold")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostLedger:
    """Charge each operation key once; reject inconsistent cached-operation cost."""

    budget: float | None = None
    cost_unit: str = "relative"
    charges: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.budget is not None:
            nonnegative(self.budget, "budget")
        if not self.cost_unit:
            raise ValueError("cost_unit required")
        for value in self.charges.values():
            nonnegative(value, "charge")
        if self.budget is not None and self.spent > self.budget:
            raise ValueError("initial charges exceed the budget")

    @property
    def spent(self) -> float:
        from math import fsum

        return fsum(self.charges.values())

    def buy(self, operations: dict[str, float]) -> bool:
        for key, amount in operations.items():
            nonnegative(amount, "cost")
            if key in self.charges and self.charges[key] != amount:
                raise ValueError(f"inconsistent cost for cached operation {key}")
        additional = sum(value for key, value in operations.items() if key not in self.charges)
        if self.budget is not None and self.spent + additional > self.budget + 1e-12:
            return False
        self.charges.update(operations)
        return True
