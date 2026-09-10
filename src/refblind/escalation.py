"""Legacy one-shot heuristic, kept for compatibility; use policy.SequentialPolicy.

This API assumes all scores are already available; it is NOT the deployed
sequential policy and must not be used to claim compute savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .validation import positive


class EscalationDecision(str, Enum):
    ACCEPT_MLIP = "accept_mlip"
    RUN_DFT = "run_dft"
    RUN_HIGH_LEVEL = "run_high_level"


@dataclass(frozen=True)
class EscalationPolicy:
    """Simple deterministic baseline for hierarchical fidelity escalation.

    Inputs are assumed to be calibrated risk scores in [0, 1]. The policy is
    intentionally transparent: it is a baseline and research scaffold, not a
    claim that thresholding is the final optimal decision rule.
    """

    mlip_uq_trigger: float = 0.5
    electronic_risk_trigger: float = 0.5
    cost_penalty: float = 0.05
    dft_relative_cost: float = 1.0
    high_level_relative_cost: float = 20.0

    def __post_init__(self) -> None:
        for name in ("mlip_uq_trigger", "electronic_risk_trigger", "cost_penalty"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        positive(self.dft_relative_cost, "dft_relative_cost")
        positive(self.high_level_relative_cost, "high_level_relative_cost")

    @staticmethod
    def _risk(value: float, name: str) -> float:
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
        return value

    def decide(self, *, mlip_uncertainty: float, electronic_risk: float) -> EscalationDecision:
        """Choose the least expensive action whose risk-adjusted trigger fires.

        Higher-level escalation is considered first because ``electronic_risk``
        is intended to target possible reference-method failure. A normalized
        relative-cost penalty prevents a nominal threshold crossing from forcing
        escalation when its margin is negligible relative to configured cost.
        """

        uq = self._risk(mlip_uncertainty, "mlip_uncertainty")
        erisk = self._risk(electronic_risk, "electronic_risk")
        max_cost = max(self.dft_relative_cost, self.high_level_relative_cost)

        hl_margin = erisk - self.electronic_risk_trigger
        hl_penalty = self.cost_penalty * (self.high_level_relative_cost / max_cost)
        if hl_margin >= hl_penalty:
            return EscalationDecision.RUN_HIGH_LEVEL

        dft_margin = uq - self.mlip_uq_trigger
        dft_penalty = self.cost_penalty * (self.dft_relative_cost / max_cost)
        if dft_margin >= dft_penalty:
            return EscalationDecision.RUN_DFT

        return EscalationDecision.ACCEPT_MLIP
