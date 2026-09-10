"""Core tools for reference-blind MLIP/QM reliability studies."""

from .errors import ErrorDecomposition, decompose_error, reference_blind_failure
from .escalation import EscalationDecision, EscalationPolicy
from .evaluation import ordering_accuracy, ordering_preserved

__all__ = [
    "ErrorDecomposition",
    "EscalationDecision",
    "EscalationPolicy",
    "decompose_error",
    "ordering_accuracy",
    "ordering_preserved",
    "reference_blind_failure",
]

__version__ = "0.1.0"
