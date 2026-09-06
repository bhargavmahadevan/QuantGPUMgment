"""
GhostLayer Feedback Module: Closed-Loop Outcome Evaluator.
Measures realized before-and-after throughput deltas and loss-shift trajectories
for applied recommendations, writing verified outcomes back to the Knowledge Base.
"""

from ghost_layer.feedback.outcome_evaluator import (
    OutcomeEvaluator,
    OutcomeEvaluation,
    OutcomeEvaluationReport,
)

__all__ = [
    "OutcomeEvaluator",
    "OutcomeEvaluation",
    "OutcomeEvaluationReport",
]
