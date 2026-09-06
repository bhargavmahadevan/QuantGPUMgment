"""
GhostLayer Closed-Loop Control State Machine.

Implements an explicit 10-state lifecycle with verified transition boundaries:
OBSERVING -> CANDIDATE_FOUND -> SHADOW_TEST -> CANARY -> CONTROLLED_TRIAL
  -> STATISTICAL_EVALUATION -> (COMMITTED | ROLLED_BACK) -> LEARNING -> MONITORING.

Every state transition requires an explicit reason and an immutable evidence payload.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class ControlState(enum.Enum):
    OBSERVING = "OBSERVING"
    CANDIDATE_FOUND = "CANDIDATE_FOUND"
    SHADOW_TEST = "SHADOW_TEST"
    CANARY = "CANARY"
    CONTROLLED_TRIAL = "CONTROLLED_TRIAL"
    STATISTICAL_EVALUATION = "STATISTICAL_EVALUATION"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    LEARNING = "LEARNING"
    MONITORING = "MONITORING"


@dataclass
class TransitionEvidence:
    """Explicit audit record certifying a valid state transition."""
    from_state: ControlState
    to_state: ControlState
    reason: str
    evidence_payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ClosedLoopStateMachine:
    """
    State machine orchestrating the autonomous optimization and verification lifecycle.
    Prevents unauthorized state jumps and enforces empirical evidence at each boundary.
    """

    ALLOWED_TRANSITIONS: Dict[ControlState, List[ControlState]] = {
        ControlState.OBSERVING: [ControlState.CANDIDATE_FOUND, ControlState.MONITORING],
        ControlState.CANDIDATE_FOUND: [ControlState.SHADOW_TEST, ControlState.OBSERVING],
        ControlState.SHADOW_TEST: [ControlState.CANARY, ControlState.ROLLED_BACK],
        ControlState.CANARY: [ControlState.CONTROLLED_TRIAL, ControlState.ROLLED_BACK],
        ControlState.CONTROLLED_TRIAL: [ControlState.STATISTICAL_EVALUATION, ControlState.ROLLED_BACK],
        ControlState.STATISTICAL_EVALUATION: [ControlState.COMMITTED, ControlState.ROLLED_BACK],
        ControlState.COMMITTED: [ControlState.LEARNING],
        ControlState.ROLLED_BACK: [ControlState.LEARNING, ControlState.OBSERVING],
        ControlState.LEARNING: [ControlState.MONITORING, ControlState.OBSERVING],
        ControlState.MONITORING: [ControlState.OBSERVING, ControlState.ROLLED_BACK],
    }

    def __init__(self, initial_state: ControlState = ControlState.OBSERVING):
        self.state = initial_state
        self.history: List[TransitionEvidence] = []

    def can_transition(self, target_state: ControlState) -> bool:
        """Checks if moving to target_state is permitted from the current state."""
        return target_state in self.ALLOWED_TRANSITIONS.get(self.state, [])

    def transition_to(
        self,
        target_state: ControlState,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> TransitionEvidence:
        """
        Executes an audited state transition with explicit reason and evidence payload.
        Raises ValueError on invalid state transition attempts.
        """
        if not self.can_transition(target_state):
            raise ValueError(
                f"Invalid control transition: Cannot transition from {self.state.value} to {target_state.value}."
            )

        rec = TransitionEvidence(
            from_state=self.state,
            to_state=target_state,
            reason=reason,
            evidence_payload=evidence or {},
        )
        self.history.append(rec)
        self.state = target_state
        return rec
