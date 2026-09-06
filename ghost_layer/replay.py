import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ReplayEvent:
    timestamp: float
    step: int
    stage: str  # OBSERVE, DIAGNOSE, SAFE_APPLY, VERIFY, MONITOR, ROLLBACK
    recommendation_id: str
    action: str
    details: Dict[str, Any]
    verified_safe: Optional[bool] = None
    throughput_delta_pct: float = 0.0
    reason: str = ""

@dataclass
class OptimizationReplayLog:
    session_id: str
    created_at: float = field(default_factory=time.time)
    events: List[ReplayEvent] = field(default_factory=list)

    def record_event(
        self,
        step: int,
        stage: str,
        recommendation_id: str,
        action: str,
        details: Dict[str, Any],
        verified_safe: Optional[bool] = None,
        throughput_delta_pct: float = 0.0,
        reason: str = ""
    ) -> ReplayEvent:
        event = ReplayEvent(
            timestamp=time.time(),
            step=step,
            stage=stage,
            recommendation_id=recommendation_id,
            action=action,
            details=details,
            verified_safe=verified_safe,
            throughput_delta_pct=throughput_delta_pct,
            reason=reason,
        )
        self.events.append(event)
        return event

    def generate_audit_trail(self) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp,
                "step": e.step,
                "stage": e.stage,
                "recommendation_id": e.recommendation_id,
                "action": e.action,
                "details": e.details,
                "verified_safe": e.verified_safe,
                "throughput_delta_pct": e.throughput_delta_pct,
                "reason": e.reason,
            }
            for e in self.events
        ]
