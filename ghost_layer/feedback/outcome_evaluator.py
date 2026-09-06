"""
Closed-Loop Outcome Evaluator:
Tracks training step latency and loss trajectory before and after applying optimizations.
Measures realized speedup (ΔThroughput) and loss-shift proxy, and writes empirical results
back to the SharedKnowledgeBase and ReplayLog to close the feedback loop.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, KnowledgeEntry
from ghost_layer.replay import OptimizationReplayLog


@dataclass
class OutcomeEvaluation:
    """Evaluation result for an applied optimization rule."""
    rule_id: str
    architecture_family: str
    hardware_type: str
    apply_step: int
    baseline_window_steps: int
    eval_window_steps: int
    baseline_avg_step_ms: float
    post_avg_step_ms: float
    realized_speedup_pct: float
    baseline_avg_loss: float
    post_avg_loss: float
    loss_shift_delta: float
    is_safe: bool
    registered_to_kb: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class OutcomeEvaluationReport:
    """Summary of all evaluated outcomes during a training session."""
    session_id: str
    total_evaluations: int
    verified_speedups: List[OutcomeEvaluation]
    rejected_outcomes: List[OutcomeEvaluation]
    avg_realized_speedup_pct: float


class OutcomeEvaluator:
    """
    Monitors live step-by-step telemetry to evaluate the empirical outcome of applied recommendations.
    
    Workflow:
    1. Records pre-apply baseline step durations and loss values.
    2. When an optimization applies, marks the transition step.
    3. Records post-apply steps over a configurable evaluation window (default: 20-30 steps).
    4. Computes realized speedup %: ((baseline_ms - post_ms) / baseline_ms) * 100.
    5. Computes loss shift proxy: |mean(L_post) - mean(L_base)| / max(1e-5, mean(L_base)).
    6. Verifies loss safety against the 0.10 threshold.
    7. Updates SharedKnowledgeBase with empirical sample (success or rejection).
    8. Emits a closed-loop EVALUATE_OUTCOME event to OptimizationReplayLog.
    """

    def __init__(
        self,
        knowledge_base: Optional[SharedKnowledgeBase] = None,
        replay_log: Optional[OptimizationReplayLog] = None,
        baseline_window_size: int = 25,
        eval_window_size: int = 25,
        loss_shift_threshold: float = 0.10,
    ):
        self.knowledge_base = knowledge_base or SharedKnowledgeBase()
        self.replay_log = replay_log
        self.baseline_window_size = baseline_window_size
        self.eval_window_size = eval_window_size
        self.loss_shift_threshold = loss_shift_threshold

        # Step history: list of (step, step_time_ms, loss)
        self.step_history: List[Tuple[int, float, Optional[float]]] = []

        # Active pending evaluations: rule_id -> evaluation state dict
        self.pending_evaluations: List[Dict[str, Any]] = []

        # Completed evaluations
        self.completed_evaluations: List[OutcomeEvaluation] = []

    def record_step(self, step: int, step_time_ms: float, loss: Optional[float] = None):
        """Record telemetry for a training step and progress active evaluations."""
        self.step_history.append((step, step_time_ms, loss))

        # Check and update pending evaluations
        to_finalize = []
        for eval_state in self.pending_evaluations:
            eval_state["post_step_times"].append(step_time_ms)
            if loss is not None:
                eval_state["post_losses"].append(loss)

            if len(eval_state["post_step_times"]) >= eval_state["eval_window_size"]:
                to_finalize.append(eval_state)

        for eval_state in to_finalize:
            self.pending_evaluations.remove(eval_state)
            self._finalize_evaluation(eval_state)

    def on_recommendation_applied(
        self,
        rule_id: str,
        step: int,
        architecture_family: str = "transformer",
        hardware_type: str = "NVIDIA GPU",
        effective_config: Optional[Dict[str, Any]] = None,
        param_count_bucket: str = "",
        framework_version: str = "",
    ):
        """
        Notify evaluator that an optimization was applied at the given step.
        Gathers prior baseline steps and starts tracking post-apply window.
        """
        # Collect baseline steps (up to baseline_window_size prior steps)
        prior_steps = [s for s in self.step_history if s[0] <= step]
        baseline_slice = prior_steps[-self.baseline_window_size:] if prior_steps else []

        baseline_step_times = [s[1] for s in baseline_slice]
        baseline_losses = [s[2] for s in baseline_slice if s[2] is not None]

        # If no prior steps, use current step time if available
        if not baseline_step_times and self.step_history:
            baseline_step_times = [self.step_history[-1][1]]

        eval_state = {
            "rule_id": rule_id,
            "apply_step": step,
            "architecture_family": architecture_family,
            "hardware_type": hardware_type,
            "effective_config": effective_config or {},
            "param_count_bucket": param_count_bucket,
            "framework_version": framework_version,
            "baseline_step_times": baseline_step_times,
            "baseline_losses": baseline_losses,
            "post_step_times": [],
            "post_losses": [],
            "eval_window_size": self.eval_window_size,
        }
        self.pending_evaluations.append(eval_state)

    def _finalize_evaluation(self, eval_state: Dict[str, Any]) -> OutcomeEvaluation:
        """Compute realized speedup, verify loss safety, and write outcome to KB."""
        rule_id = eval_state["rule_id"]
        arch = eval_state["architecture_family"]
        hw = eval_state["hardware_type"]
        config = eval_state["effective_config"]
        param_bucket = eval_state["param_count_bucket"]
        fw_ver = eval_state["framework_version"]

        base_times = eval_state["baseline_step_times"]
        post_times = eval_state["post_step_times"]
        base_losses = eval_state["baseline_losses"]
        post_losses = eval_state["post_losses"]

        base_avg_ms = float(np.mean(base_times)) if base_times else 1.0
        post_avg_ms = float(np.mean(post_times)) if post_times else base_avg_ms

        # Realized speedup %: ((T_base - T_post) / T_base) * 100
        if base_avg_ms > 0:
            speedup_pct = round(((base_avg_ms - post_avg_ms) / base_avg_ms) * 100.0, 2)
        else:
            speedup_pct = 0.0

        # Loss shift calculation
        base_loss_avg = float(np.mean(base_losses)) if base_losses else 1.0
        post_loss_avg = float(np.mean(post_losses)) if post_losses else base_loss_avg

        if abs(base_loss_avg) > 1e-5:
            loss_shift = round(abs(post_loss_avg - base_loss_avg) / abs(base_loss_avg), 4)
        else:
            loss_shift = round(abs(post_loss_avg - base_loss_avg), 4)

        is_safe = bool(loss_shift <= self.loss_shift_threshold)
        is_beneficial = bool(is_safe and speedup_pct > 0.0)

        # Write outcome to Knowledge Base
        registered_entry = None
        if self.knowledge_base is not None:
            registered_entry = self.knowledge_base.register_learning(
                architecture_family=arch,
                hardware_type=hw,
                effective_config=config,
                throughput_improvement_pct=max(0.0, speedup_pct) if is_beneficial else 0.0,
                was_verified_safe=is_beneficial,
                param_count_bucket=param_bucket,
                framework_version=fw_ver,
            )
            if hasattr(self.knowledge_base, "register_rule_outcome"):
                self.knowledge_base.register_rule_outcome(
                    rule_id=rule_id,
                    was_verified_safe=is_beneficial,
                    realized_speedup_pct=speedup_pct,
                    loss_shift=loss_shift,
                )

        evaluation = OutcomeEvaluation(
            rule_id=rule_id,
            architecture_family=arch,
            hardware_type=hw,
            apply_step=eval_state["apply_step"],
            baseline_window_steps=len(base_times),
            eval_window_steps=len(post_times),
            baseline_avg_step_ms=round(base_avg_ms, 2),
            post_avg_step_ms=round(post_avg_ms, 2),
            realized_speedup_pct=speedup_pct,
            baseline_avg_loss=round(base_loss_avg, 4),
            post_avg_loss=round(post_loss_avg, 4),
            loss_shift_delta=loss_shift,
            is_safe=is_safe,
            registered_to_kb=registered_entry is not None,
        )
        self.completed_evaluations.append(evaluation)

        # Log event to Replay Log
        if self.replay_log is not None:
            action_name = "OUTCOME_VERIFIED_BENEFICIAL" if is_beneficial else (
                "OUTCOME_LOSS_SHIFT_EXCEEDED" if not is_safe else "OUTCOME_NO_SPEEDUP"
            )
            self.replay_log.record_event(
                step=eval_state["apply_step"] + len(post_times),
                stage="EVALUATE_OUTCOME",
                recommendation_id=rule_id,
                action=action_name,
                details={
                    "baseline_ms": evaluation.baseline_avg_step_ms,
                    "post_ms": evaluation.post_avg_step_ms,
                    "speedup_pct": speedup_pct,
                    "loss_shift": loss_shift,
                    "safe": is_safe,
                    "beneficial": is_beneficial,
                    "kb_sample_count": registered_entry.sample_count if registered_entry else 0,
                },
                verified_safe=is_safe,
                throughput_delta_pct=speedup_pct,
                reason=(
                    f"Measured realized throughput delta of {speedup_pct:+.1f}% "
                    f"(loss shift: {loss_shift:.3f}, threshold <= {self.loss_shift_threshold}). "
                    f"Outcome registered to KnowledgeBase."
                ),
            )

        return evaluation

    def get_report(self, session_id: str = "default_session") -> OutcomeEvaluationReport:
        """Generate summary report of all evaluated outcomes."""
        verified = [e for e in self.completed_evaluations if e.is_safe and e.realized_speedup_pct > 0]
        rejected = [e for e in self.completed_evaluations if not e.is_safe or e.realized_speedup_pct <= 0]
        avg_speedup = float(np.mean([e.realized_speedup_pct for e in verified])) if verified else 0.0

        return OutcomeEvaluationReport(
            session_id=session_id,
            total_evaluations=len(self.completed_evaluations),
            verified_speedups=verified,
            rejected_outcomes=rejected,
            avg_realized_speedup_pct=round(avg_speedup, 2),
        )
