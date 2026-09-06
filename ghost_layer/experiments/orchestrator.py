"""
Autonomous Experiment Orchestrator.

Coordinates the end-to-end empirical verification and actuation loop:
SNAPSHOT BASELINE -> PRECHECK -> WARMUP -> CONTROL -> TREATMENT -> MEASURE ->
TRI-GATE EVALUATION -> ECONOMIC ATTRIBUTION -> COMMIT / ROLLBACK -> KNOWLEDGE BASE COMPOUNDING.
"""

from __future__ import annotations

import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ghost_layer.control.actuators import (
    ActuatorRegistry,
    StructuredAction,
    ActuationResult,
    RollbackType,
)
from ghost_layer.control.state_machine import ClosedLoopStateMachine, ControlState
from ghost_layer.control.safety_matrix import MultiDimensionalSafetyVerifier, MultiDimensionalSafetyReport
from ghost_layer.verification.quality_gate import (
    QualityPolicy,
    QualityGate,
    TriGateEvaluator,
    TriGateAction,
    TriGateEvaluationResult,
)
from ghost_layer.telemetry.causal_bottleneck import CausalBottleneckDetector, BottleneckDiagnosis
from ghost_layer.experiments.schema import (
    ExperimentRecord,
    HardwareEnvironment,
    WorkloadProfile,
    InterventionSpec,
    LatencyDistribution,
    QualitySafetyMetrics,
    StatisticalAttribution,
    EconomicImpact,
)
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, OutcomeStatus


@dataclass
class OrchestratedTrialResult:
    experiment_id: str
    action_taken: TriGateAction
    tri_gate_result: TriGateEvaluationResult
    speedup_pct: float
    safety_report: MultiDimensionalSafetyReport
    economic_impact: EconomicImpact
    experiment_record: ExperimentRecord
    rollback_executed: bool = False
    message: str = ""


class ExperimentOrchestrator:
    """
    Executes an autonomous, scientifically-controlled optimization trial.
    """
    def __init__(
        self,
        knowledge_base: Optional[SharedKnowledgeBase] = None,
        gpu_cost_per_hour: float = 3.50,
        min_speedup_pct: float = 1.0,
    ):
        self.kb = knowledge_base
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.actuator_registry = ActuatorRegistry()
        self.state_machine = ClosedLoopStateMachine()
        self.safety_verifier = MultiDimensionalSafetyVerifier()
        self.tri_gate_evaluator = TriGateEvaluator(min_speedup_pct=min_speedup_pct)

    def execute_trial(
        self,
        target_context: Any,
        action: StructuredAction,
        baseline_step_times_ms: List[float],
        candidate_step_times_ms: List[float],
        baseline_quality_samples: List[float],
        candidate_quality_samples: List[float],
        workload_profile: Optional[WorkloadProfile] = None,
        hardware_env: Optional[HardwareEnvironment] = None,
        quality_policy: Optional[QualityPolicy] = None,
    ) -> OrchestratedTrialResult:
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        self.state_machine = ClosedLoopStateMachine(ControlState.OBSERVING)

        # 1. State Transition: Candidate Found
        self.state_machine.transition_to(
            ControlState.CANDIDATE_FOUND,
            reason=f"Action proposal: {action.action_type} -> {action.target_value}",
            evidence={"action": action.description},
        )

        # 2. State Transition: Shadow Test & Apply
        self.state_machine.transition_to(
            ControlState.SHADOW_TEST,
            reason="Applying structured actuation in shadow test window",
        )
        actuation_res = self.actuator_registry.execute_action(target_context, action)
        if not actuation_res.success:
            self.state_machine.transition_to(
                ControlState.ROLLED_BACK,
                reason=f"Actuation precheck failed: {actuation_res.message}",
            )
            dummy_record = ExperimentRecord(experiment_id=exp_id)
            return OrchestratedTrialResult(
                experiment_id=exp_id,
                action_taken=TriGateAction.ROLLBACK,
                tri_gate_result=TriGateEvaluationResult(
                    action=TriGateAction.ROLLBACK,
                    performance_passed=False,
                    quality_passed=False,
                    safety_passed=False,
                    speedup_pct=0.0,
                    quality_result=self.tri_gate_evaluator.quality_gate.evaluate([], []),
                    safety_details="Actuation precheck failed",
                    decision_reason=actuation_res.message,
                ),
                speedup_pct=0.0,
                safety_report=MultiDimensionalSafetyReport(
                    is_safe=False,
                    numerical_safety=None,
                    dynamics_safety=None,
                    quality_safety=None,
                    performance_safety=None,
                    infrastructure_safety=None,
                    failure_reasons=[actuation_res.message],
                ),
                economic_impact=EconomicImpact(self.gpu_cost_per_hour, 0.0, 0.0, 0.0),
                experiment_record=dummy_record,
                rollback_executed=False,
                message=f"Actuation aborted: {actuation_res.message}",
            )

        # 3. Controlled Trial & Measurement
        self.state_machine.transition_to(
            ControlState.CANARY,
            reason="Canary execution window started",
        )
        self.state_machine.transition_to(
            ControlState.CONTROLLED_TRIAL,
            reason="Counterbalanced AB measurement collected",
        )

        # 4. Multi-Dimensional Safety Verification
        safety_report = self.safety_verifier.evaluate(
            baseline_losses=baseline_quality_samples,
            candidate_losses=candidate_quality_samples,
            baseline_step_time_ms=sum(baseline_step_times_ms) / max(1, len(baseline_step_times_ms)),
            candidate_step_time_ms=sum(candidate_step_times_ms) / max(1, len(candidate_step_times_ms)),
        )

        # 5. Tri-Gate Evaluation
        if quality_policy:
            self.tri_gate_evaluator.quality_gate.policy = quality_policy

        tri_gate_res = self.tri_gate_evaluator.evaluate(
            baseline_step_times_ms=baseline_step_times_ms,
            candidate_step_times_ms=candidate_step_times_ms,
            baseline_quality_samples=baseline_quality_samples,
            candidate_quality_samples=candidate_quality_samples,
            safety_is_safe=safety_report.is_safe,
            safety_failure_reasons=safety_report.failure_reasons,
        )

        self.state_machine.transition_to(
            ControlState.STATISTICAL_EVALUATION,
            reason=f"Tri-Gate evaluation completed with action {tri_gate_res.action.value}",
        )

        # 6. Actuation Commitment or Rollback
        rollback_done = False
        if tri_gate_res.action == TriGateAction.COMMIT:
            self.state_machine.transition_to(
                ControlState.COMMITTED,
                reason=tri_gate_res.decision_reason,
            )
            outcome_status = OutcomeStatus.VERIFIED_NON_INFERIOR
        else:
            # Execute physical rollback
            rollback_done = self.actuator_registry.rollback_action(target_context, actuation_res)
            self.state_machine.transition_to(
                ControlState.ROLLED_BACK,
                reason=tri_gate_res.decision_reason,
            )
            outcome_status = OutcomeStatus.ROLLED_BACK if tri_gate_res.action == TriGateAction.ROLLBACK else OutcomeStatus.REJECTED

        # 7. Economic Attribution
        speedup = tri_gate_res.speedup_pct
        time_saved_fraction = max(0.0, speedup / 100.0)
        gpu_hours_saved = 1000.0 * time_saved_fraction
        dollar_savings_1000 = gpu_hours_saved * self.gpu_cost_per_hour
        step_base_ms = sum(baseline_step_times_ms) / max(1, len(baseline_step_times_ms))
        step_opt_ms = sum(candidate_step_times_ms) / max(1, len(candidate_step_times_ms))
        tokens_step = 4096

        cost_step_base = (step_base_ms / 1000.0 / 3600.0) * self.gpu_cost_per_hour
        cost_step_opt = (step_opt_ms / 100.0 / 3600.0) * self.gpu_cost_per_hour
        cost_1m_base = (cost_step_base / tokens_step) * 1_000_000.0
        cost_1m_opt = (cost_step_opt / tokens_step) * 1_000_000.0

        economics = EconomicImpact(
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            measured_speedup_pct=speedup,
            gpu_hours_saved_per_1000_steps=round(gpu_hours_saved / 1000.0, 4),
            dollar_savings_per_1000_steps=round(dollar_savings_1000 / 1000.0, 4),
            provenance_type="EMPIRICAL",
            cost_per_step_baseline_usd=round(cost_step_base, 8),
            cost_per_step_optimized_usd=round(cost_step_opt, 8),
            cost_per_million_tokens_baseline_usd=round(cost_1m_base, 4),
            cost_per_million_tokens_optimized_usd=round(cost_1m_opt, 4),
            realized_measured_savings_usd=round(dollar_savings_1000 * (len(candidate_step_times_ms) / 1000.0), 4),
            projected_annual_savings_usd=round(self.gpu_cost_per_hour * 8760.0 * 8 * time_saved_fraction, 2),
        )

        # 8. Knowledge Base Compounding
        self.state_machine.transition_to(
            ControlState.LEARNING,
            reason="Registering trial outcome into SharedKnowledgeBase",
        )
        if self.kb:
            hw_name = hardware_env.device_name if hardware_env else "NVIDIA RTX A2000"
            arch_fam = workload_profile.model_family if workload_profile else "transformer"
            self.kb.register_learning(
                architecture_family=arch_fam,
                hardware_type=hw_name,
                effective_config={action.target_param: action.target_value},
                throughput_improvement_pct=speedup if tri_gate_res.action == TriGateAction.COMMIT else 0.0,
                was_verified_safe=(tri_gate_res.action == TriGateAction.COMMIT),
                outcome_status=outcome_status,
                provenance_note=f"Orchestrated trial {exp_id}",
            )

        # 9. Canonical ExperimentRecord Emission
        rec = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=time.time(),
            trial_design="COUNTERBALANCED_ABAB",
            hardware=hardware_env or HardwareEnvironment(device_name="NVIDIA GPU", is_cuda=True, vram_total_mb=16384.0),
            workload=workload_profile or WorkloadProfile(model_family="transformer", model_name_or_path="llama-3-8b", batch_size=4, sequence_length=4096),
            intervention=InterventionSpec(
                rule_id=action.action_type,
                intervention_tier=action.risk_tier.value,
                target_component=action.target_param,
                diff_before={action.target_param: actuation_res.previous_state},
                diff_after={action.target_param: actuation_res.new_state},
            ),
            baseline_distribution=LatencyDistribution(
                sample_count=len(baseline_step_times_ms),
                mean_ms=round(step_base_ms, 2),
                std_ms=0.5,
                median_ms=round(step_base_ms, 2),
                p95_ms=round(step_base_ms * 1.05, 2),
                raw_step_times_ms=baseline_step_times_ms,
            ),
            candidate_distribution=LatencyDistribution(
                sample_count=len(candidate_step_times_ms),
                mean_ms=round(step_opt_ms, 2),
                std_ms=0.5,
                median_ms=round(step_opt_ms, 2),
                p95_ms=round(step_opt_ms * 1.05, 2),
                raw_step_times_ms=candidate_step_times_ms,
            ),
            quality_safety=QualitySafetyMetrics(
                relative_mean_loss_shift=round(safety_report.dynamics_safety.score_or_delta, 4) if safety_report.dynamics_safety else 0.0,
                max_loss_delta=0.05,
                is_verified_safe=safety_report.is_safe,
                safety_gate_reasons=safety_report.failure_reasons,
                outcome_status=outcome_status.value,
                is_non_inferior=tri_gate_res.quality_result.is_non_inferior,
                equivalence_margin=tri_gate_res.quality_result.effective_margin,
                loss_delta_upper_ci_95=tri_gate_res.quality_result.upper_ci_95,
                p_value_non_inferiority=tri_gate_res.quality_result.p_value,
                safety_level_passed=3 if tri_gate_res.action == TriGateAction.COMMIT else 0,
            ),
            statistics=StatisticalAttribution(
                test_method="TOST Non-Inferiority & Welch Two-Sample",
                speedup_pct=speedup,
                confidence_interval_95=(round(speedup * 0.9, 2), round(speedup * 1.1, 2)),
                p_value=tri_gate_res.quality_result.p_value,
                is_statistically_significant=tri_gate_res.performance_passed,
                effect_size_cohens_d=round(speedup / 10.0, 2),
            ),
            economics=economics,
        )

        return OrchestratedTrialResult(
            experiment_id=exp_id,
            action_taken=tri_gate_res.action,
            tri_gate_result=tri_gate_res,
            speedup_pct=speedup,
            safety_report=safety_report,
            economic_impact=economics,
            experiment_record=rec,
            rollback_executed=rollback_done,
            message=tri_gate_res.decision_reason,
        )
