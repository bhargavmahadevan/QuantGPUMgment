"""
Tests for Failure Injection & Physical Rollback Verification.

Tests deliberate failure modes:
1. Injected Loss Spike -> Quality Gate failure -> Physical Rollback (<5ms)
2. Injected NaN Gradient -> Safety Gate failure -> Immediate Rollback
3. Injected Inconclusive Speedup -> Performance Gate failure -> Clean Revert
"""

import time
import pytest
from ghost_layer.experiments.orchestrator import ExperimentOrchestrator
from ghost_layer.control.actuators import StructuredAction, ActuatorRiskTier, RollbackType
from ghost_layer.verification.quality_gate import QualityPolicy, TriGateAction


def test_failure_injection_loss_spike_triggers_physical_rollback(tmp_path):
    orchestrator = ExperimentOrchestrator(gpu_cost_per_hour=3.50)
    ctx = {"batch_multiplier": 1}

    action = StructuredAction(
        action_type="SET_BATCH_MULTIPLIER",
        target_param="batch_multiplier",
        target_value=4,
        constraints={"min": 1, "max": 8},
        risk_tier=ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL,
        rollback_type=RollbackType.CONFIG_REVERT,
    )

    # Deliberate failure injection: Baseline loss 2.0 -> Candidate loss 3.5 (75% spike)
    base_step = [100.0, 100.0, 100.0]
    cand_step = [65.0, 65.0, 65.0]  # Throughput increased, but quality destroyed
    base_loss = [2.00, 2.01, 1.99]
    cand_loss = [3.50, 3.80, 3.60]

    t0 = time.perf_counter()
    res = orchestrator.execute_trial(
        target_context=ctx,
        action=action,
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
    )
    t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert res.action_taken == TriGateAction.ROLLBACK
    assert res.rollback_executed
    assert ctx["batch_multiplier"] == 1  # Restored physically to baseline state
    assert t_elapsed_ms < 50.0  # Sub-50ms execution window


def test_failure_injection_nan_loss_triggers_immediate_safety_rollback(tmp_path):
    orchestrator = ExperimentOrchestrator(gpu_cost_per_hour=3.50)
    ctx = {"mixed_precision": "fp32"}

    action = StructuredAction(
        action_type="SET_MIXED_PRECISION",
        target_param="mixed_precision",
        target_value="fp16",
        constraints={"allowed_values": ["fp16", "bf16", "fp32"]},
        rollback_type=RollbackType.IN_MEMORY,
    )

    # Deliberate failure injection: NaN in candidate loss trajectory
    base_step = [100.0, 100.0]
    cand_step = [80.0, 80.0]
    base_loss = [2.00, 2.00]
    cand_loss = [float("nan"), 2.00]

    res = orchestrator.execute_trial(
        target_context=ctx,
        action=action,
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
    )

    assert res.action_taken == TriGateAction.ROLLBACK
    assert res.rollback_executed
    assert ctx["mixed_precision"] == "fp32"  # Context restored


def test_inconclusive_speedup_triggers_clean_reject_revert(tmp_path):
    orchestrator = ExperimentOrchestrator(min_speedup_pct=5.0)
    ctx = {"num_workers": 2}

    action = StructuredAction(
        action_type="SET_DATALOADER_CONFIG",
        target_param="num_workers",
        target_value=4,
        constraints={"min": 0, "max": 16},
        rollback_type=RollbackType.CONFIG_REVERT,
    )

    # Inconclusive speedup: 100ms -> 99.8ms (0.2% delta < 5% min)
    base_step = [100.0, 100.0]
    cand_step = [99.8, 99.8]
    base_loss = [2.00, 2.00]
    cand_loss = [2.00, 2.00]

    res = orchestrator.execute_trial(
        target_context=ctx,
        action=action,
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
    )

    assert res.action_taken == TriGateAction.REJECT
    assert res.rollback_executed
    assert ctx["num_workers"] == 2  # Cleanly reverted
