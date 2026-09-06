"""
Tests for Autonomous Experiment Orchestrator Closed-Loop Engine.
"""

import pytest
from ghost_layer.experiments.orchestrator import (
    ExperimentOrchestrator,
    OrchestratedTrialResult,
)
from ghost_layer.control.actuators import (
    StructuredAction,
    ActuatorRiskTier,
    RollbackType,
)
from ghost_layer.verification.quality_gate import (
    QualityPolicy,
    QualityMetricType,
    QualityDirection,
    TriGateAction,
)
from ghost_layer.experiments.schema import (
    HardwareEnvironment,
    WorkloadProfile,
)
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, OutcomeStatus


def test_orchestrator_commit_lifecycle_happy_path(tmp_path):
    kb = SharedKnowledgeBase(db_file_path=str(tmp_path / "test_kb.json"))
    orchestrator = ExperimentOrchestrator(knowledge_base=kb, gpu_cost_per_hour=3.50, min_speedup_pct=5.0)

    ctx = {"mixed_precision": "fp32"}
    action = StructuredAction(
        action_type="SET_MIXED_PRECISION",
        target_param="mixed_precision",
        target_value="bf16",
        constraints={"allowed_values": ["fp16", "bf16", "fp32"]},
        risk_tier=ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL,
        rollback_type=RollbackType.IN_MEMORY,
        description="Enable BF16 Mixed Precision",
    )

    # 20% speedup, lossless
    base_step = [100.0, 101.0, 99.0, 100.0]
    cand_step = [80.0, 81.0, 79.0, 80.0]
    base_loss = [2.00, 2.01, 1.99, 2.00]
    cand_loss = [2.00, 2.01, 2.00, 2.01]

    res = orchestrator.execute_trial(
        target_context=ctx,
        action=action,
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
    )

    assert res.action_taken == TriGateAction.COMMIT
    assert not res.rollback_executed
    assert ctx["mixed_precision"] == "bf16"
    assert res.speedup_pct > 19.0
    assert res.experiment_record.record_hash is not None
    assert res.economic_impact.measured_speedup_pct > 19.0
    assert res.economic_impact.realized_measured_savings_usd > 0.0

    # Verify KB compounding
    entries = list(kb.entries.values())
    assert len(entries) == 1
    assert entries[0].outcome_status == OutcomeStatus.VERIFIED_NON_INFERIOR.value
    assert entries[0].sample_count == 1


def test_orchestrator_rollback_on_quality_degradation(tmp_path):
    kb = SharedKnowledgeBase(db_file_path=str(tmp_path / "test_kb.json"))
    orchestrator = ExperimentOrchestrator(knowledge_base=kb, gpu_cost_per_hour=3.50)

    ctx = {"batch_multiplier": 1}
    action = StructuredAction(
        action_type="SET_BATCH_MULTIPLIER",
        target_param="batch_multiplier",
        target_value=4,
        constraints={"min": 1, "max": 8},
        rollback_type=RollbackType.CONFIG_REVERT,
        description="Scale batch 4x",
    )

    # Speedup is good, but quality degrades significantly
    base_step = [100.0, 100.0]
    cand_step = [60.0, 60.0]
    base_loss = [2.00, 2.00]
    cand_loss = [2.60, 2.70]  # 30%+ degradation

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
    assert ctx["batch_multiplier"] == 1  # Successfully rolled back physically

    # KB contains rejected outcome without inflating verified throughput
    entries = list(kb.entries.values())
    assert len(entries) == 1
    assert entries[0].outcome_status == OutcomeStatus.ROLLED_BACK.value
    assert entries[0].throughput_improvement_pct == 0.0
    assert entries[0].rejected_sample_count == 1


def test_orchestrator_reject_on_inconclusive_speedup(tmp_path):
    kb = SharedKnowledgeBase(db_file_path=str(tmp_path / "test_kb.json"))
    orchestrator = ExperimentOrchestrator(knowledge_base=kb, min_speedup_pct=5.0)

    ctx = {"num_workers": 2}
    action = StructuredAction(
        action_type="SET_DATALOADER_CONFIG",
        target_param="num_workers",
        target_value=4,
        constraints={"min": 0, "max": 16},
        rollback_type=RollbackType.CONFIG_REVERT,
    )

    # 0.5% speedup (< 5% min)
    base_step = [100.0, 100.0]
    cand_step = [99.5, 99.5]
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
    assert ctx["num_workers"] == 2  # Reverted
    entries = list(kb.entries.values())
    assert entries[0].outcome_status == OutcomeStatus.REJECTED.value
