"""
Unit tests validating the 10-state ClosedLoopStateMachine,
MultiDimensionalSafetyVerifier, and RollbackManager verification.
"""

import pytest
import torch

from ghost_layer.control.state_machine import (
    ControlState,
    ClosedLoopStateMachine,
    TransitionEvidence,
)
from ghost_layer.control.safety_matrix import (
    MultiDimensionalSafetyVerifier,
    MultiDimensionalSafetyReport,
)
from ghost_layer.rollback import RollbackManager, ConfigSnapshot


def test_closed_loop_state_machine_happy_path():
    """Verify state machine follows explicit 10-state lifecycle with transition evidence."""
    sm = ClosedLoopStateMachine(initial_state=ControlState.OBSERVING)
    assert sm.state == ControlState.OBSERVING

    # Step 1: CANDIDATE_FOUND
    sm.transition_to(
        ControlState.CANDIDATE_FOUND,
        reason="Decision engine detected dataloader starvation",
        evidence={"rule_id": "RULE_DATALOADER_WORKERS"},
    )
    assert sm.state == ControlState.CANDIDATE_FOUND

    # Step 2: SHADOW_TEST
    sm.transition_to(
        ControlState.SHADOW_TEST,
        reason="Dry-run memory check passed",
    )
    assert sm.state == ControlState.SHADOW_TEST

    # Step 3: CANARY
    sm.transition_to(
        ControlState.CANARY,
        reason="Executing single-step micro canary",
    )
    assert sm.state == ControlState.CANARY

    # Step 4: CONTROLLED_TRIAL
    sm.transition_to(
        ControlState.CONTROLLED_TRIAL,
        reason="Canary passed, beginning ABAB trial",
    )
    assert sm.state == ControlState.CONTROLLED_TRIAL

    # Step 5: STATISTICAL_EVALUATION
    sm.transition_to(
        ControlState.STATISTICAL_EVALUATION,
        reason="ABAB trial completed, computing block bootstrap CI",
    )
    assert sm.state == ControlState.STATISTICAL_EVALUATION

    # Step 6: COMMITTED
    sm.transition_to(
        ControlState.COMMITTED,
        reason="Block bootstrap p < 0.01 and speedup CI lower > 0",
    )
    assert sm.state == ControlState.COMMITTED

    # Step 7: LEARNING
    sm.transition_to(
        ControlState.LEARNING,
        reason="Compounding speedup prior into shared knowledge base",
    )
    assert sm.state == ControlState.LEARNING

    # Step 8: MONITORING
    sm.transition_to(
        ControlState.MONITORING,
        reason="Monitoring for latency drift and loss shifts",
    )
    assert sm.state == ControlState.MONITORING
    assert len(sm.history) == 8


def test_closed_loop_state_machine_disallows_illegal_transitions():
    """Verify state machine prevents direct unauthorized jumps (e.g. OBSERVING -> COMMITTED)."""
    sm = ClosedLoopStateMachine(initial_state=ControlState.OBSERVING)
    with pytest.raises(ValueError, match="Invalid control transition"):
        sm.transition_to(ControlState.COMMITTED, reason="Illegal jump")


def test_closed_loop_state_machine_rollback_branch():
    """Verify state machine transitions to ROLLED_BACK from statistical evaluation."""
    sm = ClosedLoopStateMachine(initial_state=ControlState.STATISTICAL_EVALUATION)
    sm.transition_to(
        ControlState.ROLLED_BACK,
        reason="Speedup was not statistically significant",
    )
    assert sm.state == ControlState.ROLLED_BACK


def test_multidimensional_safety_verifier_all_pass():
    """Verify MultiDimensionalSafetyVerifier passes when all 5 dimensions are clean."""
    verifier = MultiDimensionalSafetyVerifier()
    report = verifier.evaluate(
        baseline_losses=[2.0, 1.9, 1.8],
        candidate_losses=[2.0, 1.91, 1.79],
        candidate_grad_norm=1.5,
        baseline_val_loss=1.85,
        candidate_val_loss=1.84,
        baseline_step_time_ms=100.0,
        candidate_step_time_ms=80.0,
        peak_vram_mb=8000.0,
        total_vram_mb=16384.0,
        is_thermal_throttled=False,
    )
    assert report.is_safe is True
    assert report.summary_label == "ALL_GATES_PASSED"
    assert report.numerical_safety.passed is True
    assert report.dynamics_safety.passed is True
    assert report.quality_safety.passed is True
    assert report.performance_safety.passed is True
    assert report.infrastructure_safety.passed is True


def test_multidimensional_safety_verifier_rejection_conditions():
    """Verify verifier isolates individual failure dimensions."""
    verifier = MultiDimensionalSafetyVerifier()

    # Rejection 1: Thermal throttle
    rep_thermal = verifier.evaluate(
        baseline_losses=[2.0],
        candidate_losses=[2.0],
        is_thermal_throttled=True,
    )
    assert rep_thermal.is_safe is False
    assert rep_thermal.infrastructure_safety.passed is False
    assert any("thermal throttling" in r.lower() for r in rep_thermal.failure_reasons)

    # Rejection 2: VRAM headroom exceeded (> 95%)
    rep_vram = verifier.evaluate(
        baseline_losses=[2.0],
        candidate_losses=[2.0],
        peak_vram_mb=98.0,
        total_vram_mb=100.0,
    )
    assert rep_vram.is_safe is False
    assert rep_vram.infrastructure_safety.passed is False


def test_rollback_manager_verify_restoration():
    """Verify RollbackManager.verify_state_restoration correctly checks config consistency."""
    mgr = RollbackManager()
    dummy_model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

    snap = mgr.snapshot(model=dummy_model, optimizer=optimizer, rule_id="RULE_TEST")
    assert mgr.verify_state_restoration(snap, optimizer=optimizer) is True

    # Mutate lr
    optimizer.param_groups[0]["lr"] = 0.05
    assert mgr.verify_state_restoration(snap, optimizer=optimizer) is False
