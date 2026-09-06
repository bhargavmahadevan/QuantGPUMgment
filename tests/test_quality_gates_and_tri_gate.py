"""
Tests for Pluggable Quality Gate & Tri-Gate Decision Evaluator.
"""

import pytest
from ghost_layer.verification.quality_gate import (
    QualityMetricType,
    QualityDirection,
    QualityPolicy,
    QualityGate,
    QualityEvaluationResult,
    TriGateAction,
    TriGateEvaluator,
    TriGateEvaluationResult,
)


def test_quality_policy_defaults_and_effective_margin():
    policy = QualityPolicy()
    assert policy.metric_type == QualityMetricType.VALIDATION_LOSS
    assert policy.direction == QualityDirection.LOWER_IS_BETTER
    assert policy.margin_type == "relative"
    assert policy.margin_value == 0.01

    # Relative margin on 2.0 baseline = 0.02
    assert policy.compute_effective_margin(2.0) == pytest.approx(0.02)

    # Absolute margin policy
    abs_policy = QualityPolicy(
        metric_type=QualityMetricType.TASK_ACCURACY,
        metric_name="accuracy_top1",
        direction=QualityDirection.HIGHER_IS_BETTER,
        margin_type="absolute",
        margin_value=0.005,  # 0.5% max drop
    )
    assert abs_policy.compute_effective_margin(0.85) == 0.005


def test_quality_gate_empty_samples():
    gate = QualityGate()
    res = gate.evaluate([], [])
    assert not res.is_passed
    assert not res.is_non_inferior
    assert "Insufficient" in res.details


def test_quality_gate_single_point_lower_is_better():
    # Baseline loss 2.0, Candidate loss 2.01 (0.5% increase <= 1% margin) -> Pass
    policy = QualityPolicy(margin_value=0.01, direction=QualityDirection.LOWER_IS_BETTER)
    gate = QualityGate(policy=policy)
    res = gate.evaluate([2.0], [2.01])
    assert res.is_passed
    assert res.is_non_inferior

    # Baseline loss 2.0, Candidate loss 2.05 (2.5% increase > 1% margin) -> Fail
    res_fail = gate.evaluate([2.0], [2.05])
    assert not res_fail.is_passed
    assert not res_fail.is_non_inferior


def test_quality_gate_single_point_higher_is_better():
    # Accuracy baseline 0.90, Candidate 0.898 (drop of 0.002 <= 0.005 margin) -> Pass
    policy = QualityPolicy(
        metric_type=QualityMetricType.TASK_ACCURACY,
        direction=QualityDirection.HIGHER_IS_BETTER,
        margin_type="absolute",
        margin_value=0.005,
    )
    gate = QualityGate(policy=policy)
    res = gate.evaluate([0.90], [0.898])
    assert res.is_passed

    # Accuracy drop from 0.90 to 0.88 (drop of 0.02 > 0.005 margin) -> Fail
    res_fail = gate.evaluate([0.90], [0.88])
    assert not res_fail.is_passed


def test_quality_gate_tost_multi_sample_non_inferiority():
    policy = QualityPolicy(
        margin_value=0.02,  # 2% margin
        direction=QualityDirection.LOWER_IS_BETTER,
        confidence_level=0.95,
    )
    gate = QualityGate(policy=policy)

    # Clean distributions with minimal loss delta
    baseline_losses = [2.00, 2.01, 1.99, 2.02, 2.00]
    candidate_losses = [2.01, 2.02, 2.00, 2.03, 2.01]
    res = gate.evaluate(baseline_losses, candidate_losses)
    assert res.is_passed
    assert res.is_non_inferior
    assert res.p_value <= 0.05

    # Degraded candidate losses
    bad_candidate_losses = [2.20, 2.25, 2.19, 2.22, 2.21]
    res_bad = gate.evaluate(baseline_losses, bad_candidate_losses)
    assert not res_bad.is_passed
    assert not res_bad.is_non_inferior


def test_tri_gate_evaluator_commit_path():
    evaluator = TriGateEvaluator(min_speedup_pct=5.0)

    # 15% speedup, safe loss, zero safety warnings
    base_step = [100.0, 102.0, 98.0, 100.0]
    cand_step = [85.0, 84.0, 86.0, 85.0]
    base_loss = [2.0, 2.0, 2.0, 2.0]
    cand_loss = [2.005, 2.004, 2.006, 2.005]

    res = evaluator.evaluate(
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
        safety_is_safe=True,
    )

    assert res.action == TriGateAction.COMMIT
    assert res.performance_passed
    assert res.quality_passed
    assert res.safety_passed
    assert res.speedup_pct > 14.0
    assert "Tri-Gate Passed" in res.decision_reason


def test_tri_gate_evaluator_rollback_on_safety_failure():
    evaluator = TriGateEvaluator()

    base_step = [100.0, 100.0]
    cand_step = [80.0, 80.0]
    base_loss = [2.0, 2.0]
    cand_loss = [2.0, 2.0]

    res = evaluator.evaluate(
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
        safety_is_safe=False,
        safety_failure_reasons=["NaN gradient detected in Layer 4"],
    )

    assert res.action == TriGateAction.ROLLBACK
    assert not res.safety_passed
    assert "Safety Gate Violated" in res.decision_reason


def test_tri_gate_evaluator_rollback_on_quality_failure():
    evaluator = TriGateEvaluator()

    base_step = [100.0, 100.0]
    cand_step = [80.0, 80.0]
    base_loss = [2.0, 2.0]
    cand_loss = [2.5, 2.6]  # Massive loss spike

    res = evaluator.evaluate(
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
        safety_is_safe=True,
    )

    assert res.action == TriGateAction.ROLLBACK
    assert not res.quality_passed
    assert "Quality Gate Violated" in res.decision_reason


def test_tri_gate_evaluator_reject_on_inconclusive_performance():
    evaluator = TriGateEvaluator(min_speedup_pct=3.0)

    # Only 1% speedup (< 3% minimum)
    base_step = [100.0, 100.0]
    cand_step = [99.0, 99.0]
    base_loss = [2.0, 2.0]
    cand_loss = [2.0, 2.0]

    res = evaluator.evaluate(
        baseline_step_times_ms=base_step,
        candidate_step_times_ms=cand_step,
        baseline_quality_samples=base_loss,
        candidate_quality_samples=cand_loss,
        safety_is_safe=True,
    )

    assert res.action == TriGateAction.REJECT
    assert not res.performance_passed
    assert res.quality_passed
    assert res.safety_passed
    assert "Performance Gate Inconclusive" in res.decision_reason
