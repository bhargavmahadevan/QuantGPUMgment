"""
Unit tests validating the 3-question verification contract:
1. Did performance improve?
2. Did training behavior remain within acceptable bounds?
3. Is the observed performance difference statistically supported?
"""

import pytest
from ghost_layer.verification.verifier import CorrectnessVerifier, VerificationResult


def test_three_question_all_pass():
    """Verify that when performance improves, losses converge safely, and p-value is solid, all 3 pass."""
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05, max_allowed_relative_loss_shift=0.02)

    base_losses = [2.00, 1.95, 1.90, 1.85, 1.80]
    opt_losses = [1.99, 1.94, 1.89, 1.84, 1.79]  # Slightly faster convergence, negligible delta

    base_times = [100.0, 102.0, 101.0, 99.0, 100.0]
    opt_times = [75.0, 76.0, 74.0, 75.0, 76.0]  # ~25% speedup

    result = verifier.verify_trajectories(
        baseline_losses=base_losses,
        optimized_losses=opt_losses,
        recommendation_id="RULE_MIXED_PRECISION",
        baseline_step_times_ms=base_times,
        candidate_step_times_ms=opt_times,
    )

    assert isinstance(result, VerificationResult)
    assert result.performance_improved is True
    assert result.training_behavior_safe is True
    assert result.statistically_supported is True
    assert result.is_safe is True
    assert result.speedup_pct is not None and result.speedup_pct > 20.0
    assert result.action_taken == "VERIFIED_SAFE"


def test_three_question_performance_regression_rejects():
    """Verify that if candidate is slower, performance_improved is False and is_safe is revoked."""
    verifier = CorrectnessVerifier()

    base_losses = [2.0, 1.9, 1.8]
    opt_losses = [2.0, 1.9, 1.8]

    base_times = [80.0, 81.0, 79.0]
    slow_times = [100.0, 105.0, 99.0]  # Latency regressed

    result = verifier.verify_trajectories(
        baseline_losses=base_losses,
        optimized_losses=opt_losses,
        recommendation_id="RULE_TEST",
        baseline_step_times_ms=base_times,
        candidate_step_times_ms=slow_times,
    )

    assert result.performance_improved is False
    assert result.is_safe is False
    assert result.action_taken == "DOWNGRADED_TO_RECOMMENDATION"
    assert "Performance did not improve" in result.reason


def test_three_question_loss_divergence_rejects():
    """Verify that if loss delta exceeds threshold, training_behavior_safe is False."""
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05)

    base_losses = [2.0, 1.9, 1.8]
    diverged_losses = [2.0, 2.2, 2.5]  # Divergence

    result = verifier.verify_trajectories(
        baseline_losses=base_losses,
        optimized_losses=diverged_losses,
        recommendation_id="RULE_TEST",
    )

    assert result.training_behavior_safe is False
    assert result.is_safe is False
    assert "Max loss delta exceeded" in result.reason


def test_three_question_scipy_welch_exactness():
    """Verify compute_welch_t_test uses SciPy and computes valid t-stat and p-value."""
    s1 = [10.0, 11.0, 12.0, 10.5, 11.5]
    s2 = [20.0, 21.0, 22.0, 20.5, 21.5]

    t_stat, p_val = CorrectnessVerifier.compute_welch_t_test(s1, s2)
    assert t_stat < -10.0
    assert p_val < 0.001
