import pytest
import math
from ghost_layer.verification.verifier import CorrectnessVerifier, VerificationResult


def test_tost_non_inferiority_safe_trajectory():
    verifier = CorrectnessVerifier(
        max_allowed_loss_delta=0.05,
        max_allowed_relative_loss_shift=0.02,
        min_p_value_threshold=0.05,
    )
    # Baseline losses around 2.00, candidate losses around 2.005 (well within 2% margin = 0.04)
    baseline = [2.00, 1.99, 2.01, 2.00, 1.98, 2.02, 1.99, 2.01, 2.00, 1.99]
    candidate = [2.005, 1.995, 2.012, 2.003, 1.985, 2.022, 1.994, 2.011, 2.002, 1.996]

    is_ni, p_ni, upper_ci, se = verifier.compute_non_inferiority_test(
        baseline, candidate, margin_delta=0.04, alpha=0.05
    )

    assert is_ni is True
    assert p_ni < 0.05
    assert upper_ci <= 0.04


def test_tost_non_inferiority_divergent_trajectory_rejected():
    verifier = CorrectnessVerifier(
        max_allowed_loss_delta=0.05,
        max_allowed_relative_loss_shift=0.02,
        min_p_value_threshold=0.05,
    )
    # Baseline around 2.00, candidate significantly degraded to 2.15 (delta = 0.15 >> margin 0.04)
    baseline = [2.00, 1.99, 2.01, 2.00, 1.98, 2.02, 1.99, 2.01, 2.00, 1.99]
    candidate = [2.15, 2.14, 2.16, 2.15, 2.13, 2.17, 2.14, 2.16, 2.15, 2.14]

    is_ni, p_ni, upper_ci, se = verifier.compute_non_inferiority_test(
        baseline, candidate, margin_delta=0.04, alpha=0.05
    )

    assert is_ni is False
    assert upper_ci > 0.04


def test_level_1_runtime_safety_nan_guard():
    verifier = CorrectnessVerifier()
    baseline = [2.0, 1.95, 1.90]
    candidate_nan = [2.0, float("nan"), 1.85]

    res = verifier.verify_trajectories(
        baseline_losses=baseline,
        optimized_losses=candidate_nan,
        recommendation_id="RULE_TEST_NAN",
    )

    assert res.is_safe is False
    assert res.safety_level_passed == 0
    assert "Level 1 Runtime Failure" in res.reason


def test_verification_result_structured_tost_fields():
    verifier = CorrectnessVerifier(
        max_allowed_loss_delta=0.05,
        max_allowed_relative_loss_shift=0.02,
    )
    baseline_losses = [2.00, 1.98, 1.96, 1.94, 1.92]
    candidate_losses = [2.00, 1.97, 1.95, 1.93, 1.91]
    baseline_times = [30.0, 31.0, 29.5, 30.5, 30.0]
    candidate_times = [15.0, 14.8, 15.2, 14.9, 15.1]

    res = verifier.verify_trajectories(
        baseline_losses=baseline_losses,
        optimized_losses=candidate_losses,
        recommendation_id="RULE_AMP",
        was_auto_applied=True,
        baseline_step_times_ms=baseline_times,
        candidate_step_times_ms=candidate_times,
    )

    assert res.is_safe is True
    assert res.speedup_pct > 40.0
    assert res.is_non_inferior is True
    assert res.equivalence_margin > 0.0
    assert res.loss_delta_upper_ci_95 <= res.equivalence_margin
    assert res.safety_level_passed == 2
    assert res.loss_metric_type == "relative_loss_shift"
