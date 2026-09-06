"""
Unit tests validating Phase 1 architectural and scientific contracts:
1. Complete purge of kl_divergence alias from VerificationResult.
2. Participation of min_p_value_threshold in CorrectnessVerifier safety decisions.
3. Exact SciPy p-value and confidence interval evaluation in SavingsVerifier.
4. Mathematical honesty: orthogonal_gradient_projection and GradientAlignmentFilter.
5. In-place and returned model reference in optimize_memory_format().
6. Recursive sanitization across diff.before and diff.after in AuditTrailRecord.
7. Risk tier categorization (Tier A, Tier B, Tier C, Startup-Only) and AutoApplier enforcement.
"""

import math
import torch
import torch.nn as nn
import pytest

from ghost_layer.verification.verifier import VerificationResult, CorrectnessVerifier
from ghost_layer.savings import SavingsVerifier, BaselineMeasurement, _SCIPY_AVAILABLE
from ghost_layer.curvature.vector_field import (
    orthogonal_gradient_projection,
    estimate_update_orthogonality_index,
    GradientAlignmentFilter,
)
from ghost_layer.runtime_optimizer import RuntimeOptimizer, RuntimeOptimizationResult
from ghost_layer.decision.audit_trail import SanitizationFilter, AuditDiffSnapshot, AuditTrailRecord
from ghost_layer.applier import (
    AutoApplier,
    ConsentLevel,
    get_rule_tier,
    TIER_A_RULES,
    STARTUP_ONLY_RULES,
    TIER_B_RULES,
    TIER_C_RULES,
)
from ghost_layer.decision.engine import Recommendation


def test_kl_divergence_completely_purged_from_verification_result():
    """Verify that VerificationResult has relative_mean_loss_shift and no kl_divergence field."""
    res = VerificationResult(
        is_safe=True,
        relative_mean_loss_shift=0.0125,
        max_loss_delta=0.04,
        reason="safe",
    )
    assert res.relative_mean_loss_shift == 0.0125
    assert not hasattr(res, "kl_divergence")


def test_min_p_value_threshold_participates_in_safety():
    """
    Verify that an unfavorable divergence that is statistically significant
    (p < min_p_value_threshold) revokes is_safe, even if scalar thresholds pass.
    """
    verifier = CorrectnessVerifier(
        max_allowed_loss_delta=1.0,
        max_allowed_relative_loss_shift=0.50,
        min_p_value_threshold=0.05,
    )
    # Baseline losses steady around 2.0
    base_losses = [2.00, 2.01, 1.99, 2.00, 2.01, 1.99, 2.00, 2.02, 1.98, 2.01]
    # Optimized losses steady around 2.08 (consistently higher with tiny variance -> p < 0.001)
    opt_losses = [2.08, 2.09, 2.07, 2.08, 2.09, 2.07, 2.08, 2.10, 2.07, 2.08]

    res = verifier.verify_trajectories(base_losses, opt_losses, "RULE_DATALOADER_WORKERS")
    # Even though max delta is ~0.10 (< 1.0) and relative shift is ~0.04 (< 0.50),
    # the regression is statistically significant (p < 0.05) and unfavorable (opt > base)
    assert res.is_safe is False
    assert res.action_taken == "DOWNGRADED_TO_RECOMMENDATION"
    assert "Statistically significant unfavorable divergence detected" in res.reason


def test_savings_verifier_uses_scipy_exact_distribution():
    """Verify that SavingsVerifier produces exact statistical metrics via scipy.stats."""
    verifier = SavingsVerifier(gpu_cost_per_hour=3.50)
    base = BaselineMeasurement(
        avg_step_time_ms=100.0,
        std_step_time_ms=5.0,
        avg_gpu_util_pct=60.0,
        avg_memory_mb=4000.0,
        peak_memory_mb=5000.0,
        loss_trajectory=[2.0, 1.9, 1.8],
        step_times_ms=[100.0 + (i % 3) for i in range(30)],
        num_steps=30,
    )
    opt = BaselineMeasurement(
        avg_step_time_ms=80.0,
        std_step_time_ms=4.0,
        avg_gpu_util_pct=75.0,
        avg_memory_mb=4100.0,
        peak_memory_mb=5100.0,
        loss_trajectory=[2.0, 1.9, 1.8],
        step_times_ms=[80.0 + (i % 3) for i in range(30)],
        num_steps=30,
    )
    report = verifier.compare(base, opt)
    assert report.is_statistically_significant is True
    assert report.p_value < 1e-6
    assert report.speedup_pct > 0.0
    assert report.is_verified is True


def test_orthogonal_gradient_projection_and_alignment_filter():
    """Verify mathematical exactness of orthogonal gradient projection."""
    g = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([2.0, 3.0, 0.0])  # Collinear 2.0 along g, orthogonal 3.0 along y

    v_collinear, v_orthogonal = orthogonal_gradient_projection(v, g)
    assert torch.allclose(v_collinear, torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(v_orthogonal, torch.tensor([0.0, 3.0, 0.0]))

    filter_module = GradientAlignmentFilter(damping_factor=0.5)
    filtered = filter_module.filter_step(v, g)
    # v_collinear + (1 - 0.5) * v_orthogonal = [2.0, 1.5, 0.0]
    assert torch.allclose(filtered, torch.tensor([2.0, 1.5, 0.0]))


def test_optimize_memory_format_returns_and_mutates_model():
    """Verify optimize_memory_format mutates model parameters and returns converted reference."""
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)

    model = SimpleConvNet()
    optimizer = RuntimeOptimizer()
    result = optimizer.optimize_memory_format(model, format_name="channels_last")

    assert result.applied is True
    assert result.optimized_model is not None
    # 4D parameter should now have channels_last memory format
    assert model.conv.weight.is_contiguous(memory_format=torch.channels_last)
    assert result.optimized_model.conv.weight.is_contiguous(memory_format=torch.channels_last)


def test_recursive_sanitization_in_audit_trail():
    """Verify that secrets in diff.before and diff.after are redacted prior to hashing."""
    sanitizer = SanitizationFilter()
    diff = AuditDiffSnapshot(
        before={"lr": 1e-4, "api_token": "sk-secret-key-1234567890"},
        after={"lr": 2e-4, "hf_token": "hf_abcdefghijklmnopqrstuvwxyz"},
    )
    sanitized = diff.to_sanitized_dict(sanitizer)
    assert sanitized["before"]["api_token"] == "[REDACTED_SECRET]"
    assert sanitized["after"]["hf_token"] == "[REDACTED_SECRET]"

    record = AuditTrailRecord(
        actor="system",
        action="APPLY",
        resource="optimizer",
        context={"cluster": "us-east-1"},
        diff=diff,
    )
    # Hash must compute deterministically and successfully over sanitized diff
    h = record.compute_hash()
    assert len(h) == 64


def test_risk_tiering_and_applier_enforcement():
    """Verify risk tier classifications and AutoApplier enforcement gating."""
    assert get_rule_tier("RULE_DATALOADER_WORKERS") == "TIER_A_PRODUCTION_SAFE"
    assert get_rule_tier("RULE_GRAPHIFY_TORCH_COMPILE") == "STARTUP_ONLY"
    assert get_rule_tier("RULE_MUON_OPTIMIZER") == "TIER_B_EXPERIMENTAL"
    assert get_rule_tier("RULE_VECTOR_FIELD_VORTICITY_DAMPING") == "TIER_C_RESEARCH"

    # With enforce_risk_tiers=True, Tier B/C and Startup-only are blocked under AUTO_APPLY_SAFE
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_SAFE, enforce_risk_tiers=True)
    assert applier.can_apply(recommendation_is_safe=True, rule_id="RULE_DATALOADER_WORKERS") is True
    assert applier.can_apply(recommendation_is_safe=True, rule_id="RULE_GRAPHIFY_TORCH_COMPILE") is False
    assert applier.can_apply(recommendation_is_safe=True, rule_id="RULE_MUON_OPTIMIZER") is False
    assert applier.can_apply(recommendation_is_safe=True, rule_id="RULE_VECTOR_FIELD_VORTICITY_DAMPING") is False


def test_recommendation_prior_and_evidence_score():
    """Verify Recommendation has recommendation_prior, evidence_score, and intervention_tier."""
    rec = Recommendation(
        rule_id="RULE_MIXED_PRECISION",
        title="Enable AMP bf16",
        impact_level="HIGH",
        speedup_estimate_label="1.8x",
        description="test",
        actionable_code_snippet="",
        safe_to_auto_apply=True,
        confidence=0.88,
    )
    assert rec.recommendation_prior == 0.88
    assert rec.evidence_score == 0.88
    assert rec.intervention_tier == "TIER_A_PRODUCTION_SAFE"
