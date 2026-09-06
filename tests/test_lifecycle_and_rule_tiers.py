import pytest
from ghost_layer.decision.engine import (
    OptimizationLifecycle,
    RuleTier,
    Recommendation,
    get_confidence_breakdown,
)


def test_lifecycle_and_rule_tier_enums():
    assert OptimizationLifecycle.STATIC.value == "STATIC"
    assert OptimizationLifecycle.STARTUP_ONLY.value == "STARTUP_ONLY"
    assert OptimizationLifecycle.EPOCH_BOUNDARY.value == "EPOCH_BOUNDARY"
    assert OptimizationLifecycle.STEP_BOUNDARY.value == "STEP_BOUNDARY"
    assert OptimizationLifecycle.DYNAMIC.value == "DYNAMIC"

    assert RuleTier.TIER_A_CORE_PRODUCTION.value == "TIER_A_CORE_PRODUCTION"
    assert RuleTier.TIER_B_RESEARCH_EXPERIMENTAL.value == "TIER_B_RESEARCH_EXPERIMENTAL"


def test_recommendation_decision_score_and_tiers():
    rec = Recommendation(
        rule_id="RULE_MIXED_PRECISION_FP16",
        title="Activate PyTorch Mixed Precision (AMP)",
        impact_level="HIGH",
        speedup_estimate_label="2.0x - 2.8x speedup",
        description="Converts FP32 operations to AMP FP16.",
        actionable_code_snippet="torch.cuda.amp.autocast()",
        safe_to_auto_apply=True,
        confidence=0.92,
        lifecycle=OptimizationLifecycle.STEP_BOUNDARY.value,
        rule_tier=RuleTier.TIER_A_CORE_PRODUCTION.value,
    )

    assert rec.decision_score == 0.92
    assert rec.evidence_score == 0.92
    assert rec.lifecycle == "STEP_BOUNDARY"
    assert rec.rule_tier == "TIER_A_CORE_PRODUCTION"


def test_confidence_breakdown_decision_score():
    breakdown_no_runs = get_confidence_breakdown(
        hardware_match=1.0,
        model_similarity=1.0,
        historical_verifications=0,
        rollback_frequency=0.0,
    )

    assert breakdown_no_runs["decision_score"] <= 0.60
    assert breakdown_no_runs["score_type"] == "HEURISTIC_PRIOR"
    assert breakdown_no_runs["no_verified_runs"] is True

    breakdown_with_runs = get_confidence_breakdown(
        hardware_match=1.0,
        model_similarity=1.0,
        historical_verifications=5,
        rollback_frequency=0.0,
    )

    assert breakdown_with_runs["decision_score"] >= 0.85
    assert breakdown_with_runs["score_type"] == "EMPIRICALLY_VERIFIED"
    assert breakdown_with_runs["no_verified_runs"] is False
