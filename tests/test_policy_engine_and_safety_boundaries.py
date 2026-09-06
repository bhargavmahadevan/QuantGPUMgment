"""
Tests for Enterprise ActionPolicyEngine and EnvironmentSafetyPolicy Boundaries.
"""

import pytest
from ghost_layer.control.policy import (
    EnvironmentTier,
    EnvironmentSafetyPolicy,
    ActionPolicyEngine,
    PolicyValidationResult,
)
from ghost_layer.control.actuators import StructuredAction, ActuatorRiskTier


def test_production_strict_policy_allows_core_actions():
    policy = EnvironmentSafetyPolicy.production_default()
    engine = ActionPolicyEngine(policy=policy)

    # Allowed mixed precision
    action_amp = StructuredAction(
        action_type="SET_MIXED_PRECISION",
        target_param="mixed_precision",
        target_value="bf16",
        constraints={"allowed_values": ["fp16", "bf16", "fp32"]},
        risk_tier=ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL,
    )
    res = engine.validate(action_amp)
    assert res.is_allowed
    assert len(res.violation_reasons) == 0


def test_production_strict_policy_blocks_torch_compile():
    policy = EnvironmentSafetyPolicy.production_default()
    engine = ActionPolicyEngine(policy=policy)

    # torch.compile is forbidden in strict production
    action_compile = StructuredAction(
        action_type="SET_TORCH_COMPILE",
        target_param="torch_compile_mode",
        target_value="reduce-overhead",
        constraints={"allowed_values": ["default", "reduce-overhead"]},
        risk_tier=ActuatorRiskTier.RISK_TIER_C_HIGH_RISK_RESEARCH,
    )
    res = engine.validate(action_compile)
    assert not res.is_allowed
    assert any("torch.compile" in r or "not allowlisted" in r for r in res.violation_reasons)


def test_production_strict_policy_blocks_excessive_batch_multiplier():
    policy = EnvironmentSafetyPolicy.production_default()
    engine = ActionPolicyEngine(policy=policy)

    # Multiplier 4x exceeds 2x production limit
    action_batch = StructuredAction(
        action_type="SET_BATCH_MULTIPLIER",
        target_param="batch_multiplier",
        target_value=4.0,
        constraints={"min": 1.0, "max": 8.0},
        risk_tier=ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL,
    )
    res = engine.validate(action_batch)
    assert not res.is_allowed
    assert any("exceeds policy ceiling" in r for r in res.violation_reasons)


def test_staging_canary_policy_allows_torch_compile_and_higher_multiplier():
    policy = EnvironmentSafetyPolicy.staging_canary()
    engine = ActionPolicyEngine(policy=policy)

    action_compile = StructuredAction(
        action_type="SET_TORCH_COMPILE",
        target_param="torch_compile_mode",
        target_value="reduce-overhead",
        constraints={"allowed_values": ["default", "reduce-overhead"]},
        risk_tier=ActuatorRiskTier.RISK_TIER_C_HIGH_RISK_RESEARCH,
    )
    res = engine.validate(action_compile)
    assert res.is_allowed

    action_batch = StructuredAction(
        action_type="SET_BATCH_MULTIPLIER",
        target_param="batch_multiplier",
        target_value=4.0,
        constraints={"min": 1.0, "max": 8.0},
        risk_tier=ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL,
    )
    res_batch = engine.validate(action_batch)
    assert res_batch.is_allowed


def test_unallowlisted_custom_action_rejected():
    policy = EnvironmentSafetyPolicy.production_default()
    engine = ActionPolicyEngine(policy=policy)

    action_unknown = StructuredAction(
        action_type="EXECUTE_RAW_CODE",
        target_param="eval_code",
        target_value="import os; os.system('ls')",
    )
    res = engine.validate(action_unknown)
    assert not res.is_allowed
    assert any("not allowlisted" in r for r in res.violation_reasons)
