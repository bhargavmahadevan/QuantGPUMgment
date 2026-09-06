"""
Tests for Structured Actuators & Physical Rollback Subsystem.
"""

import pytest
from ghost_layer.control.actuators import (
    RollbackType,
    ActuatorRiskTier,
    StructuredAction,
    ActuationResult,
    ActuatorRegistry,
    MixedPrecisionActuator,
    AttentionBackendActuator,
    DataPipelineActuator,
    VRAMBatchScalingActuator,
    TorchCompileActuator,
)


def test_structured_action_constraint_validation():
    # Valid action
    action = StructuredAction(
        action_type="SET_DATALOADER_CONFIG",
        target_param="num_workers",
        target_value=8,
        constraints={"min": 0, "max": 16},
    )
    valid, msg = action.validate_constraints()
    assert valid

    # Value above max
    invalid_action = StructuredAction(
        action_type="SET_DATALOADER_CONFIG",
        target_param="num_workers",
        target_value=32,
        constraints={"min": 0, "max": 16},
    )
    valid_bad, msg_bad = invalid_action.validate_constraints()
    assert not valid_bad
    assert "exceeds maximum" in msg_bad

    # Value not in allowed values
    action_enum = StructuredAction(
        action_type="SET_MIXED_PRECISION",
        target_param="mixed_precision",
        target_value="fp8",
        constraints={"allowed_values": ["fp16", "bf16", "fp32"]},
    )
    valid_enum, msg_enum = action_enum.validate_constraints()
    assert not valid_enum
    assert "not in allowed set" in msg_enum


def test_mixed_precision_actuator_apply_and_rollback():
    ctx = {"mixed_precision": "fp32"}
    actuator = MixedPrecisionActuator()

    action = StructuredAction(
        action_type="SET_MIXED_PRECISION",
        target_param="mixed_precision",
        target_value="bf16",
        constraints={"allowed_values": ["fp16", "bf16", "fp32"]},
        risk_tier=ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL,
        rollback_type=RollbackType.IN_MEMORY,
    )

    res = actuator.apply(ctx, action)
    assert res.success
    assert ctx["mixed_precision"] == "bf16"
    assert res.previous_state == "fp32"
    assert res.rollback_type == RollbackType.IN_MEMORY

    # Execute rollback
    ok = actuator.rollback(ctx, res)
    assert ok
    assert ctx["mixed_precision"] == "fp32"


def test_attention_backend_actuator_apply_and_rollback():
    ctx = {"attention_backend": "math"}
    actuator = AttentionBackendActuator()

    action = StructuredAction(
        action_type="SET_ATTENTION_BACKEND",
        target_param="attention_backend",
        target_value="sdpa_flash",
        constraints={"allowed_values": ["math", "sdpa_flash", "sdpa_mem_efficient"]},
    )

    res = actuator.apply(ctx, action)
    assert res.success
    assert ctx["attention_backend"] == "sdpa_flash"

    # Rollback
    actuator.rollback(ctx, res)
    assert ctx["attention_backend"] == "math"


def test_dataloader_actuator_apply_and_rollback():
    ctx = {"num_workers": 2}
    actuator = DataPipelineActuator()

    action = StructuredAction(
        action_type="SET_DATALOADER_CONFIG",
        target_param="num_workers",
        target_value=8,
        constraints={"min": 0, "max": 16},
        rollback_type=RollbackType.CONFIG_REVERT,
    )

    res = actuator.apply(ctx, action)
    assert res.success
    assert ctx["num_workers"] == 8

    # Rollback
    actuator.rollback(ctx, res)
    assert ctx["num_workers"] == 2


def test_vram_batch_scaling_actuator_apply_and_rollback():
    ctx = {"batch_multiplier": 1}
    actuator = VRAMBatchScalingActuator()

    action = StructuredAction(
        action_type="SET_BATCH_MULTIPLIER",
        target_param="batch_multiplier",
        target_value=2,
        constraints={"min": 1, "max": 4},
        rollback_type=RollbackType.CONFIG_REVERT,
    )

    res = actuator.apply(ctx, action)
    assert res.success
    assert ctx["batch_multiplier"] == 2

    # Rollback
    actuator.rollback(ctx, res)
    assert ctx["batch_multiplier"] == 1


def test_torch_compile_actuator_apply_and_rollback():
    ctx = {"torch_compile_mode": None}
    actuator = TorchCompileActuator()

    action = StructuredAction(
        action_type="SET_TORCH_COMPILE",
        target_param="torch_compile_mode",
        target_value="reduce-overhead",
        constraints={"allowed_values": ["default", "reduce-overhead", "max-autotune"]},
        rollback_type=RollbackType.CHECKPOINT_RESTORE,
    )

    res = actuator.apply(ctx, action)
    assert res.success
    assert ctx["torch_compile_mode"] == "reduce-overhead"
    assert res.rollback_type == RollbackType.CHECKPOINT_RESTORE

    # Rollback
    actuator.rollback(ctx, res)
    assert ctx["torch_compile_mode"] is None


def test_actuator_registry_dispatch_and_error_handling():
    registry = ActuatorRegistry()
    ctx = {"mixed_precision": "fp32"}

    # Unknown action
    unknown_action = StructuredAction(
        action_type="UNKNOWN_RULE",
        target_param="foo",
        target_value="bar",
    )
    res_bad = registry.execute_action(ctx, unknown_action)
    assert not res_bad.success
    assert "Unknown action type" in res_bad.message

    # Valid dispatched action
    good_action = StructuredAction(
        action_type="SET_MIXED_PRECISION",
        target_param="mixed_precision",
        target_value="fp16",
        constraints={"allowed_values": ["fp16", "bf16", "fp32"]},
    )
    res_good = registry.execute_action(ctx, good_action)
    assert res_good.success
    assert ctx["mixed_precision"] == "fp16"

    # Rollback via registry
    rb_ok = registry.rollback_action(ctx, res_good)
    assert rb_ok
    assert ctx["mixed_precision"] == "fp32"
