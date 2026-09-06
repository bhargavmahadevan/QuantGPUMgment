"""
Tests for DecisionEngine and AutoApplier extensions:
- RULE_CHUNKED_CROSS_ENTROPY
- RULE_MHC_RESIDUAL_ROUTING
- AutoApplier execution for ChunkedCrossEntropy and DataLoaderWorkers
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ghost_layer.decision.engine import DecisionEngine, Recommendation
from ghost_layer.telemetry.watcher import TelemetrySummary
from ghost_layer.verification.verifier import VerificationResult
from ghost_layer.applier import AutoApplier, ConsentLevel
from ghost_layer.curvature.loss import ChunkedCrossEntropyLoss


def test_decision_engine_emits_chunked_cross_entropy():
    engine = DecisionEngine()

    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=35.0,
        avg_gpu_utilization_pct=88.0,
        peak_gpu_memory_mb=68000.0,
        gpu_memory_total_mb=80000.0,  # 85% peak
        avg_step_time_ms=350.0,
        avg_dataloader_stall_pct=1.0,
        mixed_precision="bf16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
        sequence_length=4096,
    )

    verifications = [
        VerificationResult(
            is_safe=True,
            relative_mean_loss_shift=0.001,
            max_loss_delta=0.001,
            reason="safe",
            action_taken="AUTO_APPLIED",
            recommendation_id="RULE_CHUNKED_CROSS_ENTROPY",
        )
    ]

    recs = engine.evaluate(summary, model_type="transformer", verification_results=verifications)
    rule_ids = [r.rule_id for r in recs]

    assert "RULE_CHUNKED_CROSS_ENTROPY" in rule_ids
    ce_rec = next(r for r in recs if r.rule_id == "RULE_CHUNKED_CROSS_ENTROPY")
    assert ce_rec.impact_level == "HIGH"
    assert "ChunkedCrossEntropyLoss" in ce_rec.actionable_code_snippet
    assert ce_rec.safe_to_auto_apply is True


def test_decision_engine_emits_mhc_on_gradient_instability():
    engine = DecisionEngine()

    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=35.0,
        avg_gpu_utilization_pct=90.0,
        peak_gpu_memory_mb=35000.0,
        gpu_memory_total_mb=80000.0,
        avg_step_time_ms=400.0,
        avg_dataloader_stall_pct=1.0,
        mixed_precision="bf16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
    )
    summary.gradient_norm_variance = 8.5  # High variance in gradient norms

    recs = engine.evaluate(summary, model_type="transformer")
    rule_ids = [r.rule_id for r in recs]

    assert "RULE_MHC_RESIDUAL_ROUTING" in rule_ids
    mhc_rec = next(r for r in recs if r.rule_id == "RULE_MHC_RESIDUAL_ROUTING")
    assert mhc_rec.impact_level in ["MEDIUM", "HIGH"]
    assert "ManifoldConstrainedHyperConnections" in mhc_rec.actionable_code_snippet
    # Architectural modifications must not be auto-applied blindly
    assert mhc_rec.safe_to_auto_apply is False


def test_auto_applier_chunked_cross_entropy_audit_only():
    applier = AutoApplier(consent_level=ConsentLevel.AUDIT_ONLY)
    loss_fn = nn.CrossEntropyLoss()
    target_context = {"loss_fn": loss_fn}

    res = applier.apply_recommendation(
        "RULE_CHUNKED_CROSS_ENTROPY",
        recommendation_is_safe=True,
        target_context=target_context,
    )

    assert not res.applied
    assert "Blocked by consent level" in res.error
    assert isinstance(target_context["loss_fn"], nn.CrossEntropyLoss)


def test_auto_applier_chunked_cross_entropy_auto_apply_safe():
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_SAFE)
    loss_fn = nn.CrossEntropyLoss()
    target_context = {"loss_fn": loss_fn}

    res = applier.apply_recommendation(
        "RULE_CHUNKED_CROSS_ENTROPY",
        recommendation_is_safe=True,
        target_context=target_context,
    )

    assert res.applied is True
    assert res.rule_id == "RULE_CHUNKED_CROSS_ENTROPY"
    assert isinstance(target_context["loss_fn"], ChunkedCrossEntropyLoss)


def test_auto_applier_dataloader_workers():
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_SAFE)
    dataset = TensorDataset(torch.randn(16, 4), torch.zeros(16))
    dl = DataLoader(dataset, batch_size=4, num_workers=0)
    target_context = {"dataloader": dl}

    res = applier.apply_recommendation(
        "RULE_DATALOADER_WORKERS",
        recommendation_is_safe=True,
        target_context=target_context,
    )

    assert res.applied is True
    assert res.rule_id == "RULE_DATALOADER_WORKERS"
    assert "dataloader" in target_context
    assert target_context["dataloader"].pin_memory is True
