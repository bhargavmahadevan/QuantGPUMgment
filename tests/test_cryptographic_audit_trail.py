"""
Unit tests for GhostLayer Cryptographic Audit Trail Engine & Mid-Range Fleet Optimizer
Inspired by codenotary/immudb, auditumio/auditum, javers, and access-control-audit-backend.
"""
import pytest
import time
from datetime import datetime, timezone

from ghost_layer.decision.audit_trail import (
    AuditTrailRecord,
    AuditTrailEngine,
    SanitizationFilter,
    AuditDiffSnapshot,
)
from ghost_layer.roi.midrange_optimizer import MidRangeFleetOptimizer, MidRangeAuditReport


def test_5w_event_schema_creation_and_serialization():
    """Verify that audit records strictly enforce the 5 Ws schema and ISO-8601 UTC timestamps."""
    diff = AuditDiffSnapshot(
        before={"step_time_ms": 3200.0, "gpu_memory_mb": 78000.0, "loss_variance": 0.45},
        after={"step_time_ms": 2080.0, "gpu_memory_mb": 42000.0, "loss_variance": 0.12},
        delta_pct={"step_time_pct": -35.0, "vram_reduction_pct": -46.2, "variance_pct": -73.3}
    )
    
    record = AuditTrailRecord(
        actor="GhostWatcherHook:cuda:0",
        action="optimization.applied.flash_attention",
        resource="layer.transformer_block_12",
        context={"cluster_id": "tier-1a-8x-a100", "run_id": "run-2026-0902", "device_type": "A100-SXM4-80GB"},
        diff=diff,
    )
    
    assert record.actor == "GhostWatcherHook:cuda:0"
    assert record.action == "optimization.applied.flash_attention"
    assert record.resource == "layer.transformer_block_12"
    assert record.timestamp.endswith("Z") or "+00:00" in record.timestamp
    assert record.diff.delta_pct["step_time_pct"] == -35.0


def test_cryptographic_hash_chain_immutability_and_tamper_detection():
    """Verify tamper-evident hash chaining (immudb-style): modifying a past record invalidates the chain."""
    engine = AuditTrailEngine()
    
    # Append 5 sequential audit events
    for step in range(1, 6):
        engine.record_event(
            actor=f"GhostWatcherHook:cuda:{step % 2}",
            action=f"telemetry.step_{step}",
            resource="training_loop",
            context={"step": step, "cluster": "8x-H100"},
            diff=AuditDiffSnapshot(before={"loss": 4.0 - 0.1*step}, after={"loss": 3.8 - 0.1*step})
        )
        
    assert len(engine.records) == 5
    # Verify chain integrity
    is_valid, error_msg = engine.verify_chain_integrity()
    assert is_valid is True
    assert error_msg is None
    
    # Tamper with record #2 (simulate an unauthorized alteration)
    engine.records[2].actor = "MaliciousHacker"
    is_valid_tampered, error_msg_tampered = engine.verify_chain_integrity()
    assert is_valid_tampered is False
    assert "Tamper detected at index 2" in error_msg_tampered


def test_sanitization_filter_redaction():
    """Verify that raw secrets, API tokens, model weights tensors, and prompt PII are redacted."""
    sanitizer = SanitizationFilter()
    
    raw_context = {
        "api_key": "sk-proj-secret-1234567890",
        "huggingface_token": "hf_abc123secretToken",
        "user_email": "engineer@ai-lab.com",
        "cluster_node": "node-04",
        "weight_tensor": [0.12, 0.45, -0.89, 1.22] * 100,  # Large tensor array
        "prompt_text": "Sensitive internal customer conversation..."
    }
    
    sanitized = sanitizer.sanitize_dict(raw_context)
    
    assert sanitized["api_key"] == "[REDACTED_SECRET]"
    assert sanitized["huggingface_token"] == "[REDACTED_SECRET]"
    assert sanitized["user_email"] == "[REDACTED_PII]"
    assert sanitized["cluster_node"] == "node-04"
    assert "[REDACTED_TENSOR_BLOB" in str(sanitized["weight_tensor"])
    assert sanitized["prompt_text"] == "[REDACTED_PROMPT_TEXT]"


def test_midrange_fleet_efficiency_and_profit_maximizer():
    """Verify midrange fleet calculations for 8x A100 / 32x H100 tiers across DataLoader, VRAM, and Muon vectors."""
    optimizer = MidRangeFleetOptimizer()
    
    # Audit Tier 1A: 8x A100 80GB running 70B fine-tuning (50,000 steps)
    report_8x = optimizer.audit_and_optimize_tier(
        tier_name="Tier 1A: 8x A100 SXM4",
        num_gpus=8,
        gpu_hourly_rate=4.00,
        baseline_step_ms=3200.0,
        dataloader_stall_pct=12.5,
        activation_vram_pct=65.0,
        gradient_noise_snr=0.35,
        total_steps=50000,
    )
    
    assert report_8x.num_gpus == 8
    assert report_8x.optimized_step_ms < report_8x.baseline_step_ms
    assert report_8x.net_client_savings_usd > 0
    assert report_8x.ghost_optimization_fee_usd > 0
    assert report_8x.ghost_containment_premium_usd > 0
    assert report_8x.pre_flight_audit_fee_usd == 2500.0
    assert report_8x.total_ghost_revenue_usd > 2500.0
    
    # Audit Tier 2A: 32x H100 running 32B MoE sweep
    report_32x = optimizer.audit_and_optimize_tier(
        tier_name="Tier 2A: 32x H100 InfiniBand",
        num_gpus=32,
        gpu_hourly_rate=3.50,
        baseline_step_ms=3500.0,
        dataloader_stall_pct=10.0,
        activation_vram_pct=72.0,
        gradient_noise_snr=0.40,
        total_steps=40000,
    )
    
    assert report_32x.num_gpus == 32
    assert report_32x.net_client_savings_usd > 1000.0
    assert report_32x.total_ghost_revenue_usd > 2500.0
