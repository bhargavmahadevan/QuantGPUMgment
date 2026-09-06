"""
Unit tests for Section B: VRAM Headroom & Safe Batch Scaling Multiplier.
"""

import pytest
from ghost_layer.telemetry.vram_scaling import (
    calculate_vram_headroom_multiplier,
    VRAMScalingReport,
)
from ghost_layer.applier import AutoApplier, ConsentLevel
from ghost_layer.decision.engine import DecisionEngine, TelemetrySummary


def test_vram_scaling_abundant_headroom():
    """Verify safe 2.0x scaling multiplier when VRAM utilization is very low."""
    # 16GB total, 4GB peak (25% utilization)
    rep = calculate_vram_headroom_multiplier(
        vram_total_mb=16384.0,
        vram_peak_mb=4096.0,
        current_batch_size=4,
        vram_static_mb=2048.0,  # 2GB static, 2GB active
    )
    
    # Headroom = 16384 - 4096 = 12288 MB
    # OOM buffer = 0.15 * 16384 = 2457.6 MB
    # Available = 12288 - 2457.6 = 9830.4 MB
    # Active batch = 2048 MB
    # Increments = floor(9830.4 / 2048) * 0.25 = 4 * 0.25 = 1.0 -> 2.0x
    assert rep.is_scale_up_safe is True
    assert rep.safe_multiplier == 2.0
    assert rep.recommended_batch_size == 8


def test_vram_scaling_tight_headroom():
    """Verify conservative scaling (1.0x) when VRAM is near capacity (OOM buffer breached)."""
    # 16GB total, 14.5GB peak (> 90% utilization)
    rep = calculate_vram_headroom_multiplier(
        vram_total_mb=16384.0,
        vram_peak_mb=14500.0,
        current_batch_size=8,
    )
    
    # Headroom = 16384 - 14500 = 1884 MB < OOM buffer (2457.6 MB)
    assert rep.is_scale_up_safe is False
    assert rep.safe_multiplier == 1.0
    assert rep.recommended_batch_size == 8


def test_applier_batch_scaling_integration():
    """Verify AutoApplier integrates calculate_vram_headroom_multiplier."""
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_ALL)
    
    class DummyLoader:
        batch_size = 4
        
    res = applier.apply_recommendation("RULE_BATCH_SCALING", dataloader=DummyLoader())
    assert res.applied is True
    assert "safe_multiplier" in res.after_config
    assert "vram_headroom_mb" in res.after_config
    assert res.after_config["batch_size"] >= 4


def test_decision_engine_rule_5_integration():
    """Verify DecisionEngine Rule 5 evaluates Section B headroom multiplier."""
    engine = DecisionEngine()
    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=10.0,
        avg_gpu_utilization_pct=45.0,
        peak_gpu_memory_mb=3000.0,
        gpu_memory_total_mb=16000.0,
        avg_step_time_ms=50.0,
        avg_dataloader_stall_pct=2.0,
        mixed_precision="fp16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
    )
    
    recs = engine.evaluate(summary=summary, model_type="transformer")
    batch_rec = next((r for r in recs if r.rule_id == "RULE_BATCH_SCALING"), None)
    assert batch_rec is not None
    assert "Section B VRAM headroom" in batch_rec.description
    assert "Safe headroom multiplier:" in batch_rec.evidence
