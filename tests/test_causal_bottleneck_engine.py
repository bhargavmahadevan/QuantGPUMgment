"""
Tests for Causal Bottleneck Diagnosis Engine.
"""

import pytest
from ghost_layer.telemetry.causal_bottleneck import (
    BottleneckType,
    BottleneckDiagnosis,
    CausalBottleneckDetector,
)


def test_diagnose_input_bound_dataloader_stalls():
    # 25ms total step time, 6ms data loading (24% of step) -> INPUT_BOUND
    diag = CausalBottleneckDetector.diagnose(
        gpu_utilization_pct=60.0,
        step_time_ms=25.0,
        data_loading_time_ms=6.0,
        gpu_memory_used_mb=4000.0,
        total_vram_mb=16384.0,
    )
    assert diag.primary_bottleneck == BottleneckType.INPUT_BOUND
    assert "RULE_DATALOADER_WORKERS" in diag.recommended_rules
    assert diag.confidence_score >= 0.90
    assert diag.evidence_metrics["io_fraction"] == pytest.approx(0.24, rel=1e-2)


def test_diagnose_communication_bound_distributed():
    # 100ms total step time, 25ms communication wait (25% of step) -> COMMUNICATION_BOUND
    diag = CausalBottleneckDetector.diagnose(
        gpu_utilization_pct=70.0,
        step_time_ms=100.0,
        data_loading_time_ms=2.0,
        comm_wait_ms=25.0,
    )
    assert diag.primary_bottleneck == BottleneckType.COMMUNICATION_BOUND
    assert "RULE_GRADIENT_ACCUMULATION" in diag.recommended_rules


def test_diagnose_memory_capacity_bound():
    # 15,200 MB used of 16,384 MB (92.7% VRAM) -> MEMORY_CAPACITY_BOUND
    diag = CausalBottleneckDetector.diagnose(
        gpu_utilization_pct=85.0,
        step_time_ms=80.0,
        data_loading_time_ms=1.0,
        gpu_memory_used_mb=15200.0,
        total_vram_mb=16384.0,
    )
    assert diag.primary_bottleneck == BottleneckType.MEMORY_CAPACITY_BOUND
    assert "RULE_GRADIENT_CHECKPOINTING" in diag.recommended_rules
    assert "RULE_VRAM_HEADROOM" in diag.recommended_rules


def test_diagnose_kernel_launch_bound():
    # Low GPU util (45%) with high CPU launch overhead (30ms out of 80ms step) -> KERNEL_LAUNCH_BOUND
    diag = CausalBottleneckDetector.diagnose(
        gpu_utilization_pct=45.0,
        step_time_ms=80.0,
        data_loading_time_ms=1.0,
        cpu_overhead_ms=30.0,
        gpu_memory_used_mb=4000.0,
        total_vram_mb=16384.0,
    )
    assert diag.primary_bottleneck == BottleneckType.KERNEL_LAUNCH_BOUND
    assert "RULE_TORCH_COMPILE" in diag.recommended_rules


def test_diagnose_memory_bandwidth_bound():
    # Low arithmetic intensity (15 FLOPs/byte) -> MEMORY_BANDWIDTH_BOUND
    diag = CausalBottleneckDetector.diagnose(
        gpu_utilization_pct=65.0,
        step_time_ms=50.0,
        data_loading_time_ms=1.0,
        gpu_memory_used_mb=6000.0,
        total_vram_mb=16384.0,
        arithmetic_intensity=15.0,
    )
    assert diag.primary_bottleneck == BottleneckType.MEMORY_BANDWIDTH_BOUND
    assert "RULE_SDPA_ATTENTION" in diag.recommended_rules
    assert "RULE_MIXED_PRECISION_FP16" in diag.recommended_rules


def test_diagnose_compute_bound_saturated():
    # High GPU utilization (96%), high arithmetic intensity, clean IO -> COMPUTE_BOUND
    diag = CausalBottleneckDetector.diagnose(
        gpu_utilization_pct=96.0,
        step_time_ms=45.0,
        data_loading_time_ms=0.5,
        gpu_memory_used_mb=10000.0,
        total_vram_mb=16384.0,
        arithmetic_intensity=80.0,
    )
    assert diag.primary_bottleneck == BottleneckType.COMPUTE_BOUND
    assert "RULE_MIXED_PRECISION_FP16" in diag.recommended_rules
    assert "RULE_BATCH_SCALING" in diag.recommended_rules


def test_bottleneck_diagnosis_to_dict():
    diag = BottleneckDiagnosis(
        primary_bottleneck=BottleneckType.INPUT_BOUND,
        secondary_bottleneck=BottleneckType.MEMORY_BANDWIDTH_BOUND,
        confidence_score=0.92,
        evidence_metrics={"io_fraction": 0.22},
        recommended_rules=["RULE_DATALOADER_WORKERS"],
        rationale="DataLoader stalls detected",
    )
    d = diag.to_dict()
    assert d["primary_bottleneck"] == "INPUT_BOUND"
    assert d["secondary_bottleneck"] == "MEMORY_BANDWIDTH_BOUND"
    assert d["confidence_score"] == 0.92
    assert d["recommended_rules"] == ["RULE_DATALOADER_WORKERS"]
