"""
Unit tests validating EnvironmentFingerprint hashing/similarity,
ContaminationDetector interference alerts, and DecisionEngine calibration hardware scoping.
"""

import pytest
from ghost_layer.telemetry.environment_fingerprint import (
    EnvironmentFingerprint,
    HardwareSpec,
    SoftwareSpec,
    RuntimeSpec,
    WorkloadSpec,
)
from ghost_layer.telemetry.contamination import (
    ContaminationDetector,
    ContaminationReport,
)
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.telemetry.watcher import TelemetrySummary


def _build_sample_fingerprint(device_name="NVIDIA RTX A2000", is_cuda=True):
    return EnvironmentFingerprint(
        hardware=HardwareSpec(device_name=device_name, is_cuda=is_cuda, vram_total_mb=8192.0),
        software=SoftwareSpec(pytorch_version="2.5.1", cuda_runtime_version="12.4", python_version="3.11", os_platform="Windows"),
        runtime=RuntimeSpec(allocator_conf="expandable_segments:True", thread_count=8, amp_enabled=True, pin_memory=True),
        workload=WorkloadSpec(model_family="llama", parameter_count=52000000, sequence_length=512, batch_size=4),
    )


def test_environment_fingerprint_hashing_and_similarity():
    """Verify deterministic hashes and similarity between environments."""
    fp1 = _build_sample_fingerprint("NVIDIA RTX A2000", is_cuda=True)
    fp2 = _build_sample_fingerprint("NVIDIA RTX A2000", is_cuda=True)
    fp_cpu = _build_sample_fingerprint("CPU", is_cuda=False)

    # Identical environments should have identical composite hash and 1.0 similarity
    assert fp1.composite_hash == fp2.composite_hash
    assert fp1.calculate_similarity(fp2) == 1.0

    # Diverging environment should have lower similarity
    sim_cpu = fp1.calculate_similarity(fp_cpu)
    assert sim_cpu < 0.70


def test_contamination_detector():
    """Verify ContaminationDetector flags thermal throttling and clock degradation."""
    detector = ContaminationDetector(max_allowed_clock_drop_pct=15.0)

    # Clean run
    clean_rep = detector.evaluate_trial(
        is_thermal_throttled=False,
        baseline_clock_mhz=1500,
        candidate_clock_mhz=1480,
    )
    assert clean_rep.is_contaminated is False
    assert clean_rep.status == "VALID"

    # Thermal throttled run
    thermal_rep = detector.evaluate_trial(is_thermal_throttled=True)
    assert thermal_rep.is_contaminated is True
    assert thermal_rep.status == "CONTAMINATED"
    assert thermal_rep.thermal_throttling_detected is True

    # Severe clock drop
    clock_rep = detector.evaluate_trial(
        is_thermal_throttled=False,
        baseline_clock_mhz=1500,
        candidate_clock_mhz=1100,  # > 26% drop
    )
    assert clock_rep.is_contaminated is True
    assert clock_rep.frequency_drop_detected is True


def test_calibration_hardware_scoping_blocks_cpu_transfer_to_gpu():
    """Verify CPU calibration reports cannot be transferred to GPU target architectures."""
    engine = DecisionEngine(target_hardware="NVIDIA RTX A2000")

    class DummyCalib:
        hardware_name = "CPU"
        precision_speedup_pct = 2.95

    compat = engine.is_calibration_hardware_compatible(DummyCalib(), engine.target_hardware)
    assert compat is False

    # Summary requesting FP32 -> AMP
    summary = TelemetrySummary(
        total_steps=10,
        total_duration_sec=1.0,
        avg_gpu_utilization_pct=50.0,
        peak_gpu_memory_mb=4000.0,
        gpu_memory_total_mb=8000.0,
        avg_step_time_ms=100.0,
        avg_dataloader_stall_pct=0.0,
        mixed_precision="fp32",
        gradient_checkpointing=False,
        flash_attention=False,
        num_workers=4,
        pin_memory=True,
    )

    recs = engine.evaluate(summary=summary, calibration_report=DummyCalib())
    mixed_rec = next((r for r in recs if r.rule_id == "RULE_MIXED_PRECISION"), None)
    assert mixed_rec is not None
    # Must NOT claim Measured on CPU for a GPU target
    assert "Measured on CPU" not in mixed_rec.speedup_estimate_label
    assert mixed_rec.has_verified_runs is False
