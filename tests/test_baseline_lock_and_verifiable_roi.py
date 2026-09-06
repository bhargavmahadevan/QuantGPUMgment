"""
Unit tests validating Cryptographic BaselineLock sealing, parameter drift detection,
and ROICalculator CALCULATED vs VERIFIED ROI status gating.
"""

import pytest
from ghost_layer.experiments.baseline_lock import BaselineLock
from ghost_layer.roi.calculator import ROICalculator, ROIAuditReport


def _create_sample_lock():
    return BaselineLock.create(
        lock_id="LOCK-A2000-LLM-001",
        model_name="Transformer-52M",
        total_parameters=52200000,
        batch_size=4,
        sequence_length=512,
        token_count=100000000,
        hardware_name="NVIDIA RTX A2000",
        gpu_hourly_cost_usd=3.50,
    )


def test_baseline_lock_creation_and_validation():
    """Verify BaselineLock seals state with SHA-256 and validates candidate compatibility."""
    lock = _create_sample_lock()
    assert lock.lock_hash is not None and len(lock.lock_hash) == 64
    assert lock.lock_id == "LOCK-A2000-LLM-001"

    # Exact matching candidate
    cand_valid = {
        "model_name": "Transformer-52M",
        "batch_size": 4,
        "sequence_length": 512,
        "hardware_name": "NVIDIA RTX A2000",
    }
    is_ok, status, violations = lock.validate_candidate(cand_valid)
    assert is_ok is True
    assert status == "COMPARISON_VALID"
    assert len(violations) == 0

    # Divergent candidate (e.g. batch size changed)
    cand_drift = {
        "model_name": "Transformer-52M",
        "batch_size": 8,  # Moving baseline!
        "sequence_length": 512,
        "hardware_name": "NVIDIA RTX A2000",
    }
    is_ok, status, violations = lock.validate_candidate(cand_drift)
    assert is_ok is False
    assert status == "COMPARISON_INVALID"
    assert any("Batch size mismatch" in v for v in violations)


def test_verifiable_roi_status_flow():
    """Verify ROICalculator outputs CALCULATED vs VERIFIED vs INVALID ROI statuses."""
    calc = ROICalculator(gpu_cost_per_hour=3.50)
    lock = _create_sample_lock()

    # 1. Without lock: status = CALCULATED
    rep_calc = calc.calculate(
        baseline_step_time_ms=100.0,
        optimized_step_time_ms=60.0,
        total_training_steps=10000,
        num_gpus=8,
    )
    assert rep_calc.roi_status == "CALCULATED"
    assert rep_calc.is_verified is False
    assert "without cryptographic BaselineLock" in rep_calc.provenance_note

    # 2. With lock, p < 0.05, and 95% CI: status = VERIFIED
    rep_ver = calc.calculate(
        baseline_step_time_ms=100.0,
        optimized_step_time_ms=60.0,
        total_training_steps=10000,
        num_gpus=8,
        baseline_lock=lock,
        welch_p_value=0.0012,
        speedup_ci_95=(38.5, 41.5),
        candidate_metadata={
            "model_name": "Transformer-52M",
            "batch_size": 4,
            "sequence_length": 512,
            "hardware_name": "NVIDIA RTX A2000",
        },
    )
    assert rep_ver.roi_status == "VERIFIED"
    assert rep_ver.is_verified is True
    assert rep_ver.baseline_lock_id == "LOCK-A2000-LLM-001"
    assert "Empirically verified" in rep_ver.provenance_note

    # 3. With lock, but moving baseline: status = INVALID
    rep_inv = calc.calculate(
        baseline_step_time_ms=100.0,
        optimized_step_time_ms=60.0,
        total_training_steps=10000,
        num_gpus=8,
        baseline_lock=lock,
        welch_p_value=0.0012,
        speedup_ci_95=(38.5, 41.5),
        candidate_metadata={
            "model_name": "DifferentModel-13B",  # Incompatible!
            "batch_size": 4,
            "sequence_length": 512,
            "hardware_name": "NVIDIA RTX A2000",
        },
    )
    assert rep_inv.roi_status == "INVALID"
    assert rep_inv.is_verified is False
    assert "Baseline lock violated" in rep_inv.provenance_note
