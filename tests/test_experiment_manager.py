"""
Unit tests validating ExperimentManager, counterbalanced ABAB trial design,
autocorrelation-aware block bootstrap statistics, and ablation decomposition.
"""

import math
import pytest

from ghost_layer.experiments.schema import (
    ExperimentRecord,
    WorkloadProfile,
    InterventionSpec,
)
from ghost_layer.experiments.statistics import (
    block_bootstrap_speedup_ci,
    compute_cohens_d,
    decompose_ablation_chain,
)
from ghost_layer.experiments.manager import ExperimentManager


def test_block_bootstrap_speedup_ci():
    """Verify block bootstrap produces empirical confidence intervals over step times."""
    # Autocorrelated baseline steps around 100ms
    base = [100.0 + 5.0 * math.sin(i) for i in range(50)]
    # Autocorrelated candidate steps around 80ms (~20% speedup)
    cand = [80.0 + 4.0 * math.sin(i) for i in range(50)]

    point_sp, (ci_low, ci_high), p_val = block_bootstrap_speedup_ci(
        base, cand, block_size=5, num_resamples=500, seed=42
    )
    assert 15.0 <= point_sp <= 25.0
    assert ci_low < point_sp < ci_high
    assert p_val < 0.01  # Highly significant improvement


def test_decompose_ablation_chain():
    """Verify marginal attribution decomposition for A -> B -> C -> D."""
    stages = {
        "A_Baseline": 100.0,
        "B_PinMemory": 90.0,   # 10% marginal speedup
        "C_SDPA": 72.0,        # 20% marginal speedup
        "D_AMP": 54.0,         # 25% marginal speedup
    }
    contributions = decompose_ablation_chain(stages)
    assert contributions["B_PinMemory"] == 10.0
    assert contributions["C_SDPA"] == 20.0
    assert contributions["D_AMP"] == 25.0


def test_experiment_manager_controlled_trial_abab():
    """Verify ExperimentManager executes counterbalanced ABAB trial and generates canonical ExperimentRecord."""
    manager = ExperimentManager(warmup_steps=2, block_size=5, gpu_cost_per_hour=4.00)

    # Synthetic step functions
    step_state = {"count": 0}

    def baseline_fn():
        step_state["count"] += 1
        return 100.0 + (step_state["count"] % 3)

    def candidate_fn():
        step_state["count"] += 1
        return 75.0 + (step_state["count"] % 3)

    intervention = InterventionSpec(
        rule_id="RULE_MIXED_PRECISION",
        intervention_tier="TIER_A_PRODUCTION_SAFE",
        target_component="ModelPrecision",
        diff_before={"precision": "fp32"},
        diff_after={"precision": "bf16"},
    )
    workload = WorkloadProfile(
        model_family="gpt-neox",
        model_name_or_path="EleutherAI/pythia-70m",
        batch_size=8,
        sequence_length=1024,
    )

    base_losses = [2.5 - 0.01 * i for i in range(40)]
    cand_losses = [2.5 - 0.011 * i for i in range(40)]

    record = manager.run_controlled_trial(
        experiment_id="exp_test_001",
        baseline_step_fn=baseline_fn,
        candidate_step_fn=candidate_fn,
        intervention=intervention,
        workload=workload,
        num_blocks=4,
        steps_per_block=10,
        baseline_losses=base_losses,
        candidate_losses=cand_losses,
    )

    assert isinstance(record, ExperimentRecord)
    assert record.experiment_id == "exp_test_001"
    assert record.trial_design == "COUNTERBALANCED_ABAB"
    assert record.warmup_steps_discarded == 2
    # 4 blocks * (10 - 2 warmup) = 32 clean samples total -> 16 per arm
    assert record.baseline_distribution.sample_count == 16
    assert record.candidate_distribution.sample_count == 16
    assert record.statistics.speedup_pct > 20.0
    assert record.statistics.is_statistically_significant is True
    assert record.quality_safety.is_verified_safe is True
    assert record.economics.provenance_type == "EMPIRICAL"
    assert len(record.record_hash) == 64


def test_quality_safety_rejection_on_nan_and_loss_spike():
    """Verify ExperimentManager safety gating halts on NaN or excessive loss divergence."""
    manager = ExperimentManager()

    # Trajectory with NaN in candidate
    base_losses = [2.0, 1.9, 1.8]
    nan_losses = [2.0, float("nan"), 1.8]

    safety_nan = manager._evaluate_quality_safety(base_losses, nan_losses)
    assert safety_nan.has_nan_or_inf is True
    assert safety_nan.is_verified_safe is False
    assert "NaN or Inf detected" in safety_nan.safety_gate_reasons[0]

    # Trajectory with large loss shift (> 0.10)
    spike_losses = [2.0, 2.8, 2.7]
    safety_spike = manager._evaluate_quality_safety(base_losses, spike_losses)
    assert safety_spike.is_verified_safe is False
    assert "threshold exceeded" in safety_spike.safety_gate_reasons[0]
