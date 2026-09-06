"""
Tests for SamplePowerAnalyzer & Pre-Trial Sample Sizing.
"""

import pytest
from ghost_layer.verification.power_analysis import (
    PowerAnalysisSpec,
    PowerAnalysisResult,
    SamplePowerAnalyzer,
)


def test_calculate_required_sample_size_standard():
    spec = PowerAnalysisSpec(
        mde_relative=0.05,  # 5% MDE
        baseline_mean=100.0,
        baseline_std=2.0,  # 2% relative noise
        alpha=0.05,
        power=0.80,
    )
    n_req = SamplePowerAnalyzer.calculate_required_sample_size(spec)
    # Low noise relative to MDE -> small sample required (e.g. 5-10 steps)
    assert n_req >= 5
    assert n_req <= 20


def test_calculate_required_sample_size_high_noise():
    spec = PowerAnalysisSpec(
        mde_relative=0.02,  # 2% small MDE
        baseline_mean=100.0,
        baseline_std=10.0,  # 10% high noise
        alpha=0.05,
        power=0.90,
    )
    n_req = SamplePowerAnalyzer.calculate_required_sample_size(spec)
    # High noise + small MDE + 90% power -> requires large sample size
    assert n_req > 50


def test_evaluate_sample_adequacy_adequate():
    spec = PowerAnalysisSpec(mde_relative=0.05, baseline_mean=100.0, baseline_std=2.0)
    # 20 observations > required ~5-8
    res = SamplePowerAnalyzer.evaluate_sample_adequacy(
        observed_baseline_count=20,
        observed_candidate_count=20,
        spec=spec,
    )
    assert res.is_sufficient_sample
    assert "meets or exceeds" in res.recommendation


def test_evaluate_sample_adequacy_insufficient_power():
    spec = PowerAnalysisSpec(mde_relative=0.02, baseline_mean=100.0, baseline_std=10.0)
    # Only 4 observations when N_req > 50 -> Insufficient power
    res = SamplePowerAnalyzer.evaluate_sample_adequacy(
        observed_baseline_count=4,
        observed_candidate_count=4,
        spec=spec,
    )
    assert not res.is_sufficient_sample
    assert "INSUFFICIENT_POWER" in res.recommendation
