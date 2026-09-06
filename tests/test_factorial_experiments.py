"""
Unit tests validating 2^k Factorial Experimentation & Systems Interaction Analysis.
"""

import pytest
from ghost_layer.experiments.factorial import FactorialExperimentRunner, FactorialAnalysisReport


def test_factorial_runner_analysis():
    """Verify 2^3 factorial experiment with 3 factors (AMP, SDPA, PINNED)."""
    factors = ["AMP", "PINNED", "SDPA"]

    # Synthetic latency surface: baseline 100ms
    # AMP reduces to 60ms (40% speedup)
    # PINNED reduces to 90ms (10% speedup)
    # SDPA reduces to 80ms (20% speedup)
    # AMP+PINNED has sub-additive overlap: 55ms (45% speedup vs 50% linear)
    # AMP+SDPA has super-additive synergy: 42ms (58% speedup vs 60% linear)
    # Full bundle: 38ms (62% speedup)
    latency_map = {
        (): 100.0,
        ("AMP",): 60.0,
        ("PINNED",): 90.0,
        ("SDPA",): 80.0,
        ("AMP", "PINNED"): 55.0,
        ("AMP", "SDPA"): 42.0,
        ("PINNED", "SDPA"): 72.0,
        ("AMP", "PINNED", "SDPA"): 38.0,
    }

    report = FactorialExperimentRunner.analyze_results(factors, latency_map)

    assert isinstance(report, FactorialAnalysisReport)
    assert report.total_treatments == 8
    assert len(report.treatments) == 8

    # Verify main effects
    assert report.main_effects_pct["AMP"] > 30.0
    assert report.main_effects_pct["PINNED"] > 0.0

    # Verify optimal treatment is the full bundle
    assert report.optimal_treatment_id == "AMP+PINNED+SDPA"
    assert report.optimal_speedup_pct == 62.0
