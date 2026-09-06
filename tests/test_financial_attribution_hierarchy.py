import pytest
from ghost_layer.savings import SavingsVerifier, BaselineMeasurement
from ghost_layer.roi.calculator import ROICalculator


def test_verified_savings_multi_tier_economics():
    verifier = SavingsVerifier(gpu_cost_per_hour=4.00)
    baseline_times = [100.0, 102.0, 98.0, 101.0, 99.0]
    opt_times = [50.0, 51.0, 49.0, 50.5, 49.5]

    base = verifier.measure(baseline_times, label="baseline")
    opt = verifier.measure(opt_times, label="optimized")

    report = verifier.compare(base, opt)

    assert report.is_verified is True
    assert report.speedup_pct > 45.0

    # Step economics
    assert report.cost_per_step_baseline_usd > 0.0
    assert report.cost_per_step_optimized_usd < report.cost_per_step_baseline_usd

    # 1M tokens economics
    assert report.cost_per_million_tokens_baseline_usd > report.cost_per_million_tokens_optimized_usd

    # Realized savings in window
    assert report.realized_measured_savings_usd > 0.0

    # Fleet annualized projected savings
    annual_savings = report.calculate_projected_annual_savings(num_gpus=8, annual_operating_hours=8760.0)
    assert annual_savings > 10000.0


def test_roi_calculator_unit_and_fleet_economics():
    calculator = ROICalculator(gpu_cost_per_hour=3.50, audit_fee_usd=2500.0)

    report = calculator.calculate(
        baseline_step_time_ms=30.0,
        optimized_step_time_ms=10.0,
        total_training_steps=50_000_000,
        num_gpus=8,
    )

    assert report.time_reduction_pct > 60.0
    assert report.cost_per_step_baseline_usd > 0.0
    assert report.cost_per_step_optimized_usd < report.cost_per_step_baseline_usd
    assert report.cost_per_million_tokens_baseline_usd > report.cost_per_million_tokens_optimized_usd
    assert report.realized_measured_savings_usd > 0.0
    assert report.projected_annual_savings_usd > 0.0
    assert report.audit_roi_multiple > 1.0
