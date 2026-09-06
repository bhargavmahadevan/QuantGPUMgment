import pytest
from ghost_layer.roi.financial_impact import (
    FinancialImpactCalculator,
    FinancialImpact,
    GPU_HOURLY_RATES_USD,
)
from ghost_layer.telemetry.watcher import TelemetrySummary
from ghost_layer.decision.engine import DecisionEngine, Recommendation
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase


def test_gpu_hourly_rates_resolution():
    calc = FinancialImpactCalculator()
    assert calc.get_hourly_rate("H100") == 3.50
    assert calc.get_hourly_rate("NVIDIA H100 SXM5 80GB") == 3.50
    assert calc.get_hourly_rate("A100") == 2.00
    assert calc.get_hourly_rate("NVIDIA RTX A2000") == 0.35
    assert calc.get_hourly_rate("Unknown GPU") == 3.50  # default fallback


def test_financial_impact_calculation_8x_h100():
    calc = FinancialImpactCalculator()
    impact = calc.calculate_impact(
        hardware_type="H100",
        num_gpus=8,
        speedup_potential_pct=25.0,
        monthly_active_hours=720.0,
        audit_fee_usd=2500.0,
    )
    assert isinstance(impact, FinancialImpact)
    # 8 GPUs * $3.50 = $28.00/hr cluster cost
    assert impact.cluster_hourly_rate_usd == 28.00
    # Monthly baseline cost = 28 * 720 = $20,160.00
    assert impact.monthly_baseline_cost_usd == 20160.00
    # 25% speedup => time reduction savings fraction
    assert impact.modeled_monthly_savings_usd == 5040.00
    assert impact.modeled_annual_savings_usd == 60480.00
    # Audit payback days: 2500 / (5040 / 30) = 14.88 days
    assert 14.0 <= impact.audit_payback_days <= 16.0
    # 1-Year Audit ROI Multiple: 60480 / 2500 = 24.19x
    assert impact.audit_roi_multiple >= 20.0


def test_decision_engine_enriches_recommendations_with_financial_impact():
    kb = SharedKnowledgeBase()
    engine = DecisionEngine(knowledge_base=kb)
    
    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=25.0,
        avg_gpu_utilization_pct=40.0,
        peak_gpu_memory_mb=6000.0,
        gpu_memory_total_mb=16384.0,
        avg_step_time_ms=250.0,
        avg_dataloader_stall_pct=26.0,  # 26% stall ratio -> triggers DataLoader stall
        mixed_precision="fp32",  # triggers precision recommendation
        gradient_checkpointing=False,
        flash_attention=False,
        num_workers=0,
        pin_memory=False,
    )
    
    recs = engine.evaluate(
        summary,
        model_type="transformer",
        hardware_type="H100",
        num_gpus=8,
    )
    
    assert len(recs) > 0
    # Check that high-impact recommendations have financial impact populated
    has_financial_attribution = False
    for r in recs:
        if r.financial_impact is not None:
            has_financial_attribution = True
            assert "modeled_monthly_waste_usd" in r.financial_impact
            assert "audit_payback_days" in r.financial_impact
            assert r.financial_impact["cluster_hourly_rate_usd"] == 28.00
            assert r.financial_impact["modeled_monthly_waste_usd"] > 0
    assert has_financial_attribution, "At least one recommendation should have financial impact attributed."


def test_slack_notifier_formats_financial_impact():
    from ghost_layer.integrations.slack_notifier import SlackNotifier
    notifier = SlackNotifier(webhook_url="https://mock-hooks.slack.com/services/test")
    rec = Recommendation(
        rule_id="RULE_MIXED_PRECISION",
        title="Enable Automatic Mixed Precision",
        impact_level="HIGH",
        speedup_estimate_label="~45%",
        description="Tensor Cores idle under FP32.",
        actionable_code_snippet="torch.cuda.amp.autocast()",
        safe_to_auto_apply=True,
        financial_impact={
            "modeled_monthly_waste_usd": 5040.00,
            "audit_payback_days": 14.9,
            "audit_roi_multiple": 24.19,
        }
    )
    # Intercept _send to inspect blocks
    captured = {}
    def mock_send(text, blocks):
        captured["text"] = text
        captured["blocks"] = blocks
        return True
    notifier._send = mock_send
    assert notifier.notify_recommendation(rec) is True
    assert any("Modeled Waste" in str(b) for b in captured["blocks"])
    assert any("$5,040.00" in str(b) for b in captured["blocks"])
