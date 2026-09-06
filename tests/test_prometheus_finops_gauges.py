import pytest
from ghost_layer.integrations.prometheus_exporter import PrometheusExporter


def test_prometheus_finops_gauges_recorded():
    exporter = PrometheusExporter(port=0)  # do not bind port in test
    
    exporter.record_finops(
        cluster_hourly_spend_usd=28.00,
        wasted_hourly_spend_usd=7.00,
        recovered_monthly_savings_usd=5040.00,
        audit_roi_multiplier=24.19,
        tier_level=2,
    )
    
    metrics_text = exporter.store.format_metrics()
    assert "ghostlayer_cluster_hourly_spend_usd 28.0" in metrics_text
    assert "ghostlayer_wasted_hourly_spend_usd 7.0" in metrics_text
    assert "ghostlayer_recovered_monthly_savings_usd 5040.0" in metrics_text
    assert "ghostlayer_audit_roi_multiplier 24.19" in metrics_text
    assert "ghostlayer_commercial_tier_level 2.0" in metrics_text
