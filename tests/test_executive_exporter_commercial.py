import pytest
import os
import tempfile
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.reporting.executive_exporter import ExecutiveAuditExporter


def test_executive_exporter_commercial_pricing_and_payback():
    hook = GhostWatcherHook(gpu_memory_mb=16384.0, target_hardware="NVIDIA H100 SXM5")
    # Record a few baseline steps
    for step in range(1, 11):
        hook.watcher.record_step(
            step=step,
            gpu_utilization_pct=45.0,
            gpu_memory_used_mb=8000.0,
            step_time_ms=100.0,
            data_loading_time_ms=25.0,  # 25% stall
            mixed_precision="fp32",
        )
    
    exporter = ExecutiveAuditExporter(hook)
    summary_data = exporter.generate_metrics_summary()
    
    assert "preflight_audit_fee_usd" in summary_data
    assert summary_data["preflight_audit_fee_usd"] == 2500.0
    assert "audit_payback_days_8x" in summary_data
    assert "annual_recoverable_savings_8x" in summary_data
    assert "commercial_tiers" in summary_data
    assert len(summary_data["commercial_tiers"]) == 3
    
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "audit_exec.html")
        md_path = os.path.join(tmpdir, "audit_exec.md")
        
        exporter.export_html(html_path)
        exporter.export_markdown(md_path)
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            assert "Pre-Flight Diagnostic Audit" in html_content
            assert "$2,500" in html_content
            assert "Enterprise Compute Assurance" in html_content
            assert "Schedule 48-Hour Audit" in html_content

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "$2,500" in md_content
            assert "Team Platform SaaS" in md_content
