"""
GhostLayer Integrations, Callbacks, Exporters, & Graph DAG Test Suite.
Combines tests for:
- Monitoring integrations (SlackNotifier, WebhookBridge, WandbExporter, PrometheusExporter)
- ExecutiveAuditExporter (JSON, Markdown, HTML reporting)
- Trainer callbacks & context manager decorators (GhostTrainerCallback, GhostLightningCallback, ghost_watch)
- Execution Graph Builder & DAG Optimizer (Transformer, MoE, SSM Mamba DAGs, topological sorting, kernel fusion)
"""

import os
import json
import pytest

# GhostLayer Integrations & Reporting
from ghost_layer.integrations import (
    SlackNotifier,
    WebhookBridge,
    WandbExporter,
    PrometheusExporter,
)
from ghost_layer.decision.engine import DecisionEngine, Recommendation
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.callback import GhostTrainerCallback, GhostLightningCallback, ghost_watch
from ghost_layer.reporting.executive_exporter import ExecutiveAuditExporter
from ghost_layer.telemetry.watcher import TelemetryWatcher
from ghost_layer.graphify.graph_builder import ExecutionGraphBuilder
from ghost_layer.graphify.analyzer import GraphAnalyzer, GraphFusionEngine
from ghost_layer.graphify.optimizer import GraphOptimizer


# ==============================================================================
# 1. Monitoring Integrations (Prometheus & Webhook) Tests
# ==============================================================================

def test_prometheus_metric_store_formatting():
    exporter = PrometheusExporter(port=9999)
    exporter.record_step(step_time_ms=22.5, gpu_util_pct=90.0, memory_mb=12000.0, loss=1.5)
    exporter.record_recommendation("RULE_MIXED_PRECISION")
    exporter.record_recommendation("RULE_MUON_OPTIMIZER")

    formatted = exporter.store.format_metrics()
    assert "ghostlayer_step_time_ms 22.5" in formatted
    assert "ghostlayer_gpu_utilization_pct 90.0" in formatted
    assert 'ghostlayer_recommendations_total{rule_id="RULE_MIXED_PRECISION"} 1.0' in formatted
    assert 'ghostlayer_recommendations_total{rule_id="RULE_MUON_OPTIMIZER"} 1.0' in formatted


def test_webhook_bridge_payload_creation():
    bridge = WebhookBridge(endpoint_url="http://mock-endpoint:8080/events")
    rec = Recommendation(
        rule_id="RULE_MUON_OPTIMIZER",
        title="Switch to Muon",
        impact_level="HIGH",
        speedup_estimate_label="35%",
        description="Matrix polar updates",
        actionable_code_snippet="Muon()",
        safe_to_auto_apply=True,
    )
    assert bridge.endpoint_url == "http://mock-endpoint:8080/events"
    assert bridge.max_retries == 3


# ==============================================================================
# 2. Executive Exporter & Trainer Callbacks Tests
# ==============================================================================

def test_executive_audit_exporter(tmp_path):
    hook = GhostWatcherHook(gpu_memory_mb=16384.0, gpu_cost_per_hour=3.50)
    with hook:
        for i in range(10):
            hook.on_step_begin()
            hook.on_step_end(
                gpu_util_pct=75.0,
                gpu_mem_used_mb=10000.0,
                loss=2.5 - (i * 0.05),
                dataloader_time_ms=5.0,
                mixed_precision="fp32",
            )

    exporter = ExecutiveAuditExporter(hook)

    # Test JSON export
    json_path = str(tmp_path / "report.json")
    exported_json = exporter.export_json(json_path)
    assert os.path.exists(exported_json)
    with open(exported_json, "r") as f:
        data = json.load(f)
    assert data["total_steps_audited"] == 10
    assert "estimated_monthly_waste_per_gpu" in data
    assert data["vram_headroom_pct"] > 0

    # Test Markdown export
    md_path = str(tmp_path / "report.md")
    exported_md = exporter.export_markdown(md_path)
    assert os.path.exists(exported_md)
    with open(exported_md, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "Quant GhostLayer: Executive GPU Efficiency Audit Report" in md_text

    # Test HTML export
    html_path = str(tmp_path / "report.html")
    exported_html = exporter.export_html(html_path)
    assert os.path.exists(exported_html)
    with open(exported_html, "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "Quant GhostLayer™ Executive Diagnostic" in html_text
    assert "Prioritized Recommendations" in html_text


def test_ghost_watch_context_manager(tmp_path):
    report_file = str(tmp_path / "gw_report.html")
    with ghost_watch(gpu_memory_mb=16384.0, output_report_path=report_file) as gw:
        for step in range(5):
            gw.step(loss=1.5, gpu_util_pct=80.0, dataloader_time_ms=2.0)

    assert os.path.exists(report_file)


def test_ghost_watch_decorator():
    executed = False

    @ghost_watch(gpu_memory_mb=8192.0, output_report_path=None)
    def dummy_train_epoch():
        nonlocal executed
        executed = True
        return 42

    result = dummy_train_epoch()
    assert executed is True
    assert result == 42


def test_ghost_trainer_callback(tmp_path):
    report_file = str(tmp_path / "hf_report.html")
    cb = GhostTrainerCallback(
        gpu_memory_mb=16384.0,
        output_report_path=report_file,
        auto_generate_report=True,
    )

    cb.on_train_begin()
    for _ in range(3):
        cb.on_step_begin()
        cb.on_step_end()
    cb.on_train_end()

    assert os.path.exists(report_file)


def test_ghost_lightning_callback(tmp_path):
    report_file = str(tmp_path / "pl_report.html")
    cb = GhostLightningCallback(
        gpu_memory_mb=16384.0,
        output_report_path=report_file,
    )

    class DummyModule:
        pass
    class DummyTrainer:
        pass

    cb.on_train_start(DummyTrainer(), DummyModule())
    for i in range(3):
        cb.on_train_batch_start(DummyTrainer(), DummyModule(), None, i)
        cb.on_train_batch_end(DummyTrainer(), DummyModule(), {"loss": 1.23}, None, i)
    cb.on_train_end(DummyTrainer(), DummyModule())

    assert os.path.exists(report_file)


# ==============================================================================
# 3. Execution Graph Builder & DAG Optimizer Tests
# ==============================================================================

def test_graph_builder_transformer_dag():
    num_layers = 4
    builder = ExecutionGraphBuilder("Llama-7B")
    builder.build_transformer_dag(num_layers=num_layers, hidden_dim=4096, sequence_length=2048)

    expected_nodes = 1 + (num_layers * 3) + 1
    expected_edges = (num_layers * 3) + 1

    assert len(builder.nodes) == expected_nodes
    assert len(builder.edges) == expected_edges
    assert "embedding" in builder.nodes
    assert "head" in builder.nodes

    for i in range(num_layers):
        assert f"layer_{i}_attention" in builder.nodes
        assert f"layer_{i}_mlp" in builder.nodes
        assert f"layer_{i}_norm" in builder.nodes


def test_graph_analyzer():
    num_layers = 2
    builder = ExecutionGraphBuilder("GPT-Small")
    builder.build_transformer_dag(num_layers=num_layers, hidden_dim=2048, sequence_length=1024)

    analyzer = GraphAnalyzer(builder)
    summary = analyzer.analyze()

    expected_nodes = 1 + (num_layers * 3) + 1
    expected_edges = (num_layers * 3) + 1

    assert summary.total_nodes == expected_nodes
    assert summary.total_edges == expected_edges
    assert summary.critical_path_length_ms > 0.0
    assert summary.max_memory_node_mb == 1200.0
    assert summary.recommended_torch_compile is True


def test_graphify_decision_engine_integration():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    watcher.record_step(1, 50.0, 8000.0, 150.0, mixed_precision="bf16")
    watcher.stop()

    engine = DecisionEngine()
    recs = engine.evaluate(watcher.get_summary(), model_type="transformer")

    rule_ids = [r.rule_id for r in recs]
    assert "RULE_GRAPHIFY_TORCH_COMPILE" in rule_ids


def test_graphify_moe_and_ssm_dags():
    num_layers = 4
    num_experts = 8
    moe_builder = ExecutionGraphBuilder("Mixtral-8x7B")
    moe_builder.build_moe_dag(num_layers=num_layers, num_experts=num_experts, top_k=2)

    assert len(moe_builder.nodes) == 34
    assert any("moe_router" in nid for nid in moe_builder.nodes)

    ssm_builder = ExecutionGraphBuilder("Mamba-2.8B")
    ssm_builder.build_ssm_dag(num_layers=num_layers)
    assert len(ssm_builder.nodes) == 22
    assert any("selective_scan" in nid for nid in ssm_builder.nodes)


def test_graphify_topological_sort_and_validation():
    builder = ExecutionGraphBuilder("TestDAG")
    builder.add_node("n1", "Input", "MODULE", execution_time_ms=1.0)
    builder.add_node("n2", "Op1", "OPERATOR", execution_time_ms=2.0)
    builder.add_node("n3", "Op2", "OPERATOR", execution_time_ms=3.0)
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", "n3")

    topo = builder.topological_sort()
    assert topo == ["n1", "n2", "n3"]
    assert builder.validate_dag() is True


def test_graphify_kernel_fusion_engine():
    num_layers = 2
    builder = ExecutionGraphBuilder("FusionTest")
    builder.build_transformer_dag(num_layers=num_layers)
    clusters = GraphFusionEngine.identify_fusible_clusters(builder)

    assert len(clusters) == num_layers

    analyzer = GraphAnalyzer(builder)
    summary = analyzer.analyze()
    assert summary.fusible_clusters_count == len(clusters)
    assert summary.estimated_fusion_speedup_pct > 0.0


def test_graphify_optimizer_passes():
    builder = ExecutionGraphBuilder("LLaMA-Optimizer-Test")
    builder.build_transformer_dag(num_layers=4)

    baseline_analyzer = GraphAnalyzer(builder)
    base_summary = baseline_analyzer.analyze()

    optimizer = GraphOptimizer(builder)
    opt_builder, report = optimizer.optimize()

    opt_analyzer = GraphAnalyzer(opt_builder)
    opt_summary = opt_analyzer.analyze()

    expected_speedup_pct = round(((base_summary.critical_path_length_ms - opt_summary.critical_path_length_ms) / base_summary.critical_path_length_ms) * 100.0, 1)
    expected_mem_savings_pct = round(((base_summary.max_memory_node_mb - opt_summary.max_memory_node_mb) / base_summary.max_memory_node_mb) * 100.0, 1)

    assert report.speedup_pct == expected_speedup_pct
    assert report.memory_savings_pct == expected_mem_savings_pct
    assert len(report.applied_passes) >= 4
    assert any("MatrixFreeEpiloguePass" in p for p in report.applied_passes)
    assert opt_builder.model_name == "LLaMA-Optimizer-Test_Optimized"


def test_graphify_visual_exporters():
    builder = ExecutionGraphBuilder("VisualTest")
    builder.build_transformer_dag(num_layers=2)

    mermaid_str = builder.to_mermaid()
    assert "flowchart TD" in mermaid_str
    assert "VisualTest" in mermaid_str

    dot_str = builder.to_dot()
    assert "digraph" in dot_str

    html_str = builder.to_html_interactive()
    assert "<!DOCTYPE html>" in html_str
    assert "VisualTest" in html_str
