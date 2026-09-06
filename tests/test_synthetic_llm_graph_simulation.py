"""
Synthetic Graph & Operator Simulation Test Suite.

NOTE ON EMPIRICAL BOUNDARIES & HONESTY:
--------------------------------------
This file performs pure analytical verification of ExecutionGraphBuilder,
GraphAnalyzer, GraphOptimizer, and GhostActiveCompensator.

The operator execution times (e.g. 18.5ms MLA attention, 28.0ms MoE routed experts)
and cluster dimensions (16,384 H100s) used herein are SYNTHETIC MODELING FIXTURES
designed to test topological critical-path algorithms and compensator numerical
stability. They are NOT physical hardware benchmarks, NOT empirical measurements,
and do NOT touch the Hugging Face Hub or live models.
"""

import pytest
import numpy as np

from ghost_layer.graphify.graph_builder import ExecutionGraphBuilder
from ghost_layer.graphify.analyzer import GraphAnalyzer
from ghost_layer.graphify.optimizer import GraphOptimizer
from ghost_layer.telemetry.watcher import TelemetryWatcher
from ghost_layer.telemetry.containment import VarianceContainmentEngine
from ghost_layer.telemetry.compensator import GhostActiveCompensator
from ghost_layer.roi.calculator import ROICalculator


def test_llama_405b_synthetic_dag_construction():
    """Verify execution DAG builder for modeled 405B parameter transformer topology (synthetic fixture)."""
    builder = ExecutionGraphBuilder(model_name="Synthetic-Llama-3.1-405B")
    builder.build_transformer_dag(num_layers=16, hidden_dim=16384, sequence_length=8192)
    
    assert len(builder.nodes) > 0
    assert len(builder.edges) > 0
    
    analyzer = GraphAnalyzer(builder)
    crit_time, critical_path = analyzer.find_critical_path()
    assert len(critical_path) > 0
    assert crit_time > 0
    
    optimizer = GraphOptimizer(builder)
    opt_builder, report = optimizer.optimize()
    assert report.speedup_pct > 0
    assert report.optimized_critical_path_ms < report.original_critical_path_ms


def test_deepseek_v3_synthetic_moe_dag_construction():
    """Verify execution DAG builder for modeled 671B MoE topology with MLA & routed experts (synthetic fixture)."""
    builder = ExecutionGraphBuilder(model_name="Synthetic-DeepSeek-V3-671B-MoE")
    
    # Input embedding
    builder.add_node("emb", "TokenEmbedding", "MODULE", execution_time_ms=5.0, memory_footprint_mb=4096.0)
    
    # 4 MoE Pipeline stages with Multi-Head Latent Attention (MLA)
    prev_node = "emb"
    for stage_idx in range(4):
        norm_node = f"norm_{stage_idx}"
        mla_node = f"mla_attn_{stage_idx}"
        router_node = f"moe_router_{stage_idx}"
        expert_node = f"moe_experts_{stage_idx}"
        fused_add = f"fused_residual_{stage_idx}"
        
        builder.add_node(norm_node, f"RMSNorm_S{stage_idx}", "OPERATOR", execution_time_ms=0.5, memory_footprint_mb=512.0, is_fusible=True, op_type="norm")
        builder.add_node(mla_node, f"MLA_Attention_S{stage_idx}", "ATTENTION", execution_time_ms=18.5, memory_footprint_mb=8192.0, op_type="attention")
        builder.add_node(router_node, f"TopK_Router_S{stage_idx}", "OPERATOR", execution_time_ms=1.2, memory_footprint_mb=256.0, op_type="router")
        builder.add_node(expert_node, f"RoutedExperts256_S{stage_idx}", "MODULE", execution_time_ms=28.0, memory_footprint_mb=16384.0, op_type="gemm")
        builder.add_node(fused_add, f"ResidualAdd_S{stage_idx}", "OPERATOR", execution_time_ms=0.4, memory_footprint_mb=512.0, is_fusible=True, op_type="elementwise")
        
        builder.add_edge(prev_node, norm_node)
        builder.add_edge(norm_node, mla_node)
        builder.add_edge(mla_node, router_node)
        builder.add_edge(router_node, expert_node)
        builder.add_edge(expert_node, fused_add)
        prev_node = fused_add

    assert len(builder.nodes) == 21
    assert len(builder.edges) == 20
    
    analyzer = GraphAnalyzer(builder)
    crit_time, path = analyzer.find_critical_path()
    assert len(path) == 21
    assert crit_time > 0


def test_modeled_llm_variance_containment_and_damping():
    """Verify S-Plane and Fractional Hamiltonian compensators on simulated loss curve with synthetic shock."""
    comp_sp = GhostActiveCompensator(mode="s_plane", regime="nonpolar")
    comp_ham = GhostActiveCompensator(mode="fractional_hamiltonian", regime="nonpolar", alpha=1.5)
    
    np.random.seed(42)
    base_loss = 5.5
    for step in range(1, 51):
        decay = base_loss * np.exp(-0.01 * step)
        noise = np.random.normal(0, 0.25)
        if step == 25:
            noise += 3.0  # Synthetic gradient shock injection
        loss = max(0.01, decay + noise)
        
        res_sp = comp_sp.compute_step(step, loss)
        res_ham = comp_ham.compute_step(step, loss)
        
        if step == 25:
            assert res_sp.damping_factor_applied < 1.0
            assert res_ham.damping_factor_applied < 1.0

    sum_sp = comp_sp.get_summary()
    sum_ham = comp_ham.get_summary()
    assert sum_sp.regressive_steps_caught > 0
    assert sum_ham.regressive_steps_caught > 0
    assert sum_ham.variance_reduction_pct > 0


def test_synthetic_mega_cluster_roi_calculation():
    """Verify ROICalculator on a modeled 16,384 GPU cluster training run using explicit fee rates."""
    calc = ROICalculator(gpu_cost_per_hour=3.00, performance_fee_rate_pct=25.0, containment_premium_rate_pct=10.0)
    
    # Modeled 16,384 GPUs, 1200ms -> 960ms, 10,000 steps, 0.75 wasted GPU hours
    report = calc.calculate(
        baseline_step_time_ms=1200.0,
        optimized_step_time_ms=960.0,
        total_training_steps=10000,
        num_gpus=16384,
        leaked_gpu_hours=0.75,
    )
    
    assert report.baseline_gpu_hours > report.optimized_gpu_hours
    assert report.gpu_hours_saved > 0
    assert report.net_client_savings_usd > 0
    assert report.performance_fee_usd > 0
    assert report.containment_premium_usd > 0
    assert report.total_ghost_layer_revenue_usd == round(report.performance_fee_usd + report.containment_premium_usd, 2)
