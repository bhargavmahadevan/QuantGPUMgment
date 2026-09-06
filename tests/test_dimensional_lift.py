"""
Unit & Invariant Tests for GhostLayer Higher-Dimensional Optimization & Verification Suite.

Tests:
1. 4D Roofline Telemetry & Hyperplane Separator (ghost_layer.telemetry.roofline_4d)
2. 3D Credibility Manifold & Zero-Fabrication Decision Evaluator (ghost_layer.decision.credibility_manifold)
3. 3D Pareto-Convex Memory & Batching Hull (ghost_layer.curvature.pareto_hull)
4. 16D Privacy-Preserving Latent Execution Signatures (ghost_layer.distributed.signature_embedding)
"""

import math
import pytest
from ghost_layer.telemetry.roofline_4d import (
    TelemetryVector4D,
    BottleneckClass,
    Roofline4DClassifier,
)
from ghost_layer.decision.credibility_manifold import (
    CredibilityManifold,
    CredibilityStatus,
    CredibilityVector3D,
)
from ghost_layer.decision.engine import get_confidence_breakdown
from ghost_layer.curvature.pareto_hull import (
    ParetoMemoryHull,
    ActivationMode,
)
from ghost_layer.distributed.signature_embedding import (
    ExecutionSignatureEmbedding,
    ExecutionSignatureRegistry,
    cosine_similarity,
)


# ============================================================================
# 1. 4D Roofline Telemetry & Hyperplane Separator Tests
# ============================================================================

def test_telemetry_vector_4d_construction():
    vec = TelemetryVector4D(
        arithmetic_intensity=45.2,
        dram_bandwidth_pct=60.0,
        host_device_latency_ms=0.5,
        tensor_core_occupancy_pct=72.0,
    )
    arr = vec.to_vector()
    assert len(arr) == 4
    assert arr[0] == 45.2
    assert arr[1] == 60.0
    assert arr[2] == 0.5
    assert arr[3] == 72.0


def test_telemetry_vector_from_step_data():
    vec = TelemetryVector4D.from_step_data(
        tflops=20.0,
        memory_bandwidth_gbps=144.0,
        dram_peak_gbps=288.0,
        step_latency_ms=12.0,
        cuda_active_ms=10.0,
        sm_occupancy_pct=70.0,
    )
    assert vec.dram_bandwidth_pct == 50.0
    assert vec.host_device_latency_ms == 2.0
    assert vec.tensor_core_occupancy_pct == 70.0
    assert vec.arithmetic_intensity > 0.0


def test_roofline_4d_classifier_launch_latency_bound():
    classifier = Roofline4DClassifier()
    # High launch stall (3.5ms > 1.5ms threshold) with low SM occupancy (35%)
    vec = TelemetryVector4D(
        arithmetic_intensity=20.0,
        dram_bandwidth_pct=30.0,
        host_device_latency_ms=3.5,
        tensor_core_occupancy_pct=35.0,
    )
    res = classifier.classify(vec)
    assert res.primary_bottleneck == BottleneckClass.HOST_DEVICE_LAUNCH_BOUND
    assert res.target_rule_id == "GHOST_RULE_CUDA_GRAPHS"
    assert "CUDA Graphs" in res.recommended_action


def test_roofline_4d_classifier_memory_bandwidth_bound():
    classifier = Roofline4DClassifier()
    # High DRAM pressure (88% >= 75%) and low FLOP/Byte (18.0 < 40.0)
    vec = TelemetryVector4D(
        arithmetic_intensity=18.0,
        dram_bandwidth_pct=88.0,
        host_device_latency_ms=0.4,
        tensor_core_occupancy_pct=50.0,
    )
    res = classifier.classify(vec)
    assert res.primary_bottleneck == BottleneckClass.MEMORY_BANDWIDTH_BOUND
    assert res.target_rule_id == "GHOST_RULE_TORCH_COMPILE_FUSION"
    assert "kernel fusion" in res.recommended_action


def test_roofline_4d_classifier_compute_tensor_core_bound():
    classifier = Roofline4DClassifier()
    # High arithmetic intensity (95.0 >= 40.0) and high SM occupancy (85% >= 60%)
    vec = TelemetryVector4D(
        arithmetic_intensity=95.0,
        dram_bandwidth_pct=50.0,
        host_device_latency_ms=0.2,
        tensor_core_occupancy_pct=85.0,
    )
    res = classifier.classify(vec)
    assert res.primary_bottleneck == BottleneckClass.COMPUTE_TENSOR_CORE_BOUND
    assert res.target_rule_id == "GHOST_RULE_MIXED_PRECISION_FP8"
    assert "FP8" in res.recommended_action or "Muon" in res.recommended_action


def test_roofline_4d_classifier_dataloader_bound():
    classifier = Roofline4DClassifier()
    # Both compute and DRAM are starved
    vec = TelemetryVector4D(
        arithmetic_intensity=10.0,
        dram_bandwidth_pct=15.0,
        host_device_latency_ms=0.5,
        tensor_core_occupancy_pct=15.0,
    )
    res = classifier.classify(vec)
    assert res.primary_bottleneck == BottleneckClass.DATALOADER_IO_BOUND
    assert res.target_rule_id == "GHOST_RULE_DATALOADER_OPTIMIZATION"


def test_roofline_4d_classifier_trajectory():
    classifier = Roofline4DClassifier()
    vectors = [
        TelemetryVector4D(arithmetic_intensity=15.0, dram_bandwidth_pct=85.0, host_device_latency_ms=0.2, tensor_core_occupancy_pct=40.0),
        TelemetryVector4D(arithmetic_intensity=16.0, dram_bandwidth_pct=90.0, host_device_latency_ms=0.3, tensor_core_occupancy_pct=42.0),
        TelemetryVector4D(arithmetic_intensity=14.0, dram_bandwidth_pct=82.0, host_device_latency_ms=0.2, tensor_core_occupancy_pct=38.0),
    ]
    traj = classifier.classify_trajectory(vectors)
    assert traj["step_count"] == 3
    assert traj["dominant_bottleneck"] == BottleneckClass.MEMORY_BANDWIDTH_BOUND.value
    assert len(traj["mean_vector"]) == 4


# ============================================================================
# 2. 3D Credibility Manifold & Zero-Fabrication Tests
# ============================================================================

def test_credibility_manifold_zero_runs_capped():
    manifold = CredibilityManifold()
    # Zero verified runs must be strictly capped at 0.60
    eval_res = manifold.evaluate(
        heuristic_prior=0.95,
        verified_hardware_runs=0,
        loss_shift_margin=0.0,
    )
    assert eval_res.status == CredibilityStatus.HEURISTIC_PRIOR
    assert eval_res.calibrated_score <= 0.60
    assert eval_res.is_safe_to_auto_apply is False
    assert eval_res.is_empirical is False
    assert "HEURISTIC PRIOR" in eval_res.audit_label


def test_credibility_manifold_partial_runs():
    manifold = CredibilityManifold()
    # 2 verified runs -> EMPIRICALLY_VALIDATED, but not auto-apply safe yet (needs >= 5)
    eval_res = manifold.evaluate(
        heuristic_prior=0.85,
        verified_hardware_runs=2,
        loss_shift_margin=0.02,
    )
    assert eval_res.status == CredibilityStatus.EMPIRICALLY_VALIDATED
    assert eval_res.is_safe_to_auto_apply is False
    assert eval_res.is_empirical is True
    assert 0.50 <= eval_res.calibrated_score <= 0.90


def test_credibility_manifold_production_verified_safe():
    manifold = CredibilityManifold()
    # 6 verified runs + low loss shift (0.01) -> PRODUCTION_VERIFIED_SAFE
    eval_res = manifold.evaluate(
        heuristic_prior=0.90,
        verified_hardware_runs=6,
        loss_shift_margin=0.01,
        rollback_frequency=0.0,
    )
    assert eval_res.status == CredibilityStatus.PRODUCTION_VERIFIED_SAFE
    assert eval_res.is_safe_to_auto_apply is True
    assert eval_res.calibrated_score >= 0.80
    assert eval_res.distance_to_auto_apply_boundary == 0.0


def test_credibility_manifold_excessive_drift_hard_block():
    manifold = CredibilityManifold()
    # Loss shift delta of 0.15 exceeds 0.10 threshold -> HARD BLOCK
    eval_res = manifold.evaluate(
        heuristic_prior=0.95,
        verified_hardware_runs=10,
        loss_shift_margin=0.15,
    )
    assert eval_res.status == CredibilityStatus.BLOCKED_EXCESSIVE_DRIFT
    assert eval_res.is_safe_to_auto_apply is False
    assert "BLOCKED" in eval_res.audit_label


def test_engine_confidence_breakdown_integration():
    breakdown_zero = get_confidence_breakdown(
        hardware_match=1.0,
        model_similarity=1.0,
        historical_verifications=0,
        rollback_frequency=0.0,
    )
    assert breakdown_zero["score"] <= 0.60
    assert breakdown_zero["credibility_status"] == CredibilityStatus.HEURISTIC_PRIOR.value
    assert breakdown_zero["is_safe_to_auto_apply"] is False

    breakdown_verified = get_confidence_breakdown(
        hardware_match=1.0,
        model_similarity=1.0,
        historical_verifications=6,
        rollback_frequency=0.0,
    )
    assert breakdown_verified["score"] > 0.60
    assert breakdown_verified["credibility_status"] == CredibilityStatus.PRODUCTION_VERIFIED_SAFE.value
    assert breakdown_verified["is_safe_to_auto_apply"] is True


# ============================================================================
# 3. 3D Pareto-Convex Memory & Batching Hull Tests
# ============================================================================

def test_pareto_memory_hull_vram_estimation():
    hull = ParetoMemoryHull(
        vram_total_gb=16.0,
        safety_headroom_margin=0.15,
        model_params_billion=0.5,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
    )
    # Estimate VRAM for seq=1024, batch=4
    vram_fp32 = hull.estimate_vram_gb(1024, 4, ActivationMode.STANDARD_FP32)
    vram_amp = hull.estimate_vram_gb(1024, 4, ActivationMode.MIXED_PRECISION_AMP)
    vram_opt = hull.estimate_vram_gb(1024, 4, ActivationMode.FULL_OPTIMIZED_STACK)

    # Mixed precision and optimized stack should use strictly less memory than FP32
    assert vram_opt < vram_amp < vram_fp32
    assert hull.is_within_hull(1024, 4, ActivationMode.FULL_OPTIMIZED_STACK)


def test_pareto_convex_hull_frontier_computation():
    hull = ParetoMemoryHull(
        vram_total_gb=16.0,
        safety_headroom_margin=0.15,
        model_params_billion=0.5,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
    )
    report = hull.compute_pareto_frontier(
        candidate_seq_lens=[512, 1024, 2048],
        max_batch_size=32,
        mode=ActivationMode.FULL_OPTIMIZED_STACK,
    )
    assert len(report.pareto_frontier) > 0
    assert report.optimal_point.estimated_vram_gb <= hull.vram_usable
    assert report.optimal_point.vram_headroom_pct >= 0.0
    assert report.safe_convex_hull_volume > 0.0
    assert "microbatch" in report.recommended_scaling_strategy


# ============================================================================
# 4. 16D Latent Execution Signature Embedding Tests
# ============================================================================

def test_execution_signature_l2_normalization():
    sig = ExecutionSignatureEmbedding.from_step_telemetry(
        ai=45.0,
        dram_pct=70.0,
        launch_lat_ms=0.5,
        sm_occ_pct=80.0,
        step_time_ms=10.0,
        vram_allocated_gb=6.0,
        vram_total_gb=16.0,
    )
    raw = sig.to_raw_vector()
    assert len(raw) == 16

    emb = sig.to_unit_hypersphere_embedding()
    assert len(emb) == 16
    norm_sq = sum(x * x for x in emb)
    assert abs(math.sqrt(norm_sq) - 1.0) < 1e-4


def test_cosine_similarity_properties():
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0, 0.0]
    v4 = [-1.0, 0.0, 0.0, 0.0]

    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-5
    assert abs(cosine_similarity(v1, v4) - (-1.0)) < 1e-5


def test_execution_signature_registry_nearest_match():
    registry = ExecutionSignatureRegistry()

    # Query 1: Memory-bound query should match Fused Flash Attention recipe
    query_mem = ExecutionSignatureEmbedding(
        arithmetic_intensity_norm=0.18,
        dram_bandwidth_saturation=0.88,
        host_to_device_latency_ratio=0.04,
        tensor_core_occupancy_norm=0.30,
        sm_utilization_ratio=0.35,
        grad_norm_mean_log=0.20,
        loss_delta_snr=0.80,
        birkhoff_drift_residual=0.0,
        vram_utilization_pct=0.55,
        kernel_launch_frequency_khz=0.75,
        gemm_to_elementwise_ratio=0.32,
        activation_memory_ratio=0.68,
        forward_to_backward_ratio=0.33,
        dataloader_wait_ratio=0.02,
        effective_tflops_ratio=0.22,
        curvature_second_order_norm=0.1,
    )
    match_mem = registry.find_nearest_recipe(query_mem)
    assert match_mem is not None
    assert match_mem.matched_recipe_id == "RECIPE_FUSED_FLASH_ATTN"
    assert match_mem.similarity_score >= 0.80
    assert match_mem.is_high_confidence_match is True

    # Query 2: Dense compute query should match Muon Hybrid FP8 recipe
    query_compute = ExecutionSignatureEmbedding(
        arithmetic_intensity_norm=0.82,
        dram_bandwidth_saturation=0.50,
        host_to_device_latency_ratio=0.02,
        tensor_core_occupancy_norm=0.88,
        sm_utilization_ratio=0.92,
        grad_norm_mean_log=0.32,
        loss_delta_snr=0.92,
        birkhoff_drift_residual=0.0,
        vram_utilization_pct=0.80,
        kernel_launch_frequency_khz=0.2,
        gemm_to_elementwise_ratio=0.88,
        activation_memory_ratio=0.48,
        forward_to_backward_ratio=0.33,
        dataloader_wait_ratio=0.01,
        effective_tflops_ratio=0.78,
        curvature_second_order_norm=0.15,
    )
    match_comp = registry.find_nearest_recipe(query_compute)
    assert match_comp is not None
    assert match_comp.matched_recipe_id == "RECIPE_MUON_HYBRID_FP8"
    assert match_comp.similarity_score >= 0.85
