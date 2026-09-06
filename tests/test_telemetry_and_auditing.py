"""
GhostLayer Telemetry, Auditing, Attention, & Latency Test Suite.
Combines tests for:
- Telemetry Watcher, Decision Engine, Privacy Scrubber, ROI Calculator
- Tensor Flow & GPU Vitality Suite (TensorFlowAuditor, Rules 13 & 14)
- Telemetry Persistence & Spot Preemption Resumption
- Active S-Plane & Fractional Hamiltonian Compensator
- Attention Profiler (MHA vs GQA vs MQA vs MLA, DSA sparse scaling, arithmetic intensity)
- Latency Chain breakdown (IO, communication, memory bandwidth, compute bound)
"""

import os
import json
import pytest
from pathlib import Path
import torch
import torch.nn as nn

# GhostLayer Core & Telemetry
from ghost_layer.telemetry.watcher import TelemetryWatcher, TelemetrySummary, MetricSnapshot
from ghost_layer.telemetry.persistence import TelemetryPersistence
from ghost_layer.telemetry.tensor_auditor import TensorFlowAuditor, audit_gpu_vitality
from ghost_layer.telemetry.compensator import (
    GhostActiveCompensator,
    CompensationStepResult,
    FractionalHamiltonianCompensator,
)
from ghost_layer.telemetry.attention_profiler import (
    AttentionArchitecture,
    AttentionProfile,
    compute_kv_cache_footprint,
    calculate_gqa_compression_ratio,
    calculate_mla_compression_ratio,
    estimate_attention_arithmetic_intensity,
)
from ghost_layer.telemetry.latency_chain import (
    LatencyChainBreakdown,
    profile_latency_chain,
)
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.verification.verifier import CorrectnessVerifier
from ghost_layer.roi.calculator import ROICalculator
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, PrivacyScrubber, TelemetryBoundary
from ghost_layer.applier import AutoApplier, ConsentLevel
from ghost_layer.app import load_genuine_log, GENUINE_LOGS_DIR


# ==============================================================================
# 1. Telemetry Watcher & Decision Engine Tests
# ==============================================================================

def test_telemetry_watcher():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    for i in range(1, 11):
        watcher.record_step(
            step=i,
            gpu_utilization_pct=50.0,
            gpu_memory_used_mb=8000.0,
            step_time_ms=100.0,
            loss=1.0 / i,
            data_loading_time_ms=20.0,
            mixed_precision="fp32",
        )
    watcher.stop()
    summary = watcher.get_summary()

    assert summary.total_steps == 10
    assert summary.avg_gpu_utilization_pct == 50.0
    assert summary.peak_gpu_memory_mb == 8000.0
    assert summary.avg_step_time_ms == 100.0
    assert summary.avg_dataloader_stall_pct == 20.0


def test_decision_engine():
    engine = DecisionEngine()
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    for i in range(1, 6):
        watcher.record_step(
            step=i,
            gpu_utilization_pct=40.0,
            gpu_memory_used_mb=15000.0,
            step_time_ms=200.0,
            mixed_precision="fp32",
            num_workers=0,
            flash_attention=False,
            gradient_checkpointing=False,
        )
    watcher.stop()
    summary = watcher.get_summary()
    recs = engine.evaluate(summary, model_type="transformer")

    rule_ids = [r.rule_id for r in recs]
    assert "RULE_MIXED_PRECISION" in rule_ids
    assert "RULE_DATALOADER_WORKERS" in rule_ids
    assert "RULE_FLASH_ATTENTION" in rule_ids
    assert "RULE_GRADIENT_CHECKPOINTING" in rule_ids


def test_correctness_verifier_safe():
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05, max_allowed_kl_div=0.02)
    base_loss = [2.0, 1.8, 1.5, 1.2, 1.0]
    opt_loss = [2.01, 1.81, 1.51, 1.21, 1.01]

    result = verifier.verify_trajectories(base_loss, opt_loss, "RULE_MIXED_PRECISION", was_auto_applied=True)
    assert result.is_safe is True
    assert result.action_taken == "AUTO_APPLIED"


def test_correctness_verifier_divergent():
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05, max_allowed_kl_div=0.02)
    base_loss = [2.0, 1.8, 1.5, 1.2, 1.0]
    divergent_loss = [2.0, 3.5, 4.2, 5.0, 6.1]

    result = verifier.verify_trajectories(base_loss, divergent_loss, "RULE_MIXED_PRECISION", was_auto_applied=True)
    assert result.is_safe is False
    assert result.action_taken == "DOWNGRADED_TO_RECOMMENDATION"


def test_roi_calculator():
    calc = ROICalculator(gpu_cost_per_hour=4.00, performance_fee_rate_pct=25.0)
    report = calc.calculate(
        baseline_step_time_ms=200.0,
        optimized_step_time_ms=100.0,
        total_training_steps=360000,
        num_gpus=8,
    )

    assert report.baseline_gpu_hours == 160.0
    assert report.optimized_gpu_hours == 80.0
    assert report.gpu_hours_saved == 80.0
    assert report.baseline_cost_usd == 640.0
    assert report.optimized_cost_usd == 320.0
    assert report.performance_fee_usd == 80.0
    assert report.net_client_savings_usd == 240.0


def test_knowledge_base_privacy():
    raw_meta = {
        "architecture_family": "GPT-4-Style",
        "dataset_name": "SECRET_PROPRIETARY_CORPUS.csv",
        "file_path": "C:\\Users\\Secret\\model.pt",
        "mixed_precision": "bf16",
        "flash_attention": True,
    }
    sanitized = PrivacyScrubber.sanitize(raw_meta)
    assert "dataset_name" not in sanitized
    assert "file_path" not in sanitized
    assert sanitized["mixed_precision"] == "bf16"

    db_path = "test_kb_temp.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    kb = SharedKnowledgeBase(db_file_path=db_path)
    kb.register_learning("Llama-7B", "A100", {"mixed_precision": "bf16"}, 35.0)
    kb.register_learning("Llama-7B", "A100", {"mixed_precision": "bf16"}, 45.0)

    entry = kb.query("Llama-7B", "A100")
    assert entry is not None
    assert entry.sample_count == 2
    assert entry.throughput_improvement_pct == 40.0

    if os.path.exists(db_path):
        os.remove(db_path)


# ==============================================================================
# 2. Tensor Flow Vitality & Memory Alignment Tests (Rules 13 & 14)
# ==============================================================================

class UnalignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(13, 27, bias=False)

    def forward(self, x):
        return self.fc(x)


class AlignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64, bias=False)

    def forward(self, x):
        return self.fc(x)


class DummyDataLoader:
    def __init__(self, pin_memory=False, num_workers=0):
        self.pin_memory = pin_memory
        self.num_workers = num_workers


def test_tensor_flow_vitality_audit_unaligned():
    model = UnalignedModel()
    loader = DummyDataLoader(pin_memory=False, num_workers=0)

    diag = audit_gpu_vitality(model=model, dataloader=loader)
    assert not diag.stride_alignment_optimal
    assert diag.unpinned_memory_transfers
    assert diag.vitality_score_pct < 100.0
    assert len(diag.base_leakage_diagnoses) >= 2


def test_tensor_flow_vitality_audit_aligned():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = AlignedModel()
    loader = DummyDataLoader(pin_memory=True, num_workers=4)

    diag = audit_gpu_vitality(model=model, dataloader=loader)
    assert diag.stride_alignment_optimal
    assert not diag.unpinned_memory_transfers
    assert diag.vitality_score_pct >= 70.0


def test_decision_engine_rules_13_and_14():
    engine = DecisionEngine()
    summary = TelemetrySummary(
        total_steps=500,
        total_duration_sec=20.0,
        avg_gpu_utilization_pct=60.0,
        peak_gpu_memory_mb=7000.0,
        gpu_memory_total_mb=8000.0,
        avg_step_time_ms=85.0,
        avg_dataloader_stall_pct=15.0,
        mixed_precision="bf16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=0,
        pin_memory=False,
    )

    recs = engine.evaluate(summary=summary, model_type="transformer")
    rule_ids = [r.rule_id for r in recs]

    assert "RULE_ASYNC_TENSOR_FLOW" in rule_ids
    assert "RULE_CUDA_ALLOCATOR_TUNING" in rule_ids


def test_applier_rules_13_and_14():
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_ALL)
    loader = DummyDataLoader(pin_memory=False, num_workers=0)

    res_async = applier.apply_recommendation("RULE_ASYNC_TENSOR_FLOW", dataloader=loader)
    assert res_async.applied
    assert res_async.after_config.get("pin_memory") is True

    res_alloc = applier.apply_recommendation("RULE_CUDA_ALLOCATOR_TUNING")
    assert res_alloc.applied
    assert "expandable_segments:True" in res_alloc.after_config.get("PYTORCH_CUDA_ALLOC_CONF", "")


# ==============================================================================
# 3. Telemetry Persistence & Resumption Tests
# ==============================================================================

def test_telemetry_persistence_append_and_load(tmp_path):
    storage_dir = str(tmp_path / "sessions")
    session_id = "test_run_123"

    persistence = TelemetryPersistence(storage_dir=storage_dir, session_id=session_id)
    assert persistence.file_path.name == "test_run_123.jsonl"

    watcher = TelemetryWatcher(
        target_gpu_mb=16384.0,
        persistence_dir=storage_dir,
        session_id=session_id,
    )
    watcher.start()

    for step in range(1, 6):
        watcher.record_step(
            step=step,
            gpu_utilization_pct=80.0 + step,
            gpu_memory_used_mb=4000.0,
            step_time_ms=25.0,
            loss=2.5 - (step * 0.1),
        )

    assert persistence.file_path.exists()
    loaded_snaps = persistence.load_snapshots()
    assert len(loaded_snaps) == 5
    assert loaded_snaps[0].step == 1
    assert loaded_snaps[-1].step == 5
    assert loaded_snaps[-1].loss == pytest.approx(2.0, abs=1e-4)


def test_telemetry_watcher_resume_from_file(tmp_path):
    storage_dir = str(tmp_path / "sessions")
    session_id = "preempted_run"

    watcher1 = TelemetryWatcher(
        target_gpu_mb=16384.0,
        persistence_dir=storage_dir,
        session_id=session_id,
    )
    watcher1.start()

    for step in range(1, 11):
        watcher1.record_step(
            step=step,
            gpu_utilization_pct=85.0,
            gpu_memory_used_mb=6000.0,
            step_time_ms=30.0,
            loss=3.0 - (step * 0.05),
        )

    file_path = str(Path(storage_dir) / f"{session_id}.jsonl")
    watcher2 = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher2.resume_from_file(file_path)

    assert len(watcher2.snapshots) == 10
    assert len(watcher2.variance_errors) == 9
    assert watcher2._last_loss == pytest.approx(2.5, abs=1e-4)

    summary = watcher2.get_summary()
    assert summary.total_steps == 10
    assert summary.peak_gpu_memory_mb == 6000.0


# ==============================================================================
# 4. Active S-Plane & Fractional Hamiltonian Compensator Tests
# ==============================================================================

def test_compensator_laminar_flow():
    comp = GhostActiveCompensator(regime="polar")
    losses = [4.0, 3.8, 3.6, 3.4, 3.2]
    for step, l in enumerate(losses, 1):
        res = comp.compute_step(step, l)
        if step > 1:
            assert res.is_regressive is False
            assert res.damping_factor_applied == 1.0

    summary = comp.get_summary()
    assert summary.regressive_steps_caught == 0
    assert summary.avg_damping_applied == 1.0


def test_compensator_regressive_spike_damping():
    comp_sp = GhostActiveCompensator(mode="s_plane", regime="nonpolar")
    comp_sp.compute_step(1, 3.0)
    comp_sp.compute_step(2, 2.9)
    res_sp = comp_sp.compute_step(3, 3.8)

    assert res_sp.is_regressive is True
    assert res_sp.damping_factor_applied < 0.7
    assert "NONPOLAR_ELASTIC_DAMPING" in res_sp.action_taken

    comp_ham = GhostActiveCompensator(mode="fractional_hamiltonian", regime="nonpolar")
    comp_ham.compute_step(1, 3.0)
    comp_ham.compute_step(2, 2.9)
    res_ham = comp_ham.compute_step(3, 3.8)

    assert res_ham.is_regressive is True
    assert res_ham.damping_factor_applied < 0.7
    assert "HAMILTONIAN_LEVY_DAMPING" in res_ham.action_taken

    summary = comp_ham.get_summary()
    assert summary.regressive_steps_caught == 1
    assert summary.variance_reduction_pct > 0.0


def test_compensator_regimes():
    for regime in ["nonpolar", "polar", "interfacial"]:
        comp = GhostActiveCompensator(regime=regime)
        comp.compute_step(1, 2.5)
        res = comp.compute_step(2, 3.1)
        assert res.is_regressive is True
        assert 0.15 <= res.damping_factor_applied <= 1.0


def test_fractional_hamiltonian_symplectic_drift():
    ham = FractionalHamiltonianCompensator(alpha=1.5, friction_beta=0.1)

    res1 = ham.compute_step(1, 3.5, grad_norm_sq=1.0)
    assert res1.damping_factor_applied == 1.0

    res2 = ham.compute_step(2, 3.0, grad_norm_sq=1.0)
    assert res2.damping_factor_applied == 1.0
    assert res2.energy_drift < 0

    res3 = ham.compute_step(3, 5.5, grad_norm_sq=10.0)
    assert res3.energy_drift > 0
    assert res3.damping_factor_applied < 0.3
    assert "HAMILTONIAN_LEVY_DAMPING" in res3.action_taken


def test_compensator_modes():
    comp_ham = GhostActiveCompensator(mode="fractional_hamiltonian")
    comp_sp = GhostActiveCompensator(mode="s_plane")

    res_ham = comp_ham.compute_step(1, 4.0)
    res_sp = comp_sp.compute_step(1, 4.0)

    assert res_ham.mode == "fractional_hamiltonian"
    assert res_sp.mode == "s_plane"


def test_genuine_training_logs_exist():
    for filename in [
        "zephyr_7b_sft_trainer_state.json",
        "zephyr_7b_dpo_trainer_state.json",
        "mistral_7b_sft_trainer_state.json",
    ]:
        path = os.path.join(GENUINE_LOGS_DIR, filename)
        assert os.path.exists(path), f"Missing genuine log: {filename}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "log_history" in data
            assert len(data["log_history"]) > 0


# ==============================================================================
# 5. Attention Architecture Profiler & KV Cache Footprint Tests
# ==============================================================================

def test_kv_cache_footprint_mha_vs_gqa_vs_mqa():
    mha_res = compute_kv_cache_footprint(
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        seq_len=4096,
        batch_size=1,
        architecture="MHA",
        precision_bytes=2,
    )
    assert mha_res["kv_cache_mb"] == 2048.0
    assert mha_res["compression_ratio_vs_mha"] == 1.0

    gqa_res = compute_kv_cache_footprint(
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
        seq_len=4096,
        batch_size=1,
        architecture="GQA",
        precision_bytes=2,
    )
    assert gqa_res["kv_cache_mb"] == 512.0
    assert gqa_res["compression_ratio_vs_mha"] == 4.0

    mqa_res = compute_kv_cache_footprint(
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        seq_len=4096,
        batch_size=1,
        architecture="MQA",
        precision_bytes=2,
    )
    assert mqa_res["kv_cache_mb"] == 64.0
    assert mqa_res["compression_ratio_vs_mha"] == 32.0


def test_mla_latent_compression():
    mla_res = compute_kv_cache_footprint(
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        seq_len=4096,
        batch_size=1,
        architecture="MLA",
        mla_latent_dim=512,
        mla_rope_dim=64,
        precision_bytes=2,
    )
    assert mla_res["kv_cache_mb"] == 144.0
    assert mla_res["compression_ratio_vs_mha"] > 14.0

    comp_ratio = calculate_mla_compression_ratio(num_query_heads=32, head_dim=128, latent_dim_kv=512, rope_dim=64)
    assert comp_ratio == 14.22


def test_dsa_sparse_scaling():
    dsa_res = compute_kv_cache_footprint(
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
        seq_len=32768,
        batch_size=1,
        architecture="DSA",
        sparse_top_k=2048,
        precision_bytes=2,
    )
    assert dsa_res["compression_ratio_vs_mha"] > 30.0


def test_arithmetic_intensity():
    prefill_intensity = estimate_attention_arithmetic_intensity(
        seq_len=2048, head_dim=128, batch_size=2, num_heads=32, is_prefill=True
    )
    decode_intensity = estimate_attention_arithmetic_intensity(
        seq_len=2048, head_dim=128, batch_size=2, num_heads=32, is_prefill=False
    )
    assert prefill_intensity > 500.0
    assert decode_intensity < 2.0


def test_latency_chain_profiler_bottlenecks():
    breakdown_io = profile_latency_chain(step_time_ms=100.0, dataloader_time_ms=40.0)
    assert breakdown_io.dominant_bottleneck == "IO_BOUND"
    assert breakdown_io.dataloader_stall_ms == 40.0

    breakdown_comm = profile_latency_chain(step_time_ms=100.0, communication_time_ms=45.0)
    assert breakdown_comm.dominant_bottleneck == "COMMUNICATION_BOUND"

    breakdown_mem = profile_latency_chain(
        step_time_ms=25.0,
        is_generation=True,
        estimated_kv_cache_mb=4096.0,
        hardware_bandwidth_gb_s=900.0,
    )
    assert breakdown_mem.dominant_bottleneck == "MEMORY_BANDWIDTH_BOUND"

    breakdown_comp = profile_latency_chain(
        step_time_ms=30.0,
        dataloader_time_ms=1.0,
        communication_time_ms=1.0,
        is_generation=False,
    )
    assert breakdown_comp.dominant_bottleneck == "COMPUTE_BOUND"


def test_watcher_with_attention_and_latency_telemetry():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()

    snapshot = watcher.record_step(
        step=1,
        gpu_utilization_pct=85.0,
        gpu_memory_used_mb=10240.0,
        step_time_ms=35.0,
        loss=2.15,
        attention_architecture="GQA",
        sequence_length=4096,
        estimated_kv_cache_mb=512.0,
        ttft_ms=50.0,
        tpot_ms=12.5,
        dominant_latency_bottleneck="COMPUTE_BOUND",
    )
    assert snapshot.attention_architecture == "GQA"
    assert snapshot.sequence_length == 4096
    assert snapshot.estimated_kv_cache_mb == 512.0

    summary = watcher.get_summary()
    assert summary.attention_architecture == "GQA"
    assert summary.sequence_length == 4096
    assert summary.estimated_kv_cache_mb == 512.0
    assert summary.dominant_latency_bottleneck == "COMPUTE_BOUND"


def test_decision_engine_attention_rules():
    engine = DecisionEngine(target_hardware="NVIDIA RTX A2000")

    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=30.0,
        avg_gpu_utilization_pct=80.0,
        peak_gpu_memory_mb=12000.0,
        gpu_memory_total_mb=16384.0,
        avg_step_time_ms=45.0,
        avg_dataloader_stall_pct=2.0,
        mixed_precision="bf16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
        attention_architecture="MHA",
        sequence_length=8192,
        estimated_kv_cache_mb=2048.0,
        ttft_ms=60.0,
        tpot_ms=25.0,
        dominant_latency_bottleneck="MEMORY_BANDWIDTH_BOUND",
    )

    recs = engine.evaluate(summary, model_type="llama")
    rule_ids = [r.rule_id for r in recs]

    assert "RULE_GQA_UPTRAINING" in rule_ids
    assert "RULE_MLA_LATENT_COMPRESSION" in rule_ids
    assert "RULE_DYNAMIC_SPARSE_ATTENTION" in rule_ids
    assert "RULE_LATENCY_CHAIN_KV_BOUND" in rule_ids

    for r in recs:
        if r.rule_id in ["RULE_GQA_UPTRAINING", "RULE_MLA_LATENT_COMPRESSION", "RULE_DYNAMIC_SPARSE_ATTENTION", "RULE_LATENCY_CHAIN_KV_BOUND"]:
            assert r.has_verified_runs is False
            assert r.confidence <= 0.60


def test_telemetry_boundary_includes_attention_fields():
    audit = TelemetryBoundary.audit()
    collected_names = [f["field_name"] for f in audit["collected"]]
    assert "attention_architecture" in collected_names
    assert "kv_cache_mb" in collected_names
    assert "ttft_ms" in collected_names
    assert "tpot_ms" in collected_names
    assert "dominant_latency_bottleneck" in collected_names
    assert audit["summary"]["pii_collected"] is False
    assert audit["summary"]["model_weights_collected"] is False
