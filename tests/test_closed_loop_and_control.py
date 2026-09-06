"""
GhostLayer Closed-Loop Execution, Control, & Rollback Test Suite.
Combines tests for:
- AutoApplier and ConsentLevel gating
- RollbackManager and state restoration
- SavingsVerifier and statistical significance testing (Welch's t-test)
- OptimizationReplayLog and audit trail recording
- Closed-Loop OutcomeEvaluator and calibration wiring
- Dynamic Hive-Mind Surrogate and Replay Feedback
"""

import os
import pytest
import torch
import torch.nn as nn

# GhostLayer Closed-Loop Modules
from ghost_layer.applier import AutoApplier, ConsentLevel, ApplyResult
from ghost_layer.rollback import RollbackManager, ConfigSnapshot, RollbackResult
from ghost_layer.savings import SavingsVerifier, BaselineMeasurement, VerifiedSavingsReport
from ghost_layer.replay import OptimizationReplayLog
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.feedback.outcome_evaluator import OutcomeEvaluator
from ghost_layer.benchmarking.calibration_runner import (
    CalibrationRunner,
    CalibrationReport,
    BenchmarkResult,
    BenchmarkConfig,
)
from ghost_layer.decision.engine import DecisionEngine, calculate_evidence_confidence
from ghost_layer.telemetry.watcher import TelemetryWatcher, TelemetrySummary
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, SurrogateOptimizationModel
from ghost_layer.verification.verifier import CorrectnessVerifier
from ghost_layer.reporting.generator import ReportGenerator
from ghost_layer.roi.calculator import ROICalculator


# ==============================================================================
# Helper Dummy Models
# ==============================================================================

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False


# ==============================================================================
# 1. AutoApplier & ConsentLevel Tests
# ==============================================================================

def test_consent_level_audit_only_blocks():
    applier = AutoApplier(consent_level=ConsentLevel.AUDIT_ONLY)
    res = applier.apply_recommendation("RULE_MIXED_PRECISION", recommendation_is_safe=True)
    assert not res.applied
    assert "Blocked by consent level" in res.error


def test_consent_level_auto_apply_safe():
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_SAFE)
    model = DummyModel()

    res_safe = applier.apply_recommendation("RULE_MIXED_PRECISION", model=model, recommendation_is_safe=True)
    assert res_safe.applied
    assert res_safe.after_config.get("mixed_precision") in ["fp16", "bf16"]

    res_unsafe = applier.apply_recommendation("RULE_BATCH_SCALING", model=model, recommendation_is_safe=False)
    assert not res_unsafe.applied


def test_consent_level_recommend_and_ask():
    applier = AutoApplier(consent_level=ConsentLevel.RECOMMEND_AND_ASK)
    model = DummyModel()

    res_blocked = applier.apply_recommendation("RULE_MUON_OPTIMIZER", model=model, recommendation_is_safe=True)
    assert not res_blocked.applied

    res_approved = applier.approve_and_apply("RULE_MUON_OPTIMIZER", model=model)
    assert res_approved.applied
    assert res_approved.after_config.get("optimizer") == "HybridMuonOptimizer"


def test_apply_curvature_optimizers():
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_ALL)
    model = DummyModel()

    res_sophia = applier.apply_recommendation("RULE_SOPHIA_SECOND_ORDER", model=model)
    assert res_sophia.applied
    assert res_sophia.after_config.get("optimizer") == "SophiaG"

    res_shampoo = applier.apply_recommendation("RULE_SHAMPOO_PRECONDITIONING", model=model)
    assert res_shampoo.applied
    assert res_shampoo.after_config.get("optimizer") == "Shampoo"


# ==============================================================================
# 2. Rollback Manager Tests
# ==============================================================================

def test_rollback_snapshot_and_restore():
    mgr = RollbackManager()
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    assert not model.is_gradient_checkpointing

    snap = mgr.snapshot(model=model, optimizer=optimizer, rule_id="RULE_GRADIENT_CHECKPOINTING")
    assert snap.snapshot_id in mgr._snapshots

    model.gradient_checkpointing_enable()
    assert model.is_gradient_checkpointing

    result = mgr.rollback(snap.snapshot_id, model=model, optimizer=optimizer)
    assert result.success
    assert not model.is_gradient_checkpointing
    assert "gradient_checkpointing=False" in result.restored_fields


def test_rollback_unknown_snapshot():
    mgr = RollbackManager()
    res = mgr.rollback("non_existent_snap")
    assert not res.success
    assert "not found" in res.error


# ==============================================================================
# 3. Savings Verifier & Statistical Testing Tests
# ==============================================================================

def test_savings_verifier_statistically_significant():
    verifier = SavingsVerifier(gpu_cost_per_hour=3.50, significance_level=0.05)

    baseline_times = [50.0 + i * 0.2 for i in range(30)]
    optimized_times = [25.0 + i * 0.1 for i in range(30)]

    base = verifier.measure(baseline_times, label="baseline")
    opt = verifier.measure(optimized_times, label="optimized")

    report = verifier.compare(base, opt)
    assert report.is_verified
    assert report.speedup_pct > 40.0
    assert report.p_value < 0.001
    assert report.dollar_savings_per_1000_gpu_hours > 0.0
    assert "VERIFIED" in report.verification_label


def test_savings_verifier_noise_not_significant():
    verifier = SavingsVerifier(gpu_cost_per_hour=3.50, significance_level=0.01)

    baseline_times = [30.0 + ((-1)**i) * 5.0 for i in range(10)]
    optimized_times = [29.8 + ((-1)**i) * 5.0 for i in range(10)]

    base = verifier.measure(baseline_times)
    opt = verifier.measure(optimized_times)

    report = verifier.compare(base, opt)
    assert not report.is_verified
    assert "NOT SIGNIFICANT" in report.verification_label


# ==============================================================================
# 4. Optimization Replay Log Tests
# ==============================================================================

def test_optimization_replay_logging():
    log = OptimizationReplayLog(session_id="session_test_001")
    event = log.record_event(
        step=100,
        stage="SAFE_APPLY",
        recommendation_id="RULE_FLASH_ATTENTION",
        action="ENABLE_FLASH_ATTN_2",
        details={"speedup_est": "1.35x"},
        verified_safe=True,
        throughput_delta_pct=28.5,
        reason="Verified loss trajectory divergence below 0.01 tolerance",
    )

    assert event.stage == "SAFE_APPLY"
    assert event.verified_safe is True
    assert event.throughput_delta_pct == 28.5

    trail = log.generate_audit_trail()
    assert len(trail) == 1
    assert trail[0]["recommendation_id"] == "RULE_FLASH_ATTENTION"


# ==============================================================================
# 5. Closed-Loop Wiring & Outcome Evaluator Tests
# ==============================================================================

def test_auto_applier_triggers_outcome_evaluator(tmp_path):
    kb = SharedKnowledgeBase(db_file_path=str(tmp_path / "kb.json"))
    replay = OptimizationReplayLog(session_id="test_apply_eval")
    evaluator = OutcomeEvaluator(
        knowledge_base=kb,
        replay_log=replay,
        baseline_window_size=3,
        eval_window_size=3,
    )

    applier = AutoApplier(
        consent_level=ConsentLevel.AUTO_APPLY_SAFE,
        replay_log=replay,
        outcome_evaluator=evaluator,
    )

    for s in range(1, 4):
        evaluator.record_step(step=s, step_time_ms=100.0, loss=2.0)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self._current_step = 3
        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    res = applier.apply_recommendation(
        rule_id="RULE_MIXED_PRECISION",
        model=model,
        recommendation_is_safe=True,
        step=3,
    )
    assert res.applied is True

    assert len(evaluator.pending_evaluations) == 1
    assert evaluator.pending_evaluations[0]["rule_id"] == "RULE_MIXED_PRECISION"

    for s in range(4, 7):
        evaluator.record_step(step=s, step_time_ms=60.0, loss=2.01)

    assert len(evaluator.completed_evaluations) == 1
    eval_result = evaluator.completed_evaluations[0]
    assert eval_result.rule_id == "RULE_MIXED_PRECISION"
    assert eval_result.realized_speedup_pct == 40.0
    assert eval_result.is_safe is True
    assert eval_result.registered_to_kb is True


def test_ghost_watcher_hook_records_in_outcome_evaluator(tmp_path):
    kb = SharedKnowledgeBase(db_file_path=str(tmp_path / "kb.json"))
    hook = GhostWatcherHook(
        gpu_memory_mb=8192.0,
        knowledge_base=kb,
        consent_level=ConsentLevel.AUDIT_ONLY,
    )

    with hook:
        for s in range(1, 10):
            hook.on_step_begin()
            hook.on_step_end(gpu_util_pct=75.0, gpu_mem_used_mb=4000.0, loss=1.5 - s * 0.01)

    assert len(hook.outcome_evaluator.step_history) == 9
    assert hook.outcome_evaluator.step_history[0][0] == 1


def test_decision_engine_uses_calibration_report():
    calib = CalibrationReport(
        hardware_name="NVIDIA RTX 4090",
        precision_speedup_pct=52.4,
        workers_speedup_pct=28.1,
        grad_checkpoint_memory_saving_pct=64.0,
        empirical_results=[],
        is_empirically_measured=True,
    )

    engine = DecisionEngine(
        target_hardware="NVIDIA RTX 4090",
        calibration_report=calib,
    )

    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=12.0,
        avg_step_time_ms=120.0,
        avg_gpu_utilization_pct=50.0,
        peak_gpu_memory_mb=7500.0,
        gpu_memory_total_mb=8000.0,
        mixed_precision="fp32",
        avg_dataloader_stall_pct=20.0,
        num_workers=0,
        pin_memory=False,
        gradient_checkpointing=False,
        flash_attention=False,
    )

    recs = engine.evaluate(summary, model_type="transformer")
    rec_by_id = {r.rule_id: r for r in recs}

    mp_rec = rec_by_id["RULE_MIXED_PRECISION"]
    assert "52.4%" in mp_rec.speedup_estimate_label
    assert "Measured on NVIDIA RTX 4090" in mp_rec.speedup_estimate_label
    assert mp_rec.has_verified_runs is True

    dl_rec = rec_by_id["RULE_DATALOADER_WORKERS"]
    assert "28.1%" in dl_rec.speedup_estimate_label
    assert dl_rec.has_verified_runs is True

    gc_rec = rec_by_id["RULE_GRADIENT_CHECKPOINTING"]
    assert "64.0% memory saving on NVIDIA RTX 4090" in gc_rec.evidence
    assert gc_rec.has_verified_runs is True


def test_ghost_watcher_hook_calibrate_hardware(tmp_path):
    kb = SharedKnowledgeBase(db_file_path=str(tmp_path / "kb.json"))
    hook = GhostWatcherHook(
        gpu_memory_mb=8192.0,
        knowledge_base=kb,
    )

    cfg = BenchmarkConfig(
        model_family="transformer",
        hidden_dim=32,
        num_layers=2,
        seq_len=8,
        batch_size=4,
        steps_warmup=1,
        steps_benchmark=2,
    )

    report = hook.calibrate_hardware(config=cfg)
    assert report.is_empirically_measured is True
    assert hook.decision_engine.calibration_report is not None
    assert hook.decision_engine.calibration_report == report


# ==============================================================================
# 6. Surrogate Model Dynamic Query & Report Explainability Tests
# ==============================================================================

def test_surrogate_model_dynamic_query_and_decision_engine_wiring(tmp_path):
    db_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=db_path)

    kb.register_learning(
        architecture_family="transformer",
        hardware_type="NVIDIA A100",
        effective_config={
            "micro_batch_multiplier": 4.0,
            "inductor_mode": "max-autotune",
            "prefetch_factor": 8,
            "cuda_alloc_conf": "max_split_size_mb:512",
            "triton_epilogue_fusion": True,
        },
        throughput_improvement_pct=34.5,
        was_verified_safe=True,
    )

    predicted = kb.predict_hive_mind_config("transformer", "NVIDIA A100")
    assert predicted["micro_batch_multiplier"] == 4.0
    assert predicted["inductor_mode"] == "max-autotune"

    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    for s in range(1, 10):
        watcher.record_step(
            step=s,
            gpu_utilization_pct=45.0,
            gpu_memory_used_mb=14500.0,
            step_time_ms=200.0,
            loss=3.0 - (s * 0.01),
            data_loading_time_ms=40.0,
            mixed_precision="fp32",
            gradient_checkpointing=False,
            flash_attention=False,
            num_workers=0,
            pin_memory=False,
        )
    watcher.stop()
    summary = watcher.get_summary()

    engine = DecisionEngine(target_hardware="NVIDIA A100", knowledge_base=kb)
    recs = engine.evaluate(summary, model_type="transformer", knowledge_base=kb)

    assert len(recs) >= 5
    for r in recs:
        assert 0.50 <= r.confidence <= 0.99
        assert r.risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert len(r.evidence) > 0
        assert len(r.verification_plan) > 0
        assert len(r.rollback_plan) > 0

    hive_rec = [r for r in recs if r.rule_id == "RULE_DYNAMIC_HIVE_MIND_SURROGATE"][0]
    assert "micro_batch_multiplier=4.0" in hive_rec.description or "micro_batch_multiplier=4.0" in hive_rec.actionable_code_snippet
    assert "max-autotune" in hive_rec.description or "max-autotune" in hive_rec.actionable_code_snippet
    assert hive_rec.confidence > 0.85


def test_report_generator_renders_explainability_and_replay_log(tmp_path):
    db_path = str(tmp_path / "test_kb_report.json")
    kb = SharedKnowledgeBase(db_file_path=db_path)

    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    watcher.record_step(1, 50.0, 8000.0, 100.0, loss=2.0, mixed_precision="fp32")
    watcher.stop()
    summary = watcher.get_summary()

    engine = DecisionEngine(target_hardware="NVIDIA H100", knowledge_base=kb)
    recs = engine.evaluate(summary, model_type="transformer", knowledge_base=kb)

    replay_log = OptimizationReplayLog(session_id="test_report_replay")
    replay_log.record_event(
        step=1, stage="OBSERVE", recommendation_id="REC_01", action="RECORD_STEP", details={}, verified_safe=True
    )
    replay_log.record_event(
        step=2, stage="ROLLBACK", recommendation_id="RULE_MIXED_PRECISION", action="REVOKE_AUTO_APPLY",
        details={"reverted": True}, verified_safe=False, throughput_delta_pct=0.0, reason="Loss divergence detected"
    )

    roi = ROICalculator().calculate(100.0, 80.0, 1000, 8)

    markdown = ReportGenerator.generate_markdown(
        summary=summary,
        recommendations=recs,
        verifications=[],
        roi=roi,
        replay_log=replay_log,
        is_simulated=False,
    )

    assert "Decision Confidence:" in markdown
    assert "heuristic prior" in markdown
    assert "Risk Level:" in markdown
    assert "Evidence:" in markdown
    assert "Verification Plan:" in markdown
    assert "Rollback Plan:" in markdown
    assert "Optimization Replay & Audit Trail Log" in markdown
    assert "REVOKE_AUTO_APPLY" in markdown
    assert "Loss divergence detected" in markdown


def test_closed_loop_hook_rollback_execution(tmp_path):
    db_path = str(tmp_path / "test_kb_hook.json")
    kb = SharedKnowledgeBase(db_file_path=db_path)

    hook = GhostWatcherHook(gpu_memory_mb=16384.0, target_hardware="NVIDIA A100", knowledge_base=kb)

    with hook:
        hook.on_step_begin()
        hook.on_step_end(gpu_util_pct=50.0, gpu_mem_used_mb=8000.0, loss=2.0, mixed_precision="fp32")
        hook.on_step_begin()
        hook.on_step_end(gpu_util_pct=52.0, gpu_mem_used_mb=8100.0, loss=1.9, mixed_precision="fp32")

    divergent_losses = [2.0, 8.5]

    summary, recs, verifs, roi, md_report = hook.generate_full_report(
        model_type="transformer",
        was_auto_applied=True,
        optimized_losses=divergent_losses,
    )

    assert any(v.is_safe is False for v in verifs)
    replay_events = hook.replay_log.generate_audit_trail()
    rollback_events = [e for e in replay_events if e["stage"] == "ROLLBACK"]

    assert len(rollback_events) > 0
    assert rollback_events[0]["action"] == "REVOKE_AUTO_APPLY_AND_REVERT"
    assert "Runtime divergence threshold exceeded" in rollback_events[0]["reason"]

    entry = kb.query("transformer", "NVIDIA A100")
    assert entry is not None
    assert entry.rejected_sample_count >= 1
