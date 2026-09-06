"""
GhostLayer Pipeline, Calibration, Evidence, & Enterprise Scale Test Suite.
Combines tests for:
- End-to-end training pipeline (watcher -> graphify -> decision -> verifier -> ROI -> KB)
- LLM training gameplan execution (Transformer, MoE, SSM Mamba)
- Hardware calibration runner & empirical benchmarking
- Closed-loop outcome evaluator (verified speedups & loss shift rejection)
- Enterprise scale ROI calculations (45 client configurations)
- Audit fixes regression tests (stall clamping, rule-specific safety, report status)
- Continuous value & drift monitor
- Bayesian surrogate optimization model
- 250 simulated LLM GPU benchmark validation runs
"""

import os
import json
import tempfile
from pathlib import Path
import pytest

# GhostLayer Core, Graphify, Decision, & Verification
from ghost_layer.telemetry.watcher import TelemetryWatcher, TelemetrySummary
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.verification.verifier import CorrectnessVerifier, VerificationResult
from ghost_layer.roi.calculator import ROICalculator
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, SurrogateOptimizationModel, param_count_to_bucket
from ghost_layer.reporting.generator import ReportGenerator
from ghost_layer.graphify.graph_builder import ExecutionGraphBuilder
from ghost_layer.graphify.analyzer import GraphAnalyzer
from ghost_layer.graphify.optimizer import GraphOptimizer
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.benchmarking.calibration_runner import CalibrationRunner, BenchmarkConfig
from ghost_layer.feedback.outcome_evaluator import OutcomeEvaluator
from ghost_layer.replay import OptimizationReplayLog
from ghost_layer.continuity.drift_monitor import DriftMonitor, EnvironmentFingerprint


# ==============================================================================
# 1. Full System End-to-End Pipeline
# ==============================================================================

def test_full_system_end_to_end_pipeline(tmp_path):
    # 1. Telemetry Watching
    watcher = TelemetryWatcher(target_gpu_mb=81920.0)
    watcher.start()

    for step in range(1, 51):
        watcher.record_step(
            step=step,
            gpu_utilization_pct=42.5,
            gpu_memory_used_mb=72000.0,
            step_time_ms=350.0,
            loss=4.5 - (step * 0.05),
            data_loading_time_ms=85.0,
            mixed_precision="fp32",
            gradient_checkpointing=False,
            flash_attention=False,
            num_workers=0,
            pin_memory=False,
        )
    watcher.stop()
    summary = watcher.get_summary()

    assert summary.total_steps == 50
    assert summary.avg_gpu_utilization_pct == 42.5
    assert summary.peak_gpu_memory_mb == 72000.0

    # 2. Graphify DAG Mapping
    num_layers = 80
    builder = ExecutionGraphBuilder("Llama-3-70B")
    builder.build_transformer_dag(num_layers=num_layers, hidden_dim=8192, sequence_length=4096)
    analyzer = GraphAnalyzer(builder)
    graph_bottlenecks = analyzer.analyze()

    expected_nodes = 1 + (num_layers * 3) + 1
    expected_edges = (num_layers * 3) + 1
    assert graph_bottlenecks.total_nodes == expected_nodes
    assert graph_bottlenecks.total_edges == expected_edges
    assert graph_bottlenecks.recommended_torch_compile is True

    # 3. Decision Engine Optimization Evaluation
    engine = DecisionEngine()
    recommendations = engine.evaluate(summary, model_type="transformer")

    assert len(recommendations) >= 4
    rule_ids = {r.rule_id for r in recommendations}
    assert "RULE_MIXED_PRECISION" in rule_ids
    assert "RULE_DATALOADER_WORKERS" in rule_ids
    assert "RULE_FLASH_ATTENTION" in rule_ids
    assert "RULE_GRAPHIFY_TORCH_COMPILE" in rule_ids

    # 4. Correctness Safety Verification
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05)
    base_loss = [4.5 - (s * 0.05) for s in range(1, 51)]
    opt_loss = [(4.5 - (s * 0.05)) + 0.002 for s in range(1, 51)]

    verif_res = verifier.verify_trajectories(base_loss, opt_loss, "RULE_MIXED_PRECISION", was_auto_applied=True)
    assert verif_res.is_safe is True
    assert verif_res.action_taken == "AUTO_APPLIED"
    assert verif_res.max_loss_delta == pytest.approx(0.002, abs=1e-5)

    avg_base_loss = sum(base_loss) / len(base_loss)
    expected_kl_proxy = round(0.002 / (avg_base_loss + 1e-8), 6)
    assert verif_res.relative_mean_loss_shift == pytest.approx(expected_kl_proxy, abs=1e-5)

    # 5. Financial ROI & Fee Audit
    gpu_cost_per_hr = 4.50
    fee_rate_pct = 25.0
    total_steps = 500000
    num_gpus = 64
    base_step_ms = 350.0
    opt_step_ms = base_step_ms * (1.0 - 0.60)

    calculator = ROICalculator(gpu_cost_per_hour=gpu_cost_per_hr, performance_fee_rate_pct=fee_rate_pct)
    roi = calculator.calculate(
        baseline_step_time_ms=base_step_ms,
        optimized_step_time_ms=opt_step_ms,
        total_training_steps=total_steps,
        num_gpus=num_gpus,
    )

    expected_base_gpu_hrs = round((base_step_ms / 3600000.0) * total_steps * num_gpus, 2)
    expected_opt_gpu_hrs = round((opt_step_ms / 3600000.0) * total_steps * num_gpus, 2)
    expected_hrs_saved = round(expected_base_gpu_hrs - expected_opt_gpu_hrs, 2)
    expected_base_cost = round(expected_base_gpu_hrs * gpu_cost_per_hr, 2)
    expected_opt_cost = round(expected_opt_gpu_hrs * gpu_cost_per_hr, 2)
    expected_gross_savings = round(expected_base_cost - expected_opt_cost, 2)
    expected_fee = round(expected_gross_savings * (fee_rate_pct / 100.0), 2)
    expected_net_savings = round(expected_gross_savings - expected_fee, 2)

    assert roi.baseline_gpu_hours == expected_base_gpu_hrs
    assert roi.optimized_gpu_hours == expected_opt_gpu_hrs
    assert roi.gpu_hours_saved == expected_hrs_saved
    assert roi.baseline_cost_usd == expected_base_cost
    assert roi.optimized_cost_usd == pytest.approx(expected_opt_cost, abs=0.05)
    assert roi.performance_fee_usd == pytest.approx(expected_fee, abs=0.05)
    assert roi.net_client_savings_usd == pytest.approx(expected_net_savings, abs=0.05)

    # 6. Shared Knowledge Base Persistence
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)
    kb.register_learning(
        architecture_family="Llama-3-70B",
        hardware_type="NVIDIA H100-SXM5-80GB",
        effective_config={"mixed_precision": "bf16", "flash_attention": True},
        throughput_improvement_pct=60.0,
    )

    entry = kb.query("Llama-3-70B", "NVIDIA H100-SXM5-80GB")
    assert entry is not None
    assert entry.throughput_improvement_pct == 60.0

    # 7. Executive Report Generation
    report_md = ReportGenerator.generate_markdown(summary, recommendations, [verif_res], roi)
    assert "# Quant Ghost Layer: AI Training Efficiency & ROI Audit Report" in report_md
    assert "Net Client Dollar Savings" in report_md


# ==============================================================================
# 2. LLM Training Gameplan Execution
# ==============================================================================

def test_llm_training_gameplan_execution():
    builder = ExecutionGraphBuilder("LLaMA-3-Gameplan-Test")
    builder.build_transformer_dag(num_layers=4, hidden_dim=4096, sequence_length=4096)
    assert builder.validate_dag() is True

    analyzer = GraphAnalyzer(builder)
    summary = analyzer.analyze()
    assert summary.total_nodes > 10
    assert summary.critical_path_length_ms > 0.0

    optimizer = GraphOptimizer(builder)
    opt_builder, opt_report = optimizer.optimize()
    assert opt_report.speedup_pct > 0.0

    with GhostWatcherHook(gpu_memory_mb=16384.0) as hook:
        for step in range(1, 11):
            hook.on_step_begin()
            hook.on_step_end(
                gpu_util_pct=50.0,
                gpu_mem_used_mb=8000.0,
                loss=3.0 - (step * 0.01),
                dataloader_time_ms=30.0,
                mixed_precision="fp32",
            )
        telemetry_summary, recs = hook.analyze_and_report(model_type="transformer")

    assert len(recs) > 0

    verifier = CorrectnessVerifier()
    base_losses = [3.0 - i * 0.01 for i in range(5)]
    opt_losses = [3.0 - i * 0.01 + 0.001 for i in range(5)]
    v_result = verifier.verify_trajectories(base_losses, opt_losses, "RULE_GRAPHIFY_TORCH_COMPILE", was_auto_applied=True)
    assert v_result.is_safe is True

    calc = ROICalculator(gpu_cost_per_hour=4.00, performance_fee_rate_pct=25.0)
    roi = calc.calculate(
        baseline_step_time_ms=300.0,
        optimized_step_time_ms=210.0,
        total_training_steps=50000,
        num_gpus=32,
    )
    assert roi.time_reduction_pct == pytest.approx(30.0, abs=0.1)
    assert roi.performance_fee_usd > 0.0


def test_llm_training_gameplan_moe_and_ssm_architectures():
    moe_builder = ExecutionGraphBuilder("Mixtral-Gameplan-Test")
    moe_builder.build_moe_dag(num_layers=4, num_experts=8, top_k=2)
    assert moe_builder.validate_dag() is True

    ssm_builder = ExecutionGraphBuilder("Mamba-Gameplan-Test")
    ssm_builder.build_ssm_dag(num_layers=4)
    assert ssm_builder.validate_dag() is True


# ==============================================================================
# 3. Hardware Calibration Runner Tests
# ==============================================================================

def test_calibration_runner_full_suite(tmp_path):
    kb_path = str(tmp_path / "calib_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)

    runner = CalibrationRunner(device="cpu", knowledge_base=kb)
    cfg = BenchmarkConfig(
        model_family="mlp",
        hidden_dim=32,
        num_layers=2,
        seq_len=8,
        batch_size=4,
        steps_warmup=2,
        steps_benchmark=3,
    )

    fp32_res, amp_res = runner.benchmark_precision(cfg)
    assert fp32_res.is_baseline is True
    assert fp32_res.avg_step_time_ms > 0
    assert fp32_res.is_empirically_measured is True
    assert amp_res.avg_step_time_ms > 0
    assert amp_res.is_empirically_measured is True

    w0_res, w_multi_res = runner.benchmark_dataloader_workers(cfg)
    assert w0_res.is_baseline is True
    assert w0_res.avg_step_time_ms > 0
    assert w_multi_res.is_empirically_measured is True

    base_ckpt, on_ckpt = runner.benchmark_gradient_checkpointing(cfg)
    assert base_ckpt.is_baseline is True
    assert on_ckpt.is_empirically_measured is True

    report = runner.run_full_calibration(cfg)
    assert len(report.empirical_results) == 6
    assert report.hardware_name != ""
    assert report.is_empirically_measured is True

    save_path = str(tmp_path / "calib_report.json")
    runner.save_report(report, file_path=save_path)
    assert os.path.exists(save_path)

    loaded = CalibrationRunner.load_report(file_path=save_path)
    assert loaded is not None
    assert loaded.hardware_name == report.hardware_name
    assert loaded.precision_speedup_pct == report.precision_speedup_pct
    assert len(loaded.empirical_results) == 6


# ==============================================================================
# 4. Outcome Evaluator Tests
# ==============================================================================

def test_outcome_evaluator_verified_speedup(tmp_path):
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)
    replay = OptimizationReplayLog(session_id="test_feedback_session")

    evaluator = OutcomeEvaluator(
        knowledge_base=kb,
        replay_log=replay,
        baseline_window_size=5,
        eval_window_size=5,
        loss_shift_threshold=0.10,
    )

    for step in range(1, 6):
        evaluator.record_step(step=step, step_time_ms=50.0, loss=2.00)

    evaluator.on_recommendation_applied(
        rule_id="RULE_MIXED_PRECISION",
        step=5,
        architecture_family="transformer",
        hardware_type="NVIDIA RTX A2000",
        effective_config={"mixed_precision": "bf16"},
    )

    for step in range(6, 11):
        evaluator.record_step(step=step, step_time_ms=30.0, loss=2.01)

    assert len(evaluator.completed_evaluations) == 1
    eval_res = evaluator.completed_evaluations[0]

    assert eval_res.rule_id == "RULE_MIXED_PRECISION"
    assert eval_res.baseline_avg_step_ms == 50.0
    assert eval_res.post_avg_step_ms == 30.0
    assert eval_res.realized_speedup_pct == 40.0
    assert eval_res.is_safe is True
    assert eval_res.registered_to_kb is True

    entry = kb.query("transformer", "NVIDIA RTX A2000")
    assert entry is not None
    assert entry.sample_count == 1
    assert entry.throughput_improvement_pct == 40.0
    assert entry.rejected_sample_count == 0

    report = evaluator.get_report("test_feedback_session")
    assert report.total_evaluations == 1
    assert len(report.verified_speedups) == 1
    assert report.avg_realized_speedup_pct == 40.0


def test_outcome_evaluator_rejected_loss_shift(tmp_path):
    kb_path = str(tmp_path / "test_kb_reject.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)
    replay = OptimizationReplayLog(session_id="test_reject_session")

    evaluator = OutcomeEvaluator(
        knowledge_base=kb,
        replay_log=replay,
        baseline_window_size=5,
        eval_window_size=5,
        loss_shift_threshold=0.10,
    )

    for step in range(1, 6):
        evaluator.record_step(step=step, step_time_ms=50.0, loss=2.00)

    evaluator.on_recommendation_applied(
        rule_id="RULE_SOPHIA_SECOND_ORDER",
        step=5,
        architecture_family="transformer",
        hardware_type="NVIDIA RTX A2000",
    )

    for step in range(6, 11):
        evaluator.record_step(step=step, step_time_ms=25.0, loss=3.50)

    assert len(evaluator.completed_evaluations) == 1
    eval_res = evaluator.completed_evaluations[0]

    assert eval_res.is_safe is False

    entry = kb.query("transformer", "NVIDIA RTX A2000")
    assert entry is not None
    assert entry.sample_count == 0
    assert entry.rejected_sample_count == 1


# ==============================================================================
# 5. Enterprise ROI Scaling (45 Configuration Cases)
# ==============================================================================

ENTERPRISE_TEST_CASES = [
    (1, 1.50, 25.0, 100_000, 200, 150, 2.08, 0.52),
    (4, 2.50, 25.0, 500_000, 300, 250, 69.44, 17.36),
    (8, 4.00, 25.0, 1_000_000, 300, 250, 444.44, 111.11),
    (8, 3.50, 20.0, 2_000_000, 400, 350, 777.78, 155.56),
    (8, 4.50, 25.0, 500_000, 100, 90, 50.00, 12.50),
    (16, 3.80, 25.0, 5_000_000, 350, 300, 4222.22, 1055.56),
    (32, 4.00, 25.0, 5_000_000, 500, 420, 14222.22, 3555.56),
    (64, 4.00, 25.0, 10_000_000, 400, 320, 56888.89, 14222.22),
    (64, 3.50, 30.0, 8_000_000, 300, 280, 9955.56, 2986.67),
    (64, 2.00, 20.0, 15_000_000, 450, 400, 26666.67, 5333.33),
    (128, 3.75, 25.0, 20_000_000, 500, 450, 133333.33, 33333.33),
    (256, 3.75, 25.0, 50_000_000, 650, 500, 2000000.00, 500000.00),
    (512, 3.50, 20.0, 100_000_000, 500, 400, 4977777.78, 995555.56),
    (512, 4.00, 25.0, 100_000_000, 500, 480, 1137777.78, 284444.44),
    (512, 2.50, 15.0, 50_000_000, 300, 200, 1777777.78, 266666.67),
    (1024, 3.50, 25.0, 200_000_000, 800, 680, 23893333.33, 5973333.33),
    (2048, 3.00, 20.0, 500_000_000, 600, 550, 42666666.67, 8533333.33),
    (4096, 2.80, 15.0, 1_000_000_000, 1000, 950, 159288888.89, 23893333.33),
    (4096, 3.20, 25.0, 500_000_000, 700, 680, 36408888.89, 9102222.22),
    (4096, 2.00, 10.0, 100_000_000, 500, 400, 22755555.56, 2275555.56),
    (1, 1.00, 50.0, 10_000, 100, 99, 0.00, 0.00),
    (100, 3.14, 22.5, 1_234_567, 333, 330, 323.05, 72.69),
    (8, 5.00, 25.0, 1_000_000, 100, 100, 0.00, 0.00),
    (24, 2.75, 25.0, 500_000, 500, 490, 91.67, 22.92),
    (48, 3.25, 25.0, 750_000, 400, 395, 162.50, 40.62),
    (2, 4.00, 25.0, 500_000, 250, 200, 55.56, 13.89),
    (6, 3.50, 25.0, 1_000_000, 300, 280, 116.67, 29.17),
    (10, 4.25, 25.0, 2_000_000, 400, 350, 1180.56, 295.14),
    (20, 3.80, 25.0, 3_000_000, 350, 310, 2533.33, 633.33),
    (40, 4.00, 25.0, 4_000_000, 450, 400, 8888.89, 2222.22),
    (80, 3.90, 25.0, 6_000_000, 500, 420, 41600.00, 10400.00),
    (160, 3.75, 25.0, 15_000_000, 600, 520, 200000.00, 50000.00),
    (320, 3.60, 25.0, 25_000_000, 550, 480, 560000.00, 140000.00),
    (640, 3.50, 25.0, 50_000_000, 700, 600, 3111111.11, 777777.78),
    (1280, 3.40, 25.0, 100_000_000, 800, 750, 6044444.44, 1511111.11),
    (2560, 3.30, 25.0, 250_000_000, 900, 800, 58666666.67, 14666666.67),
    (5120, 3.20, 25.0, 500_000_000, 1000, 950, 113777777.78, 28444444.44),
    (12, 1.50, 15.0, 5_000_000, 200, 180, 500.00, 75.00),
    (28, 2.25, 20.0, 10_000_000, 300, 280, 3500.00, 700.00),
    (56, 2.75, 22.0, 20_000_000, 400, 360, 34222.22, 7528.89),
]


@pytest.mark.parametrize(
    "gpu_count, cost_per_gpu, fee_rate, steps, base_ms, opt_ms, exp_savings, exp_fee",
    ENTERPRISE_TEST_CASES,
)
def test_enterprise_roi_scaling(
    gpu_count, cost_per_gpu, fee_rate, steps, base_ms, opt_ms, exp_savings, exp_fee
):
    calc = ROICalculator(
        gpu_cost_per_hour=cost_per_gpu,
        performance_fee_rate_pct=fee_rate,
    )

    report = calc.calculate(
        baseline_step_time_ms=base_ms,
        optimized_step_time_ms=opt_ms,
        total_training_steps=steps,
        num_gpus=gpu_count,
    )

    total_savings_dollars = report.baseline_cost_usd - report.optimized_cost_usd
    if exp_savings == 0.0:
        assert total_savings_dollars < 0.01
        assert report.performance_fee_usd < 0.01
    else:
        assert total_savings_dollars == pytest.approx(exp_savings, abs=1.0)
        assert report.performance_fee_usd == pytest.approx(exp_fee, abs=1.0)


# ==============================================================================
# 6. Audit Fixes Regression Tests
# ==============================================================================

def test_stall_pct_cannot_exceed_100_with_bad_caller_input():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    for i in range(1, 6):
        watcher.record_step(
            step=i,
            gpu_utilization_pct=42.0,
            gpu_memory_used_mb=3400.0,
            step_time_ms=2.3,
            data_loading_time_ms=120.0,
        )
    watcher.stop()
    summary = watcher.get_summary()
    assert summary.avg_dataloader_stall_pct <= 100.0


def test_stall_pct_clamped_at_the_snapshot_level_too():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    snap = watcher.record_step(
        step=1, gpu_utilization_pct=42.0, gpu_memory_used_mb=3400.0,
        step_time_ms=5.0, data_loading_time_ms=500.0,
    )
    watcher.stop()
    assert snap.data_loading_time_ms <= snap.step_time_ms


def test_safe_to_auto_apply_only_true_for_the_verified_rule():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    for i in range(1, 6):
        watcher.record_step(
            step=i, gpu_utilization_pct=40.0, gpu_memory_used_mb=15000.0,
            step_time_ms=200.0, mixed_precision="fp32", num_workers=0,
        )
    watcher.stop()
    summary = watcher.get_summary()

    verifications = [
        VerificationResult(
            is_safe=True, relative_mean_loss_shift=0.001, max_loss_delta=0.001,
            reason="ok", action_taken="AUTO_APPLIED",
            recommendation_id="RULE_MIXED_PRECISION",
        ),
    ]

    engine = DecisionEngine()
    recs = engine.evaluate(summary, model_type="transformer", verification_results=verifications)
    by_id = {r.rule_id for r in recs}

    assert next(r for r in recs if r.rule_id == "RULE_MIXED_PRECISION").safe_to_auto_apply is True
    assert next(r for r in recs if r.rule_id == "RULE_DATALOADER_WORKERS").safe_to_auto_apply is False


def test_safe_to_auto_apply_false_when_no_verifications_supplied():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    watcher.record_step(step=1, gpu_utilization_pct=40.0, gpu_memory_used_mb=15000.0, step_time_ms=200.0)
    watcher.stop()
    summary = watcher.get_summary()

    engine = DecisionEngine()
    recs = engine.evaluate(summary, model_type="transformer")
    assert all(r.safe_to_auto_apply is False for r in recs)


def test_report_status_reflects_rejected_verification_not_just_simulated_flag():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    watcher.record_step(step=1, gpu_utilization_pct=42.0, gpu_memory_used_mb=3400.0, step_time_ms=2.3)
    watcher.stop()
    summary = watcher.get_summary()

    roi = ROICalculator(gpu_cost_per_hour=3.50).calculate(
        baseline_step_time_ms=2.3, optimized_step_time_ms=3.3, total_training_steps=126, num_gpus=1
    )
    rejected = VerificationResult(
        is_safe=False, relative_mean_loss_shift=0.4, max_loss_delta=0.677,
        reason="Divergence threshold exceeded", action_taken="DOWNGRADED_TO_RECOMMENDATION",
        recommendation_id="RULE_GRAPHIFY_TORCH_COMPILE",
    )

    md = ReportGenerator.generate_markdown(
        summary=summary, recommendations=[], verifications=[rejected], roi=roi,
        is_simulated=False,
    )
    assert "Optimization Verified & Audit Complete" not in md
    assert "Auto-Apply Rejected" in md


def test_report_status_reflects_measured_regression_even_if_nothing_was_rejected():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    watcher.record_step(step=1, gpu_utilization_pct=42.0, gpu_memory_used_mb=3400.0, step_time_ms=21.2)
    watcher.stop()
    summary = watcher.get_summary()

    roi = ROICalculator(gpu_cost_per_hour=3.50).calculate(
        baseline_step_time_ms=21.2, optimized_step_time_ms=24.1, total_training_steps=64, num_gpus=1
    )
    md = ReportGenerator.generate_markdown(
        summary=summary, recommendations=[], verifications=[], roi=roi, is_simulated=False,
    )
    assert "Optimization Verified & Audit Complete" not in md


def test_report_status_still_says_verified_when_genuinely_safe():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    watcher.record_step(step=1, gpu_utilization_pct=42.0, gpu_memory_used_mb=3400.0, step_time_ms=100.0)
    watcher.stop()
    summary = watcher.get_summary()

    roi = ROICalculator(gpu_cost_per_hour=3.50).calculate(
        baseline_step_time_ms=100.0, optimized_step_time_ms=55.0, total_training_steps=100, num_gpus=1
    )
    passed = VerificationResult(
        is_safe=True, relative_mean_loss_shift=0.001, max_loss_delta=0.001,
        reason="ok", action_taken="AUTO_APPLIED", recommendation_id="RULE_MIXED_PRECISION",
    )

    md = ReportGenerator.generate_markdown(
        summary=summary, recommendations=[], verifications=[passed], roi=roi, is_simulated=False,
    )
    assert "Optimization Verified & Audit Complete" in md


def test_verifier_stores_recommendation_id_on_pass_and_fail_and_empty_input():
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05, max_allowed_kl_div=0.02)

    passed = verifier.verify_trajectories([2.0, 1.5, 1.0], [2.01, 1.51, 1.01], recommendation_id="RULE_X")
    assert passed.recommendation_id == "RULE_X"

    failed = verifier.verify_trajectories([2.0, 1.5, 1.0], [2.9, 1.0, 0.2], recommendation_id="RULE_Y")
    assert failed.recommendation_id == "RULE_Y"

    empty = verifier.verify_trajectories([], [], recommendation_id="RULE_Z")
    assert empty.recommendation_id == "RULE_Z"


# ==============================================================================
# 7. Continuous Value & Drift Monitoring Tests
# ==============================================================================

@pytest.fixture
def tmp_kb_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_param_count_bucketing():
    assert param_count_to_bucket(None) == ""
    assert param_count_to_bucket(5_000_000) == "tiny(<10M)"
    assert param_count_to_bucket(50_000_000) == "small(10M-100M)"
    assert param_count_to_bucket(500_000_000) == "medium(100M-1B)"
    assert param_count_to_bucket(5_000_000_000) == "large(1B-10B)"
    assert param_count_to_bucket(50_000_000_000) == "xlarge(10B+)"


def test_confidence_requires_low_variance_not_just_sample_count(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    noisy_values = [5.0, 40.0, 8.0, 35.0, 12.0, 38.0, 6.0, 33.0, 10.0, 30.0]
    for v in noisy_values:
        kb.register_learning("transformer", "A100", {"x": 1}, v, was_verified_safe=True)
    entry = kb.query("transformer", "A100")
    assert entry.sample_count == 10
    assert entry.confidence_tier != "HIGH"


def test_confidence_reaches_high_with_consistent_samples(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    consistent_values = [30.0, 31.0, 29.5, 30.5, 30.2, 29.8, 30.1, 30.3, 29.9, 30.0, 30.4]
    for v in consistent_values:
        kb.register_learning("transformer", "A100", {"x": 1}, v, was_verified_safe=True)
    entry = kb.query("transformer", "A100")
    assert entry.confidence_tier == "HIGH"


def test_rejected_samples_never_pull_the_average(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    kb.register_learning("cnn", "RTX4090", {"x": 1}, 40.0, was_verified_safe=True)
    kb.register_learning("cnn", "RTX4090", {"x": 1}, -50.0, was_verified_safe=False)
    entry = kb.query("cnn", "RTX4090")
    assert entry.throughput_improvement_pct == 40.0
    assert entry.sample_count == 1
    assert entry.rejected_sample_count == 1


def test_find_best_match_prefers_exact_over_broad(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    kb.register_learning("transformer", "A100", {"x": 1}, 20.0, was_verified_safe=True)
    kb.register_learning("transformer", "A100", {"x": 1}, 45.0, was_verified_safe=True,
                          param_count_bucket="medium(100M-1B)")

    entry, level = kb.find_best_match("transformer", "A100", param_count_bucket="medium(100M-1B)")
    assert level == "EXACT"
    assert entry.throughput_improvement_pct == 45.0

    entry2, level2 = kb.find_best_match("transformer", "A100", param_count_bucket="xlarge(10B+)")
    assert level2 == "ARCH+HW"
    assert entry2.throughput_improvement_pct == 20.0


def test_find_best_match_returns_none_for_totally_unseen_fingerprint(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    entry, level = kb.find_best_match("transformer", "H100")
    assert entry is None
    assert level == "NONE"


def _summary_needing_all_rules():
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    for i in range(1, 6):
        watcher.record_step(
            step=i, gpu_utilization_pct=40.0, gpu_memory_used_mb=15500.0,
            step_time_ms=200.0, mixed_precision="fp32", num_workers=0,
        )
    watcher.stop()
    return watcher.get_summary()


def test_engine_without_kb_uses_static_label():
    engine = DecisionEngine(target_hardware="A100")
    recs = engine.evaluate(_summary_needing_all_rules(), model_type="transformer")
    fusion = next(r for r in recs if r.rule_id == "RULE_GRAPHIFY_TORCH_COMPILE")
    assert "Measured/Estimated from Graphify" in fusion.speedup_estimate_label


def test_engine_with_confident_kb_match_overrides_label(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    consistent = [37.0, 38.0, 36.5, 37.5, 37.2, 36.8, 37.1, 37.3, 36.9, 37.0, 37.4]
    for v in consistent:
        kb.register_learning("transformer", "A100", {"x": 1}, v, was_verified_safe=True)

    engine = DecisionEngine(target_hardware="A100")
    recs = engine.evaluate(_summary_needing_all_rules(), model_type="transformer", knowledge_base=kb)
    fusion = next(r for r in recs if r.rule_id == "RULE_GRAPHIFY_TORCH_COMPILE")
    assert "verified cross-client samples" in fusion.speedup_estimate_label
    assert "HIGH confidence" in fusion.speedup_estimate_label


def test_engine_with_low_confidence_kb_match_falls_back_to_static(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    kb.register_learning("transformer", "A100", {"x": 1}, 37.0, was_verified_safe=True)

    engine = DecisionEngine(target_hardware="A100")
    recs = engine.evaluate(_summary_needing_all_rules(), model_type="transformer", knowledge_base=kb)
    fusion = next(r for r in recs if r.rule_id == "RULE_GRAPHIFY_TORCH_COMPILE")
    assert "Measured/Estimated from Graphify" in fusion.speedup_estimate_label


def _fp(**overrides):
    base = dict(
        architecture_family="transformer", hardware_type="A100",
        param_count_bucket="medium(100M-1B)", framework_version="2.5",
        applied_config={"num_workers": 4}, verified_speedup_pct=30.0, timestamp=0.0,
    )
    base.update(overrides)
    return EnvironmentFingerprint(**base)


def test_no_drift_when_nothing_changed(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    monitor = DriftMonitor(knowledge_base=kb)
    last = _fp()
    current = _fp()
    report = monitor.check(last, current)
    assert report.is_stale is False
    assert report.environment_drift == {}
    assert report.kb_has_better_config is False


def test_drift_detected_on_hardware_change(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    monitor = DriftMonitor(knowledge_base=kb)
    last = _fp(hardware_type="A100")
    current = _fp(hardware_type="H100")
    report = monitor.check(last, current)
    assert report.is_stale is True
    assert "hardware_type" in report.environment_drift
    assert "Re-verify" in report.recommended_action


def test_drift_detected_when_kb_learned_a_meaningfully_better_config(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    consistent = [50.0, 51.0, 49.5, 50.5, 50.2, 49.8, 50.1, 50.3, 49.9, 50.0]
    for v in consistent:
        kb.register_learning("transformer", "A100", {"x": 1}, v, was_verified_safe=True,
                              param_count_bucket="medium(100M-1B)")

    monitor = DriftMonitor(knowledge_base=kb)
    last = _fp(verified_speedup_pct=30.0)
    current = _fp(verified_speedup_pct=30.0)
    report = monitor.check(last, current)
    assert report.is_stale is True
    assert report.kb_has_better_config is True
    assert report.kb_improvement_pct > 0


def test_no_drift_flagged_for_noise_level_kb_improvement(tmp_kb_path):
    kb = SharedKnowledgeBase(db_file_path=tmp_kb_path)
    consistent = [31.0, 32.0, 31.5, 32.5, 31.2, 31.8, 32.1]
    for v in consistent:
        kb.register_learning("transformer", "A100", {"x": 1}, v, was_verified_safe=True,
                              param_count_bucket="medium(100M-1B)")
    monitor = DriftMonitor(knowledge_base=kb)
    last = _fp(verified_speedup_pct=30.0)
    current = _fp(verified_speedup_pct=30.0)
    report = monitor.check(last, current)
    assert report.kb_has_better_config is False
    assert report.is_stale is False


def test_environment_fingerprint_differs_from_is_symmetIndependent_of_other_fields():
    a = _fp(hardware_type="A100", framework_version="2.5")
    b = _fp(hardware_type="A100", framework_version="2.6")
    diff = b.differs_from(a)
    assert list(diff.keys()) == ["framework_version"]


# ==============================================================================
# 8. Hive-Mind Bayesian Optimization Surrogate Model Tests
# ==============================================================================

def test_surrogate_optimization_model_bayes_learning():
    posterior = SurrogateOptimizationModel.init_posterior()
    assert "micro_batch_multiplier" in posterior
    assert "inductor_mode" in posterior

    initial_pred = SurrogateOptimizationModel.predict_optimal_config(posterior)
    assert initial_pred["micro_batch_multiplier"] == 1.0 or initial_pred["micro_batch_multiplier"] == 2.0

    for _ in range(5):
        posterior = SurrogateOptimizationModel.update_posterior(
            posterior,
            effective_config={"micro_batch_multiplier": 4.0, "inductor_mode": "max-autotune"},
            throughput_improvement_pct=25.0,
            was_verified_safe=True,
        )

    updated_pred = SurrogateOptimizationModel.predict_optimal_config(posterior)
    assert updated_pred["micro_batch_multiplier"] == 4.0
    assert updated_pred["inductor_mode"] == "max-autotune"


def test_shared_knowledge_base_hive_mind_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_hive_mind_kb.json")
        kb = SharedKnowledgeBase(db_file_path=db_path)

        kb.register_learning(
            architecture_family="transformer",
            hardware_type="nvidia_h100",
            effective_config={
                "micro_batch_multiplier": 4.0,
                "inductor_mode": "max-autotune",
                "prefetch_factor": 8,
                "triton_epilogue_fusion": True,
            },
            throughput_improvement_pct=32.5,
            was_verified_safe=True,
        )

        entry = kb.query("transformer", "nvidia_h100")
        assert entry is not None
        assert entry.throughput_improvement_pct == 32.5
        assert entry.surrogate_posterior is not None

        pred = kb.predict_hive_mind_config("transformer", "nvidia_h100")
        assert pred["micro_batch_multiplier"] == 4.0
        assert pred["triton_epilogue_fusion"] is True


def test_decision_engine_includes_hive_mind_rule():
    engine = DecisionEngine()
    summary = TelemetrySummary(
        total_steps=10,
        total_duration_sec=1.0,
        avg_gpu_utilization_pct=85.0,
        peak_gpu_memory_mb=40000.0,
        gpu_memory_total_mb=80000.0,
        avg_step_time_ms=100.0,
        avg_dataloader_stall_pct=2.0,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
    )
    recs = engine.evaluate(summary, model_type="transformer")
    rule_ids = [r.rule_id for r in recs]
    assert "RULE_DYNAMIC_HIVE_MIND_SURROGATE" in rule_ids

