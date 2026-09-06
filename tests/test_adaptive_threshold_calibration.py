"""
Unit tests for Empirically Closed-Loop Adaptive Threshold Calibration.

Verifies that:
1. When no empirical samples exist, default heuristic thresholds are returned with 'HEURISTIC_PRIOR'.
2. When high win-rate verified outcomes are registered, sensitivity thresholds expand toward max_bound.
3. When rejections / rollbacks are registered, sensitivity thresholds contract toward min_bound.
4. Thresholds are clamped strictly within [min_bound, max_bound].
5. OutcomeEvaluator registers rule outcomes into SharedKnowledgeBase.
6. DecisionEngine adapts evaluation thresholds dynamically based on empirical KB state.
"""

import os
import pytest
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase
from ghost_layer.feedback.outcome_evaluator import OutcomeEvaluator
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.telemetry.watcher import TelemetrySummary


def test_default_heuristic_prior_when_no_samples(tmp_path):
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)

    thresh, src = kb.get_calibrated_threshold(
        rule_id="RULE_SOPHIA_SECOND_ORDER",
        default_threshold=0.85,
        min_bound=0.75,
        max_bound=0.90,
    )
    assert thresh == 0.85
    assert src == "HEURISTIC_PRIOR"


def test_high_win_rate_expands_threshold(tmp_path):
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)

    # Register 4 successful runs with zero rejections (100% win-rate)
    for _ in range(4):
        kb.register_rule_outcome(
            rule_id="RULE_SOPHIA_SECOND_ORDER",
            was_verified_safe=True,
            realized_speedup_pct=35.0,
            loss_shift=0.01,
        )

    thresh, src = kb.get_calibrated_threshold(
        rule_id="RULE_SOPHIA_SECOND_ORDER",
        default_threshold=0.85,
        min_bound=0.75,
        max_bound=0.90,
    )
    # Threshold should have adapted upwards (increased sensitivity)
    assert thresh > 0.85
    assert thresh <= 0.90
    assert "EMPIRICALLY_ADAPTED" in src
    assert "win_rate=100.0%" in src


def test_rejections_tighten_threshold_conservatively(tmp_path):
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)

    # Register 1 safe run and 3 rejections (25% win rate)
    kb.register_rule_outcome(
        rule_id="RULE_SHAMPOO_PRECONDITIONING",
        was_verified_safe=True,
        realized_speedup_pct=15.0,
        loss_shift=0.03,
    )
    for _ in range(3):
        kb.register_rule_outcome(
            rule_id="RULE_SHAMPOO_PRECONDITIONING",
            was_verified_safe=False,
            realized_speedup_pct=-10.0,
            loss_shift=0.15,
        )

    thresh, src = kb.get_calibrated_threshold(
        rule_id="RULE_SHAMPOO_PRECONDITIONING",
        default_threshold=0.60,
        min_bound=0.50,
        max_bound=0.70,
    )
    # Threshold should contract (tighten) to protect training stability
    assert thresh < 0.60
    assert thresh >= 0.50
    assert "EMPIRICALLY_ADAPTED_CONSERVATIVE" in src
    assert "rejections=3" in src


def test_bounds_clamping_prevents_runaway(tmp_path):
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)

    # 50 successful runs
    for _ in range(50):
        kb.register_rule_outcome(
            rule_id="RULE_VARIANCE_CONTAINMENT",
            was_verified_safe=True,
            realized_speedup_pct=50.0,
            loss_shift=0.005,
        )

    thresh, _ = kb.get_calibrated_threshold(
        rule_id="RULE_VARIANCE_CONTAINMENT",
        default_threshold=0.70,
        min_bound=0.55,
        max_bound=0.75,
    )
    assert thresh == 0.75  # Clamped at max_bound


def test_outcome_evaluator_wires_to_kb_rule_calibrations(tmp_path):
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)
    evaluator = OutcomeEvaluator(knowledge_base=kb, baseline_window_size=3, eval_window_size=3)

    # Baseline steps
    evaluator.record_step(1, 100.0, 2.50)
    evaluator.record_step(2, 100.0, 2.48)
    evaluator.record_step(3, 100.0, 2.49)

    # Apply recommendation
    evaluator.on_recommendation_applied(
        rule_id="RULE_SOPHIA_SECOND_ORDER",
        step=3,
        architecture_family="transformer",
        hardware_type="NVIDIA RTX A2000",
    )

    # Post-apply steps (speedup ~30%, loss stable)
    evaluator.record_step(4, 70.0, 2.47)
    evaluator.record_step(5, 70.0, 2.46)
    evaluator.record_step(6, 70.0, 2.45)

    assert "RULE_SOPHIA_SECOND_ORDER" in kb.rule_calibrations
    rec = kb.rule_calibrations["RULE_SOPHIA_SECOND_ORDER"]
    assert rec["total_samples"] == 1
    assert rec["verified_safe_count"] == 1
    assert rec["mean_speedup_pct"] == 30.0


def test_decision_engine_evaluates_against_adapted_threshold(tmp_path):
    kb_path = str(tmp_path / "test_kb.json")
    kb = SharedKnowledgeBase(db_file_path=kb_path)
    engine = DecisionEngine(knowledge_base=kb)

    # Baseline summary with moderate loss stability index: 0.86
    # With default threshold 0.85, 0.86 < 0.85 is FALSE -> Sophia NOT triggered
    summary = TelemetrySummary(
        total_steps=50,
        total_duration_sec=15.0,
        avg_gpu_utilization_pct=85.0,
        peak_gpu_memory_mb=12000.0,
        gpu_memory_total_mb=24000.0,
        avg_step_time_ms=30.0,
        avg_dataloader_stall_pct=2.0,
        mixed_precision="fp16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
    )

    recs_unadapted = engine.evaluate(
        summary=summary,
        knowledge_base=kb,
        inverse_cost_weight=0.86,
    )
    sophia_recs = [r for r in recs_unadapted if r.rule_id == "RULE_SOPHIA_SECOND_ORDER"]
    assert len(sophia_recs) == 0

    # Now adapt threshold by registering 5 safe outcomes (threshold adapts above 0.86)
    for _ in range(5):
        kb.register_rule_outcome("RULE_SOPHIA_SECOND_ORDER", was_verified_safe=True, realized_speedup_pct=40.0)

    thresh, _ = kb.get_calibrated_threshold("RULE_SOPHIA_SECOND_ORDER", 0.85, 0.75, 0.90)
    assert thresh > 0.86

    # Re-evaluate with adapted threshold: now 0.86 < thresh is TRUE -> Sophia IS triggered!
    recs_adapted = engine.evaluate(
        summary=summary,
        knowledge_base=kb,
        inverse_cost_weight=0.86,
    )
    sophia_recs_adapted = [r for r in recs_adapted if r.rule_id == "RULE_SOPHIA_SECOND_ORDER"]
    assert len(sophia_recs_adapted) == 1
    assert "EMPIRICALLY_ADAPTED" in sophia_recs_adapted[0].evidence
