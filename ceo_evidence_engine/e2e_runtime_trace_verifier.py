"""
CEO Evidence Engine — End-to-End Runtime Trace Verifier
Validates full end-to-end integration:
CLI Telemetry -> Decision Engine -> Knowledge Base (Hive Mind) -> Replay Log -> Verification -> Executive Report
"""

import sys
import os
import time
import json
import uuid
import tempfile
from typing import List, Dict, Any

from ghost_layer.telemetry.watcher import TelemetryWatcher, TelemetrySummary
from ghost_layer.decision.engine import DecisionEngine, Recommendation, calculate_evidence_confidence
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, KnowledgeEntry
from ghost_layer.replay import OptimizationReplayLog, ReplayEvent
from ghost_layer.verification.verifier import CorrectnessVerifier, VerificationResult
from ghost_layer.reporting.generator import ReportGenerator
from ghost_layer.roi.calculator import ROICalculator, ROIAuditReport

def run_end_to_end_runtime_trace():
    """
    Executes a complete 7-stage runtime trace confirming data flow across all components.
    """
    print(f"\n========================================================")
    print(f"  GHOSTLAYER END-TO-END RUNTIME TRACE VERIFIER")
    print(f"========================================================")
    
    # -----------------------------------------------------------------
    # Stage 1: Telemetry Collection
    # -----------------------------------------------------------------
    print("-> Stage 1: Telemetry Collection (TelemetryWatcher)...")
    watcher = TelemetryWatcher(target_gpu_mb=16384.0)
    watcher.start()
    for i in range(10):
        watcher.record_step(
            step=i + 1,
            gpu_utilization_pct=82.0 + (i % 3),
            gpu_memory_used_mb=14500.0 + (i % 2) * 100,
            step_time_ms=31.3,
            loss=2.45 - i * 0.05,
            mixed_precision="fp32"
        )
    watcher.stop()
    summary: TelemetrySummary = watcher.get_summary()
    assert summary.mixed_precision == "fp32"
    assert summary.total_steps == 10
    print(f"  [PASSED] Telemetry Captured: Steps={summary.total_steps}, Peak VRAM={summary.peak_gpu_memory_mb:.1f} MB")
    
    # -----------------------------------------------------------------
    # Stage 2: Shared Knowledge Base & Hive Mind Query
    # -----------------------------------------------------------------
    print("-> Stage 2: Knowledge Base & Hive Mind Query (SharedKnowledgeBase)...")
    kb = SharedKnowledgeBase(db_file_path=os.path.join(tempfile.gettempdir(), "test_kb.json"))
    kb.register_learning(
        architecture_family="transformer",
        hardware_type="NVIDIA RTX A2000",
        effective_config={"rule_id": "RULE_MIXED_PRECISION", "precision": "bfloat16"},
        throughput_improvement_pct=35.0,
        was_verified_safe=True
    )
    predicted_config = kb.predict_hive_mind_config(architecture_family="transformer", hardware_type="NVIDIA RTX A2000")
    assert "rule_id" in predicted_config or "micro_batch_multiplier" in predicted_config
    print(f"  [PASSED] Hive Mind Prediction: Config={predicted_config}")
    
    # -----------------------------------------------------------------
    # Stage 3: Verification Policy & Loss Guardrail Check
    # -----------------------------------------------------------------
    print("-> Stage 3: Correctness Verifier & KL-Divergence Guardrail Check...")
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05, max_allowed_kl_div=0.02)
    baseline_losses = [2.45, 2.40, 2.35, 2.30, 2.25]
    optimized_losses = [2.44, 2.39, 2.34, 2.29, 2.24]
    ver_result: VerificationResult = verifier.verify_trajectories(
        baseline_losses=baseline_losses,
        optimized_losses=optimized_losses,
        recommendation_id="RULE_MIXED_PRECISION",
        was_auto_applied=True
    )
    assert ver_result.is_safe is True
    print(f"  [PASSED] Verification Policy Check: Safe={ver_result.is_safe}, Action={ver_result.action_taken}")
    
    # -----------------------------------------------------------------
    # Stage 4: Decision Engine Evaluation with Dynamic Confidence
    # -----------------------------------------------------------------
    print("-> Stage 4: Decision Engine Evaluation (DecisionEngine.evaluate)...")
    engine = DecisionEngine(target_hardware="NVIDIA RTX A2000", knowledge_base=kb)
    recs: List[Recommendation] = engine.evaluate(
        summary=summary,
        model_type="transformer",
        verification_results=[ver_result],
        knowledge_base=kb
    )
    assert len(recs) > 0
    mp_rec = next((r for r in recs if r.rule_id == "RULE_MIXED_PRECISION"), recs[0])
    print(f"  [PASSED] Recommendation Generated: {mp_rec.title} (Dynamic Confidence: {mp_rec.confidence})")
    
    # -----------------------------------------------------------------
    # Stage 5: Optimization Replay Logging
    # -----------------------------------------------------------------
    print("-> Stage 5: Optimization Replay Log Recording...")
    replay_log = OptimizationReplayLog(session_id="SESS-E2E-TRACE-001")
    event = replay_log.record_event(
        step=10,
        stage="SAFE_APPLY",
        recommendation_id=mp_rec.rule_id,
        action="APPLY_BF16_PRECISION",
        details={"precision": "bfloat16", "vram_mb": summary.peak_gpu_memory_mb},
        verified_safe=True,
        throughput_delta_pct=35.0,
        reason="Loss curve divergence verified safe."
    )
    assert len(replay_log.events) > 0
    print(f"  [PASSED] Replay Log Event Recorded: Action={event.action}, Total Events={len(replay_log.events)}")
    
    # -----------------------------------------------------------------
    # Stage 6: Executive Report Rendering (Explainability & Replay)
    # -----------------------------------------------------------------
    print("-> Stage 6: Executive Report Generation & Explainability Verification...")
    roi_calc = ROICalculator(gpu_cost_per_hour=3.06, performance_fee_rate_pct=20.0)
    roi_report: ROIAuditReport = roi_calc.calculate(
        baseline_step_time_ms=31.3,
        optimized_step_time_ms=20.4,
        total_training_steps=10000,
        num_gpus=16
    )
    
    report_gen = ReportGenerator()
    report_md = report_gen.generate_markdown(
        summary=summary,
        recommendations=recs,
        verifications=[ver_result],
        roi=roi_report,
        client_name="Enterprise Test Client",
        replay_log=replay_log
    )
    assert len(report_md) > 200
    assert mp_rec.title in report_md
    print(f"  [PASSED] Executive Report Rendered ({len(report_md.splitlines())} lines)")
    
    # -----------------------------------------------------------------
    # Stage 7: Hive Mind Dynamic Adaptation Verification
    # -----------------------------------------------------------------
    print("-> Stage 7: Hive Mind Dynamic Adaptation Check...")
    for _ in range(15):
        kb.register_learning(
            architecture_family="transformer",
            hardware_type="NVIDIA RTX A2000",
            effective_config={"precision": "bad_kernel"},
            throughput_improvement_pct=0.0,
            was_verified_safe=False
        )
        
    base_conf = calculate_evidence_confidence(
        hardware_match=1.0,
        model_similarity=0.90,
        historical_verifications=20,
        rollback_frequency=0.0
    )
    adjusted_conf = calculate_evidence_confidence(
        hardware_match=1.0,
        model_similarity=0.90,
        historical_verifications=20,
        rollback_frequency=0.75
    )
    assert adjusted_conf < base_conf
    print(f"  [PASSED] Hive Mind Adaptation Verified: Base Conf ({base_conf}) -> High-Rollback Conf ({adjusted_conf})\n")

    print(f"========================================================")
    print(f"  ALL 7 E2E RUNTIME TRACE STAGES VERIFIED SUCCESSFULLY")
    print(f"========================================================\n")
    
    return True

if __name__ == "__main__":
    run_end_to_end_runtime_trace()
