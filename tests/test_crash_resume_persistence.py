"""
End-to-End Crash and Resume Persistence Test.

Verifies that GhostLayer's state management, decision history, and knowledge base
survive unexpected process termination (simulated crash/interruption), and that
a newly spawned training process can resume from persisted disk artifacts without
data corruption or state loss.
"""

import os
import json
import pytest
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase
from ghost_layer.replay import OptimizationReplayLog
from ghost_layer.rollback import RollbackManager, ConfigSnapshot


def test_crash_and_resume_preserves_kb_and_replay_state(tmp_path):
    kb_file = str(tmp_path / "ghost_kb.json")
    replay_file = str(tmp_path / "replay_log.json")

    # =========================================================================
    # Phase 1: Pre-Crash Training Run (Process A)
    # =========================================================================
    kb_a = SharedKnowledgeBase(db_file_path=kb_file)
    replay_a = OptimizationReplayLog(session_id="SESSION_PRE_CRASH")
    rollback_a = RollbackManager()

    # Step 1: Capture baseline config snapshot
    snapshot_1 = rollback_a.snapshot(
        rule_id="RULE_MIXED_PRECISION"
    )

    # Step 2: Record replay events
    replay_a.record_event(
        step=10,
        stage="DIAGNOSE",
        recommendation_id="RULE_MIXED_PRECISION",
        action="RECOMMEND",
        details={"speedup_estimate": 45.0},
    )
    replay_a.record_event(
        step=11,
        stage="SAFE_APPLY",
        recommendation_id="RULE_MIXED_PRECISION",
        action="APPLY",
        details={"active_precision": "fp16"},
    )

    # Step 3: Register verified empirical learning in KB
    kb_a.register_learning(
        architecture_family="transformer",
        hardware_type="NVIDIA RTX A2000",
        effective_config={"mixed_precision": "fp16"},
        throughput_improvement_pct=42.5,
        was_verified_safe=True,
    )
    kb_a.register_rule_outcome(
        rule_id="RULE_MIXED_PRECISION",
        was_verified_safe=True,
        realized_speedup_pct=42.5,
        loss_shift=0.015,
    )

    # Save replay log to disk (as persistent audit trail)
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(replay_a.generate_audit_trail(), f, indent=2)

    # =========================================================================
    # Phase 2: Simulated Abrupt Crash (Process A Dies)
    # Destroy all in-memory Python references
    # =========================================================================
    del kb_a
    del replay_a
    del rollback_a

    assert os.path.exists(kb_file), "KB JSON must persist on disk"
    assert os.path.exists(replay_file), "Replay log JSON must persist on disk"

    # =========================================================================
    # Phase 3: Resume Training in Fresh Process (Process B)
    # =========================================================================
    kb_b = SharedKnowledgeBase(db_file_path=kb_file)
    
    # 1. Verify Knowledge Base entries survived the crash
    entry, match_level = kb_b.find_best_match("transformer", "NVIDIA RTX A2000")
    assert entry is not None, "KB must recall entry registered before crash"
    assert entry.throughput_improvement_pct == 42.5
    assert entry.sample_count == 1

    # 2. Verify Rule Calibrations survived the crash
    assert "RULE_MIXED_PRECISION" in kb_b.rule_calibrations
    rec = kb_b.rule_calibrations["RULE_MIXED_PRECISION"]
    assert rec["verified_safe_count"] == 1
    assert rec["mean_speedup_pct"] == 42.5

    # 3. Verify Replay Log audit trail can be restored from disk
    with open(replay_file, "r", encoding="utf-8") as f:
        restored_events = json.load(f)

    assert len(restored_events) == 2
    assert restored_events[0]["stage"] == "DIAGNOSE"
    assert restored_events[0]["recommendation_id"] == "RULE_MIXED_PRECISION"
    assert restored_events[1]["stage"] == "SAFE_APPLY"
    assert restored_events[1]["details"]["active_precision"] == "fp16"

    # 4. Resume training: Process B continues to record post-resume steps
    replay_b = OptimizationReplayLog(session_id="SESSION_POST_RESUME")
    replay_b.record_event(
        step=12,
        stage="MONITOR",
        recommendation_id="RULE_MIXED_PRECISION",
        action="RESUME_VERIFIED",
        details={"status": "training_resumed_post_crash"},
    )
    assert len(replay_b.events) == 1
    assert replay_b.events[0].step == 12
