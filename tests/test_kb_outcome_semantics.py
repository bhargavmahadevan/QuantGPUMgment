import pytest
import tempfile
import os
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, KnowledgeEntry, OutcomeStatus


def test_kb_outcome_status_verified_safe_updates_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kb.json")
        kb = SharedKnowledgeBase(db_file_path=db_path)

        # Register verified safe outcome
        entry = kb.register_learning(
            architecture_family="transformer",
            hardware_type="NVIDIA RTX A2000",
            effective_config={"mixed_precision": "fp16"},
            throughput_improvement_pct=50.0,
            was_verified_safe=True,
            outcome_status=OutcomeStatus.VERIFIED_SAFE,
        )

        assert entry is not None
        assert entry.sample_count == 1
        assert entry.rejected_sample_count == 0
        assert entry.throughput_improvement_pct == 50.0
        assert entry.outcome_status == "VERIFIED_SAFE"


def test_kb_outcome_status_rejected_does_not_contaminate_speedup():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kb.json")
        kb = SharedKnowledgeBase(db_file_path=db_path)

        # First entry: verified safe at 40% speedup
        kb.register_learning(
            architecture_family="llama",
            hardware_type="NVIDIA H100",
            effective_config={"mixed_precision": "bf16"},
            throughput_improvement_pct=40.0,
            was_verified_safe=True,
            outcome_status=OutcomeStatus.VERIFIED_NON_INFERIOR,
        )

        # Second entry: rejected / divergent trial claiming 90% speedup
        entry2 = kb.register_learning(
            architecture_family="llama",
            hardware_type="NVIDIA H100",
            effective_config={"mixed_precision": "bf16"},
            throughput_improvement_pct=90.0,
            was_verified_safe=False,
            outcome_status=OutcomeStatus.REJECTED,
        )

        assert entry2 is not None
        assert entry2.sample_count == 1  # Verified count stays 1
        assert entry2.rejected_sample_count == 1
        assert entry2.throughput_improvement_pct == 40.0  # Rejected 90% did NOT contaminate average!
        assert entry2.outcome_status == "REJECTED"


def test_kb_outcome_status_simulated_and_provenance():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kb.json")
        kb = SharedKnowledgeBase(db_file_path=db_path)

        entry = kb.register_learning(
            architecture_family="moe",
            hardware_type="NVIDIA B200",
            effective_config={"expert_parallelism": 8},
            throughput_improvement_pct=120.0,
            was_verified_safe=False,
            outcome_status=OutcomeStatus.SIMULATED,
            provenance_note="Simulated on cluster graph simulator",
        )

        assert entry is not None
        assert entry.outcome_status == "SIMULATED"
        assert entry.sample_count == 0
        assert entry.rejected_sample_count == 1
        assert entry.throughput_improvement_pct == 0.0
        assert "Simulated" in entry.provenance_note
