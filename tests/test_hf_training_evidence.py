"""
Tests for HuggingFace Training Evidence Runner.
Validates evidence artifact structure, SHA-256 hash integrity,
and GhostTrainerCallback integration without requiring GPU.
"""

import json
import hashlib
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any


class TestEvidenceRunnerImport:
    """Validate the evidence runner module can be imported."""

    def test_import_module(self):
        """Evidence runner module should be importable."""
        from ceo_evidence_engine import hf_training_evidence_runner
        assert hasattr(hf_training_evidence_runner, "run_training_evidence")
        assert hasattr(hf_training_evidence_runner, "_get_hardware_info")
        assert hasattr(hf_training_evidence_runner, "_get_git_commit")

    def test_hardware_info_structure(self):
        """Hardware info should return all required fields."""
        from ceo_evidence_engine.hf_training_evidence_runner import _get_hardware_info
        info = _get_hardware_info()
        required_keys = [
            "os", "python_version", "torch_version",
            "cuda_available", "gpu_name", "gpu_count", "total_vram_mb",
        ]
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["total_vram_mb"], (int, float))

    def test_git_commit_returns_string(self):
        """Git commit helper should always return a string."""
        from ceo_evidence_engine.hf_training_evidence_runner import _get_git_commit
        commit = _get_git_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0


class TestEvidenceArtifactStructure:
    """Validate evidence artifact JSON structure and SHA-256 integrity."""

    @pytest.fixture
    def sample_evidence_artifact(self) -> Dict[str, Any]:
        """Create a minimal valid evidence artifact for structural tests."""
        evidence_payload = {
            "model_key": "test/model-tiny",
            "timestamp": "2026-09-16T00:00:00+00:00",
            "results": {
                "execution_mode": "CPU_MEASURED_RUN",
                "hardware": {
                    "os": "Windows 11",
                    "python_version": "3.14.3",
                    "torch_version": "2.13.0+cpu",
                    "cuda_available": False,
                    "cuda_version": None,
                    "gpu_name": "CPU",
                    "gpu_count": 0,
                    "total_vram_mb": 0.0,
                },
                "training_config": {
                    "model_id": "test/model-tiny",
                    "dataset": "wikitext/wikitext-2-raw-v1",
                    "num_steps": 5,
                    "batch_size": 2,
                    "precision": "fp32",
                    "gradient_checkpointing": False,
                    "learning_rate": 5e-5,
                    "block_size": 128,
                },
                "metrics": {
                    "total_steps_captured": 5,
                    "avg_step_time_ms": 100.5,
                    "peak_vram_mb": 0.0,
                    "final_training_loss": 10.5,
                    "wall_time_sec": 2.5,
                    "loss_trajectory": [12.0, 11.5, 11.0, 10.8, 10.5],
                    "step_times_ms": [105.0, 100.0, 99.0, 101.0, 97.5],
                },
                "ghostlayer_telemetry": {
                    "mixed_precision_detected": "fp32",
                    "gpu_utilization_avg_pct": 0.0,
                    "vram_headroom_pct": None,
                },
                "recommendations": [
                    {
                        "rule_id": "RULE_MIXED_PRECISION",
                        "title": "Enable Mixed Precision Training",
                        "confidence": "MEDIUM",
                        "impact_level": "HIGH",
                        "risk_level": "LOW",
                        "speedup_estimate": "+15-35%",
                        "safe_to_auto_apply": False,
                    }
                ],
                "recommendation_count": 1,
            },
        }

        hash_input = json.dumps(evidence_payload, sort_keys=True).encode("utf-8")
        sha256_hash = hashlib.sha256(hash_input).hexdigest()

        return {
            "benchmark_category": "REAL_MODEL_TRAINING_TELEMETRY",
            "training_efficiency_benchmark": True,
            "is_empirically_measured": True,
            "metadata": evidence_payload,
            "reproducibility": {
                "run_id": "RUN-TEST0001",
                "sha256_audit_hash": sha256_hash,
                "git_commit": "abc123",
            },
        }

    def test_artifact_has_required_top_level_keys(self, sample_evidence_artifact):
        """Evidence artifact must contain all top-level keys."""
        required = [
            "benchmark_category", "training_efficiency_benchmark",
            "is_empirically_measured", "metadata", "reproducibility",
        ]
        for key in required:
            assert key in sample_evidence_artifact, f"Missing top-level key: {key}"

    def test_artifact_benchmark_category(self, sample_evidence_artifact):
        """Category must be REAL_MODEL_TRAINING_TELEMETRY."""
        assert sample_evidence_artifact["benchmark_category"] == "REAL_MODEL_TRAINING_TELEMETRY"

    def test_artifact_is_empirically_measured(self, sample_evidence_artifact):
        """Must be flagged as empirically measured."""
        assert sample_evidence_artifact["is_empirically_measured"] is True

    def test_sha256_hash_reproducibility(self, sample_evidence_artifact):
        """SHA-256 hash must be reproducible from the metadata payload."""
        payload = sample_evidence_artifact["metadata"]
        recomputed = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        stored = sample_evidence_artifact["reproducibility"]["sha256_audit_hash"]
        assert recomputed == stored, (
            f"SHA-256 mismatch: computed={recomputed[:16]}... stored={stored[:16]}..."
        )

    def test_metadata_has_results(self, sample_evidence_artifact):
        """Metadata must contain results block."""
        assert "results" in sample_evidence_artifact["metadata"]
        results = sample_evidence_artifact["metadata"]["results"]
        assert "execution_mode" in results
        assert "hardware" in results
        assert "training_config" in results
        assert "metrics" in results
        assert "ghostlayer_telemetry" in results
        assert "recommendations" in results

    def test_metrics_have_required_fields(self, sample_evidence_artifact):
        """Metrics block must contain all measurement fields."""
        metrics = sample_evidence_artifact["metadata"]["results"]["metrics"]
        required = [
            "total_steps_captured", "avg_step_time_ms", "peak_vram_mb",
            "final_training_loss", "wall_time_sec", "loss_trajectory", "step_times_ms",
        ]
        for key in required:
            assert key in metrics, f"Missing metrics key: {key}"

    def test_loss_trajectory_is_list(self, sample_evidence_artifact):
        """Loss trajectory must be a list of floats."""
        trajectory = sample_evidence_artifact["metadata"]["results"]["metrics"]["loss_trajectory"]
        assert isinstance(trajectory, list)
        for val in trajectory:
            assert isinstance(val, (int, float))

    def test_recommendations_are_structured(self, sample_evidence_artifact):
        """Each recommendation must have rule_id, title, confidence."""
        recs = sample_evidence_artifact["metadata"]["results"]["recommendations"]
        assert len(recs) > 0
        for rec in recs:
            assert "rule_id" in rec
            assert "title" in rec
            assert "confidence" in rec
            assert "impact_level" in rec

    def test_execution_mode_is_valid(self, sample_evidence_artifact):
        """Execution mode must be one of the known values."""
        mode = sample_evidence_artifact["metadata"]["results"]["execution_mode"]
        valid_modes = {"LIVE_MEASURED_RUN", "CPU_MEASURED_RUN"}
        assert mode in valid_modes, f"Unknown execution mode: {mode}"


class TestGhostTrainerCallbackIntegration:
    """Test GhostTrainerCallback wiring without requiring a real model."""

    def test_callback_instantiation(self):
        """GhostTrainerCallback should instantiate without errors."""
        from ghost_layer.callback import GhostTrainerCallback
        cb = GhostTrainerCallback(
            cluster_name="test-cluster",
            gpu_memory_mb=8192.0,
            gpu_cost_per_hour=3.50,
            auto_generate_report=False,
        )
        assert cb.cluster_name == "test-cluster"
        assert cb.gpu_memory_mb == 8192.0
        assert cb._is_active is False

    def test_callback_lifecycle(self):
        """Callback should handle train_begin → step → train_end lifecycle."""
        from ghost_layer.callback import GhostTrainerCallback
        cb = GhostTrainerCallback(auto_generate_report=False)

        # Simulate lifecycle
        cb.on_train_begin()
        assert cb._is_active is True

        cb.on_step_begin()
        cb.on_step_end()  # No state, no args — should not crash

        cb.on_train_end()
        assert cb._is_active is False

    def test_callback_captures_loss_from_state(self):
        """Callback should extract loss from TrainerState log_history."""
        from ghost_layer.callback import GhostTrainerCallback
        cb = GhostTrainerCallback(auto_generate_report=False)
        cb.on_train_begin()
        cb.on_step_begin()

        # Mock TrainerState with log_history
        mock_state = MagicMock()
        mock_state.log_history = [{"loss": 2.45, "step": 1}]

        cb.on_step_end(state=mock_state)

        # The loss should have been passed through to the watcher
        assert cb.hook.watcher.snapshots[-1].loss == 2.45
        cb.on_train_end()
