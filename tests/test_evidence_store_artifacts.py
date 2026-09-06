"""
Tests for EvidenceStore & Immutable Artifact Package Integrity.
"""

import os
import json
import pytest
from pathlib import Path
from ghost_layer.evidence.store import EvidenceStore
from ghost_layer.experiments.schema import (
    ExperimentRecord,
    HardwareEnvironment,
    WorkloadProfile,
    InterventionSpec,
    LatencyDistribution,
    QualitySafetyMetrics,
    StatisticalAttribution,
    EconomicImpact,
)


def test_evidence_store_persists_all_8_artifacts(tmp_path):
    store = EvidenceStore(root_dir=str(tmp_path / "evidence"))

    rec = ExperimentRecord(
        experiment_id="EXP-001-TEST",
        hardware=HardwareEnvironment("NVIDIA RTX A2000", True, 12288.0),
        workload=WorkloadProfile("transformer", "llama-3-8b", 4, 4096),
        intervention=InterventionSpec("RULE_SDPA_ATTENTION", "TIER_A_PRODUCTION_SAFE", "attention"),
        baseline_distribution=LatencyDistribution(10, 100.0, 1.0, 100.0, 102.0, [100.0] * 10),
        candidate_distribution=LatencyDistribution(10, 85.0, 1.0, 85.0, 87.0, [85.0] * 10),
        quality_safety=QualitySafetyMetrics(0.001, 0.02, False),
        statistics=StatisticalAttribution("TOST Non-Inferiority", 15.0, (14.0, 16.0), 0.01, True, 1.5),
        economics=EconomicImpact(3.50, 15.0, 0.15, 0.525, "EMPIRICAL"),
    )

    pkg = store.persist_experiment_evidence(rec, raw_telemetry={"gpu_util": 88.5})
    assert pkg.experiment_id == "EXP-001-TEST"
    assert len(pkg.files_created) == 9  # 8 data artifacts + hashes.json
    assert (Path(pkg.artifact_dir) / "hashes.json").exists()

    # Verify cryptographic integrity
    is_valid, msg = store.verify_experiment_integrity("EXP-001-TEST")
    assert is_valid
    assert "verified intact" in msg


def test_evidence_store_detects_tampered_artifact(tmp_path):
    store = EvidenceStore(root_dir=str(tmp_path / "evidence"))

    rec = ExperimentRecord(
        experiment_id="EXP-002-TAMPER",
        hardware=HardwareEnvironment("NVIDIA A100", True, 81920.0),
        workload=WorkloadProfile("transformer", "mistral-7b", 8, 4096),
    )
    pkg = store.persist_experiment_evidence(rec)

    # Tamper with statistics.json
    stats_file = Path(pkg.artifact_dir) / "statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"tampered": True}))

    # Verification must fail
    is_valid, msg = store.verify_experiment_integrity("EXP-002-TAMPER")
    assert not is_valid
    assert "Integrity check failed" in msg
