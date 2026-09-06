"""
Immutable Evidence Store Subsystem.

Creates, verifies, and manages auditable empirical experiment artifacts on disk:
evidence/
    experiments/
        EXP-XXX/
            metadata.json
            baseline.json
            treatment.json
            raw_metrics.json
            quality_metrics.json
            statistics.json
            telemetry.json
            experiment_record.json
            hashes.json
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from ghost_layer.experiments.schema import ExperimentRecord


@dataclass
class EvidenceArtifactPackage:
    experiment_id: str
    artifact_dir: str
    manifest_hash: str
    files_created: List[str]
    timestamp: float = field(default_factory=time.time)


class EvidenceStore:
    """
    Manages the persistent, cryptographically verified evidence artifact directory.
    """
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir) if root_dir else Path(os.getcwd()) / "evidence"
        self.experiments_dir = self.root_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def persist_experiment_evidence(
        self,
        record: ExperimentRecord,
        raw_telemetry: Optional[Dict[str, Any]] = None,
    ) -> EvidenceArtifactPackage:
        """
        Writes all 8 canonical evidence artifacts and computes the immutable SHA-256 manifest.
        """
        exp_dir = self.experiments_dir / record.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        files_written = []
        file_hashes = {}

        # 1. metadata.json
        meta_data = {
            "experiment_id": record.experiment_id,
            "timestamp": record.timestamp,
            "trial_design": record.trial_design,
            "warmup_steps_discarded": record.warmup_steps_discarded,
            "hardware": asdict(record.hardware),
            "workload": asdict(record.workload),
            "commit_hash": record.commit_hash,
            "record_hash": record.record_hash,
        }
        self._write_json(exp_dir / "metadata.json", meta_data, files_written, file_hashes)

        # 2. baseline.json
        self._write_json(exp_dir / "baseline.json", asdict(record.baseline_distribution), files_written, file_hashes)

        # 3. treatment.json
        self._write_json(exp_dir / "treatment.json", asdict(record.candidate_distribution), files_written, file_hashes)

        # 4. raw_metrics.json
        raw_metrics = {
            "baseline_step_times_ms": record.baseline_distribution.raw_step_times_ms,
            "candidate_step_times_ms": record.candidate_distribution.raw_step_times_ms,
        }
        self._write_json(exp_dir / "raw_metrics.json", raw_metrics, files_written, file_hashes)

        # 5. quality_metrics.json
        self._write_json(exp_dir / "quality_metrics.json", asdict(record.quality_safety), files_written, file_hashes)

        # 6. statistics.json
        self._write_json(exp_dir / "statistics.json", asdict(record.statistics), files_written, file_hashes)

        # 7. telemetry.json
        self._write_json(exp_dir / "telemetry.json", raw_telemetry or {}, files_written, file_hashes)

        # 8. experiment_record.json
        self._write_json(exp_dir / "experiment_record.json", record.to_dict(), files_written, file_hashes)

        # 9. hashes.json (Cryptographic Manifest)
        manifest_raw = json.dumps(file_hashes, sort_keys=True)
        manifest_hash = hashlib.sha256(manifest_raw.encode("utf-8")).hexdigest()
        file_hashes["_manifest_sha256"] = manifest_hash
        
        with open(exp_dir / "hashes.json", "wb") as f:
            f.write(json.dumps(file_hashes, indent=2).encode("utf-8"))
        files_written.append("hashes.json")

        return EvidenceArtifactPackage(
            experiment_id=record.experiment_id,
            artifact_dir=str(exp_dir),
            manifest_hash=manifest_hash,
            files_created=files_written,
        )

    def verify_experiment_integrity(self, experiment_id: str) -> Tuple[bool, str]:
        """
        Verifies that no artifact inside the experiment directory has been altered or tampered with.
        """
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            return False, f"Experiment directory '{experiment_id}' does not exist."

        hash_file = exp_dir / "hashes.json"
        if not hash_file.exists():
            return False, "Missing hashes.json manifest."

        with open(hash_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for filename, expected_hash in manifest.items():
            if filename.startswith("_"):
                continue
            target_file = exp_dir / filename
            if not target_file.exists():
                return False, f"Missing expected artifact: {filename}"
            
            with open(target_file, "rb") as tf:
                content = tf.read()
                current_hash = hashlib.sha256(content).hexdigest()
                if current_hash != expected_hash:
                    return False, f"Integrity check failed for {filename}: hash mismatch."

        return True, "All artifact hashes verified intact."

    def _write_json(self, path: Path, data: Any, files_written: List[str], file_hashes: Dict[str, str]) -> None:
        raw_bytes = json.dumps(data, indent=2).encode("utf-8")
        with open(path, "wb") as f:
            f.write(raw_bytes)
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        filename = path.name
        files_written.append(filename)
        file_hashes[filename] = file_hash
