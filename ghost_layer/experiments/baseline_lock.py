"""
Cryptographic Baseline Lock.

Freezes and seals workload, data, hardware, framework, and pricing parameters
prior to optimization execution to ensure empirical attribution and prevent moving-baseline drift.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BaselineLock:
    """
    Immutable sealed baseline snapshot.
    Any deviation in comparison parameters invalidates comparative claims.
    """
    lock_id: str
    model_name: str
    total_parameters: int
    batch_size: int
    sequence_length: int
    token_count: int
    hardware_name: str
    gpu_hourly_cost_usd: float
    framework_version: str
    measurement_protocol: str
    created_at_utc: str
    lock_hash: str

    @classmethod
    def create(
        cls,
        lock_id: str,
        model_name: str,
        total_parameters: int,
        batch_size: int,
        sequence_length: int,
        token_count: int,
        hardware_name: str,
        gpu_hourly_cost_usd: float,
        framework_version: str = "PyTorch 2.5",
        measurement_protocol: str = "Counterbalanced ABAB with 5-step warmup discard",
    ) -> BaselineLock:
        now_iso = datetime.now(timezone.utc).isoformat()
        payload = {
            "lock_id": lock_id,
            "model_name": model_name,
            "total_parameters": total_parameters,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "token_count": token_count,
            "hardware_name": hardware_name,
            "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
            "framework_version": framework_version,
            "measurement_protocol": measurement_protocol,
            "created_at_utc": now_iso,
        }
        canonical_str = json.dumps(payload, sort_keys=True)
        lock_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        return cls(
            lock_id=lock_id,
            model_name=model_name,
            total_parameters=total_parameters,
            batch_size=batch_size,
            sequence_length=sequence_length,
            token_count=token_count,
            hardware_name=hardware_name,
            gpu_hourly_cost_usd=gpu_hourly_cost_usd,
            framework_version=framework_version,
            measurement_protocol=measurement_protocol,
            created_at_utc=now_iso,
            lock_hash=lock_hash,
        )

    def validate_candidate(self, candidate_meta: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        """
        Validates candidate execution metadata against the locked baseline.
        Returns: (is_valid, comparison_status, violations)
        """
        violations: List[str] = []

        if candidate_meta.get("model_name") and candidate_meta["model_name"] != self.model_name:
            violations.append(f"Model mismatch: {candidate_meta['model_name']} != {self.model_name}")

        if candidate_meta.get("batch_size") and candidate_meta["batch_size"] != self.batch_size:
            violations.append(f"Batch size mismatch: {candidate_meta['batch_size']} != {self.batch_size}")

        if candidate_meta.get("sequence_length") and candidate_meta["sequence_length"] != self.sequence_length:
            violations.append(f"Sequence length mismatch: {candidate_meta['sequence_length']} != {self.sequence_length}")

        if candidate_meta.get("hardware_name") and candidate_meta["hardware_name"] != self.hardware_name:
            violations.append(f"Hardware mismatch: {candidate_meta['hardware_name']} != {self.hardware_name}")

        is_valid = len(violations) == 0
        status = "COMPARISON_VALID" if is_valid else "COMPARISON_INVALID"

        return is_valid, status, violations
