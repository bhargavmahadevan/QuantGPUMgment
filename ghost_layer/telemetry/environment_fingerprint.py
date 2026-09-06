"""
Deterministic Environment Fingerprinting.

Computes multi-dimensional hardware, software, runtime, and workload hashes
to certify exact environment replication and evaluate cross-environment similarity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class HardwareSpec:
    device_name: str
    is_cuda: bool
    vram_total_mb: float
    device_uuid: Optional[str] = None
    cuda_compute_capability: Optional[str] = None
    sm_count: Optional[int] = None


@dataclass
class SoftwareSpec:
    pytorch_version: str
    cuda_runtime_version: str
    python_version: str
    os_platform: str
    driver_version: Optional[str] = None


@dataclass
class RuntimeSpec:
    allocator_conf: str = ""
    thread_count: int = 1
    amp_enabled: bool = False
    pin_memory: bool = False


@dataclass
class WorkloadSpec:
    model_family: str
    parameter_count: int
    sequence_length: int
    batch_size: int
    precision: str = "fp32"


@dataclass
class EnvironmentFingerprint:
    """
    Cryptographic composite environment fingerprint.
    Guarantees that trials compared in benchmarks share exact execution conditions.
    """
    hardware: HardwareSpec
    software: SoftwareSpec
    runtime: RuntimeSpec
    workload: WorkloadSpec

    hardware_hash: str = field(init=False)
    software_hash: str = field(init=False)
    runtime_hash: str = field(init=False)
    workload_hash: str = field(init=False)
    composite_hash: str = field(init=False)

    def __post_init__(self):
        self.hardware_hash = self._hash_dict(asdict(self.hardware))
        self.software_hash = self._hash_dict(asdict(self.software))
        self.runtime_hash = self._hash_dict(asdict(self.runtime))
        self.workload_hash = self._hash_dict(asdict(self.workload))

        composite_raw = f"{self.hardware_hash}:{self.software_hash}:{self.runtime_hash}:{self.workload_hash}"
        self.composite_hash = hashlib.sha256(composite_raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_dict(d: Dict[str, Any]) -> str:
        canonical_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def calculate_similarity(self, other: EnvironmentFingerprint) -> float:
        """
        Calculates similarity coefficient [0.0, 1.0] between two environments:
        Weights: Hardware 0.40, Software 0.20, Runtime 0.15, Workload 0.25.
        """
        score = 0.0

        # Hardware similarity
        if self.hardware_hash == other.hardware_hash:
            score += 0.40
        elif self.hardware.device_name.lower() == other.hardware.device_name.lower():
            score += 0.35
        elif self.hardware.is_cuda == other.hardware.is_cuda:
            score += 0.15

        # Software similarity
        if self.software_hash == other.software_hash:
            score += 0.20
        elif self.software.pytorch_version == other.software.pytorch_version:
            score += 0.15

        # Runtime similarity
        if self.runtime_hash == other.runtime_hash:
            score += 0.15
        elif self.runtime.amp_enabled == other.runtime.amp_enabled:
            score += 0.10

        # Workload similarity
        if self.workload_hash == other.workload_hash:
            score += 0.25
        elif (
            self.workload.model_family.lower() == other.workload.model_family.lower()
            and self.workload.parameter_count == other.workload.parameter_count
        ):
            score += 0.20

        return round(min(1.0, max(0.0, score)), 4)
