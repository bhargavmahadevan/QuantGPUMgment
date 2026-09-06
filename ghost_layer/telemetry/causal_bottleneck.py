"""
Evidence-Based Runtime Bottleneck Diagnosis Engine.

Performs observational root-cause classification based on real hardware and runtime telemetry:
Identifies what physical resource is constraining workload throughput before proposing
targeted structured interventions.

Scientific distinction: Observational telemetry classifies the bottleneck candidate;
true causality is formally established exclusively through the subsequent counterbalanced
A/B intervention and statistical verification.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class BottleneckType(enum.Enum):
    COMPUTE_BOUND = "COMPUTE_BOUND"                      # SM ALU / Tensor Core saturation
    MEMORY_BANDWIDTH_BOUND = "MEMORY_BANDWIDTH_BOUND"    # High DRAM traffic, low arithmetic intensity
    MEMORY_CAPACITY_BOUND = "MEMORY_CAPACITY_BOUND"      # Near VRAM ceiling / OOM risk
    INPUT_BOUND = "INPUT_BOUND"                          # Host DataLoader I/O or PCIe copy stalls
    COMMUNICATION_BOUND = "COMMUNICATION_BOUND"          # AllReduce / NCCL inter-node latency
    SYNCHRONIZATION_BOUND = "SYNCHRONIZATION_BOUND"      # CUDA device-to-host sync stalls (.item(), print)
    KERNEL_LAUNCH_BOUND = "KERNEL_LAUNCH_BOUND"          # CPU-bound Python launch overhead on tiny kernels
    COMPILATION_BOUND = "COMPILATION_BOUND"              # Inductor graph breaks / JIT compilation thrashing


@dataclass
class BottleneckDiagnosis:
    primary_bottleneck: BottleneckType
    secondary_bottleneck: Optional[BottleneckType]
    confidence_score: float
    evidence_metrics: Dict[str, Any]
    recommended_rules: List[str]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_bottleneck": self.primary_bottleneck.value,
            "secondary_bottleneck": self.secondary_bottleneck.value if self.secondary_bottleneck else None,
            "confidence_score": round(self.confidence_score, 2),
            "evidence_metrics": self.evidence_metrics,
            "recommended_rules": self.recommended_rules,
            "rationale": self.rationale,
        }


class CausalBottleneckDetector:
    """
    Diagnoses the true mechanical bottleneck limiting GPU training throughput.
    """

    @classmethod
    def diagnose(
        cls,
        gpu_utilization_pct: float,
        step_time_ms: float,
        data_loading_time_ms: float = 0.0,
        gpu_memory_used_mb: float = 0.0,
        total_vram_mb: float = 16384.0,
        arithmetic_intensity: Optional[float] = None,
        cpu_overhead_ms: float = 0.0,
        comm_wait_ms: float = 0.0,
    ) -> BottleneckDiagnosis:
        evidence = {
            "gpu_utilization_pct": gpu_utilization_pct,
            "step_time_ms": step_time_ms,
            "data_loading_time_ms": data_loading_time_ms,
            "io_fraction": round(data_loading_time_ms / max(1e-6, step_time_ms), 3),
            "vram_fraction": round(gpu_memory_used_mb / max(1.0, total_vram_mb), 3),
            "arithmetic_intensity": arithmetic_intensity,
        }

        # 1. Check Input Bound (DataLoader Stalls)
        io_frac = data_loading_time_ms / max(1e-6, step_time_ms)
        if io_frac >= 0.15 or (data_loading_time_ms > 8.0 and gpu_utilization_pct < 75.0):
            return BottleneckDiagnosis(
                primary_bottleneck=BottleneckType.INPUT_BOUND,
                secondary_bottleneck=BottleneckType.MEMORY_BANDWIDTH_BOUND if gpu_utilization_pct < 60 else None,
                confidence_score=0.92,
                evidence_metrics=evidence,
                recommended_rules=["RULE_DATALOADER_WORKERS", "RULE_ASYNC_TENSOR_FLOW"],
                rationale=f"Host data pipeline stalls account for {io_frac:.1%} of step duration. GPU is starved of batches.",
            )

        # 2. Check Communication Bound (Distributed NCCL)
        comm_frac = comm_wait_ms / max(1e-6, step_time_ms)
        if comm_frac >= 0.20:
            return BottleneckDiagnosis(
                primary_bottleneck=BottleneckType.COMMUNICATION_BOUND,
                secondary_bottleneck=BottleneckType.COMPUTE_BOUND,
                confidence_score=0.88,
                evidence_metrics=evidence,
                recommended_rules=["RULE_GRADIENT_ACCUMULATION", "RULE_ASYNC_TENSOR_FLOW"],
                rationale=f"Inter-GPU communication collectives consume {comm_frac:.1%} of step time.",
            )

        # 3. Check Memory Capacity Bound (VRAM Pressure)
        vram_frac = gpu_memory_used_mb / max(1.0, total_vram_mb)
        if vram_frac >= 0.90:
            return BottleneckDiagnosis(
                primary_bottleneck=BottleneckType.MEMORY_CAPACITY_BOUND,
                secondary_bottleneck=BottleneckType.MEMORY_BANDWIDTH_BOUND,
                confidence_score=0.90,
                evidence_metrics=evidence,
                recommended_rules=["RULE_GRADIENT_CHECKPOINTING", "RULE_VRAM_HEADROOM", "RULE_ALLOCATOR_TUNING"],
                rationale=f"Allocated VRAM ({vram_frac:.1%}) is approaching hardware limits. Headroom buffer is constrained.",
            )

        # 4. Check Kernel Launch / CPU Sync Bound
        if gpu_utilization_pct < 60.0 and cpu_overhead_ms > (step_time_ms * 0.25):
            return BottleneckDiagnosis(
                primary_bottleneck=BottleneckType.KERNEL_LAUNCH_BOUND,
                secondary_bottleneck=BottleneckType.SYNCHRONIZATION_BOUND,
                confidence_score=0.85,
                evidence_metrics=evidence,
                recommended_rules=["RULE_TORCH_COMPILE", "RULE_ASYNC_TENSOR_FLOW"],
                rationale="High host CPU driver launch overhead with small GPU kernels. Operator fusion via torch.compile recommended.",
            )

        # 5. Check Memory Bandwidth Bound (Low Arithmetic Intensity)
        if (arithmetic_intensity is not None and arithmetic_intensity < 25.0) or (gpu_utilization_pct < 75.0 and vram_frac < 0.80):
            return BottleneckDiagnosis(
                primary_bottleneck=BottleneckType.MEMORY_BANDWIDTH_BOUND,
                secondary_bottleneck=BottleneckType.COMPUTE_BOUND,
                confidence_score=0.87,
                evidence_metrics=evidence,
                recommended_rules=["RULE_SDPA_ATTENTION", "RULE_FLASH_ATTENTION", "RULE_MIXED_PRECISION_FP16"],
                rationale="Workload is bound by GPU memory bus transfer speeds. Kernel fusion and fused attention will reduce DRAM roundtrips.",
            )

        # 6. Compute Bound (Default for high-occupancy saturated GPU)
        return BottleneckDiagnosis(
            primary_bottleneck=BottleneckType.COMPUTE_BOUND,
            secondary_bottleneck=None,
            confidence_score=0.95,
            evidence_metrics=evidence,
            recommended_rules=["RULE_MIXED_PRECISION_FP16", "RULE_BATCH_SCALING", "RULE_TORCH_COMPILE"],
            rationale="GPU Streaming Multiprocessors and Tensor Cores are compute-saturated. Tensor core precision or batch scaling recommended.",
        )
