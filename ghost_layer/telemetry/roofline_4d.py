"""
4D Roofline Telemetry & Hyperplane Bottleneck Separator for GhostLayer.

Lifts training step telemetry from ambiguous 1D/2D metrics into a 4D state vector:
    v = [Arithmetic Intensity (FLOP/Byte),
         DRAM Bandwidth Saturation (%),
         Host-to-Device Launch Latency (ms),
         SM Tensor Core Occupancy (%)]

Uses separating hyperplanes in 4D space to deterministically classify execution bottlenecks:
- COMPUTE_TENSOR_CORE_BOUND: High FLOP/Byte + high occupancy -> FlashAttention / FP8 GEMM.
- MEMORY_BANDWIDTH_BOUND: Low FLOP/Byte + high DRAM saturation -> torch.compile / Triton fusion.
- HOST_DEVICE_LAUNCH_BOUND: High launch latency + low GPU active time -> CUDA Graphs.
- DATALOADER_IO_BOUND: High CPU-GPU transfer stall + low compute -> DataLoader pin_memory / num_workers.
- BALANCED_OPTIMAL: Operating near theoretical hardware efficiency envelope.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import math


class BottleneckClass(str, Enum):
    COMPUTE_TENSOR_CORE_BOUND = "COMPUTE_TENSOR_CORE_BOUND"
    MEMORY_BANDWIDTH_BOUND = "MEMORY_BANDWIDTH_BOUND"
    HOST_DEVICE_LAUNCH_BOUND = "HOST_DEVICE_LAUNCH_BOUND"
    DATALOADER_IO_BOUND = "DATALOADER_IO_BOUND"
    BALANCED_OPTIMAL = "BALANCED_OPTIMAL"


@dataclass
class TelemetryVector4D:
    """4-Dimensional Hardware Telemetry State Vector."""
    arithmetic_intensity: float       # FLOPs per Byte transferred (e.g. 15.0 - 150.0)
    dram_bandwidth_pct: float         # % DRAM memory bus saturation (0.0 - 100.0)
    host_device_latency_ms: float     # Launch / synchronization overhead (ms)
    tensor_core_occupancy_pct: float  # SM Tensor Core active warp pipeline % (0.0 - 100.0)

    def to_vector(self) -> List[float]:
        """Returns the raw 4D coordinate vector."""
        return [
            float(self.arithmetic_intensity),
            float(self.dram_bandwidth_pct),
            float(self.host_device_latency_ms),
            float(self.tensor_core_occupancy_pct),
        ]

    @classmethod
    def from_step_data(
        cls,
        tflops: float,
        memory_bandwidth_gbps: float,
        dram_peak_gbps: float = 288.0,  # e.g. RTX A2000 default
        step_latency_ms: float = 10.0,
        cuda_active_ms: float = 8.5,
        sm_occupancy_pct: float = 65.0,
    ) -> "TelemetryVector4D":
        """
        Constructs a 4D Telemetry Vector from raw step measurements.
        """
        # Arithmetic Intensity: (Total FLOPs) / (Total Bytes transferred)
        effective_bytes = max(1e-6, memory_bandwidth_gbps * 1e9 * (cuda_active_ms / 1000.0))
        total_flops = max(1e-6, tflops * 1e12 * (cuda_active_ms / 1000.0))
        arithmetic_intensity = total_flops / effective_bytes

        # DRAM Bandwidth %
        dram_pct = min(100.0, max(0.0, (memory_bandwidth_gbps / max(1.0, dram_peak_gbps)) * 100.0))

        # Host-to-device launch latency
        launch_latency_ms = max(0.0, step_latency_ms - cuda_active_ms)

        # SM Tensor Core pipeline occupancy
        tensor_core_occupancy = min(100.0, max(0.0, sm_occupancy_pct))

        return cls(
            arithmetic_intensity=round(arithmetic_intensity, 4),
            dram_bandwidth_pct=round(dram_pct, 2),
            host_device_latency_ms=round(launch_latency_ms, 4),
            tensor_core_occupancy_pct=round(tensor_core_occupancy, 2),
        )


@dataclass
class ClassificationResult:
    """Result of 4D Roofline Polytope Classification."""
    primary_bottleneck: BottleneckClass
    confidence_margin: float
    recommended_action: str
    target_rule_id: str
    vector: TelemetryVector4D
    dimension_scores: Dict[str, float]
    rationale: str


class Roofline4DClassifier:
    """
    Separating Hyperplane & Polytope Classifier in 4D Telemetry Space.
    """

    def __init__(
        self,
        ai_knee_threshold: float = 40.0,       # Arithmetic intensity transition threshold
        dram_saturation_threshold: float = 75.0, # High DRAM pressure threshold %
        latency_stall_threshold_ms: float = 1.5, # CPU launch overhead threshold ms
        tensor_occupancy_threshold: float = 60.0, # SM saturation threshold %
    ):
        self.ai_knee = ai_knee_threshold
        self.dram_threshold = dram_saturation_threshold
        self.latency_threshold = latency_stall_threshold_ms
        self.occupancy_threshold = tensor_occupancy_threshold

    def classify(self, vec: TelemetryVector4D) -> ClassificationResult:
        """
        Evaluates the 4D Telemetry Vector against separating hyperplanes.
        """
        # Distance metrics from decision boundaries
        d_ai = (vec.arithmetic_intensity - self.ai_knee) / max(1.0, self.ai_knee)
        d_dram = (vec.dram_bandwidth_pct - self.dram_threshold) / 100.0
        d_lat = (vec.host_device_latency_ms - self.latency_threshold) / max(1.0, self.latency_threshold)
        d_occ = (vec.tensor_core_occupancy_pct - self.occupancy_threshold) / 100.0

        scores = {
            "compute_intensity_score": round(d_ai, 4),
            "dram_pressure_score": round(d_dram, 4),
            "launch_latency_score": round(d_lat, 4),
            "tensor_occupancy_score": round(d_occ, 4),
        }

        # Priority 1: Launch / Host-to-Device Latency Bound
        if vec.host_device_latency_ms > self.latency_threshold and vec.tensor_core_occupancy_pct < 50.0:
            margin = min(1.0, max(0.1, d_lat))
            return ClassificationResult(
                primary_bottleneck=BottleneckClass.HOST_DEVICE_LAUNCH_BOUND,
                confidence_margin=round(margin, 3),
                recommended_action="Enable CUDA Graphs or non-blocking async CUDA streams to eliminate host dispatch overhead.",
                target_rule_id="GHOST_RULE_CUDA_GRAPHS",
                vector=vec,
                dimension_scores=scores,
                rationale=f"Host launch latency ({vec.host_device_latency_ms:.2f}ms) exceeds threshold ({self.latency_threshold}ms) with low SM occupancy ({vec.tensor_core_occupancy_pct:.1f}%).",
            )

        # Priority 2: Memory Bandwidth Saturation (Low FLOPs/Byte + High DRAM %)
        if vec.dram_bandwidth_pct >= self.dram_threshold and vec.arithmetic_intensity < self.ai_knee:
            margin = min(1.0, max(0.1, d_dram - d_ai))
            return ClassificationResult(
                primary_bottleneck=BottleneckClass.MEMORY_BANDWIDTH_BOUND,
                confidence_margin=round(margin, 3),
                recommended_action="Apply kernel fusion (torch.compile or Triton FlashAttention) to reduce DRAM read/write roundtrips.",
                target_rule_id="GHOST_RULE_TORCH_COMPILE_FUSION",
                vector=vec,
                dimension_scores=scores,
                rationale=f"DRAM bandwidth saturation is high ({vec.dram_bandwidth_pct:.1f}%) with sub-critical arithmetic intensity ({vec.arithmetic_intensity:.2f} FLOP/Byte < {self.ai_knee}).",
            )

        # Priority 3: Compute / Tensor Core Pipeline Bound (High FLOPs/Byte + High Occupancy)
        if vec.arithmetic_intensity >= self.ai_knee and vec.tensor_core_occupancy_pct >= self.occupancy_threshold:
            margin = min(1.0, max(0.1, d_ai + d_occ))
            return ClassificationResult(
                primary_bottleneck=BottleneckClass.COMPUTE_TENSOR_CORE_BOUND,
                confidence_margin=round(margin, 3),
                recommended_action="Transition matrix multiplications to FP8 GEMM / scaled Mixed Precision or Hybrid Muon optimizer.",
                target_rule_id="GHOST_RULE_MIXED_PRECISION_FP8",
                vector=vec,
                dimension_scores=scores,
                rationale=f"Tensor core compute pipelines saturated ({vec.tensor_core_occupancy_pct:.1f}%) at high arithmetic intensity ({vec.arithmetic_intensity:.2f} FLOP/Byte).",
            )

        # Priority 4: DataLoader / Transfer I/O Bound
        if vec.tensor_core_occupancy_pct < 30.0 and vec.dram_bandwidth_pct < 30.0 and vec.host_device_latency_ms <= self.latency_threshold:
            return ClassificationResult(
                primary_bottleneck=BottleneckClass.DATALOADER_IO_BOUND,
                confidence_margin=0.85,
                recommended_action="Configure DataLoader with pin_memory=True, num_workers=4+, and prefetch_factor=2.",
                target_rule_id="GHOST_RULE_DATALOADER_OPTIMIZATION",
                vector=vec,
                dimension_scores=scores,
                rationale="Both compute occupancy and memory bandwidth are severely underutilized, indicating input pipeline starvation.",
            )

        # Default: Balanced Optimal
        return ClassificationResult(
            primary_bottleneck=BottleneckClass.BALANCED_OPTIMAL,
            confidence_margin=0.90,
            recommended_action="System operating within balanced efficiency envelope. Maintain current execution configuration.",
            target_rule_id="GHOST_RULE_BALANCED",
            vector=vec,
            dimension_scores=scores,
            rationale=f"Balanced operation: AI={vec.arithmetic_intensity:.2f}, DRAM={vec.dram_bandwidth_pct:.1f}%, Latency={vec.host_device_latency_ms:.2f}ms, Occupancy={vec.tensor_core_occupancy_pct:.1f}%.",
        )

    def classify_trajectory(self, vectors: List[TelemetryVector4D]) -> Dict[str, Any]:
        """Classifies a multi-step sequence of 4D telemetry vectors."""
        if not vectors:
            return {
                "dominant_bottleneck": BottleneckClass.BALANCED_OPTIMAL.value,
                "distribution": {},
                "mean_vector": None,
                "step_count": 0,
            }

        results = [self.classify(v) for v in vectors]
        counts: Dict[str, int] = {}
        for r in results:
            k = r.primary_bottleneck.value
            counts[k] = counts.get(k, 0) + 1

        dominant = max(counts.items(), key=lambda x: x[1])[0]

        # Mean vector
        mean_ai = sum(v.arithmetic_intensity for v in vectors) / len(vectors)
        mean_dram = sum(v.dram_bandwidth_pct for v in vectors) / len(vectors)
        mean_lat = sum(v.host_device_latency_ms for v in vectors) / len(vectors)
        mean_occ = sum(v.tensor_core_occupancy_pct for v in vectors) / len(vectors)

        mean_vec = TelemetryVector4D(
            arithmetic_intensity=round(mean_ai, 4),
            dram_bandwidth_pct=round(mean_dram, 2),
            host_device_latency_ms=round(mean_lat, 4),
            tensor_core_occupancy_pct=round(mean_occ, 2),
        )

        return {
            "dominant_bottleneck": dominant,
            "distribution": {k: round(v / len(vectors), 3) for k, v in counts.items()},
            "mean_vector": mean_vec.to_vector(),
            "step_count": len(vectors),
            "dominant_action": next(r.recommended_action for r in results if r.primary_bottleneck.value == dominant),
        }
