"""
16-Dimensional Latent Execution Signature Embeddings for GhostLayer.

Lifts complex training run telemetry into a normalized 16D latent embedding vector:
    z in S^{15} (Unit Hypersphere in R^{16})

Enables privacy-preserving cross-client transfer learning:
1. Embeds hardware execution metrics, memory profiles, and curvature dynamics without
   revealing client model weights, source code, or private dataset tokens.
2. Performs fast cosine / metric search across the global knowledge base to find
   the nearest empirically verified optimization recipe.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math


@dataclass
class ExecutionSignatureEmbedding:
    """
    16-Dimensional Normalized Execution Signature Vector.
    """
    # 0-3: Core 4D Roofline
    arithmetic_intensity_norm: float      # min(1.0, AI / 100.0)
    dram_bandwidth_saturation: float      # DRAM % / 100.0
    host_to_device_latency_ratio: float   # Launch stall / total step time
    tensor_core_occupancy_norm: float     # SM active warp % / 100.0

    # 4-7: Compute & Curvature Dynamics
    sm_utilization_ratio: float           # GPU compute utilization % / 100.0
    grad_norm_mean_log: float             # log10(1 + grad_norm) / 4.0
    loss_delta_snr: float                 # Welford SNR / 10.0
    birkhoff_drift_residual: float        # Residual drift / 1.0

    # 8-11: Memory & Pipeline
    vram_utilization_pct: float           # Allocated VRAM / Total VRAM
    kernel_launch_frequency_khz: float    # Kernel launches per ms / 10.0
    gemm_to_elementwise_ratio: float      # GEMM FLOPs / total FLOPs
    activation_memory_ratio: float        # Activation VRAM / Total VRAM

    # 12-15: Distributed & Efficiency
    forward_to_backward_ratio: float      # t_fwd / (t_fwd + t_bwd)
    dataloader_wait_ratio: float          # t_io / t_step
    effective_tflops_ratio: float         # Measured TFLOPs / Peak Hardware TFLOPs
    curvature_second_order_norm: float    # HVP curvature estimate / 10.0

    client_tag: str = "anonymous"
    model_family: str = "transformer"
    hardware_architecture: str = "NVIDIA"

    def to_raw_vector(self) -> List[float]:
        """Returns the unnormalized 16D list."""
        return [
            float(self.arithmetic_intensity_norm),
            float(self.dram_bandwidth_saturation),
            float(self.host_to_device_latency_ratio),
            float(self.tensor_core_occupancy_norm),
            float(self.sm_utilization_ratio),
            float(self.grad_norm_mean_log),
            float(self.loss_delta_snr),
            float(self.birkhoff_drift_residual),
            float(self.vram_utilization_pct),
            float(self.kernel_launch_frequency_khz),
            float(self.gemm_to_elementwise_ratio),
            float(self.activation_memory_ratio),
            float(self.forward_to_backward_ratio),
            float(self.dataloader_wait_ratio),
            float(self.effective_tflops_ratio),
            float(self.curvature_second_order_norm),
        ]

    def to_unit_hypersphere_embedding(self) -> List[float]:
        """Projects the 16D vector onto the unit hypersphere S^15 (L2 normalization)."""
        raw = self.to_raw_vector()
        norm = math.sqrt(sum(x * x for x in raw))
        if norm < 1e-9:
            norm = 1.0
        return [round(x / norm, 6) for x in raw]

    @classmethod
    def from_step_telemetry(
        cls,
        ai: float,
        dram_pct: float,
        launch_lat_ms: float,
        sm_occ_pct: float,
        step_time_ms: float,
        vram_allocated_gb: float,
        vram_total_gb: float,
        grad_norm: float = 1.0,
        loss_std: float = 0.05,
        tflops_measured: float = 15.0,
        tflops_peak: float = 60.0,
        client_tag: str = "anonymous",
        model_family: str = "transformer",
        hardware_architecture: str = "NVIDIA",
    ) -> "ExecutionSignatureEmbedding":
        """Builds a normalized 16D execution signature from observed telemetry."""
        step_t = max(1e-3, step_time_ms)
        vram_tot = max(1.0, vram_total_gb)

        return cls(
            arithmetic_intensity_norm=min(1.0, max(0.0, ai / 100.0)),
            dram_bandwidth_saturation=min(1.0, max(0.0, dram_pct / 100.0)),
            host_to_device_latency_ratio=min(1.0, max(0.0, launch_lat_ms / step_t)),
            tensor_core_occupancy_norm=min(1.0, max(0.0, sm_occ_pct / 100.0)),
            sm_utilization_ratio=min(1.0, max(0.0, sm_occ_pct / 100.0)),
            grad_norm_mean_log=min(1.0, max(0.0, math.log10(1.0 + grad_norm) / 4.0)),
            loss_delta_snr=min(1.0, max(0.0, 1.0 / (1.0 + 10.0 * loss_std))),
            birkhoff_drift_residual=0.0,
            vram_utilization_pct=min(1.0, max(0.0, vram_allocated_gb / vram_tot)),
            kernel_launch_frequency_khz=0.2,
            gemm_to_elementwise_ratio=0.75,
            activation_memory_ratio=0.45,
            forward_to_backward_ratio=0.33,
            dataloader_wait_ratio=min(1.0, max(0.0, max(0.0, step_t - 8.0) / step_t)),
            effective_tflops_ratio=min(1.0, max(0.0, tflops_measured / max(1.0, tflops_peak))),
            curvature_second_order_norm=0.1,
            client_tag=client_tag,
            model_family=model_family,
            hardware_architecture=hardware_architecture,
        )


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two high-dimensional vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return round(max(-1.0, min(1.0, dot / (norm_a * norm_b))), 6)


@dataclass
class MatchResult:
    """Match result against the Global Recipe Registry."""
    matched_recipe_id: str
    target_hardware: str
    similarity_score: float
    recommended_optimizations: List[str]
    is_high_confidence_match: bool


class ExecutionSignatureRegistry:
    """
    Privacy-Preserving Cross-Client Optimization Matcher.
    """

    def __init__(self) -> None:
        self.registry: List[Dict[str, Any]] = []
        self._populate_reference_benchmarks()

    def _populate_reference_benchmarks(self) -> None:
        """Populates canonical reference execution signatures for known workload profiles."""
        # 1. Memory-Bound Small Batch Profile (FlashAttention / Fusion)
        sig_mem = ExecutionSignatureEmbedding(
            arithmetic_intensity_norm=0.15,
            dram_bandwidth_saturation=0.90,
            host_to_device_latency_ratio=0.05,
            tensor_core_occupancy_norm=0.35,
            sm_utilization_ratio=0.40,
            grad_norm_mean_log=0.25,
            loss_delta_snr=0.85,
            birkhoff_drift_residual=0.0,
            vram_utilization_pct=0.60,
            kernel_launch_frequency_khz=0.8,
            gemm_to_elementwise_ratio=0.30,
            activation_memory_ratio=0.70,
            forward_to_backward_ratio=0.33,
            dataloader_wait_ratio=0.02,
            effective_tflops_ratio=0.25,
            curvature_second_order_norm=0.1,
            client_tag="kb_ref_memory_bound",
            model_family="transformer",
            hardware_architecture="NVIDIA",
        )
        self.register_recipe(
            recipe_id="RECIPE_FUSED_FLASH_ATTN",
            target_hardware="NVIDIA",
            signature=sig_mem,
            recommended_optimizations=[
                "Apply torch.compile(mode='reduce-overhead')",
                "Enable FlashAttention-2 kernels",
                "Use Chunked Cross-Entropy Loss",
            ],
        )

        # 2. Host Dispatch / Small Kernel Overhead Profile (CUDA Graphs)
        sig_lat = ExecutionSignatureEmbedding(
            arithmetic_intensity_norm=0.25,
            dram_bandwidth_saturation=0.30,
            host_to_device_latency_ratio=0.45,
            tensor_core_occupancy_norm=0.25,
            sm_utilization_ratio=0.30,
            grad_norm_mean_log=0.20,
            loss_delta_snr=0.90,
            birkhoff_drift_residual=0.0,
            vram_utilization_pct=0.35,
            kernel_launch_frequency_khz=0.95,
            gemm_to_elementwise_ratio=0.40,
            activation_memory_ratio=0.30,
            forward_to_backward_ratio=0.35,
            dataloader_wait_ratio=0.05,
            effective_tflops_ratio=0.20,
            curvature_second_order_norm=0.05,
            client_tag="kb_ref_launch_bound",
            model_family="transformer",
            hardware_architecture="NVIDIA",
        )
        self.register_recipe(
            recipe_id="RECIPE_CUDA_GRAPHS_DISPATCH",
            target_hardware="NVIDIA",
            signature=sig_lat,
            recommended_optimizations=[
                "Capture forward/backward graph with CUDA Graphs",
                "Enable non-blocking D2H/H2D streams",
                "Increase microbatch size to amortize launch costs",
            ],
        )

        # 3. Large Dense Compute Profile (Muon / FP8 Scaling)
        sig_compute = ExecutionSignatureEmbedding(
            arithmetic_intensity_norm=0.85,
            dram_bandwidth_saturation=0.55,
            host_to_device_latency_ratio=0.02,
            tensor_core_occupancy_norm=0.85,
            sm_utilization_ratio=0.90,
            grad_norm_mean_log=0.30,
            loss_delta_snr=0.95,
            birkhoff_drift_residual=0.0,
            vram_utilization_pct=0.85,
            kernel_launch_frequency_khz=0.2,
            gemm_to_elementwise_ratio=0.90,
            activation_memory_ratio=0.50,
            forward_to_backward_ratio=0.33,
            dataloader_wait_ratio=0.01,
            effective_tflops_ratio=0.80,
            curvature_second_order_norm=0.15,
            client_tag="kb_ref_compute_dense",
            model_family="transformer",
            hardware_architecture="NVIDIA",
        )
        self.register_recipe(
            recipe_id="RECIPE_MUON_HYBRID_FP8",
            target_hardware="NVIDIA",
            signature=sig_compute,
            recommended_optimizations=[
                "Deploy HybridMuonOptimizer (Newton-Schulz polar decomposition for 2D matrices)",
                "Enable FP8 Scaled GEMM matrix multiplication",
                "Tune gradient accumulation to maximize Tensor Core tile saturation",
            ],
        )

    def register_recipe(
        self,
        recipe_id: str,
        target_hardware: str,
        signature: ExecutionSignatureEmbedding,
        recommended_optimizations: List[str],
    ) -> None:
        """Registers an empirical recipe with its normalized 16D embedding."""
        embedding = signature.to_unit_hypersphere_embedding()
        self.registry.append({
            "recipe_id": recipe_id,
            "target_hardware": target_hardware,
            "embedding": embedding,
            "signature": signature,
            "recommended_optimizations": recommended_optimizations,
        })

    def find_nearest_recipe(
        self,
        query_signature: ExecutionSignatureEmbedding,
        min_similarity_threshold: float = 0.80,
    ) -> Optional[MatchResult]:
        """
        Finds the closest verified recipe on the 16D unit hypersphere.
        """
        if not self.registry:
            return None

        query_emb = query_signature.to_unit_hypersphere_embedding()
        best_match = None
        best_sim = -1.0

        for item in self.registry:
            sim = cosine_similarity(query_emb, item["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = item

        if best_match is None:
            return None

        return MatchResult(
            matched_recipe_id=best_match["recipe_id"],
            target_hardware=best_match["target_hardware"],
            similarity_score=best_sim,
            recommended_optimizations=best_match["recommended_optimizations"],
            is_high_confidence_match=(best_sim >= min_similarity_threshold),
        )
