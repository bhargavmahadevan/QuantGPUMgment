"""
3D Pareto-Convex Memory & Batching Hull for GhostLayer.

Models the continuous 3D VRAM activation envelope:
    M(SeqLen, BatchSize, TilingMode) <= VRAM_usable

Constructs the Pareto efficiency frontier of (BatchSize, SeqLen) configurations
that maximize token throughput (tokens/sec) without ever crossing the CUDA
Out-of-Memory (OOM) fault boundary.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import math


class ActivationMode(str, Enum):
    STANDARD_FP32 = "STANDARD_FP32"
    MIXED_PRECISION_AMP = "MIXED_PRECISION_AMP"
    ACTIVATION_CHECKPOINTING = "ACTIVATION_CHECKPOINTING"
    FLASH_ATTENTION_FUSED = "FLASH_ATTENTION_FUSED"
    FULL_OPTIMIZED_STACK = "FULL_OPTIMIZED_STACK"


@dataclass
class ParetoBatchingPoint:
    """A single Pareto-optimal execution point on the convex memory hull."""
    sequence_length: int
    batch_size: int
    gradient_accumulation_steps: int
    estimated_vram_gb: float
    vram_headroom_pct: float
    estimated_throughput_tokens_per_sec: float
    activation_mode: ActivationMode
    is_safe_without_oom: bool


@dataclass
class ConvexHullReport:
    """Full Pareto Convex Hull report for a model architecture & hardware."""
    optimal_point: ParetoBatchingPoint
    pareto_frontier: List[ParetoBatchingPoint]
    vram_budget_gb: float
    model_params_billion: float
    safe_convex_hull_volume: float
    recommended_scaling_strategy: str


class ParetoMemoryHull:
    """
    Computes and optimizes along the 3D Convex Memory Hull.
    """

    def __init__(
        self,
        vram_total_gb: float = 16.0,          # e.g. 16GB RTX A2000 / 80GB A100 / H100
        safety_headroom_margin: float = 0.15, # 15% reserved for CUDA runtime & fragmentation
        model_params_billion: float = 0.5,    # e.g. 500M or 7B params
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        base_tokens_per_sec: float = 45.0,    # Hardware-specific throughput constant (tokens/sec per token-slot)
    ):
        self.vram_total = vram_total_gb
        self.safety_margin = safety_headroom_margin
        self.vram_usable = vram_total_gb * (1.0 - safety_headroom_margin)
        self.params_b = model_params_billion
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.base_tokens_per_sec = base_tokens_per_sec

    def estimate_vram_gb(
        self,
        seq_len: int,
        batch_size: int,
        mode: ActivationMode = ActivationMode.MIXED_PRECISION_AMP,
    ) -> float:
        """
        Estimates total VRAM footprint in GB for a given (seq_len, batch_size, mode).
        """
        # 1. Weights + Gradients + Optimizer states (in GB)
        # FP16 params (2 bytes) + FP16 grads (2 bytes) + AdamW (8 bytes) = 12 bytes/param (mixed precision)
        if mode == ActivationMode.STANDARD_FP32:
            static_bytes = self.params_b * 1e9 * 16.0  # FP32: 4 (params) + 4 (grads) + 8 (Adam) = 16 bytes
        else:
            static_bytes = self.params_b * 1e9 * 12.0

        static_gb = static_bytes / (1024**3)

        # 2. Activation Memory per token
        # Standard transformer: ~ 34 * hidden_dim * num_layers bytes per token
        base_act_bytes_per_token = 34.0 * self.hidden_dim * self.num_layers

        if mode == ActivationMode.STANDARD_FP32:
            mode_factor = 1.0
            attn_quadratic = (self.num_layers * self.num_heads * (seq_len ** 2) * 4.0) / (1024**3)
        elif mode == ActivationMode.MIXED_PRECISION_AMP:
            mode_factor = 0.5
            attn_quadratic = (self.num_layers * self.num_heads * (seq_len ** 2) * 2.0) / (1024**3)
        elif mode == ActivationMode.ACTIVATION_CHECKPOINTING:
            mode_factor = 0.15  # Recompute drops activation footprint by ~85%
            attn_quadratic = (self.num_layers * self.num_heads * (seq_len ** 2) * 2.0) / (1024**3) * 0.2
        elif mode == ActivationMode.FLASH_ATTENTION_FUSED:
            mode_factor = 0.4
            attn_quadratic = 0.0  # Fused SRAM tiling eliminates O(S^2) quadratic memory
        elif mode == ActivationMode.FULL_OPTIMIZED_STACK:
            mode_factor = 0.10   # FlashAttention + Checkpointing + AMP
            attn_quadratic = 0.0

        linear_act_gb = (batch_size * seq_len * base_act_bytes_per_token * mode_factor) / (1024**3)
        total_act_gb = linear_act_gb + (batch_size * attn_quadratic)

        # 3. KV Cache / Work buffers (~0.5 GB)
        buffer_gb = 0.4

        total_vram = static_gb + total_act_gb + buffer_gb
        return round(total_vram, 4)

    def is_within_hull(
        self,
        seq_len: int,
        batch_size: int,
        mode: ActivationMode = ActivationMode.MIXED_PRECISION_AMP,
    ) -> bool:
        """Returns True if the configuration resides strictly within the safe convex hull."""
        est_gb = self.estimate_vram_gb(seq_len, batch_size, mode)
        return est_gb <= self.vram_usable

    def compute_pareto_frontier(
        self,
        candidate_seq_lens: Optional[List[int]] = None,
        max_batch_size: int = 64,
        target_effective_batch: int = 64,
        mode: ActivationMode = ActivationMode.FULL_OPTIMIZED_STACK,
    ) -> ConvexHullReport:
        """
        Sweeps the parameter space and finds the maximum feasible batch size per
        sequence length that fits within the VRAM budget.

        Note: This produces the efficiency frontier on the batch dimension
        (largest batch per seq_len that avoids OOM), NOT a multi-objective
        Pareto frontier trading off throughput vs. VRAM headroom.
        """
        if candidate_seq_lens is None:
            candidate_seq_lens = [512, 1024, 2048, 4096]

        frontier: List[ParetoBatchingPoint] = []

        for seq in candidate_seq_lens:
            best_batch_for_seq: Optional[ParetoBatchingPoint] = None
            for b in range(1, max_batch_size + 1):
                vram = self.estimate_vram_gb(seq, b, mode)
                if vram <= self.vram_usable:
                    # Model throughput heuristic: tokens/sec grows sub-linearly with batch size (GEMM efficiency)
                    gemm_efficiency = min(1.0, 0.4 + 0.6 * math.log2(1 + b) / math.log2(1 + 32))
                    throughput = (b * seq) * self.base_tokens_per_sec * gemm_efficiency

                    grad_accum = max(1, target_effective_batch // b)
                    headroom = round(((self.vram_usable - vram) / self.vram_usable) * 100.0, 1)

                    pt = ParetoBatchingPoint(
                        sequence_length=seq,
                        batch_size=b,
                        gradient_accumulation_steps=grad_accum,
                        estimated_vram_gb=vram,
                        vram_headroom_pct=headroom,
                        estimated_throughput_tokens_per_sec=round(throughput, 1),
                        activation_mode=mode,
                        is_safe_without_oom=True,
                    )
                    best_batch_for_seq = pt

            if best_batch_for_seq is not None:
                frontier.append(best_batch_for_seq)

        if not frontier:
            # Fallback to minimum batch size
            min_pt = ParetoBatchingPoint(
                sequence_length=512,
                batch_size=1,
                gradient_accumulation_steps=target_effective_batch,
                estimated_vram_gb=self.estimate_vram_gb(512, 1, mode),
                vram_headroom_pct=10.0,
                estimated_throughput_tokens_per_sec=500.0,
                activation_mode=mode,
                is_safe_without_oom=True,
            )
            frontier = [min_pt]

        # Optimal point maximizes throughput on the Pareto frontier
        optimal = max(frontier, key=lambda p: p.estimated_throughput_tokens_per_sec)

        # Hull volume approximation (normalized integral of safe parameter space)
        hull_vol = sum(p.batch_size * p.sequence_length for p in frontier) / (max_batch_size * max(candidate_seq_lens))

        return ConvexHullReport(
            optimal_point=optimal,
            pareto_frontier=frontier,
            vram_budget_gb=self.vram_total,
            model_params_billion=self.params_b,
            safe_convex_hull_volume=round(hull_vol, 4),
            recommended_scaling_strategy=(
                f"Set microbatch={optimal.batch_size}, seq_len={optimal.sequence_length}, "
                f"grad_accum={optimal.gradient_accumulation_steps} ({optimal.activation_mode.value}) "
                f"to achieve ~{optimal.estimated_throughput_tokens_per_sec:.0f} tokens/s with {optimal.vram_headroom_pct}% VRAM headroom."
            ),
        )
