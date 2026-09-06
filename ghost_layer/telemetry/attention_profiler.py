import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class AttentionArchitecture(str, Enum):
    MHA = "MHA"  # Multi-Head Attention
    MQA = "MQA"  # Multi-Query Attention
    GQA = "GQA"  # Grouped-Query Attention
    MLA = "MLA"  # Multi-Head Latent Attention (DeepSeek-style)
    DSA = "DSA"  # Dynamic Sparse Attention


@dataclass
class AttentionProfile:
    """Detailed memory and computational profile for an attention layer/model."""
    architecture: AttentionArchitecture
    sequence_length: int
    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    precision_bytes: int = 2  # 2 for fp16/bf16, 4 for fp32, 1 for fp8
    mla_latent_dim: Optional[int] = None  # Latent KV compression dimension c_t^{KV}
    mla_rope_dim: Optional[int] = None    # Decoupled RoPE dimension k_t^R
    sparse_top_k: Optional[int] = None     # Top-k active token budget for DSA

    @property
    def kv_cache_bytes_per_token(self) -> float:
        """Computes KV cache bytes required per single token across all layers."""
        if self.architecture == AttentionArchitecture.MLA:
            # MLA stores low-rank compressed KV vector c_t^{KV} plus decoupled RoPE key k_t^R
            latent_dim = self.mla_latent_dim if self.mla_latent_dim is not None else 512
            rope_dim = self.mla_rope_dim if self.mla_rope_dim is not None else 64
            # Per token per layer: latent_dim (for both K&V latent) + rope_dim (for positional key)
            return (latent_dim + rope_dim) * self.num_layers * self.precision_bytes
        
        elif self.architecture == AttentionArchitecture.MQA:
            # 1 key head + 1 value head
            return 2 * 1 * self.head_dim * self.num_layers * self.precision_bytes
        
        elif self.architecture == AttentionArchitecture.GQA:
            # G key/value heads
            return 2 * self.num_kv_heads * self.head_dim * self.num_layers * self.precision_bytes
        
        elif self.architecture == AttentionArchitecture.DSA:
            # Dynamic sparse attention over top-k tokens only (active KV working set)
            k = self.sparse_top_k if self.sparse_top_k is not None else min(self.sequence_length, 2048)
            effective_ratio = min(1.0, k / max(1, self.sequence_length))
            return 2 * self.num_kv_heads * self.head_dim * self.num_layers * self.precision_bytes * effective_ratio

        else:  # Standard MHA
            return 2 * self.num_query_heads * self.head_dim * self.num_layers * self.precision_bytes

    @property
    def total_kv_cache_mb(self) -> float:
        """Total KV cache memory in Megabytes for current batch size and sequence length."""
        total_bytes = self.kv_cache_bytes_per_token * self.sequence_length * self.batch_size
        return total_bytes / (1024 * 1024)

    @property
    def mha_baseline_kv_cache_mb(self) -> float:
        """Theoretical standard MHA KV cache memory for comparison."""
        mha_bytes_per_token = 2 * self.num_query_heads * self.head_dim * self.num_layers * self.precision_bytes
        return (mha_bytes_per_token * self.sequence_length * self.batch_size) / (1024 * 1024)

    @property
    def kv_compression_ratio(self) -> float:
        """Ratio of standard MHA KV footprint to this architecture's footprint (e.g. 8.0x)."""
        curr = self.total_kv_cache_mb
        baseline = self.mha_baseline_kv_cache_mb
        if curr <= 0:
            return 1.0
        return round(baseline / curr, 2)


def compute_kv_cache_footprint(
    num_layers: int,
    num_query_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    num_kv_heads: Optional[int] = None,
    architecture: str = "MHA",
    mla_latent_dim: Optional[int] = 512,
    mla_rope_dim: Optional[int] = 64,
    sparse_top_k: Optional[int] = None,
    precision_bytes: int = 2,
) -> Dict[str, Any]:
    """
    Computes precise KV cache memory footprint and compression metrics across
    different attention architectures (MHA, MQA, GQA, MLA, DSA).
    """
    try:
        arch = AttentionArchitecture(architecture.upper())
    except ValueError:
        arch = AttentionArchitecture.MHA

    kv_heads = num_kv_heads if num_kv_heads is not None else (
        1 if arch == AttentionArchitecture.MQA else num_query_heads
    )

    profile = AttentionProfile(
        architecture=arch,
        sequence_length=seq_len,
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        precision_bytes=precision_bytes,
        mla_latent_dim=mla_latent_dim,
        mla_rope_dim=mla_rope_dim,
        sparse_top_k=sparse_top_k,
    )

    return {
        "architecture": arch.value,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "kv_cache_mb": round(profile.total_kv_cache_mb, 2),
        "mha_baseline_mb": round(profile.mha_baseline_kv_cache_mb, 2),
        "compression_ratio_vs_mha": profile.kv_compression_ratio,
        "kv_cache_bytes_per_token": round(profile.kv_cache_bytes_per_token, 2),
    }


def calculate_gqa_compression_ratio(num_query_heads: int, num_kv_heads: int) -> float:
    """Calculates KV memory compression ratio achieved by Grouped-Query Attention."""
    if num_kv_heads <= 0:
        return 1.0
    return round(num_query_heads / num_kv_heads, 2)


def calculate_mla_compression_ratio(
    num_query_heads: int,
    head_dim: int,
    latent_dim_kv: int = 512,
    rope_dim: int = 64,
) -> float:
    """
    Calculates KV memory compression ratio achieved by Multi-Head Latent Attention
    compared to standard Multi-Head Attention.
    """
    mha_dim_per_token = 2 * num_query_heads * head_dim
    mla_dim_per_token = latent_dim_kv + rope_dim
    if mla_dim_per_token <= 0:
        return 1.0
    return round(mha_dim_per_token / mla_dim_per_token, 2)


def estimate_attention_arithmetic_intensity(
    seq_len: int,
    head_dim: int,
    batch_size: int,
    num_heads: int,
    is_prefill: bool = False,
) -> float:
    """
    Estimates the arithmetic intensity (FLOPs / Byte) of the attention operation.
    - Prefill phase: O(N^2) FLOPs over O(N) memory -> High arithmetic intensity (Compute Bound).
    - Decode phase: O(N) FLOPs over O(N) memory -> Low arithmetic intensity (Memory Bandwidth Bound).
    """
    if is_prefill:
        # Attention score FLOPs: 2 * batch * heads * seq_len^2 * head_dim
        flops = 2.0 * batch_size * num_heads * (seq_len ** 2) * head_dim
        # Memory traffic (Q, K, V activations): 3 * batch * heads * seq_len * head_dim * 2 bytes
        bytes_transferred = 3.0 * batch_size * num_heads * seq_len * head_dim * 2.0
        return round(flops / max(1.0, bytes_transferred), 2)
    else:
        # Generation step: 1 new token attending to seq_len past tokens
        flops = 2.0 * batch_size * num_heads * seq_len * head_dim
        # Must read entire historical KV cache from HBM
        bytes_transferred = 2.0 * batch_size * num_heads * seq_len * head_dim * 2.0
        return round(flops / max(1.0, bytes_transferred), 2)
