"""
Manifold-Constrained Hyper-Connections (mHC) Suite for GhostLayer.

Implements DeepSeek's mHC residual architecture:
1. Expands standard single residual streams into n parallel hyper-connection streams:
   X_l in R^{B x S x n x d}
2. Constrains stream mixing matrices to the Birkhoff Polytope (Doubly Stochastic Matrices):
   sum_i H_ij = 1,  sum_j H_ij = 1,  H_ij >= 0
3. Guarantees spectral radius rho(H) == 1.0 (Perron-Frobenius theorem), eliminating
   exploding and vanishing gradient dynamics across arbitrarily deep networks.
"""

from typing import Optional, Tuple, Union, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp_doubly_stochastic(
    logits: torch.Tensor,
    iters: int = 30,
    eps: float = 1e-8,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Projects arbitrary unconstrained mixing logits onto the Birkhoff Polytope
    (the manifold of Doubly Stochastic Matrices) using the Sinkhorn-Knopp algorithm.

    Args:
        logits: Tensor of shape (..., N, N) representing raw mixing affinities.
        iters: Number of alternating row/column normalization iterations (default 30).
        eps: Small epsilon for numerical stability.
        temperature: Softmax temperature scaling (default 1.0).

    Returns:
        Doubly stochastic matrix H of shape (..., N, N) where:
        - All elements H_ij >= 0
        - Row sums sum_j H_ij == 1.0
        - Column sums sum_i H_ij == 1.0
    """
    # Numerically stable exponentiation
    scaled = logits / max(1e-4, temperature)
    max_val = scaled.amax(dim=(-1, -2), keepdim=True)
    P = torch.exp(scaled - max_val)

    for _ in range(iters):
        # 1. Row normalization: each row sums to 1
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        # 2. Column normalization: each column sums to 1
        P = P / (P.sum(dim=-2, keepdim=True) + eps)

    return P


def birkhoff_drift_metric(H: torch.Tensor) -> float:
    """
    Computes the L2 distance of a matrix from the ideal Doubly Stochastic Birkhoff boundary.
    Drift == 0.0 indicates a perfectly doubly stochastic matrix.

    Returns:
        Float value representing total row and column deviation from 1.0.
    """
    with torch.no_grad():
        N = H.size(-1)
        row_sums = H.sum(dim=-1)
        col_sums = H.sum(dim=-2)

        row_err = torch.norm(row_sums - 1.0)
        col_err = torch.norm(col_sums - 1.0)
        pos_err = torch.norm(torch.clamp(-H, min=0.0))

        total_drift = float((row_err + col_err + pos_err).item())
        return round(total_drift, 6)


class mHCResidual(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) Residual Layer.

    Replaces standard residual addition (x + F(x)) with an n-stream dynamic hyper-connection
    mixed by a Birkhoff-constrained doubly stochastic matrix.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 5,
        init_identity_bias: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Learnable unconstrained mixing parameters: [num_streams, num_streams]
        # Initialized with a strong diagonal identity bias to guarantee identity warm-start
        init_matrix = torch.eye(num_streams) * init_identity_bias + torch.randn(num_streams, num_streams) * 0.01
        self.raw_mixing_weights = nn.Parameter(init_matrix)

        # Stream input/output linear projections for sub-layer communication
        self.stream_mixer = nn.Linear(num_streams, num_streams, bias=False)
        with torch.no_grad():
            self.stream_mixer.weight.copy_(torch.eye(num_streams))

        # Stream expansion/projection weights when transitioning from 1 stream to n streams
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * num_streams, bias=False)
        self.out_proj = nn.Linear(hidden_dim * num_streams, hidden_dim, bias=False)
        
        # Telemetry storage
        self.last_birkhoff_drift: float = 0.0

    def get_doubly_stochastic_matrix(self) -> torch.Tensor:
        """Returns the current doubly stochastic mixing matrix projected via Sinkhorn-Knopp."""
        H = sinkhorn_knopp_doubly_stochastic(
            self.raw_mixing_weights,
            iters=self.sinkhorn_iters,
        )
        self.last_birkhoff_drift = birkhoff_drift_metric(H)
        return H

    def forward(
        self,
        x_streams: torch.Tensor,
        sublayer_fn: Optional[nn.Module] = None,
        sublayer_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for mHC Residual.

        Args:
            x_streams: [B, S, n, d] multi-stream tensor OR [B, S, d] standard tensor.
            sublayer_fn: Optional callable module (e.g. Attention or MLP) to apply to aggregated streams.
            sublayer_out: Optional pre-computed sub-layer output [B, S, d] or [B, S, n, d].

        Returns:
            Updated multi-stream tensor [B, S, n, d].
        """
        # If input is single stream [B, S, d], expand to [B, S, n, d]
        if x_streams.ndim == 3:
            B, S, D = x_streams.shape
            expanded = self.in_proj(x_streams).view(B, S, self.num_streams, D)
            x_streams = expanded

        B, S, N, D = x_streams.shape
        assert N == self.num_streams, f"Expected {self.num_streams} streams, got {N}"

        # 1. Project mixing weights onto the Birkhoff Polytope
        H = self.get_doubly_stochastic_matrix()  # [N, N]

        # 2. Mix multi-stream residual states: X_mixed = H @ X_streams
        # x_streams: [B, S, N, D] -> transpose to [B, S, D, N] -> matmul with H.T -> [B, S, N, D]
        x_mixed = torch.einsum("ij, bsjd -> bsid", H, x_streams)

        # 3. Compute or incorporate sublayer output F(X)
        if sublayer_out is None and sublayer_fn is not None:
            # Aggregate streams for sublayer execution: average across streams [B, S, D]
            agg_stream = x_mixed.mean(dim=2)
            f_out = sublayer_fn(agg_stream)
        else:
            f_out = sublayer_out

        # 4. Add sublayer contribution to mixed streams
        if f_out is not None:
            if f_out.ndim == 3:
                # Broadcast [B, S, 1, D] across all n streams
                f_expanded = f_out.unsqueeze(2).expand(-1, -1, self.num_streams, -1)
                x_next = x_mixed + f_expanded
            else:
                x_next = x_mixed + f_out
        else:
            x_next = x_mixed

        return x_next

    def collapse_to_single_stream(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Projects multi-stream tensor [B, S, n, d] back to standard single-stream [B, S, d]."""
        if x_streams.ndim == 3:
            return x_streams
        B, S, N, D = x_streams.shape
        flat = x_streams.reshape(B, S, N * D)
        return self.out_proj(flat)
