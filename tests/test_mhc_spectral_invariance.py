"""
Test Suite: Manifold-Constrained Hyper-Connections (mHC) Spectral Invariance
Validates mathematical invariants of DeepSeek's mHC residual architecture:
1. Sinkhorn-Knopp doubly stochastic projection onto the Birkhoff Polytope
2. Perron-Frobenius spectral radius invariance: rho(H) == 1.0
3. Closure under deep composition: prod_{l=1}^L H_l in Birkhoff Polytope
4. Gradient flow stability across 32-layer deep stacks (no vanishing/exploding)
5. Numerical precision resilience (FP32 & BF16)
6. Diagonal dominance under identity warm-start
"""

import math
import pytest
import torch
import torch.nn as nn

from ghost_layer.curvature.mhc import (
    sinkhorn_knopp_doubly_stochastic,
    birkhoff_drift_metric,
    mHCResidual,
)


def test_sinkhorn_batch_broadcasting():
    """Verify Sinkhorn-Knopp operates correctly across arbitrary batch dimensions."""
    torch.manual_seed(42)
    # 3D: (B, N, N)
    logits_3d = torch.randn(4, 8, 8) * 2.0
    H_3d = sinkhorn_knopp_doubly_stochastic(logits_3d, iters=25)
    assert H_3d.shape == (4, 8, 8)
    assert (H_3d >= 0.0).all()
    assert torch.allclose(H_3d.sum(dim=-1), torch.ones(4, 8), atol=1e-2)
    assert torch.allclose(H_3d.sum(dim=-2), torch.ones(4, 8), atol=1e-2)

    # 4D: (B, S, N, N)
    logits_4d = torch.randn(2, 3, 6, 6)
    H_4d = sinkhorn_knopp_doubly_stochastic(logits_4d, iters=25)
    assert H_4d.shape == (2, 3, 6, 6)
    assert torch.allclose(H_4d.sum(dim=-1), torch.ones(2, 3, 6), atol=1e-2)
    assert torch.allclose(H_4d.sum(dim=-2), torch.ones(2, 3, 6), atol=1e-2)


def test_sinkhorn_stream_scaling_convergence():
    """Verify Sinkhorn-Knopp converges across stream counts N in [2, 4, 8, 16, 32, 64]."""
    torch.manual_seed(42)
    for N in [2, 4, 8, 16, 32, 64]:
        logits = torch.randn(N, N) * 3.0
        # Low iterations vs high iterations
        H_low = sinkhorn_knopp_doubly_stochastic(logits, iters=5)
        H_high = sinkhorn_knopp_doubly_stochastic(logits, iters=30)

        drift_low = birkhoff_drift_metric(H_low)
        drift_high = birkhoff_drift_metric(H_high)

        assert drift_high <= drift_low or drift_high < 0.05, (
            f"Expected monotonic convergence for N={N}: low={drift_low}, high={drift_high}"
        )
        assert (H_high >= 0.0).all()
        # Row and column sums must be bounded within tolerance
        assert torch.allclose(H_high.sum(dim=-1), torch.ones(N), atol=3e-2)
        assert torch.allclose(H_high.sum(dim=-2), torch.ones(N), atol=3e-2)


def test_birkhoff_polytope_composition_closure():
    """
    Mathematical Invariant: The Birkhoff Polytope is closed under matrix multiplication.
    The product of any sequence of doubly stochastic matrices P = H_L ... H_1 is doubly stochastic.
    """
    torch.manual_seed(42)
    N = 8
    for depth in [4, 8, 16, 32]:
        P = torch.eye(N)
        for _ in range(depth):
            raw = torch.randn(N, N) * 1.5
            H_l = sinkhorn_knopp_doubly_stochastic(raw, iters=30)
            P = P @ H_l

        # P must be doubly stochastic
        assert (P >= -1e-6).all(), f"Negative entry in cumulative product at depth {depth}"
        assert torch.allclose(P.sum(dim=-1), torch.ones(N), atol=5e-2), (
            f"Row sums drifted from 1.0 at depth {depth}: {P.sum(dim=-1)}"
        )
        assert torch.allclose(P.sum(dim=-2), torch.ones(N), atol=5e-2), (
            f"Col sums drifted from 1.0 at depth {depth}: {P.sum(dim=-2)}"
        )


def test_perron_frobenius_spectral_radius_under_deep_composition():
    """
    Perron-Frobenius Theorem Guarantee:
    For any doubly stochastic matrix H or composite product P, spectral radius rho(P) == 1.0.
    """
    torch.manual_seed(42)
    N = 6
    for depth in [8, 16, 32, 64]:
        P = torch.eye(N)
        for _ in range(depth):
            raw = torch.randn(N, N)
            H_l = sinkhorn_knopp_doubly_stochastic(raw, iters=30)
            P = P @ H_l

        eigvals = torch.linalg.eigvals(P)
        spectral_radius = float(torch.max(torch.abs(eigvals)).item())
        assert spectral_radius == pytest.approx(1.0, abs=0.03), (
            f"Spectral radius at depth {depth} deviated: rho = {spectral_radius}"
        )


def test_gradient_flow_stability_across_32_layers():
    """
    Verify backpropagation across a 32-layer mHCResidual stack.
    Gradients must neither vanish (||grad|| -> 0) nor explode (||grad|| -> inf / NaN).
    """
    torch.manual_seed(42)
    B, S, D, N = 2, 4, 16, 4
    depth = 32

    layers = nn.ModuleList([
        mHCResidual(hidden_dim=D, num_streams=N, sinkhorn_iters=10)
        for _ in range(depth)
    ])

    x = torch.randn(B, S, D, requires_grad=True)
    curr = x

    for layer in layers:
        curr = layer(curr)

    # Collapse final output and compute scalar objective
    out = layers[-1].collapse_to_single_stream(curr)
    loss = out.sum()
    loss.backward()

    # Input gradient checks
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    grad_norm = float(x.grad.norm().item())
    assert grad_norm > 1e-4, f"Vanishing gradient detected: norm = {grad_norm}"
    assert grad_norm < 1e5, f"Exploding gradient detected: norm = {grad_norm}"

    # Verify every layer's raw mixing weights received valid non-zero gradients
    for i, layer in enumerate(layers):
        assert layer.raw_mixing_weights.grad is not None, f"Layer {i} mixing grad is None"
        layer_grad_norm = float(layer.raw_mixing_weights.grad.norm().item())
        assert not math.isnan(layer_grad_norm), f"Layer {i} grad is NaN"
        assert layer_grad_norm > 0.0, f"Layer {i} grad is zero"


def test_numerical_precision_resilience_bf16_fp32():
    """Verify Sinkhorn-Knopp and mHCResidual operate stably in both FP32 and BF16."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dtype in [torch.float32, torch.bfloat16]:
        logits = torch.randn(8, 8, dtype=dtype, device=device)
        H = sinkhorn_knopp_doubly_stochastic(logits, iters=20)

        assert not torch.isnan(H).any()
        assert not torch.isinf(H).any()
        assert (H >= 0.0).all()

        row_sums = H.sum(dim=-1).float()
        assert torch.allclose(row_sums, torch.ones(8, device=device), atol=5e-2)

        # Module execution
        mhc = mHCResidual(hidden_dim=16, num_streams=4, sinkhorn_iters=10).to(device=device, dtype=dtype)
        x = torch.randn(2, 4, 16, dtype=dtype, device=device, requires_grad=True)
        out = mhc(x)
        assert out.shape == (2, 4, 4, 16)
        assert not torch.isnan(out).any()


def test_identity_warmstart_diagonal_dominance():
    """Verify that init_identity_bias creates initial mixing matrix close to Identity."""
    torch.manual_seed(42)
    N = 4
    mhc = mHCResidual(hidden_dim=16, num_streams=N, sinkhorn_iters=15, init_identity_bias=3.0)
    H = mhc.get_doubly_stochastic_matrix()

    diag_mean = float(torch.diag(H).mean().item())
    off_diag_mask = ~torch.eye(N, dtype=torch.bool)
    off_diag_mean = float(H[off_diag_mask].mean().item())

    # Diagonal entries should dominate off-diagonal by a substantial factor
    assert diag_mean > 0.6, f"Expected strong diagonal warm-start, got {diag_mean}"
    assert diag_mean > off_diag_mean * 3.0, (
        f"Diagonal ({diag_mean}) does not dominate off-diagonal ({off_diag_mean})"
    )


def test_temperature_scaling_birkhoff_preservation():
    """Verify temperature scaling preserves doubly stochastic constraints."""
    torch.manual_seed(42)
    N = 6
    logits = torch.randn(N, N) * 2.0

    for temp in [0.1, 0.5, 1.0, 3.0, 10.0]:
        H = sinkhorn_knopp_doubly_stochastic(logits, iters=35, temperature=temp)
        assert (H >= 0.0).all()
        assert torch.allclose(H.sum(dim=-1), torch.ones(N), atol=0.08), (
            f"Row sum failed at temp {temp}"
        )
        assert torch.allclose(H.sum(dim=-2), torch.ones(N), atol=0.08), (
            f"Col sum failed at temp {temp}"
        )


def test_single_stream_collapse_roundtrip():
    """Verify expansion from [B, S, D] -> [B, S, N, D] -> collapse back to [B, S, D]."""
    torch.manual_seed(42)
    B, S, D, N = 2, 8, 32, 4
    x_single = torch.randn(B, S, D)

    mhc = mHCResidual(hidden_dim=D, num_streams=N, sinkhorn_iters=10)

    # Forward with optional sublayer
    mlp = nn.Sequential(
        nn.Linear(D, D * 2),
        nn.GELU(),
        nn.Linear(D * 2, D),
    )

    x_multi = mhc(x_single, sublayer_fn=mlp)
    assert x_multi.shape == (B, S, N, D)

    # Collapse back
    x_reconstructed = mhc.collapse_to_single_stream(x_multi)
    assert x_reconstructed.shape == (B, S, D)

    # Test backward pass through full cycle
    loss = x_reconstructed.sum()
    loss.backward()

    assert mhc.raw_mixing_weights.grad is not None
    assert mhc.in_proj.weight.grad is not None
    assert mhc.out_proj.weight.grad is not None
