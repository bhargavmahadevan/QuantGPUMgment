"""
Pearlmutter Hessian-Vector Product (HVP) & Curvature Estimation.

Evaluates second-order directional derivatives:
    H v = ∇^2 L(θ) v = lim_{ε -> 0} [∇ L(θ + ε v) - ∇ L(θ)] / ε

Avoids constructing the full O(N^2) Hessian matrix, requiring only one
additional forward/backward pass or dual-autograd graph traversal.
"""

from typing import List, Tuple, Optional, Callable
import torch
from dataclasses import dataclass


@dataclass
class CurvatureDiagnostic:
    """Diagnostic profile of the local loss surface curvature."""
    dominant_eigenvalue_est: float
    rayleigh_quotient: float
    gradient_norm: float
    is_ill_conditioned: bool
    recommended_damping: float
    sample_regime: str  # 'low_data_few_shot', 'medium', 'large'


def compute_hvp_pearlmutter(
    loss_fn: Callable[[], torch.Tensor],
    parameters: List[torch.nn.Parameter],
    vector: List[torch.Tensor],
    epsilon: float = 1e-4,
) -> List[torch.Tensor]:
    """
    Computes Hessian-Vector Product using Pearlmutter's finite difference trick
    or autograd-of-gradient:
        H v ≈ [∇ L(θ + ε v) - ∇ L(θ - ε v)] / (2 ε)
    
    Guarantees O(N) memory without allocating full Hessian.
    """
    # Filter only parameters that require gradients
    params = [p for p in parameters if p.requires_grad]
    if not params or not vector:
        return []

    # Store original parameters
    orig_data = [p.data.clone() for p in params]

    # Positive perturbation: θ + ε v
    for p, v in zip(params, vector):
        p.data.add_(v, alpha=epsilon)
    loss_pos = loss_fn()
    grads_pos = torch.autograd.grad(loss_pos, params, create_graph=False, retain_graph=False)

    # Negative perturbation: θ - ε v
    for p, orig, v in zip(params, orig_data, vector):
        p.data.copy_(orig).sub_(v, alpha=epsilon)
    loss_neg = loss_fn()
    grads_neg = torch.autograd.grad(loss_neg, params, create_graph=False, retain_graph=False)

    # Restore original parameters
    for p, orig in zip(params, orig_data):
        p.data.copy_(orig)

    # Central difference H v ≈ (∇L+ - ∇L-) / (2ε)
    hvp = [(g_p - g_n) / (2.0 * epsilon) for g_p, g_n in zip(grads_pos, grads_neg)]
    return hvp


def estimate_local_curvature(
    model: torch.nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    num_power_iterations: int = 5,
    sample_count: int = 100,
) -> CurvatureDiagnostic:
    """
    Power iteration to estimate dominant eigenvalue of the Hessian (spectral radius)
    and diagnose if the optimization is trapped in an ill-conditioned ravine.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return CurvatureDiagnostic(
            dominant_eigenvalue_est=0.0,
            rayleigh_quotient=0.0,
            gradient_norm=0.0,
            is_ill_conditioned=False,
            recommended_damping=1e-3,
            sample_regime="low_data_few_shot" if sample_count < 1000 else "large"
        )

    # Compute base gradients
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, retain_graph=True)
    grad_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads)).item()

    # Initialize random unit vector
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum(x.norm() ** 2 for x in v))
    v = [x / (v_norm + 1e-8) for x in v]

    eigenvalue_est = 0.0
    for _ in range(num_power_iterations):
        # Compute H v using autograd graph
        hvp = compute_hvp_pearlmutter(loss_fn, params, v)
        if not hvp:
            break
        # Rayleigh quotient: (v^T H v) / (v^T v)
        eigenvalue_est = sum((vi * hvpi).sum() for vi, hvpi in zip(v, hvp)).item()
        
        # Normalize for next iteration
        hvp_norm = torch.sqrt(sum(h.norm() ** 2 for h in hvp))
        if hvp_norm.item() > 1e-8:
            v = [h / hvp_norm for h in hvp]
        else:
            break

    # Determine condition state
    # High top eigenvalue relative to gradient norm indicates sharp curvature / ravine
    is_ill_conditioned = eigenvalue_est > 50.0 or (grad_norm > 0 and (eigenvalue_est / (grad_norm + 1e-6)) > 20.0)
    damping = max(1e-4, min(1.0, eigenvalue_est * 0.05))

    sample_regime = "low_data_few_shot" if sample_count < 1000 else ("medium" if sample_count < 50000 else "large")

    return CurvatureDiagnostic(
        dominant_eigenvalue_est=round(float(eigenvalue_est), 4),
        rayleigh_quotient=round(float(eigenvalue_est), 4),
        gradient_norm=round(float(grad_norm), 4),
        is_ill_conditioned=is_ill_conditioned,
        recommended_damping=round(float(damping), 6),
        sample_regime=sample_regime
    )
