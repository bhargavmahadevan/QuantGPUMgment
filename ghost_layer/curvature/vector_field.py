"""
Curvature Diagnostics & Gradient Alignment Engine (Hessian-Trace Divergence, Sequential Update Orthogonality, & Gradient Alignment Filtering).

Provides step-level trajectory diagnostics and update filtering:
1. Hessian-Trace Estimation (Hutchinson Randomized Trace Estimator):
   Evaluates Tr(H(θ)) ≈ div(g) via Hutchinson Rademacher projections with Pearlmutter HVP.
   Measures mean local curvature along the gradient field:
   - Tr(H) > 0: Convex local curvature (stable descent basin).
   - Tr(H) < 0: Negative/indefinite curvature (saddle point or ill-conditioned landscape).
2. Sequential Update Orthogonality Index:
   Computes the mean sine of the angle between consecutive parameter update vectors (v_t, v_{t+1}).
   Measures oscillatory vs collinear trajectory behavior:
   - Orthogonality ~ 0.0: Collinear updates (efficient directional progress).
   - Orthogonality ~ 1.0: Orthogonal update cycling (limit-cycle oscillation around ravines).
3. Orthogonal Gradient Projection & Filtering:
   Decomposes an update vector v into collinear gradient descent (v_collinear) and orthogonal residual (v_orthogonal).
   Dampens the orthogonal oscillatory component to stabilize optimizer trajectory.
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
import math
import torch
import torch.nn as nn


def estimate_divergence_flux(
    model: nn.Module,
    loss_fn: Optional[Callable[[], torch.Tensor]] = None,
    num_samples: int = 50,
    seed: Optional[int] = None,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes the trace of the Hessian (Hessian-trace divergence):
        div(g) = nabla . g = Tr(H(theta))
    using the Hutchinson randomized trace estimator with Pearlmutter HVP.
    """
    if seed is not None:
        torch.manual_seed(seed)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0

    if loss_fn is None:
        # Fallback to model forward if callable with no args
        try:
            loss_fn = lambda: model()
        except Exception:
            return 0.0

    # Evaluate gradient with computation graph enabled for HVP
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)

    total_trace = 0.0

    for _ in range(num_samples):
        # Generate Rademacher vector z in {-1, +1}
        zs = [torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype) * 2.0 - 1.0 for p in params]
        
        # Inner product g . z
        grad_dot_z = sum(torch.sum(g * z) for g, z in zip(grads, zs))
        
        # Second derivative HVP = grad(g . z, theta)
        hvp = torch.autograd.grad(grad_dot_z, params, retain_graph=True)
        
        # Quadratic form z^T H z
        z_h_z = sum(torch.sum(z * h).item() for z, h in zip(zs, hvp))
        total_trace += z_h_z

    return total_trace / float(num_samples)



def estimate_update_orthogonality_index(velocity_history: List[torch.Tensor], eps: float = 1e-10) -> float:
    """
    Estimates the angular update orthogonality index (circulation heuristic) of parameter velocity vectors:
        OrthogonalityIndex in [0.0, 1.0]
    
    Measures the mean sine of the angle between sequential velocity vectors:
        sin(theta) = sqrt( ||v_t||^2 ||v_{t+1}||^2 - (v_t . v_{t+1})^2 ) / (||v_t|| ||v_{t+1}|| + eps)
    
    - OrthogonalityIndex ~ 0.0: Collinear radial descent (aligned with consecutive updates).
    - OrthogonalityIndex ~ 1.0: Orthogonal update cycling (rotational limit-cycle behavior).
    """
    if len(velocity_history) < 2:
        return 0.0

    sine_angles: List[float] = []
    for t in range(len(velocity_history) - 1):
        v1 = velocity_history[t].flatten()
        v2 = velocity_history[t + 1].flatten()

        v1_norm_sq = torch.dot(v1, v1).item()
        v2_norm_sq = torch.dot(v2, v2).item()

        if v1_norm_sq < eps or v2_norm_sq < eps:
            continue

        dot = torch.dot(v1, v2).item()
        # Clamp to avoid numerical float precision errors
        cos_sq = min(1.0, max(0.0, (dot * dot) / (v1_norm_sq * v2_norm_sq + eps)))
        sin_val = math.sqrt(max(0.0, 1.0 - cos_sq))
        sine_angles.append(sin_val)

    if not sine_angles:
        return 0.0

    return sum(sine_angles) / len(sine_angles)


# Backward-compatible alias
estimate_vorticity_curl = estimate_update_orthogonality_index


def orthogonal_gradient_projection(
    v: torch.Tensor,
    ref_gradient: torch.Tensor,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonal Gradient Decomposition:
    Decomposes velocity vector v into collinear gradient descent and orthogonal residual components:
        v = v_collinear + v_orthogonal
    where:
        v_collinear = (v . ref_grad / ||ref_grad||^2) * ref_grad   (collinear gradient flow)
        v_orthogonal = v - v_collinear                             (orthogonal residual / circulation)

    Note: In 1-form continuous field theory this represents a 1D directional slice
    of Helmholtz-Hodge decomposition; in discrete parameter updates it is an orthogonal
    gradient projection.
    """
    ref_norm_sq = torch.dot(ref_gradient.flatten(), ref_gradient.flatten()).item()
    if ref_norm_sq < eps:
        # Cannot project onto zero gradient, return v as collinear
        return v.clone(), torch.zeros_like(v)

    proj_scalar = torch.dot(v.flatten(), ref_gradient.flatten()).item() / ref_norm_sq
    v_collinear = ref_gradient * proj_scalar
    v_orthogonal = v - v_collinear
    return v_collinear, v_orthogonal


# Backward-compatible alias
helmholtz_hodge_decomposition = orthogonal_gradient_projection


class GradientAlignmentFilter:
    """
    Dynamically filters orthogonal oscillatory components out of optimizer steps,
    projecting updates onto the aligned gradient descent manifold.
    """

    def __init__(self, damping_factor: float = 0.5, history_len: int = 10, vorticity_damping_factor: Optional[float] = None):
        factor = vorticity_damping_factor if vorticity_damping_factor is not None else damping_factor
        self.damping_factor = max(0.0, min(1.0, factor))
        self.history_len = history_len
        self.velocity_history: List[torch.Tensor] = []

    def filter_step(self, step_vector: torch.Tensor, ref_gradient: torch.Tensor) -> torch.Tensor:
        """
        Dampens the orthogonal component of the step vector while preserving
        the collinear gradient descent component.
        """
        v_collinear, v_orthogonal = orthogonal_gradient_projection(step_vector, ref_gradient)
        v_filtered = v_collinear + (1.0 - self.damping_factor) * v_orthogonal

        # Maintain detached history for orthogonality tracking
        self.velocity_history.append(v_filtered.flatten().detach())
        if len(self.velocity_history) > self.history_len:
            self.velocity_history.pop(0)

        return v_filtered

    @property
    def current_vorticity(self) -> float:
        """Returns the current orthogonal circulation index of recent steps."""
        return estimate_update_orthogonality_index(self.velocity_history)

    @property
    def current_orthogonality_index(self) -> float:
        """Returns the current orthogonal circulation index of recent steps."""
        return estimate_update_orthogonality_index(self.velocity_history)


# Backward-compatible alias
HelmholtzHodgeFilter = GradientAlignmentFilter
