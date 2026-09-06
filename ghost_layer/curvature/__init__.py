"""
GhostLayer Curvature & Higher-Order Calculus Optimization Suite.

Optimizers and diagnostics designed for 2D VRAM tensor manifolds and data-constrained regimes:
- Muon: Momentum Orthogonalized by Newton-Schulz matrix polar decomposition.
- HybridMuonOptimizer: Default partitioner (2D Matrices -> Muon, 1D Vectors -> AdamW).
- Sophia: Second-order Clipped Stochastic Optimization with Hutchinson diagonal Hessian estimation.
- Shampoo: Matrix inverse 2p-th root tensor preconditioning.
- Pearlmutter HVP: Exact Hessian-Vector Product evaluation without full Hessian construction.
"""

from ghost_layer.curvature.hvp import (
    compute_hvp_pearlmutter,
    estimate_local_curvature,
    CurvatureDiagnostic,
)
from ghost_layer.curvature.sophia import SophiaG
from ghost_layer.curvature.muon import (
    Muon,
    HybridMuonOptimizer,
    create_muon_hybrid_optimizer,
    newton_schulz5,
)
from ghost_layer.curvature.shampoo import Shampoo, matrix_power_inverse_root
from ghost_layer.curvature.loss import (
    ChunkedCrossEntropyLoss,
    chunked_cross_entropy,
)
from ghost_layer.curvature.mhc import (
    mHCResidual,
    sinkhorn_knopp_doubly_stochastic,
    birkhoff_drift_metric,
)
from ghost_layer.curvature.pareto_hull import (
    ParetoMemoryHull,
    ParetoBatchingPoint,
    ConvexHullReport,
    ActivationMode,
)
from ghost_layer.curvature.lbfgs import (
    LBFGSCurvatureOptimizer,
    lbfgs_two_loop_recursion,
    damped_powell_update,
)
from ghost_layer.curvature.vector_field import (
    estimate_divergence_flux,
    estimate_vorticity_curl,
    estimate_update_orthogonality_index,
    helmholtz_hodge_decomposition,
    orthogonal_gradient_projection,
    HelmholtzHodgeFilter,
    GradientAlignmentFilter,
)

__all__ = [
    "compute_hvp_pearlmutter",
    "estimate_local_curvature",
    "CurvatureDiagnostic",
    "SophiaG",
    "Muon",
    "HybridMuonOptimizer",
    "create_muon_hybrid_optimizer",
    "newton_schulz5",
    "Shampoo",
    "matrix_power_inverse_root",
    "ChunkedCrossEntropyLoss",
    "chunked_cross_entropy",
    "mHCResidual",
    "sinkhorn_knopp_doubly_stochastic",
    "birkhoff_drift_metric",
    "ParetoMemoryHull",
    "ParetoBatchingPoint",
    "ConvexHullReport",
    "ActivationMode",
    "LBFGSCurvatureOptimizer",
    "lbfgs_two_loop_recursion",
    "damped_powell_update",
    "estimate_divergence_flux",
    "estimate_vorticity_curl",
    "estimate_update_orthogonality_index",
    "helmholtz_hodge_decomposition",
    "orthogonal_gradient_projection",
    "HelmholtzHodgeFilter",
    "GradientAlignmentFilter",
]


