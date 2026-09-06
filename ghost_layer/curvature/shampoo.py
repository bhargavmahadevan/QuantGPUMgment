"""
Shampoo Preconditioned Optimizer (Matrix Inverse 2p-th Roots).

Preconditions tensor gradients along each dimension using empirical covariance matrices
and matrix inverse p-th roots:
    G_preconditioned = L^{-1/2p} G R^{-1/2p}

Accelerates convergence along ill-conditioned loss ravines where standard AdamW oscillates.

Hardware Consideration:
Matrix inverse root via `torch.linalg.eigh` has O(d^3) complexity. For very large dimensions
(d >= 4096), amortize computation via update intervals (update_preconditioner_interval >= 10-20).
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
import torch
from torch.optim.optimizer import Optimizer


def matrix_power_inverse_root(
    matrix: torch.Tensor,
    root: int = 4,
    eps: float = 1e-4,
    max_eigh_dim: int = 8192
) -> torch.Tensor:
    """
    Computes M^{-1/root} via symmetric eigendecomposition.
    M must be symmetric positive semi-definite.
    
    Complexity: O(d^3) where d = matrix.size(0).
    For dimensions exceeding max_eigh_dim, falls back to diagonal root preconditioning.
    """
    dim = matrix.size(0)
    if dim > max_eigh_dim:
        import warnings
        warnings.warn(
            f"Matrix dimension {dim} exceeds max_eigh_dim={max_eigh_dim}. "
            "Falling back to diagonal inverse root scaling to prevent O(d^3) GPU stalls.",
            UserWarning,
            stacklevel=2,
        )
        # Diagonal inverse root fallback for ultra-large dimensions to prevent O(d^3) single-GPU compute stalls
        diag = torch.diag(matrix)
        inv_diag = torch.clamp(diag + eps, min=eps).pow(-1.0 / root)
        return torch.diag(inv_diag)

    # Regularize with epsilon on diagonal
    M_reg = matrix + eps * torch.eye(dim, device=matrix.device, dtype=matrix.dtype)
    try:
        L, Q = torch.linalg.eigh(M_reg)
        # Invert and take power
        L_inv_root = torch.clamp(L, min=eps).pow(-1.0 / root)
        return Q @ torch.diag(L_inv_root) @ Q.T
    except Exception:
        # Fallback to identity scaling on numerical instability
        return torch.eye(dim, device=matrix.device, dtype=matrix.dtype)


class Shampoo(Optimizer):
    """
    Shampoo tensor-preconditioned optimizer.
    Computes left and right empirical covariance matrices for 2D tensors.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-4,
        update_preconditioner_interval: int = 10,
        weight_decay: float = 0.0,
        max_preconditioner_dim: int = 8192,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta2=beta2,
            epsilon=epsilon,
            update_preconditioner_interval=update_preconditioner_interval,
            weight_decay=weight_decay,
            max_preconditioner_dim=max_preconditioner_dim,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            eps = group["epsilon"]
            interval = group["update_preconditioner_interval"]
            weight_decay = group["weight_decay"]
            max_dim = group.get("max_preconditioner_dim", 8192)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                # Initialize covariance accumulators for 2D matrices
                if p.ndim == 2:
                    dim0, dim1 = p.size()
                    if "cov_L" not in state:
                        state["cov_L"] = torch.zeros((dim0, dim0), device=p.device, dtype=p.dtype)
                        state["cov_R"] = torch.zeros((dim1, dim1), device=p.device, dtype=p.dtype)
                        state["inv_root_L"] = torch.eye(dim0, device=p.device, dtype=p.dtype)
                        state["inv_root_R"] = torch.eye(dim1, device=p.device, dtype=p.dtype)

                    # Update covariance: L = G G^T, R = G^T G
                    state["cov_L"].mul_(beta2).add_(grad @ grad.T, alpha=1.0 - beta2)
                    state["cov_R"].mul_(beta2).add_(grad.T @ grad, alpha=1.0 - beta2)

                    # Periodically recompute matrix roots (amortizing O(d^3) eigendecompositions)
                    if self._step_count % interval == 1:
                        state["inv_root_L"] = matrix_power_inverse_root(state["cov_L"], root=4, eps=eps, max_eigh_dim=max_dim)
                        state["inv_root_R"] = matrix_power_inverse_root(state["cov_R"], root=4, eps=eps, max_eigh_dim=max_dim)

                    # Precondition gradient: G_prec = L^{-1/4} G R^{-1/4}
                    preconditioned_grad = state["inv_root_L"] @ grad @ state["inv_root_R"]
                else:
                    # 1D tensors: scalar preconditioning
                    if "cov_diag" not in state:
                        state["cov_diag"] = torch.zeros_like(p.data)
                    state["cov_diag"].mul_(beta2).add_(grad * grad, alpha=1.0 - beta2)
                    preconditioned_grad = grad / (state["cov_diag"].sqrt() + eps)

                # Momentum buffer
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(preconditioned_grad)

                # Weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update parameter
                p.data.add_(buf, alpha=-lr)

        return loss
