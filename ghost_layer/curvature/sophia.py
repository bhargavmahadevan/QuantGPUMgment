"""
Sophia Optimizer: Second-order Clipped Stochastic Optimization.

Implements Sophia (Liu et al., 2023) with two explicit Hessian diagonal estimator modes:
1. **Hutchinson HVP Mode (`hessian_estimator='hvp'` or with loss closure):**
   Samples Rademacher random vectors u ~ Rademacher(±1) and evaluates exact directional
   second-order Hessian-vector products:
       diag_H = |u ⊙ (∇^2 L(θ) u)|
   via Pearlmutter central differences (`compute_hvp_pearlmutter`). Requires passing a loss closure
   to `step(closure)` or `update_hessian(loss_fn)`.

2. **Gauss-Newton / Empirical Surrogate Mode (`hessian_estimator='gauss_newton'`):**
   Estimates diagonal curvature via gradient squares (h_est = |g|^2) for standard training loops
   where forward loss closures cannot easily be re-evaluated.

References:
    Liu et al., "Sophia: A Scalable Second-Order Optimizer for Language Model Pre-training" (2023).
"""

import math
import warnings
from typing import List, Optional, Tuple, Dict, Any, Callable, Literal
import torch
from torch.optim.optimizer import Optimizer
from ghost_layer.curvature.hvp import compute_hvp_pearlmutter


class SophiaG(Optimizer):
    """
    Sophia-G (Second-order Clipped Stochastic Optimizer).
    
    Features:
    - Hutchinson Diagonal Hessian Estimation via Pearlmutter HVP (`compute_hvp_pearlmutter`)
    - Empirical Gauss-Newton curvature fallback when forward closures are unavailable
    - Element-wise update clipping by curvature: clip(m / max(h, gamma), -rho, rho)
    - Momentum tracking with decoupled weight decay
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        gamma: float = 0.01,
        k: int = 10,  # update hessian every k steps
        hvp_epsilon: float = 1e-4,
        hessian_estimator: Literal["auto", "hvp", "gauss_newton"] = "auto",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if hessian_estimator not in ("auto", "hvp", "gauss_newton"):
            raise ValueError(f"Invalid hessian_estimator: {hessian_estimator}. Must be 'auto', 'hvp', or 'gauss_newton'.")

        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            gamma=gamma,
            k=k,
            hvp_epsilon=hvp_epsilon,
            hessian_estimator=hessian_estimator,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._warned_surrogate_fallback = False

    @torch.no_grad()
    def update_hessian(self, loss_fn: Optional[Callable[[], torch.Tensor]] = None):
        """
        Updates diagonal Hessian estimate.
        
        - When loss_fn is provided or hessian_estimator='hvp':
          Evaluates exact Hessian-Vector Product via Pearlmutter central difference:
              u ~ Rademacher(±1), h_est = |u ⊙ (∇^2 L(θ) u)|
        - When loss_fn is None and hessian_estimator='gauss_newton' / 'auto':
          Uses empirical Gauss-Newton gradient-squared surrogate: h_est = |g|^2.
        """
        for group in self.param_groups:
            beta2 = group["betas"][1]
            eps_hvp = group.get("hvp_epsilon", 1e-4)
            estimator_mode = group.get("hessian_estimator", "auto")
            active_params = [p for p in group["params"] if p.requires_grad and p.grad is not None]
            if not active_params:
                continue

            if estimator_mode == "hvp" and loss_fn is None:
                raise ValueError(
                    "SophiaG configured with hessian_estimator='hvp' requires a loss evaluation closure. "
                    "Pass loss_fn to update_hessian(loss_fn) or pass closure to step(closure)."
                )

            if loss_fn is not None and estimator_mode in ("auto", "hvp"):
                # 1. True Hutchinson Estimator with Pearlmutter HVP
                rademacher_u = [
                    (torch.randint_like(p.data, low=0, high=2) * 2 - 1).float().to(device=p.device, dtype=p.dtype)
                    for p in active_params
                ]
                
                with torch.enable_grad():
                    hvps = compute_hvp_pearlmutter(
                        loss_fn=loss_fn,
                        parameters=active_params,
                        vector=rademacher_u,
                        epsilon=eps_hvp,
                    )

                if len(hvps) == len(active_params):
                    for p, u, hvp_val in zip(active_params, rademacher_u, hvps):
                        state = self.state[p]
                        if "hessian" not in state:
                            state["hessian"] = torch.zeros_like(p.data)
                        
                        diag_h_est = (u * hvp_val).abs()
                        state["hessian"].mul_(beta2).add_(diag_h_est, alpha=1.0 - beta2)
                        state["hessian_initialized"] = True
                else:
                    # Fallback if HVP failed
                    for p in active_params:
                        state = self.state[p]
                        if "hessian" not in state:
                            state["hessian"] = torch.zeros_like(p.data)
                        h_est = (p.grad * p.grad)
                        state["hessian"].mul_(beta2).add_(h_est, alpha=1.0 - beta2)
                        state["hessian_initialized"] = True
            else:
                # 2. Empirical Gauss-Newton surrogate fallback
                if not self._warned_surrogate_fallback and estimator_mode == "auto":
                    warnings.warn(
                        "SophiaG: No loss closure provided to step() or update_hessian(). "
                        "Using empirical gradient-squared curvature surrogate (|g|^2). "
                        "Pass closure to step(closure) to activate exact Pearlmutter Hutchinson HVP.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_surrogate_fallback = True

                for p in active_params:
                    state = self.state[p]
                    if "hessian" not in state:
                        state["hessian"] = torch.zeros_like(p.data)
                    h_est = (p.grad * p.grad)
                    state["hessian"].mul_(beta2).add_(h_est, alpha=1.0 - beta2)
                    state["hessian_initialized"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]
            k = group["k"]

            # Periodically update Hessian estimate
            if self._step_count % k == 1:
                if closure is not None:
                    self.update_hessian(loss_fn=closure)
                else:
                    self.update_hessian(loss_fn=None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p.data)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p.data)
                    state["hessian_initialized"] = False

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                # Decoupled Weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # EMA of gradient (Momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Skip Newton step until Hessian has been estimated at least once.
                # Zero-initialized hessian causes denom = gamma (very small), making
                # step_val = exp_avg / gamma ≈ 100× gradient, which clips to ±rho
                # on every parameter — a cold-start artifact.
                if not state.get("hessian_initialized", False):
                    # Fall back to first-order gradient descent for cold-start steps
                    p.data.add_(grad, alpha=-lr)
                    continue

                # Second-order Newton step with element-wise clipping:
                # update = clip(m / max(h, gamma), -rho, rho)
                denom = torch.clamp(hessian, min=gamma)
                step_val = exp_avg / denom
                step_clipped = torch.clamp(step_val, min=-rho, max=rho)

                # Parameter update
                p.data.add_(step_clipped, alpha=-lr)

        return loss
