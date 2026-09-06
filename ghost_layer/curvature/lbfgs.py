"""
Matrix-Free Quasi-Newton (L-BFGS) Curvature Engine.

Implements Limited-Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization with:
1. Nocedal Two-Loop Recursion: Computes inverse Hessian-gradient direction H_k g_k
   matrix-free in O(m * d) operations.
2. Damped Powell Safeguard: Enforces strict positive-definiteness (s_k^T y_k > 0)
   on non-convex neural network loss landscapes, preventing negative curvature collapse.
3. Zero Memory Leakage: Bounded ring buffer of depth m with explicit tensor detachment
   (.detach()), completely eliminating autograd graph retention and O(d^2) dense matrix allocations.
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
from collections import deque
import torch
from torch.optim.optimizer import Optimizer


def damped_powell_update(
    s: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 1.0,
    sigma: float = 0.2,
) -> Tuple[torch.Tensor, bool]:
    """
    Applies Powell's damping to the displacement vector y:
        s = theta_{k+1} - theta_k
        y = g_{k+1} - g_k

    When s^T y < sigma * s^T (B_0 s), non-convexity or negative curvature can break
    positive-definiteness of the BFGS update.
    Powell's modification smoothly interpolates:
        y_tilde = theta * y + (1 - theta) * B_0 s
    guaranteeing:
        s^T y_tilde >= sigma * s^T (B_0 s) > 0.
    """
    s_dot_y = torch.dot(s, y).item()
    s_norm_sq = torch.dot(s, s).item()
    b0_s_dot_s = gamma * s_norm_sq

    threshold = sigma * b0_s_dot_s

    if s_dot_y >= threshold:
        return y, False

    # Damping required
    denom = b0_s_dot_s - s_dot_y
    if abs(denom) < 1e-12:
        theta_val = 0.0
    else:
        theta_val = ( (1.0 - sigma) * b0_s_dot_s ) / denom

    theta_val = max(0.0, min(1.0, theta_val))
    b0_s = gamma * s
    y_damped = theta_val * y + (1.0 - theta_val) * b0_s
    return y_damped, True


def lbfgs_two_loop_recursion(
    grad: torch.Tensor,
    s_history: List[torch.Tensor],
    y_history: List[torch.Tensor],
    damping_gamma: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the Quasi-Newton search direction r = - H_k grad using Nocedal's
    two-loop recursion algorithm.

    Complexity:
        Time: O(m * d)
        Memory: O(m * d) (No d x d matrix allocations)
    """
    m = len(s_history)
    if m == 0:
        return -grad

    q = grad.clone()
    alphas: List[float] = [0.0] * m
    rhos: List[float] = [0.0] * m

    # 1. Backward loop (from k-1 down to k-m)
    for i in reversed(range(m)):
        s_i = s_history[i]
        y_i = y_history[i]
        s_dot_y = torch.dot(s_i, y_i).item()
        
        rho_i = 1.0 / max(1e-10, s_dot_y)
        rhos[i] = rho_i
        
        alpha_i = rho_i * torch.dot(s_i, q).item()
        alphas[i] = alpha_i
        
        q.add_(y_i, alpha=-alpha_i)

    # 2. Initial Hessian scaling: gamma_k = (s_{k-1}^T y_{k-1}) / ||y_{k-1}||^2
    s_last = s_history[-1]
    y_last = y_history[-1]
    y_norm_sq = torch.dot(y_last, y_last).item()
    
    if damping_gamma is not None:
        gamma_0 = damping_gamma
    elif y_norm_sq > 1e-10:
        gamma_0 = torch.dot(s_last, y_last).item() / y_norm_sq
    else:
        gamma_0 = 1.0

    r = q * gamma_0

    # 3. Forward loop (from k-m up to k-1)
    for i in range(m):
        s_i = s_history[i]
        y_i = y_history[i]
        rho_i = rhos[i]
        alpha_i = alphas[i]

        beta_i = rho_i * torch.dot(y_i, r).item()
        r.add_(s_i, alpha=(alpha_i - beta_i))

    return -r


class LBFGSCurvatureOptimizer(Optimizer):
    """
    Limited-Memory BFGS (L-BFGS) Optimizer with Damped Powell Curvature Updates.

    Eliminates dense O(d^2) matrix memory leaks by maintaining a bounded ring buffer
    of detached displacement vectors.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        history_size: int = 10,
        powell_damping: bool = True,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if history_size < 1:
            raise ValueError(f"Invalid history size: {history_size}")

        defaults = dict(
            lr=lr,
            history_size=history_size,
            powell_damping=powell_damping,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            m_history = group["history_size"]
            use_damping = group["powell_damping"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LBFGSCurvatureOptimizer does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["prev_param"] = p.data.clone().detach()
                    state["prev_grad"] = grad.clone().detach()
                    state["s_history"] = deque(maxlen=m_history)
                    state["y_history"] = deque(maxlen=m_history)

                step = state["step"]
                prev_param = state["prev_param"]
                prev_grad = state["prev_grad"]
                s_hist = state["s_history"]
                y_hist = state["y_history"]

                # Weight decay
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Compute displacement vectors from previous step (step >= 1)
                if step > 0:
                    s_k = (p.data - prev_param).flatten().detach()
                    y_k = (grad - prev_grad).flatten().detach()

                    s_norm = torch.norm(s_k).item()
                    y_norm = torch.norm(y_k).item()

                    # Only register non-trivial steps
                    if s_norm > 1e-10 and y_norm > 1e-10:
                        if use_damping:
                            y_k_damped, _ = damped_powell_update(s_k, y_k)
                        else:
                            y_k_damped = y_k

                        s_hist.append(s_k)
                        y_hist.append(y_k_damped)

                # Compute Quasi-Newton direction via Two-Loop Recursion
                grad_flat = grad.flatten()
                if len(s_hist) > 0:
                    direction_flat = lbfgs_two_loop_recursion(
                        grad_flat,
                        list(s_hist),
                        list(y_hist),
                    )
                else:
                    direction_flat = -grad_flat

                direction = direction_flat.view_as(p.data)

                # Cache current values before parameter update
                prev_param.copy_(p.data)
                prev_grad.copy_(grad)

                # Parameter update
                p.data.add_(direction, alpha=lr)
                state["step"] += 1

        return loss
