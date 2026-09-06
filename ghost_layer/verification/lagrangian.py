"""
Lagrangian Error & Dual Safety Controller for GhostLayer.

Formulates training safety as a constrained primal-dual optimization problem:
    min_θ L_CE(θ)  subject to  D_drift(θ, θ_base) <= τ_safe

Updates Lagrange multiplier λ dynamically:
    λ_{t+1} = max(0, λ_t + η_λ * (Δ_loss^{(t)} - τ_adaptive))

Tracks online Welford variance to distinguish stochastic batch variance from true diverge.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math


@dataclass
class WelfordState:
    """Online incremental mean and variance tracker (Welford algorithm)."""
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, val: float) -> None:
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return (self.M2 / (self.count - 1)) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))


@dataclass
class LagrangianStepReport:
    """Detailed step-level snapshot of the Lagrangian constraint plane."""
    step: int
    base_loss: float
    opt_loss: float
    loss_delta: float
    relative_drift: float
    lambda_multiplier: float
    running_mean_base: float
    running_std_base: float
    running_mean_opt: float
    running_std_opt: float
    is_safe: bool
    adaptive_tolerance: float
    dual_violation: float


class LagrangianErrorController:
    """
    Adaptive Primal-Dual Safety & Cross-Entropy Preservation Controller.
    
    Dynamically adjusts the safety tolerance based on batch noise SNR and accumulates
    dual penalty λ when model trajectories exceed safe divergence bounds.
    """

    def __init__(
        self,
        tau_base: float = 0.10,
        eta_lambda: float = 0.05,
        max_allowed_delta: float = 0.05,
        window_size: int = 50,
        initial_lambda: float = 0.0,
    ):
        self.tau_base = tau_base
        self.eta_lambda = eta_lambda
        self.max_allowed_delta = max_allowed_delta
        self.window_size = window_size
        self.lambda_val = initial_lambda

        self.welford_base = WelfordState()
        self.welford_opt = WelfordState()
        self.step_count = 0
        self.history: List[LagrangianStepReport] = []

    def evaluate_step(self, base_loss: float, opt_loss: float) -> LagrangianStepReport:
        """
        Processes a single training step's cross-entropy losses and updates the dual multiplier.
        """
        self.step_count += 1
        self.welford_base.update(base_loss)
        self.welford_opt.update(opt_loss)

        loss_delta = abs(base_loss - opt_loss)
        relative_drift = loss_delta / (abs(base_loss) + 1e-8)

        # Adaptive tolerance based on base loss variance (noise SNR)
        base_std = self.welford_base.std
        base_mean = self.welford_base.mean
        coeff_variation = (base_std / (abs(base_mean) + 1e-8)) if self.step_count > 5 else 0.0
        
        # Scale tolerance upwards slightly for noisy datasets, bounded within [0.05, 0.15]
        adaptive_tau = max(0.05, min(0.15, self.tau_base * (1.0 + 0.5 * coeff_variation)))

        # Primal-dual multiplier update: λ_{t+1} = max(0, λ_t + η_λ * (relative_drift - adaptive_tau))
        constraint_violation = relative_drift - adaptive_tau
        self.lambda_val = max(0.0, self.lambda_val + self.eta_lambda * constraint_violation)

        # Safety decision:
        # 1. Point delta must not exceed max point threshold
        # 2. Accumulated dual penalty λ must not exceed 1.0 (indicating sustained drift)
        # 3. Relative drift must be within adaptive bound
        is_safe = (loss_delta <= self.max_allowed_delta) and (self.lambda_val < 1.0) and (relative_drift <= adaptive_tau)

        report = LagrangianStepReport(
            step=self.step_count,
            base_loss=base_loss,
            opt_loss=opt_loss,
            loss_delta=round(loss_delta, 6),
            relative_drift=round(relative_drift, 6),
            lambda_multiplier=round(self.lambda_val, 6),
            running_mean_base=round(self.welford_base.mean, 6),
            running_std_base=round(self.welford_base.std, 6),
            running_mean_opt=round(self.welford_opt.mean, 6),
            running_std_opt=round(self.welford_opt.std, 6),
            is_safe=is_safe,
            adaptive_tolerance=round(adaptive_tau, 6),
            dual_violation=round(constraint_violation, 6),
        )

        self.history.append(report)
        if len(self.history) > 1000:
            self.history.pop(0)

        return report

    def verify_trajectories(
        self,
        baseline_losses: List[float],
        optimized_losses: List[float]
    ) -> Dict[str, Any]:
        """Evaluates complete sequences of baseline vs optimized loss trajectories."""
        self.reset()
        min_len = min(len(baseline_losses), len(optimized_losses))
        if min_len == 0:
            return {
                "is_safe": False,
                "reason": "Empty loss trajectory provided.",
                "final_lambda": 0.0,
                "max_drift": 0.0,
                "steps_evaluated": 0,
            }

        reports = [
            self.evaluate_step(baseline_losses[i], optimized_losses[i])
            for i in range(min_len)
        ]

        unsafe_steps = [r for r in reports if not r.is_safe]
        max_drift = max(r.relative_drift for r in reports)
        final_lambda = self.lambda_val

        overall_safe = (len(unsafe_steps) / min_len <= 0.05) and (final_lambda < 0.8)

        return {
            "is_safe": overall_safe,
            "final_lambda": round(final_lambda, 6),
            "max_drift": round(max_drift, 6),
            "unsafe_step_count": len(unsafe_steps),
            "steps_evaluated": min_len,
            "reason": (
                f"Lagrangian constraint satisfied (λ={final_lambda:.4f} < 0.8, max drift: {max_drift:.4f})."
                if overall_safe
                else f"Lagrangian dual penalty exceeded (λ={final_lambda:.4f} >= 0.8, unsafe steps: {len(unsafe_steps)}/{min_len})."
            ),
        }

    def reset(self) -> None:
        """Resets running state for new trajectory verification."""
        self.lambda_val = 0.0
        self.welford_base = WelfordState()
        self.welford_opt = WelfordState()
        self.step_count = 0
        self.history.clear()
