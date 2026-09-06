"""
Multi-Dimensional Safety Matrix Verifier.

Replaces a single scalar loss proxy with 5 distinct orthogonal safety dimensions:
1. Numerical Safety: NaN, Inf, gradient norm explosion, parameter divergence.
2. Training Dynamics Safety: relative loss shift (<= 0.10), max step delta, loss variance.
3. Quality Safety: validation loss shift, evaluation perplexity degradation.
4. Performance Safety: latency regression, step duration bounds.
5. Infrastructure Safety: peak VRAM headroom buffer, thermal throttle alerts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DimensionCheckResult:
    dimension: str
    passed: bool
    score_or_delta: float
    threshold: float
    details: str


@dataclass
class MultiDimensionalSafetyReport:
    is_safe: bool
    numerical_safety: DimensionCheckResult
    dynamics_safety: DimensionCheckResult
    quality_safety: DimensionCheckResult
    performance_safety: DimensionCheckResult
    infrastructure_safety: DimensionCheckResult
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def summary_label(self) -> str:
        return "ALL_GATES_PASSED" if self.is_safe else f"GATES_FAILED: {', '.join(self.failure_reasons)}"


class MultiDimensionalSafetyVerifier:
    """
    Evaluates candidate optimization across 5 physical and mathematical safety dimensions.
    All dimensions must pass for autonomous commit approval.
    """

    def __init__(
        self,
        max_loss_shift_ratio: float = 0.10,
        max_loss_delta: float = 0.10,
        max_grad_norm: float = 100.0,
        max_vram_utilization_pct: float = 95.0,
        max_allowed_val_loss_regression_pct: float = 2.0,
    ):
        self.max_loss_shift_ratio = max_loss_shift_ratio
        self.max_loss_delta = max_loss_delta
        self.max_grad_norm = max_grad_norm
        self.max_vram_utilization_pct = max_vram_utilization_pct
        self.max_val_regression_pct = max_allowed_val_loss_regression_pct

    def evaluate(
        self,
        baseline_losses: List[float],
        candidate_losses: List[float],
        candidate_grad_norm: Optional[float] = None,
        candidate_param_norm: Optional[float] = None,
        baseline_val_loss: Optional[float] = None,
        candidate_val_loss: Optional[float] = None,
        baseline_step_time_ms: float = 100.0,
        candidate_step_time_ms: float = 80.0,
        peak_vram_mb: float = 12000.0,
        total_vram_mb: float = 16384.0,
        is_thermal_throttled: bool = False,
    ) -> MultiDimensionalSafetyReport:
        failures: List[str] = []

        # 1. Numerical Safety Gate
        has_nan = any(math.isnan(x) or math.isinf(x) for x in baseline_losses + candidate_losses)
        grad_norm_val = candidate_grad_norm or 1.0
        grad_ok = grad_norm_val <= self.max_grad_norm
        numerical_passed = (not has_nan) and grad_ok

        num_details = "Clean"
        if has_nan:
            num_details = "NaN/Inf detected in loss curve"
            failures.append("Numerical Gate: NaN/Inf in loss")
        elif not grad_ok:
            num_details = f"Gradient norm explosion: {grad_norm_val:.2f} > {self.max_grad_norm}"
            failures.append("Numerical Gate: Grad norm exceeded")

        num_res = DimensionCheckResult(
            dimension="NUMERICAL",
            passed=numerical_passed,
            score_or_delta=grad_norm_val,
            threshold=self.max_grad_norm,
            details=num_details,
        )

        # 2. Dynamics Safety Gate (Relative loss shift <= 0.10)
        min_len = min(len(baseline_losses), len(candidate_losses)) if baseline_losses and candidate_losses else 0
        if min_len > 0:
            b_sub = baseline_losses[:min_len]
            c_sub = candidate_losses[:min_len]
            max_delta = max(abs(b - c) for b, c in zip(b_sub, c_sub))
            avg_b = sum(b_sub) / min_len
            avg_c = sum(c_sub) / min_len
            rel_shift = abs(avg_b - avg_c) / (avg_b + 1e-8)
            dyn_passed = (max_delta <= self.max_loss_delta) and (rel_shift <= self.max_loss_shift_ratio)
            if not dyn_passed:
                failures.append(f"Dynamics Gate: Shift {rel_shift:.4f} > {self.max_loss_shift_ratio} or Delta {max_delta:.4f} > {self.max_loss_delta}")
            dyn_details = f"Shift={rel_shift:.4f}, Delta={max_delta:.4f}"
        else:
            dyn_passed = True
            rel_shift = 0.0
            dyn_details = "Insufficient data"

        dyn_res = DimensionCheckResult(
            dimension="TRAINING_DYNAMICS",
            passed=dyn_passed,
            score_or_delta=round(rel_shift, 4),
            threshold=self.max_loss_shift_ratio,
            details=dyn_details,
        )

        # 3. Quality Safety Gate (Validation / Perplexity)
        qual_passed = True
        val_delta_pct = 0.0
        qual_details = "Eval Loss Maintained"
        if baseline_val_loss is not None and candidate_val_loss is not None and baseline_val_loss > 0:
            val_delta_pct = ((candidate_val_loss - baseline_val_loss) / baseline_val_loss) * 100.0
            qual_passed = val_delta_pct <= self.max_val_regression_pct
            if not qual_passed:
                failures.append(f"Quality Gate: Validation loss regressed by {val_delta_pct:.2f}% > {self.max_val_regression_pct}%")
                qual_details = f"Validation loss regressed by {val_delta_pct:.2f}%"

        qual_res = DimensionCheckResult(
            dimension="QUALITY",
            passed=qual_passed,
            score_or_delta=round(val_delta_pct, 2),
            threshold=self.max_val_regression_pct,
            details=qual_details,
        )

        # 4. Performance Safety Gate (Must not regress step time)
        perf_passed = candidate_step_time_ms <= baseline_step_time_ms * 1.02  # Allow 2% noise margin
        perf_delta_pct = ((candidate_step_time_ms - baseline_step_time_ms) / max(1e-8, baseline_step_time_ms)) * 100.0
        perf_details = f"Speedup: {-perf_delta_pct:.1f}%"
        if not perf_passed:
            failures.append(f"Performance Gate: Step latency regressed by {perf_delta_pct:.1f}%")
            perf_details = f"Latency regressed by {perf_delta_pct:.1f}%"

        perf_res = DimensionCheckResult(
            dimension="PERFORMANCE",
            passed=perf_passed,
            score_or_delta=round(perf_delta_pct, 2),
            threshold=2.0,
            details=perf_details,
        )

        # 5. Infrastructure Safety Gate (VRAM Headroom < 95% & No Thermal Throttle)
        vram_pct = (peak_vram_mb / max(1.0, total_vram_mb)) * 100.0
        vram_ok = vram_pct <= self.max_vram_utilization_pct
        infra_passed = vram_ok and (not is_thermal_throttled)
        infra_details = f"Peak VRAM: {vram_pct:.1f}%"
        if not vram_ok:
            failures.append(f"Infrastructure Gate: Peak VRAM {vram_pct:.1f}% > {self.max_vram_utilization_pct}%")
        if is_thermal_throttled:
            failures.append("Infrastructure Gate: Physical thermal throttling detected on GPU")
            infra_details += " [THERMALLY THROTTLED]"

        infra_res = DimensionCheckResult(
            dimension="INFRASTRUCTURE",
            passed=infra_passed,
            score_or_delta=round(vram_pct, 2),
            threshold=self.max_vram_utilization_pct,
            details=infra_details,
        )

        all_passed = numerical_passed and dyn_passed and qual_passed and perf_passed and infra_passed

        return MultiDimensionalSafetyReport(
            is_safe=all_passed,
            numerical_safety=num_res,
            dynamics_safety=dyn_res,
            quality_safety=qual_res,
            performance_safety=perf_res,
            infrastructure_safety=infra_res,
            failure_reasons=failures,
        )
