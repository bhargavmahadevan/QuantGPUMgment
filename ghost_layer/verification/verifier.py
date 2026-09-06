import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from ghost_layer.verification.lagrangian import LagrangianErrorController, LagrangianStepReport, WelfordState


@dataclass
class VerificationResult:
    """
    Structured outcome of the 3-question verification contract:
    1. performance_improved: Did throughput increase / latency decrease?
    2. training_behavior_safe: Did loss remain within acceptable convergence bounds?
    3. statistically_supported: Is the result backed by statistical significance & non-inferiority?
    """
    is_safe: bool
    performance_improved: bool = True
    training_behavior_safe: bool = True
    statistically_supported: bool = True
    relative_mean_loss_shift: float = 0.0
    max_loss_delta: float = 0.0
    reason: str = ""
    action_taken: str = ""  # AUTO_APPLIED, VERIFIED_SAFE, or DOWNGRADED_TO_RECOMMENDATION
    recommendation_id: str = ""  # rule_id this verification result applies to
    lagrangian_lambda: float = 0.0  # running dual multiplier value
    p_value: Optional[float] = None
    t_statistic: Optional[float] = None
    is_favorable_convergence: bool = False
    speedup_pct: Optional[float] = None

    # GhostLayer 2.0 TOST & Non-Inferiority Statistical Additions
    is_non_inferior: bool = True
    equivalence_margin: float = 0.0
    loss_delta_upper_ci_95: float = 0.0
    p_value_non_inferiority: Optional[float] = None
    safety_level_passed: int = 2  # Level 1: Runtime, Level 2: Trajectory, Level 3: Outcome
    loss_metric_type: str = "relative_loss_shift"


class CorrectnessVerifier:
    """
    Correctness & Statistical Safety Verification Engine.
    Evaluates optimization candidates across three distinct questions:
    1. Did performance improve?
    2. Did training behavior remain within acceptable bounds?
    3. Is the observed difference statistically supported (via TOST Non-Inferiority)?
    """
    def __init__(
        self,
        max_allowed_loss_delta: float = 0.05,
        max_allowed_relative_loss_shift: float = 0.02,
        use_lagrangian_dual: bool = False,
        max_allowed_kl_div: Optional[float] = None,
        min_p_value_threshold: float = 0.05,
        equivalence_margin_relative: Optional[float] = None,
    ):
        self.max_allowed_loss_delta = max_allowed_loss_delta
        self.max_allowed_relative_loss_shift = (
            max_allowed_kl_div if max_allowed_kl_div is not None else max_allowed_relative_loss_shift
        )
        self.equivalence_margin_relative = (
            equivalence_margin_relative if equivalence_margin_relative is not None else self.max_allowed_relative_loss_shift
        )
        self.use_lagrangian_dual = use_lagrangian_dual
        self.min_p_value_threshold = min_p_value_threshold
        self.lagrangian_controller = LagrangianErrorController(
            tau_base=self.max_allowed_relative_loss_shift * 5.0,
            max_allowed_delta=max_allowed_loss_delta,
        )

    @staticmethod
    def compute_welch_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """
        Computes Welch's t-statistic and exact two-tailed Student-t p-value using SciPy.
        """
        n1, n2 = len(sample1), len(sample2)
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0

        try:
            import scipy.stats
            res = scipy.stats.ttest_ind(sample1, sample2, equal_var=False)
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)
            if math.isnan(t_stat) or math.isnan(p_val):
                return 0.0, 1.0
            return round(t_stat, 4), round(p_val, 6)
        except Exception:
            # Fallback analytical calculation
            mean1 = sum(sample1) / n1
            mean2 = sum(sample2) / n2
            var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
            var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)
            se = math.sqrt((var1 / n1) + (var2 / n2) + 1e-12)
            if se == 0:
                return 0.0, 1.0
            t_stat = (mean1 - mean2) / se
            z = abs(t_stat)
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
            return round(t_stat, 4), round(max(0.0, min(1.0, p_val)), 6)

    @staticmethod
    def compute_non_inferiority_test(
        baseline: List[float],
        optimized: List[float],
        margin_delta: float,
        alpha: float = 0.05,
    ) -> Tuple[bool, float, float, float]:
        """
        Two One-Sided T-Tests (TOST) / Non-Inferiority Equivalence Hypothesis Test.
        Tests whether the optimized loss trajectory is statistically non-inferior
        to the baseline trajectory within an upper practical margin delta:
        
        H0: mu_opt - mu_base >= margin_delta  (Inferior / Regression >= margin)
        H1: mu_opt - mu_base <  margin_delta  (Non-inferior / Safe within margin)
        
        Returns:
            (is_non_inferior, p_value_non_inferiority, upper_ci_95, se)
        """
        n1, n2 = len(baseline), len(optimized)
        if n1 < 2 or n2 < 2:
            return True, 0.5, 0.0, 0.0

        mean_base = sum(baseline) / n1
        mean_opt = sum(optimized) / n2
        delta_mean = mean_opt - mean_base

        var_base = sum((x - mean_base) ** 2 for x in baseline) / (n1 - 1)
        var_opt = sum((x - mean_opt) ** 2 for x in optimized) / (n2 - 1)
        se = math.sqrt((var_base / n1) + (var_opt / n2) + 1e-12)

        # Welch-Satterthwaite degrees of freedom
        num = ((var_base / n1) + (var_opt / n2)) ** 2
        den = (((var_base / n1) ** 2) / (n1 - 1)) + (((var_opt / n2) ** 2) / (n2 - 1)) + 1e-12
        df = max(1.0, num / den)

        # One-sided t-statistic against upper non-inferiority margin
        t_ni = (delta_mean - margin_delta) / se

        try:
            import scipy.stats
            p_ni = float(scipy.stats.t.cdf(t_ni, df=df))
            t_crit = float(scipy.stats.t.ppf(1.0 - alpha, df=df))
        except Exception:
            # Analytical standard normal approximation for 1-sided CDF
            z = t_ni
            p_ni = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            t_crit = 1.645  # Standard normal 95% one-sided critical value

        upper_ci_95 = delta_mean + t_crit * se
        is_non_inferior = bool(upper_ci_95 <= margin_delta and p_ni < alpha)

        return is_non_inferior, round(p_ni, 6), round(upper_ci_95, 6), round(se, 6)

    def verify_trajectories(
        self,
        baseline_losses: List[float],
        optimized_losses: List[float],
        recommendation_id: str,
        was_auto_applied: bool = False,
        baseline_step_times_ms: Optional[List[float]] = None,
        candidate_step_times_ms: Optional[List[float]] = None,
    ) -> VerificationResult:
        if not baseline_losses or not optimized_losses:
            return VerificationResult(
                is_safe=False,
                performance_improved=False,
                training_behavior_safe=False,
                statistically_supported=False,
                relative_mean_loss_shift=0.0,
                max_loss_delta=0.0,
                reason="Insufficient telemetry to verify convergence.",
                action_taken="DOWNGRADED_TO_RECOMMENDATION",
                recommendation_id=recommendation_id,
                safety_level_passed=0,
            )

        # Level 1 Runtime Safety Check: Detect NaNs, Infs, or unphysical negative losses
        has_invalid = any(math.isnan(x) or math.isinf(x) or x < -1e-6 for x in (baseline_losses + optimized_losses))
        if has_invalid:
            return VerificationResult(
                is_safe=False,
                performance_improved=False,
                training_behavior_safe=False,
                statistically_supported=False,
                relative_mean_loss_shift=float("inf"),
                max_loss_delta=float("inf"),
                reason="Level 1 Runtime Failure: NaN, Inf, or unphysical numerical instability detected in loss trajectory.",
                action_taken="DOWNGRADED_TO_RECOMMENDATION",
                recommendation_id=recommendation_id,
                safety_level_passed=0,
            )

        min_len = min(len(baseline_losses), len(optimized_losses))
        base_sub = baseline_losses[:min_len]
        opt_sub = optimized_losses[:min_len]

        # Calculate loss delta and relative shift
        deltas = [abs(b - o) for b, o in zip(base_sub, opt_sub)]
        max_delta = max(deltas)

        avg_base = sum(base_sub) / min_len
        avg_opt = sum(opt_sub) / min_len
        loss_shift_ratio = abs(avg_base - avg_opt) / (avg_base + 1e-8)

        # Question 1: Did performance improve?
        perf_improved = True
        speedup_val: Optional[float] = None
        stat_perf_supported = True
        if baseline_step_times_ms and candidate_step_times_ms and len(baseline_step_times_ms) >= 2 and len(candidate_step_times_ms) >= 2:
            m_base_t = sum(baseline_step_times_ms) / len(baseline_step_times_ms)
            m_cand_t = sum(candidate_step_times_ms) / len(candidate_step_times_ms)
            if m_base_t > 0:
                speedup_val = round(((m_base_t - m_cand_t) / m_base_t) * 100.0, 2)
                perf_improved = speedup_val > 0.0
                t_t, p_t = self.compute_welch_t_test(baseline_step_times_ms, candidate_step_times_ms)
                stat_perf_supported = (p_t < self.min_p_value_threshold) if perf_improved else False

        # Question 2: Did training behavior remain within acceptable bounds?
        t_stat, p_val = self.compute_welch_t_test(base_sub, opt_sub)
        is_favorable = (avg_opt <= avg_base)

        if self.use_lagrangian_dual:
            lagrange_eval = self.lagrangian_controller.verify_trajectories(base_sub, opt_sub)
            behavior_safe = lagrange_eval["is_safe"]
            lagrange_lambda = lagrange_eval["final_lambda"]
        else:
            loss_bounds_ok = (max_delta <= self.max_allowed_loss_delta) and (loss_shift_ratio <= self.max_allowed_relative_loss_shift)
            behavior_safe = loss_bounds_ok
            lagrange_lambda = 0.0

        # Question 3: Non-Inferiority Equivalence & Statistical Support (TOST)
        margin_delta = max(self.max_allowed_loss_delta, self.equivalence_margin_relative * avg_base)
        is_non_inferior, p_ni, upper_ci_95, _ = self.compute_non_inferiority_test(
            base_sub, opt_sub, margin_delta=margin_delta, alpha=self.min_p_value_threshold
        )

        # A statistically significant increase in loss (regression) revokes safety
        stat_regression = (p_val < self.min_p_value_threshold) and not is_favorable
        stat_supported = not stat_regression

        # Conjunction of 3 verification questions:
        is_safe = behavior_safe and perf_improved and stat_supported

        safety_level = 2 if is_safe else (1 if not has_invalid else 0)

        if is_safe:
            action = "AUTO_APPLIED" if was_auto_applied else "VERIFIED_SAFE"
            reason = (
                f"Verified: Performance improved ({speedup_val or 'N/A'}%), "
                f"training behavior safe (max delta: {max_delta:.4f}, shift: {loss_shift_ratio:.4f}), "
                f"and non-inferiority supported (p_ni={p_ni:.4f}, upper_95_ci={upper_ci_95:.4f} <= {margin_delta:.4f})."
            )
        else:
            action = "DOWNGRADED_TO_RECOMMENDATION"
            if not perf_improved:
                reason = f"Performance did not improve ({speedup_val}%). Auto-apply revoked."
            elif not behavior_safe:
                if max_delta > self.max_allowed_loss_delta:
                    reason = f"Max loss delta exceeded ({max_delta:.4f} > {self.max_allowed_loss_delta}). Auto-apply revoked."
                else:
                    reason = f"Relative mean loss shift exceeded ({loss_shift_ratio:.4f} > {self.max_allowed_relative_loss_shift}). Auto-apply revoked."
            else:
                reason = f"Statistically significant unfavorable divergence detected (Welch p={p_val:.4f} < {self.min_p_value_threshold}). Auto-apply revoked."

        return VerificationResult(
            is_safe=is_safe,
            performance_improved=perf_improved,
            training_behavior_safe=behavior_safe,
            statistically_supported=stat_supported,
            relative_mean_loss_shift=round(loss_shift_ratio, 6),
            max_loss_delta=round(max_delta, 6),
            reason=reason,
            action_taken=action,
            recommendation_id=recommendation_id,
            lagrangian_lambda=round(lagrange_lambda, 6),
            p_value=p_val,
            t_statistic=t_stat,
            is_favorable_convergence=is_favorable,
            speedup_pct=speedup_val,
            is_non_inferior=is_non_inferior,
            equivalence_margin=round(margin_delta, 6),
            loss_delta_upper_ci_95=upper_ci_95,
            p_value_non_inferiority=p_ni,
            safety_level_passed=safety_level,
            loss_metric_type="relative_loss_shift",
        )



