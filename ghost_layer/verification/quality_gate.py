"""
Pluggable Quality Gate & Tri-Gate Decision Framework.

Formalizes quality verification beyond scalar loss by evaluating domain-specific
metrics (validation loss, perplexity, task accuracy, BLEU/ROUGE, reward) against
pre-registered non-inferiority margins.

Integrates with Performance and Multi-Dimensional Safety gates to form the
Tri-Gate Decision Model: Performance Gate + Quality Gate + Safety Gate -> COMMIT / REJECT / ROLLBACK.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class QualityMetricType(enum.Enum):
    TRAINING_LOSS = "TRAINING_LOSS"
    VALIDATION_LOSS = "VALIDATION_LOSS"
    PERPLEXITY = "PERPLEXITY"
    TASK_ACCURACY = "TASK_ACCURACY"
    BLEU_ROUGE = "BLEU_ROUGE"
    REWARD = "REWARD"
    CUSTOM = "CUSTOM"


class QualityDirection(enum.Enum):
    LOWER_IS_BETTER = "LOWER_IS_BETTER"    # Loss, Perplexity, Error Rate
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"  # Accuracy, BLEU, ROUGE, Reward


class TriGateAction(enum.Enum):
    COMMIT = "COMMIT"        # All 3 gates passed -> permanently retain intervention
    REJECT = "REJECT"        # Performance or Quality inconclusive -> do not commit, cleanly revert
    ROLLBACK = "ROLLBACK"    # Safety violation or divergence -> immediate physical rollback


@dataclass
class QualityPolicy:
    """
    Task-specific quality specification defining the acceptable degradation boundary.
    """
    metric_type: QualityMetricType = QualityMetricType.VALIDATION_LOSS
    metric_name: str = "validation_loss"
    direction: QualityDirection = QualityDirection.LOWER_IS_BETTER
    margin_type: str = "relative"  # "relative" (e.g. 0.01 = 1%) or "absolute" (e.g. 0.05)
    margin_value: float = 0.01     # 1% allowable degradation
    confidence_level: float = 0.95

    def compute_effective_margin(self, baseline_value: float) -> float:
        """Calculates absolute delta margin allowable for this baseline."""
        if self.margin_type == "relative":
            return abs(baseline_value) * self.margin_value
        return self.margin_value


@dataclass
class QualityEvaluationResult:
    is_passed: bool
    is_non_inferior: bool
    baseline_value: float
    candidate_value: float
    delta_observed: float
    effective_margin: float
    p_value: float
    upper_ci_95: float
    metric_name: str
    details: str


class QualityGate:
    """
    Evaluates candidate optimization against a pre-registered QualityPolicy
    using One-Sided Non-Inferiority Hypothesis Testing.
    """
    def __init__(self, policy: Optional[QualityPolicy] = None):
        self.policy = policy or QualityPolicy()

    def evaluate(
        self,
        baseline_samples: List[float],
        candidate_samples: List[float],
    ) -> QualityEvaluationResult:
        n1, n2 = len(baseline_samples), len(candidate_samples)
        if n1 == 0 or n2 == 0:
            return QualityEvaluationResult(
                is_passed=False,
                is_non_inferior=False,
                baseline_value=0.0,
                candidate_value=0.0,
                delta_observed=0.0,
                effective_margin=0.0,
                p_value=1.0,
                upper_ci_95=0.0,
                metric_name=self.policy.metric_name,
                details="Insufficient quality samples provided for verification.",
            )

        mean_base = sum(baseline_samples) / n1
        mean_cand = sum(candidate_samples) / n2
        margin_delta = self.policy.compute_effective_margin(mean_base)

        if n1 < 2 or n2 < 2:
            # Single-point comparison
            if self.policy.direction == QualityDirection.LOWER_IS_BETTER:
                degradation = mean_cand - mean_base
            else:
                degradation = mean_base - mean_cand
            passed = degradation <= margin_delta
            return QualityEvaluationResult(
                is_passed=passed,
                is_non_inferior=passed,
                baseline_value=round(mean_base, 4),
                candidate_value=round(mean_cand, 4),
                delta_observed=round(degradation, 4),
                effective_margin=round(margin_delta, 4),
                p_value=0.05 if passed else 0.50,
                upper_ci_95=round(degradation, 4),
                metric_name=self.policy.metric_name,
                details=f"Single-point check: degradation={degradation:.4f} <= margin={margin_delta:.4f}",
            )

        # Multi-sample Two One-Sided T-Test (TOST) / Non-Inferiority
        var_base = sum((x - mean_base) ** 2 for x in baseline_samples) / (n1 - 1)
        var_cand = sum((x - mean_cand) ** 2 for x in candidate_samples) / (n2 - 1)
        se = math.sqrt((var_base / n1) + (var_cand / n2) + 1e-12)

        num = ((var_base / n1) + (var_cand / n2)) ** 2
        den = (((var_base / n1) ** 2) / (n1 - 1)) + (((var_cand / n2) ** 2) / (n2 - 1)) + 1e-12
        df = max(1.0, num / den)

        if self.policy.direction == QualityDirection.LOWER_IS_BETTER:
            # We want mean_cand - mean_base <= margin_delta
            delta_observed = mean_cand - mean_base
        else:
            # For accuracy/reward, we want mean_base - mean_cand <= margin_delta
            delta_observed = mean_base - mean_cand

        t_ni = (delta_observed - margin_delta) / se

        try:
            import scipy.stats
            p_ni = float(scipy.stats.t.cdf(t_ni, df=df))
            t_crit = float(scipy.stats.t.ppf(self.policy.confidence_level, df=df))
        except Exception:
            # Standard normal fallback
            p_ni = 0.5 * (1.0 + math.erf(t_ni / math.sqrt(2.0)))
            t_crit = 1.645

        upper_ci = delta_observed + t_crit * se
        alpha = 1.0 - self.policy.confidence_level
        is_non_inferior = bool(upper_ci <= margin_delta and p_ni < (alpha + 1e-6))
        is_passed = is_non_inferior

        details = (
            f"Quality non-inferiority: degradation={delta_observed:.4f}, "
            f"upper_95_ci={upper_ci:.4f} <= margin={margin_delta:.4f} (p={p_ni:.4f})"
        )

        return QualityEvaluationResult(
            is_passed=is_passed,
            is_non_inferior=is_non_inferior,
            baseline_value=round(mean_base, 4),
            candidate_value=round(mean_cand, 4),
            delta_observed=round(delta_observed, 4),
            effective_margin=round(margin_delta, 4),
            p_value=round(p_ni, 6),
            upper_ci_95=round(upper_ci, 6),
            metric_name=self.policy.metric_name,
            details=details,
        )


@dataclass
class TriGateEvaluationResult:
    action: TriGateAction
    performance_passed: bool
    quality_passed: bool
    safety_passed: bool
    speedup_pct: float
    quality_result: QualityEvaluationResult
    safety_details: str
    decision_reason: str


class TriGateEvaluator:
    """
    Unified Tri-Gate Decision Evaluator:
    Performance Gate + Quality Gate + Multi-Dimensional Safety Gate -> COMMIT / REJECT / ROLLBACK.
    """
    def __init__(
        self,
        min_speedup_pct: float = 1.0,
        quality_policy: Optional[QualityPolicy] = None,
    ):
        self.min_speedup_pct = min_speedup_pct
        self.quality_gate = QualityGate(policy=quality_policy)

    def evaluate(
        self,
        baseline_step_times_ms: List[float],
        candidate_step_times_ms: List[float],
        baseline_quality_samples: List[float],
        candidate_quality_samples: List[float],
        safety_is_safe: bool = True,
        safety_failure_reasons: Optional[List[str]] = None,
    ) -> TriGateEvaluationResult:
        safety_reasons = safety_failure_reasons or []

        # 1. Performance Gate
        n_b = len(baseline_step_times_ms)
        n_c = len(candidate_step_times_ms)
        if n_b > 0 and n_c > 0:
            m_b = sum(baseline_step_times_ms) / n_b
            m_c = sum(candidate_step_times_ms) / n_c
            speedup_pct = ((m_b - m_c) / m_b) * 100.0 if m_b > 0 else 0.0
        else:
            speedup_pct = 0.0
        perf_passed = speedup_pct >= self.min_speedup_pct

        # 2. Quality Gate
        qual_res = self.quality_gate.evaluate(baseline_quality_samples, candidate_quality_samples)
        qual_passed = qual_res.is_passed

        # 3. Safety Gate
        safety_passed = safety_is_safe and (len(safety_reasons) == 0)

        # Deterministic Decision Tri-Gate
        if not safety_passed:
            action = TriGateAction.ROLLBACK
            reason = f"Safety Gate Violated ({', '.join(safety_reasons)}). Physical rollback triggered."
        elif not qual_passed:
            action = TriGateAction.ROLLBACK
            reason = f"Quality Gate Violated ({qual_res.details}). Model degradation exceeded margin."
        elif not perf_passed:
            action = TriGateAction.REJECT
            reason = f"Performance Gate Inconclusive: Speedup ({speedup_pct:.1f}%) below minimum practical threshold ({self.min_speedup_pct:.1f}%)."
        else:
            action = TriGateAction.COMMIT
            reason = f"Tri-Gate Passed: Speedup (+{speedup_pct:.1f}%), Quality Verified ({qual_res.metric_name} preserved), Safety Clean."

        return TriGateEvaluationResult(
            action=action,
            performance_passed=perf_passed,
            quality_passed=qual_passed,
            safety_passed=safety_passed,
            speedup_pct=round(speedup_pct, 2),
            quality_result=qual_res,
            safety_details="Clean" if safety_passed else ", ".join(safety_reasons),
            decision_reason=reason,
        )
