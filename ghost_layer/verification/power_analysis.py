"""
Pre-Trial Statistical Power Analysis & Sample Size Sizing.

Calculates the required sample size N BEFORE trial execution based on:
- Minimum Detectable Effect (MDE)
- Baseline variance / noise estimate
- Significance level alpha (default 0.05)
- Statistical power 1 - beta (default 0.80)

Enforces sample size sufficiency to prevent p-hacking and premature stopping bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PowerAnalysisSpec:
    mde_relative: float = 0.05        # 5% minimum detectable effect
    baseline_mean: float = 100.0      # Expected mean step time or loss
    baseline_std: float = 2.0         # Expected standard deviation
    alpha: float = 0.05               # Type I error rate
    power: float = 0.80               # Type II error guarantee (1 - beta)
    non_inferiority_margin: float = 0.01  # Acceptable degradation boundary (1%)


@dataclass
class PowerAnalysisResult:
    required_samples_per_group: int
    minimum_detectable_effect_absolute: float
    effective_cohens_d: float
    is_sufficient_sample: bool
    observed_sample_count: int
    alpha: float
    power: float
    recommendation: str


class SamplePowerAnalyzer:
    """
    Computes required N for two-sample comparisons and non-inferiority trials.
    """

    @classmethod
    def calculate_required_sample_size(
        cls,
        spec: PowerAnalysisSpec,
    ) -> int:
        """
        Calculates required sample size per arm:
        N = 2 * ((z_{1 - alpha/2} + z_{1 - beta}) * sigma / delta)^2
        """
        # Standard normal quantiles
        z_alpha = 1.96 if spec.alpha <= 0.05 else 1.645
        z_power = 0.84 if spec.power <= 0.80 else (1.28 if spec.power <= 0.90 else 1.645)

        delta = max(1e-6, spec.baseline_mean * spec.mde_relative)
        sigma = max(1e-6, spec.baseline_std)

        n_required = 2.0 * (((z_alpha + z_power) * sigma / delta) ** 2)
        # Minimum practical sample size is 5 steps to establish variance
        return max(5, int(math.ceil(n_required)))

    @classmethod
    def evaluate_sample_adequacy(
        cls,
        observed_baseline_count: int,
        observed_candidate_count: int,
        spec: Optional[PowerAnalysisSpec] = None,
    ) -> PowerAnalysisResult:
        analysis_spec = spec or PowerAnalysisSpec()
        n_req = cls.calculate_required_sample_size(analysis_spec)

        min_observed = min(observed_baseline_count, observed_candidate_count)
        is_adequate = min_observed >= n_req

        delta = analysis_spec.baseline_mean * analysis_spec.mde_relative
        cohens_d = delta / max(1e-6, analysis_spec.baseline_std)

        if is_adequate:
            rec = f"Sample size (N={min_observed}) meets or exceeds required power threshold (N_req={n_req})."
        else:
            rec = (
                f"INSUFFICIENT_POWER: Observed samples (N={min_observed}) below statistical power requirement "
                f"(N_req={n_req}). Trial outcome risks being inconclusive or underpowered."
            )

        return PowerAnalysisResult(
            required_samples_per_group=n_req,
            minimum_detectable_effect_absolute=round(delta, 4),
            effective_cohens_d=round(cohens_d, 3),
            is_sufficient_sample=is_adequate,
            observed_sample_count=min_observed,
            alpha=analysis_spec.alpha,
            power=analysis_spec.power,
            recommendation=rec,
        )
