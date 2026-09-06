"""
Scientific Statistical Attribution Suite: Block Bootstrap for Autocorrelated Step Series
and Ablation Decomposition.

Addresses serial correlation (thermal drift, caching, memory fragmentation) that violates
the independence assumption of naive Welch's t-tests.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


def compute_cohens_d(sample_a: List[float], sample_b: List[float]) -> float:
    """
    Computes Cohen's d effect size for two independent distributions.
    d = (mean_a - mean_b) / s_pooled
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)

    s_pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if s_pooled == 0:
        return 0.0

    return (mean_a - mean_b) / s_pooled


def _generate_blocks(series: List[float], block_size: int) -> List[List[float]]:
    """Creates overlapping moving blocks of fixed size to preserve autocorrelation."""
    n = len(series)
    if n <= block_size:
        return [series]
    return [series[i : i + block_size] for i in range(n - block_size + 1)]


def _resample_block_series(blocks: List[List[float]], target_len: int, rng: random.Random) -> List[float]:
    """Resamples blocks with replacement until reaching target sample length."""
    resampled: List[float] = []
    while len(resampled) < target_len:
        block = rng.choice(blocks)
        resampled.extend(block)
    return resampled[:target_len]


def block_bootstrap_speedup_ci(
    baseline_series: List[float],
    candidate_series: List[float],
    block_size: int = 5,
    num_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Moving Block Bootstrap for autocorrelated step-time series.
    
    Returns:
        (point_speedup_pct, (ci_lower_pct, ci_upper_pct), p_value)
    """
    if len(baseline_series) < 2 or len(candidate_series) < 2:
        return 0.0, (0.0, 0.0), 1.0

    mean_base = sum(baseline_series) / len(baseline_series)
    mean_cand = sum(candidate_series) / len(candidate_series)

    if mean_base <= 0:
        return 0.0, (0.0, 0.0), 1.0

    point_speedup = ((mean_base - mean_cand) / mean_base) * 100.0

    # Form overlapping blocks
    b_size = max(1, min(block_size, len(baseline_series) // 2, len(candidate_series) // 2))
    base_blocks = _generate_blocks(baseline_series, b_size)
    cand_blocks = _generate_blocks(candidate_series, b_size)

    rng = random.Random(seed)
    bootstrap_speedups: List[float] = []
    null_violations = 0

    for _ in range(num_resamples):
        b_resampled = _resample_block_series(base_blocks, len(baseline_series), rng)
        c_resampled = _resample_block_series(cand_blocks, len(candidate_series), rng)

        m_b = sum(b_resampled) / len(b_resampled)
        m_c = sum(c_resampled) / len(c_resampled)

        if m_b > 0:
            sp = ((m_b - m_c) / m_b) * 100.0
            bootstrap_speedups.append(sp)
            # Count null hypothesis occurrences (speedup <= 0.0)
            if sp <= 0.0:
                null_violations += 1

    if not bootstrap_speedups:
        return point_speedup, (point_speedup, point_speedup), 1.0

    bootstrap_speedups.sort()
    lower_idx = int(math.floor((alpha / 2.0) * len(bootstrap_speedups)))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * len(bootstrap_speedups))) - 1

    ci_lower = bootstrap_speedups[max(0, lower_idx)]
    ci_upper = bootstrap_speedups[min(len(bootstrap_speedups) - 1, upper_idx)]

    # Empirical p-value under null hypothesis H0: speedup <= 0
    p_value = max(1.0 / num_resamples, float(null_violations) / float(num_resamples))

    return round(point_speedup, 2), (round(ci_lower, 2), round(ci_upper, 2)), round(p_value, 4)


def decompose_ablation_chain(
    step_latencies_ordered: Dict[str, float],
) -> Dict[str, float]:
    """
    Decomposes an ablation chain:
        A (Baseline) -> B (Baseline + Int 1) -> C (Baseline + Int 1 + Int 2) -> ...
    into marginal latency reduction contributions for each isolated component.
    """
    items = list(step_latencies_ordered.items())
    if len(items) < 2:
        return {}

    contributions: Dict[str, float] = {}
    base_label, prev_ms = items[0]

    for label, curr_ms in items[1:]:
        marginal_saved_ms = max(0.0, prev_ms - curr_ms)
        marginal_pct = (marginal_saved_ms / prev_ms * 100.0) if prev_ms > 0 else 0.0
        contributions[label] = round(marginal_pct, 2)
        prev_ms = curr_ms

    return contributions
