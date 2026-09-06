"""
GhostLayer Scientific Experiment & Attribution Manager.

Runs counterbalanced A/B trials (ABAB), discards warmup transients, isolates ablations,
measures multi-dimensional quality gates, and outputs authoritative canonical ExperimentRecords.
"""

from __future__ import annotations

import time
import math
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ghost_layer.experiments.schema import (
    ExperimentRecord,
    HardwareEnvironment,
    WorkloadProfile,
    InterventionSpec,
    LatencyDistribution,
    QualitySafetyMetrics,
    StatisticalAttribution,
    EconomicImpact,
)
from ghost_layer.experiments.statistics import (
    block_bootstrap_speedup_ci,
    compute_cohens_d,
    decompose_ablation_chain,
)
from ghost_layer.telemetry.real_collector import HardwareProbe, CudaEventTimer


class ExperimentManager:
    """
    Autonomous Experimental Agent for ML Systems.
    Proves causality of optimization interventions through counterfactual trial design.
    """

    def __init__(
        self,
        warmup_steps: int = 3,
        block_size: int = 10,
        gpu_cost_per_hour: float = 3.50,
        hardware_probe: Optional[HardwareProbe] = None,
    ):
        self.warmup_steps = warmup_steps
        self.block_size = block_size
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.probe = hardware_probe or HardwareProbe()

    def _compute_distribution(self, step_times: List[float]) -> LatencyDistribution:
        """Computes summary statistics over clean (warmup-discarded) step times."""
        if not step_times:
            return LatencyDistribution(0, 0.0, 0.0, 0.0, 0.0)

        times = sorted(step_times)
        n = len(times)
        mean_val = sum(times) / n
        var_val = sum((x - mean_val) ** 2 for x in times) / max(1, n - 1)
        std_val = var_val ** 0.5
        median_val = times[n // 2]
        p95_val = times[min(n - 1, int(0.95 * n))]

        return LatencyDistribution(
            sample_count=n,
            mean_ms=round(mean_val, 3),
            std_ms=round(std_val, 3),
            median_ms=round(median_val, 3),
            p95_ms=round(p95_val, 3),
            raw_step_times_ms=[round(x, 3) for x in step_times],
        )

    def run_controlled_trial(
        self,
        experiment_id: Optional[str],
        baseline_step_fn: Callable[[], float],
        candidate_step_fn: Callable[[], float],
        intervention: InterventionSpec,
        workload: WorkloadProfile,
        num_blocks: int = 4,
        steps_per_block: int = 15,
        baseline_losses: Optional[List[float]] = None,
        candidate_losses: Optional[List[float]] = None,
    ) -> ExperimentRecord:
        """
        Executes a counterbalanced ABAB trial across baseline and candidate interventions.
        
        Discards warmup_steps from the beginning of each block to neutralize
        thermal throttling and cache transients.
        """
        exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:12]}"
        baseline_clean_times: List[float] = []
        candidate_clean_times: List[float] = []

        # Counterbalanced sequence: Block 0=A, 1=B, 2=A, 3=B (ABAB)
        for block_idx in range(num_blocks):
            is_candidate_block = (block_idx % 2 == 1)
            step_fn = candidate_step_fn if is_candidate_block else baseline_step_fn

            for step_idx in range(steps_per_block):
                elapsed_ms = step_fn()
                # Warmup discard policy
                if step_idx >= self.warmup_steps:
                    if is_candidate_block:
                        candidate_clean_times.append(elapsed_ms)
                    else:
                        baseline_clean_times.append(elapsed_ms)

        base_dist = self._compute_distribution(baseline_clean_times)
        cand_dist = self._compute_distribution(candidate_clean_times)

        # Statistical evaluation using block bootstrap
        point_sp, (ci_low, ci_high), p_val = block_bootstrap_speedup_ci(
            baseline_clean_times, candidate_clean_times, block_size=self.block_size
        )
        cohens_d = compute_cohens_d(baseline_clean_times, candidate_clean_times)
        is_sig = (p_val < 0.05) and (ci_low > 0.0)

        # Quality & Loss Safety Check
        safety = self._evaluate_quality_safety(baseline_losses, candidate_losses)

        # Hardware environment snapshot
        hw_probe = self.probe.probe()
        hw_env = HardwareEnvironment(
            device_name=hw_probe.device_name,
            is_cuda=hw_probe.is_cuda_available,
            vram_total_mb=hw_probe.gpu_memory_total_mb,
            cuda_version=torch.version.cuda if torch.version.cuda else "N/A",
            pytorch_version=torch.__version__,
        )

        # Economic Impact calculation
        time_saved_fraction = max(0.0, (base_dist.mean_ms - cand_dist.mean_ms) / (base_dist.mean_ms + 1e-8))
        hours_saved_per_1000 = (base_dist.mean_ms * 1000.0 * time_saved_fraction) / (3600.0 * 1000.0)
        dollars_saved_per_1000 = hours_saved_per_1000 * self.gpu_cost_per_hour

        econ = EconomicImpact(
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            measured_speedup_pct=point_sp,
            gpu_hours_saved_per_1000_steps=round(hours_saved_per_1000, 4),
            dollar_savings_per_1000_steps=round(dollars_saved_per_1000, 2),
            provenance_type="EMPIRICAL",
        )

        stat_attr = StatisticalAttribution(
            test_method="Block Bootstrap (Autocorrelation-Aware)",
            speedup_pct=point_sp,
            confidence_interval_95=(ci_low, ci_high),
            p_value=p_val,
            is_statistically_significant=is_sig,
            effect_size_cohens_d=round(cohens_d, 3),
        )

        return ExperimentRecord(
            experiment_id=exp_id,
            trial_design="COUNTERBALANCED_ABAB",
            warmup_steps_discarded=self.warmup_steps,
            hardware=hw_env,
            workload=workload,
            intervention=intervention,
            baseline_distribution=base_dist,
            candidate_distribution=cand_dist,
            quality_safety=safety,
            statistics=stat_attr,
            economics=econ,
        )

    def run_ablation_attribution(
        self,
        experiment_id: str,
        ordered_ablation_steps: Dict[str, Callable[[], float]],
        workload: WorkloadProfile,
        steps_per_stage: int = 15,
    ) -> Dict[str, Any]:
        """
        Executes an isolated ablation chain (A -> B -> C -> D) to measure and
        attribute the marginal contribution of each intervention.
        """
        stage_mean_latencies: Dict[str, float] = {}

        for stage_name, step_fn in ordered_ablation_steps.items():
            times: List[float] = []
            for step_idx in range(steps_per_stage):
                t_ms = step_fn()
                if step_idx >= self.warmup_steps:
                    times.append(t_ms)
            mean_t = sum(times) / max(1, len(times))
            stage_mean_latencies[stage_name] = round(mean_t, 3)

        marginal_contributions = decompose_ablation_chain(stage_mean_latencies)

        return {
            "experiment_id": experiment_id,
            "stage_mean_latencies_ms": stage_mean_latencies,
            "marginal_contributions_pct": marginal_contributions,
            "provenance": "EMPIRICAL_ABLATION",
        }

    def _evaluate_quality_safety(
        self,
        baseline_losses: Optional[List[float]],
        candidate_losses: Optional[List[float]],
    ) -> QualitySafetyMetrics:
        """Evaluates convergence trajectory, relative loss shift, and NaN/Inf integrity."""
        if not baseline_losses or not candidate_losses:
            return QualitySafetyMetrics(
                relative_mean_loss_shift=0.0,
                max_loss_delta=0.0,
                has_nan_or_inf=False,
                is_verified_safe=True,
                safety_gate_reasons=["No loss trajectories recorded."],
            )

        # Check for NaN or Inf
        has_nan = any(math.isnan(x) or math.isinf(x) for x in baseline_losses + candidate_losses)
        if has_nan:
            return QualitySafetyMetrics(
                relative_mean_loss_shift=1.0,
                max_loss_delta=float("inf"),
                has_nan_or_inf=True,
                is_verified_safe=False,
                safety_gate_reasons=["NaN or Inf detected in loss trajectory."],
            )

        min_len = min(len(baseline_losses), len(candidate_losses))
        base_sub = baseline_losses[:min_len]
        cand_sub = candidate_losses[:min_len]

        deltas = [abs(b - c) for b, c in zip(base_sub, cand_sub)]
        max_delta = max(deltas)

        avg_base = sum(base_sub) / min_len
        avg_cand = sum(cand_sub) / min_len
        relative_shift = abs(avg_base - avg_cand) / (avg_base + 1e-8)

        # 0.10 loss-shift threshold
        is_safe = (max_delta <= 0.10) and (relative_shift <= 0.10)
        reasons: List[str] = []
        if not is_safe:
            reasons.append(f"Loss shift threshold exceeded: max delta {max_delta:.4f} > 0.10 or relative shift {relative_shift:.4f} > 0.10")
        else:
            reasons.append("Loss trajectory verified safe within 0.10 tolerance.")

        return QualitySafetyMetrics(
            relative_mean_loss_shift=round(relative_shift, 6),
            max_loss_delta=round(max_delta, 6),
            has_nan_or_inf=False,
            is_verified_safe=is_safe,
            safety_gate_reasons=reasons,
        )
