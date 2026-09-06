from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class ROIAuditReport:
    baseline_gpu_hours: float
    optimized_gpu_hours: float
    gpu_hours_saved: float
    time_reduction_pct: float
    gpu_cost_per_hour: float
    baseline_cost_usd: float
    optimized_cost_usd: float
    net_client_savings_usd: float
    performance_fee_usd: float
    performance_fee_rate_pct: float
    total_training_steps: int
    # Containment premium (compute leak recovery)
    leaked_gpu_hours: float = 0.0
    leaked_cost_usd: float = 0.0
    containment_premium_usd: float = 0.0
    containment_premium_rate_pct: float = 0.0
    total_ghost_layer_revenue_usd: float = 0.0
    # Active commercial model: Flat-fee pre-flight audit guarantee
    audit_fee_usd: float = 2500.0
    audit_roi_multiple: float = 0.0
    # Multi-Tier Economics (GhostLayer 2.0 Hierarchy)
    cost_per_step_baseline_usd: float = 0.0
    cost_per_step_optimized_usd: float = 0.0
    cost_per_million_tokens_baseline_usd: float = 0.0
    cost_per_million_tokens_optimized_usd: float = 0.0
    realized_measured_savings_usd: float = 0.0
    projected_annual_savings_usd: float = 0.0
    # Verifiable ROI provenance & integrity fields
    roi_status: str = "CALCULATED"  # "CALCULATED", "VERIFIED", "INVALID"
    is_verified: bool = False
    baseline_lock_id: Optional[str] = None
    welch_p_value: Optional[float] = None
    speedup_ci_95: Optional[Tuple[float, float]] = None
    provenance_note: str = ""


class ROICalculator:
    """
    Calculates exact GPU-hour and financial cost savings (baseline vs. optimized).
    
    Commercial Model Alignment (see docs/understand_me/12_HOW_WE_MAKE_MONEY.md):
    - Active Commercial Model: Flat-Fee Diagnostic Pre-Flight Audits ($2,500) and
      Team Platform SaaS ($199–$999/mo).
    - Variable performance fees (e.g. 25%) and containment premiums (e.g. 10%) are
      deprecated due to baseline counterfactual attribution conflicts and legal liability.
      Defaults are set to 0.0%. Callers can explicitly specify fee rates for historical
      or hypothetical modeling comparisons.
    - Produces CALCULATED vs VERIFIED ROI status depending on whether a cryptographic
      BaselineLock and Welch p-value < 0.05 are present.
    """
    def __init__(
        self,
        gpu_cost_per_hour: float = 3.50,
        performance_fee_rate_pct: float = 0.0,
        containment_premium_rate_pct: float = 0.0,
        audit_fee_usd: float = 2500.0,
    ):
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.performance_fee_rate_pct = performance_fee_rate_pct
        self.containment_premium_rate_pct = containment_premium_rate_pct
        self.audit_fee_usd = audit_fee_usd

    def calculate(
        self,
        baseline_step_time_ms: float,
        optimized_step_time_ms: float,
        total_training_steps: int,
        num_gpus: int = 8,
        leaked_gpu_hours: float = 0.0,
        baseline_lock: Optional[Any] = None,
        welch_p_value: Optional[float] = None,
        speedup_ci_95: Optional[Tuple[float, float]] = None,
        candidate_metadata: Optional[Dict[str, Any]] = None,
    ) -> ROIAuditReport:
        # Convert step_time_ms * total_steps to hours: divide by 3,600,000 ms/hour
        steps_to_hours = total_training_steps / 3_600_000.0
        gpu_hours_multiplier = steps_to_hours * num_gpus

        baseline_gpu_hours = baseline_step_time_ms * gpu_hours_multiplier
        optimized_gpu_hours = optimized_step_time_ms * gpu_hours_multiplier

        gpu_hours_saved = max(0.0, baseline_gpu_hours - optimized_gpu_hours)
        time_reduction_pct = ((baseline_step_time_ms - optimized_step_time_ms) / baseline_step_time_ms * 100.0) if baseline_step_time_ms > 0 else 0.0

        baseline_cost = baseline_gpu_hours * self.gpu_cost_per_hour
        optimized_cost = optimized_gpu_hours * self.gpu_cost_per_hour
        gross_savings = baseline_cost - optimized_cost

        performance_fee = gross_savings * (self.performance_fee_rate_pct / 100.0)
        net_client_savings = gross_savings - performance_fee

        # Containment premium on leaked/wasted compute
        leaked_cost = leaked_gpu_hours * self.gpu_cost_per_hour
        containment_premium = leaked_cost * (self.containment_premium_rate_pct / 100.0)
        total_revenue = performance_fee + containment_premium
        audit_roi_multiple = round(gross_savings / max(1.0, self.audit_fee_usd), 2)

        # Determine Verifiable ROI status
        roi_status = "CALCULATED"
        is_verified = False
        lock_id = None
        note = "Modeled projection; unverified without cryptographic BaselineLock."

        if baseline_lock is not None:
            lock_id = getattr(baseline_lock, "lock_id", str(baseline_lock))
            if hasattr(baseline_lock, "validate_candidate") and candidate_metadata is not None:
                is_valid, _, violations = baseline_lock.validate_candidate(candidate_metadata)
                if not is_valid:
                    roi_status = "INVALID"
                    note = f"Baseline lock violated: {', '.join(violations)}"
                else:
                    if welch_p_value is not None and welch_p_value < 0.05 and speedup_ci_95 is not None:
                        roi_status = "VERIFIED"
                        is_verified = True
                        note = f"Empirically verified against BaselineLock [{lock_id}] with Welch p={welch_p_value:.4f} < 0.05."
                    else:
                        note = f"Locked to [{lock_id}], but Welch p-value >= 0.05 or multi-run CI missing; status remains CALCULATED."
            else:
                if welch_p_value is not None and welch_p_value < 0.05:
                    roi_status = "VERIFIED"
                    is_verified = True
                    note = f"Empirically verified against BaselineLock [{lock_id}] with Welch p={welch_p_value:.4f} < 0.05."

        # Multi-Tier Economics
        tokens_per_step = 4096
        cost_per_step_base = (baseline_step_time_ms / 1000.0 / 3600.0) * self.gpu_cost_per_hour * num_gpus
        cost_per_step_opt = (optimized_step_time_ms / 1000.0 / 3600.0) * self.gpu_cost_per_hour * num_gpus
        cost_per_1m_base = (cost_per_step_base / tokens_per_step) * 1_000_000.0 if tokens_per_step > 0 else 0.0
        cost_per_1m_opt = (cost_per_step_opt / tokens_per_step) * 1_000_000.0 if tokens_per_step > 0 else 0.0
        projected_annual = (time_reduction_pct / 100.0) * self.gpu_cost_per_hour * 8760.0 * num_gpus if time_reduction_pct > 0 else 0.0

        return ROIAuditReport(
            baseline_gpu_hours=round(baseline_gpu_hours, 2),
            optimized_gpu_hours=round(optimized_gpu_hours, 2),
            gpu_hours_saved=round(gpu_hours_saved, 2),
            time_reduction_pct=round(time_reduction_pct, 2),
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            baseline_cost_usd=round(baseline_cost, 2),
            optimized_cost_usd=round(optimized_cost, 2),
            net_client_savings_usd=round(net_client_savings, 2),
            performance_fee_usd=round(performance_fee, 2),
            performance_fee_rate_pct=self.performance_fee_rate_pct,
            total_training_steps=total_training_steps,
            leaked_gpu_hours=round(leaked_gpu_hours, 4),
            leaked_cost_usd=round(leaked_cost, 2),
            containment_premium_usd=round(containment_premium, 2),
            containment_premium_rate_pct=self.containment_premium_rate_pct,
            total_ghost_layer_revenue_usd=round(total_revenue, 2),
            audit_fee_usd=self.audit_fee_usd,
            audit_roi_multiple=audit_roi_multiple,
            cost_per_step_baseline_usd=round(cost_per_step_base, 8),
            cost_per_step_optimized_usd=round(cost_per_step_opt, 8),
            cost_per_million_tokens_baseline_usd=round(cost_per_1m_base, 4),
            cost_per_million_tokens_optimized_usd=round(cost_per_1m_opt, 4),
            realized_measured_savings_usd=round(gross_savings, 2),
            projected_annual_savings_usd=round(projected_annual, 2),
            roi_status=roi_status,
            is_verified=is_verified,
            baseline_lock_id=lock_id,
            welch_p_value=welch_p_value,
            speedup_ci_95=speedup_ci_95,
            provenance_note=note,
        )


