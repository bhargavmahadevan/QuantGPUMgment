"""
Financial Impact Attribution: Converts technical efficiency gaps into concrete dollar figures,
cluster-scaled monthly waste, and audit payback metrics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

GPU_HOURLY_RATES_USD: Dict[str, float] = {
    "b200": 6.00,
    "h200": 4.50,
    "h100": 3.50,
    "a100-80g": 2.20,
    "a100": 2.00,
    "l40s": 1.25,
    "l40": 1.15,
    "a6000": 1.10,
    "rtx 4090": 0.80,
    "4090": 0.80,
    "rtx 3090": 0.50,
    "3090": 0.50,
    "a2000": 0.35,
    "rtx a2000": 0.35,
    "default": 3.50,
}


@dataclass
class FinancialImpact:
    cluster_hourly_rate_usd: float
    monthly_baseline_cost_usd: float
    modeled_monthly_waste_usd: float
    modeled_monthly_savings_usd: float
    modeled_annual_savings_usd: float
    audit_payback_days: float
    audit_roi_multiple: float
    num_gpus: int = 8
    hardware_type: str = "H100"
    audit_fee_usd: float = 2500.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinancialImpactCalculator:
    """Calculates cluster-level financial spend, wasted compute, and audit ROI."""

    def __init__(self, default_rate_usd: float = 3.50):
        self.default_rate_usd = default_rate_usd

    def get_hourly_rate(self, hardware_type: Optional[str]) -> float:
        if not hardware_type:
            return self.default_rate_usd
        hw_lower = hardware_type.lower()
        for key, rate in GPU_HOURLY_RATES_USD.items():
            if key != "default" and key in hw_lower:
                return rate
        return self.default_rate_usd

    def calculate_impact(
        self,
        hardware_type: str = "H100",
        num_gpus: int = 8,
        speedup_potential_pct: float = 20.0,
        monthly_active_hours: float = 720.0,
        audit_fee_usd: float = 2500.0,
    ) -> FinancialImpact:
        hourly_gpu_rate = self.get_hourly_rate(hardware_type)
        cluster_hourly = round(hourly_gpu_rate * max(1, num_gpus), 2)
        monthly_baseline = round(cluster_hourly * monthly_active_hours, 2)

        # Savings = fraction of runtime reduced
        savings_fraction = min(0.60, max(0.01, speedup_potential_pct / 100.0))
        modeled_monthly_savings = round(monthly_baseline * savings_fraction, 2)
        # Waste equals savings under the assumption of 100% recoverability.
        # If partial recoverability is needed, introduce a recovery_rate parameter.
        modeled_monthly_waste = modeled_monthly_savings
        modeled_annual_savings = round(modeled_monthly_savings * 12.0, 2)

        daily_savings = modeled_monthly_savings / 30.0 if modeled_monthly_savings > 0 else 1.0
        payback_days = round(audit_fee_usd / daily_savings, 1)
        roi_multiple = round(modeled_annual_savings / max(1.0, audit_fee_usd), 2)

        return FinancialImpact(
            cluster_hourly_rate_usd=cluster_hourly,
            monthly_baseline_cost_usd=monthly_baseline,
            modeled_monthly_waste_usd=modeled_monthly_waste,
            modeled_monthly_savings_usd=modeled_monthly_savings,
            modeled_annual_savings_usd=modeled_annual_savings,
            audit_payback_days=payback_days,
            audit_roi_multiple=roi_multiple,
            num_gpus=num_gpus,
            hardware_type=hardware_type,
            audit_fee_usd=audit_fee_usd,
        )
