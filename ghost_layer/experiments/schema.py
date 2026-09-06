"""
Canonical ExperimentRecord Schema: The authoritative data contract for all
scientific experiments, empirical receipts, executive reports, and knowledge base compounding.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HardwareEnvironment:
    device_name: str
    is_cuda: bool
    vram_total_mb: float
    cuda_version: str = "N/A"
    pytorch_version: str = "N/A"
    driver_version: Optional[str] = None
    device_uuid: Optional[str] = None
    node_name: str = "localhost"


@dataclass
class WorkloadProfile:
    model_family: str
    model_name_or_path: str
    batch_size: int
    sequence_length: int
    parameter_count: int = 0
    dataset_identifier: str = "synthetic"
    precision: str = "fp32"
    attention_implementation: str = "eager"
    num_workers: int = 0
    pin_memory: bool = False
    torch_compile_mode: Optional[str] = None


@dataclass
class InterventionSpec:
    rule_id: str
    intervention_tier: str  # TIER_A_PRODUCTION_SAFE, STARTUP_ONLY, TIER_B_EXPERIMENTAL, TIER_C_RESEARCH
    target_component: str  # DataLoader, Optimizer, Model, Allocator, Attention
    diff_before: Dict[str, Any] = field(default_factory=dict)
    diff_after: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyDistribution:
    sample_count: int
    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    raw_step_times_ms: List[float] = field(default_factory=list)


@dataclass
class QualitySafetyMetrics:
    relative_mean_loss_shift: float
    max_loss_delta: float
    has_nan_or_inf: bool = False
    validation_loss_baseline: Optional[float] = None
    validation_loss_candidate: Optional[float] = None
    is_verified_safe: bool = True
    safety_gate_reasons: List[str] = field(default_factory=list)
    # GhostLayer 2.0 TOST & Equivalence Metrics
    outcome_status: str = "MEASURED"
    is_non_inferior: bool = True
    equivalence_margin: float = 0.05
    loss_delta_upper_ci_95: float = 0.0
    p_value_non_inferiority: Optional[float] = None
    safety_level_passed: int = 2


@dataclass
class StatisticalAttribution:
    test_method: str  # "Block Bootstrap (Autocorrelation-Aware)" or "Paired Replicate Test"
    speedup_pct: float
    confidence_interval_95: Tuple[float, float]
    p_value: float
    is_statistically_significant: bool
    effect_size_cohens_d: float
    ablation_marginal_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class EconomicImpact:
    gpu_cost_per_hour: float
    measured_speedup_pct: float
    gpu_hours_saved_per_1000_steps: float
    dollar_savings_per_1000_steps: float
    provenance_type: str = "EMPIRICAL"  # EMPIRICAL or MODELLED
    cost_per_step_baseline_usd: float = 0.0
    cost_per_step_optimized_usd: float = 0.0
    cost_per_million_tokens_baseline_usd: float = 0.0
    cost_per_million_tokens_optimized_usd: float = 0.0
    realized_measured_savings_usd: float = 0.0
    projected_annual_savings_usd: float = 0.0


@dataclass
class ExperimentRecord:
    """
    Authoritative canonical contract recording an empirical trial,
    counterfactual baseline comparison, statistical validation, and causal attribution.
    """
    experiment_id: str
    timestamp: float = field(default_factory=time.time)
    trial_design: str = "COUNTERBALANCED_ABAB"  # ABAB, BABA, PAIRED_REPLICATE
    warmup_steps_discarded: int = 5
    hardware: HardwareEnvironment = field(default_factory=lambda: HardwareEnvironment(device_name="CPU", is_cuda=False, vram_total_mb=0.0))
    workload: WorkloadProfile = field(default_factory=lambda: WorkloadProfile(model_family="unknown", model_name_or_path="unknown", batch_size=1, sequence_length=2048))
    intervention: InterventionSpec = field(default_factory=lambda: InterventionSpec(rule_id="RULE_UNKNOWN", intervention_tier="TIER_A_PRODUCTION_SAFE", target_component="unknown"))
    baseline_distribution: LatencyDistribution = field(default_factory=lambda: LatencyDistribution(0, 0.0, 0.0, 0.0, 0.0))
    candidate_distribution: LatencyDistribution = field(default_factory=lambda: LatencyDistribution(0, 0.0, 0.0, 0.0, 0.0))
    quality_safety: QualitySafetyMetrics = field(default_factory=lambda: QualitySafetyMetrics(0.0, 0.0))
    statistics: StatisticalAttribution = field(default_factory=lambda: StatisticalAttribution("Block Bootstrap", 0.0, (0.0, 0.0), 1.0, False, 0.0))
    economics: EconomicImpact = field(default_factory=lambda: EconomicImpact(3.50, 0.0, 0.0, 0.0, "EMPIRICAL"))
    commit_hash: Optional[str] = None
    record_hash: str = ""

    def __post_init__(self):
        if not self.record_hash:
            # Deterministic fingerprint
            raw = f"{self.experiment_id}:{self.intervention.rule_id}:{self.statistics.speedup_pct}:{self.timestamp}"
            self.record_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ExperimentRecord to standard dictionary."""
        return asdict(self)
