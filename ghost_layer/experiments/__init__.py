"""
GhostLayer Scientific Experiment, Attribution, and Verification Suite.
"""

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
from ghost_layer.experiments.manager import ExperimentManager

__all__ = [
    "ExperimentRecord",
    "HardwareEnvironment",
    "WorkloadProfile",
    "InterventionSpec",
    "LatencyDistribution",
    "QualitySafetyMetrics",
    "StatisticalAttribution",
    "EconomicImpact",
    "block_bootstrap_speedup_ci",
    "compute_cohens_d",
    "decompose_ablation_chain",
    "ExperimentManager",
]
