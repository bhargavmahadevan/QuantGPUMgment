"""
Evidence Taxonomy & The 7-Level Evidence Ladder (E0 - E6).

Formalizes the evidentiary hierarchy across all GhostLayer recommendations,
hardware experiments, financial projections, and Knowledge Base entries.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class EvidenceLevel(enum.Enum):
    """
    The 7-Level Evidence Ladder:
    E0 — Theoretical / First Principles
    E1 — Simulated / Synthetic Graph Simulation
    E2 — Observational / Passive Telemetry Capture
    E3 — Single Controlled Physical Experiment (Counterbalanced ABAB)
    E4 — Replicated Physical Experiment (Multi-Run Verification)
    E5 — Production Workload Telemetry (Customer In-Flight Run)
    E6 — Independently Reproduced Production Result (Multi-Cluster / Multi-Tenant)
    """
    E0_THEORETICAL = "E0_THEORETICAL"
    E1_SIMULATED = "E1_SIMULATED"
    E2_OBSERVATIONAL = "E2_OBSERVATIONAL"
    E3_CONTROLLED_PHYSICAL = "E3_CONTROLLED_PHYSICAL"
    E4_REPLICATED_PHYSICAL = "E4_REPLICATED_PHYSICAL"
    E5_PRODUCTION_WORKLOAD = "E5_PRODUCTION_WORKLOAD"
    E6_INDEPENDENTLY_REPRODUCED = "E6_INDEPENDENTLY_REPRODUCED"

    @property
    def rank(self) -> int:
        ranks = {
            "E0_THEORETICAL": 0,
            "E1_SIMULATED": 1,
            "E2_OBSERVATIONAL": 2,
            "E3_CONTROLLED_PHYSICAL": 3,
            "E4_REPLICATED_PHYSICAL": 4,
            "E5_PRODUCTION_WORKLOAD": 5,
            "E6_INDEPENDENTLY_REPRODUCED": 6,
        }
        return ranks.get(self.value, 0)


class ExecutionProvenance(enum.Enum):
    """
    Execution taxonomy classifying telemetry provenance.
    Strictly prevents CPU runs from masquerading as GPU evidence.
    """
    REAL_GPU = "REAL_GPU"
    REAL_CPU = "REAL_CPU"
    REAL_MULTI_GPU = "REAL_MULTI_GPU"
    SIMULATED = "SIMULATED"
    ANALYTICAL = "ANALYTICAL"
    PUBLIC_DATA = "PUBLIC_DATA"


@dataclass
class EvidenceTag:
    level: EvidenceLevel
    provenance: ExecutionProvenance
    hardware_verified: bool
    experiment_id: Optional[str] = None

    def format_label(self) -> str:
        exp_suffix = f" [{self.experiment_id}]" if self.experiment_id else ""
        return f"[{self.level.value} | {self.provenance.value}{exp_suffix}]"
