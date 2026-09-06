"""
Why-NOT Decision Intelligence & Counterfactual Utility Calculus.

Answers:
1. Why was an optimization selected?
2. Why were alternative candidates REJECTED?
3. What is the counterfactual utility: E[utility] = E[savings] - E[risk_cost]?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ghost_layer.evidence.taxonomy import EvidenceLevel


@dataclass
class RejectedIntervention:
    rule_id: str
    title: str
    rejection_reasons: List[str]
    risk_factors: List[str]
    evidence_level: EvidenceLevel = EvidenceLevel.E0_THEORETICAL


@dataclass
class DecisionCounterfactual:
    """
    Evaluates expected utility for a candidate decision:
    E[utility] = E[savings_usd] - E[risk_cost_usd]
    """
    action_name: str
    expected_cost_usd: float
    expected_savings_usd: float
    risk_score: float  # [0.0, 1.0]
    risk_cost_usd: float
    expected_utility_usd: float
    rationale: str


class WhyNotEngine:
    """
    Evaluates candidates not chosen and articulates explicit rejection reasoning.
    """

    @staticmethod
    def evaluate_rejections(
        model_type: str,
        mixed_precision: str,
        is_compiled: bool,
        peak_vram_mb: float,
        total_vram_mb: float,
        active_rule_ids: List[str],
    ) -> List[RejectedIntervention]:
        rejections: List[RejectedIntervention] = []

        # Why not FP16 when BF16 is possible?
        if "RULE_MIXED_PRECISION" in active_rule_ids:
            rejections.append(
                RejectedIntervention(
                    rule_id="RULE_FP16_PRECISION",
                    title="FP16 Automatic Mixed Precision",
                    rejection_reasons=[
                        "BF16 selected over FP16 due to superior dynamic range margin (8 exponent bits vs 5).",
                        "FP16 requires GradScaler dynamic loss scaling, introducing risk of underflow/overflow spikes.",
                    ],
                    risk_factors=["Loss scale collapse", "NaN gradient norm"],
                    evidence_level=EvidenceLevel.E3_CONTROLLED_PHYSICAL,
                )
            )

        # Why not torch.compile?
        if not is_compiled:
            rejections.append(
                RejectedIntervention(
                    rule_id="RULE_GRAPHIFY_TORCH_COMPILE",
                    title="Inductor Graph Compilation (torch.compile)",
                    rejection_reasons=[
                        "Compilation requires pre-start execution phase; runtime toggle risk of dynamic recompilation stall.",
                        "Initial warmup compile overhead (1-3 minutes) exceeds expected savings for short campaigns.",
                    ],
                    risk_factors=["Graph break latency spike", "CUDA OOM during kernel compilation"],
                    evidence_level=EvidenceLevel.E2_OBSERVATIONAL,
                )
            )

        # Why not Muon?
        rejections.append(
            RejectedIntervention(
                rule_id="RULE_MUON_OPTIMIZER",
                title="Muon 2D Matrix Polar Root Optimizer",
                rejection_reasons=[
                    "Muon is classified as Tier B (Experimental); restricted from dynamic auto-apply under Tier A safety budgets.",
                    "Only accelerates internal 2D weight matrices; embedding and 1D normalization layers still require standard AdamW.",
                ],
                risk_factors=["Newton-Schulz iteration divergence on ill-conditioned weights"],
                evidence_level=EvidenceLevel.E1_SIMULATED,
            )
        )

        return rejections

    @staticmethod
    def calculate_counterfactual(
        baseline_cost_usd: float,
        candidate_speedup_pct: float,
        risk_level: str,  # "LOW", "MEDIUM", "HIGH"
        hourly_rate_usd: float = 3.50,
    ) -> DecisionCounterfactual:
        """
        Calculates E[Utility] = E[Savings] - E[Risk Cost].
        """
        # Risk factor weights
        risk_multipliers = {"LOW": 0.02, "MEDIUM": 0.08, "HIGH": 0.25}
        risk_score = risk_multipliers.get(risk_level.upper(), 0.05)

        raw_savings = baseline_cost_usd * (candidate_speedup_pct / 100.0)
        expected_savings = max(0.0, raw_savings)
        risk_cost = expected_savings * risk_score
        expected_utility = expected_savings - risk_cost
        optimized_cost = baseline_cost_usd - expected_savings

        rationale = (
            f"Applying candidate yields gross savings of ${expected_savings:.2f} "
            f"with risk penalty of ${risk_cost:.2f} ({risk_level} risk), resulting in net expected utility of ${expected_utility:.2f}."
        )

        return DecisionCounterfactual(
            action_name="APPLY_RECOMMENDED",
            expected_cost_usd=round(optimized_cost, 2),
            expected_savings_usd=round(expected_savings, 2),
            risk_score=round(risk_score, 4),
            risk_cost_usd=round(risk_cost, 2),
            expected_utility_usd=round(expected_utility, 2),
            rationale=rationale,
        )
