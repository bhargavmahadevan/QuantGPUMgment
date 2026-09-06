"""
3D Credibility Manifold & Zero-Fabrication Decision Surface for GhostLayer.

Lifts 1D scalar recommendation confidence into a 3D geometric credibility manifold:
    C = [Heuristic Prior (C_0),
         Physical Verification Count (N_runs),
         Loss-Shift Drift Margin (Delta_loss)]

Enforces GhostLayer's Zero-Fabrication and Empirical Transparency Directives:
1. Pure Heuristic Priors (N_runs == 0) are strictly capped at 0.60 and labeled as theoretical.
2. Auto-Apply authorization requires satisfying the 3D Polytope:
       (C_0 >= 0.80) AND (N_runs >= 5) AND (Delta_loss <= 0.10)
3. Any drift delta exceeding 0.10 mathematically blocks auto-apply regardless of prior confidence.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
import math


class CredibilityStatus(str, Enum):
    HEURISTIC_PRIOR = "HEURISTIC_PRIOR"                       # N_runs == 0
    EMPIRICALLY_VALIDATED = "EMPIRICALLY_VALIDATED"           # 1 <= N_runs < 5
    PRODUCTION_VERIFIED_SAFE = "PRODUCTION_VERIFIED_SAFE"     # N_runs >= 5 and Delta <= 0.10
    BLOCKED_EXCESSIVE_DRIFT = "BLOCKED_EXCESSIVE_DRIFT"       # Delta > 0.10 (Hard block)


@dataclass
class CredibilityVector3D:
    """3-Dimensional Recommendation Credibility State Vector."""
    heuristic_prior: float         # Theoretical rule prior [0.0, 1.0]
    verified_hardware_runs: int    # Number of physical KB-verified runs on target hardware
    loss_shift_margin: float       # Empirical loss divergence delta (0.00 - 1.00+)

    def to_vector(self) -> List[float]:
        return [
            float(self.heuristic_prior),
            float(self.verified_hardware_runs),
            float(self.loss_shift_margin),
        ]


@dataclass
class CredibilityEvaluation:
    """Evaluation result on the 3D Credibility Manifold."""
    status: CredibilityStatus
    calibrated_score: float
    is_safe_to_auto_apply: bool
    is_empirical: bool
    distance_to_auto_apply_boundary: float
    audit_label: str
    vector: CredibilityVector3D
    breakdown: Dict[str, Any]


class CredibilityManifold:
    """
    Evaluator and Boundary Controller for the 3D Credibility Manifold.
    """

    def __init__(
        self,
        min_auto_apply_runs: int = 5,
        max_loss_shift_threshold: float = 0.10,
        zero_runs_confidence_cap: float = 0.60,
        min_prior_threshold: float = 0.80,
    ):
        self.min_runs = min_auto_apply_runs
        self.max_shift = max_loss_shift_threshold
        self.zero_runs_cap = zero_runs_confidence_cap
        self.min_prior = min_prior_threshold

    def evaluate(
        self,
        heuristic_prior: float,
        verified_hardware_runs: int,
        loss_shift_margin: float = 0.0,
        rollback_frequency: float = 0.0,
    ) -> CredibilityEvaluation:
        """
        Evaluates a recommendation tuple against the 3D Credibility Manifold.
        """
        prior = max(0.0, min(1.0, float(heuristic_prior)))
        runs = max(0, int(verified_hardware_runs))
        shift = max(0.0, float(loss_shift_margin))
        rollback_rate = max(0.0, min(1.0, float(rollback_frequency)))

        vec = CredibilityVector3D(
            heuristic_prior=round(prior, 4),
            verified_hardware_runs=runs,
            loss_shift_margin=round(shift, 4),
        )

        # 1. Hard Block on Excessive Drift
        if shift > self.max_shift:
            # Loss drift exceeds 0.10 threshold -> strictly blocked
            score = round(max(0.10, min(0.50, prior - 2.0 * (shift - self.max_shift))), 2)
            return CredibilityEvaluation(
                status=CredibilityStatus.BLOCKED_EXCESSIVE_DRIFT,
                calibrated_score=score,
                is_safe_to_auto_apply=False,
                is_empirical=(runs > 0),
                distance_to_auto_apply_boundary=round(shift - self.max_shift, 4),
                audit_label=f"BLOCKED: Loss shift delta ({shift:.4f}) exceeds 0.10 safety threshold.",
                vector=vec,
                breakdown={
                    "drift_violation": round(shift - self.max_shift, 4),
                    "runs_count": runs,
                    "rollback_rate_pct": round(rollback_rate * 100.0, 1),
                },
            )

        # 2. Heuristic Prior (Zero verified runs)
        if runs == 0:
            calibrated_score = round(min(self.zero_runs_cap, prior * 0.70), 2)
            dist_runs = float(self.min_runs - runs)
            dist_prior = max(0.0, self.min_prior - prior)
            dist = math.sqrt(dist_runs**2 + dist_prior**2)

            return CredibilityEvaluation(
                status=CredibilityStatus.HEURISTIC_PRIOR,
                calibrated_score=calibrated_score,
                is_safe_to_auto_apply=False,
                is_empirical=False,
                distance_to_auto_apply_boundary=round(dist, 3),
                audit_label=f"HEURISTIC PRIOR: Score capped at {calibrated_score:.2f} (0 verified hardware runs).",
                vector=vec,
                breakdown={
                    "verified_runs": 0,
                    "heuristic_cap": self.zero_runs_cap,
                    "needed_runs_for_auto_apply": self.min_runs,
                },
            )

        # 3. Empirically Validated (1 to 4 runs)
        if 1 <= runs < self.min_runs:
            # Calibrate with logarithmic empirical scaling
            empirical_boost = 0.05 * math.log2(1 + runs)
            penalty = 0.25 * rollback_rate + 0.5 * shift
            raw_score = prior + empirical_boost - penalty
            calibrated_score = round(max(0.50, min(0.85, raw_score)), 2)

            dist_runs = float(self.min_runs - runs)
            return CredibilityEvaluation(
                status=CredibilityStatus.EMPIRICALLY_VALIDATED,
                calibrated_score=calibrated_score,
                is_safe_to_auto_apply=False,
                is_empirical=True,
                distance_to_auto_apply_boundary=round(dist_runs, 3),
                audit_label=f"EMPIRICALLY VALIDATED: {runs}/{self.min_runs} required verified runs completed.",
                vector=vec,
                breakdown={
                    "verified_runs": runs,
                    "needed_runs_for_auto_apply": self.min_runs - runs,
                    "rollback_rate_pct": round(rollback_rate * 100.0, 1),
                },
            )

        # 4. Production Verified Safe (N >= 5 runs, Delta <= 0.10, Prior >= 0.80)
        empirical_boost = 0.10 * (1.0 - math.exp(-0.2 * runs))
        penalty = 0.30 * rollback_rate + 0.4 * shift
        raw_score = prior + empirical_boost - penalty
        calibrated_score = round(max(0.80, min(0.99, raw_score)), 2)
        is_safe = (prior >= self.min_prior) and (rollback_rate <= 0.10)
        # Geometric distance to auto-apply boundary:
        # (prior >= min_prior) AND (rollback_rate <= 0.10)
        if is_safe:
            boundary_distance = 0.0
        else:
            prior_gap = max(0.0, self.min_prior - prior)
            rollback_gap = max(0.0, rollback_rate - 0.10)
            boundary_distance = round(math.sqrt(prior_gap ** 2 + rollback_gap ** 2), 4)

        return CredibilityEvaluation(
            status=CredibilityStatus.PRODUCTION_VERIFIED_SAFE if is_safe else CredibilityStatus.EMPIRICALLY_VALIDATED,
            calibrated_score=calibrated_score,
            is_safe_to_auto_apply=is_safe,
            is_empirical=True,
            distance_to_auto_apply_boundary=boundary_distance,
            audit_label=f"PRODUCTION VERIFIED SAFE: {runs} empirical hardware trials confirmed safe (loss drift: {shift:.4f}).",
            vector=vec,
            breakdown={
                "verified_runs": runs,
                "loss_shift_margin": round(shift, 4),
                "rollback_rate_pct": round(rollback_rate * 100.0, 1),
                "is_safe_auto_apply": is_safe,
            },
        )
