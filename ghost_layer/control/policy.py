"""
Enterprise Policy Engine & Pre-Actuation Safety Boundary.

Evaluates candidate structured actions against environment-specific policy rules
(e.g., PRODUCTION_STRICT, STAGING_CANARY, RESEARCH_EXPLORATORY)
BEFORE allowing parameter modifications or trial dispatch.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from ghost_layer.control.actuators import StructuredAction, ActuatorRiskTier


class EnvironmentTier(enum.Enum):
    PRODUCTION_STRICT = "PRODUCTION_STRICT"          # Highest safety, minimal blast radius
    STAGING_CANARY = "STAGING_CANARY"                # Balanced, allows bounded experimental rules
    RESEARCH_EXPLORATORY = "RESEARCH_EXPLORATORY"    # Permissive, allows research/compilation exploration


@dataclass
class PolicyValidationResult:
    is_allowed: bool
    action: StructuredAction
    environment_tier: EnvironmentTier
    violation_reasons: List[str] = field(default_factory=list)
    enforced_constraints: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class EnvironmentSafetyPolicy:
    """
    Enterprise policy contract defining allowlisted actions and maximum parameter bounds.
    """
    environment_tier: EnvironmentTier = EnvironmentTier.PRODUCTION_STRICT
    allowed_action_types: Set[str] = field(default_factory=lambda: {
        "SET_MIXED_PRECISION",
        "SET_ATTENTION_BACKEND",
        "SET_DATALOADER_CONFIG",
    })
    max_risk_tier: ActuatorRiskTier = ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL
    max_batch_multiplier: float = 2.0
    allow_torch_compile: bool = False
    allow_optimizer_replacement: bool = False
    enforce_pre_registered_margins: bool = True

    @classmethod
    def production_default(cls) -> EnvironmentSafetyPolicy:
        """Strict production policy: Low blast radius only, no experimental compilers."""
        return cls(
            environment_tier=EnvironmentTier.PRODUCTION_STRICT,
            allowed_action_types={
                "SET_MIXED_PRECISION",
                "SET_ATTENTION_BACKEND",
                "SET_DATALOADER_CONFIG",
                "SET_BATCH_MULTIPLIER",
            },
            max_risk_tier=ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL,
            max_batch_multiplier=2.0,
            allow_torch_compile=False,
            allow_optimizer_replacement=False,
            enforce_pre_registered_margins=True,
        )

    @classmethod
    def staging_canary(cls) -> EnvironmentSafetyPolicy:
        """Staging canary policy: Allows torch.compile and higher batch scaling."""
        return cls(
            environment_tier=EnvironmentTier.STAGING_CANARY,
            allowed_action_types={
                "SET_MIXED_PRECISION",
                "SET_ATTENTION_BACKEND",
                "SET_DATALOADER_CONFIG",
                "SET_BATCH_MULTIPLIER",
                "SET_TORCH_COMPILE",
            },
            max_risk_tier=ActuatorRiskTier.RISK_TIER_C_HIGH_RISK_RESEARCH,
            max_batch_multiplier=4.0,
            allow_torch_compile=True,
            allow_optimizer_replacement=False,
            enforce_pre_registered_margins=True,
        )

    @classmethod
    def research_exploratory(cls) -> EnvironmentSafetyPolicy:
        """Permissive research policy for offline algorithmic exploration."""
        return cls(
            environment_tier=EnvironmentTier.RESEARCH_EXPLORATORY,
            allowed_action_types={
                "SET_MIXED_PRECISION",
                "SET_ATTENTION_BACKEND",
                "SET_DATALOADER_CONFIG",
                "SET_BATCH_MULTIPLIER",
                "SET_TORCH_COMPILE",
                "SET_OPTIMIZER_CURVATURE",
            },
            max_risk_tier=ActuatorRiskTier.RISK_TIER_C_HIGH_RISK_RESEARCH,
            max_batch_multiplier=8.0,
            allow_torch_compile=True,
            allow_optimizer_replacement=True,
            enforce_pre_registered_margins=False,
        )


class ActionPolicyEngine:
    """
    Enforces enterprise safety boundaries before passing structured actions to actuators.
    """
    def __init__(self, policy: Optional[EnvironmentSafetyPolicy] = None):
        self.policy = policy or EnvironmentSafetyPolicy.production_default()

    def validate(self, action: StructuredAction) -> PolicyValidationResult:
        violations: List[str] = []

        # 1. Check Allowlisted Action Type
        if action.action_type not in self.policy.allowed_action_types:
            violations.append(
                f"Action '{action.action_type}' is not allowlisted in {self.policy.environment_tier.value}"
            )

        # 2. Check Risk Tier Boundary
        tier_hierarchy = {
            ActuatorRiskTier.RISK_TIER_A_LOW_BLAST: 1,
            ActuatorRiskTier.RISK_TIER_B_BOUNDED_EXPERIMENTAL: 2,
            ActuatorRiskTier.RISK_TIER_C_HIGH_RISK_RESEARCH: 3,
        }
        action_rank = tier_hierarchy.get(action.risk_tier, 99)
        max_rank = tier_hierarchy.get(self.policy.max_risk_tier, 1)
        if action_rank > max_rank:
            violations.append(
                f"Risk tier {action.risk_tier.value} exceeds maximum permitted {self.policy.max_risk_tier.value}"
            )

        # 3. Check Batch Multiplier Bound
        if action.action_type == "SET_BATCH_MULTIPLIER":
            val = float(action.target_value) if isinstance(action.target_value, (int, float)) else 1.0
            if val > self.policy.max_batch_multiplier:
                violations.append(
                    f"Batch multiplier {val}x exceeds policy ceiling {self.policy.max_batch_multiplier}x"
                )

        # 4. Check torch.compile policy
        if action.action_type == "SET_TORCH_COMPILE" and not self.policy.allow_torch_compile:
            violations.append(
                f"torch.compile optimization is disabled in {self.policy.environment_tier.value}"
            )

        # 5. Check Action-Level Schema Constraints
        valid_schema, schema_msg = action.validate_constraints()
        if not valid_schema:
            violations.append(f"Constraint validation failed: {schema_msg}")

        is_allowed = len(violations) == 0
        rationale = "Policy check passed: action conforms to enterprise safety boundaries." if is_allowed else "; ".join(violations)

        return PolicyValidationResult(
            is_allowed=is_allowed,
            action=action,
            environment_tier=self.policy.environment_tier,
            violation_reasons=violations,
            enforced_constraints=action.constraints,
            rationale=rationale,
        )
