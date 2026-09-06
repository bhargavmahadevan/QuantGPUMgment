"""
Continuous-value / staleness detection.

The problem this module exists to solve: a performance-fee optimization service is only worth
paying for on an ongoing basis if it keeps finding NEW value. A client who is handed a one-time
config (`num_workers=4, fp16, torch.compile(mode="reduce-overhead")`) has no reason to keep paying
once they've copied it into their own training script.

This module makes "keep paying" a defensible ask by tracking two things that genuinely change
over time and genuinely make a previously-applied config stale:

1. The CLIENT'S OWN environment drifting away from what was last verified (a PyTorch upgrade,
   a different GPU, a model that grew past its original parameter-count bucket). A config that
   was verified safe for one environment is NOT verified safe for a materially different one --
   re-verification is a real, honest, billable event, not an upsell tactic.

2. The SHARED KNOWLEDGE BASE learning something better since the client's last engagement. If
   30 other verified runs have landed on a faster config for this same fingerprint since this
   client was last optimized, that's a genuine, quantifiable new opportunity -- not a repeat of
   the same pitch.

Neither of these requires "AI" in the deep-learning sense to be real and defensible. They require
honest bookkeeping, which is what this module does.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time

from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, KnowledgeEntry


@dataclass
class EnvironmentFingerprint:
    """A snapshot of the environment a config was last verified against."""
    architecture_family: str
    hardware_type: str
    param_count_bucket: str
    framework_version: str
    applied_config: Dict[str, Any]
    verified_speedup_pct: float
    timestamp: float

    def differs_from(self, other: "EnvironmentFingerprint") -> Dict[str, str]:
        """Returns a dict of {dimension: description} for every dimension that changed.
        Empty dict means the environment is unchanged."""
        changes = {}
        if self.hardware_type != other.hardware_type:
            changes["hardware_type"] = f"{other.hardware_type} \u2192 {self.hardware_type}"
        if self.framework_version != other.framework_version:
            changes["framework_version"] = f"{other.framework_version} \u2192 {self.framework_version}"
        if self.param_count_bucket != other.param_count_bucket:
            changes["param_count_bucket"] = f"{other.param_count_bucket} \u2192 {self.param_count_bucket}"
        return changes


@dataclass
class DriftReport:
    is_stale: bool
    reasons: list  # list of human-readable reasons, empty if not stale
    environment_drift: Dict[str, str]
    kb_has_better_config: bool
    kb_improvement_pct: float  # how much better the KB's current best is vs. what's applied, if any
    kb_match_level: str
    recommended_action: str


class DriftMonitor:
    """
    Compares a client's currently-applied, previously-verified config against (a) their current
    environment and (b) the current state of the shared knowledge base, and reports whether
    re-optimization is warranted.

    This is meant to run periodically (e.g. a scheduled check, or triggered on every N-th training
    run) against a stored EnvironmentFingerprint from the client's last engagement.
    """

    # A KB improvement below this threshold isn't worth flagging -- avoids nagging a client over
    # noise-level differences (the same noise floor that showed up in the MLP CPU benchmark).
    MIN_MEANINGFUL_IMPROVEMENT_PCT = 5.0

    def __init__(self, knowledge_base: Optional[SharedKnowledgeBase] = None):
        self.kb = knowledge_base or SharedKnowledgeBase()

    def check(self, last_applied: EnvironmentFingerprint, current_env: EnvironmentFingerprint) -> DriftReport:
        reasons = []
        env_drift = current_env.differs_from(last_applied)
        if env_drift:
            reasons.append(
                "Environment changed since last verified optimization: " +
                "; ".join(f"{k} ({v})" for k, v in env_drift.items())
            )

        best_entry, match_level = self.kb.find_best_match(
            architecture_family=current_env.architecture_family,
            hardware_type=current_env.hardware_type,
            param_count_bucket=current_env.param_count_bucket,
            framework_version=current_env.framework_version,
        )

        kb_has_better = False
        kb_improvement = 0.0
        if best_entry is not None:
            gap = best_entry.throughput_improvement_pct - last_applied.verified_speedup_pct
            if gap >= self.MIN_MEANINGFUL_IMPROVEMENT_PCT and best_entry.confidence_tier in ("MEDIUM", "HIGH"):
                kb_has_better = True
                kb_improvement = round(gap, 2)
                reasons.append(
                    f"The shared knowledge base now has a {best_entry.confidence_tier.lower()}-confidence "
                    f"config ({best_entry.sample_count} verified samples, {match_level} match) averaging "
                    f"{best_entry.throughput_improvement_pct}% \u2014 {kb_improvement:+.1f} pts better than "
                    f"what's currently applied for this client."
                )

        is_stale = bool(env_drift) or kb_has_better

        if not is_stale:
            action = "No action needed \u2014 currently applied config is still current and still the best known option."
        elif env_drift and kb_has_better:
            action = "Re-run full audit: environment changed AND a better config now exists. This is a billable re-optimization event."
        elif env_drift:
            action = "Re-verify the existing config against the new environment before trusting it further; do not assume last verification still holds."
        else:
            action = "Offer a re-optimization pass to capture the newly-learned config. Billable only if the client accepts and the new config is verified faster."

        return DriftReport(
            is_stale=is_stale,
            reasons=reasons,
            environment_drift=env_drift,
            kb_has_better_config=kb_has_better,
            kb_improvement_pct=kb_improvement,
            kb_match_level=match_level,
            recommended_action=action,
        )
