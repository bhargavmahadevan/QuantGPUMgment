"""
Weights & Biases Exporter: Logs GhostLayer telemetry and decision records
to W&B as complementary monitoring data alongside existing dashboards.

This does NOT replace W&B — it enriches it with decision-level data that
W&B's native tracking doesn't capture.
"""

import time
from typing import Optional, Dict, Any, List

# Graceful import — wandb is optional
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


class WandbExporter:
    """
    Logs GhostLayer telemetry and decision records to Weights & Biases.
    
    Usage:
        exporter = WandbExporter(project="my-training", run_name="ghostlayer-audit")
        exporter.init()
        exporter.log_step(snapshot)
        exporter.log_recommendation(recommendation)
        exporter.log_verification(verification_result)
        exporter.finish()
    """

    def __init__(
        self,
        project: str = "ghostlayer",
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.project = project
        self.run_name = run_name or f"ghostlayer_{int(time.time())}"
        self.entity = entity
        self.tags = tags or ["ghostlayer", "decision-audit"]
        self._run = None
        self._step_count = 0
        self._initialized = False

    @property
    def is_available(self) -> bool:
        return _WANDB_AVAILABLE

    def init(self) -> bool:
        """Initialize a W&B run. Returns True if successful."""
        if not _WANDB_AVAILABLE:
            return False

        try:
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                entity=self.entity,
                tags=self.tags,
                reinit=True,
            )
            self._initialized = True
            return True
        except Exception:
            return False

    def log_step(self, snapshot: Any) -> bool:
        """Log step-level telemetry to W&B."""
        if not self._initialized or not _WANDB_AVAILABLE:
            return False

        try:
            wandb.log({
                "ghostlayer/step": snapshot.step,
                "ghostlayer/gpu_utilization_pct": snapshot.gpu_utilization_pct,
                "ghostlayer/gpu_memory_used_mb": snapshot.gpu_memory_used_mb,
                "ghostlayer/step_time_ms": snapshot.step_time_ms,
                "ghostlayer/loss": snapshot.loss,
                "ghostlayer/data_loading_time_ms": snapshot.data_loading_time_ms,
                "ghostlayer/mixed_precision": snapshot.mixed_precision,
            }, step=self._step_count)
            self._step_count += 1
            return True
        except Exception:
            return False

    def log_recommendation(self, recommendation: Any) -> bool:
        """Log a recommendation as a W&B table row."""
        if not self._initialized or not _WANDB_AVAILABLE:
            return False

        try:
            table = wandb.Table(columns=[
                "rule_id", "title", "impact", "confidence",
                "speedup_estimate", "safe_to_auto_apply", "evidence"
            ])
            table.add_data(
                recommendation.rule_id,
                recommendation.title,
                recommendation.impact_level,
                recommendation.confidence,
                recommendation.speedup_estimate_label,
                recommendation.safe_to_auto_apply,
                recommendation.evidence,
            )
            wandb.log({"ghostlayer/recommendations": table})
            return True
        except Exception:
            return False

    def log_verification(self, verification_result: Any) -> bool:
        """Log a verification result to W&B."""
        if not self._initialized or not _WANDB_AVAILABLE:
            return False

        try:
            wandb.log({
                "ghostlayer/verification/is_safe": verification_result.is_safe,
                "ghostlayer/verification/max_loss_delta": verification_result.max_loss_delta,
                "ghostlayer/verification/relative_mean_loss_shift": verification_result.relative_mean_loss_shift,
                "ghostlayer/verification/action_taken": verification_result.action_taken,
                "ghostlayer/verification/rule_id": verification_result.recommendation_id,
            })
            return True
        except Exception:
            return False

    def log_savings(self, savings_report: Any) -> bool:
        """Log a verified savings report to W&B."""
        if not self._initialized or not _WANDB_AVAILABLE:
            return False

        try:
            wandb.log({
                "ghostlayer/savings/speedup_pct": savings_report.speedup_pct,
                "ghostlayer/savings/is_verified": savings_report.is_verified,
                "ghostlayer/savings/p_value": savings_report.p_value,
                "ghostlayer/savings/dollar_per_1000_gpu_hrs": savings_report.dollar_savings_per_1000_gpu_hours,
            })
            return True
        except Exception:
            return False

    def log_decision_record(self, audit_trail: List[Dict[str, Any]]) -> bool:
        """Log a complete decision record as a W&B table."""
        if not self._initialized or not _WANDB_AVAILABLE:
            return False

        try:
            table = wandb.Table(columns=[
                "timestamp", "step", "stage", "recommendation_id",
                "action", "verified_safe", "reason"
            ])
            for event in audit_trail:
                table.add_data(
                    event.get("timestamp", 0),
                    event.get("step", 0),
                    event.get("stage", ""),
                    event.get("recommendation_id", ""),
                    event.get("action", ""),
                    event.get("verified_safe", None),
                    event.get("reason", ""),
                )
            wandb.log({"ghostlayer/decision_record": table})
            return True
        except Exception:
            return False

    def finish(self):
        """Finish the W&B run."""
        if self._initialized and _WANDB_AVAILABLE and self._run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
            self._initialized = False
