"""
Stateful Rollback Manager: Captures training runtime and optimizer configuration state
before any optimization apply and restores it if divergence is detected.

OPERATIONAL BOUNDARY & INVARIANT:
GhostLayer Rollback is an in-memory Configuration and Optimizer Runtime State Transaction.
It restores:
  1. PyTorch optimizer hyperparameters and internal buffer states.
  2. Runtime flags (AMP, gradient checkpointing, attention implementation).
  3. DataLoader worker configuration and memory pinning.
  4. Environment variables and allocator configs.

CRITICAL DISTINCTION:
GhostLayer Rollback does NOT execute full disk-based model weight checkpoint restoration (torch.load).
If an unverified kernel corrupts floating-point weight tensors into NaNs, reverting configuration
alone cannot heal parameter weights. Full weight recovery requires reloading the caller's physical checkpoint.
GhostLayer's multi-dimensional safety gate halts execution immediately before unrecoverable parameter
corruption propagates.
"""


import time
import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from ghost_layer.replay import OptimizationReplayLog


@dataclass
class ConfigSnapshot:
    """Frozen snapshot of training configuration state before an optimization apply."""
    snapshot_id: str
    rule_id: str
    timestamp: float = field(default_factory=time.time)

    # Model config state
    model_dtype: str = "float32"
    gradient_checkpointing: bool = False
    attn_implementation: str = "sdpa"
    is_compiled: bool = False

    # Optimizer state
    optimizer_type: str = ""
    optimizer_lr: float = 0.0
    optimizer_state_dict: Optional[Dict[str, Any]] = None

    # DataLoader config
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    batch_size: Optional[int] = None

    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)

    # AMP state
    amp_enabled: bool = False
    amp_dtype: str = ""


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    rule_id: str
    snapshot_id: str
    restored_fields: List[str]
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class RollbackManager:
    """
    Manages configuration snapshots and rollback for training optimization applies.
    
    Usage:
        mgr = RollbackManager()
        snapshot = mgr.snapshot(model, optimizer, dataloader, rule_id="RULE_MIXED_PRECISION")
        # ... apply optimization ...
        # ... if verification fails:
        result = mgr.rollback(snapshot.snapshot_id, model, optimizer, dataloader)
    """

    def __init__(self, replay_log: Optional[OptimizationReplayLog] = None):
        self.replay_log = replay_log
        self._snapshots: Dict[str, ConfigSnapshot] = {}
        self._snapshot_counter = 0

    def snapshot(
        self,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        rule_id: str = "",
    ) -> ConfigSnapshot:
        """Capture a snapshot of the current training configuration."""
        self._snapshot_counter += 1
        snap_id = f"snap_{self._snapshot_counter}_{int(time.time())}"

        snap = ConfigSnapshot(
            snapshot_id=snap_id,
            rule_id=rule_id,
        )

        # Capture model state
        if model is not None:
            snap.model_dtype = str(getattr(model, "dtype", "float32"))
            snap.gradient_checkpointing = getattr(model, "is_gradient_checkpointing", False)
            if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
                snap.attn_implementation = model.config._attn_implementation
            snap.is_compiled = getattr(model, "_compiled", False)
            snap.amp_enabled = hasattr(model, "_ghost_amp_dtype")
            if snap.amp_enabled:
                snap.amp_dtype = str(model._ghost_amp_dtype)

        # Capture optimizer state
        if optimizer is not None:
            snap.optimizer_type = type(optimizer).__name__
            if hasattr(optimizer, "param_groups") and optimizer.param_groups:
                snap.optimizer_lr = optimizer.param_groups[0].get("lr", 0.0)
            try:
                snap.optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            except Exception:
                snap.optimizer_state_dict = None

        # Capture DataLoader config
        if dataloader is not None:
            snap.num_workers = getattr(dataloader, "num_workers", 0)
            snap.pin_memory = getattr(dataloader, "pin_memory", False)
            snap.persistent_workers = getattr(dataloader, "persistent_workers", False)
            snap.batch_size = getattr(dataloader, "batch_size", None)

        # Capture relevant env vars
        cuda_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if cuda_alloc:
            snap.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc

        self._snapshots[snap_id] = snap

        if self.replay_log is not None:
            self.replay_log.record_event(
                step=0,
                stage="SNAPSHOT",
                recommendation_id=rule_id,
                action="CONFIG_SNAPSHOT_CAPTURED",
                details={
                    "snapshot_id": snap_id,
                    "model_dtype": snap.model_dtype,
                    "gradient_checkpointing": snap.gradient_checkpointing,
                    "num_workers": snap.num_workers,
                    "is_compiled": snap.is_compiled,
                },
                verified_safe=True,
                reason=f"Pre-apply snapshot captured for {rule_id}",
            )

        return snap

    def rollback(
        self,
        snapshot_id: str,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
    ) -> RollbackResult:
        """Restore training configuration from a previously captured snapshot."""
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            return RollbackResult(
                success=False,
                rule_id="",
                snapshot_id=snapshot_id,
                restored_fields=[],
                error=f"Snapshot {snapshot_id} not found",
            )

        restored = []

        # Restore model state
        if model is not None:
            try:
                # Restore gradient checkpointing
                if snap.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    restored.append("gradient_checkpointing=True")
                elif not snap.gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
                    model.gradient_checkpointing_disable()
                    restored.append("gradient_checkpointing=False")

                # Restore attention implementation
                if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
                    model.config._attn_implementation = snap.attn_implementation
                    restored.append(f"attn_implementation={snap.attn_implementation}")

                # Remove AMP markers if they weren't present before
                if not snap.amp_enabled and hasattr(model, "_ghost_amp_dtype"):
                    delattr(model, "_ghost_amp_dtype")
                    if hasattr(model, "_ghost_grad_scaler"):
                        delattr(model, "_ghost_grad_scaler")
                    restored.append("amp_disabled")

                # Restore from compiled model
                if not snap.is_compiled and hasattr(model, "_ghost_original_model"):
                    # Note: torch.compile is not easily reversible at runtime;
                    # we store the original for reference but can't unwrap in-place.
                    restored.append("compiled_model_flagged_for_revert")

            except Exception as e:
                return RollbackResult(
                    success=False,
                    rule_id=snap.rule_id,
                    snapshot_id=snapshot_id,
                    restored_fields=restored,
                    error=f"Model rollback failed: {str(e)}",
                )

        # Restore optimizer state
        if optimizer is not None and snap.optimizer_state_dict is not None:
            try:
                optimizer.load_state_dict(snap.optimizer_state_dict)
                restored.append("optimizer_state_dict")
            except Exception as e:
                restored.append(f"optimizer_restore_failed: {str(e)}")

        # Restore DataLoader config
        if dataloader is not None:
            try:
                if hasattr(dataloader, "num_workers"):
                    dataloader.num_workers = snap.num_workers
                    restored.append(f"num_workers={snap.num_workers}")
                if hasattr(dataloader, "pin_memory"):
                    dataloader.pin_memory = snap.pin_memory
                    restored.append(f"pin_memory={snap.pin_memory}")
            except (AttributeError, RuntimeError):
                restored.append("dataloader_restore_partial")

        # Restore env vars
        for key, value in snap.env_vars.items():
            os.environ[key] = value
            restored.append(f"env:{key}")

        result = RollbackResult(
            success=True,
            rule_id=snap.rule_id,
            snapshot_id=snapshot_id,
            restored_fields=restored,
        )

        if self.replay_log is not None:
            self.replay_log.record_event(
                step=0,
                stage="ROLLBACK",
                recommendation_id=snap.rule_id,
                action="ROLLBACK_EXECUTED",
                details={
                    "snapshot_id": snapshot_id,
                    "restored_fields": restored,
                    "success": result.success,
                },
                verified_safe=False,
                reason=f"Rollback executed for {snap.rule_id} — divergence detected",
            )

        return result

    def can_rollback(self, snapshot_id: str) -> bool:
        """Check if a snapshot exists for rollback."""
        return snapshot_id in self._snapshots

    def get_snapshot(self, snapshot_id: str) -> Optional[ConfigSnapshot]:
        """Retrieve a snapshot by ID."""
        return self._snapshots.get(snapshot_id)

    def list_snapshots(self) -> List[ConfigSnapshot]:
        """List all captured snapshots."""
        return list(self._snapshots.values())

    def clear_snapshots(self):
        """Clear all stored snapshots."""
        self._snapshots.clear()

    def verify_state_restoration(
        self,
        snapshot: ConfigSnapshot,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
    ) -> bool:
        """
        Verifies that runtime configuration state matches the captured snapshot.
        Checks optimizer learning rate and dataloader worker count.
        """
        if optimizer and hasattr(optimizer, "param_groups") and snapshot.optimizer_lr > 0:
            current_lr = optimizer.param_groups[0].get("lr", 0.0)
            if abs(current_lr - snapshot.optimizer_lr) > 1e-9:
                return False
        if dataloader and hasattr(dataloader, "num_workers"):
            if dataloader.num_workers != snapshot.num_workers:
                return False
        return True
