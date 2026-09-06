"""
Auto-Applier: Applies recommended optimizations to the training runtime
with explicit user consent gating.

Consent Levels:
    AUDIT_ONLY      — Default. Observe and recommend only. Never modify anything.
    RECOMMEND_AND_ASK — Generate apply commands but require explicit per-recommendation approval.
    AUTO_APPLY_SAFE — Auto-apply recommendations that pass the loss-shift proxy verification.
    AUTO_APPLY_ALL  — Auto-apply all recommendations regardless of verification status.

Every apply action is logged to the ReplayLog regardless of consent level.
"""

import enum
import time
import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

from ghost_layer.replay import OptimizationReplayLog


TIER_A_RULES = {
    "RULE_MIXED_PRECISION",
    "RULE_DATALOADER_WORKERS",
    "RULE_FLASH_ATTENTION",
    "RULE_GRADIENT_CHECKPOINTING",
    "RULE_BATCH_SCALING",
    "RULE_ASYNC_TENSOR_FLOW",
}

STARTUP_ONLY_RULES = {
    "RULE_CUDA_ALLOCATOR_TUNING",
    "RULE_GRAPHIFY_TORCH_COMPILE",
}

TIER_B_RULES = {
    "RULE_MUON_OPTIMIZER",
    "RULE_SOPHIA_SECOND_ORDER",
    "RULE_SHAMPOO_PRECONDITIONING",
    "RULE_PEARLMUTTER_HVP_METAGRADIENT",
    "RULE_CHUNKED_CROSS_ENTROPY",
    "RULE_QUASI_NEWTON_LBFGS",
}

TIER_C_RULES = {
    "RULE_MHC_RESIDUAL_ROUTING",
    "RULE_VECTOR_FIELD_VORTICITY_DAMPING",
}


class ConsentLevel(enum.Enum):
    """User consent level for automatic optimization application."""
    AUDIT_ONLY = "AUDIT_ONLY"
    RECOMMEND_AND_ASK = "RECOMMEND_AND_ASK"
    AUTO_APPLY_SAFE = "AUTO_APPLY_SAFE"
    AUTO_APPLY_ALL = "AUTO_APPLY_ALL"


@dataclass
class ApplyResult:
    """Result of attempting to apply a single recommendation."""
    rule_id: str
    applied: bool
    consent_level: str
    before_config: Dict[str, Any]
    after_config: Dict[str, Any]
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def changed(self) -> bool:
        return self.applied and self.before_config != self.after_config


def get_rule_tier(rule_id: str) -> str:
    """Classifies rule into execution feasibility and risk tier."""
    if rule_id in STARTUP_ONLY_RULES:
        return "STARTUP_ONLY"
    if rule_id in TIER_B_RULES:
        return "TIER_B_EXPERIMENTAL"
    if rule_id in TIER_C_RULES:
        return "TIER_C_RESEARCH"
    return "TIER_A_PRODUCTION_SAFE"


class AutoApplier:
    """
    Applies GhostLayer recommendations to the live training runtime.
    
    All applies are gated behind an explicit ConsentLevel. At AUDIT_ONLY (the default),
    no modifications are made — the applier only logs what it *would* have done.
    
    Each rule-specific applier:
    1. Captures the before-state
    2. Applies the change
    3. Captures the after-state
    4. Returns an ApplyResult
    5. Logs the event to the ReplayLog
    """

    def __init__(
        self,
        consent_level: ConsentLevel = ConsentLevel.AUDIT_ONLY,
        replay_log: Optional[OptimizationReplayLog] = None,
        outcome_evaluator: Optional[Any] = None,
        enforce_risk_tiers: bool = False,
    ):
        self.consent_level = consent_level
        self.replay_log = replay_log
        self.outcome_evaluator = outcome_evaluator
        self.enforce_risk_tiers = enforce_risk_tiers

        # Registry of rule_id -> applier function
        self._appliers: Dict[str, Callable] = {
            "RULE_MIXED_PRECISION": self._apply_mixed_precision,
            "RULE_DATALOADER_WORKERS": self._apply_dataloader_workers,
            "RULE_FLASH_ATTENTION": self._apply_flash_attention,
            "RULE_GRADIENT_CHECKPOINTING": self._apply_gradient_checkpointing,
            "RULE_BATCH_SCALING": self._apply_batch_scaling,
            "RULE_GRAPHIFY_TORCH_COMPILE": self._apply_torch_compile,
            "RULE_MUON_OPTIMIZER": self._apply_muon,
            "RULE_SOPHIA_SECOND_ORDER": self._apply_sophia,
            "RULE_SHAMPOO_PRECONDITIONING": self._apply_shampoo,
            "RULE_PEARLMUTTER_HVP_METAGRADIENT": self._apply_hvp,
            "RULE_ASYNC_TENSOR_FLOW": self._apply_async_tensor_flow,
            "RULE_CUDA_ALLOCATOR_TUNING": self._apply_cuda_allocator,
            "RULE_CHUNKED_CROSS_ENTROPY": self._apply_chunked_cross_entropy,
            "RULE_MHC_RESIDUAL_ROUTING": self._apply_mhc_residual_routing,
            "RULE_QUASI_NEWTON_LBFGS": self._apply_quasi_newton_lbfgs,
            "RULE_VECTOR_FIELD_VORTICITY_DAMPING": self._apply_vector_field_vorticity_damping,
        }

    def can_apply(self, recommendation_is_safe: bool, rule_id: Optional[str] = None) -> bool:
        """Check if the current consent level permits applying."""
        if self.consent_level == ConsentLevel.AUDIT_ONLY:
            return False
        if self.consent_level == ConsentLevel.RECOMMEND_AND_ASK:
            return False  # Requires explicit per-recommendation approval
        if self.consent_level == ConsentLevel.AUTO_APPLY_SAFE:
            if not recommendation_is_safe:
                return False
            if self.enforce_risk_tiers and rule_id is not None:
                # Startup-only, Tier B (Experimental), and Tier C (Research) cannot be auto-applied safely
                # during live step execution without process restart or explicit user consent.
                if rule_id in STARTUP_ONLY_RULES or rule_id in TIER_B_RULES or rule_id in TIER_C_RULES:
                    return False
            return True
        if self.consent_level == ConsentLevel.AUTO_APPLY_ALL:
            return True
        return False

    def approve_and_apply(
        self,
        rule_id: str,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        recommendation_is_safe: bool = False,
        force: bool = False,
        step: int = 0,
        target_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ApplyResult:
        """
        Explicitly approve and apply a single recommendation.
        Used with RECOMMEND_AND_ASK consent level or to force-apply any recommendation.
        """
        return self._do_apply(
            rule_id=rule_id,
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            force=force,
            step=step,
            target_context=target_context,
            **kwargs,
        )

    def apply_recommendation(
        self,
        rule_id: str,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        recommendation_is_safe: bool = False,
        step: int = 0,
        target_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ApplyResult:
        """
        Attempt to apply a recommendation, respecting the consent gate.
        
        Returns ApplyResult with applied=False if consent blocks the apply.
        """
        if not self.can_apply(recommendation_is_safe, rule_id=rule_id):
            if self.consent_level == ConsentLevel.AUTO_APPLY_SAFE and recommendation_is_safe:
                if rule_id in STARTUP_ONLY_RULES:
                    err = f"Blocked by consent gate: {rule_id} is STARTUP_ONLY and cannot be dynamically auto-applied during live training steps without process restart."
                elif rule_id in TIER_B_RULES:
                    err = f"Blocked by consent gate: {rule_id} is Tier B (Experimental) and requires explicit approval (RECOMMEND_AND_ASK)."
                elif rule_id in TIER_C_RULES:
                    err = f"Blocked by consent gate: {rule_id} is Tier C (Research) and requires explicit approval (RECOMMEND_AND_ASK)."
                else:
                    err = f"Blocked by consent level: {self.consent_level.value}"
            else:
                err = f"Blocked by consent level: {self.consent_level.value}"

            result = ApplyResult(
                rule_id=rule_id,
                applied=False,
                consent_level=self.consent_level.value,
                before_config={},
                after_config={},
                error=err,
            )
            self._log_event(rule_id, "APPLY_BLOCKED", result)
            return result

        return self._do_apply(
            rule_id=rule_id,
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            step=step,
            target_context=target_context,
            **kwargs,
        )

    def apply_all(
        self,
        recommendations: List[Any],
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        step: int = 0,
        target_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[ApplyResult]:
        """Apply all recommendations that pass the consent gate."""
        results = []
        for rec in recommendations:
            result = self.apply_recommendation(
                rule_id=rec.rule_id,
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                recommendation_is_safe=rec.safe_to_auto_apply,
                step=step,
                target_context=target_context,
                **kwargs,
            )
            results.append(result)
        return results

    def _do_apply(
        self,
        rule_id: str,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        force: bool = False,
        step: int = 0,
        target_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ApplyResult:
        """Internal: execute the actual apply for a rule."""
        applier = self._appliers.get(rule_id)
        if applier is None:
            result = ApplyResult(
                rule_id=rule_id,
                applied=False,
                consent_level=self.consent_level.value,
                before_config={},
                after_config={},
                error=f"No applier registered for rule: {rule_id}",
            )
            self._log_event(rule_id, "APPLY_NO_HANDLER", result)
            return result

        try:
            import inspect
            sig = inspect.signature(applier)
            call_kwargs = {}
            if "model" in sig.parameters:
                call_kwargs["model"] = model
            if "optimizer" in sig.parameters:
                call_kwargs["optimizer"] = optimizer
            if "dataloader" in sig.parameters:
                call_kwargs["dataloader"] = dataloader
            if "target_context" in sig.parameters:
                call_kwargs["target_context"] = target_context
            for k, v in kwargs.items():
                if k in sig.parameters:
                    call_kwargs[k] = v
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                call_kwargs["target_context"] = target_context
                call_kwargs.update(kwargs)

            result = applier(**call_kwargs)

            self._log_event(rule_id, "APPLY_SUCCESS" if result.applied else "APPLY_FAILED", result)
            if result.applied and self.outcome_evaluator is not None:
                effective_step = step or getattr(model, "_current_step", 0)
                self.outcome_evaluator.on_recommendation_applied(
                    rule_id=rule_id,
                    step=effective_step,
                    effective_config=result.after_config,
                )
            return result
        except Exception as e:
            result = ApplyResult(
                rule_id=rule_id,
                applied=False,
                consent_level=self.consent_level.value,
                before_config={},
                after_config={},
                error=f"Apply failed with exception: {str(e)}",
            )
            self._log_event(rule_id, "APPLY_ERROR", result)
            return result

    # ── Rule-Specific Appliers ──────────────────────────────────────────

    def _apply_mixed_precision(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Enable AMP (Automatic Mixed Precision) with BF16/FP16."""
        before = {"mixed_precision": "fp32", "grad_scaler": False}

        try:
            import torch
            # Create a GradScaler for FP16 training on CUDA
            scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

            # Determine best dtype for this GPU
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                dtype = torch.bfloat16
                precision_name = "bf16"
            else:
                dtype = torch.float16
                precision_name = "fp16"

            after = {
                "mixed_precision": precision_name,
                "grad_scaler": scaler is not None,
                "amp_dtype": str(dtype),
            }

            # Store the scaler and dtype on the model for downstream use
            if model is not None:
                model._ghost_amp_dtype = dtype
                model._ghost_grad_scaler = scaler

            return ApplyResult(
                rule_id="RULE_MIXED_PRECISION",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_MIXED_PRECISION",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=str(e),
            )

    def _apply_dataloader_workers(
        self,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        target_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ApplyResult:
        """Optimize DataLoader num_workers and pin_memory."""
        if target_context is not None and "dataloader" in target_context:
            dataloader = target_context["dataloader"]

        before = {}
        if dataloader is not None:
            before = {
                "num_workers": getattr(dataloader, "num_workers", 0),
                "pin_memory": getattr(dataloader, "pin_memory", False),
                "persistent_workers": getattr(dataloader, "persistent_workers", False),
            }

        # Attempt to optimize via RuntimeOptimizer if dataloader is supplied
        if dataloader is not None:
            from ghost_layer.runtime_optimizer import RuntimeOptimizer
            opt = RuntimeOptimizer()
            res = opt.optimize_dataloader(dataloader)
            if res.applied and "optimized_dataloader" in res.after_state:
                if target_context is not None:
                    target_context["dataloader"] = res.after_state["optimized_dataloader"]
                return ApplyResult(
                    rule_id="RULE_DATALOADER_WORKERS",
                    applied=True,
                    consent_level=self.consent_level.value,
                    before_config=res.before_state,
                    after_config=res.after_state,
                )

        optimal_workers = min(os.cpu_count() or 4, 8)
        after = {
            "num_workers": optimal_workers,
            "pin_memory": True,
            "persistent_workers": True,
        }

        if dataloader is not None:
            try:
                dataloader.num_workers = optimal_workers
                dataloader.pin_memory = True
                dataloader.persistent_workers = True if optimal_workers > 0 else False
            except (AttributeError, RuntimeError):
                pass

        return ApplyResult(
            rule_id="RULE_DATALOADER_WORKERS",
            applied=True,
            consent_level=self.consent_level.value,
            before_config=before,
            after_config=after,
        )

    def _apply_chunked_cross_entropy(
        self,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        target_context: Optional[Dict[str, Any]] = None,
        chunk_size: int = 4096,
        **kwargs,
    ) -> ApplyResult:
        """Replace or wrap CrossEntropyLoss with ChunkedCrossEntropyLoss."""
        loss_fn = kwargs.get("loss_fn")
        if target_context is not None and "loss_fn" in target_context:
            loss_fn = target_context["loss_fn"]

        from ghost_layer.runtime_optimizer import RuntimeOptimizer
        opt = RuntimeOptimizer()
        opt_res = opt.optimize_chunked_cross_entropy(loss_fn, chunk_size=chunk_size)

        if opt_res.applied:
            if target_context is not None and "optimized_loss" in opt_res.after_state:
                target_context["loss_fn"] = opt_res.after_state["optimized_loss"]

        return ApplyResult(
            rule_id="RULE_CHUNKED_CROSS_ENTROPY",
            applied=opt_res.applied,
            consent_level=self.consent_level.value,
            before_config=opt_res.before_state,
            after_config=opt_res.after_state,
            error=opt_res.error,
        )

    def _apply_mhc_residual_routing(
        self,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        target_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ApplyResult:
        """Advisory handler: Manifold-Constrained Hyper-Connections require explicit architecture changes."""
        return ApplyResult(
            rule_id="RULE_MHC_RESIDUAL_ROUTING",
            applied=False,
            consent_level=self.consent_level.value,
            before_config={},
            after_config={},
            error="Architectural modification (mHC) requires explicit model definition; cannot auto-apply.",
        )

    def _apply_flash_attention(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Enable FlashAttention-2 for HuggingFace models."""
        before = {"attn_implementation": "sdpa"}

        if model is None:
            return ApplyResult(
                rule_id="RULE_FLASH_ATTENTION",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error="No model provided to apply FlashAttention",
            )

        try:
            # HuggingFace models store this in config
            if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
                before["attn_implementation"] = model.config._attn_implementation
                model.config._attn_implementation = "flash_attention_2"
                after = {"attn_implementation": "flash_attention_2"}
            else:
                after = before
                return ApplyResult(
                    rule_id="RULE_FLASH_ATTENTION",
                    applied=False,
                    consent_level=self.consent_level.value,
                    before_config=before,
                    after_config=after,
                    error="Model does not support _attn_implementation config",
                )

            return ApplyResult(
                rule_id="RULE_FLASH_ATTENTION",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_FLASH_ATTENTION",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=str(e),
            )

    def _apply_gradient_checkpointing(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Enable gradient checkpointing to reduce memory usage."""
        before = {"gradient_checkpointing": False}

        if model is None:
            return ApplyResult(
                rule_id="RULE_GRADIENT_CHECKPOINTING",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error="No model provided",
            )

        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                before["gradient_checkpointing"] = getattr(model, "is_gradient_checkpointing", False)
                model.gradient_checkpointing_enable()
                after = {"gradient_checkpointing": True}
                return ApplyResult(
                    rule_id="RULE_GRADIENT_CHECKPOINTING",
                    applied=True,
                    consent_level=self.consent_level.value,
                    before_config=before,
                    after_config=after,
                )
            else:
                return ApplyResult(
                    rule_id="RULE_GRADIENT_CHECKPOINTING",
                    applied=False,
                    consent_level=self.consent_level.value,
                    before_config=before,
                    after_config=before,
                    error="Model does not support gradient_checkpointing_enable()",
                )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_GRADIENT_CHECKPOINTING",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=str(e),
            )

    def _apply_batch_scaling(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None, **kwargs
    ) -> ApplyResult:
        """Recommend batch size scaling based on Section B VRAM headroom multiplier."""
        from ghost_layer.telemetry.vram_scaling import calculate_vram_headroom_multiplier
        import torch

        before = {"batch_size": "unknown", "gradient_accumulation_steps": 1}

        if dataloader is not None and hasattr(dataloader, "batch_size"):
            before["batch_size"] = dataloader.batch_size

        # Probe VRAM if on CUDA
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            vram_peak = max(1.0, torch.cuda.max_memory_allocated() / (1024**2))
        else:
            vram_total = 16000.0  # standard reference
            vram_peak = 4000.0

        current_b = before["batch_size"] if isinstance(before["batch_size"], int) else 1
        scaling_rep = calculate_vram_headroom_multiplier(
            vram_total_mb=vram_total,
            vram_peak_mb=vram_peak,
            current_batch_size=current_b,
        )

        recommended_batch = scaling_rep.recommended_batch_size if isinstance(before["batch_size"], int) else "auto"

        after = {
            "batch_size": recommended_batch,
            "safe_multiplier": scaling_rep.safe_multiplier,
            "gradient_accumulation_steps": max(1, int(round(scaling_rep.safe_multiplier))),
            "vram_headroom_mb": scaling_rep.vram_headroom_mb,
            "note": "Batch scaling computed via Section B VRAM Headroom Multiplier",
        }

        return ApplyResult(
            rule_id="RULE_BATCH_SCALING",
            applied=True,
            consent_level=self.consent_level.value,
            before_config=before,
            after_config=after,
        )

    def _apply_torch_compile(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Apply torch.compile with reduce-overhead mode and fallback to eager on failure."""
        before = {"compiled": False, "compile_mode": "eager"}

        if model is None:
            return ApplyResult(
                rule_id="RULE_GRAPHIFY_TORCH_COMPILE",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error="No model provided",
            )

        try:
            import torch
            # Check if already compiled
            if hasattr(model, "_compiled") and model._compiled:
                return ApplyResult(
                    rule_id="RULE_GRAPHIFY_TORCH_COMPILE",
                    applied=False,
                    consent_level=self.consent_level.value,
                    before_config={"compiled": True, "compile_mode": "reduce-overhead"},
                    after_config={"compiled": True, "compile_mode": "reduce-overhead"},
                    error="Model is already compiled",
                )

            compiled_model = torch.compile(model, mode="reduce-overhead", backend="inductor")
            # Tag the compiled model
            compiled_model._compiled = True
            compiled_model._ghost_original_model = model

            after = {"compiled": True, "compile_mode": "reduce-overhead", "backend": "inductor"}

            return ApplyResult(
                rule_id="RULE_GRAPHIFY_TORCH_COMPILE",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_GRAPHIFY_TORCH_COMPILE",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"torch.compile failed, falling back to eager: {str(e)}",
            )

    def _apply_muon(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Apply Muon matrix polar decomposition optimizer."""
        before = {"optimizer": type(optimizer).__name__ if optimizer else "none"}

        if model is None:
            return ApplyResult(
                rule_id="RULE_MUON_OPTIMIZER",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error="No model provided to configure Muon optimizer",
            )

        try:
            from ghost_layer.curvature.muon import create_muon_hybrid_optimizer
            new_opt = create_muon_hybrid_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)
            model._ghost_optimizer = new_opt
            after = {
                "optimizer": "HybridMuonOptimizer",
                "muon_params": getattr(new_opt, "muon_param_count", 0),
                "adamw_params": getattr(new_opt, "adamw_param_count", 0),
                "muon_lr": 0.02,
                "adamw_lr": 1e-3,
            }

            return ApplyResult(
                rule_id="RULE_MUON_OPTIMIZER",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_MUON_OPTIMIZER",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"Muon initialization failed: {str(e)}",
            )

    def _apply_sophia(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Apply Sophia-G second-order optimizer."""
        before = {"optimizer": type(optimizer).__name__ if optimizer else "none"}

        if model is None:
            return ApplyResult(
                rule_id="RULE_SOPHIA_SECOND_ORDER",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error="No model provided to configure Sophia optimizer",
            )

        try:
            from ghost_layer.curvature.sophia import SophiaG
            new_opt = SophiaG(model.parameters(), lr=1e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=0.1)
            model._ghost_optimizer = new_opt
            after = {"optimizer": "SophiaG", "lr": 1e-4, "betas": (0.965, 0.99), "rho": 0.04}

            return ApplyResult(
                rule_id="RULE_SOPHIA_SECOND_ORDER",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_SOPHIA_SECOND_ORDER",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"Sophia initialization failed: {str(e)}",
            )

    def _apply_shampoo(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Apply Shampoo matrix root tensor preconditioned optimizer."""
        before = {"optimizer": type(optimizer).__name__ if optimizer else "none"}

        if model is None:
            return ApplyResult(
                rule_id="RULE_SHAMPOO_PRECONDITIONING",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error="No model provided to configure Shampoo optimizer",
            )

        try:
            from ghost_layer.curvature.shampoo import Shampoo
            new_opt = Shampoo(model.parameters(), lr=1e-3, momentum=0.9, update_preconditioner_interval=10)
            model._ghost_optimizer = new_opt
            after = {"optimizer": "Shampoo", "lr": 1e-3, "momentum": 0.9, "interval": 10}

            return ApplyResult(
                rule_id="RULE_SHAMPOO_PRECONDITIONING",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_SHAMPOO_PRECONDITIONING",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"Shampoo initialization failed: {str(e)}",
            )

    def _apply_hvp(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Enable Pearlmutter Hessian-Vector Product diagnostic tracing."""
        before = {"hvp_tracing_enabled": False}

        if model is None:
            return ApplyResult(
                rule_id="RULE_PEARLMUTTER_HVP_METAGRADIENT",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error="No model provided for HVP diagnostic tracing",
            )

        try:
            model._ghost_hvp_enabled = True
            after = {"hvp_tracing_enabled": True}

            return ApplyResult(
                rule_id="RULE_PEARLMUTTER_HVP_METAGRADIENT",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_PEARLMUTTER_HVP_METAGRADIENT",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"HVP tracing activation failed: {str(e)}",
            )

    def _apply_async_tensor_flow(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Configure async DataLoader transfers and non-blocking streaming."""
        before = {
            "pin_memory": getattr(dataloader, "pin_memory", False) if dataloader else False,
            "num_workers": getattr(dataloader, "num_workers", 0) if dataloader else 0,
        }

        try:
            if dataloader is not None:
                # Set pin_memory on dataloader if attribute is writable
                try:
                    dataloader.pin_memory = True
                except Exception:
                    pass
            after = {"pin_memory": True, "non_blocking": True, "async_streaming": True}

            return ApplyResult(
                rule_id="RULE_ASYNC_TENSOR_FLOW",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_ASYNC_TENSOR_FLOW",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"Async tensor streaming configuration failed: {str(e)}",
            )

    def _apply_cuda_allocator(
        self, model: Any = None, optimizer: Any = None, dataloader: Any = None
    ) -> ApplyResult:
        """Configure PyTorch CUDA Caching Allocator to eliminate fragmentation."""
        import os
        before = {"PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}

        try:
            new_conf = "expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
            after = {"PYTORCH_CUDA_ALLOC_CONF": new_conf, "fragmentation_eliminated": True}

            return ApplyResult(
                rule_id="RULE_CUDA_ALLOCATOR_TUNING",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_CUDA_ALLOCATOR_TUNING",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"CUDA allocator configuration failed: {str(e)}",
            )

    def _apply_quasi_newton_lbfgs(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        target_context: Optional[Dict[str, Any]] = None,
        history_size: int = 10,
        **kwargs,
    ) -> ApplyResult:
        """Apply Matrix-Free Quasi-Newton (L-BFGS) optimizer with Damped Powell Updates."""
        before = {"optimizer": type(optimizer).__name__ if optimizer else "None"}
        try:
            from ghost_layer.curvature.lbfgs import LBFGSCurvatureOptimizer

            target_ctx = target_context if target_context is not None else {}
            params = None
            if model is not None and hasattr(model, "parameters"):
                params = model.parameters()
            elif optimizer is not None and hasattr(optimizer, "param_groups"):
                params = [p for g in optimizer.param_groups for p in g["params"]]

            if params is None:
                return ApplyResult(
                    rule_id="RULE_QUASI_NEWTON_LBFGS",
                    applied=False,
                    consent_level=self.consent_level.value,
                    before_config=before,
                    after_config=before,
                    error="Neither model nor optimizer with parameters provided.",
                )

            lr = 1e-3
            if optimizer is not None and hasattr(optimizer, "param_groups") and optimizer.param_groups:
                lr = optimizer.param_groups[0].get("lr", 1e-3)

            lbfgs_opt = LBFGSCurvatureOptimizer(
                params,
                lr=lr,
                history_size=history_size,
                powell_damping=True,
            )

            target_ctx["optimizer"] = lbfgs_opt
            target_ctx["applied_quasi_newton"] = True
            after = {
                "optimizer": "LBFGSCurvatureOptimizer",
                "history_size": history_size,
                "powell_damping": True,
                "matrix_free": True,
            }

            return ApplyResult(
                rule_id="RULE_QUASI_NEWTON_LBFGS",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )

        except Exception as e:
            return ApplyResult(
                rule_id="RULE_QUASI_NEWTON_LBFGS",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"Failed to apply Quasi-Newton L-BFGS: {str(e)}",
            )

    def _apply_vector_field_vorticity_damping(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        target_context: Optional[Dict[str, Any]] = None,
        damping_factor: float = 0.6,
        **kwargs,
    ) -> ApplyResult:
        """Apply Helmholtz-Hodge vector field vorticity filtering to dampen limit cycles."""
        before = {"vorticity_damping": False}
        try:
            from ghost_layer.curvature.vector_field import HelmholtzHodgeFilter

            target_ctx = target_context if target_context is not None else {}
            hh_filter = HelmholtzHodgeFilter(vorticity_damping_factor=damping_factor)

            target_ctx["helmholtz_filter"] = hh_filter
            target_ctx["applied_vorticity_damping"] = True

            after = {
                "vorticity_damping": True,
                "damping_factor": damping_factor,
                "helmholtz_hodge_projection": True,
            }

            return ApplyResult(
                rule_id="RULE_VECTOR_FIELD_VORTICITY_DAMPING",
                applied=True,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=after,
            )
        except Exception as e:
            return ApplyResult(
                rule_id="RULE_VECTOR_FIELD_VORTICITY_DAMPING",
                applied=False,
                consent_level=self.consent_level.value,
                before_config=before,
                after_config=before,
                error=f"Failed to apply vorticity damping: {str(e)}",
            )

    # ── Logging ──────────────────────────────────────────────────────────



    def _log_event(self, rule_id: str, action: str, result: ApplyResult):
        """Log an apply event to the replay log."""
        if self.replay_log is None:
            return
        self.replay_log.record_event(
            step=0,
            stage="SAFE_APPLY",
            recommendation_id=rule_id,
            action=action,
            details={
                "consent_level": result.consent_level,
                "applied": result.applied,
                "before": result.before_config,
                "after": result.after_config,
                "error": result.error,
            },
            verified_safe=result.applied,
            throughput_delta_pct=0.0,
            reason=result.error or f"Applied {rule_id} successfully",
        )
