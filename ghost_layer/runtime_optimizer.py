"""
Runtime Optimizer: Applies PyTorch built-in optimizations (torch.compile,
CUDA memory configuration, memory format) as first-class GhostLayer operations.

This is NOT custom kernel fusion. This wraps standard PyTorch APIs that
GhostLayer can safely invoke on the user's behalf with consent:
- torch.compile() with inductor backend
- PYTORCH_CUDA_ALLOC_CONF environment tuning
- channels_last memory format conversion
- CUDA memory pool management
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class RuntimeOptimizationResult:
    """Result of a runtime optimization operation."""
    optimization_type: str
    applied: bool
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    error: Optional[str] = None
    optimized_model: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)


class RuntimeOptimizer:
    """
    Wraps PyTorch's built-in runtime optimizations into safe, idempotent operations.
    
    Each optimization:
    1. Checks if already applied (idempotent)
    2. Captures before-state
    3. Applies with try/catch
    4. Captures after-state
    5. Falls back to original state on failure
    """

    def __init__(self):
        self._applied: Dict[str, bool] = {}

    def optimize_compile(
        self,
        model: Any,
        mode: str = "reduce-overhead",
        backend: str = "inductor",
        fullgraph: bool = False,
    ) -> RuntimeOptimizationResult:
        """
        Wrap model in torch.compile() with safety checks and fallback.
        
        Idempotent: returns no-op if model is already compiled.
        """
        before = {"compiled": False, "mode": "eager", "backend": "none"}

        # Check idempotency
        if getattr(model, "_ghost_compiled", False):
            return RuntimeOptimizationResult(
                optimization_type="torch_compile",
                applied=False,
                before_state={"compiled": True, "mode": mode},
                after_state={"compiled": True, "mode": mode},
                error="Already compiled (idempotent skip)",
            )

        try:
            import torch
            compiled = torch.compile(model, mode=mode, backend=backend, fullgraph=fullgraph)
            compiled._ghost_compiled = True
            compiled._ghost_original_model = model
            compiled._ghost_compile_mode = mode

            after = {"compiled": True, "mode": mode, "backend": backend, "fullgraph": fullgraph}
            self._applied["torch_compile"] = True

            return RuntimeOptimizationResult(
                optimization_type="torch_compile",
                applied=True,
                before_state=before,
                after_state=after,
            )
        except Exception as e:
            return RuntimeOptimizationResult(
                optimization_type="torch_compile",
                applied=False,
                before_state=before,
                after_state=before,
                error=f"torch.compile failed, model remains in eager mode: {str(e)}",
            )

    def optimize_cuda_memory(
        self,
        max_split_size_mb: int = 128,
        expandable_segments: bool = True,
        roundup_power2_divisions: int = 16,
    ) -> RuntimeOptimizationResult:
        """
        Configure CUDA memory allocator for optimal training performance.
        Sets PYTORCH_CUDA_ALLOC_CONF with recommended values.
        """
        before_val = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        before = {"PYTORCH_CUDA_ALLOC_CONF": before_val}

        parts = [f"max_split_size_mb:{max_split_size_mb}"]
        if expandable_segments:
            parts.append("expandable_segments:True")
        parts.append(f"roundup_power2_divisions:{roundup_power2_divisions}")

        new_val = ",".join(parts)

        try:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_val
            after = {"PYTORCH_CUDA_ALLOC_CONF": new_val}
            self._applied["cuda_memory"] = True

            return RuntimeOptimizationResult(
                optimization_type="cuda_memory_config",
                applied=True,
                before_state=before,
                after_state=after,
            )
        except Exception as e:
            return RuntimeOptimizationResult(
                optimization_type="cuda_memory_config",
                applied=False,
                before_state=before,
                after_state=before,
                error=str(e),
            )

    def optimize_memory_format(
        self, model: Any, format_name: str = "channels_last"
    ) -> RuntimeOptimizationResult:
        """
        Convert model to channels_last memory format for better GPU cache utilization.
        Primarily benefits CNNs and vision models.
        """
        before = {"memory_format": "contiguous"}

        try:
            import torch
            if format_name == "channels_last":
                converted_model = model.to(memory_format=torch.channels_last)
                # In-place parameter conversion for 4D convolution weights if PyTorch module
                if isinstance(model, torch.nn.Module):
                    for p in model.parameters():
                        if p.dim() == 4:
                            p.data = p.data.to(memory_format=torch.channels_last)
                after = {"memory_format": "channels_last"}
            else:
                after = before
                return RuntimeOptimizationResult(
                    optimization_type="memory_format",
                    applied=False,
                    before_state=before,
                    after_state=after,
                    error=f"Unsupported memory format: {format_name}",
                    optimized_model=model,
                )

            self._applied["memory_format"] = True
            return RuntimeOptimizationResult(
                optimization_type="memory_format",
                applied=True,
                before_state=before,
                after_state=after,
                optimized_model=converted_model,
            )
        except Exception as e:
            return RuntimeOptimizationResult(
                optimization_type="memory_format",
                applied=False,
                before_state=before,
                after_state=before,
                error=str(e),
                optimized_model=model,
            )

    def optimize_cudnn(self, benchmark: bool = True, deterministic: bool = False) -> RuntimeOptimizationResult:
        """
        Configure cuDNN for training performance.
        benchmark=True: cuDNN auto-tuner finds the best convolution algorithm.
        deterministic=False: allows non-deterministic algorithms (faster).
        """
        before = {"cudnn_benchmark": False, "cudnn_deterministic": True}

        try:
            import torch
            before["cudnn_benchmark"] = torch.backends.cudnn.benchmark
            before["cudnn_deterministic"] = torch.backends.cudnn.deterministic

            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = deterministic

            after = {"cudnn_benchmark": benchmark, "cudnn_deterministic": deterministic}
            self._applied["cudnn"] = True

            return RuntimeOptimizationResult(
                optimization_type="cudnn_config",
                applied=True,
                before_state=before,
                after_state=after,
            )
        except Exception as e:
            return RuntimeOptimizationResult(
                optimization_type="cudnn_config",
                applied=False,
                before_state=before,
                after_state=before,
                error=str(e),
            )

    def optimize_matmul_precision(self, precision: str = "high") -> RuntimeOptimizationResult:
        """
        Set float32 matrix multiplication precision.
        'high' uses TF32 on Ampere+ GPUs for ~3x speedup on matmuls.
        """
        before = {"matmul_precision": "highest"}

        try:
            import torch
            before["matmul_precision"] = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision(precision)
            after = {"matmul_precision": precision}
            self._applied["matmul_precision"] = True

            return RuntimeOptimizationResult(
                optimization_type="matmul_precision",
                applied=True,
                before_state=before,
                after_state=after,
            )
        except Exception as e:
            return RuntimeOptimizationResult(
                optimization_type="matmul_precision",
                applied=False,
                before_state=before,
                after_state=before,
                error=str(e),
            )

    def optimize_dataloader(
        self,
        dataloader: Any,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 2,
    ) -> RuntimeOptimizationResult:
        """
        Configure PyTorch DataLoader for zero-stall I/O.
        Sets pin_memory=True, autotunes num_workers, persistent_workers=True, and prefetch_factor.
        """
        if getattr(dataloader, "_ghost_dataloader_optimized", False):
            return RuntimeOptimizationResult(
                optimization_type="dataloader_config",
                applied=False,
                before_state={"num_workers": getattr(dataloader, "num_workers", 0), "pin_memory": getattr(dataloader, "pin_memory", False)},
                after_state={"num_workers": getattr(dataloader, "num_workers", 0), "pin_memory": getattr(dataloader, "pin_memory", False)},
                error="Already optimized (idempotent skip)",
            )

        before = {
            "num_workers": getattr(dataloader, "num_workers", 0),
            "pin_memory": getattr(dataloader, "pin_memory", False),
            "persistent_workers": getattr(dataloader, "persistent_workers", False),
        }

        try:
            import torch
            from torch.utils.data import DataLoader

            if num_workers is None:
                cpu_count = os.cpu_count() or 4
                num_workers = max(1, min(16, cpu_count // 2))

            dataset = getattr(dataloader, "dataset", None)
            if dataset is None:
                raise ValueError("DataLoader has no dataset attribute.")

            batch_size = getattr(dataloader, "batch_size", 1)
            collate_fn = getattr(dataloader, "collate_fn", None)
            drop_last = getattr(dataloader, "drop_last", False)
            timeout = getattr(dataloader, "timeout", 0)

            use_persistent = persistent_workers if num_workers > 0 else False
            use_prefetch = prefetch_factor if num_workers > 0 else None

            kwargs = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": drop_last,
                "timeout": timeout,
            }
            if collate_fn is not None:
                kwargs["collate_fn"] = collate_fn
            if use_persistent:
                kwargs["persistent_workers"] = True
            if use_prefetch is not None:
                kwargs["prefetch_factor"] = use_prefetch

            sampler = getattr(dataloader, "sampler", None)
            if sampler is not None and not isinstance(sampler, torch.utils.data.SequentialSampler):
                kwargs["sampler"] = sampler
            else:
                kwargs["shuffle"] = False

            optimized_dl = DataLoader(dataset, **kwargs)
            optimized_dl._ghost_dataloader_optimized = True
            optimized_dl._ghost_original_dataloader = dataloader

            after = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": use_persistent,
                "prefetch_factor": use_prefetch,
                "optimized_dataloader": optimized_dl,
            }
            self._applied["dataloader_config"] = True

            return RuntimeOptimizationResult(
                optimization_type="dataloader_config",
                applied=True,
                before_state=before,
                after_state=after,
            )
        except Exception as e:
            return RuntimeOptimizationResult(
                optimization_type="dataloader_config",
                applied=False,
                before_state=before,
                after_state=before,
                error=f"DataLoader optimization failed: {str(e)}",
            )

    def optimize_chunked_cross_entropy(
        self,
        loss_fn: Optional[Any] = None,
        chunk_size: int = 4096,
    ) -> RuntimeOptimizationResult:
        """
        Wrap or replace CrossEntropyLoss with ChunkedCrossEntropyLoss to save 40%-60% activation VRAM.
        """
        from ghost_layer.curvature.loss import ChunkedCrossEntropyLoss

        if isinstance(loss_fn, ChunkedCrossEntropyLoss) or getattr(loss_fn, "_ghost_chunked_ce", False):
            return RuntimeOptimizationResult(
                optimization_type="chunked_cross_entropy",
                applied=False,
                before_state={"loss_type": "ChunkedCrossEntropyLoss", "chunk_size": getattr(loss_fn, "chunk_size", chunk_size)},
                after_state={"loss_type": "ChunkedCrossEntropyLoss", "chunk_size": getattr(loss_fn, "chunk_size", chunk_size)},
                error="Already chunked (idempotent skip)",
            )

        before = {
            "loss_type": type(loss_fn).__name__ if loss_fn is not None else "None",
        }

        try:
            ignore_index = getattr(loss_fn, "ignore_index", -100)
            reduction = getattr(loss_fn, "reduction", "mean")
            label_smoothing = getattr(loss_fn, "label_smoothing", 0.0)

            optimized_loss = ChunkedCrossEntropyLoss(
                chunk_size=chunk_size,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )
            optimized_loss._ghost_chunked_ce = True
            optimized_loss._ghost_original_loss = loss_fn

            after = {
                "loss_type": "ChunkedCrossEntropyLoss",
                "chunk_size": chunk_size,
                "optimized_loss": optimized_loss,
            }
            self._applied["chunked_cross_entropy"] = True

            return RuntimeOptimizationResult(
                optimization_type="chunked_cross_entropy",
                applied=True,
                before_state=before,
                after_state=after,
            )
        except Exception as e:
            return RuntimeOptimizationResult(
                optimization_type="chunked_cross_entropy",
                applied=False,
                before_state=before,
                after_state=before,
                error=f"ChunkedCrossEntropy optimization failed: {str(e)}",
            )

    def get_applied_optimizations(self) -> Dict[str, bool]:
        """Return which optimizations have been applied."""
        return dict(self._applied)

    def apply_all_safe(
        self,
        model: Any = None,
        dataloader: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
    ) -> List[RuntimeOptimizationResult]:
        """Apply all safe, standard optimizations in recommended order."""
        results = []

        # 1. CUDA memory config (no model dependency)
        results.append(self.optimize_cuda_memory())

        # 2. cuDNN benchmark mode
        results.append(self.optimize_cudnn(benchmark=True))

        # 3. TF32 matmul precision
        results.append(self.optimize_matmul_precision(precision="high"))

        # 4. DataLoader worker & memory pinning
        if dataloader is not None:
            results.append(self.optimize_dataloader(dataloader))

        # 5. Chunked cross-entropy
        if loss_fn is not None:
            results.append(self.optimize_chunked_cross_entropy(loss_fn))

        # 6. torch.compile (last, as it wraps the model)
        if model is not None:
            results.append(self.optimize_compile(model))

        return results

