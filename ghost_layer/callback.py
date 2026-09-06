"""
Universal Zero-Friction Callback & Decorator Integration for GhostLayer.
Supports:
1. HuggingFace Transformers `TrainerCallback`
2. PyTorch Lightning `Callback`
3. Raw PyTorch `ghost_watch()` context manager and decorator
"""

import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.reporting.generator import ReportGenerator

# Try importing Hugging Face TrainerCallback gracefully
try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    _HF_AVAILABLE = True
except ImportError:
    # Graceful fallback mock so transformers is not a hard mandatory dependency
    class TrainerCallback:
        pass
    class TrainerControl:
        pass
    class TrainerState:
        pass
    class TrainingArguments:
        pass
    _HF_AVAILABLE = False


class GhostTrainerCallback(TrainerCallback):
    """
    HuggingFace Transformers Trainer integration.
    One-line attach:
        trainer.add_callback(GhostTrainerCallback(cluster_name="node-01"))
    """
    def __init__(
        self,
        cluster_name: str = "default-cluster",
        gpu_memory_mb: float = 16384.0,
        gpu_cost_per_hour: float = 3.50,
        output_report_path: Optional[str] = "ghost_executive_audit.html",
        auto_generate_report: bool = True,
    ):
        self.cluster_name = cluster_name
        self.gpu_memory_mb = gpu_memory_mb
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.output_report_path = output_report_path
        self.auto_generate_report = auto_generate_report
        
        self.hook = GhostWatcherHook(
            gpu_memory_mb=gpu_memory_mb,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        self._is_active = False

    def on_train_begin(self, args: Any = None, state: Any = None, control: Any = None, **kwargs):
        self.hook.watcher.start()
        self._is_active = True

    def on_step_begin(self, args: Any = None, state: Any = None, control: Any = None, **kwargs):
        if self._is_active:
            self.hook.on_step_begin()

    def on_step_end(self, args: Any = None, state: Any = None, control: Any = None, **kwargs):
        if not self._is_active:
            return

        # Extract loss and step metrics if available from TrainerState or kwargs
        loss = None
        if state is not None and hasattr(state, "log_history") and state.log_history:
            latest = state.log_history[-1]
            loss = latest.get("loss", None)

        # Detect GPU memory usage if torch.cuda is available
        vram_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass

        self.hook.on_step_end(
            gpu_util_pct=None,
            gpu_mem_used_mb=vram_mb if vram_mb > 0 else None,
            loss=loss,
            mixed_precision="bf16" if (args and getattr(args, "bf16", False)) else ("fp16" if (args and getattr(args, "fp16", False)) else "fp32"),
            gradient_checkpointing=bool(args and getattr(args, "gradient_checkpointing", False)),
        )

    def on_train_end(self, args: Any = None, state: Any = None, control: Any = None, **kwargs):
        if not self._is_active:
            return
            
        self.hook.watcher.stop()
        self._is_active = False
        
        if self.auto_generate_report and self.output_report_path:
            from ghost_layer.reporting.executive_exporter import ExecutiveAuditExporter
            exporter = ExecutiveAuditExporter(self.hook)
            exporter.export_html(self.output_report_path)


class GhostLightningCallback:
    """
    PyTorch Lightning integration callback.
    Attach via:
        trainer = pl.Trainer(callbacks=[GhostLightningCallback()])
    """
    def __init__(
        self,
        gpu_memory_mb: float = 16384.0,
        gpu_cost_per_hour: float = 3.50,
        output_report_path: Optional[str] = "ghost_executive_audit.html",
    ):
        self.gpu_memory_mb = gpu_memory_mb
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.output_report_path = output_report_path
        self.hook = GhostWatcherHook(
            gpu_memory_mb=gpu_memory_mb,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )

    def on_train_start(self, trainer: Any, pl_module: Any):
        self.hook.watcher.start()

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int):
        self.hook.on_step_begin()

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int):
        loss_val = None
        if isinstance(outputs, dict):
            loss_val = outputs.get("loss", None)
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
        elif hasattr(outputs, "item"):
            loss_val = outputs.item()

        vram_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass

        self.hook.on_step_end(
            gpu_util_pct=None,
            gpu_mem_used_mb=vram_mb if vram_mb > 0 else None,
            loss=loss_val,
        )

    def on_train_end(self, trainer: Any, pl_module: Any):
        self.hook.watcher.stop()
        if self.output_report_path:
            from ghost_layer.reporting.executive_exporter import ExecutiveAuditExporter
            exporter = ExecutiveAuditExporter(self.hook)
            exporter.export_html(self.output_report_path)


class ghost_watch:
    """
    Context manager & decorator for raw PyTorch scripts.
    Usage 1 (Context Manager):
        with ghost_watch(gpu_memory_mb=16384.0) as gw:
            for step, batch in enumerate(dataloader):
                gw.step(loss=loss.item())
                
    Usage 2 (Decorator):
        @ghost_watch()
        def train_epoch():
            ...
    """
    def __init__(
        self,
        gpu_memory_mb: float = 16384.0,
        gpu_cost_per_hour: float = 3.50,
        output_report_path: Optional[str] = "ghost_audit.html",
    ):
        self.gpu_memory_mb = gpu_memory_mb
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.output_report_path = output_report_path
        self.hook = GhostWatcherHook(
            gpu_memory_mb=gpu_memory_mb,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        self._in_step = False

    def __enter__(self):
        self.hook.watcher.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hook.watcher.stop()
        if self.output_report_path:
            from ghost_layer.reporting.executive_exporter import ExecutiveAuditExporter
            exporter = ExecutiveAuditExporter(self.hook)
            exporter.export_html(self.output_report_path)

    def step(
        self,
        loss: Optional[float] = None,
        gpu_util_pct: Optional[float] = None,
        gpu_mem_used_mb: Optional[float] = None,
        mixed_precision: str = "fp32",
        dataloader_time_ms: float = 0.0,
    ):
        """Record a single step iteration with direct hardware probe fallback."""
        if not self._in_step:
            self.hook.on_step_begin()
            
        self.hook.on_step_end(
            gpu_util_pct=gpu_util_pct,
            gpu_mem_used_mb=gpu_mem_used_mb,
            loss=loss,
            dataloader_time_ms=dataloader_time_ms,
            mixed_precision=mixed_precision,
        )
        self._in_step = False


    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
