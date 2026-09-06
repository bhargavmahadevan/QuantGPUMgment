import time
from dataclasses import dataclass
from typing import List, Optional
import os
import numpy as np

@dataclass
class MetricSnapshot:
    step: int
    timestamp: float
    gpu_utilization_pct: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    step_time_ms: float
    loss: Optional[float] = None
    data_loading_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    cpu_utilization_pct: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    mixed_precision: str = "fp32"  # fp32, fp16, bf16
    gradient_checkpointing: bool = False
    flash_attention: bool = False
    rank: int = 0
    world_size: int = 1
    attention_architecture: str = "MHA"
    sequence_length: int = 2048
    estimated_kv_cache_mb: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    dominant_latency_bottleneck: str = "COMPUTE_BOUND"

@dataclass
class TelemetrySummary:
    total_steps: int
    total_duration_sec: float
    avg_gpu_utilization_pct: float
    peak_gpu_memory_mb: float
    gpu_memory_total_mb: float
    avg_step_time_ms: float
    avg_dataloader_stall_pct: float
    mixed_precision: str
    gradient_checkpointing: bool
    flash_attention: bool
    num_workers: int
    pin_memory: bool
    rank: int = 0
    world_size: int = 1
    attention_architecture: str = "MHA"
    sequence_length: int = 2048
    estimated_kv_cache_mb: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    dominant_latency_bottleneck: str = "COMPUTE_BOUND"

class TelemetryWatcher:
    """
    Lightweight, non-invasive telemetry watcher that tracks ML training step performance,
    GPU/CPU utilization, memory allocation, and data loading bottlenecks.
    Overhead target: < 1.5% runtime penalty.
    """
    def __init__(
        self,
        target_gpu_mb: float = 16384.0,
        rank: int = 0,
        world_size: int = 1,
        persistence_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.snapshots: List[MetricSnapshot] = []
        self.gpu_memory_total_mb = target_gpu_mb
        self.rank = rank
        self.world_size = world_size
        self.is_active = False
        self._start_time = 0.0
        self.variance_errors: List[float] = []
        self._last_loss: Optional[float] = None
        
        self.persistence = None
        if persistence_dir or session_id:
            from ghost_layer.telemetry.persistence import TelemetryPersistence
            self.persistence = TelemetryPersistence(storage_dir=persistence_dir, session_id=session_id)

    def start(self, resume_prior: bool = False):
        if not resume_prior:
            self.snapshots.clear()
            self.variance_errors.clear()
            self._last_loss = None
        self.is_active = True
        self._start_time = time.time()

    def resume_from_file(self, file_path: str):
        """Resume telemetry state from a saved session JSONL file."""
        from ghost_layer.telemetry.persistence import TelemetryPersistence
        loaded = TelemetryPersistence.load_from_file(file_path)
        if loaded:
            self.snapshots = loaded
            self.variance_errors.clear()
            self._last_loss = None
            for s in loaded:
                if s.loss is not None:
                    if self._last_loss is not None:
                        self.variance_errors.append(s.loss - self._last_loss)
                    self._last_loss = s.loss
            self.is_active = True
            self._start_time = time.time() - (self.snapshots[-1].timestamp if self.snapshots else 0.0)

    def stop(self):
        self.is_active = False

    def record_step(
        self,
        step: int,
        gpu_utilization_pct: float,
        gpu_memory_used_mb: float,
        step_time_ms: float,
        loss: Optional[float] = None,
        data_loading_time_ms: float = 0.0,
        cpu_utilization_pct: float = 20.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        mixed_precision: str = "fp32",
        gradient_checkpointing: bool = False,
        flash_attention: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        attention_architecture: str = "MHA",
        sequence_length: int = 2048,
        estimated_kv_cache_mb: float = 0.0,
        ttft_ms: float = 0.0,
        tpot_ms: float = 0.0,
        dominant_latency_bottleneck: str = "COMPUTE_BOUND",
    ) -> MetricSnapshot:
        # Data loading time is caller-supplied (the watcher doesn't instrument the DataLoader itself),
        # so it can't be trusted to stay within the measured wall-clock step time. Clamp it at the
        # source rather than only at the aggregate, so a single bad caller value can't silently produce
        # a >100% stall reading in the summary.
        safe_data_loading_time_ms = max(0.0, min(data_loading_time_ms, step_time_ms)) if step_time_ms > 0 else 0.0

        active_rank = self.rank if rank is None else rank
        active_world = self.world_size if world_size is None else world_size

        snapshot = MetricSnapshot(
            step=step,
            timestamp=time.time() - self._start_time if self._start_time else 0.0,
            gpu_utilization_pct=max(0.0, min(100.0, gpu_utilization_pct)),
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=self.gpu_memory_total_mb,
            step_time_ms=step_time_ms,
            loss=loss,
            data_loading_time_ms=safe_data_loading_time_ms,
            cpu_utilization_pct=cpu_utilization_pct,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing,
            flash_attention=flash_attention,
            rank=active_rank,
            world_size=active_world,
            attention_architecture=attention_architecture,
            sequence_length=sequence_length,
            estimated_kv_cache_mb=estimated_kv_cache_mb,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            dominant_latency_bottleneck=dominant_latency_bottleneck,
        )
        self.snapshots.append(snapshot)
        
        # Persist snapshot if persistence is configured
        if self.persistence is not None:
            self.persistence.append_snapshot(snapshot)

        # Track loss variance/errors
        if loss is not None:
            if self._last_loss is not None:
                delta = loss - self._last_loss
                self.variance_errors.append(delta)
            self._last_loss = loss
            
        return snapshot

    def get_summary(self) -> TelemetrySummary:
        if not self.snapshots:
            return TelemetrySummary(
                total_steps=0,
                total_duration_sec=0.0,
                avg_gpu_utilization_pct=0.0,
                peak_gpu_memory_mb=0.0,
                gpu_memory_total_mb=self.gpu_memory_total_mb,
                avg_step_time_ms=0.0,
                avg_dataloader_stall_pct=0.0,
                mixed_precision="fp32",
                gradient_checkpointing=False,
                flash_attention=False,
                num_workers=0,
                pin_memory=False,
                rank=self.rank,
                world_size=self.world_size,
                attention_architecture="MHA",
                sequence_length=2048,
                estimated_kv_cache_mb=0.0,
                ttft_ms=0.0,
                tpot_ms=0.0,
                dominant_latency_bottleneck="COMPUTE_BOUND",
            )

        total_steps = len(self.snapshots)
        duration = self.snapshots[-1].timestamp if self.snapshots else 0.0
        avg_gpu_util = sum(s.gpu_utilization_pct for s in self.snapshots) / total_steps
        peak_mem = max(s.gpu_memory_used_mb for s in self.snapshots)
        avg_step_ms = sum(s.step_time_ms for s in self.snapshots) / total_steps

        # Calculate stall percentage from data loading
        total_loading = sum(s.data_loading_time_ms for s in self.snapshots)
        total_step_time = sum(s.step_time_ms for s in self.snapshots)
        stall_pct = min(100.0, (total_loading / max(total_step_time, 1e-5) * 100.0)) if total_step_time > 0 else 0.0

        last_snap = self.snapshots[-1]
        avg_kv_mb = max(s.estimated_kv_cache_mb for s in self.snapshots)
        avg_ttft = sum(s.ttft_ms for s in self.snapshots) / total_steps
        avg_tpot = sum(s.tpot_ms for s in self.snapshots) / total_steps

        return TelemetrySummary(
            total_steps=total_steps,
            total_duration_sec=duration,
            avg_gpu_utilization_pct=round(avg_gpu_util, 2),
            peak_gpu_memory_mb=round(peak_mem, 2),
            gpu_memory_total_mb=self.gpu_memory_total_mb,
            avg_step_time_ms=round(avg_step_ms, 2),
            avg_dataloader_stall_pct=round(stall_pct, 2),
            mixed_precision=last_snap.mixed_precision,
            gradient_checkpointing=last_snap.gradient_checkpointing,
            flash_attention=last_snap.flash_attention,
            num_workers=last_snap.num_workers,
            pin_memory=last_snap.pin_memory,
            rank=self.rank,
            world_size=self.world_size,
            attention_architecture=last_snap.attention_architecture,
            sequence_length=last_snap.sequence_length,
            estimated_kv_cache_mb=round(avg_kv_mb, 2),
            ttft_ms=round(avg_ttft, 2),
            tpot_ms=round(avg_tpot, 2),
            dominant_latency_bottleneck=last_snap.dominant_latency_bottleneck,
        )

    def calculate_loss_stability_index(self) -> float:
        """
        Computes an empirical loss delta stability index: mean(exp(-|Δloss_t|)) in (0, 1].
        
        Note: This is an uncalibrated empirical heuristic indicating short-term loss variance.
        Values near 1.0 indicate smooth monotonic descent; values below 0.70 indicate
        high step-to-step variance (potential gradient noise or ill-conditioned loss surface).
        """
        if not self.variance_errors:
            return 1.0
            
        errors = np.array(self.variance_errors)
        stability_weights = np.exp(-np.abs(errors))
        return float(np.mean(stability_weights))

    def calculate_euler_inverse_cost(self) -> float:
        """
        Legacy alias for calculate_loss_stability_index().
        Formerly referred to as 'Euler inverse cost' (heuristic metric).
        """
        return self.calculate_loss_stability_index()

    def calculate_fractional_levy_cost(self, alpha: float = 1.5) -> float:
        """
        [EXPERIMENTAL / UNVALIDATED HEURISTIC]
        Computes a heavy-tailed penalty: exp(-|Δloss|^(alpha/2)).
        Provided as an experimental research heuristic; not calibrated against real compute waste.
        """
        if not self.variance_errors:
            return 1.0
        errors = np.array(self.variance_errors)
        penalties = np.power(np.abs(errors) + 1e-7, alpha / 2.0)
        return float(np.mean(np.exp(-penalties)))

    def calculate_hamiltonian_energy_drift(self, friction_beta: float = 0.1) -> float:
        """
        [EXPERIMENTAL / UNVALIDATED HEURISTIC]
        Computes a discrete pseudo-kinetic energy drift heuristic across loss deltas.
        Provided as an experimental research heuristic; not a formal physical symplectic invariant.
        """
        if len(self.variance_errors) < 2:
            return 0.0
        errors = np.array(self.variance_errors)
        kinetics = 0.5 * (errors ** 2)
        drifts = errors[1:] + kinetics[1:] - (1.0 - friction_beta) * kinetics[:-1]
        return float(np.mean(np.maximum(0.0, drifts)))

    def generate_variance_plot(self, output_path: str = "variance_sphere_telemetry.png") -> None:
        """
        Generates a 100% empirical Telemetry Dashboard from measured snapshots:
        1. Measured Step Loss Trajectory & Step Delta (ΔL)
        2. Measured GPU Utilization (%) and Memory Allocation (MB)
        3. Step Latency & DataLoader Stall Breakdown (ms)
        4. Step-to-Step Loss Delta Distribution (empirical histogram)
        """
        if not self.snapshots:
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        steps = [s.step for s in self.snapshots]
        losses = [s.loss for s in self.snapshots if s.loss is not None]
        loss_steps = [s.step for s in self.snapshots if s.loss is not None]
        gpu_utils = [s.gpu_utilization_pct for s in self.snapshots]
        gpu_mems = [s.gpu_memory_used_mb for s in self.snapshots]
        step_times = [s.step_time_ms for s in self.snapshots]
        stall_times = [s.data_loading_time_ms for s in self.snapshots]

        fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("GhostLayer Empirical Training Telemetry Dashboard", fontsize=14, fontweight="bold")

        # Panel 1: Step Loss Trajectory
        if losses:
            axs[0, 0].plot(loss_steps, losses, color="#2563eb", lw=2, label="Step Loss")
            if len(self.variance_errors) > 0 and len(loss_steps) == len(self.variance_errors) + 1:
                axs[0, 0].scatter(loss_steps[1:], self.variance_errors, color="#dc2626", s=15, alpha=0.6, label="Step ΔLoss")
            axs[0, 0].set_title("1. Measured Step Loss Trajectory")
            axs[0, 0].set_xlabel("Step")
            axs[0, 0].set_ylabel("Loss")
            axs[0, 0].grid(True, alpha=0.3)
            axs[0, 0].legend()
        else:
            axs[0, 0].text(0.5, 0.5, "No Loss Telemetry Captured", ha="center", va="center")
            axs[0, 0].set_title("1. Step Loss Trajectory")

        # Panel 2: GPU Utilization & Memory
        ax_gpu = axs[0, 1]
        ax_gpu.plot(steps, gpu_utils, color="#16a34a", lw=2, label="GPU Compute (%)")
        ax_gpu.set_ylabel("GPU Utilization (%)", color="#16a34a")
        ax_gpu.set_ylim(0, 105)
        ax_gpu.grid(True, alpha=0.3)

        ax_mem = ax_gpu.twinx()
        ax_mem.plot(steps, gpu_mems, color="#9333ea", lw=1.5, linestyle="--", label="VRAM (MB)")
        ax_mem.set_ylabel("Memory Used (MB)", color="#9333ea")
        ax_gpu.set_title(f"2. GPU Compute & VRAM Usage (Peak: {max(gpu_mems, default=0):.0f} MB)")
        ax_gpu.set_xlabel("Step")

        # Panel 3: Step Latency Breakdown
        axs[1, 0].plot(steps, step_times, color="#0284c7", lw=1.5, label="Total Step Time (ms)")
        axs[1, 0].plot(steps, stall_times, color="#ea580c", lw=1.5, label="DataLoader Stall (ms)")
        axs[1, 0].set_title("3. Step Latency & DataLoader Stalls")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Time (ms)")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        # Panel 4: Loss Delta Stability Distribution
        if self.variance_errors:
            stability_idx = self.calculate_loss_stability_index()
            axs[1, 1].hist(self.variance_errors, bins=min(20, max(5, len(self.variance_errors))), color="#4f46e5", edgecolor="black", alpha=0.7)
            axs[1, 1].set_title(f"4. Step ΔLoss Distribution (Stability Index: {stability_idx:.3f})")
            axs[1, 1].set_xlabel("Loss Delta (Step_t - Step_{t-1})")
            axs[1, 1].set_ylabel("Frequency")
            axs[1, 1].grid(True, alpha=0.3)
        else:
            axs[1, 1].text(0.5, 0.5, "No Loss Delta Variance Captured", ha="center", va="center")
            axs[1, 1].set_title("4. Loss Delta Stability Distribution")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
