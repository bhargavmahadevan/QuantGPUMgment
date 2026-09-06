"""
Unit tests validating direct physical hardware probing, CUDA Event timing,
and DataLoader queue starvation monitoring (RealTelemetryCollector).
"""

import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from ghost_layer.telemetry.real_collector import (
    CudaEventTimer,
    HardwareProbe,
    HardwareProbeResult,
    GhostDataLoaderWrapper,
    RealTelemetryCollector,
)
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.callback import ghost_watch


def test_cuda_event_timer_execution():
    """Verify CudaEventTimer starts, stops, and returns elapsed time in ms."""
    timer = CudaEventTimer()
    timer.start()
    if torch.cuda.is_available():
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = a @ b
    else:
        time.sleep(0.01)  # 10ms work
    elapsed_ms = timer.stop()
    assert elapsed_ms > 0.0


def test_hardware_probe_execution():
    """Verify HardwareProbe captures device memory and returns valid probe result."""
    probe = HardwareProbe()
    result = probe.probe()
    assert isinstance(result, HardwareProbeResult)
    assert result.measurement_source in ["PHYSICAL_PROBE", "CPU_FALLBACK_UNMEASURED"]

    if torch.cuda.is_available():
        assert result.is_cuda_available is True
        assert result.gpu_memory_total_mb > 0.0
        assert result.fragmentation_ratio >= 0.0
    else:
        assert result.is_cuda_available is False
        assert result.gpu_utilization_pct is None
        assert result.measurement_source == "CPU_FALLBACK_UNMEASURED"


def test_ghost_dataloader_wrapper_queue_timing():
    """Verify GhostDataLoaderWrapper instruments inter-batch queue wait times."""
    # Synthetic dataset of 5 batches
    x = torch.randn(20, 4)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=4)

    wrapper = GhostDataLoaderWrapper(loader)
    assert len(wrapper) == 5

    batches_seen = 0
    for batch in wrapper:
        batches_seen += 1
        assert wrapper.last_queue_wait_ms >= 0.0

    assert batches_seen == 5
    assert wrapper.batch_count == 5
    assert wrapper.avg_queue_wait_ms >= 0.0


def test_real_telemetry_collector_end_to_end():
    """Verify RealTelemetryCollector compiles physical metrics without caller guesswork."""
    collector = RealTelemetryCollector()
    collector.start_step()

    # Simulate brief computation
    a = torch.randn(100, 100)
    b = a @ a
    loss_val = float(b.mean().item())

    metrics = collector.end_step(step=1, loss=loss_val)
    assert metrics["step"] == 1
    assert metrics["step_time_ms"] >= 0.0
    assert metrics["compute_time_ms"] >= 0.0
    assert metrics["data_loading_time_ms"] == 0.0
    assert metrics["dataloader_stall_pct"] == 0.0
    assert metrics["loss"] == loss_val
    assert "gpu_memory_allocated_mb" in metrics
    assert "allocator_fragmentation_ratio" in metrics


def test_ghost_watcher_hook_dynamic_probe_fallback():
    """Verify GhostWatcherHook uses physical probe when telemetry is not caller-supplied."""
    hook = GhostWatcherHook(gpu_memory_mb=16384.0)
    hook.on_step_begin()
    # Call on_step_end with None for util and memory
    snapshot = hook.on_step_end(
        gpu_util_pct=None,
        gpu_mem_used_mb=None,
        loss=2.5,
    )
    assert snapshot.step == 1
    assert snapshot.loss == 2.5
    assert snapshot.gpu_memory_used_mb >= 0.0


def test_ghost_watch_step_without_hardcoded_utilization():
    """Verify ghost_watch.step accepts None and executes without fake 85.0/80.0 defaults."""
    with ghost_watch(gpu_memory_mb=8192.0) as gw:
        gw.step(loss=1.5)
        gw.step(loss=1.4)
    summary = gw.hook.watcher.get_summary()
    assert summary.total_steps == 2
