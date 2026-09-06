"""
GhostLayer Real Telemetry Collector: Direct in-process physical instrumentation
for PyTorch training workloads.

Replaces caller-supplied numbers and simulated constants with:
1. Asynchronous CUDA Event timing (torch.cuda.Event) for compute latency without CPU sync stalls.
2. Direct PyTorch CUDA memory allocator probing (torch.cuda.memory_stats, fragmentation ratio).
3. Physical GPU interrogation via NVML (SM utilization %, board power, clocks, thermal flags).
4. DataLoader queue starvation profiling via GhostDataLoaderWrapper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch


@dataclass
class HardwareProbeResult:
    """Direct hardware telemetry snapshot captured from NVML and PyTorch runtime."""
    is_cuda_available: bool
    is_nvml_available: bool
    device_name: str
    gpu_utilization_pct: Optional[float]  # Physical SM utilization from NVML (None if unmeasured)
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_memory_total_mb: float
    fragmentation_ratio: float  # 1.0 - (allocated / reserved)
    power_watts: Optional[float] = None
    sm_clock_mhz: Optional[int] = None
    is_thermal_throttled: Optional[bool] = None
    measurement_source: str = "PHYSICAL_PROBE"  # PHYSICAL_PROBE or CPU_FALLBACK_UNMEASURED
    timestamp: float = field(default_factory=time.time)


class CudaEventTimer:
    """
    High-precision GPU execution timer using CUDA Events.
    Avoids host-device synchronization overhead during training iteration.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.is_cuda = torch.cuda.is_available()
        self.device = device
        self._start_event: Optional[torch.cuda.Event] = None
        self._stop_event: Optional[torch.cuda.Event] = None
        self._host_start: float = 0.0

    def start(self) -> None:
        """Starts kernel execution measurement."""
        if self.is_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._stop_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._host_start = time.perf_counter()

    def stop(self) -> float:
        """
        Stops measurement and returns elapsed time in milliseconds.
        Synchronizes only the recorded event, not the full CUDA stream.
        """
        if self.is_cuda and self._start_event is not None and self._stop_event is not None:
            self._stop_event.record()
            self._stop_event.synchronize()
            return float(self._start_event.elapsed_time(self._stop_event))
        else:
            return (time.perf_counter() - self._host_start) * 1000.0


class HardwareProbe:
    """
    Probes physical GPU metrics via PyTorch CUDA APIs and NVML.
    If running without physical GPU or NVML, gracefully reports unmeasured status.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.is_cuda = torch.cuda.is_available()
        self._nvml_initialized = False
        self._nvml_handle = None

        if self.is_cuda:
            self._init_nvml()

    def _init_nvml(self) -> None:
        """Attempts to initialize NVML bindings via pynvml if present."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml_initialized = True
        except Exception:
            self._nvml_initialized = False
            self._nvml_handle = None

    def probe(self) -> HardwareProbeResult:
        """Captures active hardware and memory metrics directly from device."""
        if not self.is_cuda:
            return HardwareProbeResult(
                is_cuda_available=False,
                is_nvml_available=False,
                device_name="CPU_FALLBACK",
                gpu_utilization_pct=None,
                gpu_memory_allocated_mb=0.0,
                gpu_memory_reserved_mb=0.0,
                gpu_memory_total_mb=0.0,
                fragmentation_ratio=0.0,
                measurement_source="CPU_FALLBACK_UNMEASURED",
            )

        # PyTorch allocator metrics
        dev = self.device_index
        allocated_bytes = torch.cuda.memory_allocated(dev)
        reserved_bytes = torch.cuda.memory_reserved(dev)
        total_bytes = torch.cuda.get_device_properties(dev).total_memory
        device_name = torch.cuda.get_device_name(dev)

        allocated_mb = allocated_bytes / (1024.0 * 1024.0)
        reserved_mb = reserved_bytes / (1024.0 * 1024.0)
        total_mb = total_bytes / (1024.0 * 1024.0)

        # Allocator fragmentation ratio
        frag_ratio = 1.0 - (allocated_bytes / (reserved_bytes + 1e-8)) if reserved_bytes > 0 else 0.0
        frag_ratio = max(0.0, min(1.0, frag_ratio))

        # NVML physical readings
        gpu_util_pct: Optional[float] = None
        power_watts: Optional[float] = None
        clock_mhz: Optional[int] = None
        throttled: Optional[bool] = None

        if self._nvml_initialized and self._nvml_handle is not None:
            try:
                import pynvml
                rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util_pct = float(rates.gpu)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                power_watts = power_mw / 1000.0
                clock_mhz = int(pynvml.nvmlDeviceGetClockInfo(self._nvml_handle, pynvml.NVML_CLOCK_SM))
                clocks_throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._nvml_handle)
                throttled = bool(clocks_throttle != 0)
            except Exception:
                gpu_util_pct = None

        return HardwareProbeResult(
            is_cuda_available=True,
            is_nvml_available=self._nvml_initialized,
            device_name=device_name,
            gpu_utilization_pct=gpu_util_pct,
            gpu_memory_allocated_mb=round(allocated_mb, 2),
            gpu_memory_reserved_mb=round(reserved_mb, 2),
            gpu_memory_total_mb=round(total_mb, 2),
            fragmentation_ratio=round(frag_ratio, 4),
            power_watts=round(power_watts, 2) if power_watts is not None else None,
            sm_clock_mhz=clock_mhz,
            is_thermal_throttled=throttled,
            measurement_source="PHYSICAL_PROBE",
        )


class GhostDataLoaderWrapper:
    """
    Wraps PyTorch DataLoader to accurately instrument inter-batch worker wait times.
    Directly measures data starvation without requiring caller-supplied stall estimates.
    """

    def __init__(self, dataloader: Any):
        self.dataloader = dataloader
        self.last_queue_wait_ms: float = 0.0
        self.total_queue_wait_ms: float = 0.0
        self.batch_count: int = 0
        self._last_yield_time: Optional[float] = None

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.dataloader)
        self._last_yield_time = time.perf_counter()

        while True:
            t_req = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            t_avail = time.perf_counter()

            wait_ms = (t_avail - t_req) * 1000.0
            self.last_queue_wait_ms = wait_ms
            self.total_queue_wait_ms += wait_ms
            self.batch_count += 1
            self._last_yield_time = time.perf_counter()

            yield batch

    def __len__(self) -> int:
        if hasattr(self.dataloader, "__len__"):
            return len(self.dataloader)
        return 0

    @property
    def avg_queue_wait_ms(self) -> float:
        """Returns average queue wait latency per batch in milliseconds."""
        return self.total_queue_wait_ms / max(1, self.batch_count)


class RealTelemetryCollector:
    """
    Coordinates hardware probing, CUDA timing, and DataLoader instrumentation
    into authentic physical telemetry snapshots.
    """

    def __init__(self, device_index: int = 0):
        self.probe = HardwareProbe(device_index=device_index)
        self.timer = CudaEventTimer()

    def start_step(self) -> None:
        """Begins timing of compute step."""
        self.timer.start()

    def end_step(
        self,
        step: int,
        loss: Optional[float] = None,
        dataloader_wrapper: Optional[GhostDataLoaderWrapper] = None,
    ) -> Dict[str, Any]:
        """
        Ends compute timing, samples physical hardware state, and compiles metrics.
        """
        compute_ms = self.timer.stop()
        hw = self.probe.probe()

        data_wait_ms = dataloader_wrapper.last_queue_wait_ms if dataloader_wrapper is not None else 0.0
        total_step_ms = compute_ms + data_wait_ms

        stall_pct = (data_wait_ms / (total_step_ms + 1e-8)) * 100.0 if total_step_ms > 0 else 0.0

        return {
            "step": step,
            "step_time_ms": round(total_step_ms, 3),
            "compute_time_ms": round(compute_ms, 3),
            "data_loading_time_ms": round(data_wait_ms, 3),
            "dataloader_stall_pct": round(min(100.0, max(0.0, stall_pct)), 2),
            "loss": loss,
            "gpu_utilization_pct": hw.gpu_utilization_pct,
            "gpu_memory_allocated_mb": hw.gpu_memory_allocated_mb,
            "gpu_memory_reserved_mb": hw.gpu_memory_reserved_mb,
            "gpu_memory_total_mb": hw.gpu_memory_total_mb,
            "allocator_fragmentation_ratio": hw.fragmentation_ratio,
            "power_watts": hw.power_watts,
            "is_thermal_throttled": hw.is_thermal_throttled,
            "measurement_source": hw.measurement_source,
            "timestamp": time.time(),
        }
