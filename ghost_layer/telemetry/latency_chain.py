import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LatencyChainBreakdown:
    """
    Decomposes step latency across the end-to-end execution chain to pinpoint
    whether performance is compute-bound, memory-bandwidth-bound, or stalled on I/O / communication.
    """
    ttft_ms: float = 0.0                      # Time-To-First-Token / Prefill latency
    tpot_ms: float = 0.0                      # Time-Per-Output-Token / Decode step latency
    hbm_fetch_time_ms: float = 0.0            # Estimated HBM memory bus read/write time
    compute_time_ms: float = 0.0              # Tensor Core / ALU arithmetic execution time
    dataloader_stall_ms: float = 0.0          # Host CPU / DataLoader I/O stall
    communication_stall_ms: float = 0.0       # Distributed collective all-reduce / barrier stall
    memory_allocation_stall_ms: float = 0.0   # VRAM page allocation & cache defragmentation stall
    dominant_bottleneck: str = "COMPUTE_BOUND" # COMPUTE_BOUND, MEMORY_BANDWIDTH_BOUND, IO_BOUND, COMMUNICATION_BOUND

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_ms": round(self.ttft_ms, 2),
            "tpot_ms": round(self.tpot_ms, 2),
            "hbm_fetch_time_ms": round(self.hbm_fetch_time_ms, 2),
            "compute_time_ms": round(self.compute_time_ms, 2),
            "dataloader_stall_ms": round(self.dataloader_stall_ms, 2),
            "communication_stall_ms": round(self.communication_stall_ms, 2),
            "memory_allocation_stall_ms": round(self.memory_allocation_stall_ms, 2),
            "dominant_bottleneck": self.dominant_bottleneck,
        }


def profile_latency_chain(
    step_time_ms: float,
    dataloader_time_ms: float = 0.0,
    gpu_memory_used_mb: float = 0.0,
    gpu_memory_total_mb: float = 16384.0,
    is_generation: bool = False,
    estimated_kv_cache_mb: float = 0.0,
    hardware_bandwidth_gb_s: float = 900.0,  # e.g. 900 GB/s on A100 SXM, 288 GB/s on A2000
    communication_time_ms: float = 0.0,
) -> LatencyChainBreakdown:
    """
    Analyzes step time telemetry and memory traffic to diagnose the primary bottleneck
    in the latency chain.
    """
    safe_step_time = max(0.001, step_time_ms)
    safe_dl_time = max(0.0, min(dataloader_time_ms, safe_step_time))
    safe_comm_time = max(0.0, min(communication_time_ms, safe_step_time))

    # Memory allocation stall heuristic: when VRAM exceeds 92%, page allocation latency rises
    mem_usage_pct = (gpu_memory_used_mb / max(1.0, gpu_memory_total_mb)) * 100.0
    mem_alloc_stall_ms = 0.0
    if mem_usage_pct > 92.0:
        mem_alloc_stall_ms = round(safe_step_time * min(0.30, (mem_usage_pct - 92.0) * 0.03), 2)

    # Calculate theoretical HBM memory transfer time for reading active weights + KV cache
    # In generation (decode), reading KV cache from HBM is the dominant physical step
    if is_generation and estimated_kv_cache_mb > 0:
        # KV bytes read per step (in MB) converted to ms at given bandwidth (GB/s = MB/ms)
        hbm_fetch_time_ms = min(safe_step_time, (estimated_kv_cache_mb / max(1.0, hardware_bandwidth_gb_s)))
    else:
        # Standard forward/backward pass memory traffic estimation (~25% of step time typical)
        hbm_fetch_time_ms = min(safe_step_time * 0.40, (gpu_memory_used_mb * 0.05 / max(1.0, hardware_bandwidth_gb_s)))

    # Compute execution time is remaining non-stall time
    non_compute = safe_dl_time + safe_comm_time + mem_alloc_stall_ms
    compute_time_ms = max(0.0, safe_step_time - non_compute)

    # Determine dominant bottleneck
    if safe_dl_time / safe_step_time > 0.25:
        dominant = "IO_BOUND"
    elif safe_comm_time / safe_step_time > 0.30:
        dominant = "COMMUNICATION_BOUND"
    elif is_generation and (hbm_fetch_time_ms / safe_step_time > 0.40 or estimated_kv_cache_mb > 2048.0):
        dominant = "MEMORY_BANDWIDTH_BOUND"
    elif mem_alloc_stall_ms / safe_step_time > 0.20:
        dominant = "MEMORY_ALLOCATION_STALL"
    else:
        dominant = "COMPUTE_BOUND"

    ttft = safe_step_time if not is_generation else safe_step_time * 2.5
    tpot = safe_step_time if is_generation else safe_step_time * 0.15

    return LatencyChainBreakdown(
        ttft_ms=round(ttft, 2),
        tpot_ms=round(tpot, 2),
        hbm_fetch_time_ms=round(hbm_fetch_time_ms, 2),
        compute_time_ms=round(compute_time_ms, 2),
        dataloader_stall_ms=round(safe_dl_time, 2),
        communication_stall_ms=round(safe_comm_time, 2),
        memory_allocation_stall_ms=round(mem_alloc_stall_ms, 2),
        dominant_bottleneck=dominant,
    )
