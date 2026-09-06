"""
VRAM Headroom & Safe Batch Scaling Multiplier for GhostLayer.

Calculates the maximum feasible batch size scaling multiplier without crossing
the CUDA Out-of-Memory (OOM) fault boundary:

    Headroom_VRAM = VRAM_Total - VRAM_Peak
    Buffer_OOM   = 0.15 * VRAM_Total  (15% safety reserve)
    Safe Multiplier = min(2.0, 1.0 + floor((Headroom_VRAM - Buffer_OOM) / VRAM_ActiveBatch) * 0.25)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math
import torch


@dataclass
class VRAMScalingReport:
    """Detailed report on VRAM headroom and safe batch size scaling."""
    vram_total_mb: float
    vram_peak_mb: float
    vram_headroom_mb: float
    oom_buffer_mb: float
    vram_active_batch_mb: float
    current_batch_size: int
    safe_multiplier: float
    recommended_batch_size: int
    is_scale_up_safe: bool
    breakdown: Dict[str, Any]


def calculate_vram_headroom_multiplier(
    vram_total_mb: float,
    vram_peak_mb: float,
    current_batch_size: int = 1,
    vram_static_mb: Optional[float] = None,
    oom_buffer_ratio: float = 0.15,
    max_multiplier: float = 2.0,
) -> VRAMScalingReport:
    """
    Computes the exact VRAM headroom and safe batch scaling multiplier (Section B).

    Parameters:
        vram_total_mb: Total device VRAM in MB.
        vram_peak_mb: Peak allocated + reserved memory during step execution in MB.
        current_batch_size: Active micro-batch size.
        vram_static_mb: Static model weights + optimizer memory in MB (defaults to 0.5 * peak if omitted).
        oom_buffer_ratio: Fraction of total VRAM reserved for CUDA runtime & transient spike headroom (default 0.15).
        max_multiplier: Cap on batch size scaling in a single step (default 2.0x).
    """
    vram_total = max(1.0, float(vram_total_mb))
    vram_peak = max(0.0, min(vram_total, float(vram_peak_mb)))
    b_curr = max(1, int(current_batch_size))
    
    # 1. Total headroom above peak
    headroom = max(0.0, vram_total - vram_peak)
    
    # 2. OOM safety reserve buffer (15% by standard)
    buffer_oom = oom_buffer_ratio * vram_total
    
    # 3. Active per-batch activation memory
    if vram_static_mb is not None:
        vram_static = max(0.0, min(vram_peak, float(vram_static_mb)))
        vram_active_batch = max(1.0, vram_peak - vram_static)
    else:
        # Default heuristic: activations comprise ~50% of peak memory in standard training
        vram_active_batch = max(1.0, vram_peak * 0.50)
    
    available_for_scaling = headroom - buffer_oom
    
    if available_for_scaling > 0 and vram_active_batch > 0:
        # Compute discrete step increments of 0.25x
        raw_scale_steps = math.floor(available_for_scaling / vram_active_batch)
        increment = max(0.0, raw_scale_steps * 0.25)
        safe_multiplier = min(max_multiplier, 1.0 + increment)
        recommended_batch = max(b_curr, int(round(b_curr * safe_multiplier)))
        is_safe = (safe_multiplier > 1.0)
    else:
        safe_multiplier = 1.0
        recommended_batch = b_curr
        is_safe = False

    return VRAMScalingReport(
        vram_total_mb=round(vram_total, 2),
        vram_peak_mb=round(vram_peak, 2),
        vram_headroom_mb=round(headroom, 2),
        oom_buffer_mb=round(buffer_oom, 2),
        vram_active_batch_mb=round(vram_active_batch, 2),
        current_batch_size=b_curr,
        safe_multiplier=round(safe_multiplier, 3),
        recommended_batch_size=recommended_batch,
        is_scale_up_safe=is_safe,
        breakdown={
            "headroom_pct": round((headroom / vram_total) * 100.0, 1),
            "oom_buffer_pct": round(oom_buffer_ratio * 100.0, 1),
            "usable_scaling_headroom_mb": round(max(0.0, available_for_scaling), 2),
        },
    )
