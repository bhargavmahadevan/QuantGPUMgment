"""
Hardware Contamination & Interference Detection Engine.

Detects background process contention, thermal throttling, and frequency degradation
to protect benchmark integrity and prevent contaminated trials from updating the Knowledge Base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContaminationReport:
    is_contaminated: bool
    status: str  # "VALID" or "CONTAMINATED"
    reasons: List[str] = field(default_factory=list)
    thermal_throttling_detected: bool = False
    frequency_drop_detected: bool = False
    background_contention_detected: bool = False
    measured_clock_drop_pct: float = 0.0
    unexpected_vram_delta_mb: float = 0.0


class ContaminationDetector:
    """
    Monitors hardware probes for environmental noise, contention, and thermal events.
    """

    def __init__(
        self,
        max_allowed_clock_drop_pct: float = 15.0,
        max_allowed_background_vram_mb: float = 512.0,
    ):
        self.max_clock_drop_pct = max_allowed_clock_drop_pct
        self.max_bg_vram_mb = max_allowed_background_vram_mb

    def evaluate_trial(
        self,
        is_thermal_throttled: bool = False,
        baseline_clock_mhz: Optional[int] = None,
        candidate_clock_mhz: Optional[int] = None,
        external_vram_consumed_mb: float = 0.0,
    ) -> ContaminationReport:
        reasons: List[str] = []
        thermal_flag = False
        clock_flag = False
        bg_flag = False
        drop_pct = 0.0

        # Check 1: Thermal throttling
        if is_thermal_throttled:
            thermal_flag = True
            reasons.append("GPU thermal throttling active during trial.")

        # Check 2: SM Clock drop
        if baseline_clock_mhz and candidate_clock_mhz and baseline_clock_mhz > 0:
            if candidate_clock_mhz < baseline_clock_mhz:
                drop_pct = ((baseline_clock_mhz - candidate_clock_mhz) / baseline_clock_mhz) * 100.0
                if drop_pct > self.max_clock_drop_pct:
                    clock_flag = True
                    reasons.append(f"SM clock frequency degraded by {drop_pct:.1f}% (> {self.max_clock_drop_pct}% limit).")

        # Check 3: External process VRAM contention
        if external_vram_consumed_mb > self.max_bg_vram_mb:
            bg_flag = True
            reasons.append(f"Background process contention detected: {external_vram_consumed_mb:.1f} MB external VRAM allocated.")

        is_contam = thermal_flag or clock_flag or bg_flag
        status = "CONTAMINATED" if is_contam else "VALID"

        return ContaminationReport(
            is_contaminated=is_contam,
            status=status,
            reasons=reasons,
            thermal_throttling_detected=thermal_flag,
            frequency_drop_detected=clock_flag,
            background_contention_detected=bg_flag,
            measured_clock_drop_pct=round(drop_pct, 2),
            unexpected_vram_delta_mb=round(external_vram_consumed_mb, 2),
        )
