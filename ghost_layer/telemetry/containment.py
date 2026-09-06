"""
Variance Containment Engine

Classifies variance errors as normal fluctuations vs. compute "leaks",
quantifies wasted GPU-hours from regressive steps, and computes the
containment premium.

A "leak" is a training step where the loss increased instead of decreased,
meaning the GPU-hours spent on that step produced negative training progress.
The containment system catches these, quantifies the waste, and feeds the
dollar value into the ROI calculator as a second revenue stream.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class LeakEvent:
    """A single detected variance leak (regressive step)."""
    step: int
    loss_delta: float           # positive = loss went UP (bad)
    step_time_ms: float
    wasted_gpu_hours: float     # GPU-hours burned on this regressive step
    severity: str               # MINOR, MODERATE, SEVERE
    energy_drift: float = 0.0   # Ad-hoc spike severity score (not physical Hamiltonian energy)
    fractional_penalty: float = 0.0 # Superlinear penalty magnitude (|severity_score|^0.75)


@dataclass
class ContainmentReport:
    """Summary of all variance leaks detected during a training run."""
    total_steps_observed: int
    total_leak_events: int
    leak_rate_pct: float                    # % of steps that were leaks
    total_wasted_gpu_hours: float
    total_wasted_cost_usd: float
    containment_premium_usd: float
    containment_premium_rate_pct: float
    avg_leak_magnitude: float               # average |delta| of leak events
    max_leak_magnitude: float               # worst single spike
    severity_breakdown: dict                # {"MINOR": N, "MODERATE": N, "SEVERE": N}
    leak_events: List[LeakEvent] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)


class VarianceContainmentEngine:
    """
    Analyzes variance errors from the TelemetryWatcher and classifies them
    into normal fluctuations vs. compute leaks. Quantifies wasted GPU-hours
    and computes the containment premium.

    The engine uses a rolling-window adaptive threshold: a loss delta is
    classified as a "leak" only if it exceeds a dynamic threshold based on
    the recent error distribution. This prevents flagging normal gradient
    noise as waste.
    """

    # Severity thresholds (multiples of rolling std dev)
    MINOR_THRESHOLD = 1.0       # > 1.0 std devs above mean
    MODERATE_THRESHOLD = 2.0    # > 2.0 std devs
    SEVERE_THRESHOLD = 3.0      # > 3.0 std devs

    def __init__(
        self,
        containment_premium_rate_pct: float = 10.0,
        rolling_window_size: int = 20,
    ):
        self.containment_premium_rate_pct = containment_premium_rate_pct
        self.rolling_window_size = rolling_window_size

    def analyze(
        self,
        variance_errors: List[float],
        step_times_ms: List[float],
        gpu_cost_per_hour: float,
        num_gpus: int = 1,
    ) -> ContainmentReport:
        """
        Analyze variance errors from a completed training run.

        Args:
            variance_errors: List of step-to-step loss deltas from the watcher.
            step_times_ms: List of step durations (ms) corresponding to each error.
            gpu_cost_per_hour: Cluster-wide cost per hour (all GPUs combined).
            num_gpus: Number of GPUs (for GPU-hour calculation).
        """
        if not variance_errors or not step_times_ms:
            return ContainmentReport(
                total_steps_observed=0,
                total_leak_events=0,
                leak_rate_pct=0.0,
                total_wasted_gpu_hours=0.0,
                total_wasted_cost_usd=0.0,
                containment_premium_usd=0.0,
                containment_premium_rate_pct=self.containment_premium_rate_pct,
                avg_leak_magnitude=0.0,
                max_leak_magnitude=0.0,
                severity_breakdown={"MINOR": 0, "MODERATE": 0, "SEVERE": 0},
            )

        errors = np.array(variance_errors)
        times = np.array(step_times_ms[:len(errors)])

        leak_events: List[LeakEvent] = []
        severity_counts = {"MINOR": 0, "MODERATE": 0, "SEVERE": 0}

        for i, (delta, step_ms) in enumerate(zip(errors, times)):
            # Only positive deltas are leaks (loss went UP)
            if delta <= 0:
                continue

            # Compute rolling statistics for adaptive thresholding
            window_start = max(0, i - self.rolling_window_size)
            window = errors[window_start:i + 1]
            abs_window = np.abs(window)
            rolling_mean = float(np.mean(abs_window))
            rolling_std = float(np.std(abs_window)) if len(window) > 1 else float(np.mean(abs_window))

            # Classify severity based on how far above the rolling norm this spike is
            if rolling_std < 1e-8:
                # If std is essentially zero, any positive delta is a leak
                z_score = delta / max(rolling_mean, 1e-8)
            else:
                z_score = (delta - rolling_mean) / rolling_std

            if z_score < self.MINOR_THRESHOLD:
                continue  # Within normal noise band, not a leak

            if z_score >= self.SEVERE_THRESHOLD:
                severity = "SEVERE"
            elif z_score >= self.MODERATE_THRESHOLD:
                severity = "MODERATE"
            else:
                severity = "MINOR"

            severity_counts[severity] += 1

            # Compute wasted GPU-hours for this single regressive step
            wasted_hours = (step_ms / 1000.0 / 3600.0) * num_gpus

            # Spike severity heuristic: penalizes large positive deltas more than
            # small ones, with a momentum term from the previous step. This is an
            # ad-hoc severity score, NOT a physically meaningful Hamiltonian energy.
            # Higher values indicate more severe spikes relative to recent history.
            current_energy = 0.5 * (delta ** 2)
            prev_delta = errors[i - 1] if i > 0 else 0.0
            prev_energy = 0.5 * (prev_delta ** 2)
            severity_score = float(delta + current_energy - 0.9 * prev_energy)
            # Superlinear penalty: exponent 0.75 amplifies large spikes
            frac_pen = float(abs(severity_score + 1e-7) ** 0.75)

            leak_events.append(LeakEvent(
                step=i + 1,  # 1-indexed step (error list is offset by 1 from steps)
                loss_delta=float(delta),
                step_time_ms=float(step_ms),
                wasted_gpu_hours=wasted_hours,
                severity=severity,
                energy_drift=round(severity_score, 4),
                fractional_penalty=round(frac_pen, 4),
            ))

        total_wasted_hours = sum(e.wasted_gpu_hours for e in leak_events)
        total_wasted_cost = total_wasted_hours * gpu_cost_per_hour
        containment_premium = total_wasted_cost * (self.containment_premium_rate_pct / 100.0)

        leak_magnitudes = [e.loss_delta for e in leak_events]

        # Generate containment action recommendations
        containment_actions = self._generate_containment_actions(
            leak_events, severity_counts, errors
        )

        return ContainmentReport(
            total_steps_observed=len(errors),
            total_leak_events=len(leak_events),
            leak_rate_pct=round(len(leak_events) / len(errors) * 100.0, 2) if len(errors) > 0 else 0.0,
            total_wasted_gpu_hours=round(total_wasted_hours, 4),
            total_wasted_cost_usd=round(total_wasted_cost, 2),
            containment_premium_usd=round(containment_premium, 2),
            containment_premium_rate_pct=self.containment_premium_rate_pct,
            avg_leak_magnitude=round(float(np.mean(leak_magnitudes)), 4) if leak_magnitudes else 0.0,
            max_leak_magnitude=round(float(np.max(leak_magnitudes)), 4) if leak_magnitudes else 0.0,
            severity_breakdown=severity_counts,
            leak_events=leak_events,
            containment_actions=containment_actions,
        )

    def _generate_containment_actions(
        self,
        leak_events: List[LeakEvent],
        severity_counts: dict,
        all_errors: np.ndarray,
    ) -> List[str]:
        """Generate specific containment recommendations based on leak patterns."""
        actions = []

        if not leak_events:
            return ["No containment actions needed -- training variance is within normal bounds."]

        total_leaks = len(leak_events)

        # Action 1: Gradient clipping (for severe spikes)
        if severity_counts["SEVERE"] > 0:
            actions.append(
                f"CRITICAL: {severity_counts['SEVERE']} severe loss spikes detected. "
                f"Enable or tighten gradient clipping (torch.nn.utils.clip_grad_norm_, max_norm=1.0) "
                f"to contain explosive gradient updates."
            )

        # Action 2: Learning rate warmup (for early-run instability)
        early_leaks = [e for e in leak_events if e.step < len(all_errors) * 0.2]
        if len(early_leaks) > total_leaks * 0.4:
            actions.append(
                f"WARMUP: {len(early_leaks)}/{total_leaks} leaks occurred in the first 20% of training. "
                f"Extend learning rate warmup period (e.g., warmup_steps *= 2) to stabilize early convergence."
            )

        # Action 3: Batch size adjustment (for sustained moderate variance)
        if severity_counts["MODERATE"] > total_leaks * 0.3:
            actions.append(
                f"BATCH: {severity_counts['MODERATE']} moderate variance events suggest noisy gradient estimates. "
                f"Increase effective batch size via gradient accumulation (accumulation_steps *= 2) "
                f"to reduce per-step variance."
            )

        # Action 4: Learning rate reduction (for late-run instability)
        late_leaks = [e for e in leak_events if e.step > len(all_errors) * 0.7]
        if len(late_leaks) > total_leaks * 0.4:
            actions.append(
                f"LR DECAY: {len(late_leaks)}/{total_leaks} leaks occurred in the final 30% of training. "
                f"The learning rate may be too high for the convergence phase. "
                f"Apply cosine annealing or reduce LR by 50%."
            )

        # Action 5: General containment summary
        leak_pct = len(leak_events) / len(all_errors) * 100.0 if len(all_errors) > 0 else 0.0
        if leak_pct > 15.0:
            actions.append(
                f"WARNING: {leak_pct:.1f}% of all steps are leaking compute. "
                f"Consider pausing the run to diagnose fundamental training instability "
                f"before continuing to burn GPU-hours."
            )

        return actions
