"""
Automated 2^k Factorial Experimentation & Systems Interaction Analysis.

Evaluates all 2^k combinatorial combinations of optimization factors to compute:
1. Individual factor main effects.
2. Two-way interaction effects (synergy vs interference).
3. Super-additive vs sub-additive multi-optimization coupling.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class FactorialTreatment:
    treatment_id: str
    active_factors: List[str]
    mean_latency_ms: float
    throughput_tok_per_sec: float
    speedup_vs_baseline_pct: float


@dataclass
class FactorialAnalysisReport:
    total_treatments: int
    factors: List[str]
    treatments: List[FactorialTreatment]
    main_effects_pct: Dict[str, float]
    two_way_interactions_pct: Dict[str, float]
    synergistic_pairs: List[str]
    interfering_pairs: List[str]
    optimal_treatment_id: str
    optimal_speedup_pct: float


class FactorialExperimentRunner:
    """
    Executes 2^k factorial design of experiments across k optimization factors.
    """

    @staticmethod
    def analyze_results(
        factors: List[str],
        latency_map: Dict[Tuple[str, ...], float],
    ) -> FactorialAnalysisReport:
        """
        Analyzes full factorial latency results.
        latency_map maps sorted tuple of active factors (e.g. ('AMP', 'WORKERS')) to mean latency (ms).
        Empty tuple () represents the baseline.
        """
        baseline_latency = latency_map.get((), 100.0)
        k = len(factors)
        total_treatments = 2 ** k

        treatments: List[FactorialTreatment] = []
        for r in range(k + 1):
            for combo in itertools.combinations(factors, r):
                lat = latency_map.get(combo, baseline_latency)
                speedup = ((baseline_latency - lat) / baseline_latency * 100.0) if baseline_latency > 0 else 0.0
                t_id = "+".join(combo) if combo else "BASELINE"
                treatments.append(
                    FactorialTreatment(
                        treatment_id=t_id,
                        active_factors=list(combo),
                        mean_latency_ms=round(lat, 2),
                        throughput_tok_per_sec=round(1000.0 / lat * 1000.0, 1) if lat > 0 else 0.0,
                        speedup_vs_baseline_pct=round(speedup, 2),
                    )
                )

        # Compute Main Effects: mean difference in speedup between (factor present) and (factor absent)
        main_effects: Dict[str, float] = {}
        for f in factors:
            with_f = [t.speedup_vs_baseline_pct for t in treatments if f in t.active_factors]
            without_f = [t.speedup_vs_baseline_pct for t in treatments if f not in t.active_factors]
            avg_with = sum(with_f) / len(with_f) if with_f else 0.0
            avg_without = sum(without_f) / len(without_f) if without_f else 0.0
            main_effects[f] = round(avg_with - avg_without, 2)

        # Compute 2-way Interaction Effects
        two_way: Dict[str, float] = {}
        synergies: List[str] = []
        interferences: List[str] = []

        for f1, f2 in itertools.combinations(factors, 2):
            pair_key = f"{f1} x {f2}"
            # Interaction: (y_{11} - y_{10}) - (y_{01} - y_{00})
            both = latency_map.get(tuple(sorted([f1, f2])), baseline_latency)
            only_f1 = latency_map.get(tuple(sorted([f1])), baseline_latency)
            only_f2 = latency_map.get(tuple(sorted([f2])), baseline_latency)
            base = baseline_latency

            # Speedups
            s_both = (base - both) / base * 100.0
            s_f1 = (base - only_f1) / base * 100.0
            s_f2 = (base - only_f2) / base * 100.0

            # Super-additive delta: actual both vs expected linear addition (s_f1 + s_f2)
            interaction = round(s_both - (s_f1 + s_f2), 2)
            two_way[pair_key] = interaction

            if interaction > 2.0:
                synergies.append(f"{pair_key} (Super-additive: +{interaction:.1f}%)")
            elif interaction < -2.0:
                interferences.append(f"{pair_key} (Sub-additive interference: {interaction:.1f}%)")

        optimal_treatment = max(treatments, key=lambda t: t.speedup_vs_baseline_pct)

        return FactorialAnalysisReport(
            total_treatments=total_treatments,
            factors=factors,
            treatments=treatments,
            main_effects_pct=main_effects,
            two_way_interactions_pct=two_way,
            synergistic_pairs=synergies,
            interfering_pairs=interferences,
            optimal_treatment_id=optimal_treatment.treatment_id,
            optimal_speedup_pct=optimal_treatment.speedup_vs_baseline_pct,
        )
