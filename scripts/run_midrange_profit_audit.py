"""
GhostLayer Mid-Range Fleet Audit & Profit Maximizer Runner
Executes cryptographic 5-W audit logging and profit calculus across:
  - Tier 1A: 8x A100 80GB SXM4 ($32/hr node)
  - Tier 1B: 8x H100 80GB SXM5 ($36/hr node)
  - Tier 2A: 32x H100 SXM5 InfiniBand ($112/hr cluster)
  - Tier 2B: 64x H100 SXM5 Quantum-2 ($224/hr cluster)
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ghost_layer.roi.midrange_optimizer import MidRangeFleetOptimizer


def main():
    print("=" * 110)
    print("  GHOSTLAYER MID-RANGE FLEET AUDIT & PROFIT MAXIMIZATION ENGINE")
    print("  Powered by Cryptographic 5-W Audit Trail & Immudb-Style Hash Chaining")
    print("=" * 110)

    optimizer = MidRangeFleetOptimizer(
        performance_fee_pct=25.0,
        containment_premium_pct=10.0,
        pre_flight_audit_fee=2500.0,
        team_saas_monthly_fee=499.0
    )

    tiers = [
        {
            "tier_name": "Tier 1A: Single Cloud Node (8x A100 80GB SXM4)",
            "gpus": 8, "cost_hr": 4.00, "base_ms": 3200.0, "stall_pct": 14.5,
            "vram_pct": 75.0, "snr": 0.30, "steps": 120000,
            "model": "Llama-3-70B (LoRA / QLoRA / SFT)"
        },
        {
            "tier_name": "Tier 1B: High-End Pod (8x H100 80GB SXM5)",
            "gpus": 8, "cost_hr": 4.50, "base_ms": 2800.0, "stall_pct": 11.0,
            "vram_pct": 68.0, "snr": 0.38, "steps": 150000,
            "model": "Qwen-2.5-32B / DeepSeek-R1-Distill-32B"
        },
        {
            "tier_name": "Tier 2A: Small Cluster (32x H100 InfiniBand)",
            "gpus": 32, "cost_hr": 3.50, "base_ms": 3500.0, "stall_pct": 10.5,
            "vram_pct": 72.0, "snr": 0.42, "steps": 80000,
            "model": "Mistral-Small-24B / CodeLlama-34B"
        },
        {
            "tier_name": "Tier 2B: Mid-Market Fleet (64x H100 Quantum-2)",
            "gpus": 64, "cost_hr": 3.50, "base_ms": 4000.0, "stall_pct": 9.8,
            "vram_pct": 78.0, "snr": 0.45, "steps": 100000,
            "model": "Llama-3.3-70B Full Pre-Training Sweep"
        },
    ]

    total_gross_savings = 0.0
    total_net_client_savings = 0.0
    total_ghost_revenue = 0.0

    print("\n[MID-RANGE FLEET AUDIT & PROFIT PROJECTIONS]")
    for t in tiers:
        rep = optimizer.audit_and_optimize_tier(
            tier_name=t["tier_name"],
            num_gpus=t["gpus"],
            gpu_hourly_rate=t["cost_hr"],
            baseline_step_ms=t["base_ms"],
            dataloader_stall_pct=t["stall_pct"],
            activation_vram_pct=t["vram_pct"],
            gradient_noise_snr=t["snr"],
            total_steps=t["steps"]
        )

        total_gross_savings += rep.gross_savings_usd
        total_net_client_savings += rep.net_client_savings_usd
        total_ghost_revenue += rep.total_ghost_revenue_usd

        print(f"\n• {rep.tier_name} | Workload: {t['model']}")
        print(f"  Cluster Burn: {rep.num_gpus}x GPUs @ ${rep.gpu_hourly_rate:.2f}/hr (${rep.num_gpus * rep.gpu_hourly_rate:,.2f}/hr)")
        print(f"  Latency Tuning: {rep.baseline_step_ms:.1f}ms -> {rep.optimized_step_ms:.1f}ms (-{rep.step_latency_reduction_pct}%)")
        print(f"    - DataLoader I/O Fix:      -{rep.optimization_breakdown['dataloader_io_ms']:.1f}ms (Pinned Buffer & Prefetch)")
        print(f"    - Chunked VRAM Headroom:   -{rep.optimization_breakdown['chunked_vram_ms']:.1f}ms (ChunkedCrossEntropy)")
        print(f"    - Muon / Symplectic Damp:  -{rep.optimization_breakdown['muon_damping_ms']:.1f}ms (Newton-Schulz Orthogonalization)")
        print(f"  Financials (Per Run):")
        print(f"    - Baseline Compute Spend:  ${rep.baseline_cost_usd:>10,.2f}")
        print(f"    - Optimized Spend:         ${rep.optimized_cost_usd:>10,.2f}")
        print(f"    - Net Client Savings:      ${rep.net_client_savings_usd:>10,.2f}")
        print(f"  GhostLayer Captured Revenue (Per Run):")
        print(f"    - 25% Optimization Share:  ${rep.ghost_optimization_fee_usd:>8,.2f}")
        print(f"    - 10% Containment Premium: ${rep.ghost_containment_premium_usd:>8,.2f}")
        print(f"    - Pre-Flight Audit Fee:    ${rep.pre_flight_audit_fee_usd:>8,.2f}")
        print(f"    - TOTAL Ghost Revenue:     ${rep.total_ghost_revenue_usd:>8,.2f}")
        print(f"  Cryptographic Hash Chain:  {'[OK] VERIFIED (0-Tamper)' if rep.audit_chain_verified else '[FAIL] TAMPERED'}")

    # Verify overall cryptographic audit chain
    is_valid, error = optimizer.audit_engine.verify_chain_integrity()
    print("\n" + "=" * 110)
    print(f"  AGGREGATE MID-RANGE FLEET PORTFOLIO TOTALS (4 FLEETS, 1 RUN EACH)")
    print(f"  * Total Gross Compute Saved:   ${total_gross_savings:>12,.2f}")
    print(f"  * Net Client Compute Savings:  ${total_net_client_savings:>12,.2f}")
    print(f"  * Total GhostLayer Revenue:    ${total_ghost_revenue:>12,.2f}")
    print(f"  * Team SaaS ARR Potential:     ${optimizer.team_saas_monthly_fee * 12 * 4:>12,.2f} ($499/mo x 4 fleets)")
    print(f"  * Audit Chain Hash Status:     {'[OK] CRYPTOGRAPHICALLY SECURE & IMMUTABLE' if is_valid else '[FAIL] ERROR: ' + str(error)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
