# Honest Baseline Changelog

This file documents every correction made to align this repository's claims with what has actually
been measured or built, so anyone reviewing this project (technical due diligence, a pilot client,
an investor) can see exactly what changed and why.

## Fabricated data removed

- **PHYSICAL_GPU_EVIDENCE_AUDIT.md & real_multi_run_llm_report.md**: Reconciled the 5-trial physical A2000 benchmark under canonical identifier `EXP-A2000-LLM-001`. The throughput improvement is physically verified (+58.41% mean speedup, range +58.29% to +58.56%), but auto-apply was blocked on all 5 trials (`Safe Status: False`, `RECOMMENDATION_ONLY`) because the uncalibrated loss trajectory shift exceeded the 0.10 proxy threshold.
- **README.md**: Updated Section A table to reflect `EXP-A2000-LLM-001` with explicit `Production Action` column distinguishing `Auto-Applied` deployments from `Blocked (Recommendation)` guardrail interventions.
- **examples/run_real_llm_benchmark.py**: Defaulted to labeling its output report
  `Model Target: Llama-3-8B-Empirical-Test` regardless of what was actually instantiated (a
  ~52M-parameter toy transformer). The script now computes and reports the real architecture and
  parameter count every time; any requested model name is shown only as a separate, clearly-marked
  note that never overwrites the real label.
- Removed `real_training_validation_report.md`, a stale, unreferenced report from before the
  correctness fixes that showed an unlabeled -45.9% regression.

## Overclaiming language corrected

- `TARGET_CLIENTS_PROFIT_MARGINS_AND_PITCHES.md`: "Guaranteed 0.0000 KL divergence" → corrected to
  describe what the verifier actually does (enforces a configurable threshold, downgrades anything
  that exceeds it to recommendation-only).
- `LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md`: "guarantees maximum hardware throughput (+20% to +45%)"
  removed — this repo's own measured results range from a small regression to +19.8%, not a
  guaranteed floor of +20%. Replaced with an accurate description tied to the README's real numbers.
  Same fix applied to a "guarantees zero training loss deviation" claim.
- `GHOST_LAYER_PITCH_AND_NETWORKING.md`: Two sales-script lines implied direct prior knowledge or
  experience this project doesn't have ("we already know the exact kernel bottlenecks on a 1,024x
  H100 cluster", "we've seen 20%+ waste on standard 8B architectures running on H100s" — no H100
  testing has occurred). Reframed around what the free-trial pitch actually offers to find out, and
  cited industry-published utilization data instead of an invented first-person result.
- `ENTERPRISE_EFFICIENCY_SCALE.md`: Added a note clarifying that "mathematically verified" refers to
  the arithmetic being correct given the stated inputs, not that the inputs themselves are measured
  client results — the cluster-scale scenarios in this document are illustrative models, not reports
  of actual deployments.
- **Financial Table & Evidence Boundary Audit (Canonical Alignment across docs)**:
  - **Tier 0B Precision:** Corrected displayed latency to `2.400 → 1.488 s (-38.0%)` so displayed step times reconcile with underlying financial spend ($1,271.56 baseline spend, $788.37 optimized spend, $483.19 gross savings, $120.80 fee).
  - **Tier 0A Evidence Label:** Removed misleading `*Empirical Baseline Verified on A2000*` label from the $29.40 / 40% modeled scenario. Labeled as `**Model scenario; physical A2000 evidence exists separately**` to clearly distinguish modeled workload projections from the separate physical A2000 benchmark receipt (52.2M parameter synthetic transformer, 31.27 ms -> 10.74 ms raw, 5-run mean +59.63% speedup with auto-apply blocked by the safety verifier).
  - **Tiers 0B–4B Evidence Labels:** Explicitly categorized as `**Analytical projection**` across all summary tables.
  - **Token-to-Step Derivation:** Explicitly annotated that global batch sizes in distributed PyTorch (Megatron-LM / FSDP) use power-of-two token allocations ($B \times S$, e.g. $2^{21} = 2,097,152$ tokens for Tier 1A/1B), explaining why dividing decimal token budgets ($1.0\text{T} = 10^{12}$) yields $\lfloor 10^{12} / 2^{21} \rfloor = 476,837$ steps rather than a naive decimal $500,000$.
  - **Wall-Clock Duration & Sensitivity Analysis:** Added explicit per-campaign wall-clock durations (e.g. 86.1 days for Tier 4A, 62.1 days for Tier 4B) and an enterprise multi-scenario sensitivity matrix (Conservative 1 campaign/yr, Base Modeled Scenario, and High-Utilization 4–12 campaigns/yr).

## Test suite count history & reconciliation

- **Historical Milestones:**
  - Initial OSS Core baseline: 385 tests
  - Post-remediation suite: 421 tests
  - Multi-dimensional & synthetic lift suite: 951 tests
  - Empirical verification suite: 971 tests
  - Current comprehensive suite: **995 active tests**
- **Canonical Machine Registry:** Test counts are dynamically managed and verified by [`TEST_INVENTORY.json`](file:///c:/Users/bharg/OneDrive/Documents/bhargav_projects/QuantGPUMgment/TEST_INVENTORY.json) (generated via `scripts/generate_test_inventory.py`). All 995 tests pass with 100% Green status.

