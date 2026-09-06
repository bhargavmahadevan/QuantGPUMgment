# GhostLayer Master Chart Taxonomy & Visual Index

Welcome to the unified visual catalog for **GhostLayer (`QuantGPUMgment`)**.

This document indexes all **18 publication-quality analytical and telemetry charts** across their respective operational pillars. All charts are rendered at **300 DPI** using modern high-contrast dark aesthetics, with full data lineage linking back to our technical documentation.

---

## 🧭 Master Chart Taxonomy Matrix

| Pillar | Chart Filename | Resolution | Aspect | Primary Metric / Focus | Generating Script | Primary Documentation |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Pillar 1** | [`enterprise_cloud_cost_scaling.png`](01_revenue_and_commercial/enterprise_cloud_cost_scaling.png) | 4500 × 1800 | 2.5:1 | Baseline vs Optimized spend ($878.9k saved/run) | `scripts/generate_analysis_charts.py` | [GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md](../technical_docs/01_revenue_and_commercial/GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md) |
| **Pillar 1** | [`commercial_cashflow_trajectory.png`](01_revenue_and_commercial/commercial_cashflow_trajectory.png) | 3000 × 1650 | 1.82:1 | 12-Month ARR ramp ($180k $\to$ $6.54M ARR) | `scripts/generate_analysis_charts.py` | [GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md](../technical_docs/01_revenue_and_commercial/GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md) |
| **Pillar 1** | [`enterprise_65day_training_resource_allocation.png`](01_revenue_and_commercial/enterprise_65day_training_resource_allocation.png) | 3900 × 2850 | 1.37:1 | 65-Day GPU scaling, MFU boost (+25.6%), & dollar burn | `scripts/generate_roadmap_and_timeline_charts.py` | [GANTT_ROADMAP_AND_TRAINING_TIMELINE.md](../technical_docs/01_revenue_and_commercial/GANTT_ROADMAP_AND_TRAINING_TIMELINE.md) |
| **Pillar 1** | [`commercial_18month_workstream_gantt.png`](01_revenue_and_commercial/commercial_18month_workstream_gantt.png) | 4200 × 2250 | 1.87:1 | 6-Quarter engineering & commercial Gantt timeline | `scripts/generate_roadmap_and_timeline_charts.py` | [GANTT_ROADMAP_AND_TRAINING_TIMELINE.md](../technical_docs/01_revenue_and_commercial/GANTT_ROADMAP_AND_TRAINING_TIMELINE.md) |
| **Pillar 1** | [`financial_roi_breakdown.png`](01_revenue_and_commercial/financial_roi_breakdown.png) | 2670 × 1470 | 1.82:1 | Gross monthly compute savings vs $2,500 audit fee | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 2** | [`containment_and_rollback_sla_timeline.png`](02_prospects_and_pipeline/containment_and_rollback_sla_timeline.png) | 3900 × 2040 | 1.91:1 | Incident MTTR (<450 ms GhostLayer vs 5.9h manual stall) | `scripts/generate_roadmap_and_timeline_charts.py` | [GANTT_ROADMAP_AND_TRAINING_TIMELINE.md](../technical_docs/01_revenue_and_commercial/GANTT_ROADMAP_AND_TRAINING_TIMELINE.md) |
| **Pillar 2** | [`subrun_taxonomy_breakdown.png`](02_prospects_and_pipeline/subrun_taxonomy_breakdown.png) | 3900 × 1860 | 2.1:1 | Operational breakdown across 5 sub-runs (A through E) | `scripts/generate_roadmap_and_timeline_charts.py` | [GANTT_ROADMAP_AND_TRAINING_TIMELINE.md](../technical_docs/01_revenue_and_commercial/GANTT_ROADMAP_AND_TRAINING_TIMELINE.md) |
| **Pillar 2** | [`hive_mind_throughput_flywheel.png`](02_prospects_and_pipeline/hive_mind_throughput_flywheel.png) | 2670 × 1409 | 1.9:1 | Bayesian cross-client confidence flywheel (45% $\to$ 98%) | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 3** | [`step_latency_waterfall.png`](03_engineering_and_algorithms/step_latency_waterfall.png) | 2970 × 1470 | 2.02:1 | Latency reduction across 7 passes (350 ms $\to$ 78 ms) | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 3** | [`tensor_core_throughput.png`](03_engineering_and_algorithms/tensor_core_throughput.png) | 2676 × 1320 | 2.03:1 | Tensor Core saturation (110 $\to$ 580 TFLOPS / GPU) | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 3** | [`vram_allocation_breakdown.png`](03_engineering_and_algorithms/vram_allocation_breakdown.png) | 2670 × 1470 | 1.82:1 | VRAM footprint (activations reduced from 21.5 to 5.2 GB) | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 3** | [`hbm_bandwidth_reduction.png`](03_engineering_and_algorithms/hbm_bandwidth_reduction.png) | 2461 × 1320 | 1.86:1 | HBM read/write reduction (2,850 $\to$ 640 GB/s traffic) | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 3** | [`dataloader_stall_rate.png`](03_engineering_and_algorithms/dataloader_stall_rate.png) | 2341 × 1319 | 1.77:1 | GPU idle bubble percentage vs workers (42.5% $\to$ 1.2%) | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 3** | [`loss_convergence_comparison.png`](03_engineering_and_algorithms/loss_convergence_comparison.png) | 3000 × 1650 | 1.82:1 | Curvature speedup (60k steps Muon vs 100k steps AdamW) | `scripts/generate_analysis_charts.py` | [GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md](../technical_docs/01_revenue_and_commercial/GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md) |
| **Pillar 3** | [`loss_convergence_trajectory.png`](03_engineering_and_algorithms/loss_convergence_trajectory.png) | 2970 × 1470 | 2.02:1 | Loss preservation trajectory ($\Delta L \le 0.01$) | `scripts/generate_engineering_and_telemetry_charts.py` | [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md) |
| **Pillar 3** | [`gpu_time_allocation_decomposition.png`](03_engineering_and_algorithms/gpu_time_allocation_decomposition.png) | 3300 × 1560 | 2.12:1 | Compute time decomposition (useful compute 61.5% $\to$ 91%) | `scripts/generate_analysis_charts.py` | [GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md](../technical_docs/01_revenue_and_commercial/GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md) |
| **Pillar 3** | [`statistical_significance_distribution.png`](03_engineering_and_algorithms/statistical_significance_distribution.png) | 3000 × 1560 | 1.92:1 | Paired Welch's t-test density ($t=7.12, p<0.0001$) | `scripts/generate_analysis_charts.py` | [GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md](../technical_docs/01_revenue_and_commercial/GRAPHICAL_ANALYSIS_AND_FINANCIAL_MODELS.md) |
| **Pillar 3** | [`variance_sphere_telemetry.png`](03_engineering_and_algorithms/variance_sphere_telemetry.png) | 2100 × 1350 | 1.56:1 | Multi-scale gradient spherical variance geometry | Live `GhostWatcherHook.generate_variance_plot()` | [ENGINEERING_CALCULUS_AND_TELEMETRY.md](../technical_docs/03_engineering_and_algorithms/ENGINEERING_CALCULUS_AND_TELEMETRY.md) |

---

## 🗂️ Pillar Galleries

Explore detailed visual breakdowns, equations, and executive talking points in each pillar catalog:

1. **[Pillar 1: Revenue & Commercial](01_revenue_and_commercial/README.md)** (5 charts)
2. **[Pillar 2: Prospects & Pipeline](02_prospects_and_pipeline/README.md)** (3 charts)
3. **[Pillar 3: Engineering & Algorithms](03_engineering_and_algorithms/README.md)** (10 charts)

---

## ⚙️ Automated Chart Pipeline & Auditing

All charts are verified and audited programmatically:

```bash
# 1. Master generation of all 18 charts (300 DPI dark theme)
python scripts/generate_all_charts.py

# 2. Run Chart Analysis Agent audit
python scripts/analyze_charts.py
```

*Audit report:* See [`CHART_AUDIT_REPORT.md`](CHART_AUDIT_REPORT.md) for full metadata verification.
