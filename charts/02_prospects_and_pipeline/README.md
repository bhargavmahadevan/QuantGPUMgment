# Pillar 2: Prospects & Pipeline Charts Gallery

Welcome to the visual analytical catalog for **Pillar 2: Prospects & Pipeline**.

These charts illustrate prospect qualification economics, client SLA containment boundaries, sub-run training taxonomy, and cross-client Bayesian transfer learning flywheels.

---

## 🎯 Visual Showcase

### 1. Containment & Rollback SLA Timeline
![Containment and Rollback SLA Timeline](containment_and_rollback_sla_timeline.png)

- **File**: `containment_and_rollback_sla_timeline.png`
- **Resolution**: 3900 × 2040 px (300 DPI)
- **Primary Citation**: [GANTT_ROADMAP_AND_TRAINING_TIMELINE.md](../../technical_docs/01_revenue_and_commercial/GANTT_ROADMAP_AND_TRAINING_TIMELINE.md)
- **Core Thesis**: Incident lifecycle comparison demonstrating why enterprises buy GhostLayer. A traditional gradient explosion or NaN corruption in unmonitored clusters causes an average of 5.9 hours of wasted compute, costing upwards of $28,000 in direct cloud waste plus checkpoint rollback rework. GhostLayer detects loss divergence in step 1, isolates the corrupt checkpoint, and executes an automated surgical rollback in under 450 ms with zero wasted compute hours.
- **Generator**: `python scripts/generate_roadmap_and_timeline_charts.py`

---

### 2. Subrun Operational Taxonomy & Campaign Matrix
![Subrun Taxonomy Breakdown](subrun_taxonomy_breakdown.png)

- **File**: `subrun_taxonomy_breakdown.png`
- **Resolution**: 3900 × 1860 px (300 DPI)
- **Primary Citation**: [GANTT_ROADMAP_AND_TRAINING_TIMELINE.md](../../technical_docs/01_revenue_and_commercial/GANTT_ROADMAP_AND_TRAINING_TIMELINE.md)
- **Core Thesis**: Deconstructs a 65-day frontier LLM pre-training campaign into 5 distinct operational sub-runs:
  1. *Subrun A (Data Sweeps & Filtering)*: 64 GPUs, 5 days, $8,448/day, $2,120 saved/day
  2. *Subrun B (MoE Architecture Search)*: 128 GPUs, 8 days, $16,896/day, $4,240 saved/day
  3. *Subrun C (Dense Foundation Pre-Training)*: 1,024 GPUs, 38 days, $135,168/day, $33,920 saved/day
  4. *Subrun D (Long-Context Extension)*: 512 GPUs, 9 days, $67,584/day, $16,960 saved/day
  5. *Subrun E (RLHF & Alignment)*: 256 GPUs, 5 days, $33,792/day, $8,480 saved/day
- **Generator**: `python scripts/generate_roadmap_and_timeline_charts.py`

---

### 3. Cross-Client Hive Mind Confidence Flywheel
![Hive Mind Throughput Flywheel](hive_mind_throughput_flywheel.png)

- **File**: `hive_mind_throughput_flywheel.png`
- **Resolution**: 2670 × 1409 px (300 DPI)
- **Primary Citation**: [LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md](../../technical_docs/03_engineering_and_algorithms/LLM_TRAINING_GAMEPLAN_AND_PLAYBOOK.md)
- **Core Thesis**: Illustrates GhostLayer's compounding data moat. Under cold-start conditions (1 client), tuning recommendations rely on heuristic priors (45% confidence). As clients join (5 $\to$ 20 $\to$ 100 clusters), the Bayesian Thompson sampling engine warm-starts hyperparameters across identical hardware topologies, driving recommendation confidence to 98% and providing instant zero-shot tuning acceleration.
- **Generator**: `python scripts/generate_engineering_and_telemetry_charts.py`
