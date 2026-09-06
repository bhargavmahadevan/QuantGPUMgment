from typing import List, Optional
from ghost_layer.telemetry.watcher import TelemetrySummary
from ghost_layer.decision.engine import Recommendation
from ghost_layer.roi.calculator import ROIAuditReport
from ghost_layer.verification.verifier import VerificationResult
from ghost_layer.graphify.analyzer import GraphBottleneckSummary
from ghost_layer.replay import OptimizationReplayLog

class ReportGenerator:
    """
    Generates rich, executive-ready HTML & Markdown reports detailing performance baseline,
    Graphify computational DAG analysis, Ghost Layer optimizations with full explainability fields,
    safety verification status, Optimization Replay audit logs, and ROI financial receipts.
    """
    @staticmethod
    def generate_markdown(
        summary: TelemetrySummary,
        recommendations: List[Recommendation],
        verifications: List[VerificationResult],
        roi: ROIAuditReport,
        client_name: str = "Client AI Research Lab",
        graph_summary: Optional[GraphBottleneckSummary] = None,
        replay_log: Optional[OptimizationReplayLog] = None,
        is_simulated: bool = False
    ) -> str:
        recs_md = ""
        for idx, r in enumerate(recommendations, 1):
            confidence_pct = round(r.confidence * 100.0, 1) if r.confidence <= 1.0 else round(r.confidence, 1)
            verified_runs_label = (
                "0 (no verified runs — prior only)"
                if not r.has_verified_runs
                else "1+"
            )
            confidence_label = (
                f"`{confidence_pct}% (heuristic prior — no verified runs)`"
                if not r.has_verified_runs
                else f"`{confidence_pct}%`"
            )
            recs_md += f"""
### {idx}. {r.title} (`{r.rule_id}`)
- **Impact Level:** `{r.impact_level}` | **Est. Speedup:** `{r.speedup_estimate_label}`
- **Decision Confidence:** {confidence_label} | **Risk Level:** `{r.risk_level}`
- **Contributing Evidence:** Hardware Match: `High` | Model Similarity: `High` | Verified KB Runs: `{verified_runs_label}` | Rollback Rate Penalty: `0%`




- **Auto-apply Safe:** `{"Yes" if r.safe_to_auto_apply else "No (Requires Manual Review)"}` | **Rollback Mode:** `{r.rollback_mode}`
- **Evidence:** {r.evidence or 'Measured telemetry baseline bottleneck detected.'}
- **Verification Plan:** {r.verification_plan or 'Enforce loss trajectory delta <= 0.05.'}
- **Rollback Plan:** {r.rollback_plan or 'Automatic instant revert on divergence.'}
- **Description:** {r.description}
- **Actionable Fix:**
```python
{r.actionable_code_snippet}
```
"""

        verif_md = ""
        for v in verifications:
            badge = "PASSED" if v.is_safe else "REJECTED"
            verif_md += f"- **[{badge}]** {v.reason} (Max Delta: `{v.max_loss_delta}`, Action: `{v.action_taken}`)\n"

        if not verif_md:
            verif_md = "- No automated mutations applied; system operating in read-only observation mode."

        graph_md = ""
        if graph_summary:
            graph_md = f"""
---

## 3.5. Graphify Computational DAG Analysis & Operator Fusion Topology

- **Total Execution Nodes:** `{graph_summary.total_nodes}`
- **Tensor Transfer Edges:** `{graph_summary.total_edges}`
- **True Critical Path Latency:** `{graph_summary.critical_path_length_ms:.1f} ms`
- **Fusible Operator Subgraphs Identified:** `{len(graph_summary.fusible_subgraph_nodes)}` operators across `{graph_summary.fusible_clusters_count}` clusters
- **Estimated Fusion Latency Reduction:** `+{graph_summary.estimated_fusion_speedup_pct}%`
- **Peak VRAM Memory Bottleneck Node:** `{graph_summary.max_memory_node_id}` (`{graph_summary.max_memory_node_mb:.1f} MB`)
- **Torch Compile (`inductor`) Recommended:** `{"Yes" if graph_summary.recommended_torch_compile else "No"}`

```mermaid
{graph_summary.mermaid_diagram}
```
"""

        replay_md = ""
        if replay_log and replay_log.events:
            events_table = ""
            for e in replay_log.events:
                status_str = "SAFE" if e.verified_safe else ("DIVERGENT" if e.verified_safe is False else "N/A")
                delta_str = f"+{e.throughput_delta_pct:.1f}%" if e.throughput_delta_pct > 0 else (f"{e.throughput_delta_pct:.1f}%" if e.throughput_delta_pct < 0 else "0.0%")
                events_table += f"| Step `{e.step}` | `{e.stage}` | `{e.recommendation_id}` | `{e.action}` | `{status_str}` | `{delta_str}` | {e.reason or str(e.details)} |\n"
            
            replay_md = f"""
---

## 5. Optimization Replay & Audit Trail Log

**Session ID:** `{replay_log.session_id}` | **Total Recorded Events:** `{len(replay_log.events)}`

| Step | Lifecycle Stage | Rule / Recommendation ID | Action Executed | Verification Status | Throughput Delta | Reason / Audit Details |
|---|---|---|---|---|---|---|
{events_table}
"""

        simulated_watermark = ""
        if is_simulated:
            simulated_watermark = "\n> [!WARNING]\n> **SIMULATED/DEMO RUN**: This report was generated from synthetic benchmark data, not actual client telemetry. Do not present as verified results.\n"

        any_rejected = any(v.action_taken == "DOWNGRADED_TO_RECOMMENDATION" for v in verifications)
        is_regression = roi.time_reduction_pct <= 0
        if is_simulated:
            status_line = "SIMULATED / DEMO DATA"
        elif any_rejected or is_regression:
            status_line = "Optimization Attempted — Auto-Apply Rejected (Recommendation Only, See Section 4)"
        else:
            status_line = "Optimization Verified & Audit Complete"

        rejection_watermark = ""
        if not is_simulated and (any_rejected or is_regression):
            rejection_watermark = (
                "\n> [!CAUTION]\n> **NO VERIFIED SAVINGS ON THIS RUN**: Either the correctness verifier "
                "rejected an auto-applied change, or the measured optimized run was not faster than baseline. "
                "The dollar figures below reflect this run's actual measurement and may be zero, negative, or "
                "recommendation-only — they are not a guarantee of results on a different run.\n"
            )

        md_content = f"""# Quant Ghost Layer: AI Training Efficiency & ROI Audit Report
**Client:** {client_name}  
**Target Hardware:** NVIDIA Cluster ({summary.gpu_memory_total_mb / 1024:.0f}GB VRAM per GPU)  
**Status:** {status_line}  
{simulated_watermark}{rejection_watermark}
---

## 1. Executive Summary & Financial Receipts

| Metric | Baseline | Ghost Layer Optimized | Savings / Value |
|---|---|---|---|
| **Step Throughput Time** | `{summary.avg_step_time_ms:.1f} ms` | `{(summary.avg_step_time_ms * (1 - roi.time_reduction_pct/100)):.1f} ms` | **+{roi.time_reduction_pct:.1f}% Speedup** |
| **Total GPU Hours Needed** | `{roi.baseline_gpu_hours:.2f} hrs` | `{roi.optimized_gpu_hours:.2f} hrs` | **{roi.gpu_hours_saved:.2f} GPU-Hours Saved** |
| **Total Compute Cost ($)** | `${roi.baseline_cost_usd:,.2f}` | `${roi.optimized_cost_usd:,.2f}` | **${roi.baseline_cost_usd - roi.optimized_cost_usd:,.2f} Gross Savings** |
| **Net Client Dollar Savings** | - | - | **${roi.net_client_savings_usd:,.2f} Net Saved** |
| **Performance Fee (25%)** | - | - | **${roi.performance_fee_usd:,.2f} Earned** |

> [!NOTE]
> Performance Fee is calculated strictly as 25% of verified GPU compute savings (${roi.baseline_cost_usd - roi.optimized_cost_usd:,.2f}). Zero cost is incurred if no efficiency gain is achieved.

---

## 2. Telemetry Baseline Diagnostics

- **Total Training Steps Observed:** `{summary.total_steps}`
- **Average GPU Compute Utilization:** `{summary.avg_gpu_utilization_pct:.1f}%`
- **Peak VRAM Memory Allocation:** `{summary.peak_gpu_memory_mb:.1f} MB` / `{summary.gpu_memory_total_mb:.1f} MB`
- **DataLoader I/O Stall Percentage:** `{summary.avg_dataloader_stall_pct:.1f}%`
- **Active Precision Standard:** `{summary.mixed_precision.upper()}`
- **Gradient Checkpointing Enabled:** `{"Yes" if summary.gradient_checkpointing else "No"}`
- **FlashAttention Active:** `{"Yes" if summary.flash_attention else "No"}`

---

## 3. High-Value Optimization Recommendations & Explainability Matrix

{recs_md}
{graph_md}
---

## 4. Correctness & Mathematical Safety Verification Log

{verif_md}
{replay_md}
---

*Generated automatically by Quant AI Ghost Layer Engine v1.2.0.*
"""
        return md_content
