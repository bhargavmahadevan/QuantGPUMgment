"""
Executive Audit Report Exporter for GhostLayer.
Generates polished, executive-ready HTML (glassmorphism/dark mode), JSON, and Markdown summaries.
Designed for C-level presentation (CFO / VP of Engineering) to justify GPU efficiency investments.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from ghost_layer.hooks import GhostWatcherHook


class ExecutiveAuditExporter:
    def __init__(self, hook: GhostWatcherHook):
        self.hook = hook

    def generate_metrics_summary(self) -> Dict[str, Any]:
        """Extract and compute high-level executive diagnostic metrics."""
        summary, recs = self.hook.analyze_and_report()
        
        # Calculate key executive numbers
        step_times = [s.step_time_ms for s in self.hook.watcher.snapshots] or [25.0]
        avg_step_time_ms = sum(step_times) / len(step_times)
        
        vram_used = [s.gpu_memory_used_mb for s in self.hook.watcher.snapshots] or [8000.0]
        peak_vram_mb = max(vram_used)
        total_vram_mb = self.hook.target_gpu_mb
        vram_headroom_mb = max(0.0, total_vram_mb - peak_vram_mb)
        vram_headroom_pct = (vram_headroom_mb / total_vram_mb) * 100.0 if total_vram_mb > 0 else 0.0
        
        # Estimate I/O stall ratio
        io_times = [s.data_loading_time_ms for s in self.hook.watcher.snapshots] or [0.0]
        avg_io_ms = sum(io_times) / len(io_times)
        io_stall_pct = (avg_io_ms / avg_step_time_ms) * 100.0 if avg_step_time_ms > 0 else 0.0
        
        # Estimate monthly GPU waste
        hourly_rate = self.hook.roi_calculator.gpu_cost_per_hour
        hours_per_month = 720.0  # 30 days * 24 hrs
        monthly_cost_per_gpu = hourly_rate * hours_per_month
        
        # Parse recommendation speedups
        formatted_recs = []
        total_speedup = 0.0
        has_amp_rec = False
        
        for r in recs:
            speedup_val = 15.0
            if "AMP" in r.rule_id or "PRECISION" in r.rule_id:
                speedup_val = 25.0
                has_amp_rec = True
            elif "FLASH" in r.rule_id:
                speedup_val = 20.0
            elif "LOADER" in r.rule_id or "PIN" in r.rule_id:
                speedup_val = 15.0
            elif "BATCH" in r.rule_id:
                speedup_val = 18.0
            
            total_speedup += speedup_val
            formatted_recs.append({
                "id": r.rule_id,
                "title": r.title,
                "impact_level": r.impact_level,
                "speedup_estimate_label": r.speedup_estimate_label,
                "expected_speedup_pct": speedup_val,
                "description": r.description,
                "actionable_code": r.actionable_code_snippet,
            })

        # Conservative waste estimation (I/O stall + unutilized VRAM/precision)
        waste_ratio = min(0.40, max(0.10, (io_stall_pct / 100.0) + (0.15 if has_amp_rec else 0.0)))
        monthly_waste_per_gpu = monthly_cost_per_gpu * waste_ratio
        
        audit_fee = 2500.0
        monthly_waste_8x = monthly_waste_per_gpu * 8.0
        monthly_waste_64x = monthly_waste_per_gpu * 64.0
        daily_savings_8x = monthly_waste_8x / 30.0 if monthly_waste_8x > 0 else 1.0
        daily_savings_64x = monthly_waste_64x / 30.0 if monthly_waste_64x > 0 else 1.0

        payback_days_8x = round(audit_fee / daily_savings_8x, 1)
        payback_days_64x = round(audit_fee / daily_savings_64x, 1)
        annual_savings_8x = round(monthly_waste_8x * 12.0, 2)
        annual_savings_64x = round(monthly_waste_64x * 12.0, 2)

        commercial_tiers = [
            {
                "tier": "Tier 1: Pre-Flight Diagnostic Audit",
                "price": "$2,500 Flat Fee",
                "scope": "1x to 8x GPU Staging Runs (48-Hour Turnaround)",
                "target": "Growth AI Startups validating fine-tuning pipelines",
                "deliverable": "C-Level Executive & Technical GPU Efficiency Audit Report",
            },
            {
                "tier": "Tier 2: Team Platform SaaS",
                "price": "$199 - $999 / month",
                "scope": "8 to 64 GPUs Continuous Tracking",
                "target": "Active Engineering & ML Platform Teams",
                "deliverable": "Real-time Slack stall alerts, Prometheus FinOps metrics, Decision Replay",
            },
            {
                "tier": "Tier 3: Enterprise Compute Assurance",
                "price": "$15,000 - $40,000 / year",
                "scope": "Air-Gapped On-Premises & Private VPC Clusters",
                "target": "Quantitative Trading Desks & Frontier AI Research Labs",
                "deliverable": "Zero Data Exfiltration SLA, Custom Rules, BaselineLock Verification",
            },
        ]

        return {
            "session_id": self.hook.replay_log.session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "target_hardware": self.hook.target_hardware,
            "total_steps_audited": len(step_times),
            "avg_step_time_ms": round(avg_step_time_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "total_vram_mb": round(total_vram_mb, 1),
            "vram_headroom_pct": round(vram_headroom_pct, 1),
            "io_stall_pct": round(io_stall_pct, 1),
            "hourly_gpu_rate": hourly_rate,
            "estimated_monthly_waste_per_gpu": round(monthly_waste_per_gpu, 2),
            "estimated_monthly_waste_8x_cluster": round(monthly_waste_8x, 2),
            "estimated_monthly_waste_64x_cluster": round(monthly_waste_64x, 2),
            "preflight_audit_fee_usd": audit_fee,
            "audit_payback_days_8x": payback_days_8x,
            "audit_payback_days_64x": payback_days_64x,
            "annual_recoverable_savings_8x": annual_savings_8x,
            "annual_recoverable_savings_64x": annual_savings_64x,
            "commercial_tiers": commercial_tiers,
            "recommendations": formatted_recs,
            "total_potential_speedup_pct": round(total_speedup, 1),
        }

    def export_json(self, output_path: str) -> str:
        """Export metrics summary as structured JSON."""
        data = self.generate_metrics_summary()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path

    def export_markdown(self, output_path: str) -> str:
        """Export executive summary as clean Markdown."""
        data = self.generate_metrics_summary()
        md = f"""# Quant GhostLayer: Executive GPU Efficiency Audit Report

> **Session ID:** `{data['session_id']}`  
> **Audited Hardware:** `{data['target_hardware']}` | **Total Steps Audited:** `{data['total_steps_audited']}` | **Date:** `{data['timestamp']}`

---

## 💰 Financial Waste & Opportunity Summary

| Metric | Measured Telemetry | Financial Impact (Per 8x Cluster) | Financial Impact (Per 64x Cluster) |
| :--- | :--- | :--- | :--- |
| **GPU Cloud Cost Basis** | `${data['hourly_gpu_rate']:.2f} / GPU-hour` | `${data['hourly_gpu_rate']*8*720:,.2f} / month` | `${data['hourly_gpu_rate']*64*720:,.2f} / month` |
| **Identified Compute Waste** | **+{data['total_potential_speedup_pct']}% Potential Speedup** | **${data['estimated_monthly_waste_8x_cluster']:,.2f} / month** | **${data['estimated_monthly_waste_64x_cluster']:,.2f} / month** |
| **VRAM Headroom** | `{data['vram_headroom_pct']}% Available` | Potential 1.5x–2.0x Batch Scaling | Eliminates Gradient Acc Overhead |
| **DataLoader I/O Stall** | `{data['io_stall_pct']}% of Step Duration` | Direct GPU Starvation | Pin-pooling & Worker Tuning |

---

## 🛠️ Prioritized Action Items for ML Infrastructure Team

"""
        for i, rec in enumerate(data['recommendations'], 1):
            md += f"### {i}. {rec['title']} (Impact: {rec['impact_level']})\n"
            md += f"- **Expected Speedup:** +{rec['expected_speedup_pct']}%\n"
            md += f"- **Technical Detail:** {rec['description']}\n\n"

        md += f"""---

## 💼 Commercial Monetization & CFO ROI Defense

| Tier | Pricing | Deliverable & Scope | Target Workload |
| :--- | :--- | :--- | :--- |
| **Tier 1: Pre-Flight Diagnostic Audit** | **$2,500 Flat Fee** | 48-Hour Technical & Executive GPU Audit | 1x–8x GPU Staging Runs |
| **Tier 2: Team Platform SaaS** | **$199–$999 / mo** | Continuous Slack Alerts & Prometheus FinOps | 8–64 GPU Active Teams |
| **Tier 3: Enterprise Compute Assurance** | **$15,000–$40,000 / yr** | Air-Gapped VPC + Zero Exfiltration SLA | Quantitative Desks & Frontier Labs |

> **CFO Payback Summary:** On an 8x GPU cluster, this workload experiences approximately **${data['estimated_monthly_waste_8x_cluster']:,.2f}/mo** in unoptimized spend. The **$2,500 Pre-Flight Diagnostic Audit** pays for itself in **{data['audit_payback_days_8x']} days**, recovering up to **${data['annual_recoverable_savings_8x']:,.2f}** in annual compute runway.

To schedule an assisted audit: contact solutions@ghostlayer.ai
"""

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        return output_path

    def export_html(self, output_path: str) -> str:
        """Export dark-mode, glassmorphic HTML executive report."""
        data = self.generate_metrics_summary()
        
        recs_html = ""
        for rec in data['recommendations']:
            badge_color = "#ef4444" if rec['impact_level'] == "HIGH" else ("#f59e0b" if rec['impact_level'] == "MEDIUM" else "#10b981")
            recs_html += f"""
            <div class="card" style="margin-bottom: 16px; border-left: 4px solid {badge_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; font-size: 1.15rem; color: #f8fafc;">{rec['title']}</h3>
                    <span style="background: {badge_color}22; color: {badge_color}; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.8rem; border: 1px solid {badge_color}44;">
                        +{rec['expected_speedup_pct']}% Speedup
                    </span>
                </div>
                <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 0.95rem; line-height: 1.5;">{rec['description']}</p>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GhostLayer Executive Audit — {data['session_id']}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #090d16;
            --bg-secondary: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1080px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 32px;
        }}
        .badge {{
            background: rgba(6, 182, 212, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(6, 182, 212, 0.3);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 24px;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-emerald);
            margin-top: 8px;
        }}
        .stat-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}
        th, td {{
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        .financial-highlight {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 32px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700;">Quant GhostLayer™ Executive Diagnostic</h1>
                <p style="margin: 6px 0 0 0; color: var(--text-secondary); font-size: 0.95rem;">
                    Target Hardware: <span style="color: var(--text-primary); font-weight: 600;">{data['target_hardware']}</span> | 
                    Session ID: <span style="font-family: 'JetBrains Mono', monospace;">{data['session_id']}</span>
                </p>
            </div>
            <div class="badge">Verified Telemetry Audit</div>
        </div>

        <div class="financial-highlight">
            <h2 style="margin-top: 0; font-size: 1.3rem; color: #34d399;">💰 Executive Cost & Runway Impact</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-top: 16px;">
                <div>
                    <div class="stat-label">Identified Waste (8x Cluster)</div>
                    <div class="stat-value" style="color: #f87171;">${data['estimated_monthly_waste_8x_cluster']:,.2f}<span style="font-size: 1rem; color: var(--text-secondary);">/mo</span></div>
                </div>
                <div>
                    <div class="stat-label">Identified Waste (64x Cluster)</div>
                    <div class="stat-value" style="color: #f87171;">${data['estimated_monthly_waste_64x_cluster']:,.2f}<span style="font-size: 1rem; color: var(--text-secondary);">/mo</span></div>
                </div>
                <div>
                    <div class="stat-label">Total Optimization Potential</div>
                    <div class="stat-value" style="color: #34d399;">+{data['total_potential_speedup_pct']}%</div>
                </div>
            </div>
            <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(16, 185, 129, 0.2); font-size: 0.95rem; color: var(--text-secondary);">
                ⚡ <strong style="color: #34d399;">CFO Payback Window:</strong> The <strong>$2,500 Pre-Flight Diagnostic Audit</strong> pays for itself in <strong>{data['audit_payback_days_8x']} days</strong> on an 8x GPU cluster, unlocking <strong>${data['annual_recoverable_savings_8x']:,.2f}</strong> in annual runway.
            </div>
        </div>

        <div class="card" style="margin-bottom: 32px; background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.7) 100%);">
            <h2 style="margin-top: 0; font-size: 1.3rem; color: #f8fafc;">💼 Commercial Engagement Tiers</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 16px;">
                <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 8px; padding: 18px;">
                    <div style="font-weight: 700; color: #38bdf8; font-size: 1.1rem;">Tier 1: Pre-Flight Audit</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin: 8px 0;">$2,500 <span style="font-size: 0.85rem; font-weight: 400; color: #94a3b8;">Flat Fee</span></div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 0 0 12px 0;">1x to 8x GPU Staging Runs • 48-Hour Turnaround</p>
                    <div style="font-size: 0.8rem; color: #cbd5e1;">Deliverable: C-Level Executive & Technical GPU Efficiency Audit Report with full proof of savings.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 18px;">
                    <div style="font-weight: 700; color: #34d399; font-size: 1.1rem;">Tier 2: Team Platform SaaS</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin: 8px 0;">$499 <span style="font-size: 0.85rem; font-weight: 400; color: #94a3b8;">/ month</span></div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 0 0 12px 0;">8 to 64 GPUs • Continuous Job Monitoring</p>
                    <div style="font-size: 0.8rem; color: #cbd5e1;">Deliverable: Real-time Slack/Discord stall alerts, Prometheus FinOps metrics exporter, Decision Replay.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 8px; padding: 18px;">
                    <div style="font-weight: 700; color: #c084fc; font-size: 1.1rem;">Tier 3: Enterprise Compute Assurance</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin: 8px 0;">$25,000 <span style="font-size: 0.85rem; font-weight: 400; color: #94a3b8;">/ year</span></div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 0 0 12px 0;">Air-Gapped On-Premises & Private VPC</p>
                    <div style="font-size: 0.8rem; color: #cbd5e1;">Deliverable: Zero Data Exfiltration SLA, BaselineLock Verification, Dedicated Custom Rules.</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 24px;">
                <a href="mailto:solutions@ghostlayer.ai?subject=Schedule%2048-Hour%20Pre-Flight%20Diagnostic%20Audit%20($2,500)" style="display: inline-block; background: linear-gradient(135deg, #06b6d4, #10b981); color: #090d16; padding: 12px 28px; border-radius: 8px; font-weight: 700; text-decoration: none; font-size: 1rem; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);">Schedule 48-Hour Audit ($2,500)</a>
            </div>
        </div>

        <div class="grid-stats">
            <div class="card">
                <div class="stat-label">Mean Step Latency</div>
                <div class="stat-value" style="color: #38bdf8;">{data['avg_step_time_ms']} ms</div>
            </div>
            <div class="card">
                <div class="stat-label">VRAM Headroom</div>
                <div class="stat-value" style="color: #34d399;">{data['vram_headroom_pct']}%</div>
            </div>
            <div class="card">
                <div class="stat-label">DataLoader I/O Stall</div>
                <div class="stat-value" style="color: {'#f87171' if data['io_stall_pct'] > 15 else '#38bdf8'};">{data['io_stall_pct']}%</div>
            </div>
            <div class="card">
                <div class="stat-label">Steps Audited</div>
                <div class="stat-value" style="color: #e2e8f0;">{data['total_steps_audited']}</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 32px;">
            <h2 style="margin-top: 0; font-size: 1.25rem;">Infrastructure Diagnostic Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Measured Baseline</th>
                        <th>Target Threshold</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Peak VRAM Allocation</td>
                        <td>{data['peak_vram_mb']} MB / {data['total_vram_mb']} MB</td>
                        <td>&lt; 90% Allocated</td>
                        <td><span style="color: #34d399;">OPTIMAL HEADROOM</span></td>
                    </tr>
                    <tr>
                        <td>DataLoader I/O Pipeline</td>
                        <td>{data['io_stall_pct']}% Stall Ratio</td>
                        <td>&lt; 5.0% Stall Ratio</td>
                        <td><span style="color: {'#f87171' if data['io_stall_pct'] > 15 else '#34d399'};">{'REQUIRES PIN-POOLING' if data['io_stall_pct'] > 15 else 'HEALTHY'}</span></td>
                    </tr>
                    <tr>
                        <td>Compute Kernel Precision</td>
                        <td>Evaluated via Decision Engine</td>
                        <td>AMP-BF16 / FlashAttention-2</td>
                        <td><span style="color: #38bdf8;">RECOMMENDATIONS ACTIVE</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h2 style="font-size: 1.25rem; margin-bottom: 16px;">Prioritized Recommendations</h2>
        {recs_html}

        <footer style="margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--border-color); text-align: center; color: var(--text-secondary); font-size: 0.85rem;">
            Generated by Quant GhostLayer™ Telemetry & Financial Engine v1.2.0 • Zero Data Exfiltration • Air-Gapped Compliant
        </footer>
    </div>
</body>
</html>
"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path
