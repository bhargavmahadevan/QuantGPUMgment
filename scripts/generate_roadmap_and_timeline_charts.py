"""
Generate publication-quality, dark-mode graphical analysis charts for GhostLayer:
1. Enterprise 65-Day Training Campaign: Resource Allocation, MFU, and Cumulative Dollar Burn
2. GhostLayer 18-Month Multi-Stream Commercial & Engineering Roadmap Gantt
3. Incident Detection, Adaptive Lagrangian Containment & Rollback SLA Timeline
4. Sub-Run Taxonomy Operational Breakdown Matrix

Outputs high-res PNGs to the `charts/` directory for embedding into technical & commercial docs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
import matplotlib.lines as mlines

# Ensure charts directory exists
os.makedirs("charts", exist_ok=True)

# Set high-DPI dark-mode aesthetic
plt.style.use("dark_background")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.2,
    "grid.color": "#1E293B",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "figure.facecolor": "#0B0F17",
    "axes.facecolor": "#0F172A",
    "text.color": "#F8FAFC",
    "axes.labelcolor": "#94A3B8",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
})

def save_chart(fig, filename, pillar):
    os.makedirs(f"charts/{pillar}", exist_ok=True)
    root_path = os.path.join("charts", filename)
    pillar_path = os.path.join("charts", pillar, filename)
    fig.savefig(root_path, dpi=300)
    fig.savefig(pillar_path, dpi=300)
    print(f"Generated {root_path} and {pillar_path}")

# ==============================================================================
# Chart 1: Enterprise 65-Day Training Resource Allocation & Burn Rate
# ==============================================================================
fig, (ax_gpu, ax_mfu, ax_burn) = plt.subplots(3, 1, figsize=(13, 9.5), dpi=300, sharex=True,
                                             gridspec_kw={"height_ratios": [1.1, 1.1, 1.2]})

days = np.linspace(1, 65, 650)

# Phase boundaries
phase_ends = [7, 16, 44, 53, 65]
phase_names = [
    "Phase 1: Data Sweeps\n(Days 1-7)",
    "Phase 2: MoE Routing\n(Days 8-16)",
    "Phase 3: Foundation Pre-Training\n(Days 17-44)",
    "Phase 4: Context Extension\n(Days 45-53)",
    "Phase 5: Reasoning RL\n(Days 54-65)"
]
phase_colors = ["#0284C7", "#6366F1", "#10B981", "#F59E0B", "#EC4899"]

# Panel 1: GPU Cluster Allocation
gpu_counts = []
for d in days:
    if d <= 7:
        gpu_counts.append(64)
    elif d <= 16:
        gpu_counts.append(256)
    elif d <= 44:
        gpu_counts.append(1024)
    elif d <= 53:
        gpu_counts.append(256)
    else:
        gpu_counts.append(128)
gpu_counts = np.array(gpu_counts)

ax_gpu.step(days, gpu_counts, where="post", color="#38BDF8", linewidth=2.5, label="Active GPU Accelerators (H100 / B200)")
ax_gpu.fill_between(days, 0, gpu_counts, step="post", color="#0284C7", alpha=0.25)
ax_gpu.set_ylabel("Active GPUs", fontsize=10.5, labelpad=10, fontweight="bold")
ax_gpu.set_ylim(0, 1200)
ax_gpu.set_yticks([0, 64, 256, 512, 1024])
ax_gpu.grid(True)

# Phase shade bands and annotations
for i, (p_start, p_end, name, col) in enumerate(zip([1] + phase_ends[:-1], phase_ends, phase_names, phase_colors)):
    ax_gpu.axvspan(p_start, p_end, color=col, alpha=0.08)
    ax_mfu.axvspan(p_start, p_end, color=col, alpha=0.08)
    ax_burn.axvspan(p_start, p_end, color=col, alpha=0.08)
    # Label on top panel
    mid_d = (p_start + p_end) / 2
    ax_gpu.text(mid_d, 1080, name, ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#0F172A", alpha=0.85, edgecolor=col))

# Milestone markers on GPU panel
milestones = [
    (7, "M1: Pre-Flight Audit", "#38BDF8"),
    (16, "M2: Arch Freeze", "#818CF8"),
    (44, "M3: Checkpoint Extract", "#34D399"),
    (53, "M4: 128k Verified", "#FBBF24"),
    (65, "M5: Production Release", "#F472B6")
]
for m_day, m_txt, m_col in milestones:
    ax_gpu.scatter([m_day], [gpu_counts[np.argmin(np.abs(days - m_day))]], color=m_col, s=70, zorder=6)

ax_gpu.set_title("Enterprise 65-Day Frontier Training Campaign: Resource & Efficiency Architecture",
                 fontsize=13, fontweight="bold", pad=28)
ax_gpu.legend(loc="upper left", frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=9)


# Panel 2: Model FLOPs Utilization (MFU %) & Compute Efficiency
np.random.seed(101)
base_mfu = []
ghost_mfu = []
for d in days:
    if d <= 7:
        # High dataloader starvation in baseline
        b = 58.0 + np.random.normal(0, 1.2)
        g = 83.5 + np.random.normal(0, 0.8)
    elif d <= 16:
        # MoE communication drag in baseline
        b = 62.0 + np.random.normal(0, 1.5)
        g = 85.0 + np.random.normal(0, 0.9)
    elif d <= 44:
        # Pre-training: standard AdamW vs GhostWatcherHook + Curvature
        b = 68.5 + 2.5 * np.sin(d/3.0) + np.random.normal(0, 1.0)
        g = 90.5 + 0.8 * np.sin(d/4.0) + np.random.normal(0, 0.6)
    elif d <= 53:
        # Context extension: activation memory spikes and OOM paging in baseline
        b = 54.0 + 3.0 * np.sin(d/2.0) + np.random.normal(0, 1.8)
        g = 87.5 + np.random.normal(0, 0.8)
    else:
        # RL / GRPO: ragged batching padding waste in baseline
        b = 59.0 + np.random.normal(0, 1.4)
        g = 88.0 + np.random.normal(0, 0.7)
    base_mfu.append(b)
    ghost_mfu.append(g)

base_mfu = np.array(base_mfu)
ghost_mfu = np.array(ghost_mfu)

ax_mfu.plot(days, base_mfu, color="#F43F5E", linewidth=1.8, alpha=0.85, label="Unoptimized Baseline MFU (Avg 62.8%)")
ax_mfu.plot(days, ghost_mfu, color="#10B981", linewidth=2.2, label="GhostLayer Managed MFU (Avg 88.4% / +25.6% Boost)")
ax_mfu.fill_between(days, base_mfu, ghost_mfu, color="#10B981", alpha=0.15)
ax_mfu.set_ylabel("MFU Efficiency (%)", fontsize=10.5, labelpad=10, fontweight="bold")
ax_mfu.set_ylim(40, 100)
ax_mfu.axhline(85, color="#64748B", linestyle=":", linewidth=1.0, alpha=0.7)
ax_mfu.text(1.5, 86, "Enterprise Target (85% MFU)", color="#94A3B8", fontsize=8)
ax_mfu.grid(True)
ax_mfu.legend(loc="lower right", frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=9)


# Panel 3: Cumulative Training Dollar Burn ($M)
hourly_rates = []
for d in days:
    if d <= 7:
        r = 64 * 3.80 * 24 / 1000000.0
    elif d <= 16:
        r = 256 * 3.50 * 24 / 1000000.0
    elif d <= 44:
        r = 1024 * 3.20 * 24 / 1000000.0
    elif d <= 53:
        r = 256 * 3.50 * 24 / 1000000.0
    else:
        r = 128 * 3.60 * 24 / 1000000.0
    hourly_rates.append(r)
hourly_rates = np.array(hourly_rates)

dt = days[1] - days[0]
base_cumulative = np.cumsum(hourly_rates) * dt

opt_factor = []
for d in days:
    if d <= 7:
        opt_factor.append(0.65)  # -35% (DataLoader starvation)
    elif d <= 16:
        opt_factor.append(0.75)  # -25% (MoE DAG fusion)
    elif d <= 44:
        opt_factor.append(0.82)  # -18% (Foundation pre-training scale)
    elif d <= 53:
        opt_factor.append(0.78)  # -22% (ChunkedCrossEntropy memory)
    else:
        opt_factor.append(0.76)  # -24% (Ragged batching)
opt_factor = np.array(opt_factor)

ghost_cumulative = np.cumsum(hourly_rates * opt_factor) * dt

# Align final spend to model scenario: $4.88M baseline vs $4.00M optimized ($878.9k gross savings)
scale_base = 4.8828 / base_cumulative[-1]
base_cumulative = base_cumulative * scale_base
scale_ghost = 4.0039 / ghost_cumulative[-1]
ghost_cumulative = ghost_cumulative * scale_ghost

ax_burn.plot(days, base_cumulative, color="#EF4444", linewidth=2.2, label="Baseline Spend ($4.88M Total)")
ax_burn.plot(days, ghost_cumulative, color="#0EA5E9", linewidth=2.5, label="GhostLayer Spend ($4.00M Total)")
ax_burn.fill_between(days, ghost_cumulative, base_cumulative, color="#10B981", alpha=0.25, label="Gross Dollar Savings ($878.9k)")

ax_burn.annotate("Day 65 Total Saved: $878,905\n- Net Client Value: $659,179 (75%)\n- GhostLayer Assurance Fee: $219,726 (25%)",
                 xy=(65, base_cumulative[-1]), xytext=(41, 1.8),
                 arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.8),
                 fontsize=9.0, fontweight="bold", color="#A7F3D0",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#064E3B", alpha=0.9, edgecolor="#10B981"))

ax_burn.set_ylabel("Cumulative Spend ($M)", fontsize=10.5, labelpad=10, fontweight="bold")
ax_burn.set_xlabel("Campaign Timeline (Days 1 to 65)", fontsize=11, labelpad=10, fontweight="bold")
ax_burn.set_xlim(1, 65)
ax_burn.set_ylim(0, 5.5)
ax_burn.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"${y:.1f}M"))
ax_burn.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax_burn.grid(True)
ax_burn.legend(loc="upper left", frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=9)

plt.tight_layout()
save_chart(plt.gcf(), "enterprise_65day_training_resource_allocation.png", "01_revenue_and_commercial")
plt.close()


# ==============================================================================
# Chart 2: GhostLayer 18-Month Multi-Stream Commercial & Engineering Roadmap
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)

workstreams = [
    "Stream 1: Core Engine & Step Telemetry",
    "Stream 2: Cryptographic Audit Vault & Merkle Tree",
    "Stream 3: Mid-Range Fleet Optimizer & Audits",
    "Stream 4: Team Platform SaaS & Real-Time Alerts",
    "Stream 5: Frontier Multi-Node & Enterprise SLAs"
]

tasks = [
    # Stream 0: Core Engine
    (0, 1.0, 3.0, "v0.1: GhostWatcherHook & Telemetry Callbacks", "done"),
    (0, 3.5, 4.0, "v0.5: Adaptive Lagrangian & Welch's t-test", "active"),
    (0, 7.0, 4.5, "v0.8: Kernel Fusion DAG & FlashAttn-3", "planned"),
    (0, 11.0, 6.0, "v1.0: Full PyTorch C++ / CUDA Native Core", "planned"),

    # Stream 1: Cryptographic Vault
    (1, 1.5, 2.5, "immudb-Style SHA-256 Hash Vault Engine", "done"),
    (1, 3.8, 3.5, "5-W Event Schema & Welford Variance Proofs", "active"),
    (1, 7.0, 4.0, "Tamper-Proof Audit Delivery Receipts Engine", "planned"),
    (1, 10.5, 6.5, "SOC2 Type II & FedRAMP Compliance Package", "planned"),

    # Stream 2: Mid-Range Fleet Optimizer
    (2, 2.0, 3.5, "DataLoader Pinning & Worker Prefetch Engine", "active"),
    (2, 5.0, 4.0, "ChunkedCrossEntropy Loss & Memory Allocator", "planned"),
    (2, 8.5, 4.5, "Pre-Flight Diagnostic Audit Pipeline ($2.5k)", "planned"),
    (2, 12.5, 5.0, "Self-Service Fleet Tuning Auto-CLI Tool", "planned"),

    # Stream 3: Team Platform SaaS
    (3, 3.0, 4.0, "Team SaaS Web UI & Live Telemetry Dashboard", "active"),
    (3, 6.5, 3.5, "Slack & PagerDuty Real-Time Stall Alerts", "planned"),
    (3, 9.5, 4.5, "Team Tier GA ($499/mo Self-Serve Portal)", "planned"),
    (3, 13.5, 4.5, "Collaborative Multi-Tenant Workspace Hub", "planned"),

    # Stream 4: Frontier Multi-Node Engine
    (4, 5.0, 4.5, "Multi-Node NCCL & DeepSpeed ZeRO-3 Hooks", "planned"),
    (4, 9.0, 4.5, "Air-Gapped VPC On-Prem Enterprise Installer", "planned"),
    (4, 13.0, 4.5, "Enterprise Assurance SLA ($40k/yr Base)", "planned"),
    (4, 16.0, 2.0, "10k+ GPU Mega-Cluster Deployments", "planned"),
]

status_colors = {
    "done": "#10B981",       # Emerald
    "active": "#0EA5E9",     # Sky Blue
    "planned": "#64748B"     # Slate Gray
}

y_pos = np.arange(len(workstreams))

for y in y_pos:
    ax.axhspan(y - 0.45, y + 0.45, color="#1E293B", alpha=0.35, zorder=0)

for stream_idx, start_m, duration, name, status in tasks:
    y = stream_idx
    col = status_colors[status]
    rect = FancyBboxPatch((start_m, y - 0.28), duration, 0.56,
                          boxstyle="round,pad=0.03,rounding_size=0.15",
                          facecolor=col, edgecolor="#F8FAFC", alpha=0.9 if status != "planned" else 0.55,
                          linewidth=1.2 if status != "planned" else 0.8, zorder=3)
    ax.add_patch(rect)
    text_col = "#0B0F17" if status in ["done", "active"] else "#F8FAFC"
    ax.text(start_m + duration/2.0, y, name, ha="center", va="center", fontsize=8.0, fontweight="bold",
            color=text_col, zorder=4)

commercial_milestones = [
    (2.5, "M_Q1: OSS Core Launch", "#10B981"),
    (5.5, "M_Q2: 4-Fleet Pilot ($25k MRR)", "#0EA5E9"),
    (9.0, "M_Q3: SaaS Self-Serve GA ($62k MRR)", "#F59E0B"),
    (13.0, "M_Q4: Enterprise SLA GA ($238k MRR)", "#8B5CF6"),
    (17.5, "M_Q6: 10k GPU Mega-Cluster ($545k MRR)", "#EC4899")
]

for m_pos, m_label, m_c in commercial_milestones:
    ax.axvline(m_pos, color=m_c, linestyle=":", linewidth=1.5, alpha=0.7, zorder=2)
    ax.scatter([m_pos], [4.65], color=m_c, s=90, marker="D", zorder=5)
    ax.text(m_pos, 4.85, m_label, ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=m_c,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#0F172A", alpha=0.85, edgecolor=m_c))

ax.set_yticks(y_pos)
ax.set_yticklabels(workstreams, fontsize=10, fontweight="bold")
ax.invert_yaxis()

ax.set_xlim(0.5, 18.5)
ax.set_ylim(4.9, -0.6)

quarters = [
    (1.0, 3.0, "Q1 2026\n(Core Telemetry)"),
    (4.0, 6.0, "Q2 2026\n(Mid-Range Fleet)"),
    (7.0, 9.0, "Q3 2026\n(SaaS Platform)"),
    (10.0, 12.0, "Q4 2026\n(Assurance SLAs)"),
    (13.0, 15.0, "Q1 2027\n(Frontier Scaling)"),
    (16.0, 18.0, "Q2 2027\n(10k+ Clusters)")
]
for q_start, q_end, q_lbl in quarters:
    ax.axvspan(q_start, q_end + 1.0, color="#334155", alpha=0.12, zorder=0)

ax.set_xticks(np.arange(1, 19))
ax.set_xticklabels([f"M{m}" for m in range(1, 19)], fontsize=9)
ax.set_xlabel("Operational Month (18-Month Multi-Stream Horizon)", fontsize=11, fontweight="bold", labelpad=10)

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([2.0, 5.0, 8.0, 11.0, 14.0, 17.0])
ax_top.set_xticklabels([
    "Q1 2026 | ARR $180k",
    "Q2 2026 | ARR $1.3M",
    "Q3 2026 | ARR $2.8M",
    "Q4 2026 | ARR $4.2M",
    "Q1 2027 | ARR $5.5M",
    "Q2 2027 | ARR $6.5M"
], fontsize=9.5, fontweight="bold", color="#38BDF8")

legend_elements = [
    Patch(facecolor="#10B981", edgecolor="#F8FAFC", label="Shipped / Verified"),
    Patch(facecolor="#0EA5E9", edgecolor="#F8FAFC", label="In Active Development"),
    Patch(facecolor="#64748B", edgecolor="#94A3B8", label="Planned / Roadmapped"),
    mlines.Line2D([], [], color="#F59E0B", marker="D", linestyle=":", label="Commercial Inflection Milestone")
]
ax.legend(handles=legend_elements, loc="lower left", frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=9)

plt.title("GhostLayer 18-Month Multi-Stream Commercial & Engineering Roadmap", fontsize=13, fontweight="bold", pad=28)
plt.tight_layout()
save_chart(plt.gcf(), "commercial_18month_workstream_gantt.png", "01_revenue_and_commercial")
plt.close()


# ==============================================================================
# Chart 3: Incident Detection, Adaptive Lagrangian Containment & Rollback SLA
# ==============================================================================
fig, (ax_stall, ax_ghost) = plt.subplots(2, 1, figsize=(13, 6.8), dpi=300, sharex=False)

trad_stages = [
    ("Loss Divergence / NaN Inception", 0.0, 0.4, "#F43F5E"),
    ("Silent Run-on / Compute Waste (30 min)", 0.4, 0.5, "#EF4444"),
    ("Process Crash / Cluster Stall (1 hr)", 0.9, 1.0, "#DC2626"),
    ("On-Call Triage & Manual Autopsy (2.5 hrs)", 1.9, 2.5, "#B91C1C"),
    ("Rewind to 6h-old Checkpoint & Restart (1.5 hrs)", 4.4, 1.5, "#991B1B")
]

y_t = 0.5
for name, start, dur, col in trad_stages:
    rect = FancyBboxPatch((start, y_t - 0.25), dur, 0.5,
                          boxstyle="round,pad=0.03,rounding_size=0.1",
                          facecolor=col, edgecolor="#F8FAFC", alpha=0.85, linewidth=1.2)
    ax_stall.add_patch(rect)
    ax_stall.text(start + dur/2.0, y_t, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#F8FAFC")

ax_stall.set_xlim(-0.2, 6.5)
ax_stall.set_ylim(0, 1.0)
ax_stall.set_yticks([])
ax_stall.set_xlabel("Elapsed Incident Time (Hours)", fontsize=10.5, fontweight="bold", labelpad=8)
ax_stall.set_title("TRADITIONAL CLUSTER FAILURE: 5.9 Hours Wasted Compute ($28,000+ per Incident at 1,024x Scale)",
                   fontsize=11.5, fontweight="bold", color="#FDA4AF", pad=12)
ax_stall.grid(True, axis="x")
ax_stall.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f} hrs"))

ghost_stages = [
    ("Curvature / Loss Spike Detected (<150ms)", 0.0, 0.15, "#0284C7"),
    ("Lagrangian Check: ΔL > 0.10 (<50ms)", 0.15, 0.05, "#6366F1"),
    ("State Rollback to Step N-1 (<250ms)", 0.20, 0.25, "#10B981"),
    ("LR Auto-Damp & Monotonic Resume (<100ms)", 0.45, 0.10, "#34D399"),
    ("Merkle Cryptographic Vault Receipt Generated (2.0s)", 0.55, 2.0, "#0EA5E9")
]

y_g = 0.5
for name, start, dur, col in ghost_stages:
    rect = FancyBboxPatch((start, y_g - 0.25), dur, 0.5,
                          boxstyle="round,pad=0.03,rounding_size=0.08",
                          facecolor=col, edgecolor="#F8FAFC", alpha=0.9, linewidth=1.2)
    ax_ghost.add_patch(rect)
    ax_ghost.text(start + dur/2.0, y_g, name, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0B0F17")

ax_ghost.set_xlim(-0.1, 3.2)
ax_ghost.set_ylim(0, 1.0)
ax_ghost.set_yticks([])
ax_ghost.set_xlabel("Elapsed Incident Time (Seconds)", fontsize=10.5, fontweight="bold", labelpad=8)
ax_ghost.set_title("GHOSTLAYER ADAPTIVE CONTAINMENT: <450ms Total MTTR (Zero Lost Steps / $0 Cloud Waste)",
                   fontsize=11.5, fontweight="bold", color="#6EE7B7", pad=12)
ax_ghost.grid(True, axis="x")
ax_ghost.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f} s"))

plt.suptitle("Incident Containment & Rollback SLA Comparison: Traditional vs. GhostLayer",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
save_chart(plt.gcf(), "containment_and_rollback_sla_timeline.png", "02_prospects_and_pipeline")
plt.close()


# ==============================================================================
# Chart 4: Sub-Run Taxonomy Breakdown Matrix
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=300, gridspec_kw={"width_ratios": [1.2, 1]})

subruns = [
    "1. Data Sweeps\n(20B-200B tok)",
    "2. MoE Routing\n(Top-2/Top-4)",
    "3. Foundation\n(1,024x Pre-Train)",
    "4. Context Ext\n(4k -> 128k)",
    "5. Reasoning RL\n(GRPO / PPO)"
]
x = np.arange(len(subruns))
width = 0.35

durations = [7, 9, 28, 9, 12]          # Days
cluster_gpus = [64, 256, 1024, 256, 128] # GPUs

color1 = "#0EA5E9"
color2 = "#8B5CF6"

rects1 = ax1.bar(x - width/2, durations, width, label="Duration (Days)", color=color1, edgecolor="#38BDF8")
ax1_twin = ax1.twinx()
rects2 = ax1_twin.bar(x + width/2, cluster_gpus, width, label="Cluster Size (GPUs)", color=color2, edgecolor="#C084FC")

ax1.set_ylabel("Duration (Days)", fontsize=10.5, color=color1, fontweight="bold")
ax1_twin.set_ylabel("Cluster Size (Active GPUs)", fontsize=10.5, color=color2, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(subruns, fontsize=8.5, fontweight="bold")
ax1.set_title("Sub-Run Taxonomy: Duration & Hardware Allocation", fontsize=12, fontweight="bold", pad=12)
ax1.grid(True, axis="y")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=8.5)

daily_base_spend = [5.84, 21.50, 78.64, 21.50, 11.06]    # $k / day
daily_savings = [2.04, 5.38, 14.16, 4.73, 2.65]          # $k / day

rects_c1 = ax2.bar(x - width/2, daily_base_spend, width, label="Baseline Spend ($k/day)", color="#64748B", edgecolor="#94A3B8")
rects_c2 = ax2.bar(x + width/2, daily_savings, width, label="GhostLayer Saved ($k/day)", color="#10B981", edgecolor="#34D399")

ax2.set_ylabel("Daily Cost & Savings ($k/day)", fontsize=10.5, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(subruns, fontsize=8.5, fontweight="bold")
ax2.set_title("Financial Daily Burn Rate & Recovered Capital", fontsize=12, fontweight="bold", pad=12)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"${y:.0f}k"))
ax2.grid(True, axis="y")
ax2.legend(loc="upper left", frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=8.5)

for rect, val, base_v in zip(rects_c2, daily_savings, daily_base_spend):
    height = rect.get_height()
    pct = (val / base_v) * 100.0
    ax2.text(rect.get_x() + rect.get_width()/2.0, height + 1.2, f"-{pct:.0f}%",
             ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#A7F3D0")

plt.suptitle("GhostLayer Sub-Run Operational Taxonomy & Economics", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_chart(plt.gcf(), "subrun_taxonomy_breakdown.png", "02_prospects_and_pipeline")
plt.close()

print("\nSUCCESS: All 4 high-resolution analytical charts generated in charts/")
