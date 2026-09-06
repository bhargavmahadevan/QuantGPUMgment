# 🚀 03: How Distributed Scaling Works

> **Document Type:** Architecture & Scaling Overview  
> **Target:** Engineering Teams Scaling from Single Workstations to Multi-GPU Nodes

---

### 🔌 Lightweight Integration Across Nodes

As training workloads scale from a single developer workstation to multi-GPU nodes, manual profiling with heavyweight tools becomes increasingly complex:

1. **Standard Callback Hook:** Integrates via standard PyTorch callbacks (`GhostTrainerCallback` for HuggingFace Trainer, `GhostLightningCallback` for PyTorch Lightning, or `GhostWatcherHook` context manager).
2. **Low Telemetry Overhead:** Collects step times, memory allocations, and I/O metrics in $<0.5\%$ runtime overhead per step.
3. **Rank-Aware Collection:** In distributed environments (DDP/FSDP), primary rank 0 collects aggregate telemetry to output consolidated diagnostic reports without swamping worker nodes.

---

### 📈 Diagnostic Scaling Across Hardware Formats

| Workload Scale | Traditional Manual Profiling | GhostLayer Automated Diagnostic Approach |
| :--- | :--- | :--- |
| **1 GPU (Workstation)** | Manual `torch.cuda` logging & inspection | Local CLI output and terminal audit card |
| **8 GPUs (Single Node)** | Multi-GB Chrome trace capture via `torch.profiler` | Aggregated step-timing and DataLoader starvation metrics |
| **Multi-Node Clusters** | Complex distributed tracing and log parsing | Centralized diagnostic report with actionable configuration checklists |

> [!NOTE]
> GhostLayer operates as an observational telemetry and advisory layer. It does not replace cluster orchestration (Slurm/Kubernetes) or hardware virtualization tools (Run:ai).
