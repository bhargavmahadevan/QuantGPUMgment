# 🛡️ 13: Competitive Landscape & Product Positioning

> **Document Type:** Competitive Positioning & Market Analysis  
> **Core Category:** PyTorch Training Diagnostic & Telemetry Advisory Layer  
> **Key Focus:** Non-intrusive runtime observation and structured decision records vs. full platform lock-in.

---

## 1. The Actual Competitive Landscape

Training efficiency is a crowded, active market with well-funded competitors across different layers of the infrastructure stack:

| Category | Key Players | How They Operate | Where GhostLayer Differs |
| :--- | :--- | :--- | :--- |
| **Framework Rewrites** | Unsloth, Axolotl, LLaMA-Factory | Fast, hand-tuned Triton kernels for specific open models. Require rewriting training scripts into their framework. | **Zero Framework Lock-in:** GhostLayer attaches as a standard PyTorch callback (`GhostTrainerCallback`) without rewriting custom pipelines. |
| **Managed AI Cloud & Runtimes** | MosaicML (Databricks), Together AI, Modular (MAX) | Full-stack managed training clouds, compilers, and specialized inference/training runtimes. | **Infrastructure Agnostic:** Runs inside the client's existing AWS, GCP, CoreWeave, or on-prem cluster; does not require moving workloads. |
| **Cluster Orchestrators** | Run:ai (NVIDIA), CentML, Slurm, Ray | Low-level GPU scheduling, multi-node slicing, and dynamic hardware resource allocation. | **Workload-Level Telemetry:** Focuses on in-process step timing, memory fragmentation, and DataLoader starvation, not raw hardware virtualization. |
| **ML Observability** | Weights & Biases, MLflow, TensorBoard | General experiment tracking, hyperparameter logging, and metric visualization. | **Actionable Decision Records:** Observability shows metric charts; GhostLayer generates structured configuration checklists (e.g., pin_memory, worker counts, Inductor compile). |

---

## 2. GhostLayer's Strategic Wedge

GhostLayer does not compete by building a proprietary cloud or custom kernel compiler. Its commercial wedge is defined by three deliberate constraints:

1. **Lightweight & Non-Intrusive:** Under 0.5% runtime overhead, standard PyTorch hook API, and default read-only execution (`ConsentLevel.AUDIT_ONLY`).
2. **Deterministic Diagnostic Checklists:** Produces structured decision records identifying common training inefficiencies (DataLoader I/O starvation, VRAM fragmentation, precision mismatches).
3. **Frictionless Onboarding:** $2,500 flat-fee assisted audit or $199–$999/mo SaaS subscription, avoiding $100k+ enterprise platform lock-in.

---

## 3. Product Scope & Operational Boundaries

* **What GhostLayer Is:** An open-source, read-only telemetry hook and diagnostic decision engine for PyTorch training workloads.
* **What GhostLayer Is NOT:** GhostLayer is **not** an autonomous agent that alters production model weights or cluster schedules without human sign-off, nor is it a proprietary replacement for CUDA kernels.
