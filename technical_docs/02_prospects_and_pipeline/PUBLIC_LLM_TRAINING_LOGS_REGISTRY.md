# Public LLM GPU Training Logs & Failure Mode Mapping Registry

> [!WARNING]
> **Academic Reference & Retrospective Analysis Notice:**  
> This registry catalogs documented failure pathologies from published open-science frontier training logs and maps them to GhostLayer's diagnostic heuristic rules.  
> **GhostLayer was NOT deployed on these historical training runs.** This registry exists as a technical reference mapping known distributed training pathologies to diagnostic detection mechanisms.

---

## 1. Public Literature Failure Pathologies & Mapped Diagnostic Rules

| Initiative / Model | Scale & Topology (Literature) | Documented Public Artifacts | Documented Training Pathologies | Mapped Diagnostic Rule in GhostLayer |
| :--- | :--- | :--- | :--- | :--- |
| **BigScience BLOOM (176B)** | 384 × 80GB A100 (Jean Zay) | TensorBoard traces, public markdown logbook | • BF16 optimizer state desynchronization<br>• Dynamic loss scaling collapse<br>• Slurm node heartbeat timeouts | `RULE_MIXED_PRECISION` & `RULE_CUDA_ALLOCATOR_TUNING` flag precision and allocator state anomalies. |
| **Meta OPT-175B** | 992 × 80GB A100 | 100+ page chronological logbook (`metaseq`) | • Hardware resets & loss spikes<br>• NCCL InfiniBand silent hangs | `RULE_ROLLBACK_GUARD` triggers alert on loss-shift delta ($\Delta L > 0.10$). |
| **AI2 OLMo & OLMoE (1B–32B)** | Multi-node A100/H100 | Public W&B dashboards, checkpoints | • Multi-node gradient norm variance<br>• DataLoader worker queue starvation | `RULE_DATALOADER_WORKERS` flags when I/O stall ratio exceeds 8%. |
| **LLM360 (Amber 7B, K2 65B)** | DGX A100/H100 clusters | 360 intermediate weight revisions | • Mid-training loss divergence<br>• Layer-wise gradient norm drift | `RULE_ROLLBACK_GUARD` monitors loss stability bounds. |
| **Hugging Face SmolLM (135M–3B)** | H100 clusters (Nanotron) | W&B telemetry, Smol Training Playbook | • MFU degradation from un-fused kernels<br>• LR schedule transition shocks | `RULE_GRAPHIFY_TORCH_COMPILE` recommends kernel fusion. |
| **EleutherAI Pythia (70M–12B)** | Multi-node GPU clusters | 154 checkpoint series per scale | • Step-time variance and batch pipeline latency | `RULE_ASYNC_TENSOR_FLOW` checks pinned host-to-device transfers. |

---

## 2. Public Literature Source Links

1. **BigScience BLOOM Logbook:** [Hugging Face `bigscience/tr11-176B-logs`](https://huggingface.co/bigscience/tr11-176B-logs)
2. **Meta OPT Chronicles:** [Meta OPT Logbook in `facebookresearch/metaseq`](https://github.com/facebookresearch/metaseq)
3. **AI2 OLMo Dashboards:** [AI2 OLMo GitHub](https://github.com/allenai/OLMo)
4. **LLM360 Transparency Hub:** [LLM360 GitHub](https://github.com/LLM360)
5. **Hugging Face SmolLM Playbook:** [Smol Training Playbook](https://gist.github.com/jph00/3c97a2c6c5075c4e7b98faae634b033a)
6. **EleutherAI Pythia:** [EleutherAI Hugging Face](https://huggingface.co/EleutherAI)
