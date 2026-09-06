# Real Multi-Run GPU Empirical Benchmark Receipt: EXP-A2000-LLM-001

- **Experiment Identifier:** `EXP-A2000-LLM-001`
- **Hardware Platform:** Physical NVIDIA RTX A2000 8GB Laptop GPU (PCIe ID / Device 0)
- **Workload:** 52.2M-parameter Transformer (6 layers, 8 attention heads, hidden dim 512, sequence length 512, batch size 4)
- **Number of Consecutive Physical Trials:** 5
- **Intervention Type:** Configuration Bundle (Baseline: FP32, unpinned, eager attention → Candidate: AMP-FP16, PyTorch SDPA, pinned memory)
- **Operational Verdict:** **RECOMMENDATION_ONLY** (Auto-apply blocked on all 5 trials by the loss-shift proxy guardrail: pointwise delta > 0.10)

---

## 1. Measured Multi-Trial Execution Data

| Trial # | Baseline (ms/step) | Baseline (tok/s) | Ghost Layer Candidate (ms/step) | Ghost Layer Candidate (tok/s) | Speedup Gain | Loss Shift Proxy Delta | Safe Status (Auto-Apply Gate) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trial 1** | 207.68 ms | 9,861 tok/s | 86.62 ms | 23,643 tok/s | **+58.29%** | $\Delta = 0.142$ | `False (Blocked: delta > 0.10)` |
| **Trial 2** | 209.00 ms | 9,799 tok/s | 86.95 ms | 23,552 tok/s | **+58.39%** | $\Delta = 0.138$ | `False (Blocked: delta > 0.10)` |
| **Trial 3** | 209.66 ms | 9,768 tok/s | 86.96 ms | 23,551 tok/s | **+58.52%** | $\Delta = 0.145$ | `False (Blocked: delta > 0.10)` |
| **Trial 4** | 210.34 ms | 9,737 tok/s | 87.71 ms | 23,349 tok/s | **+58.30%** | $\Delta = 0.140$ | `False (Blocked: delta > 0.10)` |
| **Trial 5** | 210.55 ms | 9,727 tok/s | 87.24 ms | 23,475 tok/s | **+58.56%** | $\Delta = 0.139$ | `False (Blocked: delta > 0.10)` |

---

## 2. Empirical Summary Statistics
- **Mean Latency (Baseline):** `209.45 ms/step` (std dev: 1.15 ms)
- **Mean Latency (Candidate):** `87.10 ms/step` (std dev: 0.42 ms)
- **Mean Throughput Speedup:** **`+58.41%`**
- **Speedup Range:** `+58.29%` to `+58.56%`
- **Speedup Standard Deviation:** `0.11%`
- **Throughput Stability Assessment:** Highly Stable / Physically Replicated
- **Quality & Safety Gate:** **BLOCKED (Safe Status: False)**. Pointwise relative loss shift exceeds 0.10 threshold.
- **Evidentiary Classification:** `E4_REPLICATED_PHYSICAL` throughput observation; `RECOMMENDATION_ONLY` deployment status.

---

## 3. Scientific Integrity & Attribution Note
The +58.41% latency reduction is physically measured on real hardware. However, GhostLayer's safety policy correctly blocked automatic runtime deployment because the uncalibrated loss trajectory delta exceeded 0.10. GhostLayer provided diagnostic identification and parameter selection, not verified automated execution.
