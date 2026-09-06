# Real LLM Empirical Benchmark & Marginal Ablation Receipt

**Hardware Platform:** NVIDIA RTX A2000 8GB Laptop GPU
**Model Target Architecture:** `PyTorchLLMModel-6L-512d (52.2M params)`
**Measurement Protocol:** Synchronized CUDA Events, 10 Warmup Discarded, 50 Timed Steps

### Marginal Ablation Matrix

| Optimization Layer | Step Time (ms) | Marginal Speedup | Cumulative Speedup | Primary Driver |
| :--- | ---: | ---: | ---: | :--- |
| **0. Baseline (FP32)** | 234.84 ms | — | 0.0% | Unoptimized reference |
| **1. + Pinned Memory** | 235.55 ms | -0.3% | -0.3% | Host-to-Device transfer pinning |
| **2. + SDPA Kernel** | 208.11 ms | +11.6% | +11.4% | FlashAttention HBM roundtrip fusion |
| **3. + Mixed Precision** | 85.06 ms | +59.1% | +63.8% | Tensor Core acceleration |

### Statistical Trajectory Verification

- **Safety Status:** `True` (`AUTO_APPLIED`)
- **Welch's Two-Sample t-Test:** `p = 0.999241` (t-stat: `-0.001`)
- **Max Loss Delta:** `0.0011`
- **Relative Loss Shift:** `0.0000`
- **Lagrangian Dual Multiplier (λ):** `0.0000`
