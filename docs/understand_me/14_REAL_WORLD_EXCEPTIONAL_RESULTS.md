# 📊 14: Empirical Test Results — Corrected

> [!WARNING]
> The CPU-only results below (SimpleCNN, MLP) were measured on a CPU PyTorch install (`cuda_available: false`). They demonstrate the report pipeline, not GPU optimization. Only the RTX A2000 GPU rows represent actual CUDA hardware measurements.

---

### 🏆 Physical Hardware Benchmarks (Empirical Runs)

| AI Workload | Hardware | Baseline Speed | Ghost Layer Speed | Speedup Gain | Verifier Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LLM Transformer (52.2M Params)** | RTX A2000 GPU | `~31 ms/step` | `~11 ms/step` | **~65% faster (single run)** | Blocked on multi-run (Safe Status: False × 5) |
| **LLM Transformer (52.2M, 5-trial)** | RTX A2000 GPU | `207–241 ms/step` | `86–87 ms/step` | **+58.4%–64.1% (mean 59.6%)** | Blocked — recommendation-only |
| **High-VRAM LLM (166.3M)** | RTX A2000 GPU | `1,299 ms/step` | `441 ms/step` | **+66.1%** | Blocked — loss delta exceeded 0.10 |
| **SimpleCNN (2-Layer Synthetic)** | CPU only (`cuda_available: false`) | `13.9 ms` | `11.1 ms` | +19.8% | CPU-only — no GPU verification |
| **MLP (Synthetic)** | CPU only (`cuda_available: false`) | `6.7 ms` | `6.5 ms` | +2.3% | CPU-only — no GPU verification |

> [!CAUTION]
> **Previous version of this file incorrectly labeled the 2-layer synthetic CNN as "ResNet-50 Vision AI".** The actual model is `custom/SimpleCNN-2Layer` — a 2-layer ConvNet on synthetic tensors. ResNet-50 has never been benchmarked by GhostLayer.

> [!NOTE]
> The "speedup" on RTX A2000 rows reflects the difference between FP32 and AMP-FP16 execution — a known PyTorch optimization that GhostLayer recommended but did not cause. The verifier blocked auto-apply on every multi-run trial.

### 📈 Simulated Cluster Projections (250 Scenarios)
> [!WARNING]
> The 250-scenario projection below is a **simulation using synthetic parameters**, not measured GPU data. It projects throughput improvements using the rules engine's heuristic model and should not be cited as empirical evidence.

Across 250 simulated fleet cluster scenarios (including Llama-3 and Mistral configurations), the rules engine projects an average +22.99% speedup. These are **untested projections**, not verified results.
