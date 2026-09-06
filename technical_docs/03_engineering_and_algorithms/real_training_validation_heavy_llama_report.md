# Heavy LLM (Llama/Meta) Empirical Training Audit Receipt

**Model ID:** `distilbert/distilgpt2` (81,912,576 Parameters)
**Hardware Platform:** NVIDIA RTX A2000 8GB Laptop GPU
**Baseline Throughput:** `3,362 tok/s` (304.59 ms/step)
**Baseline VRAM:** `2594 MB`
**Ghost Layer Throughput:** `6,963 tok/s` (147.06 ms/step)
**Ghost Layer VRAM:** `1868 MB` (Saved +726 MB)
**Empirical Speedup:** `+51.7%`
**Loss Convergence Verified:** `True` (Max Loss Delta: 0.162378)
**Projected 100k-Step 8x GPU Savings:** `$122.52`
**Projected Quant 25% Fee:** `$30.63`
