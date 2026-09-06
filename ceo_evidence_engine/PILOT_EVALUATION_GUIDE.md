# GhostLayer Design Partner & Pilot Evaluation Guide

This guide outlines the 5-step protocol for ML Platform Engineering Teams and enterprise design partners to independently evaluate GhostLayer on their own GPU infrastructure.

---

## 🎯 Pilot Evaluation Objectives

1. **Passive Observation & Telemetry Profiling:** Detect hidden FlashAttention un-fused kernel stalls and CUDA VRAM page fragmentation without modifying existing training pipelines.
2. **Safety Guardrail Verification:** Confirm sub-millisecond parameter rollbacks when loss divergence thresholds (`KL > 0.05`) are exceeded.
3. **Empirical Throughput Acceleration:** Measure token throughput (tokens/sec) and peak VRAM allocation deltas under real production training workloads (Llama 3, Mistral, Vision Transformers).

---

## 🚀 Step-by-Step Pilot Deployment

### Step 1: Install Package & Verify Environment Integration
```bash
# Clone & install GhostLayer package
pip install -e .

# Run pre-flight release integration verification
python -m ceo_evidence_engine.benchmark_runner --verify
```

---

### Step 2: Attach Non-Invasive Context Hook (`GhostWatcherHook`)
Attach the single-line context hook to your PyTorch training script:

```python
from ghost_layer.hooks import GhostWatcherHook

# Wrap existing PyTorch training loop (Passive 48-Hour Shadow Mode)
with GhostWatcherHook(gpu_memory_mb=81920.0) as hook:
    for step, batch in enumerate(dataloader):
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        hook.on_step_end(
            gpu_util_pct=88.0,
            gpu_mem_used_mb=42100.0,
            loss=loss.item(),
            mixed_precision="fp32"
        )
```

---

### Step 3: Execute Baseline & Live Benchmarks
Run reproducible baseline vs. GhostLayer benchmarks on your target hardware:

```bash
# Benchmark on local/cloud GPU hardware
python -m ceo_evidence_engine.real_hardware_benchmarks --model llama3-8b

# Benchmark Hugging Face Cloud Inference API (Requires HF_TOKEN)
# Note: In environments without HF_TOKEN set, this executes in offline simulation mode.
# Live cloud execution is UNEXECUTED until a valid HF_TOKEN is provided.
$env:HF_TOKEN="your_token_here"  # PowerShell
python -m ceo_evidence_engine.hf_cloud_inference_client --provider hf-inference
```

---

### Step 4: Review Auditable Evidence Artifacts & Cryptographic Hashes
Inspect generated JSON evidence artifacts inside `ceo_evidence_engine/sample_outputs/`:

- `hardware_evidence_llama3-8b.json` (Includes full environment telemetry & 64-char SHA-256 audit hash)
- `evidence_report_RUN-XXXXXX.md` (Executive Markdown Audit Report with ROI receipts and verification logs)

---

### Step 5: Independent Verification Checklist

| Verification Metric | Target Threshold | Validation Method |
| :--- | :--- | :--- |
| **Telemetry Overhead** | `< 1.5%` runtime penalty | Measure step latency delta before vs after hook attachment |
| **Convergence Safety** | `KL Divergence < 0.05` | Verify zero loss curve degradation |
| **Rollback Latency** | `< 1.0 ms` | Trigger synthetic loss spike and measure auto-revert latency |
| **Decision Confidence (heuristic)** | `> 0.80` | Check `calculate_evidence_confidence()` heuristic in audit report |

---

*For technical support or custom cluster topology integration, contact `engineering@ghostlayer.ai`.*
