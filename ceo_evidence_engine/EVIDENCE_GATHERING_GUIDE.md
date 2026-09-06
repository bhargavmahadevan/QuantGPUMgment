# GhostLayer CEO Evidence Engine — Empirical Evidence Gathering Guide

> **Absolute Realism Policy:** This guide only describes how to gather **real, physically-measured benchmark data**. All collected metrics must be clearly labeled with execution mode (`LIVE_MEASURED_RUN`, `PUBLISHED_REFERENCE_BASELINE`, or `NOT_YET_TESTED`). Never present estimated or projected figures as physically measured results.

---

## 🎯 Purpose

This guide walks an ML Platform Engineer or technical evaluator through the complete workflow for gathering verifiable, SHA-256-auditable benchmark data from physical GPU hardware using the GhostLayer CEO Evidence Engine.

---

## Step 0 — Detect Your Hardware Target

Before running any benchmarks, identify what GPU hardware you have available:

```bash
# Auto-detect hardware target (CUDA, MPS, or CPU fallback)
python -m ceo_evidence_engine.real_hardware_benchmarks --detect
```

Expected output will show one of three execution modes:

| Detected Hardware | Execution Mode | Description |
|:---|:---|:---|
| NVIDIA CUDA GPU | `LIVE_MEASURED_RUN` | Physically measured on your GPU |
| Apple Silicon (MPS) | `LIVE_MEASURED_RUN` | Physically measured on M-series GPU |
| No GPU / CPU only | `PUBLISHED_REFERENCE_BASELINE` | Loaded from `hardware_reference_database.json` |

> [!IMPORTANT]
> Only `LIVE_MEASURED_RUN` results constitute empirical evidence suitable for investor or enterprise audit purposes. `PUBLISHED_REFERENCE_BASELINE` figures are clearly labeled reference projections.

---

## Step 1 — Prepare a GPU-Enabled Environment

### Option A: Local Machine (NVIDIA GPU)

```bash
# Verify CUDA is accessible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install GhostLayer in editable mode
pip install -e .
```

### Option B: Lambda Labs Cloud (Recommended — A100 / H100)

```bash
# Provision a Lambda GPU instance (e.g., A100 80GB SXM4)
# SSH into the instance, then:
git clone <your-repo>
cd QuantGPUMgment
pip install -e .

# Verify hardware detection
python -m ceo_evidence_engine.real_hardware_benchmarks --detect
```

### Option C: RunPod / Vast.ai (Budget NVIDIA RTX)

```bash
# Use the provided Docker template: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
pip install -e .
python -m ceo_evidence_engine.real_hardware_benchmarks --detect
```

### Option D: Hugging Face Cloud Inference API (No GPU Required)

> **Status Notice**: Live cloud inference via `hf_cloud_inference_client.py` has **not** been executed in this repository environment because `HF_TOKEN` is not provisioned. Without a token, the runner executes in transparent simulated mode (`SIMULATED_CLOUD_INFERENCE_RUN`). The instructions below allow anyone with an active token to execute live.

```bash
# Set HF_TOKEN environment variable (free tier works for benchmarking)
export HF_TOKEN=<your_huggingface_token>  # Linux/macOS
$env:HF_TOKEN="<your_huggingface_token>"   # Windows PowerShell

# Run cloud inference benchmark
python -m ceo_evidence_engine.hf_cloud_inference_client --provider hf-inference
```

---

## Step 2 — Run the Hardware Benchmark Suite

```bash
# Benchmark a specific model (generates SHA-256 verified JSON artifact)
python -m ceo_evidence_engine.real_hardware_benchmarks --model llama3-8b

# Benchmark all models and also export HTML audit cards
python -m ceo_evidence_engine.real_hardware_benchmarks --model all --html

# Available model keys:
#   llama3-8b           — LLM token throughput benchmark
#   mistral-7b          — Mistral 7B reference benchmark
#   resnet-50           — Vision model inference benchmark
#   small-cnn-empirical — Small CNN empirical training run
```

---

## Step 3 — Locate the Evidence Artifacts

All outputs are saved to `ceo_evidence_engine/sample_outputs/`:

| File | Description |
|:---|:---|
| `hardware_evidence_<model>.json` | Machine-readable JSON artifact with SHA-256 audit hash |
| `hardware_audit_card_<model>.html` | Self-contained HTML audit card (open in browser) |
| `evidence_report_RUN-XXXXXX.md` | Executive Markdown audit report |

---

## Step 4 — Verify the SHA-256 Cryptographic Audit Hash

Each benchmark JSON artifact contains a 64-character SHA-256 hash of the payload:

```python
import json, hashlib

with open("ceo_evidence_engine/sample_outputs/hardware_evidence_llama3-8b.json") as f:
    artifact = json.load(f)

# Re-compute the hash from the raw payload
raw_payload = {
    "timestamp": artifact["metadata"]["timestamp"],
    "model_key": artifact["metadata"]["model_key"],
    "results": artifact["results"]
}
recomputed = hashlib.sha256(
    json.dumps(raw_payload, sort_keys=True).encode("utf-8")
).hexdigest()

stored = artifact["metadata"]["reproducibility"]["sha256_audit_hash"]
print("Hash match:", recomputed == stored)  # Must be True
print("SHA-256:", stored)
```

---

## Step 5 — Run the Full Verification Test Suite

```bash
# Verify all 10 CEO Evidence Engine tests pass (SHA-256, A/B fleet, HF Cloud, e2e trace)
pytest ceo_evidence_engine/test_evidence_suite.py -v

# Run the full project test suite (336 tests)
pytest tests/ -v
```

---

## Step 6 — Recommended Hardware Targets for Enterprise Evidence

| Hardware | Cloud Provider | Approximate Cost | Notes |
|:---|:---|:---|:---|
| NVIDIA H100 80GB SXM5 | Lambda Labs, CoreWeave | ~$2.50/hr | Highest throughput, best for LLM benchmarks |
| NVIDIA A100 80GB SXM4 | AWS (p4d), GCP (a2), Lambda | ~$1.80/hr | Industry standard reference hardware |
| NVIDIA RTX 4090 24GB | Vast.ai, RunPod | ~$0.50/hr | Budget option for initial validation |
| NVIDIA RTX A2000 8GB | Local workstation | Hardware owned | Matches existing empirical runs |
| Apple M3 Max (MPS) | Local MacBook Pro | Hardware owned | MPS execution path validation |

---

## Step 7 — Attaching Evidence to Investor Materials

When referencing benchmark data in pitch decks or investor materials:

1. Always cite the **execution mode** (`LIVE_MEASURED_RUN` vs `PUBLISHED_REFERENCE_BASELINE`)
2. Include the **SHA-256 audit hash** as a cryptographic provenance receipt
3. Clearly label the **hardware model** and **date of measurement**
4. For projected fleet-scale numbers, explicitly state: *"Projection based on analytical_kernel_estimator methodology with 0.88 confidence score. Not physically measured on that hardware configuration."*

---

## Contacts & Support

For technical integration support:
- Email: `engineering@ghostlayer.ai`
- Pilot evaluation protocol: see [PILOT_EVALUATION_GUIDE.md](./PILOT_EVALUATION_GUIDE.md)
