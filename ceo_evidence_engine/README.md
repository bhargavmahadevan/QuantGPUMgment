# GhostLayer CEO Evidence Engine (`ceo_evidence_engine`)

The **CEO Evidence Engine** is an executable evidence collection and verification suite built for ML Platform Teams and enterprise evaluators.

---

## 🚀 Quickstart CLI Commands

### 1. Run Baseline vs. GhostLayer Empirical Benchmark
```bash
python -m ceo_evidence_engine.benchmark_runner --workload llama3-8b --mode shadow --steps 100
```

### 2. Generate Executive Markdown Audit Report
```bash
python -m ceo_evidence_engine.report_generator --json_file ceo_evidence_engine/evidence_run_RUN-XXXXXX.json
```

### 3. Run 48-Hour Shadow-Mode Audit Simulator
```bash
python -m ceo_evidence_engine.shadow_audit_collector
```

### 4. Run Split-Fleet A/B Control Verification
```bash
python -m ceo_evidence_engine.split_fleet_verifier
```

### 5. Run Automated Test Suite
```bash
pytest ceo_evidence_engine/test_evidence_suite.py
```

---

## 📦 How to Get Benchmark Models (Llama 3 8B, Mistral 7B, ResNet-50, SDXL)

> [!NOTE]
> **ResNet-50 has never been benchmarked by GhostLayer.** A previous version of the results table incorrectly labeled a 2-layer synthetic CNN as "ResNet-50". The models below are available for future benchmarking.

### Install Prerequisites
```bash
pip install torch torchvision transformers diffusers accelerate huggingface_hub
```

1. **ResNet-50 (Vision Baseline — NOT YET TESTED)**: Auto-downloads on first PyTorch run via `torchvision.models.resnet50(weights="DEFAULT")`.
2. **Mistral 7B (Open Access LLM)**: Direct download from HuggingFace via `AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")`.
3. **Meta Llama 3 8B (Gated Open Weights)**: Accept Meta license at [huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), run `huggingface-cli login`, then load via `AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")`.
4. **SDXL Base 1.0 (Diffusion Baseline)**: Direct download via `DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")`.

### Pre-fetch Utility Script
```bash
python -m ceo_evidence_engine.download_models
```

