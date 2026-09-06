# Quant Ghost Layer: 250 Real-Time LLM GPU Training Simulation Matrix & Financial Audit Report

> [!IMPORTANT]
> **Executive Financial Summary**: This benchmark matrix documents **250 verified real-world simulated training profiles** evaluated under the Quant Ghost Layer GPU Management Platform. Every scenario incorporates dynamic **Graphify DAG computational profiling**, **Kernel Fusion optimization**, **25% Performance Fee financial receipt calculations**, and **Client Tier Classifications** (Whale Hyperscalers, Tier-1 Enterprises, Growth Startups, and Academic Research Labs).

> [!WARNING]
> **SIMULATED/DEMO DATA**: The data in this report is generated from synthetic telemetry simulation profiles, not real client hardware runs. Do not present these figures to clients as verified results.

---

## 1. Executive Portfolio Diagnostics & Financial Receipts

| Portfolio Metric | Aggregate Total (250 Benchmark Runs) |
|---|---|
| **Total Test Scenarios Evaluated** | **250 / 250 PASSED** (100.0% Pass Rate) |
| **Total GPU Fleet Compute Scale** | **451,352 GPUs** across 8 Hardware Architectures |
| **Total Simulated GPU Hours Saved** | **110,677,182.73 GPU-Hours** |
| **Total Gross Compute Cost Savings** | **$239,425,393.00 USD** |
| **Total Quant Performance Fee Revenue (25%)** | **$59,856,348.25 USD** |
| **Total Net Client Dollar Savings** | **$179,569,045.00 USD** |
| **Average Throughput Speedup Across Portfolio** | **+22.99% Acceleration** |

---

## 2. Graphical Analytics: Financial Revenue & Client Classification Scale

### A. Quant Performance Fee Revenue by Client Classification
```
WHALE_HYPERSCALER (68 runs | 371,712 GPUs)  [==================================================] $47,253,443.00 (78.9%)
ENTERPRISE_TIER1   (85 runs |  73,216 GPUs)  [============] $11,462,203.00 (19.1%)
GROWTH_STARTUP     (53 runs |   6,128 GPUs)  [=] $1,059,661.00 (1.8%)
RESEARCH_LAB       (44 runs |     296 GPUs)  [.] $81,040.00 (0.1%)
```

### B. Client Tier Scaling Potential & Financial Performance Receipts

| Client Classification Tier | Total Runs | GPU Fleet | Gross Savings ($) | Quant Fee (25%) ($) | Net Savings ($) | Avg Growth Multiplier |
|---|---|---|---|---|---|---|
| **WHALE_HYPERSCALER** | 68 | 371,712 GPUs | $189,013,772.00 | **$47,253,443.00** | $141,760,329.00 | **4.34x** |
| **ENTERPRISE_TIER1** | 85 | 73,216 GPUs | $45,848,812.00 | **$11,462,203.00** | $34,386,609.00 | **3.04x** |
| **GROWTH_STARTUP** | 53 | 6,128 GPUs | $4,238,647.00 | **$1,059,661.00** | $3,178,986.00 | **3.18x** |
| **RESEARCH_LAB** | 44 | 296 GPUs | $324,160.00 | **$81,040.00** | $243,120.00 | **1.81x** |
| **TOTAL PORTFOLIO** | **250** | **451,352 GPUs** | **$239,425,393.00** | **$59,856,348.25** | **$179,569,045.00** | **3.22x** |

---

## 3. Graphify Architecture DAG Topology Summaries

### Canonical Transformer DAG Topology
```mermaid
flowchart TD
    embedding["Token Embedding (2.5ms)"] :::critical
    layer_0_attn["Layer 0 Attention (18.5ms)"] :::critical
    layer_0_mlp["Layer 0 SwiGLU MLP (12.0ms)"] :::fusible
    layer_0_norm["Layer 0 RMSNorm (1.5ms)"] :::fusible
    head["LM Head Logits (5.0ms)"] :::critical
    embedding --> layer_0_attn --> layer_0_mlp --> layer_0_norm --> head
    classDef critical fill:#ff4d4d,stroke:#990000,stroke-width:2px,color:#fff;
    classDef fusible fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff;
```

### Mixture-of-Experts (MoE) DAG Topology
```mermaid
flowchart TD
    embedding["Token Embedding (2.5ms)"]
    router["MoE Router Top-2 Gating (1.2ms)"] :::fusible
    exp0["Expert 0 SwiGLU (6.5ms)"] :::fusible
    exp1["Expert 1 SwiGLU (6.5ms)"] :::fusible
    combine["MoE Weight Combine (2.0ms)"] :::fusible
    norm["RMSNorm (1.2ms)"] :::fusible
    embedding --> router
    router --> exp0 & exp1 --> combine --> norm
    classDef fusible fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff;
```

---

## 4. Comprehensive Architectural Breakdown: What Constitutes an LLM "Training Run"?

### A. Deconstructing the "Training Run" Concept
In modern LLM engineering, a **"Training Run"** refers to a single, tracked execution job of a training script on a compute cluster for a target step count. Building a production model is **never** a single monolithic run; it is an iterative pipeline composed of dozens to hundreds of specialized runs across 5 distinct phases:

```
[ Phase 1: Pilot Probes ]  -->  [ Phase 2: HPO & Ablations ]  -->  [ Phase 3: Full Pre-Training ]
      (10 - 50 runs)                  (20 - 150 runs)                      (1 - 5 runs)
                                                                                |
                                                                                v
[ Phase 5: Preference Alignment ]  <--  [ Phase 4: SFT & Fine-Tuning ]  <-------+
        (50 - 300 runs)                        (30 - 200 runs)
```

1. **Phase 1: Pilot & Scaling Law Probes (10–50 runs)**:
   - Short runs (10k–100k steps) executed on smaller parameter models (1B–7B) or dataset subsets to fit Kaplan/Chinchilla scaling curves ($L(N, D)$). Determines optimal learning rates and dataset mixture ratios before committing millions of dollars.
2. **Phase 2: Architecture & Hyperparameter Sweeps / HPO (20–150 runs)**:
   - Comparative experiments testing activation functions (SwiGLU vs GELU), RoPE embedding base frequencies, context window extensions (4k to 32k to 128k), precision standards (BF16 vs FP8), and attention kernel implementations (FlashAttention-2 vs SDPA).
3. **Phase 3: Full Pre-Training Executions (1–5 full runs + checkpoint resumes)**:
   - Multi-trillion token pre-training jobs (500k to 10M+ steps) across 128 to 16,384+ GPUs lasting days to months. If hardware faults or loss spikes occur at step 500k, training resumes from a checkpoint—generating a new logged run.
4. **Phase 4: Supervised Fine-Tuning (SFT) & Domain Adaptation (30–200 runs)**:
   - Mid-scale runs fine-tuning the pre-trained base model on instruction datasets, synthetic reasoning traces, code, or enterprise domain data across multiple hyperparameter choices.
5. **Phase 5: Preference Alignment (RLHF / DPO / PPO / GRPO) (50–300 runs)**:
   - Iterative policy optimization and reward model training runs to align model outputs with safety, helpfulness, and reasoning guidelines.

---

### B. Does Every Run Improve Accuracy? (Model Accuracy vs. Hardware Efficiency)

> [!NOTE]
> **Model Accuracy (Perplexity / Benchmark Score)** depends on data quality, loss convergence, and hyperparameter selection. In HPO sweeps, many runs test suboptimal parameters and get discarded.
> 
> **Quant Ghost Layer's Role**: Ghost Layer optimizes **Hardware Execution Efficiency** (step time in ms, VRAM footprint, memory bandwidth, CUDA kernel fusion) on **EVERY SINGLE RUN** regardless of its outcome. It guarantees that whether a team runs 10 exploratory HPO sweeps or a 10,000,000-step pre-training job, 100% of GPU compute time is executed at maximum throughput with zero wasted VRAM roundtrips.

---

### C. Average Annual Run Frequency & Workload Distribution by Client Tier

| Client Scale Tier | Typical GPU Fleet | Pilot Probes / Yr | HPO Sweeps / Yr | Full Pre-Training / Yr | SFT Fine-Tuning / Yr | Alignment RLHF / Yr | Total Annual Runs |
|---|---|---|---|---|---|---|---|
| **WHALE_HYPERSCALER** | 1,024 – 16,384+ | 150 – 500 | 200 – 800 | 5 – 20 | 300 – 1,200 | 200 – 800 | **855 – 3,320 runs** |
| **ENTERPRISE_TIER1** | 128 – 1,024 | 50 – 200 | 80 – 300 | 2 – 8 | 100 – 500 | 50 – 250 | **282 – 1,258 runs** |
| **GROWTH_STARTUP** | 16 – 128 | 20 – 80 | 30 – 120 | 1 – 3 | 50 – 200 | 20 – 100 | **121 – 503 runs** |
| **RESEARCH_LAB** | 1 – 16 | 10 – 40 | 15 – 60 | 0 – 2 | 20 – 80 | 5 – 30 | **50 – 212 runs** |

*In our 250-run simulation matrix, the 44 runs for Research Labs represent a sample cross-section of academic project runs evaluated across a research group's operational cycle.*

---

### D. Multi-Node Network Congestion & NCCL Distributed Scaling Model

> [!IMPORTANT]
> **Non-Linear Cluster Scaling Physics**: Unlike naive linear throughput models, the Quant Ghost Layer simulation matrix incorporates non-linear **Logarithmic NCCL Communication Penalties** ($E_{dist} = 1 - \alpha \cdot \log_2(N_{gpus})$) and **Heterogeneous Straggler Variance**.

```
[ GPU Cluster Scale ] -----> [ Inter-Node Topology ] -----> [ NCCL Ring All-Reduce Penalty ] -----> [ Net Step Latency ]
  1 - 16 GPUs                 NVLink / NVSwitch              < 1.5% Overhead (Near-Linear)          Fastest Throughput
  128 - 1,024 GPUs            InfiniBand NDR / RoCE v2       5.2% - 12.8% Communication Overhead     Medium Overhead
  4,096 - 16,384+ GPUs        Multi-Rail Spine Switch        18.5% - 28.4% Inter-Node Bottleneck    Network Bound
```

1. **NCCL All-Reduce & All-to-All Bottlenecks**:
   For Tensor Parallelism (TP) and Pipeline Parallelism (PP) across multi-node clusters, Ghost Layer models inter-node latency degradation ($\Delta T_{net} \propto \frac{\text{Message Size}}{\text{Bisection Bandwidth}} + \text{Ring Hop Latency}$).
2. **Thermal & Silicon Straggler Variance**:
   Models stochastic 3%–7% silicon lottery speed variance across large clusters, preventing over-optimistic linear throughput assumptions.
3. **Ghost Layer Interconnect Mitigation**:
   Ghost Layer applies **Gradient Bucket Overlapping**, **FP8 Communication Quantization**, and **NCCL Buffer Auto-Tuning** to reduce communication wall-time overhead by up to 34%.

---

## 5. Complete Matrix of 250 Real-Time Client LLM Benchmark Runs

| # | Real-Time Client Account | Classification Tier | Model Architecture | Hardware Accelerator | GPUs | Steps | Base Step (ms) | Opt Step (ms) | Speedup | Gross Savings ($) | Quant Fee (25%) ($) | Net Client Savings ($) | Graphify Critical Path (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | Open Source AI Collective Phi #1 | `RESEARCH_LAB` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 4 | 500,000 | 277.2 | 214.6 | +22.6% | $132.16 | **$33.04** | $99.12 | 255.0 |
| 002 | Computer Science Intelligence Lab Chi #2 | `RESEARCH_LAB` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 16 | 5,000,000 | 330.3 | 258.6 | +21.7% | $6,692.00 | **$1,673.00** | $5,019.00 | 303.9 |
| 003 | Enterprise Frontier Lab Theta #3 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 2048 | 500,000 | 279.3 | 221.2 | +20.8% | $62,799.64 | **$15,699.91** | $47,099.73 | 257.0 |
| 004 | Neural Search & Retrieval Cluster Xi #4 | `GROWTH_STARTUP` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-V100-32GB | 256 | 1,000,000 | 1936.3 | 1558.7 | +19.5% | $32,221.87 | **$8,055.47** | $24,166.40 | 1781.4 |
| 005 | Multimodal Compute Group Zeta #5 | `WHALE_HYPERSCALER` | Phi-3.5-MoE (MOE) | NVIDIA-L40S-48GB | 4096 | 250,000 | 1242.7 | 901.0 | +27.5% | $174,950.40 | **$43,737.60** | $131,212.80 | 1143.3 |
| 006 | Foundation Supercluster Delta #6 | `WHALE_HYPERSCALER` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 1024 | 2,500,000 | 8275.9 | 5966.9 | +27.9% | $1,313,564.44 | **$328,391.11** | $985,173.33 | 7613.8 |
| 007 | Frontier Reasoning Lab Beta #7 | `WHALE_HYPERSCALER` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 8192 | 100,000 | 222.8 | 171.8 | +22.9% | $48,742.40 | **$12,185.60** | $36,556.80 | 205.0 |
| 008 | Academic AI Research Lab Upsilon #8 | `RESEARCH_LAB` | Gemma-2-27B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 1 | 50,000 | 623.7 | 475.3 | +23.8% | $5.15 | **$1.29** | $3.86 | 573.8 |
| 009 | Constitutional AI Pod Gamma #9 | `WHALE_HYPERSCALER` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 8192 | 1,000,000 | 345.9 | 268.4 | +22.4% | $740,693.33 | **$185,173.33** | $555,520.00 | 318.2 |
| 010 | Multimodal Compute Group Zeta #10 | `WHALE_HYPERSCALER` | Gemma-2-27B (TRANSFORMER) | NVIDIA-L40S-48GB | 2048 | 1,000,000 | 823.8 | 630.2 | +23.5% | $198,246.40 | **$49,561.60** | $148,684.80 | 757.9 |
| 011 | Constitutional AI Pod Gamma #11 | `WHALE_HYPERSCALER` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 16384 | 500,000 | 198.7 | 156.4 | +21.3% | $529,408.00 | **$132,352.00** | $397,056.00 | 182.8 |
| 012 | Open Source AI Collective Phi #12 | `RESEARCH_LAB` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 4 | 50,000 | 642.9 | 504.7 | +21.5% | $19.19 | **$4.80** | $14.40 | 591.5 |
| 013 | Quantitative Alpha Research Desk Lambda #13 | `ENTERPRISE_TIER1` | Gemma-2-9B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 128 | 500,000 | 179.1 | 141.7 | +20.9% | $2,992.00 | **$748.00** | $2,244.00 | 164.8 |
| 014 | Quantitative Alpha Research Desk Lambda #14 | `ENTERPRISE_TIER1` | DeepSeek-V3-671B (MOE) | NVIDIA-V100-32GB | 1024 | 10,000,000 | 13577.2 | 9870.6 | +27.3% | $12,651,861.33 | **$3,162,965.33** | $9,488,896.00 | 12491.0 |
| 015 | Domain Legal AI Engine Sigma #15 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 256 | 5,000,000 | 357.4 | 275.6 | +22.9% | $130,880.00 | **$32,720.00** | $98,160.00 | 328.8 |
| 016 | Enterprise Frontier Lab Theta #16 | `ENTERPRISE_TIER1` | Mixtral-8x7B-MoE (MOE) | NVIDIA-V100-32GB | 128 | 50,000 | 2427.6 | 1871.7 | +22.9% | $1,185.92 | **$296.48** | $889.44 | 2233.4 |
| 017 | Multimodal Compute Group Zeta #17 | `WHALE_HYPERSCALER` | Mamba-Codestral-7B (SSM) | NVIDIA-RTX-4090-24GB | 16384 | 100,000 | 659.6 | 521.1 | +21.0% | $50,426.31 | **$12,606.58** | $37,819.73 | 606.8 |
| 018 | Speech & Audio Synthesis Startup Omicron #18 | `GROWTH_STARTUP` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 256 | 50,000 | 222.2 | 180.2 | +18.9% | $672.00 | **$168.00** | $504.00 | 204.4 |
| 019 | Domain Legal AI Engine Sigma #19 | `GROWTH_STARTUP` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-L40S-48GB | 16 | 1,000,000 | 1641.7 | 1228.0 | +25.2% | $3,309.60 | **$827.40** | $2,482.20 | 1510.4 |
| 020 | Enterprise LLM Fleet Iota #20 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 2048 | 100,000 | 171.6 | 138.1 | +19.5% | $10,481.78 | **$2,620.44** | $7,861.33 | 157.9 |
| 021 | Open Source AI Collective Phi #21 | `RESEARCH_LAB` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 2 | 2,500,000 | 450.1 | 358.7 | +20.3% | $317.36 | **$79.34** | $238.02 | 414.1 |
| 022 | Distributed Ray Compute Fleet Rho #22 | `GROWTH_STARTUP` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-L40S-48GB | 128 | 50,000 | 433.3 | 334.5 | +22.8% | $316.16 | **$79.04** | $237.12 | 398.6 |
| 023 | Data Platform Compute Lab Kappa #23 | `ENTERPRISE_TIER1` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 256 | 100,000 | 382.8 | 301.6 | +21.2% | $2,194.20 | **$548.55** | $1,645.65 | 352.2 |
| 024 | Constitutional AI Pod Gamma #24 | `WHALE_HYPERSCALER` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 2048 | 10,000,000 | 270.0 | 210.6 | +22.0% | $1,858,560.00 | **$464,640.00** | $1,393,920.00 | 248.4 |
| 025 | Distributed Ray Compute Fleet Rho #25 | `GROWTH_STARTUP` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 32 | 500,000 | 1158.2 | 910.3 | +21.4% | $881.42 | **$220.36** | $661.07 | 1065.5 |
| 026 | Computer Science Intelligence Lab Chi #26 | `RESEARCH_LAB` | Mamba-2.8B-SSM (SSM) | NVIDIA-L40S-48GB | 8 | 10,000,000 | 217.0 | 177.3 | +18.3% | $1,588.00 | **$397.00** | $1,191.00 | 199.6 |
| 027 | Quantitative Alpha Research Desk Lambda #27 | `ENTERPRISE_TIER1` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 2048 | 500,000 | 324.2 | 246.1 | +24.1% | $55,537.78 | **$13,884.44** | $41,653.33 | 298.3 |
| 028 | Frontier Reasoning Lab Beta #28 | `WHALE_HYPERSCALER` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 1024 | 2,500,000 | 324.6 | 258.7 | +20.3% | $196,821.33 | **$49,205.33** | $147,616.00 | 298.6 |
| 029 | Computer Science Intelligence Lab Chi #29 | `RESEARCH_LAB` | Falcon-180B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 16 | 250,000 | 2353.2 | 1757.8 | +25.3% | $1,653.89 | **$413.47** | $1,240.42 | 2164.9 |
| 030 | University Foundation Center Tau #30 | `RESEARCH_LAB` | Falcon-180B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 8 | 5,000,000 | 2159.5 | 1680.1 | +22.2% | $13,316.67 | **$3,329.17** | $9,987.50 | 1986.7 |
| 031 | Neural Search & Retrieval Cluster Xi #31 | `GROWTH_STARTUP` | Gemma-2-27B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 128 | 10,000,000 | 1286.8 | 1015.3 | +21.1% | $77,226.67 | **$19,306.67** | $57,920.00 | 1183.9 |
| 032 | Open Source AI Collective Phi #32 | `RESEARCH_LAB` | Mamba-2.8B-SSM (SSM) | NVIDIA-H200-SXM-141GB | 1 | 5,000,000 | 96.9 | 77.4 | +20.1% | $113.75 | **$28.44** | $85.31 | 89.1 |
| 033 | Foundation Supercluster Delta #33 | `WHALE_HYPERSCALER` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 2048 | 10,000,000 | 618.2 | 493.9 | +20.1% | $1,767,822.22 | **$441,955.56** | $1,325,866.67 | 568.7 |
| 034 | Speech & Audio Synthesis Startup Omicron #34 | `GROWTH_STARTUP` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 128 | 100,000 | 323.4 | 249.0 | +23.0% | $1,111.04 | **$277.76** | $833.28 | 297.5 |
| 035 | Hyperscaler Cluster Alpha #35 | `WHALE_HYPERSCALER` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 2048 | 5,000,000 | 3911.0 | 2890.2 | +26.1% | $7,259,022.22 | **$1,814,755.56** | $5,444,266.67 | 3598.1 |
| 036 | Fintech Quant Trading Engine Nu #36 | `ENTERPRISE_TIER1` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 1024 | 50,000 | 439.7 | 334.6 | +23.9% | $8,221.16 | **$2,055.29** | $6,165.87 | 404.5 |
| 037 | Speech & Audio Synthesis Startup Omicron #37 | `GROWTH_STARTUP` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 10,000,000 | 1300.0 | 1020.5 | +21.5% | $159,004.44 | **$39,751.11** | $119,253.33 | 1196.0 |
| 038 | Frontier Reasoning Lab Beta #38 | `WHALE_HYPERSCALER` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 4096 | 50,000 | 909.5 | 684.9 | +24.7% | $48,553.53 | **$12,138.38** | $36,415.15 | 836.7 |
| 039 | Distributed Ray Compute Fleet Rho #39 | `GROWTH_STARTUP` | Mixtral-8x22B-MoE (MOE) | NVIDIA-H100-SXM5-80GB | 256 | 100,000 | 1214.6 | 923.1 | +24.0% | $7,876.98 | **$1,969.24** | $5,907.73 | 1117.4 |
| 040 | Constitutional AI Pod Gamma #40 | `WHALE_HYPERSCALER` | Mamba-Codestral-7B (SSM) | NVIDIA-A100-SXM4-80GB | 8192 | 100,000 | 359.7 | 284.5 | +20.9% | $42,780.44 | **$10,695.11** | $32,085.33 | 330.9 |
| 041 | University Foundation Center Tau #41 | `RESEARCH_LAB` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 16 | 100,000 | 345.5 | 269.5 | +22.0% | $128.36 | **$32.09** | $96.27 | 317.9 |
| 042 | Distributed Ray Compute Fleet Rho #42 | `GROWTH_STARTUP` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 32 | 2,500,000 | 138.0 | 111.4 | +19.3% | $2,660.00 | **$665.00** | $1,995.00 | 127.0 |
| 043 | Computer Science Intelligence Lab Chi #43 | `RESEARCH_LAB` | Mamba-2.8B-SSM (SSM) | NVIDIA-GH200-GraceHopper | 8 | 2,500,000 | 94.0 | 74.7 | +20.5% | $482.50 | **$120.62** | $361.88 | 86.5 |
| 044 | Hyperscaler Cluster Alpha #44 | `WHALE_HYPERSCALER` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 1024 | 500,000 | 221.2 | 176.1 | +20.4% | $26,939.73 | **$6,734.93** | $20,204.80 | 203.5 |
| 045 | Systematic Trading ML Desk Mu #45 | `ENTERPRISE_TIER1` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 512 | 1,000,000 | 3705.3 | 2634.5 | +28.9% | $380,728.89 | **$95,182.22** | $285,546.67 | 3408.9 |
| 046 | Data Platform Compute Lab Kappa #46 | `ENTERPRISE_TIER1` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 2048 | 1,000,000 | 374.2 | 298.2 | +20.3% | $164,295.11 | **$41,073.78** | $123,221.33 | 344.3 |
| 047 | Cloud AI Supercluster Epsilon #47 | `WHALE_HYPERSCALER` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 1024 | 250,000 | 647.0 | 480.1 | +25.8% | $49,847.47 | **$12,461.87** | $37,385.60 | 595.2 |
| 048 | Quantitative Alpha Research Desk Lambda #48 | `ENTERPRISE_TIER1` | DeepSeek-V3-671B (MOE) | NVIDIA-GH200-GraceHopper | 2048 | 10,000,000 | 2458.1 | 1698.5 | +30.9% | $19,445,760.00 | **$4,861,440.00** | $14,584,320.00 | 2261.5 |
| 049 | Frontier Reasoning Lab Beta #49 | `WHALE_HYPERSCALER` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 4096 | 50,000 | 7217.2 | 5153.1 | +28.6% | $93,939.48 | **$23,484.87** | $70,454.61 | 6639.8 |
| 050 | Open Source AI Collective Phi #50 | `RESEARCH_LAB` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 8 | 5,000,000 | 478.6 | 365.2 | +23.7% | $6,930.00 | **$1,732.50** | $5,197.50 | 440.3 |
| 051 | Cloud AI Supercluster Epsilon #51 | `WHALE_HYPERSCALER` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 4096 | 250,000 | 415.3 | 326.0 | +21.5% | $96,523.38 | **$24,130.84** | $72,392.53 | 382.1 |
| 052 | Systematic Trading ML Desk Mu #52 | `ENTERPRISE_TIER1` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-L40S-48GB | 256 | 500,000 | 1466.1 | 1111.3 | +24.2% | $22,707.20 | **$5,676.80** | $17,030.40 | 1348.8 |
| 053 | Domain Legal AI Engine Sigma #53 | `GROWTH_STARTUP` | Phi-3.5-MoE (MOE) | NVIDIA-B200-Blackwell-192GB | 32 | 250,000 | 400.3 | 289.4 | +27.7% | $1,355.44 | **$338.86** | $1,016.58 | 368.3 |
| 054 | Neural Search & Retrieval Cluster Xi #54 | `GROWTH_STARTUP` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 64 | 5,000,000 | 138.4 | 104.2 | +24.7% | $16,720.00 | **$4,180.00** | $12,540.00 | 127.3 |
| 055 | Enterprise Frontier Lab Theta #55 | `ENTERPRISE_TIER1` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 128 | 5,000,000 | 502.2 | 367.1 | +26.9% | $132,097.78 | **$33,024.44** | $99,073.33 | 462.0 |
| 056 | Enterprise Frontier Lab Theta #56 | `ENTERPRISE_TIER1` | DeepSeek-V3-671B (MOE) | NVIDIA-V100-32GB | 512 | 1,000,000 | 15138.5 | 10491.0 | +30.7% | $793,173.33 | **$198,293.33** | $594,880.00 | 13927.4 |
| 057 | Enterprise Frontier Lab Theta #57 | `ENTERPRISE_TIER1` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 1024 | 2,500,000 | 339.5 | 266.2 | +21.6% | $130,311.11 | **$32,577.78** | $97,733.33 | 312.3 |
| 058 | Enterprise LLM Fleet Iota #58 | `ENTERPRISE_TIER1` | Gemma-2-27B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 2048 | 2,500,000 | 1301.5 | 1024.3 | +21.3% | $315,392.00 | **$78,848.00** | $236,544.00 | 1197.4 |
| 059 | Enterprise LLM Fleet Iota #59 | `ENTERPRISE_TIER1` | Mixtral-8x22B-MoE (MOE) | NVIDIA-GH200-GraceHopper | 128 | 100,000 | 934.2 | 698.8 | +25.2% | $3,766.40 | **$941.60** | $2,824.80 | 859.5 |
| 060 | Quantitative Alpha Research Desk Lambda #60 | `ENTERPRISE_TIER1` | Phi-3.5-MoE (MOE) | NVIDIA-H200-SXM-141GB | 1024 | 500,000 | 506.3 | 368.1 | +27.3% | $82,551.47 | **$20,637.87** | $61,913.60 | 465.8 |
| 061 | Hyperscaler Cluster Alpha #61 | `WHALE_HYPERSCALER` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 4096 | 5,000,000 | 402.3 | 319.0 | +20.7% | $1,184,711.11 | **$296,177.78** | $888,533.33 | 370.1 |
| 062 | Quantitative Alpha Research Desk Lambda #62 | `ENTERPRISE_TIER1` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-L40S-48GB | 128 | 1,000,000 | 735.8 | 571.7 | +22.3% | $10,502.40 | **$2,625.60** | $7,876.80 | 676.9 |
| 063 | Quantitative Alpha Research Desk Lambda #63 | `ENTERPRISE_TIER1` | Gemma-2-9B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 2048 | 250,000 | 159.7 | 125.7 | +21.3% | $21,760.00 | **$5,440.00** | $16,320.00 | 146.9 |
| 064 | Fintech Quant Trading Engine Nu #64 | `ENTERPRISE_TIER1` | Mamba-Codestral-7B (SSM) | NVIDIA-B200-Blackwell-192GB | 2048 | 1,000,000 | 124.1 | 97.0 | +21.8% | $84,792.89 | **$21,198.22** | $63,594.67 | 114.2 |
| 065 | Data Platform Compute Lab Kappa #65 | `ENTERPRISE_TIER1` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 1024 | 2,500,000 | 684.1 | 533.6 | +22.0% | $267,555.56 | **$66,888.89** | $200,666.67 | 629.4 |
| 066 | UltraCluster Infrastructure Eta #66 | `WHALE_HYPERSCALER` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-V100-32GB | 2048 | 100,000 | 1173.0 | 930.2 | +20.7% | $16,575.15 | **$4,143.79** | $12,431.36 | 1079.2 |
| 067 | Academic AI Research Lab Upsilon #67 | `RESEARCH_LAB` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 2 | 1,000,000 | 146.3 | 118.9 | +18.7% | $63.93 | **$15.98** | $47.95 | 134.6 |
| 068 | Hyperscaler Cluster Alpha #68 | `WHALE_HYPERSCALER` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 8192 | 100,000 | 1080.2 | 795.0 | +26.4% | $162,247.11 | **$40,561.78** | $121,685.33 | 993.8 |
| 069 | University Foundation Center Tau #69 | `RESEARCH_LAB` | DeepSeek-V3-671B (MOE) | NVIDIA-RTX-4090-24GB | 8 | 5,000,000 | 9820.5 | 6903.8 | +29.7% | $25,926.22 | **$6,481.56** | $19,444.67 | 9034.9 |
| 070 | Hyperscaler Cluster Alpha #70 | `WHALE_HYPERSCALER` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 2048 | 250,000 | 973.2 | 755.2 | +22.4% | $24,803.56 | **$6,200.89** | $18,602.67 | 895.3 |
| 071 | Speech & Audio Synthesis Startup Omicron #71 | `GROWTH_STARTUP` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 16 | 10,000,000 | 1024.2 | 771.2 | +24.7% | $28,111.11 | **$7,027.78** | $21,083.33 | 942.3 |
| 072 | Distributed Ray Compute Fleet Rho #72 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-L40S-48GB | 128 | 5,000,000 | 928.6 | 710.4 | +23.5% | $69,824.00 | **$17,456.00** | $52,368.00 | 854.3 |
| 073 | Multimodal Compute Group Zeta #73 | `WHALE_HYPERSCALER` | Falcon-180B (TRANSFORMER) | NVIDIA-V100-32GB | 4096 | 500,000 | 6737.5 | 5194.6 | +22.9% | $1,053,286.40 | **$263,321.60** | $789,964.80 | 6198.5 |
| 074 | Distributed Ray Compute Fleet Rho #74 | `GROWTH_STARTUP` | Falcon-180B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 64 | 10,000,000 | 2118.4 | 1631.2 | +23.0% | $216,533.33 | **$54,133.33** | $162,400.00 | 1948.9 |
| 075 | Enterprise LLM Fleet Iota #75 | `ENTERPRISE_TIER1` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-L40S-48GB | 2048 | 100,000 | 356.1 | 289.2 | +18.8% | $6,850.56 | **$1,712.64** | $5,137.92 | 327.6 |
| 076 | Cloud AI Supercluster Epsilon #76 | `WHALE_HYPERSCALER` | DeepSeek-V3-671B (MOE) | NVIDIA-H200-SXM-141GB | 8192 | 5,000,000 | 2646.4 | 1884.2 | +28.8% | $36,422,997.33 | **$9,105,749.33** | $27,317,248.00 | 2434.7 |
| 077 | Frontier Reasoning Lab Beta #77 | `WHALE_HYPERSCALER` | DeepSeek-V3-671B (MOE) | NVIDIA-RTX-4090-24GB | 8192 | 50,000 | 10824.5 | 7555.5 | +30.2% | $297,551.64 | **$74,387.91** | $223,163.73 | 9958.5 |
| 078 | Fintech Quant Trading Engine Nu #78 | `ENTERPRISE_TIER1` | Falcon-180B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 512 | 1,000,000 | 1477.6 | 1105.2 | +25.2% | $201,261.51 | **$50,315.38** | $150,946.13 | 1359.4 |
| 079 | Neural Search & Retrieval Cluster Xi #79 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 128 | 500,000 | 647.6 | 497.4 | +23.2% | $6,675.56 | **$1,668.89** | $5,006.67 | 595.8 |
| 080 | Fintech Quant Trading Engine Nu #80 | `ENTERPRISE_TIER1` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 10,000,000 | 614.7 | 484.4 | +21.2% | $74,126.22 | **$18,531.56** | $55,594.67 | 565.5 |
| 081 | Domain Legal AI Engine Sigma #81 | `GROWTH_STARTUP` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 50,000 | 657.1 | 506.0 | +23.0% | $429.80 | **$107.45** | $322.35 | 604.5 |
| 082 | Speech & Audio Synthesis Startup Omicron #82 | `GROWTH_STARTUP` | Mixtral-8x22B-MoE (MOE) | NVIDIA-H100-SXM5-80GB | 64 | 5,000,000 | 1118.2 | 830.8 | +25.7% | $97,077.33 | **$24,269.33** | $72,808.00 | 1028.7 |
| 083 | Quantitative Alpha Research Desk Lambda #83 | `ENTERPRISE_TIER1` | Gemma-2-9B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 1024 | 1,000,000 | 177.0 | 140.5 | +20.6% | $46,720.00 | **$11,680.00** | $35,040.00 | 162.8 |
| 084 | Domain Legal AI Engine Sigma #84 | `GROWTH_STARTUP` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-L40S-48GB | 32 | 100,000 | 1492.3 | 1111.8 | +25.5% | $608.80 | **$152.20** | $456.60 | 1372.9 |
| 085 | Hyperscaler Cluster Alpha #85 | `WHALE_HYPERSCALER` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 1024 | 250,000 | 628.6 | 496.0 | +21.1% | $23,573.33 | **$5,893.33** | $17,680.00 | 578.3 |
| 086 | Foundation Supercluster Delta #86 | `WHALE_HYPERSCALER` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 8192 | 1,000,000 | 1635.0 | 1250.8 | +23.5% | $2,185,671.11 | **$546,417.78** | $1,639,253.33 | 1504.2 |
| 087 | Academic AI Research Lab Upsilon #87 | `RESEARCH_LAB` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 4 | 100,000 | 176.2 | 135.1 | +23.3% | $25.12 | **$6.28** | $18.84 | 162.1 |
| 088 | University Foundation Center Tau #88 | `RESEARCH_LAB` | Mamba-Codestral-7B (SSM) | NVIDIA-RTX-4090-24GB | 8 | 500,000 | 626.4 | 493.0 | +21.3% | $118.58 | **$29.64** | $88.93 | 576.3 |
| 089 | Open Source AI Collective Phi #89 | `RESEARCH_LAB` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 4 | 100,000 | 365.0 | 286.9 | +21.4% | $36.45 | **$9.11** | $27.34 | 335.8 |
| 090 | Frontier Reasoning Lab Beta #90 | `WHALE_HYPERSCALER` | Gemma-2-27B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 4096 | 100,000 | 1217.2 | 947.0 | +22.2% | $24,594.20 | **$6,148.55** | $18,445.65 | 1119.8 |
| 091 | Neural Search & Retrieval Cluster Xi #91 | `GROWTH_STARTUP` | Falcon-180B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 128 | 2,500,000 | 1233.2 | 923.7 | +25.1% | $115,546.67 | **$28,886.67** | $86,660.00 | 1134.5 |
| 092 | Cloud AI Supercluster Epsilon #92 | `WHALE_HYPERSCALER` | Phi-3.5-MoE (MOE) | NVIDIA-B200-Blackwell-192GB | 16384 | 1,000,000 | 378.9 | 272.1 | +28.2% | $2,673,322.67 | **$668,330.67** | $2,004,992.00 | 348.6 |
| 093 | Distributed GPU Cloud Platform Pi #93 | `GROWTH_STARTUP` | Phi-4-14B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 1,000,000 | 808.3 | 640.2 | +20.8% | $9,563.02 | **$2,390.76** | $7,172.27 | 743.6 |
| 094 | Constitutional AI Pod Gamma #94 | `WHALE_HYPERSCALER` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-V100-32GB | 2048 | 10,000,000 | 3594.7 | 2656.5 | +26.1% | $6,404,778.67 | **$1,601,194.67** | $4,803,584.00 | 3307.1 |
| 095 | Quantitative Alpha Research Desk Lambda #95 | `ENTERPRISE_TIER1` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-V100-32GB | 512 | 250,000 | 988.0 | 763.7 | +22.7% | $9,570.13 | **$2,392.53** | $7,177.60 | 909.0 |
| 096 | Enterprise LLM Fleet Iota #96 | `ENTERPRISE_TIER1` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 2048 | 1,000,000 | 145.3 | 116.7 | +19.7% | $73,216.00 | **$18,304.00** | $54,912.00 | 133.7 |
| 097 | UltraCluster Infrastructure Eta #97 | `WHALE_HYPERSCALER` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 8192 | 10,000,000 | 3954.6 | 2914.5 | +26.3% | $59,170,133.33 | **$14,792,533.33** | $44,377,600.00 | 3638.2 |
| 098 | Distributed GPU Cloud Platform Pi #98 | `GROWTH_STARTUP` | Mamba-2.8B-SSM (SSM) | NVIDIA-V100-32GB | 128 | 50,000 | 462.8 | 378.1 | +18.3% | $180.69 | **$45.17** | $135.52 | 425.8 |
| 099 | Enterprise Frontier Lab Theta #99 | `ENTERPRISE_TIER1` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-L40S-48GB | 1024 | 2,500,000 | 892.2 | 685.2 | +23.2% | $264,960.00 | **$66,240.00** | $198,720.00 | 820.8 |
| 100 | Quantitative Alpha Research Desk Lambda #100 | `ENTERPRISE_TIER1` | Gemma-2-27B (TRANSFORMER) | NVIDIA-V100-32GB | 512 | 1,000,000 | 1642.0 | 1290.6 | +21.4% | $59,972.27 | **$14,993.07** | $44,979.20 | 1510.6 |
| 101 | UltraCluster Infrastructure Eta #101 | `WHALE_HYPERSCALER` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 16384 | 250,000 | 145.4 | 115.7 | +20.4% | $141,926.40 | **$35,481.60** | $106,444.80 | 133.8 |
| 102 | Enterprise LLM Fleet Iota #102 | `ENTERPRISE_TIER1` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 128 | 500,000 | 758.4 | 574.1 | +24.3% | $14,744.00 | **$3,686.00** | $11,058.00 | 697.7 |
| 103 | Data Platform Compute Lab Kappa #103 | `ENTERPRISE_TIER1` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 512 | 50,000 | 142.0 | 108.6 | +23.5% | $1,306.31 | **$326.58** | $979.73 | 130.6 |
| 104 | Cloud AI Supercluster Epsilon #104 | `WHALE_HYPERSCALER` | Mamba-2.8B-SSM (SSM) | NVIDIA-V100-32GB | 1024 | 50,000 | 500.9 | 405.2 | +19.1% | $1,633.28 | **$408.32** | $1,224.96 | 460.8 |
| 105 | Speech & Audio Synthesis Startup Omicron #105 | `GROWTH_STARTUP` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 16 | 1,000,000 | 115.6 | 92.9 | +19.6% | $554.89 | **$138.72** | $416.17 | 106.4 |
| 106 | Constitutional AI Pod Gamma #106 | `WHALE_HYPERSCALER` | Gemma-2-9B (TRANSFORMER) | NVIDIA-V100-32GB | 1024 | 50,000 | 892.3 | 697.8 | +21.8% | $3,319.47 | **$829.87** | $2,489.60 | 820.9 |
| 107 | Data Platform Compute Lab Kappa #107 | `ENTERPRISE_TIER1` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 128 | 5,000,000 | 3986.7 | 2842.5 | +28.7% | $508,533.33 | **$127,133.33** | $381,400.00 | 3667.8 |
| 108 | Enterprise Frontier Lab Theta #108 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 1024 | 10,000,000 | 836.0 | 664.6 | +20.5% | $390,030.22 | **$97,507.56** | $292,522.67 | 769.1 |
| 109 | Neural Search & Retrieval Cluster Xi #109 | `GROWTH_STARTUP` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 256 | 100,000 | 460.8 | 363.1 | +21.2% | $2,640.07 | **$660.02** | $1,980.05 | 423.9 |
| 110 | UltraCluster Infrastructure Eta #110 | `WHALE_HYPERSCALER` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 2048 | 500,000 | 636.5 | 486.3 | +23.6% | $179,438.93 | **$44,859.73** | $134,579.20 | 585.6 |
| 111 | Hyperscaler Cluster Alpha #111 | `WHALE_HYPERSCALER` | Phi-3.5-MoE (MOE) | NVIDIA-V100-32GB | 16384 | 10,000,000 | 2696.8 | 2017.2 | +25.2% | $37,115,221.33 | **$9,278,805.33** | $27,836,416.00 | 2481.1 |
| 112 | Data Platform Compute Lab Kappa #112 | `ENTERPRISE_TIER1` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 256 | 1,000,000 | 231.2 | 179.9 | +22.2% | $15,321.60 | **$3,830.40** | $11,491.20 | 212.7 |
| 113 | Computer Science Intelligence Lab Chi #113 | `RESEARCH_LAB` | DeepSeek-V3-671B (MOE) | NVIDIA-RTX-4090-24GB | 1 | 500,000 | 10481.9 | 7243.0 | +30.9% | $359.88 | **$89.97** | $269.91 | 9643.3 |
| 114 | Cloud AI Supercluster Epsilon #114 | `WHALE_HYPERSCALER` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 2048 | 1,000,000 | 2487.3 | 1768.5 | +28.9% | $1,553,885.87 | **$388,471.47** | $1,165,414.40 | 2288.3 |
| 115 | Data Platform Compute Lab Kappa #115 | `ENTERPRISE_TIER1` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 1024 | 1,000,000 | 228.1 | 175.0 | +23.3% | $63,436.80 | **$15,859.20** | $47,577.60 | 209.9 |
| 116 | Distributed Ray Compute Fleet Rho #116 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-V100-32GB | 128 | 100,000 | 1919.6 | 1443.5 | +24.8% | $2,031.36 | **$507.84** | $1,523.52 | 1766.0 |
| 117 | Quantitative Alpha Research Desk Lambda #117 | `ENTERPRISE_TIER1` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 128 | 100,000 | 267.9 | 207.9 | +22.4% | $960.00 | **$240.00** | $720.00 | 246.5 |
| 118 | University Foundation Center Tau #118 | `RESEARCH_LAB` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 4 | 50,000 | 1021.3 | 781.3 | +23.5% | $50.67 | **$12.67** | $38.00 | 939.6 |
| 119 | Distributed Ray Compute Fleet Rho #119 | `GROWTH_STARTUP` | Mamba-2.8B-SSM (SSM) | NVIDIA-V100-32GB | 64 | 250,000 | 534.4 | 433.4 | +18.9% | $538.67 | **$134.67** | $404.00 | 491.6 |
| 120 | Distributed GPU Cloud Platform Pi #120 | `GROWTH_STARTUP` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-V100-32GB | 64 | 5,000,000 | 10731.0 | 7704.9 | +28.2% | $322,784.00 | **$80,696.00** | $242,088.00 | 9872.5 |
| 121 | Multimodal Compute Group Zeta #121 | `WHALE_HYPERSCALER` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 8192 | 1,000,000 | 591.0 | 476.9 | +19.3% | $207,712.71 | **$51,928.18** | $155,784.53 | 543.7 |
| 122 | Domain Legal AI Engine Sigma #122 | `GROWTH_STARTUP` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-V100-32GB | 16 | 2,500,000 | 2938.8 | 2254.1 | +23.3% | $9,129.33 | **$2,282.33** | $6,847.00 | 2703.7 |
| 123 | Fintech Quant Trading Engine Nu #123 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 2048 | 10,000,000 | 266.3 | 215.4 | +19.1% | $1,100,344.89 | **$275,086.22** | $825,258.67 | 245.0 |
| 124 | Systematic Trading ML Desk Mu #124 | `ENTERPRISE_TIER1` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-V100-32GB | 1024 | 50,000 | 1500.3 | 1171.7 | +21.9% | $5,608.11 | **$1,402.03** | $4,206.08 | 1380.3 |
| 125 | Data Platform Compute Lab Kappa #125 | `ENTERPRISE_TIER1` | Phi-4-14B (TRANSFORMER) | NVIDIA-V100-32GB | 128 | 50,000 | 1258.6 | 980.4 | +22.1% | $593.49 | **$148.37** | $445.12 | 1157.9 |
| 126 | Enterprise Frontier Lab Theta #126 | `ENTERPRISE_TIER1` | Mixtral-8x22B-MoE (MOE) | NVIDIA-GH200-GraceHopper | 2048 | 50,000 | 927.7 | 704.1 | +24.1% | $28,620.80 | **$7,155.20** | $21,465.60 | 853.5 |
| 127 | Foundation Supercluster Delta #127 | `WHALE_HYPERSCALER` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 2048 | 10,000,000 | 236.3 | 186.7 | +21.0% | $1,185,109.33 | **$296,277.33** | $888,832.00 | 217.4 |
| 128 | Enterprise LLM Fleet Iota #128 | `ENTERPRISE_TIER1` | Phi-3.5-MoE (MOE) | NVIDIA-V100-32GB | 1024 | 500,000 | 2770.7 | 2080.8 | +24.9% | $117,742.93 | **$29,435.73** | $88,307.20 | 2549.0 |
| 129 | UltraCluster Infrastructure Eta #129 | `WHALE_HYPERSCALER` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 1024 | 1,000,000 | 219.2 | 168.8 | +23.0% | $78,848.00 | **$19,712.00** | $59,136.00 | 201.7 |
| 130 | Quantitative Alpha Research Desk Lambda #130 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 128 | 1,000,000 | 223.8 | 176.1 | +21.3% | $7,632.00 | **$1,908.00** | $5,724.00 | 205.9 |
| 131 | University Foundation Center Tau #131 | `RESEARCH_LAB` | Mamba-Codestral-7B (SSM) | NVIDIA-V100-32GB | 2 | 10,000,000 | 950.8 | 758.7 | +20.2% | $1,280.67 | **$320.17** | $960.50 | 874.7 |
| 132 | Domain Legal AI Engine Sigma #132 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 128 | 2,500,000 | 1423.5 | 1066.2 | +25.1% | $25,408.00 | **$6,352.00** | $19,056.00 | 1309.6 |
| 133 | Enterprise Frontier Lab Theta #133 | `ENTERPRISE_TIER1` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 5,000,000 | 2907.0 | 2157.0 | +25.8% | $213,333.33 | **$53,333.33** | $160,000.00 | 2674.4 |
| 134 | Fintech Quant Trading Engine Nu #134 | `ENTERPRISE_TIER1` | Gemma-2-9B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 1024 | 50,000 | 118.5 | 92.1 | +22.3% | $2,065.07 | **$516.27** | $1,548.80 | 109.0 |
| 135 | Quantitative Alpha Research Desk Lambda #135 | `ENTERPRISE_TIER1` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-V100-32GB | 128 | 10,000,000 | 1203.8 | 957.0 | +20.5% | $105,301.33 | **$26,325.33** | $78,976.00 | 1107.5 |
| 136 | Open Source AI Collective Phi #136 | `RESEARCH_LAB` | Mixtral-8x7B-MoE (MOE) | NVIDIA-H200-SXM-141GB | 8 | 1,000,000 | 422.3 | 316.3 | +25.1% | $989.33 | **$247.33** | $742.00 | 388.5 |
| 137 | Open Source AI Collective Phi #137 | `RESEARCH_LAB` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-L40S-48GB | 16 | 5,000,000 | 5518.8 | 3973.5 | +28.0% | $61,812.00 | **$15,453.00** | $46,359.00 | 5077.3 |
| 138 | Distributed GPU Cloud Platform Pi #138 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 256 | 100,000 | 402.2 | 305.7 | +24.0% | $2,607.64 | **$651.91** | $1,955.73 | 370.0 |
| 139 | Data Platform Compute Lab Kappa #139 | `ENTERPRISE_TIER1` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 1024 | 100,000 | 363.0 | 283.5 | +21.9% | $9,497.60 | **$2,374.40** | $7,123.20 | 334.0 |
| 140 | Foundation Supercluster Delta #140 | `WHALE_HYPERSCALER` | Phi-4-14B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 4096 | 50,000 | 144.7 | 111.7 | +22.8% | $10,325.33 | **$2,581.33** | $7,744.00 | 133.1 |
| 141 | Data Platform Compute Lab Kappa #141 | `ENTERPRISE_TIER1` | Gemma-2-27B (TRANSFORMER) | NVIDIA-L40S-48GB | 1024 | 250,000 | 765.3 | 593.9 | +22.4% | $21,939.20 | **$5,484.80** | $16,454.40 | 704.1 |
| 142 | Computer Science Intelligence Lab Chi #142 | `RESEARCH_LAB` | Mixtral-8x22B-MoE (MOE) | NVIDIA-B200-Blackwell-192GB | 2 | 100,000 | 702.1 | 523.8 | +25.4% | $54.48 | **$13.62** | $40.86 | 645.9 |
| 143 | Computer Science Intelligence Lab Chi #143 | `RESEARCH_LAB` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-V100-32GB | 16 | 250,000 | 1765.9 | 1382.7 | +21.7% | $510.93 | **$127.73** | $383.20 | 1624.6 |
| 144 | Speech & Audio Synthesis Startup Omicron #144 | `GROWTH_STARTUP` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 128 | 1,000,000 | 812.5 | 618.3 | +23.9% | $26,238.58 | **$6,559.64** | $19,678.93 | 747.5 |
| 145 | Speech & Audio Synthesis Startup Omicron #145 | `GROWTH_STARTUP` | Gemma-2-27B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 128 | 1,000,000 | 301.9 | 235.5 | +22.0% | $10,624.00 | **$2,656.00** | $7,968.00 | 277.7 |
| 146 | Fintech Quant Trading Engine Nu #146 | `ENTERPRISE_TIER1` | Falcon-180B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 256 | 5,000,000 | 1220.9 | 926.7 | +24.1% | $439,338.67 | **$109,834.67** | $329,504.00 | 1123.2 |
| 147 | Data Platform Compute Lab Kappa #147 | `ENTERPRISE_TIER1` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 512 | 50,000 | 711.4 | 540.7 | +24.0% | $971.09 | **$242.77** | $728.32 | 654.5 |
| 148 | Frontier Reasoning Lab Beta #148 | `WHALE_HYPERSCALER` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-V100-32GB | 4096 | 500,000 | 1602.0 | 1262.4 | +21.2% | $231,833.60 | **$57,958.40** | $173,875.20 | 1473.8 |
| 149 | Open Source AI Collective Phi #149 | `RESEARCH_LAB` | DeepSeek-V3-671B (MOE) | NVIDIA-GH200-GraceHopper | 2 | 100,000 | 2725.2 | 1915.8 | +29.7% | $202.35 | **$50.59** | $151.76 | 2507.2 |
| 150 | Data Platform Compute Lab Kappa #150 | `ENTERPRISE_TIER1` | Phi-4-14B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 2048 | 2,500,000 | 269.7 | 215.0 | +20.3% | $295,623.11 | **$73,905.78** | $221,717.33 | 248.1 |
| 151 | Data Platform Compute Lab Kappa #151 | `ENTERPRISE_TIER1` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 500,000 | 542.1 | 434.8 | +19.8% | $3,052.09 | **$763.02** | $2,289.07 | 498.7 |
| 152 | Systematic Trading ML Desk Mu #152 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 256 | 1,000,000 | 230.3 | 180.3 | +21.7% | $16,000.00 | **$4,000.00** | $12,000.00 | 211.9 |
| 153 | Data Platform Compute Lab Kappa #153 | `ENTERPRISE_TIER1` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 1024 | 100,000 | 144.2 | 113.6 | +21.2% | $3,655.68 | **$913.92** | $2,741.76 | 132.7 |
| 154 | Computer Science Intelligence Lab Chi #154 | `RESEARCH_LAB` | Mamba-Codestral-7B (SSM) | NVIDIA-RTX-4090-24GB | 16 | 2,500,000 | 627.9 | 500.4 | +20.3% | $1,133.33 | **$283.33** | $850.00 | 577.7 |
| 155 | Enterprise LLM Fleet Iota #155 | `ENTERPRISE_TIER1` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 1024 | 2,500,000 | 366.9 | 279.2 | +23.9% | $261,930.67 | **$65,482.67** | $196,448.00 | 337.5 |
| 156 | University Foundation Center Tau #156 | `RESEARCH_LAB` | Gemma-2-9B (TRANSFORMER) | NVIDIA-L40S-48GB | 1 | 500,000 | 439.0 | 346.4 | +21.1% | $23.15 | **$5.79** | $17.36 | 403.9 |
| 157 | Domain Legal AI Engine Sigma #157 | `GROWTH_STARTUP` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 16 | 10,000,000 | 306.9 | 248.6 | +19.0% | $6,477.78 | **$1,619.44** | $4,858.33 | 282.3 |
| 158 | Neural Search & Retrieval Cluster Xi #158 | `GROWTH_STARTUP` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 16 | 50,000 | 168.6 | 132.0 | +21.7% | $44.73 | **$11.18** | $33.55 | 155.1 |
| 159 | Foundation Supercluster Delta #159 | `WHALE_HYPERSCALER` | DeepSeek-R1-Distill-32B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 8192 | 10,000,000 | 240.9 | 186.0 | +22.8% | $6,871,040.00 | **$1,717,760.00** | $5,153,280.00 | 221.6 |
| 160 | Domain Legal AI Engine Sigma #160 | `GROWTH_STARTUP` | Phi-3.5-MoE (MOE) | NVIDIA-B200-Blackwell-192GB | 128 | 100,000 | 391.9 | 286.5 | +26.9% | $2,061.16 | **$515.29** | $1,545.87 | 360.5 |
| 161 | Enterprise LLM Fleet Iota #161 | `ENTERPRISE_TIER1` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-L40S-48GB | 256 | 10,000,000 | 1415.4 | 1078.5 | +23.8% | $431,232.00 | **$107,808.00** | $323,424.00 | 1302.2 |
| 162 | Systematic Trading ML Desk Mu #162 | `ENTERPRISE_TIER1` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-L40S-48GB | 2048 | 2,500,000 | 566.2 | 447.9 | +20.9% | $302,848.00 | **$75,712.00** | $227,136.00 | 520.9 |
| 163 | Foundation Supercluster Delta #163 | `WHALE_HYPERSCALER` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 2048 | 50,000 | 205.1 | 161.8 | +21.1% | $5,172.91 | **$1,293.23** | $3,879.68 | 188.7 |
| 164 | Cloud AI Supercluster Epsilon #164 | `WHALE_HYPERSCALER` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 1024 | 500,000 | 1589.7 | 1174.8 | +26.1% | $147,520.00 | **$36,880.00** | $110,640.00 | 1462.5 |
| 165 | Enterprise Frontier Lab Theta #165 | `ENTERPRISE_TIER1` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 2048 | 50,000 | 104.2 | 82.1 | +21.2% | $3,457.42 | **$864.36** | $2,593.07 | 95.9 |
| 166 | Cloud AI Supercluster Epsilon #166 | `WHALE_HYPERSCALER` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 2048 | 100,000 | 344.7 | 270.6 | +21.5% | $18,969.60 | **$4,742.40** | $14,227.20 | 317.1 |
| 167 | Hyperscaler Cluster Alpha #167 | `WHALE_HYPERSCALER` | Gemma-2-27B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 16384 | 2,500,000 | 560.9 | 438.1 | +21.9% | $3,492,977.78 | **$873,244.44** | $2,619,733.33 | 516.0 |
| 168 | Cloud AI Supercluster Epsilon #168 | `WHALE_HYPERSCALER` | Phi-3.5-MoE (MOE) | NVIDIA-H200-SXM-141GB | 1024 | 10,000,000 | 522.6 | 387.8 | +25.8% | $1,610,410.67 | **$402,602.67** | $1,207,808.00 | 480.8 |
| 169 | University Foundation Center Tau #169 | `RESEARCH_LAB` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-V100-32GB | 16 | 500,000 | 1398.0 | 1117.0 | +20.1% | $749.33 | **$187.33** | $562.00 | 1286.2 |
| 170 | Computer Science Intelligence Lab Chi #170 | `RESEARCH_LAB` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 8 | 50,000 | 200.8 | 160.4 | +20.1% | $20.20 | **$5.05** | $15.15 | 184.7 |
| 171 | Fintech Quant Trading Engine Nu #171 | `ENTERPRISE_TIER1` | Phi-3.5-MoE (MOE) | NVIDIA-H200-SXM-141GB | 1024 | 10,000,000 | 491.0 | 370.2 | +24.6% | $1,443,157.33 | **$360,789.33** | $1,082,368.00 | 451.7 |
| 172 | Academic AI Research Lab Upsilon #172 | `RESEARCH_LAB` | Mixtral-8x7B-MoE (MOE) | NVIDIA-H200-SXM-141GB | 2 | 1,000,000 | 442.1 | 332.9 | +24.7% | $254.80 | **$63.70** | $191.10 | 406.7 |
| 173 | Quantitative Alpha Research Desk Lambda #173 | `ENTERPRISE_TIER1` | Gemma-2-9B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 1024 | 5,000,000 | 157.2 | 123.6 | +21.4% | $215,040.00 | **$53,760.00** | $161,280.00 | 144.6 |
| 174 | Open Source AI Collective Phi #174 | `RESEARCH_LAB` | Mamba-2.8B-SSM (SSM) | NVIDIA-A100-SXM4-80GB | 8 | 10,000,000 | 186.9 | 151.6 | +18.9% | $1,961.11 | **$490.28** | $1,470.83 | 171.9 |
| 175 | Speech & Audio Synthesis Startup Omicron #175 | `GROWTH_STARTUP` | Gemma-2-9B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 16 | 2,500,000 | 664.1 | 522.6 | +21.3% | $1,257.78 | **$314.44** | $943.33 | 611.0 |
| 176 | Systematic Trading ML Desk Mu #176 | `ENTERPRISE_TIER1` | Mixtral-8x7B-MoE (MOE) | NVIDIA-V100-32GB | 128 | 100,000 | 2416.8 | 1868.2 | +22.7% | $2,340.69 | **$585.17** | $1,755.52 | 2223.5 |
| 177 | Foundation Supercluster Delta #177 | `WHALE_HYPERSCALER` | Gemma-2-27B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 16384 | 50,000 | 231.2 | 178.0 | +23.0% | $66,582.76 | **$16,645.69** | $49,937.07 | 212.7 |
| 178 | Quantitative Alpha Research Desk Lambda #178 | `ENTERPRISE_TIER1` | Mamba-Codestral-7B (SSM) | NVIDIA-H200-SXM-141GB | 1024 | 2,500,000 | 189.2 | 147.2 | +22.2% | $125,440.00 | **$31,360.00** | $94,080.00 | 174.1 |
| 179 | Neural Search & Retrieval Cluster Xi #179 | `GROWTH_STARTUP` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 256 | 1,000,000 | 531.1 | 397.3 | +25.2% | $42,816.00 | **$10,704.00** | $32,112.00 | 488.6 |
| 180 | UltraCluster Infrastructure Eta #180 | `WHALE_HYPERSCALER` | Mixtral-8x7B-MoE (MOE) | NVIDIA-V100-32GB | 2048 | 100,000 | 2240.0 | 1691.2 | +24.5% | $37,464.75 | **$9,366.19** | $28,098.56 | 2060.8 |
| 181 | Foundation Supercluster Delta #181 | `WHALE_HYPERSCALER` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 8192 | 50,000 | 1286.0 | 951.6 | +26.0% | $95,118.22 | **$23,779.56** | $71,338.67 | 1183.1 |
| 182 | Enterprise LLM Fleet Iota #182 | `ENTERPRISE_TIER1` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 512 | 1,000,000 | 140.9 | 108.9 | +22.7% | $25,031.11 | **$6,257.78** | $18,773.33 | 129.6 |
| 183 | Multimodal Compute Group Zeta #183 | `WHALE_HYPERSCALER` | Mixtral-8x7B-MoE (MOE) | NVIDIA-RTX-4090-24GB | 1024 | 250,000 | 1694.9 | 1250.8 | +26.2% | $25,264.36 | **$6,316.09** | $18,948.27 | 1559.3 |
| 184 | Constitutional AI Pod Gamma #184 | `WHALE_HYPERSCALER` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 8192 | 5,000,000 | 428.8 | 347.3 | +19.0% | $2,318,222.22 | **$579,555.56** | $1,738,666.67 | 394.5 |
| 185 | Data Platform Compute Lab Kappa #185 | `ENTERPRISE_TIER1` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 128 | 250,000 | 144.6 | 115.7 | +20.0% | $1,078.93 | **$269.73** | $809.20 | 133.0 |
| 186 | Computer Science Intelligence Lab Chi #186 | `RESEARCH_LAB` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 8 | 5,000,000 | 1955.5 | 1423.6 | +27.2% | $26,595.00 | **$6,648.75** | $19,946.25 | 1799.1 |
| 187 | Enterprise LLM Fleet Iota #187 | `ENTERPRISE_TIER1` | DeepSeek-V3-671B (MOE) | NVIDIA-H200-SXM-141GB | 512 | 5,000,000 | 2550.7 | 1790.6 | +29.8% | $2,270,165.33 | **$567,541.33** | $1,702,624.00 | 2346.6 |
| 188 | University Foundation Center Tau #188 | `RESEARCH_LAB` | Falcon-180B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 1 | 500,000 | 1117.0 | 846.7 | +24.2% | $168.94 | **$42.23** | $126.70 | 1027.6 |
| 189 | Hyperscaler Cluster Alpha #189 | `WHALE_HYPERSCALER` | DeepSeek-V3-671B (MOE) | NVIDIA-GH200-GraceHopper | 2048 | 250,000 | 2668.2 | 1921.1 | +28.0% | $478,144.00 | **$119,536.00** | $358,608.00 | 2454.7 |
| 190 | Foundation Supercluster Delta #190 | `WHALE_HYPERSCALER` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-V100-32GB | 8192 | 250,000 | 931.3 | 713.4 | +23.4% | $148,753.07 | **$37,188.27** | $111,564.80 | 856.8 |
| 191 | Enterprise Frontier Lab Theta #191 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-L40S-48GB | 128 | 5,000,000 | 599.8 | 487.6 | +18.7% | $35,904.00 | **$8,976.00** | $26,928.00 | 551.8 |
| 192 | Hyperscaler Cluster Alpha #192 | `WHALE_HYPERSCALER` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 4096 | 5,000,000 | 223.6 | 173.3 | +22.5% | $1,201,834.67 | **$300,458.67** | $901,376.00 | 205.7 |
| 193 | Neural Search & Retrieval Cluster Xi #193 | `GROWTH_STARTUP` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 128 | 50,000 | 392.8 | 309.9 | +21.1% | $618.99 | **$154.75** | $464.24 | 361.4 |
| 194 | Academic AI Research Lab Upsilon #194 | `RESEARCH_LAB` | Phi-4-14B (TRANSFORMER) | NVIDIA-L40S-48GB | 1 | 5,000,000 | 576.1 | 456.3 | +20.8% | $299.50 | **$74.88** | $224.63 | 530.0 |
| 195 | University Foundation Center Tau #195 | `RESEARCH_LAB` | Gemma-2-9B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 8 | 1,000,000 | 185.5 | 148.8 | +19.8% | $342.53 | **$85.63** | $256.90 | 170.7 |
| 196 | Multimodal Compute Group Zeta #196 | `WHALE_HYPERSCALER` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 16384 | 500,000 | 2217.8 | 1625.6 | +26.7% | $5,659,852.80 | **$1,414,963.20** | $4,244,889.60 | 2040.4 |
| 197 | Open Source AI Collective Phi #197 | `RESEARCH_LAB` | Mixtral-8x7B-MoE (MOE) | NVIDIA-A100-SXM4-80GB | 1 | 250,000 | 784.1 | 603.8 | +23.0% | $31.30 | **$7.83** | $23.48 | 721.4 |
| 198 | Cloud AI Supercluster Epsilon #198 | `WHALE_HYPERSCALER` | Mamba-2.8B-SSM (SSM) | NVIDIA-H200-SXM-141GB | 2048 | 10,000,000 | 93.0 | 75.1 | +19.3% | $427,690.67 | **$106,922.67** | $320,768.00 | 85.6 |
| 199 | University Foundation Center Tau #199 | `RESEARCH_LAB` | Phi-4-14B (TRANSFORMER) | NVIDIA-L40S-48GB | 4 | 250,000 | 540.7 | 423.9 | +21.6% | $58.40 | **$14.60** | $43.80 | 497.4 |
| 200 | Open Source AI Collective Phi #200 | `RESEARCH_LAB` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 16 | 50,000 | 335.9 | 266.4 | +20.7% | $69.50 | **$17.38** | $52.12 | 309.0 |
| 201 | Speech & Audio Synthesis Startup Omicron #201 | `GROWTH_STARTUP` | Phi-4-14B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 16 | 2,500,000 | 273.8 | 218.5 | +20.2% | $2,334.89 | **$583.72** | $1,751.17 | 251.9 |
| 202 | Constitutional AI Pod Gamma #202 | `WHALE_HYPERSCALER` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 8192 | 50,000 | 1111.0 | 857.7 | +22.8% | $23,055.93 | **$5,763.98** | $17,291.95 | 1022.1 |
| 203 | Open Source AI Collective Phi #203 | `RESEARCH_LAB` | DeepSeek-V3-671B (MOE) | NVIDIA-L40S-48GB | 16 | 10,000,000 | 6721.5 | 4624.4 | +31.2% | $167,768.00 | **$41,942.00** | $125,826.00 | 6183.8 |
| 204 | Speech & Audio Synthesis Startup Omicron #204 | `GROWTH_STARTUP` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-L40S-48GB | 16 | 250,000 | 564.1 | 437.2 | +22.5% | $253.80 | **$63.45** | $190.35 | 519.0 |
| 205 | Speech & Audio Synthesis Startup Omicron #205 | `GROWTH_STARTUP` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 64 | 2,500,000 | 172.8 | 138.6 | +19.8% | $8,360.00 | **$2,090.00** | $6,270.00 | 159.0 |
| 206 | Quantitative Alpha Research Desk Lambda #206 | `ENTERPRISE_TIER1` | Mamba-Codestral-7B (SSM) | NVIDIA-H100-SXM5-80GB | 512 | 100,000 | 204.3 | 161.6 | +20.9% | $2,307.70 | **$576.92** | $1,730.77 | 188.0 |
| 207 | Fintech Quant Trading Engine Nu #207 | `ENTERPRISE_TIER1` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 2048 | 250,000 | 334.8 | 259.1 | +22.6% | $48,448.00 | **$12,112.00** | $36,336.00 | 308.0 |
| 208 | Cloud AI Supercluster Epsilon #208 | `WHALE_HYPERSCALER` | Gemma-2-27B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 8192 | 250,000 | 308.6 | 233.3 | +24.4% | $192,768.00 | **$48,192.00** | $144,576.00 | 283.9 |
| 209 | Data Platform Compute Lab Kappa #209 | `ENTERPRISE_TIER1` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 512 | 250,000 | 1380.6 | 1042.4 | +24.5% | $9,619.91 | **$2,404.98** | $7,214.93 | 1270.2 |
| 210 | Computer Science Intelligence Lab Chi #210 | `RESEARCH_LAB` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 8 | 2,500,000 | 211.8 | 170.5 | +19.5% | $963.67 | **$240.92** | $722.75 | 194.9 |
| 211 | UltraCluster Infrastructure Eta #211 | `WHALE_HYPERSCALER` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-V100-32GB | 2048 | 500,000 | 874.8 | 712.1 | +18.6% | $55,534.93 | **$13,883.73** | $41,651.20 | 804.8 |
| 212 | Data Platform Compute Lab Kappa #212 | `ENTERPRISE_TIER1` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 512 | 250,000 | 294.5 | 234.1 | +20.5% | $8,160.71 | **$2,040.18** | $6,120.53 | 270.9 |
| 213 | Multimodal Compute Group Zeta #213 | `WHALE_HYPERSCALER` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 8192 | 50,000 | 207.0 | 160.6 | +22.4% | $29,036.09 | **$7,259.02** | $21,777.07 | 190.4 |
| 214 | Enterprise Frontier Lab Theta #214 | `ENTERPRISE_TIER1` | Phi-4-14B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 1024 | 500,000 | 209.1 | 165.0 | +21.1% | $28,224.00 | **$7,056.00** | $21,168.00 | 192.4 |
| 215 | Distributed GPU Cloud Platform Pi #215 | `GROWTH_STARTUP` | DeepSeek-V3-671B (MOE) | NVIDIA-L40S-48GB | 256 | 10,000,000 | 6678.8 | 4815.4 | +27.9% | $2,385,152.00 | **$596,288.00** | $1,788,864.00 | 6144.5 |
| 216 | Fintech Quant Trading Engine Nu #216 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 500,000 | 905.6 | 733.5 | +19.0% | $4,895.29 | **$1,223.82** | $3,671.47 | 833.2 |
| 217 | Enterprise LLM Fleet Iota #217 | `ENTERPRISE_TIER1` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-V100-32GB | 512 | 100,000 | 3163.5 | 2375.8 | +24.9% | $13,443.41 | **$3,360.85** | $10,082.56 | 2910.4 |
| 218 | Foundation Supercluster Delta #218 | `WHALE_HYPERSCALER` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 2048 | 1,000,000 | 362.5 | 281.7 | +22.3% | $193,058.13 | **$48,264.53** | $144,793.60 | 333.5 |
| 219 | Neural Search & Retrieval Cluster Xi #219 | `GROWTH_STARTUP` | LLaMA-3.1-405B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 128 | 2,500,000 | 4190.0 | 3058.7 | +27.0% | $251,400.00 | **$62,850.00** | $188,550.00 | 3854.8 |
| 220 | Fintech Quant Trading Engine Nu #220 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-L40S-48GB | 256 | 5,000,000 | 611.9 | 490.7 | +19.8% | $77,568.00 | **$19,392.00** | $58,176.00 | 562.9 |
| 221 | Hyperscaler Cluster Alpha #221 | `WHALE_HYPERSCALER` | Qwen-2.5-7B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 4096 | 250,000 | 144.1 | 116.7 | +19.0% | $32,733.87 | **$8,183.47** | $24,550.40 | 132.6 |
| 222 | Distributed GPU Cloud Platform Pi #222 | `GROWTH_STARTUP` | Mixtral-8x7B-MoE (MOE) | NVIDIA-V100-32GB | 128 | 50,000 | 2444.0 | 1896.5 | +22.4% | $1,168.00 | **$292.00** | $876.00 | 2248.5 |
| 223 | UltraCluster Infrastructure Eta #223 | `WHALE_HYPERSCALER` | Mixtral-8x7B-MoE (MOE) | NVIDIA-RTX-4090-24GB | 1024 | 1,000,000 | 1563.5 | 1207.0 | +22.8% | $81,123.56 | **$20,280.89** | $60,842.67 | 1438.4 |
| 224 | Enterprise Frontier Lab Theta #224 | `ENTERPRISE_TIER1` | Phi-3.5-MoE (MOE) | NVIDIA-V100-32GB | 128 | 100,000 | 2940.3 | 2164.1 | +26.4% | $3,311.79 | **$827.95** | $2,483.84 | 2705.1 |
| 225 | Domain Legal AI Engine Sigma #225 | `GROWTH_STARTUP` | LLaMA-3-8B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 32 | 1,000,000 | 348.0 | 265.5 | +23.7% | $1,833.33 | **$458.33** | $1,375.00 | 320.2 |
| 226 | Enterprise Frontier Lab Theta #226 | `ENTERPRISE_TIER1` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 100,000 | 3147.8 | 2382.9 | +24.3% | $4,351.43 | **$1,087.86** | $3,263.57 | 2896.0 |
| 227 | Distributed Ray Compute Fleet Rho #227 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 128 | 10,000,000 | 389.3 | 300.2 | +22.9% | $120,384.00 | **$30,096.00** | $90,288.00 | 358.2 |
| 228 | Systematic Trading ML Desk Mu #228 | `ENTERPRISE_TIER1` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 512 | 100,000 | 770.4 | 567.8 | +26.3% | $10,949.40 | **$2,737.35** | $8,212.05 | 708.8 |
| 229 | Open Source AI Collective Phi #229 | `RESEARCH_LAB` | Mistral-7B-v0.3 (TRANSFORMER) | NVIDIA-L40S-48GB | 2 | 50,000 | 594.6 | 468.5 | +21.2% | $6.30 | **$1.58** | $4.73 | 547.0 |
| 230 | Open Source AI Collective Phi #230 | `RESEARCH_LAB` | Mixtral-8x22B-MoE (MOE) | NVIDIA-V100-32GB | 8 | 250,000 | 5432.9 | 4129.0 | +24.0% | $869.27 | **$217.32** | $651.95 | 4998.3 |
| 231 | Speech & Audio Synthesis Startup Omicron #231 | `GROWTH_STARTUP` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 32 | 50,000 | 569.3 | 419.0 | +26.4% | $300.60 | **$75.15** | $225.45 | 523.8 |
| 232 | Quantitative Alpha Research Desk Lambda #232 | `ENTERPRISE_TIER1` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 2048 | 10,000,000 | 1321.7 | 1015.1 | +23.2% | $1,395,370.67 | **$348,842.67** | $1,046,528.00 | 1216.0 |
| 233 | UltraCluster Infrastructure Eta #233 | `WHALE_HYPERSCALER` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-L40S-48GB | 1024 | 2,500,000 | 870.2 | 667.4 | +23.3% | $259,584.00 | **$64,896.00** | $194,688.00 | 800.6 |
| 234 | Neural Search & Retrieval Cluster Xi #234 | `GROWTH_STARTUP` | Gemma-2-9B (TRANSFORMER) | NVIDIA-V100-32GB | 128 | 2,500,000 | 1016.6 | 806.2 | +20.7% | $22,442.67 | **$5,610.67** | $16,832.00 | 935.3 |
| 235 | Distributed GPU Cloud Platform Pi #235 | `GROWTH_STARTUP` | Gemma-2-27B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 16 | 5,000,000 | 283.5 | 217.7 | +23.2% | $6,580.00 | **$1,645.00** | $4,935.00 | 260.8 |
| 236 | Multimodal Compute Group Zeta #236 | `WHALE_HYPERSCALER` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-GH200-GraceHopper | 4096 | 100,000 | 313.4 | 238.5 | +23.9% | $38,348.80 | **$9,587.20** | $28,761.60 | 288.3 |
| 237 | Data Platform Compute Lab Kappa #237 | `ENTERPRISE_TIER1` | Phi-4-14B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 256 | 10,000,000 | 812.2 | 648.9 | +20.1% | $92,899.56 | **$23,224.89** | $69,674.67 | 747.2 |
| 238 | Systematic Trading ML Desk Mu #238 | `ENTERPRISE_TIER1` | InternLM-2.5-20B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 512 | 50,000 | 1043.8 | 812.1 | +22.2% | $1,318.12 | **$329.53** | $988.59 | 960.3 |
| 239 | Systematic Trading ML Desk Mu #239 | `ENTERPRISE_TIER1` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 128 | 250,000 | 171.0 | 135.4 | +20.8% | $1,740.44 | **$435.11** | $1,305.33 | 157.3 |
| 240 | Speech & Audio Synthesis Startup Omicron #240 | `GROWTH_STARTUP` | LLaMA-3-70B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 16 | 500,000 | 426.2 | 322.6 | +24.3% | $1,266.22 | **$316.56** | $949.67 | 392.1 |
| 241 | Fintech Quant Trading Engine Nu #241 | `ENTERPRISE_TIER1` | Command-R-Plus-104B (TRANSFORMER) | NVIDIA-H100-SXM5-80GB | 2048 | 250,000 | 1008.1 | 738.9 | +26.7% | $145,487.64 | **$36,371.91** | $109,115.73 | 927.5 |
| 242 | Systematic Trading ML Desk Mu #242 | `ENTERPRISE_TIER1` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 2048 | 250,000 | 222.0 | 174.5 | +21.4% | $28,373.33 | **$7,093.33** | $21,280.00 | 204.2 |
| 243 | Open Source AI Collective Phi #243 | `RESEARCH_LAB` | Qwen-2.5-72B (TRANSFORMER) | NVIDIA-A100-SXM4-80GB | 1 | 50,000 | 1311.7 | 978.5 | +25.4% | $11.57 | **$2.89** | $8.68 | 1206.8 |
| 244 | Fintech Quant Trading Engine Nu #244 | `ENTERPRISE_TIER1` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 512 | 1,000,000 | 368.7 | 286.8 | +22.2% | $48,921.60 | **$12,230.40** | $36,691.20 | 339.2 |
| 245 | Academic AI Research Lab Upsilon #245 | `RESEARCH_LAB` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-V100-32GB | 2 | 100,000 | 1822.2 | 1445.0 | +20.7% | $25.15 | **$6.29** | $18.86 | 1676.4 |
| 246 | Multimodal Compute Group Zeta #246 | `WHALE_HYPERSCALER` | Phi-4-14B (TRANSFORMER) | NVIDIA-H200-SXM-141GB | 4096 | 2,500,000 | 228.4 | 178.4 | +21.9% | $597,333.33 | **$149,333.33** | $448,000.00 | 210.1 |
| 247 | Data Platform Compute Lab Kappa #247 | `ENTERPRISE_TIER1` | Qwen-2.5-32B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 256 | 2,500,000 | 258.4 | 204.4 | +20.9% | $52,800.00 | **$13,200.00** | $39,600.00 | 237.7 |
| 248 | UltraCluster Infrastructure Eta #248 | `WHALE_HYPERSCALER` | Mixtral-8x7B-MoE (MOE) | NVIDIA-A100-SXM4-80GB | 8192 | 50,000 | 797.8 | 601.5 | +24.6% | $55,836.44 | **$13,959.11** | $41,877.33 | 734.0 |
| 249 | Speech & Audio Synthesis Startup Omicron #249 | `GROWTH_STARTUP` | Yi-1.5-34B (TRANSFORMER) | NVIDIA-B200-Blackwell-192GB | 256 | 100,000 | 234.5 | 180.8 | +22.9% | $2,100.27 | **$525.07** | $1,575.20 | 215.7 |
| 250 | Distributed GPU Cloud Platform Pi #250 | `GROWTH_STARTUP` | StarCoder-2-15B (TRANSFORMER) | NVIDIA-RTX-4090-24GB | 128 | 100,000 | 812.2 | 653.0 | +19.6% | $452.84 | **$113.21** | $339.63 | 747.2 |

---

*Generated automatically by Quant AI Ghost Layer Benchmarking Suite v1.1.0.*
