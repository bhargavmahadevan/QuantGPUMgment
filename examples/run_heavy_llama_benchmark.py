"""
Heavy LLM (Llama/Meta) Fine-Tuning Validation & Benchmark Runner
=================================================================

Loads advanced pre-trained open-weights LLMs (e.g., Llama-3.2-1B, Qwen) and applies
extreme GhostLayer optimizations including BF16 AMP, SDPA (FlashAttention), and 
Gradient Checkpointing for VRAM reduction. Evaluates throughput, loss divergence,
and auto-generates audit-grade empirical receipts.

Usage:
  python examples/run_heavy_llama_benchmark.py [--model_id MODEL_ID] [--steps STEPS] [--batch_size BATCH_SIZE]
"""

import time
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ghost_layer.verification.verifier import CorrectnessVerifier
from ghost_layer.roi.calculator import ROICalculator
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase

class HeavyTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=64, seq_len=512):
        # A simple synthetic dataset mimicking heavy instruction-tuning tokens
        sample_texts = [
            "GhostLayer dynamically applies Gradient Checkpointing to heavily parameter-bound LLMs to fit within VRAM constraints.",
            "Optimizing Meta Llama architectures requires FlashAttention-2 and mixed-precision (BF16) to maximize Tensor Core utilization.",
            "Mathematical divergence verifiers protect against NaN gradients and sudden spikes in KL divergence.",
            "Executing advanced operator fusion and attention rewrite on large causal language models improves throughput significantly."
        ] * (num_samples // 4 + 1)
        
        encodings = tokenizer(
            sample_texts[:num_samples],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt"
        )
        self.input_ids = encodings["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

def run_heavy_llama_benchmark(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", steps=30, batch_size=2, seq_len=512):
    print("======================================================================")
    print("     GHOST LAYER: HEAVY LLM (META/LLAMA) TRAINING VALIDATION SUITE    ")
    print("======================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"Host Device ({device.type.upper()})"
    
    print(f"[Device Diagnostics] Benchmarking on: {gpu_name}")
    print(f"[HuggingFace Model] Loading tokenizer for: '{model_id}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = HeavyTextDataset(tokenizer, num_samples=steps * batch_size, seq_len=seq_len)
    
    print(f"[Model Architecture] Loading weights... (This may take a moment for heavy models)")
    
    # Load model with explicitly standard attention first for baseline
    model_baseline = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager").to(device)
    param_count = sum(p.numel() for p in model_baseline.parameters())
    print(f"[Model Architecture] Total Parameters: {param_count:,}\n")

    torch.manual_seed(42)
    # Store initial state to perfectly replicate the starting conditions for both phases
    initial_state = {k: v.cpu().clone() for k, v in model_baseline.state_dict().items()}

    # ---------------------------------------------------------------------------
    # Phase 1: Baseline Unoptimized Real LLM Training Pass
    # ---------------------------------------------------------------------------
    print("--- Phase 1: Executing Baseline Loop (FP32, Eager Attention, No Checkpointing) ---")
    optimizer_base = torch.optim.AdamW(model_baseline.parameters(), lr=1e-5)
    baseline_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    baseline_losses = []
    baseline_step_times = []
    model_baseline.train()

    # Track VRAM if on CUDA
    baseline_peak_vram = 0

    for step, input_ids in enumerate(baseline_loader, 1):
        if step > steps:
            break
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
        t0 = time.perf_counter()
        input_ids = input_ids.to(device)
        
        optimizer_base.zero_grad()
        outputs = model_baseline(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer_base.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            baseline_peak_vram = max(baseline_peak_vram, peak_mb)

        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000.0
        baseline_step_times.append(step_ms)
        baseline_losses.append(loss.item())

    avg_base_ms = sum(baseline_step_times) / len(baseline_step_times)
    base_tok_sec = (batch_size * seq_len) / (avg_base_ms / 1000.0)
    print(f"  [Baseline] Avg Step Latency: {avg_base_ms:.2f} ms | Throughput: {base_tok_sec:,.0f} tok/s")
    if torch.cuda.is_available():
        print(f"  [Baseline VRAM] Peak Memory: {baseline_peak_vram:.0f} MB")
    print(f"  [Baseline Loss] Initial Loss: {baseline_losses[0]:.4f} -> Final Loss: {baseline_losses[-1]:.4f}\n")

    del model_baseline, optimizer_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Phase 2: Ghost Layer Optimized Real LLM Training Pass
    # ---------------------------------------------------------------------------
    print("--- Phase 2: Executing Optimized Loop (BF16, SDPA, Gradient Checkpointing) ---")
    torch.manual_seed(42)
    
    # Load model with SDPA (FlashAttention)
    model_opt = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa").to(device)
    model_opt.load_state_dict(initial_state)
    
    # Enable Gradient Checkpointing
    if hasattr(model_opt, "gradient_checkpointing_enable"):
        model_opt.gradient_checkpointing_enable()
        print("  [GhostLayer Optimization] Gradient Checkpointing ENABLED.")
    
    optimizer_opt = torch.optim.AdamW(model_opt.parameters(), lr=1e-5)

    # Enable Pinned Memory
    is_cuda = (device.type == 'cuda')
    opt_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=is_cuda)
    
    use_amp = is_cuda
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"  [GhostLayer Optimization] AMP Precision: {amp_dtype}")

    opt_losses = []
    opt_step_times = []
    opt_peak_vram = 0
    model_opt.train()

    for step, input_ids in enumerate(opt_loader, 1):
        if step > steps:
            break
            
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
        t0 = time.perf_counter()
        input_ids = input_ids.to(device)
        
        optimizer_opt.zero_grad()
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                outputs = model_opt(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
        else:
            outputs = model_opt(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        loss.backward()
        optimizer_opt.step()
        
        if is_cuda:
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            opt_peak_vram = max(opt_peak_vram, peak_mb)

        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000.0
        opt_step_times.append(step_ms)
        opt_losses.append(loss.item())

    avg_opt_ms = sum(opt_step_times) / len(opt_step_times)
    opt_tok_sec = (batch_size * seq_len) / (avg_opt_ms / 1000.0)
    speedup_pct = ((avg_base_ms - avg_opt_ms) / avg_base_ms) * 100.0
    
    print(f"  [Optimized] Avg Step Latency: {avg_opt_ms:.2f} ms | Throughput: {opt_tok_sec:,.0f} tok/s")
    print(f"  [Optimized] Speedup Gain: {speedup_pct:+.1f}%")
    if is_cuda:
        vram_diff = baseline_peak_vram - opt_peak_vram
        print(f"  [Optimized VRAM] Peak Memory: {opt_peak_vram:.0f} MB (Saved {vram_diff:+.0f} MB)")
    print(f"  [Optimized Loss] Initial Loss: {opt_losses[0]:.4f} -> Final Loss: {opt_losses[-1]:.4f}\n")

    # ---------------------------------------------------------------------------
    # Phase 3: Mathematical Safety & Divergence Verification
    # ---------------------------------------------------------------------------
    print("--- Phase 3: Executing Mathematical Safety Verification ---")
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.20, max_allowed_relative_loss_shift=0.05, use_lagrangian_dual=True)
    verification = verifier.verify_trajectories(
        baseline_losses=baseline_losses,
        optimized_losses=opt_losses,
        recommendation_id="RULE_HEAVY_LLM_SDPA_AMP_GC",
        was_auto_applied=True
    )

    print(f"  [Verification Status] Is Safe: {verification.is_safe}")
    print(f"  [Verification Status] Action Taken: {verification.action_taken}")
    print(f"  [Verification Status] Reason: {verification.reason}\n")

    # ---------------------------------------------------------------------------
    # Phase 4: Financial ROI Audit Receipt Generation
    # ---------------------------------------------------------------------------
    print("--- Phase 4: Generating Financial ROI Audit Receipts ---")
    calculator = ROICalculator(gpu_cost_per_hour=3.50 if device.type == 'cuda' else 1.50, performance_fee_rate_pct=25.0)
    roi = calculator.calculate(
        baseline_step_time_ms=avg_base_ms,
        optimized_step_time_ms=avg_opt_ms,
        total_training_steps=100000,
        num_gpus=8 if device.type == 'cuda' else 1
    )

    print(f"  Gross Savings ($)    : ${roi.baseline_cost_usd - roi.optimized_cost_usd:,.2f}")
    print(f"  Net Client Savings   : ${roi.net_client_savings_usd:,.2f}")
    print(f"  Quant Fee (25%)      : ${roi.performance_fee_usd:,.2f}")

    # Register into Knowledge Base ONLY if on actual GPU hardware
    if is_cuda:
        kb = SharedKnowledgeBase(db_file_path="ghost_knowledge_base.json")
        kb.register_learning(
            architecture_family=f"HeavyHF-{model_id}",
            hardware_type=gpu_name,
            effective_config={
                "mixed_precision": "bf16" if use_amp else "fp32",
                "sdpa_flash_attention": True,
                "gradient_checkpointing": hasattr(model_opt, "gradient_checkpointing_enable")
            },
            throughput_improvement_pct=speedup_pct,
            was_verified_safe=verification.is_safe
        )
    else:
        print("\n  [KB Audit] CPU/Smoke test detected. Bypassing Knowledge Base telemetry ingestion to prevent data pollution.")

    report_paths = [
        "real_training_validation_heavy_llama_report.md",
        os.path.join("technical_docs", "03_engineering_and_algorithms", "real_training_validation_heavy_llama_report.md")
    ]
    report_text = f"""# Heavy LLM (Llama/Meta) Empirical Training Audit Receipt

"""
    if not is_cuda:
        report_text += "> [!WARNING]\n> **CPU ENVIRONMENT RECORDED**: This report was generated on a CPU host device. GPU optimizations (AMP, SDPA) typically regress on CPU environments. These numbers reflect structural test validation, not production hardware verification.\n\n"
    report_text += f"""**Model ID:** `{model_id}` ({param_count:,} Parameters)
**Hardware Platform:** {gpu_name}
**Baseline Throughput:** `{base_tok_sec:,.0f} tok/s` ({avg_base_ms:.2f} ms/step)
"""
    if is_cuda:
        report_text += f"**Baseline VRAM:** `{baseline_peak_vram:.0f} MB`\n"
    report_text += f"**Ghost Layer Throughput:** `{opt_tok_sec:,.0f} tok/s` ({avg_opt_ms:.2f} ms/step)\n"
    if is_cuda:
        report_text += f"**Ghost Layer VRAM:** `{opt_peak_vram:.0f} MB` (Saved {vram_diff:+.0f} MB)\n"
    report_text += f"""**Empirical Speedup:** `{speedup_pct:+.1f}%`
**Loss Convergence Verified:** `{verification.is_safe}` (Max Loss Delta: {verification.max_loss_delta:.6f})
**Projected 100k-Step 8x GPU Savings:** `${roi.baseline_cost_usd - roi.optimized_cost_usd:,.2f}`
**Projected Quant 25% Fee:** `${roi.performance_fee_usd:,.2f}`
"""
    for r_path in report_paths:
        os.makedirs(os.path.dirname(r_path) if os.path.dirname(r_path) else ".", exist_ok=True)
        with open(r_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    print(f"\n[Report Generator] Exporting verified receipt to '{report_paths}'...")

    print("======================================================================")
    print("   HEAVY LLM (META/LLAMA) TRAINING BENCHMARK COMPLETED SUCCESSFULLY   ")
    print("======================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heavy LLM Training Validation Runner")
    parser.add_argument("--model_id", type=str, default="distilbert/distilgpt2", help="Hugging Face Model ID (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--steps", type=int, default=30, help="Number of fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step (keep low for heavy models)")
    args = parser.parse_args()

    run_heavy_llama_benchmark(model_id=args.model_id, steps=args.steps, batch_size=args.batch_size)
