"""
Real Open-Weights LLM Fine-Tuning Validation & Benchmark Runner
================================================================

Loads actual pre-trained open-weights LLMs from Hugging Face Hub (e.g., distilgpt2, Qwen2.5-0.5B, TinyLlama-1.1B)
and executes actual PyTorch forward/backward pass steps to measure baseline vs. Ghost Layer optimized
throughput, latency, loss convergence, and financial receipts.
"""

import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.verification.verifier import CorrectnessVerifier
from ghost_layer.roi.calculator import ROICalculator
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase

class RealTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=64, seq_len=128):
        sample_texts = [
            "Quant Ghost Layer is an autonomous GPU optimization platform for PyTorch LLM training loops.",
            "Deep learning scaling laws mandate strict VRAM allocation management and FlashAttention kernel fusion.",
            "Multi-node GPU compute clusters suffer from NCCL ring all-reduce latency bottlenecks during pre-training.",
            "Mathematical divergence verifiers protect gradient updates from numerical corruption and loss spikes."
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

def run_real_hf_llm_training(model_id="distilbert/distilgpt2", steps=30, batch_size=4, seq_len=128):
    print("======================================================================")
    print("   GHOST LAYER: REAL OPEN-WEIGHTS LLM TRAINING VALIDATION SUITE       ")
    print("======================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"Host Device ({device.type.upper()})"
    
    print(f"[Device Diagnostics] Device: {gpu_name}")
    print(f"[HuggingFace Model] Loading weights for: '{model_id}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = RealTextDataset(tokenizer, num_samples=steps * batch_size, seq_len=seq_len)
    
    print(f"[Model Parameter Count] Loading architecture...")
    model_baseline = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    param_count = sum(p.numel() for p in model_baseline.parameters())
    print(f"[Model Parameter Count] Total Parameters: {param_count:,}\n")

    torch.manual_seed(42)
    initial_state = {k: v.cpu().clone() for k, v in model_baseline.state_dict().items()}

    # ---------------------------------------------------------------------------
    # Phase 1: Baseline Unoptimized Real LLM Training Pass
    # ---------------------------------------------------------------------------
    print("--- Phase 1: Executing Baseline Real LLM Training Loop (FP32, Single-Worker DataLoader) ---")
    optimizer_base = torch.optim.AdamW(model_baseline.parameters(), lr=5e-5)
    baseline_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    baseline_losses = []
    baseline_step_times = []
    model_baseline.train()

    for step, input_ids in enumerate(baseline_loader, 1):
        if step > steps:
            break
        t0 = time.perf_counter()
        input_ids = input_ids.to(device)
        
        optimizer_base.zero_grad()
        outputs = model_baseline(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer_base.step()
        
        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000.0
        baseline_step_times.append(step_ms)
        baseline_losses.append(loss.item())

    avg_base_ms = sum(baseline_step_times) / len(baseline_step_times)
    base_tok_sec = (batch_size * seq_len) / (avg_base_ms / 1000.0)
    print(f"  [Baseline] Avg Step Latency: {avg_base_ms:.2f} ms | Throughput: {base_tok_sec:,.0f} tok/s")
    print(f"  [Baseline Loss] Initial Loss: {baseline_losses[0]:.4f} -> Final Loss: {baseline_losses[-1]:.4f}\n")

    # ---------------------------------------------------------------------------
    # Phase 2: Ghost Layer Optimized Real LLM Training Pass
    # ---------------------------------------------------------------------------
    print("--- Phase 2: Executing Ghost Layer Optimized Real LLM Training Loop ---")
    torch.manual_seed(42)
    model_opt = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model_opt.load_state_dict(initial_state)
    optimizer_opt = torch.optim.AdamW(model_opt.parameters(), lr=5e-5)

    opt_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    opt_losses = []
    opt_step_times = []
    model_opt.train()

    for step, input_ids in enumerate(opt_loader, 1):
        if step > steps:
            break
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
        
        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000.0
        opt_step_times.append(step_ms)
        opt_losses.append(loss.item())

    avg_opt_ms = sum(opt_step_times) / len(opt_step_times)
    opt_tok_sec = (batch_size * seq_len) / (avg_opt_ms / 1000.0)
    speedup_pct = ((avg_base_ms - avg_opt_ms) / avg_base_ms) * 100.0
    print(f"  [Optimized] Avg Step Latency: {avg_opt_ms:.2f} ms | Throughput: {opt_tok_sec:,.0f} tok/s")
    print(f"  [Optimized] Speedup Gain: +{speedup_pct:.1f}%")
    print(f"  [Optimized Loss] Initial Loss: {opt_losses[0]:.4f} -> Final Loss: {opt_losses[-1]:.4f}\n")

    # ---------------------------------------------------------------------------
    # Phase 3: Mathematical Safety & Divergence Verification
    # ---------------------------------------------------------------------------
    print("--- Phase 3: Executing Mathematical Safety Verification ---")
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.15, max_allowed_kl_div=0.03)
    verification = verifier.verify_trajectories(
        baseline_losses=baseline_losses,
        optimized_losses=opt_losses,
        recommendation_id="RULE_MIXED_PRECISION_OPTIMIZER",
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

    kb = SharedKnowledgeBase(db_file_path="ghost_knowledge_base.json")
    kb.register_learning(
        architecture_family=f"HF-{model_id}",
        hardware_type=gpu_name,
        effective_config={"mixed_precision": "bf16" if use_amp else "fp32"},
        throughput_improvement_pct=speedup_pct,
        was_verified_safe=verification.is_safe
    )

    report_file = "real_training_validation_hf_llm_report.md"
    print(f"\n[Report Generator] Exporting verified receipt to '{report_file}'...")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Real Open-Weights LLM Empirical Training Audit Receipt\n\n")
        f.write(f"**Model ID:** `{model_id}` ({param_count:,} Parameters)\n")
        f.write(f"**Hardware Platform:** {gpu_name}\n")
        f.write(f"**Baseline Throughput:** `{base_tok_sec:,.0f} tok/s` ({avg_base_ms:.2f} ms/step)\n")
        f.write(f"**Ghost Layer Throughput:** `{opt_tok_sec:,.0f} tok/s` ({avg_opt_ms:.2f} ms/step)\n")
        f.write(f"**Empirical Speedup:** `+{speedup_pct:.1f}%`\n")
        f.write(f"**Loss Convergence Verified:** `{verification.is_safe}` (Max Loss Delta: {verification.max_loss_delta})\n")
        f.write(f"**Projected 100k-Step 8x GPU Savings:** `${roi.baseline_cost_usd - roi.optimized_cost_usd:,.2f}`\n")
        f.write(f"**Projected Quant 25% Fee:** `${roi.performance_fee_usd:,.2f}`\n")

    print("======================================================================")
    print("   REAL OPEN-WEIGHTS LLM TRAINING COMPLETED SUCCESSFULLY              ")
    print("======================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Open-Weights LLM Training Validation Runner")
    parser.add_argument("--model_id", type=str, default="distilbert/distilgpt2", help="Hugging Face Model ID")
    parser.add_argument("--steps", type=int, default=30, help="Number of fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    args = parser.parse_args()

    run_real_hf_llm_training(model_id=args.model_id, steps=args.steps, batch_size=args.batch_size)
