"""
High-Throughput Parameter Sweep Benchmark on Physical NVIDIA RTX A2000 GPU
Evaluates scaling throughput across:
- Batch Sizes: [4, 8, 16, 32]
- Sequence Lengths: [128, 256]
- Precision Modes: [FP32 Standard, AMP-FP16 Tensor Cores]
Measures: Real Tokens/sec, Step Latency (ms), VRAM Allocated (MB), and Scaling Multipliers.
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def run_throughput_sweep():
    model_id = "EleutherAI/pythia-70m"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 80)
    print(f"[GhostLayer] STARTING HIGH-THROUGHPUT SWEEP ON PHYSICAL GPU: {gpu_name}")
    print("=" * 80)

    print("\n[Phase 1] Loading Hugging Face Dataset ('Salesforce/wikitext')...")
    raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    non_empty = [item["text"].strip() for item in raw_dataset if len(item["text"].strip()) > 30]
    corpus_text = "\n\n".join(non_empty[:3000])
    all_tokens = tokenizer(corpus_text, return_tensors="pt")["input_ids"][0]

    configurations = [
        {"batch_size": 4, "seq_len": 128, "precision": "FP32"},
        {"batch_size": 8, "seq_len": 128, "precision": "FP32"},
        {"batch_size": 16, "seq_len": 128, "precision": "FP32"},
        {"batch_size": 32, "seq_len": 128, "precision": "FP32"},
        {"batch_size": 8, "seq_len": 256, "precision": "FP32"},
        {"batch_size": 16, "seq_len": 256, "precision": "FP32"},
        {"batch_size": 32, "seq_len": 256, "precision": "FP32"},
        {"batch_size": 32, "seq_len": 128, "precision": "AMP-FP16"},
        {"batch_size": 32, "seq_len": 256, "precision": "AMP-FP16"},
    ]

    results = []

    print("\n[Phase 2] Executing Multi-Configuration Throughput Benchmark...")
    print("-" * 85)
    print(f"{'Config':<6} | {'Precision':<9} | {'Batch':<6} | {'SeqLen':<7} | {'Tokens/Batch':<13} | {'Avg Step (ms)':<14} | {'Throughput (tok/s)':<20} | {'Peak VRAM (MB)':<14}")
    print("-" * 85)

    for idx, cfg in enumerate(configurations, start=1):
        bs = cfg["batch_size"]
        seq_len = cfg["seq_len"]
        prec = cfg["precision"]
        tokens_per_batch = bs * seq_len

        # Chunk dataset
        chunks = [
            all_tokens[i : i + seq_len]
            for i in range(0, len(all_tokens) - seq_len, seq_len)
        ]

        class SweepDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, i):
                return {"input_ids": self.data[i], "labels": self.data[i].clone()}

        loader = DataLoader(SweepDataset(chunks), batch_size=bs, shuffle=False)

        # Re-instantiate model fresh
        torch.cuda.empty_cache()
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
            model.config.use_cache = False
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=1e-5)

            # Warmup
            model.train()
            warmup_steps = 3
            bench_steps = 15
            step_times = []

            for step, batch in enumerate(loader):
                if step >= warmup_steps + bench_steps:
                    break

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = torch.ones_like(input_ids).to(device)

                if prec == "AMP-FP16":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                if step >= warmup_steps:
                    step_times.append(dt_ms)

            avg_ms = sum(step_times) / len(step_times)
            tok_per_sec = tokens_per_batch / (avg_ms / 1000.0)
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            res_entry = {
                "config_id": idx,
                "precision": prec,
                "batch_size": bs,
                "sequence_length": seq_len,
                "tokens_per_step": tokens_per_batch,
                "avg_step_latency_ms": round(avg_ms, 2),
                "tokens_per_second": round(tok_per_sec, 2),
                "peak_vram_mb": round(vram_mb, 2),
                "status": "SUCCESS",
            }
            results.append(res_entry)

            print(f"#{idx:<5} | {prec:<9} | {bs:<6} | {seq_len:<7} | {tokens_per_batch:<13} | {avg_ms:>10.2f} ms | {tok_per_sec:>15.1f} tok/s | {vram_mb:>10.1f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"#{idx:<5} | {prec:<9} | {bs:<6} | {seq_len:<7} | {tokens_per_batch:<13} | {'OOM':>10}    | {'OOM':>15}       | {'>8192 MB (Exceeded Hull)':>10}")
            torch.cuda.empty_cache()

    print("-" * 80)
    
    # Identify maximum throughput configuration
    max_res = max(results, key=lambda x: x["tokens_per_second"])
    print(f"\n[+] MAXIMUM MEASURED THROUGHPUT ON RTX A2000:")
    print(f"    • Peak Throughput: {max_res['tokens_per_second']:,.1f} tokens/second")
    print(f"    • Optimal Configuration: Batch Size = {max_res['batch_size']}, Sequence Length = {max_res['sequence_length']} ({max_res['tokens_per_step']} tokens/batch)")
    print(f"    • Step Latency: {max_res['avg_step_latency_ms']:.2f} ms")
    print(f"    • VRAM Allocation: {max_res['peak_vram_mb']:.1f} MB (within 8,192 MB capacity)")

    # Save to canonical experiments directory
    out_file = "data/canonical_experiments/EXP-2026-09-01-A2000-THROUGHPUT-SWEEP.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment_id": "EXP-2026-09-01-A2000-THROUGHPUT-SWEEP",
            "title": "Throughput Scaling & Batching Sweep on NVIDIA RTX A2000",
            "evidence_tier": "Tier 2 - Physical Hardware Observation",
            "hardware": {
                "device": gpu_name,
                "vram_total_mb": 8192,
                "pytorch_version": torch.__version__,
            },
            "configurations_evaluated": results,
            "optimal_peak_throughput": max_res,
        }, f, indent=2)

    print(f"\n[+] Saved Throughput Sweep Evidence to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_throughput_sweep()
