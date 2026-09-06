"""
Multi-Run GPU Benchmark Verification Script
===========================================

Executes 5 consecutive trials of the real LLM benchmark on physical CUDA GPU hardware
to compute the mean, min, max, and variance of the measured speedup gain.
"""

import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ghost_layer.verification.verifier import CorrectnessVerifier

class PyTorchLLMModel(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.embedding(x)
        h = self.transformer(h)
        return self.head(h)

def run_trial(trial_num, num_steps=30, warmup_steps=5, batch_size=4, seq_len=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32000
    
    gen = torch.Generator().manual_seed(42 + trial_num)
    total_samples = batch_size * (num_steps + warmup_steps)
    raw_tokens = torch.randint(0, vocab_size, (total_samples, seq_len + 1), generator=gen)
    input_ids = raw_tokens[:, :-1]
    target_ids = raw_tokens[:, 1:]
    dataset = TensorDataset(input_ids, target_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # 1. Baseline FP32 Run
    torch.manual_seed(42 + trial_num)
    model_base = PyTorchLLMModel().to(device)
    # CRITICAL FIX: Capture state dict at step 0 BEFORE any parameter updates occur
    initial_weights = {k: v.cpu().clone() for k, v in model_base.state_dict().items()}
    optimizer_base = torch.optim.AdamW(model_base.parameters(), lr=1e-4)

    baseline_times = []
    baseline_losses = []
    
    model_base.train()
    loader_iter = iter(loader)

    # Discarded Warmup
    for _ in range(warmup_steps):
        batch_x, batch_y = next(loader_iter)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer_base.zero_grad(set_to_none=True)
        logits = model_base(batch_x)
        loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        optimizer_base.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measured Steps
    for _ in range(num_steps):
        batch_x, batch_y = next(loader_iter)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        t0 = time.perf_counter()
        optimizer_base.zero_grad(set_to_none=True)
        logits = model_base(batch_x)
        loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        optimizer_base.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline_times.append((t1 - t0) * 1000.0)
        baseline_losses.append(loss.item())

    del model_base, optimizer_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Optimized AMP FP16 Run
    model_opt = PyTorchLLMModel().to(device)
    model_opt.load_state_dict(initial_weights)
    optimizer_opt = torch.optim.AdamW(model_opt.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    opt_times = []
    opt_losses = []

    model_opt.train()
    loader_iter = iter(loader)

    # Discarded Warmup
    for _ in range(warmup_steps):
        batch_x, batch_y = next(loader_iter)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer_opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model_opt(batch_x)
                loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer_opt)
            scaler.update()
            torch.cuda.synchronize()
        else:
            logits = model_opt(batch_x)
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer_opt.step()

    # Measured Steps
    for _ in range(num_steps):
        batch_x, batch_y = next(loader_iter)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        t0 = time.perf_counter()
        optimizer_opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model_opt(batch_x)
                loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer_opt)
            scaler.update()
            torch.cuda.synchronize()
        else:
            logits = model_opt(batch_x)
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer_opt.step()
        t1 = time.perf_counter()
        opt_times.append((t1 - t0) * 1000.0)
        opt_losses.append(loss.item())

    del model_opt, optimizer_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_base_ms = sum(baseline_times) / len(baseline_times)
    avg_opt_ms = sum(opt_times) / len(opt_times)
    base_tok_s = (batch_size * seq_len) / (avg_base_ms / 1000.0)
    opt_tok_s = (batch_size * seq_len) / (avg_opt_ms / 1000.0)
    speedup_pct = ((avg_base_ms - avg_opt_ms) / avg_base_ms) * 100.0

    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.10, max_allowed_relative_loss_shift=0.05, use_lagrangian_dual=True)
    verification = verifier.verify_trajectories(
        baseline_losses=baseline_losses,
        optimized_losses=opt_losses,
        recommendation_id="RULE_MULTI_RUN_TEST",
        was_auto_applied=True
    )

    return {
        "trial": trial_num,
        "base_ms": avg_base_ms,
        "opt_ms": avg_opt_ms,
        "base_tok_s": base_tok_s,
        "opt_tok_s": opt_tok_s,
        "speedup_pct": speedup_pct,
        "is_safe": verification.is_safe,
        "max_loss_delta": verification.max_loss_delta,
        "p_value": verification.p_value
    }

def main():
    print("======================================================================")
    print("      MULTI-RUN GPU BENCHMARK VERIFICATION (5 CONSECUTIVE TRIALS)    ")
    print("======================================================================\n")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Hardware Device] {device_name}\n")

    trials = []
    for t in range(1, 6):
        print(f"--- Running Trial {t} / 5 ---")
        res = run_trial(t)
        trials.append(res)
        print(f"  Trial {t}: Baseline {res['base_ms']:.2f} ms ({res['base_tok_s']:,.0f} tok/s) | Opt {res['opt_ms']:.2f} ms ({res['opt_tok_s']:,.0f} tok/s) | Speedup: {res['speedup_pct']:+.2f}% | Safe: {res['is_safe']}")

    speedups = [r["speedup_pct"] for r in trials]
    avg_speedup = sum(speedups) / len(speedups)
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    variance = sum((x - avg_speedup) ** 2 for x in speedups) / len(speedups)
    std_dev = variance ** 0.5

    print("\n======================================================================")
    print("                 EMPIRICAL MULTI-RUN SUMMARY STATS                    ")
    print("======================================================================")
    print(f"  Total Trials Executed  : {len(trials)}")
    print(f"  Mean Speedup Gain      : {avg_speedup:+.2f}%")
    print(f"  Min Speedup Gain       : {min_speedup:+.2f}%")
    print(f"  Max Speedup Gain       : {max_speedup:+.2f}%")
    print(f"  Standard Deviation     : {std_dev:.2f}%")
    print(f"  Consistency Metric     : {'CONSTANT / HIGHLY STABLE' if std_dev < 3.0 else 'VARIABLE'}")
    print("======================================================================\n")

    report_lines = [
        "# Real Multi-Run GPU Empirical Benchmark Receipt\n",
        f"**Hardware Platform:** {device_name}\n",
        "**Number of Consecutive Trials:** 5\n\n",
        "| Trial # | Baseline (ms/step) | Baseline (tok/s) | Ghost Layer Opt (ms/step) | Ghost Layer Opt (tok/s) | Speedup Gain | Safe Status |\n",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    ]
    for r in trials:
        report_lines.append(f"| Trial {r['trial']} | {r['base_ms']:.2f} ms | {r['base_tok_s']:,.0f} tok/s | {r['opt_ms']:.2f} ms | {r['opt_tok_s']:,.0f} tok/s | **{r['speedup_pct']:+.2f}%** | `{r['is_safe']}` |\n")

    report_lines.append(f"\n### Empirical Summary Statistics\n")
    report_lines.append(f"- **Mean Speedup:** `{avg_speedup:+.2f}%`\n")
    report_lines.append(f"- **Range:** `{min_speedup:+.2f}%` to `{max_speedup:+.2f}%`\n")
    report_lines.append(f"- **Standard Deviation:** `{std_dev:.2f}%`\n")
    report_lines.append(f"- **Stability Assessment:** `{ 'Highly Stable / Consistent' if std_dev < 3.0 else 'Host Timing Variance Detected' }`\n")

    report_paths = [
        "real_multi_run_llm_report.md",
        os.path.join("technical_docs", "03_engineering_and_algorithms", "real_multi_run_llm_report.md")
    ]
    for r_path in report_paths:
        os.makedirs(os.path.dirname(r_path) if os.path.dirname(r_path) else ".", exist_ok=True)
        with open(r_path, "w", encoding="utf-8") as f:
            f.writelines(report_lines)

    print(f"[Report Generator] Multi-run empirical receipt written to {report_paths}")

if __name__ == "__main__":
    main()
