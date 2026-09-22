"""
CEO Evidence Engine — HuggingFace Training Evidence Runner

Runs a REAL HuggingFace model training loop with GhostTrainerCallback attached.
Captures step-level telemetry, generates decision records, and produces SHA-256
verified evidence artifacts.

Designed for the iterative "Run → Assess → Fix → Improve" workflow:
  1. Pick a model
  2. Run N training steps
  3. Inspect the evidence artifact
  4. Fix issues found
  5. Move to the next model

Usage:
    python -m ceo_evidence_engine.hf_training_evidence_runner --model EleutherAI/pythia-70m --steps 30
    python -m ceo_evidence_engine.hf_training_evidence_runner --model EleutherAI/pythia-160m --steps 20 --bf16
"""

import os
import sys
import json
import time
import uuid
import hashlib
import argparse
import platform
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# --------------------------------------------------------------------------- #
# Lazy imports — fail early with clear messages
# --------------------------------------------------------------------------- #

def _check_dependency(name: str) -> None:
    """Fail fast with a clear message if a required package is missing."""
    try:
        __import__(name)
    except ImportError:
        print(f"[FATAL] Required package '{name}' is not installed.")
        print(f"  Install via: pip install {name}")
        sys.exit(1)


def _get_hardware_info() -> Dict[str, Any]:
    """Capture hardware telemetry without fabrication."""
    import torch
    cuda_available = torch.cuda.is_available()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "total_vram_mb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2
        ) if cuda_available else 0.0,
    }


def run_training_evidence(
    model_id: str = "EleutherAI/pythia-70m",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_steps: int = 30,
    batch_size: int = 4,
    use_bf16: bool = False,
    use_fp16: bool = False,
    gradient_checkpointing: bool = False,
    learning_rate: float = 5e-5,
    output_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a real HuggingFace Trainer run with GhostTrainerCallback attached.

    Returns the evidence artifact dictionary, or None on failure.
    """
    # ---- Dependency checks ---- #
    for dep in ["torch", "transformers", "datasets"]:
        _check_dependency(dep)

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset
    from ghost_layer.callback import GhostTrainerCallback

    # ---- Banner ---- #
    hw_info = _get_hardware_info()
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 64)
    print("  GHOSTLAYER HF TRAINING EVIDENCE RUNNER")
    print("=" * 64)
    print(f"  Run ID      : {run_id}")
    print(f"  Model       : {model_id}")
    print(f"  Dataset     : {dataset_name}/{dataset_config}")
    print(f"  Steps       : {num_steps}")
    print(f"  Batch Size  : {batch_size}")
    print(f"  Precision   : {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
    print(f"  Hardware    : {hw_info['gpu_name']}")
    print(f"  CUDA        : {hw_info['cuda_available']} ({hw_info['cuda_version']})")
    print(f"  VRAM        : {hw_info['total_vram_mb']} MB")
    print("-" * 64)

    # ---- Execution mode ---- #
    execution_mode = "LIVE_MEASURED_RUN" if hw_info["cuda_available"] else "CPU_MEASURED_RUN"
    device_target = "cuda" if hw_info["cuda_available"] else "cpu"

    # ---- Load tokenizer & model ---- #
    print(f"\n-> Stage 1: Loading tokenizer and model ({model_id})...")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  [OK] Model loaded: {param_count:,} parameters ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  [FATAL] Model load failed: {e}")
        return None

    # ---- Load dataset ---- #
    print(f"\n-> Stage 2: Loading dataset ({dataset_name}/{dataset_config})...")
    t0 = time.time()
    try:
        raw_dataset = load_dataset(dataset_name, dataset_config, split="train")
        print(f"  [OK] Dataset loaded: {len(raw_dataset)} examples ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  [FATAL] Dataset load failed: {e}")
        return None

    # ---- Tokenize ---- #
    print("\n-> Stage 3: Tokenizing dataset...")
    t0 = time.time()
    block_size = 128  # Small block size for evidence gathering (not training quality)

    def tokenize_fn(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_special_tokens_mask=True,
        )
        return outputs

    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )
    # Filter out empty sequences
    tokenized = tokenized.filter(lambda x: sum(x["attention_mask"]) > 10)
    # Limit dataset size to what we need
    max_samples = min(len(tokenized), num_steps * batch_size * 2)
    tokenized = tokenized.select(range(max_samples))
    print(f"  [OK] Tokenized {len(tokenized)} samples (block_size={block_size}, {time.time() - t0:.1f}s)")

    # ---- Setup output ---- #
    if output_dir is None:
        safe_model_name = model_id.replace("/", "_").replace("-", "_")
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sample_outputs",
            f"training_{safe_model_name}_{run_id}",
        )
    os.makedirs(output_dir, exist_ok=True)

    # ---- Training arguments ---- #
    print(f"\n-> Stage 4: Configuring Trainer ({num_steps} steps, {execution_mode})...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,  # Log every step for full telemetry
        save_strategy="no",  # Don't save checkpoints during evidence runs
        report_to="none",  # No W&B or other reporting
        bf16=use_bf16 and hw_info["cuda_available"],
        fp16=use_fp16 and hw_info["cuda_available"],
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=0,  # Start simple, optimize in later runs
        seed=42,
        remove_unused_columns=False,
    )

    # ---- Data collator ---- #
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )

    # ---- GhostLayer callback ---- #
    ghost_callback = GhostTrainerCallback(
        cluster_name=f"evidence-{hw_info['gpu_name'].replace(' ', '_')}",
        gpu_memory_mb=hw_info["total_vram_mb"] if hw_info["total_vram_mb"] > 0 else 8192.0,
        gpu_cost_per_hour=3.50,
        output_report_path=os.path.join(output_dir, "ghost_executive_audit.html"),
        auto_generate_report=True,
    )

    # ---- Run training ---- #
    print(f"\n-> Stage 5: RUNNING TRAINING ({num_steps} steps)...")
    print("-" * 64)
    wall_start = time.time()
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
            callbacks=[ghost_callback],
        )
        train_result = trainer.train()
        wall_end = time.time()
        wall_time_sec = wall_end - wall_start
        print("-" * 64)
        print(f"  [OK] Training completed in {wall_time_sec:.1f}s")
    except Exception as e:
        print(f"\n  [FATAL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ---- Extract telemetry from GhostLayer ---- #
    print(f"\n-> Stage 6: Extracting GhostLayer telemetry...")
    hook = ghost_callback.hook

    try:
        summary, recommendations = hook.analyze_and_report(
            model_type="transformer",
            client_name=f"Evidence Run: {model_id}",
        )
    except Exception as e:
        print(f"  [WARNING] analyze_and_report failed: {e}")
        # Still capture what we can from the watcher
        summary = hook.watcher.get_summary()
        recommendations = []

    # ---- Extract training metrics ---- #
    train_metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
    final_loss = train_metrics.get("train_loss", None)

    # Extract step-by-step loss from trainer log history
    loss_trajectory: List[float] = []
    step_times_ms: List[float] = []
    if hasattr(trainer.state, "log_history"):
        for entry in trainer.state.log_history:
            if "loss" in entry:
                loss_trajectory.append(entry["loss"])

    # Extract step times from GhostLayer watcher
    for snap in hook.watcher.snapshots:
        step_times_ms.append(snap.step_time_ms)

    avg_step_time_ms = summary.avg_step_time_ms if summary.avg_step_time_ms > 0 else 0.0
    peak_vram_mb = summary.peak_gpu_memory_mb

    print(f"  Steps captured   : {summary.total_steps}")
    print(f"  Avg step time    : {avg_step_time_ms:.2f} ms")
    print(f"  Peak VRAM        : {peak_vram_mb:.1f} MB")
    print(f"  Final loss       : {final_loss}")
    print(f"  Recommendations  : {len(recommendations)}")
    for i, rec in enumerate(recommendations):
        print(f"    [{i+1}] {rec.title} (confidence: {rec.confidence}, impact: {rec.impact_level})")

    # ---- Build evidence artifact ---- #
    print(f"\n-> Stage 7: Building SHA-256 verified evidence artifact...")

    evidence_payload = {
        "model_key": model_id,
        "timestamp": timestamp,
        "results": {
            "execution_mode": execution_mode,
            "hardware": hw_info,
            "training_config": {
                "model_id": model_id,
                "dataset": f"{dataset_name}/{dataset_config}",
                "num_steps": num_steps,
                "batch_size": batch_size,
                "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
                "gradient_checkpointing": gradient_checkpointing,
                "learning_rate": learning_rate,
                "block_size": block_size,
            },
            "metrics": {
                "total_steps_captured": summary.total_steps,
                "avg_step_time_ms": round(avg_step_time_ms, 3),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "final_training_loss": final_loss,
                "wall_time_sec": round(wall_time_sec, 2),
                "loss_trajectory": loss_trajectory,
                "step_times_ms": [round(t, 3) for t in step_times_ms[:num_steps]],
            },
            "ghostlayer_telemetry": {
                "mixed_precision_detected": summary.mixed_precision,
                "gpu_utilization_avg_pct": round(summary.avg_gpu_utilization_pct, 2),
                "vram_headroom_pct": round(
                    (1.0 - peak_vram_mb / max(1.0, hw_info["total_vram_mb"])) * 100.0, 2
                ) if hw_info["total_vram_mb"] > 0 else None,
            },
            "recommendations": [
                {
                    "rule_id": r.rule_id,
                    "title": r.title,
                    "confidence": r.confidence,
                    "impact_level": r.impact_level,
                    "risk_level": r.risk_level,
                    "speedup_estimate": r.speedup_estimate_label,
                    "safe_to_auto_apply": r.safe_to_auto_apply,
                }
                for r in recommendations
            ],
            "recommendation_count": len(recommendations),
        },
    }

    # SHA-256 hash for audit provenance
    hash_input = json.dumps(evidence_payload, sort_keys=True).encode("utf-8")
    sha256_hash = hashlib.sha256(hash_input).hexdigest()

    evidence_artifact = {
        "benchmark_category": "REAL_MODEL_TRAINING_TELEMETRY",
        "training_efficiency_benchmark": True,
        "is_empirically_measured": True,
        "metadata": evidence_payload,
        "reproducibility": {
            "run_id": run_id,
            "sha256_audit_hash": sha256_hash,
            "git_commit": _get_git_commit(),
        },
    }

    # ---- Save artifacts ---- #
    # 1. Save to sample_outputs
    evidence_file = os.path.join(output_dir, "training_evidence.json")
    with open(evidence_file, "w", encoding="utf-8") as f:
        json.dump(evidence_artifact, f, indent=2)
    print(f"  [SAVED] {evidence_file}")

    # 2. Sync to canonical .ghostlayer/ store
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ghostlayer_dir = os.path.join(root_dir, ".ghostlayer")
    os.makedirs(ghostlayer_dir, exist_ok=True)
    safe_name = model_id.replace("/", "_")
    canonical_file = os.path.join(ghostlayer_dir, f"training_evidence_{safe_name}.json")
    with open(canonical_file, "w", encoding="utf-8") as f:
        json.dump(evidence_artifact, f, indent=2)
    print(f"  [SAVED] {canonical_file}")

    # ---- Post-run assessment checklist ---- #
    print("\n" + "=" * 64)
    print("  POST-RUN ASSESSMENT CHECKLIST")
    print("=" * 64)
    _print_assessment(evidence_artifact, hw_info, recommendations, summary)

    return evidence_artifact


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _print_assessment(
    artifact: Dict[str, Any],
    hw_info: Dict[str, Any],
    recommendations: list,
    summary: Any,
) -> None:
    """Print a structured assessment checklist for manual review after each run."""
    results = artifact["metadata"]["results"]
    metrics = results["metrics"]
    telemetry = results["ghostlayer_telemetry"]

    checks = []

    # 1. Execution mode
    mode = results["execution_mode"]
    if mode == "LIVE_MEASURED_RUN":
        checks.append(("✅", "Execution Mode", f"{mode} on {hw_info['gpu_name']}"))
    else:
        checks.append(("⚠️", "Execution Mode", f"{mode} — no GPU speedup data possible"))

    # 2. Steps captured
    captured = metrics["total_steps_captured"]
    expected = results["training_config"]["num_steps"]
    if captured >= expected:
        checks.append(("✅", "Steps Captured", f"{captured}/{expected}"))
    else:
        checks.append(("❌", "Steps Captured", f"Only {captured}/{expected} — telemetry loss"))

    # 3. Loss trajectory
    trajectory = metrics.get("loss_trajectory", [])
    if len(trajectory) >= 2 and trajectory[-1] < trajectory[0]:
        checks.append(("✅", "Loss Trajectory", f"Decreasing: {trajectory[0]:.4f} → {trajectory[-1]:.4f}"))
    elif len(trajectory) >= 2:
        checks.append(("⚠️", "Loss Trajectory", f"Not decreasing: {trajectory[0]:.4f} → {trajectory[-1]:.4f}"))
    else:
        checks.append(("❌", "Loss Trajectory", "Insufficient data"))

    # 4. GPU utilization
    gpu_util = telemetry["gpu_utilization_avg_pct"]
    if hw_info["cuda_available"] and gpu_util > 0:
        checks.append(("✅", "GPU Utilization", f"{gpu_util:.1f}% average"))
    elif hw_info["cuda_available"]:
        checks.append(("⚠️", "GPU Utilization", "0% — probe may not be capturing (pynvml missing?)"))
    else:
        checks.append(("ℹ️", "GPU Utilization", "N/A (CPU mode)"))

    # 5. VRAM usage
    vram_headroom = telemetry.get("vram_headroom_pct")
    if vram_headroom is not None and vram_headroom > 0:
        checks.append(("✅", "VRAM Headroom", f"{vram_headroom:.1f}% free"))
    elif hw_info["cuda_available"]:
        checks.append(("⚠️", "VRAM Usage", "Not captured"))
    else:
        checks.append(("ℹ️", "VRAM Usage", "N/A (CPU mode)"))

    # 6. Recommendations
    if len(recommendations) > 0:
        checks.append(("✅", "Recommendations", f"{len(recommendations)} generated"))
    else:
        checks.append(("❌", "Recommendations", "None generated — decision engine issue?"))

    # 7. SHA-256 hash
    checks.append(("✅", "SHA-256 Hash", artifact["reproducibility"]["sha256_audit_hash"][:32] + "..."))

    # 8. Precision check
    precision = results["training_config"]["precision"]
    detected = telemetry["mixed_precision_detected"]
    if precision == detected:
        checks.append(("✅", "Precision Match", f"Config={precision}, Detected={detected}"))
    else:
        checks.append(("⚠️", "Precision Mismatch", f"Config={precision}, Detected={detected}"))

    # Print results
    for icon, label, detail in checks:
        print(f"  {icon} {label:20s}: {detail}")

    # Suggest next actions
    issues = [c for c in checks if c[0] in ("❌", "⚠️")]
    if issues:
        print(f"\n  ⚠️  {len(issues)} issue(s) found — fix before proceeding to next model.")
        for icon, label, detail in issues:
            print(f"    → {label}: {detail}")
    else:
        print(f"\n  ✅ All checks passed — safe to proceed to next model.")

    print("=" * 64 + "\n")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GhostLayer HF Training Evidence Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ceo_evidence_engine.hf_training_evidence_runner --model EleutherAI/pythia-70m --steps 30
  python -m ceo_evidence_engine.hf_training_evidence_runner --model EleutherAI/pythia-160m --steps 20 --bf16
  python -m ceo_evidence_engine.hf_training_evidence_runner --model distilbert/distilgpt2 --steps 25
        """,
    )
    parser.add_argument(
        "--model", type=str, default="EleutherAI/pythia-70m",
        help="HuggingFace model ID (default: EleutherAI/pythia-70m)",
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        help="HuggingFace dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config", type=str, default="wikitext-2-raw-v1",
        help="Dataset config/subset (default: wikitext-2-raw-v1)",
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of training steps (default: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    run_training_evidence(
        model_id=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        num_steps=args.steps,
        batch_size=args.batch_size,
        use_bf16=args.bf16,
        use_fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
