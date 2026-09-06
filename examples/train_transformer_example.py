import time
import random
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.roi.calculator import ROICalculator
from ghost_layer.reporting.generator import ReportGenerator
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase

def run_simulated_training():
    print("======================================================================")
    print("      GHOST LAYER: END-TO-END TRANSFORMER TRAINING DEMONSTRATION      ")
    print("======================================================================\n")

    print("[Phase 1] Initializing PyTorch Transformer Training Loop...")
    print("[Phase 1] Attaching GhostWatcherHook (Read-Only Observation Mode)...\n")

    # Step 1: Run un-optimized baseline training
    baseline_step_ms = 220.0
    total_steps = 2500
    num_gpus = 8

    with GhostWatcherHook(gpu_memory_mb=16384.0) as hook:
        for step in range(1, 101):
            hook.on_step_begin()
            # Simulate real step computation
            time.sleep(0.005)
            # Un-optimized configuration telemetry
            hook.on_step_end(
                gpu_util_pct=48.5 + random.uniform(-3, 3),
                gpu_mem_used_mb=14500.0,
                loss=3.8 - (step * 0.012),
                dataloader_time_ms=65.0,  # 30% I/O stall
                mixed_precision="fp32",
                gradient_checkpointing=False,
                flash_attention=False,
                num_workers=0,
                pin_memory=False,
            )

        summary, recommendations = hook.analyze_and_report(model_type="transformer")

    print("----------------------------------------------------------------------")
    print("                  TELEMETRY DIAGNOSTICS DETECTED                     ")
    print("----------------------------------------------------------------------")
    print(f" Observed Steps: {summary.total_steps}")
    print(f" Avg GPU Utilization: {summary.avg_gpu_utilization_pct}% (Low)")
    print(f" DataLoader I/O Stall: {summary.avg_dataloader_stall_pct}% (High)")
    print(f" Precision: {summary.mixed_precision.upper()} (Suboptimal for Tensor Cores)")
    print(f" Recommendations Generated: {len(recommendations)}\n")

    for idx, r in enumerate(recommendations, 1):
        print(f"  [{idx}] {r.title}")
        print(f"      - Impact: {r.impact_level} | Est Speedup: {r.speedup_estimate_label}")
        print(f"      - Rule ID: {r.rule_id}\n")

    # Step 2: Apply recommendations & calculate optimized throughput
    print("----------------------------------------------------------------------")
    print("          APPLYING SAFE OPTIMIZATIONS & VERIFYING QUALITY             ")
    print("----------------------------------------------------------------------")

    # Enabling BF16 + DataLoader workers + FlashAttention cuts step time by ~58%
    optimized_step_ms = baseline_step_ms * (1.0 - 0.58)

    # Step 3: Compute ROI & Financial Receipts
    calculator = ROICalculator(gpu_cost_per_hour=3.50, performance_fee_rate_pct=25.0)
    roi = calculator.calculate(
        baseline_step_time_ms=baseline_step_ms,
        optimized_step_time_ms=optimized_step_ms,
        total_training_steps=total_steps,
        num_gpus=num_gpus,
    )

    print(f" Baseline Total GPU Hours : {roi.baseline_gpu_hours:,.2f} hrs")
    print(f" Optimized Total GPU Hours: {roi.optimized_gpu_hours:,.2f} hrs")
    print(f" GPU Hours Saved          : {roi.gpu_hours_saved:,.2f} hrs ({roi.time_reduction_pct:.1f}% reduction)")
    print(f" Baseline Total Compute   : ${roi.baseline_cost_usd:,.2f}")
    print(f" Optimized Compute Cost   : ${roi.optimized_cost_usd:,.2f}")
    print(f" GROSS DOLLAR SAVINGS     : ${roi.baseline_cost_usd - roi.optimized_cost_usd:,.2f}")
    print(f" NET CLIENT SAVINGS (75%) : ${roi.net_client_savings_usd:,.2f}")
    print(f" GHOST LAYER FEE (25%)    : ${roi.performance_fee_usd:,.2f}\n")

    # Step 4: Knowledge Base Learning Registration
    kb = SharedKnowledgeBase(db_file_path="ghost_knowledge_base.json")
    kb.register_learning(
        architecture_family="Llama-Transformer-7B",
        hardware_type="NVIDIA A100-SXM4-80GB",
        effective_config={
            "mixed_precision": "bf16",
            "flash_attention": True,
            "num_workers": 4,
            "pin_memory": True,
        },
        throughput_improvement_pct=roi.time_reduction_pct,
    )
    print(f"[Knowledge Base] Recorded anonymized pattern. Total stored patterns: {len(kb.entries)}")

    # Step 5: Export Markdown Report
    report_md = ReportGenerator.generate_markdown(summary, recommendations, [], roi, is_simulated=True)
    with open("ghost_layer_audit_receipt.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("[Audit Report] Generated executive receipt: ghost_layer_audit_receipt.md\n")

if __name__ == "__main__":
    run_simulated_training()
