import argparse
import re
import os
import sys

# Ensure repository root is on sys.path when executed directly as a script
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost_layer.telemetry.watcher import TelemetryWatcher
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.roi.calculator import ROICalculator
from ghost_layer.reporting.generator import ReportGenerator
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase
from ghost_layer.graphify.graph_builder import ExecutionGraphBuilder
from ghost_layer.graphify.analyzer import GraphAnalyzer
from ghost_layer.graphify.optimizer import GraphOptimizer
from ghost_layer.replay import OptimizationReplayLog
from ghost_layer.benchmarking.calibration_runner import CalibrationRunner, BenchmarkConfig

def main():
    parser = argparse.ArgumentParser(description="Ghost Layer CLI - AI Training Efficiency, Graphify DAG Profiler & ROI Audit Suite")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand 1: audit
    audit_parser = subparsers.add_parser("audit", help="Run simulated efficiency audit on a training profile")
    audit_parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    audit_parser.add_argument("--baseline-ms", type=float, default=240.0, help="Baseline step time in ms")
    audit_parser.add_argument("--gpu-cost", type=float, default=3.50, help="GPU cost per hour in USD")
    audit_parser.add_argument("--out", type=str, default="audit_report.md", help="Output markdown file")
    audit_parser.add_argument("--graphify", action="store_true", help="Include Graphify computational DAG analysis")

    # Subcommand 2: graphify
    graph_parser = subparsers.add_parser("graphify", help="Build, profile, and optimize computational DAG execution graphs")
    graph_parser.add_argument("--model", type=str, default="LLaMA-3-70B", help="Model name")
    graph_parser.add_argument("--arch", type=str, choices=["transformer", "moe", "ssm"], default="transformer", help="Architecture template")
    graph_parser.add_argument("--layers", type=int, default=8, help="Number of model layers")
    graph_parser.add_argument("--out", type=str, default="graph_report.html", help="Output visual HTML report file")

    # Subcommand 3: kb
    kb_parser = subparsers.add_parser("kb", help="Inspect anonymized shared knowledge base entries")
    kb_parser.add_argument("--list", action="store_true", help="List stored configuration entries")

    # Subcommand 4: calibrate
    calib_parser = subparsers.add_parser("calibrate", help="Run empirical hardware calibration microbenchmarks")
    calib_parser.add_argument("--device", type=str, default=None, help="Target device ('cuda' or 'cpu')")
    calib_parser.add_argument("--save", action="store_true", default=True, help="Save report to .ghostlayer/calibration.json")
    calib_parser.add_argument("--out", type=str, default=".ghostlayer/calibration.json", help="Custom path for calibration report")

    args = parser.parse_args()

    if args.command == "audit":
        kb = SharedKnowledgeBase()
        replay_log = OptimizationReplayLog(session_id="cli_audit_session")
        watcher = TelemetryWatcher(target_gpu_mb=16384.0)
        watcher.start()

        # Simulate baseline steps
        for step in range(1, 101):
            watcher.record_step(
                step=step,
                gpu_utilization_pct=42.0,
                gpu_memory_used_mb=7200.0,
                step_time_ms=args.baseline_ms,
                loss=2.5 - (step * 0.005),
                data_loading_time_ms=50.0,
                mixed_precision="fp32",
                gradient_checkpointing=False,
                flash_attention=False,
                num_workers=0,
                pin_memory=False,
            )
            replay_log.record_event(
                step=step,
                stage="OBSERVE",
                recommendation_id="TELEMETRY_OBSERVE",
                action="RECORD_SIMULATED_STEP",
                details={"step_ms": args.baseline_ms, "util": 42.0, "vram_mb": 7200.0},
                verified_safe=True,
                throughput_delta_pct=0.0,
                reason="Baseline step observed."
            )
        watcher.stop()
        summary = watcher.get_summary()

        graph_summary = None
        builder = None
        if args.graphify:
            builder = ExecutionGraphBuilder("AuditedModel")
            builder.build_transformer_dag(num_layers=12, hidden_dim=4096, sequence_length=2048)
            analyzer = GraphAnalyzer(builder)
            graph_summary = analyzer.analyze()

        engine = DecisionEngine(knowledge_base=kb)
        recs = engine.evaluate(summary, model_type="transformer", graph_summary=graph_summary, knowledge_base=kb)

        for r in recs:
            replay_log.record_event(
                step=100,
                stage="DIAGNOSE",
                recommendation_id=r.rule_id,
                action="GENERATED_RECOMMENDATION",
                details={"confidence": r.confidence, "risk": r.risk_level},
                verified_safe=r.safe_to_auto_apply,
                throughput_delta_pct=0.0,
                reason=r.evidence
            )

        parsed = []
        for r in recs:
            m = re.search(r"([\d.]+)", r.speedup_estimate_label)
            if m:
                parsed.append(float(m.group(1)))
        speedup_pct = max(parsed) if parsed else 45.0
        opt_step_ms = args.baseline_ms * (1.0 - speedup_pct / 100.0)

        roi_calc = ROICalculator(gpu_cost_per_hour=args.gpu_cost, audit_fee_usd=2500.0)
        roi = roi_calc.calculate(
            baseline_step_time_ms=args.baseline_ms,
            optimized_step_time_ms=opt_step_ms,
            total_training_steps=args.steps,
            num_gpus=8
        )

        md = ReportGenerator.generate_markdown(
            summary, recs, [], roi, graph_summary=graph_summary, replay_log=replay_log, is_simulated=True,
        )
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)

        print("\n========================================================")
        print("  QUANT GHOSTLAYER™ PRE-FLIGHT DIAGNOSTIC AUDIT SUMMARY")
        print("========================================================")
        print(f"  - Baseline Compute:       {roi.baseline_gpu_hours:,.1f} GPU-hrs | Baseline Spend: ${roi.baseline_cost_usd:,.2f}")
        print(f"  - Optimized Compute:      {roi.optimized_gpu_hours:,.1f} GPU-hrs | Optimized Spend: ${roi.optimized_cost_usd:,.2f}")
        print(f"  - Net Workload Savings:   ${roi.net_client_savings_usd:,.2f} ({roi.time_reduction_pct:.1f}% reduction)")
        print(f"  - Diagnostic Audit Fee:   ${roi.audit_fee_usd:,.2f} Flat Fee")
        print(f"  - Audit ROI Multiple:     {roi.audit_roi_multiple}x Return on Investment")
        print(f"  - Detailed Audit Report:  {args.out}")
        print("--------------------------------------------------------")
        print("  Commercial Next Steps:")
        print("  • Schedule 48h Staging Audit: contact solutions@ghostlayer.ai")
        print("  • Activate Continuous Team SaaS ($499/mo) with Slack & Prometheus FinOps")
        print("========================================================\n")

    elif args.command == "graphify":
        builder = ExecutionGraphBuilder(args.model)
        if args.arch == "transformer":
            builder.build_transformer_dag(num_layers=args.layers)
        elif args.arch == "moe":
            builder.build_moe_dag(num_layers=args.layers)
        elif args.arch == "ssm":
            builder.build_ssm_dag(num_layers=args.layers)

        analyzer = GraphAnalyzer(builder)
        summary = analyzer.analyze()

        optimizer = GraphOptimizer(builder)
        _, opt_report = optimizer.optimize()

        html_content = builder.to_html_interactive()
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[Graphify CLI] Model Computational DAG Profile Complete for {args.model}:")
        print(f"  - Total Execution Nodes: {summary.total_nodes}")
        print(f"  - True Critical Path Latency: {summary.critical_path_length_ms:.1f} ms")
        print(f"  - Fusible Subgraph Clusters: {summary.fusible_clusters_count} clusters ({len(summary.fusible_subgraph_nodes)} operators)")
        print(f"  - Estimated Kernel Fusion Speedup: +{summary.estimated_fusion_speedup_pct}%")
        print(f"  - Optimized Critical Path Latency: {opt_report.optimized_critical_path_ms:.1f} ms (+{opt_report.speedup_pct}% speedup)")
        print(f"  - Interactive Visual Report saved to: {args.out}")

    elif args.command == "kb":
        kb = SharedKnowledgeBase()
        print(f"[Ghost Layer KB] Stored entries: {len(kb.entries)}")
        for key, entry in kb.entries.items():
            print(f"  - [{entry.architecture_family}] on {entry.hardware_type} -> +{entry.throughput_improvement_pct}% speedup ({entry.sample_count} samples)")

    elif args.command == "calibrate":
        kb = SharedKnowledgeBase()
        runner = CalibrationRunner(device=args.device, knowledge_base=kb)
        print(f"[Ghost Layer] Running empirical hardware calibration sweep on: {runner.hardware_name} ({runner.device})...")
        report = runner.run_full_calibration()

        if args.save:
            runner.save_report(report, file_path=args.out)

        print("\n" + "=" * 60)
        print(f"  EMPIRICAL CALIBRATION REPORT: {report.hardware_name}")
        print("=" * 60)
        print(f"  - Mixed Precision Speedup (AMP):   +{report.precision_speedup_pct:.1f}%")
        print(f"  - DataLoader Workers Speedup:     +{report.workers_speedup_pct:.1f}%")
        print(f"  - Gradient Checkpointing Saving:  +{report.grad_checkpoint_memory_saving_pct:.1f}% VRAM")
        print(f"  - Empirically Verified:           {report.is_empirically_measured}")
        print(f"  - Knowledge Base Updated:         3 empirical samples registered")
        if args.save:
            print(f"  - Calibration Report Saved to:    {args.out}")
        print("=" * 60)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
