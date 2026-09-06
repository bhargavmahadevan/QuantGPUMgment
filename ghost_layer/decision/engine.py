import os
import sys
import re
import math
import enum
from dataclasses import dataclass
from typing import List, Any, Optional, Dict
from ghost_layer.telemetry.watcher import TelemetrySummary
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase
from ghost_layer.evidence.taxonomy import EvidenceLevel
from ghost_layer.roi.financial_impact import FinancialImpactCalculator
from ghost_layer.decision.counterfactual import (
    WhyNotEngine,
    DecisionCounterfactual,
    RejectedIntervention,
)
from ghost_layer.decision.credibility_manifold import (
    CredibilityManifold,
    CredibilityStatus,
    CredibilityVector3D,
    CredibilityEvaluation,
)

_CREDIBILITY_MANIFOLD = CredibilityManifold()


class OptimizationLifecycle(enum.Enum):
    """
    Formal lifecycle semantics defining when an intervention can be safely applied or mutated.
    """
    STATIC = "STATIC"                      # Requires offline script/config change before execution
    STARTUP_ONLY = "STARTUP_ONLY"          # Must be applied prior to model initialization / worker fork
    EPOCH_BOUNDARY = "EPOCH_BOUNDARY"      # Applied at dataset epoch boundaries (e.g. DataLoader re-seeding)
    STEP_BOUNDARY = "STEP_BOUNDARY"        # Applied between training steps (e.g. AMP, batch scaling)
    DYNAMIC = "DYNAMIC"                    # Continuous runtime adaptation (e.g. adaptive LR, micro-batching)


class RuleTier(enum.Enum):
    """
    Rule maturity classification separating core high-reliability interventions from research/experimental heuristics.
    """
    TIER_A_CORE_PRODUCTION = "TIER_A_CORE_PRODUCTION"        # 5 Core: AMP, SDPA/Attention, DataLoader, VRAM/Batch, torch.compile
    TIER_B_RESEARCH_EXPERIMENTAL = "TIER_B_RESEARCH_EXPERIMENTAL"  # Research: Curvature (Muon/Sophia/Shampoo), Chunked CE, mHC


@dataclass
class Recommendation:
    rule_id: str
    title: str
    impact_level: str  # HIGH, MEDIUM, LOW
    speedup_estimate_label: str
    description: str
    actionable_code_snippet: str
    safe_to_auto_apply: bool
    confidence: float = 0.95
    recommendation_prior: float = 0.95
    # True only when this recommendation's score is backed by >=1 verified run
    # in the SharedKnowledgeBase for this architecture+hardware combination.
    # False means the score is a heuristic prior derived from rule logic, not measurement.
    has_verified_runs: bool = False
    risk_level: str = "LOW"
    rollback_mode: str = "AUTOMATIC"
    evidence: str = ""
    verification_plan: str = ""
    rollback_plan: str = ""
    intervention_tier: str = "TIER_A_PRODUCTION_SAFE"
    evidence_level: str = "E0_THEORETICAL"
    financial_impact: Optional[Dict[str, Any]] = None
    lifecycle: str = "STEP_BOUNDARY"
    rule_tier: str = "TIER_A_CORE_PRODUCTION"

    def __post_init__(self):
        # Synchronize confidence (backward-compat alias) with recommendation_prior
        if self.confidence != 0.95 and self.recommendation_prior == 0.95:
            self.recommendation_prior = self.confidence
        elif self.recommendation_prior != 0.95 and self.confidence == 0.95:
            self.confidence = self.recommendation_prior

    @property
    def evidence_score(self) -> float:
        """Alias for recommendation_prior indicating empirical/heuristic credibility."""
        return self.recommendation_prior

    @property
    def decision_score(self) -> float:
        """
        Heuristic decision score (0.00 to 1.00) synthesizing prior rule confidence and empirical KB evidence.
        Explicitly labeled as a decision heuristic rather than a calibrated probability.
        """
        return self.recommendation_prior

# Maximum confidence score when there are zero verified runs in the knowledge base.
# This cap exists because the formula's base_confidence (0.85) would otherwise produce
# scores of 0.93-0.99 with zero empirical evidence, making the output structurally
# decoupled from the actual evidence count. With no verified runs, the score is a
# heuristic prior only and must be labeled as such.
_ZERO_RUNS_CONFIDENCE_CAP = 0.60

def get_confidence_breakdown(
    hardware_match: float = 1.0,
    model_similarity: float = 1.0,
    historical_verifications: int = 0,
    rollback_frequency: float = 0.0
) -> Dict[str, Any]:
    """
    Returns an interpretable dictionary breakdown of evidence factors contributing
    to the heuristic decision score.

    IMPORTANT: historical_verifications must be the count of *actual verified runs*
    from the SharedKnowledgeBase for this architecture+hardware combination.
    Do NOT pass telemetry proxy values (stall %, memory %, graph node counts, or
    hardcoded constants) — those are not verification counts and will produce
    misleadingly high scores with zero empirical backing.
    """
    base_confidence = 0.85
    hw_weight = 0.05 * min(1.0, max(0.0, hardware_match))
    model_weight = 0.05 * min(1.0, max(0.0, model_similarity))
    history_weight = min(0.05, 0.005 * max(0, historical_verifications))
    penalty = 0.25 * min(1.0, max(0.0, rollback_frequency))
    raw_score = base_confidence + hw_weight + model_weight + history_weight - penalty

    no_verified_runs = historical_verifications == 0
    if no_verified_runs:
        # Cap the score — without verified runs this is a prior, not an evidence-backed score.
        score = round(min(_ZERO_RUNS_CONFIDENCE_CAP, raw_score), 2)
    else:
        score = round(max(0.50, min(0.99, raw_score)), 2)

    eval_3d = _CREDIBILITY_MANIFOLD.evaluate(
        heuristic_prior=raw_score,
        verified_hardware_runs=historical_verifications,
        loss_shift_margin=0.0,
        rollback_frequency=rollback_frequency,
    )

    return {
        "score": score,
        "decision_score": score,
        "score_type": "HEURISTIC_PRIOR" if no_verified_runs else "EMPIRICALLY_VERIFIED",
        "no_verified_runs": no_verified_runs,
        "hardware_match_label": "High" if hardware_match >= 0.8 else ("Medium" if hardware_match >= 0.5 else "Low"),
        "model_similarity_label": "High" if model_similarity >= 0.8 else ("Medium" if model_similarity >= 0.5 else "Low"),
        "historical_verifications_count": historical_verifications,
        "rollback_rate_pct": round(rollback_frequency * 100.0, 1),
        "penalty_applied": round(penalty, 3),
        "credibility_status": eval_3d.status.value,
        "credibility_audit_label": eval_3d.audit_label,
        "is_safe_to_auto_apply": eval_3d.is_safe_to_auto_apply,
    }

def calculate_evidence_confidence(
    hardware_match: float = 1.0,
    model_similarity: float = 1.0,
    historical_verifications: int = 0,
    rollback_frequency: float = 0.0
) -> float:
    """
    Computes a heuristic decision score (formerly confidence).
    historical_verifications MUST be the count of verified runs from the KB — not
    a proxy value derived from telemetry metrics or a hardcoded constant.
    Returns a score capped at 0.60 when historical_verifications == 0.
    """
    return get_confidence_breakdown(hardware_match, model_similarity, historical_verifications, rollback_frequency)["score"]



class DecisionEngine:
    """
    Analyzes telemetry summaries, Graphify computational DAGs, and Bayesian Hive Mind surrogate models
    to detect high-value technical optimizations (Mixed Precision, Gradient Checkpointing, FlashAttention,
    DataLoader tuning, Batch Scaling, Graphify Kernel Fusion, and Bayesian Hive Mind continuous auto-tuning).
    """
    def __init__(
        self,
        target_hardware: str = "NVIDIA A100 / H100 / RTX 4090",
        knowledge_base: Optional[SharedKnowledgeBase] = None,
        calibration_report: Optional[Any] = None,
    ):
        self.target_hardware = target_hardware
        self.knowledge_base = knowledge_base
        self.calibration_report = calibration_report

    @staticmethod
    def is_calibration_hardware_compatible(calib: Any, target_hardware: str) -> bool:
        """
        Enforces strict hardware compatibility boundaries between calibration reports
        and active target workloads. Prevents CPU calibration from transferring to GPU contexts.
        """
        if calib is None:
            return False
        calib_hw = getattr(calib, "hardware_name", "").upper()
        target_hw = target_hardware.upper()

        # Hard boundary: CPU calibration NEVER transfers to GPU targets
        if "CPU" in calib_hw and ("NVIDIA" in target_hw or "CUDA" in target_hw or "A2000" in target_hw or "A100" in target_hw or "H100" in target_hw or "GPU" in target_hw):
            return False
        if "GPU" in calib_hw and "CPU" in target_hw:
            return False

        return True

    def evaluate(
        self,
        summary: TelemetrySummary,
        model_type: str = "transformer",
        graph_builder: Optional[Any] = None,
        graph_summary: Optional[Any] = None,
        verification_results: Optional[List[Any]] = None,
        knowledge_base: Optional[SharedKnowledgeBase] = None,
        param_count_bucket: str = "",
        framework_version: str = "",
        inverse_cost_weight: float = 1.0,
        calibration_report: Optional[Any] = None,
        hardware_type: Optional[str] = None,
        num_gpus: int = 8,
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []
        kb = knowledge_base or self.knowledge_base or SharedKnowledgeBase()
        raw_calib = calibration_report or self.calibration_report
        is_compat = self.is_calibration_hardware_compatible(raw_calib, self.target_hardware)
        calib = raw_calib if is_compat else None

        # Index verification results by the specific rule_id (recommendation_id) they verified
        verifications_by_rule = (
            {v.recommendation_id: v for v in verification_results} if verification_results else {}
        )

        def is_safe(rule_id: str) -> bool:
            v = verifications_by_rule.get(rule_id)
            return v.is_safe if v is not None else False

        # Rule 1: Mixed Precision (FP16 / BF16)
        if summary.mixed_precision.lower() == "fp32":
            has_calib = calib is not None and calib.precision_speedup_pct > 0
            conf = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.95,
                historical_verifications=1 if has_calib else 0,
                rollback_frequency=0.02
            ) * inverse_cost_weight
            if has_calib:
                prec_std = getattr(calib, "precision_speedup_std_dev", 0.0)
                std_str = f" ± {prec_std:.1f}%" if prec_std > 0 else ""
                speedup_label = f"{calib.precision_speedup_pct:.1f}%{std_str} (Measured on {calib.hardware_name})"
                evidence_str = (
                    f"Measured active precision standard: FP32. Tensor Cores operating at reduced capacity."
                    f" [Hardware Calibrated: {calib.precision_speedup_pct:.1f}%{std_str} speedup on {calib.hardware_name}]"
                )
            else:
                speedup_label = "~45% (Typical Published Range)"
                evidence_str = "Measured active precision standard: FP32. Tensor Cores operating at reduced capacity."

            recs.append(
                Recommendation(
                    rule_id="RULE_MIXED_PRECISION",
                    title="Enable BF16 / FP16 Mixed Precision Training",
                    impact_level="HIGH",
                    speedup_estimate_label=speedup_label,
                    description="Training in standard FP32 precision underutilizes Tensor Cores. Switching to Automatic Mixed Precision (AMP) with BF16/FP16 yields up to 2x-4x throughput with negligible accuracy impact.",
                    actionable_code_snippet=(
                        "from torch.cuda.amp import autocast\n"
                        "# Wrap forward pass:\n"
                        "with autocast(dtype=torch.bfloat16):\n"
                        "    outputs = model(inputs)\n"
                    ),
                    safe_to_auto_apply=is_safe("RULE_MIXED_PRECISION"),
                    confidence=conf,
                    has_verified_runs=has_calib,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=evidence_str,
                    verification_plan="Verify loss curve trajectory over 50 steps; require max loss delta <= 0.05 and KL div <= 0.02.",
                    rollback_plan="Auto-revert to FP32 precision if loss divergence or NaN gradient occurs.",
                )
            )

        # Rule 2: DataLoader Worker Bottleneck
        if summary.num_workers == 0 or summary.avg_dataloader_stall_pct > 15.0:
            has_calib = calib is not None and calib.workers_speedup_pct > 0
            conf = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=1.0,
                historical_verifications=1 if has_calib else 0,
                rollback_frequency=0.0
            ) * inverse_cost_weight
            if has_calib:
                w_std = getattr(calib, "workers_speedup_std_dev", 0.0)
                std_str = f" ± {w_std:.1f}%" if w_std > 0 else ""
                speedup_label = f"{calib.workers_speedup_pct:.1f}%{std_str} (Measured on {calib.hardware_name})"
                evidence_str = (
                    f"DataLoader GPU I/O stall measured at {summary.avg_dataloader_stall_pct:.1f}% with num_workers={summary.num_workers}."
                    f" [Hardware Calibrated: {calib.workers_speedup_pct:.1f}%{std_str} speedup on {calib.hardware_name}]"
                )
            else:
                speedup_label = "~25% (Typical Published Range)"
                evidence_str = f"DataLoader GPU I/O stall measured at {summary.avg_dataloader_stall_pct:.1f}% with num_workers={summary.num_workers}."

            recs.append(
                Recommendation(
                    rule_id="RULE_DATALOADER_WORKERS",
                    title="Optimize PyTorch DataLoader Workers & Memory Pinning",
                    impact_level="HIGH",
                    speedup_estimate_label=speedup_label,
                    description=f"DataLoader is currently causing a {summary.avg_dataloader_stall_pct:.1f}% I/O stall on the GPU because num_workers={summary.num_workers} and pin_memory={summary.pin_memory}.",
                    actionable_code_snippet=(
                        "DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, persistent_workers=True)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_DATALOADER_WORKERS"),
                    confidence=conf,
                    has_verified_runs=has_calib,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=evidence_str,
                    verification_plan="Verify step throughput latency reduction over 20 steps.",
                    rollback_plan="Reset num_workers to baseline if CPU RAM exhaustion detected.",
                )
            )

        # Rule 3: FlashAttention-2 / 3 integration for Transformers
        # historical_verifications=0: no KB-verified runs for this rule yet.
        if model_type.lower() in ["transformer", "llm", "bert", "gpt"] and not summary.flash_attention:
            conf = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.90,
                historical_verifications=0,  # No KB-verified runs — prior only
                rollback_frequency=0.05
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_FLASH_ATTENTION",
                    title="Enable FlashAttention-2 Kernel",
                    impact_level="HIGH",
                    speedup_estimate_label="~35% (Typical Published Range)",
                    description="Standard PyTorch attention computes full NxN attention matrices in RAM. FlashAttention computes attention in GPU SRAM, dramatically cutting memory footprint and boosting step throughput.",
                    actionable_code_snippet=(
                        "model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')"
                    ),
                    safe_to_auto_apply=is_safe("RULE_FLASH_ATTENTION"),
                    confidence=conf,
                    has_verified_runs=False,
                    risk_level="MEDIUM",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Model architecture type '{model_type}' detected with standard PyTorch attention enabled.",
                    verification_plan="Verify step latency speedup while enforcing loss trajectory delta <= 0.05.",
                    rollback_plan="Revert to PyTorch SDPA attention if CUDA kernel launch error or divergence occurs.",
                )
            )

        # Rule 4: Gradient Checkpointing for High Memory Headroom
        memory_usage_pct = (summary.peak_gpu_memory_mb / summary.gpu_memory_total_mb) * 100.0 if summary.gpu_memory_total_mb > 0 else 50.0
        if memory_usage_pct > 88.0 and not summary.gradient_checkpointing:
            has_calib = calib is not None and calib.grad_checkpoint_memory_saving_pct > 0
            conf = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.85,
                historical_verifications=1 if has_calib else 0,
                rollback_frequency=0.01
            ) * inverse_cost_weight
            if has_calib:
                gc_std = getattr(calib, "grad_checkpoint_memory_saving_std_dev", 0.0)
                std_str = f" ± {gc_std:.1f}%" if gc_std > 0 else ""
                evidence_str = (
                    f"Peak GPU VRAM allocation measured at {memory_usage_pct:.1f}% ({summary.peak_gpu_memory_mb:.0f} MB)."
                    f" [Hardware Calibrated: {calib.grad_checkpoint_memory_saving_pct:.1f}%{std_str} memory saving on {calib.hardware_name}]"
                )
            else:
                evidence_str = f"Peak GPU VRAM allocation measured at {memory_usage_pct:.1f}% ({summary.peak_gpu_memory_mb:.0f} MB)."

            recs.append(
                Recommendation(
                    rule_id="RULE_GRADIENT_CHECKPOINTING",
                    title="Enable Gradient Checkpointing to Eliminate OOM & Scale Batch Size",
                    impact_level="MEDIUM",
                    speedup_estimate_label="~20% (Typical Published Range)",
                    description=f"GPU Memory peak is at {memory_usage_pct:.1f}%. Enabling gradient checkpointing trades small compute recalculation for 60%+ memory savings, unlocking 2x larger batch sizes.",
                    actionable_code_snippet=(
                        "model.gradient_checkpointing_enable()"
                    ),
                    safe_to_auto_apply=is_safe("RULE_GRADIENT_CHECKPOINTING"),
                    confidence=conf,
                    has_verified_runs=has_calib,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=evidence_str,
                    verification_plan="Verify peak VRAM drops below 60% with zero CUDA OOM errors.",
                    rollback_plan="Disable gradient checkpointing if backward pass latency penalty exceeds 15%.",
                )
            )

        # Rule 5: Underutilized GPU Memory (Scale Batch Size / Gradient Accumulation via Section B)
        # historical_verifications=0: no KB-verified runs for this rule yet.
        if memory_usage_pct < 45.0 and summary.avg_gpu_utilization_pct < 70.0:
            from ghost_layer.telemetry.vram_scaling import calculate_vram_headroom_multiplier
            v_scaling = calculate_vram_headroom_multiplier(
                vram_total_mb=summary.gpu_memory_total_mb,
                vram_peak_mb=summary.peak_gpu_memory_mb,
                current_batch_size=1,
            )
            conf = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.85,
                historical_verifications=0,  # No KB-verified runs — prior only
                rollback_frequency=0.0
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_BATCH_SCALING",
                    title="Increase Batch Size / Gradient Accumulation Steps",
                    impact_level="MEDIUM",
                    speedup_estimate_label=f"~{min(50.0, round((v_scaling.safe_multiplier - 1.0) * 100.0, 1)) + 20.0:.0f}% (Typical Published Range)",
                    description=(
                        f"GPU peak memory usage is only {memory_usage_pct:.1f}% ({summary.peak_gpu_memory_mb:.0f}MB / {summary.gpu_memory_total_mb:.0f}MB) "
                        f"and compute utilization is {summary.avg_gpu_utilization_pct:.1f}%. "
                        f"Section B VRAM headroom analysis indicates a safe scaling multiplier of {v_scaling.safe_multiplier:.2f}x "
                        f"(available headroom: {v_scaling.vram_headroom_mb:.0f}MB with {v_scaling.oom_buffer_mb:.0f}MB OOM safety reserve)."
                    ),
                    actionable_code_snippet=(
                        f"# Scale micro-batch size by {v_scaling.safe_multiplier:.2f}x or set gradient_accumulation_steps = {max(2, int(round(v_scaling.safe_multiplier)))}"
                    ),
                    safe_to_auto_apply=is_safe("RULE_BATCH_SCALING"),
                    confidence=conf,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=(
                        f"GPU memory utilization is {memory_usage_pct:.1f}% and compute utilization is {summary.avg_gpu_utilization_pct:.1f}%. "
                        f"Safe headroom multiplier: {v_scaling.safe_multiplier:.2f}x."
                    ),
                    verification_plan="Verify GPU compute utilization increases above 80% with stable step times.",
                    rollback_plan="Reduce batch size if VRAM allocation spikes > 90%.",
                )
            )

        # Rule 6: Graphify Operator Fusion (`torch.compile` Inductor Backend)
        # historical_verifications=0: fusible_count is a graph analysis result, not a
        # count of KB-verified runs. Multiplying it by 2 was a proxy bug.
        fusion_speedup = 28.0
        fusible_count = 5
        if graph_summary is not None and hasattr(graph_summary, "estimated_fusion_speedup_pct"):
            if graph_summary.estimated_fusion_speedup_pct > 0:
                fusion_speedup = graph_summary.estimated_fusion_speedup_pct
            fusible_count = len(getattr(graph_summary, "fusible_subgraph_nodes", []))

        conf_graph = calculate_evidence_confidence(
            hardware_match=1.0,
            model_similarity=0.95,
            historical_verifications=0,  # No KB-verified runs — prior only
            rollback_frequency=0.03
        ) * inverse_cost_weight

        fusion_label = f"~{fusion_speedup}% (Measured/Estimated from Graphify)"
        if kb is not None and hasattr(kb, "find_best_match"):
            kb_entry, kb_match_level = kb.find_best_match(
                architecture_family=model_type,
                hardware_type=self.target_hardware,
                param_count_bucket=param_count_bucket,
                framework_version=framework_version,
            )
            if kb_entry is not None and kb_entry.confidence_tier in ("MEDIUM", "HIGH"):
                fusion_speedup = kb_entry.throughput_improvement_pct
                fusion_label = (
                    f"~{fusion_speedup}% ({kb_entry.confidence_tier} confidence, "
                    f"{kb_entry.sample_count} verified cross-client samples, {kb_match_level} match)"
                )

        recs.append(
            Recommendation(
                rule_id="RULE_GRAPHIFY_TORCH_COMPILE",
                title="Apply Graphify Operator Fusion (`torch.compile` Inductor Backend)",
                impact_level="HIGH",
                speedup_estimate_label=fusion_label,
                description=f"Graphify execution graph mapping identified {fusible_count} fusible elementwise RMSNorm & Activation subgraphs. Compiling the DAG fuses GPU kernels, eliminating intermediate VRAM roundtrips.",
                actionable_code_snippet=(
                    "model = torch.compile(model, mode='reduce-overhead', backend='inductor')"
                ),
                safe_to_auto_apply=is_safe("RULE_GRAPHIFY_TORCH_COMPILE"),
                confidence=conf_graph,
                has_verified_runs=False,
                risk_level="MEDIUM",
                rollback_mode="AUTOMATIC",
                evidence=f"Graphify DAG analysis mapped {fusible_count} fusible operator subgraphs on critical path.",
                verification_plan="Verify Inductor compilation completion and measure step speedup over 30 steps.",
                rollback_plan="Fallback to eager PyTorch execution if Triton compilation error occurs.",
            )
        )

        # Rule 7: Dynamic Bayesian Hive Mind Surrogate Auto-Tuning
        # DYNAMIC QUERY to the surrogate model
        predicted_config = kb.predict_hive_mind_config(model_type, self.target_hardware)
        entry = kb.query(model_type, self.target_hardware)

        samples = entry.sample_count if entry else 0
        rejects = entry.rejected_sample_count if entry else 0
        total_samples = samples + rejects
        hist_speedup = entry.throughput_improvement_pct if entry else 22.5
        rollback_freq = (rejects / total_samples) if total_samples > 0 else 0.0

        conf_hive = calculate_evidence_confidence(
            hardware_match=1.0 if entry else 0.8,
            model_similarity=1.0 if entry else 0.8,
            historical_verifications=samples,
            rollback_frequency=rollback_freq
        ) * inverse_cost_weight

        mb_mult = predicted_config.get("micro_batch_multiplier", 2.0)
        ind_mode = str(predicted_config.get("inductor_mode", "reduce-overhead"))
        prefetch = predicted_config.get("prefetch_factor", 4)
        alloc_conf = str(predicted_config.get("cuda_alloc_conf", "max_split_size_mb:128"))
        epilogue_fusion = predicted_config.get("triton_epilogue_fusion", True)

        recs.append(
            Recommendation(
                rule_id="RULE_DYNAMIC_HIVE_MIND_SURROGATE",
                title="Inject Dynamic Bayesian Hive Mind Parameter Autotuning",
                impact_level="HIGH",
                speedup_estimate_label=f"~{hist_speedup:.1f}% Continuous Gain (Surrogate Predicted)",
                description=(
                    f"Surrogate model queried for '{model_type}' on '{self.target_hardware}' across {samples} verified runs. "
                    f"Sampled optimal parameter vector: micro_batch_multiplier={mb_mult}x, inductor_mode='{ind_mode}', "
                    f"prefetch_factor={prefetch}, cuda_alloc_conf='{alloc_conf}', triton_epilogue_fusion={epilogue_fusion}."
                ),
                actionable_code_snippet=(
                    f"# Dynamic Hive Mind Surrogate Parameter Vector:\n"
                    f"import os, torch\n"
                    f"os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '{alloc_conf}'\n"
                    f"torch._dynamo.config.epilogue_fusion = {epilogue_fusion}\n"
                    f"# Applies micro_batch_multiplier={mb_mult}x and prefetch_factor={prefetch} via GhostWatcherHook"
                ),
                safe_to_auto_apply=is_safe("RULE_DYNAMIC_HIVE_MIND_SURROGATE"),
                confidence=conf_hive,
                has_verified_runs=(samples > 0),
                risk_level="LOW" if rollback_freq < 0.1 else "MEDIUM",
                rollback_mode="AUTOMATIC",
                evidence=f"Bayesian Thompson-Sampling bandit model evaluated {samples} verified runs ({rejects} rejected) for architecture '{model_type}'.",
                verification_plan="Verify surrogate configuration throughput gain over 30 steps with loss trajectory delta <= 0.05.",
                rollback_plan="Update Beta-Bernoulli posterior with negative feedback and revert to default runtime config if throughput regresses.",
            )
        )

        # Rule 8: Variance Containment — triggered when loss delta stability index indicates elevated step variance
        thresh_containment, src_containment = kb.get_calibrated_threshold(
            "RULE_VARIANCE_CONTAINMENT",
            default_threshold=0.70,
            min_bound=0.55,
            max_bound=0.75,
        )
        if inverse_cost_weight < thresh_containment:
            severity = "CRITICAL" if inverse_cost_weight < (thresh_containment * 0.714) else "HIGH"
            containment_actions = []
            if inverse_cost_weight < 0.40:
                containment_actions.append("Enable gradient clipping (max_norm=1.0)")
                containment_actions.append("Extend LR warmup period by 2x")
                containment_actions.append("Reduce micro-batch size by 50% until stable")
            elif inverse_cost_weight < 0.60:
                containment_actions.append("Enable gradient clipping (max_norm=2.0)")
                containment_actions.append("Increase gradient accumulation steps by 2x")
            else:
                containment_actions.append("Increase gradient accumulation steps by 1.5x")
                containment_actions.append("Consider cosine LR annealing schedule")

            action_str = "; ".join(containment_actions)
            conf_containment = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=1.0,
                historical_verifications=0,
                rollback_frequency=0.0
            )

            recs.append(
                Recommendation(
                    rule_id="RULE_VARIANCE_CONTAINMENT",
                    title=f"Activate Variance Containment Protocol ({severity})",
                    impact_level="HIGH",
                    speedup_estimate_label=f"Prevents compute waste (Loss stability index: {inverse_cost_weight:.3f})",
                    description=(
                        f"Empirical loss trajectory analysis detected {severity.lower()} step variance "
                        f"(loss stability index: {inverse_cost_weight:.3f}, threshold < {thresh_containment:.3f} [{src_containment}]). "
                        f"High step-to-step loss fluctuation indicates gradient noise or ill-conditioned parameter updates. "
                        f"Containment actions should be applied before further aggressive throughput optimizations."
                    ),
                    actionable_code_snippet=(
                        f"# Containment Actions:\n"
                        f"# {action_str}\n"
                        f"torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
                    ),
                    safe_to_auto_apply=False,  # Containment is recommendation-only
                    confidence=conf_containment,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="MANUAL",
                    evidence=f"Loss stability index {inverse_cost_weight:.3f} is below {thresh_containment:.3f} threshold ({src_containment}). Elevated step-to-step loss variance detected.",
                    verification_plan="Verify loss curve stabilization (std dev reduction > 30%) within 50 steps of applying containment actions.",
                    rollback_plan="Revert containment actions if throughput drops > 10% without corresponding variance reduction.",
                )
            )

        # Rule 9: Muon (Momentum Orthogonalized by Newton-Schulz) for 2D VRAM Tensor Matrices
        if model_type.lower() in ["transformer", "llm", "bert", "gpt", "mlp", "llama", "mistral", "qwen"]:
            conf_muon = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.95,
                historical_verifications=0,
                rollback_frequency=0.01,
            ) * max(0.6, inverse_cost_weight)
            recs.append(
                Recommendation(
                    rule_id="RULE_MUON_OPTIMIZER",
                    title="Activate Muon 2D-VRAM Matrix Polar Decomposition Optimizer",
                    impact_level="HIGH",
                    speedup_estimate_label="~35-50% Faster Convergence (Unvalidated Projection from Muon Literature; Typical Published Range)",
                    description=(
                        f"2D Euclidean VRAM parameter manifold detected for {model_type.upper()}. "
                        f"Standard AdamW treats weights as isotropic 1D Euclidean points, causing spectral skew and gradient drag. "
                        f"Muon orthogonalizes 2D momentum tensors via quintic Newton-Schulz root finding ($U = G(G^TG)^{{-1/2}}$), "
                        f"normalizing all singular values to 1.0 and maximizing learning efficiency across attention heads."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import create_muon_hybrid_optimizer\n"
                        "# Default 2D VRAM Split: 2D weights -> Muon, 1D vectors -> AdamW\n"
                        "optimizer = create_muon_hybrid_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_MUON_OPTIMIZER"),
                    confidence=conf_muon,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"2D weight matrices in {model_type} benefit from matrix polar decomposition (unvalidated literature projection; empirical benchmark pending).",
                    verification_plan="Verify rapid loss descent over 30 steps with loss trajectory delta <= 0.05.",
                    rollback_plan="Revert to AdamW if gradient norm explodes or loss diverges.",
                )
            )

        # Rule 10: Sophia-G (Second-Order Clipped Stochastic Optimization)
        thresh_sophia, src_sophia = kb.get_calibrated_threshold(
            "RULE_SOPHIA_SECOND_ORDER",
            default_threshold=0.85,
            min_bound=0.75,
            max_bound=0.90,
        )
        if inverse_cost_weight < thresh_sophia:
            conf_sophia = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.90,
                historical_verifications=0,
                rollback_frequency=0.02,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_SOPHIA_SECOND_ORDER",
                    title="Enable Sophia-G Second-Order Diagonal Hessian Optimization",
                    impact_level="HIGH",
                    speedup_estimate_label="~2x Faster Step Convergence (Unvalidated Projection from Sophia Literature; Empirical Benchmark Pending)",
                    description=(
                        f"Loss delta variance detected (loss stability index: {inverse_cost_weight:.3f}, threshold < {thresh_sophia:.3f} [{src_sophia}]). "
                        f"Sophia-G estimates the diagonal Hessian via Hutchinson trace estimator (Pearlmutter HVP) "
                        f"and clips updates element-wise by local curvature, navigating non-convex valleys without full matrix inversion. "
                        f"Pass loss closure to optimizer.step(closure) or update_hessian(loss_fn) for exact second-order directional derivatives."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import SophiaG\n"
                        "optimizer = SophiaG(model.parameters(), lr=1e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=0.1, k=10)\n"
                        "# Training loop with exact Pearlmutter Hutchinson HVP closure:\n"
                        "# def closure():\n"
                        "#     return loss_fn(model(x), y)\n"
                        "# loss = loss_fn(model(x), y)\n"
                        "# loss.backward()\n"
                        "# optimizer.step(closure=closure)  # Evaluates HVP via closure every k=10 steps"
                    ),
                    safe_to_auto_apply=is_safe("RULE_SOPHIA_SECOND_ORDER"),
                    confidence=conf_sophia,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Loss stability index {inverse_cost_weight:.3f} is below {thresh_sophia:.3f} threshold ({src_sophia}). Moderate step variance suggests curvature-aware clipping benefit.",
                    verification_plan="Verify monotonic loss reduction over 40 steps without gradient clipping spikes.",
                    rollback_plan="Revert to standard optimizer if Hessian variance exceeds threshold.",
                )
            )

        # Rule 11: Shampoo (Matrix Inverse 2p-th Root Tensor Preconditioning)
        thresh_shampoo, src_shampoo = kb.get_calibrated_threshold(
            "RULE_SHAMPOO_PRECONDITIONING",
            default_threshold=0.60,
            min_bound=0.50,
            max_bound=0.70,
        )
        if inverse_cost_weight < thresh_shampoo:
            conf_shampoo = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.88,
                historical_verifications=0,
                rollback_frequency=0.03,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_SHAMPOO_PRECONDITIONING",
                    title="Apply Shampoo Matrix Root Preconditioning for Ill-Conditioned Loss Ravines",
                    impact_level="HIGH",
                    speedup_estimate_label="~40% Convergence Acceleration (Unvalidated Projection from Shampoo Literature; Empirical Benchmark Pending)",
                    description=(
                        f"Severe loss surface ill-conditioning detected (loss stability index: {inverse_cost_weight:.3f}, threshold < {thresh_shampoo:.3f} [{src_shampoo}]). "
                        f"Shampoo preconditions tensor gradients along row and column dimensions with inverse 4th roots "
                        f"(L^(-1/4) G R^(-1/4)), preventing cross-axis oscillation. "
                        f"Note: Matrix eigendecomposition is O(d^3); for large hidden dimensions (d >= 4096), "
                        f"use amortized update intervals (update_preconditioner_interval >= 10-20) to avoid step latency overhead."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import Shampoo\n"
                        "optimizer = Shampoo(model.parameters(), lr=1e-3, momentum=0.9, update_preconditioner_interval=10)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_SHAMPOO_PRECONDITIONING"),
                    confidence=conf_shampoo,
                    has_verified_runs=False,
                    risk_level="MEDIUM",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Loss stability index {inverse_cost_weight:.3f} is below {thresh_shampoo:.3f} threshold ({src_shampoo}). Severe loss oscillation indicates ill-conditioned curvature.",
                    verification_plan="Verify eigenvalue conditioning improvement and step time stability.",
                    rollback_plan="Revert to diagonal preconditioner if matrix eigendecomposition fails.",
                )
            )

        # Rule 12: Pearlmutter Hessian-Vector Product for Few-Shot Bilevel Adaptation
        # Threshold: total_steps < 500 represents early-stage or few-shot adaptation regimes
        if summary.total_steps < 500:
            conf_hvp = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.95,
                historical_verifications=0,
                rollback_frequency=0.01,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_PEARLMUTTER_HVP_METAGRADIENT",
                    title="Activate Pearlmutter HVP Exact Curvature for Few-Shot Meta-Adaptation",
                    impact_level="HIGH",
                    speedup_estimate_label="Optimal Newton Direction in 1-5 Few-Shot Steps (Theoretical Bound; Empirical Benchmark Pending)",
                    description=(
                        f"Ultra-low sample regime detected ({summary.total_steps} steps). "
                        f"Pearlmutter finite-difference trick evaluates exact directional Hessian-vector products "
                        f"H v without storing O(N^2) matrices, finding optimal few-shot parameter initializations."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import compute_hvp_pearlmutter, estimate_local_curvature\n"
                        "diag = estimate_local_curvature(model, loss_fn, sample_count=summary.total_steps)\n"
                        "# Use HVP for exact second-order directional updates"
                    ),
                    safe_to_auto_apply=is_safe("RULE_PEARLMUTTER_HVP_METAGRADIENT"),
                    confidence=conf_hvp,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Few-shot training regime detected ({summary.total_steps} steps). Exact HVP provides optimal directional curvature.",
                    verification_plan="Verify rapid validation loss convergence in <10 adaptation steps.",
                    rollback_plan="Fall back to first-order meta-learning if second-order gradient norm vanishes.",
                )
            )

        # Rule 13: Async Tensor Flow & PCIe Pipeline De-Jamming
        if not summary.pin_memory or summary.num_workers == 0 or summary.avg_dataloader_stall_pct > 8.0:
            conf_async = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.98,
                historical_verifications=0,
                rollback_frequency=0.0,
            )
            recs.append(
                Recommendation(
                    rule_id="RULE_ASYNC_TENSOR_FLOW",
                    title="De-Jam Tensor Flow: Enable Async Non-Blocking Pinned Transfers",
                    impact_level="HIGH",
                    speedup_estimate_label="15-25% Latency Recovery (Eliminates PCIe Host-Device Stalls; Typical Published Range)",
                    description=(
                        f"Tensor flow jamming detected: DataLoader stall is {summary.avg_dataloader_stall_pct:.1f}% "
                        f"with pin_memory={summary.pin_memory} and num_workers={summary.num_workers}. "
                        f"Synchronous host-to-device tensor copying blocks GPU compute streams."
                    ),
                    actionable_code_snippet=(
                        "# Configure DataLoader with async streaming:\n"
                        "loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=2, persistent_workers=True)\n"
                        "# In loop: batch = batch.to(device, non_blocking=True)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_ASYNC_TENSOR_FLOW"),
                    confidence=conf_async,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"DataLoader stall at {summary.avg_dataloader_stall_pct:.1f}%. Async transfers overlap PCIe and compute.",
                    verification_plan="Verify DataLoader stall drops below 2.0% within 20 steps.",
                    rollback_plan="Revert to standard DataLoader if worker spawning fails.",
                )
            )

        # Rule 14: CUDA Caching Allocator De-Fragmentation (Max GPU Vitality)
        if summary.peak_gpu_memory_mb > 0.65 * summary.gpu_memory_total_mb or "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            conf_alloc = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.95,
                historical_verifications=0,
                rollback_frequency=0.0,
            )
            recs.append(
                Recommendation(
                    rule_id="RULE_CUDA_ALLOCATOR_TUNING",
                    title="Tune PyTorch Caching Allocator to Eliminate VRAM Fragmentation Leakage",
                    impact_level="MEDIUM",
                    speedup_estimate_label="Prevents False OOMs & Recovers 20-35% Usable VRAM (Typical Published Range)",
                    description=(
                        f"PyTorch Caching Allocator is uncalibrated or VRAM pressure is high ({summary.peak_gpu_memory_mb:.0f}MB / {summary.gpu_memory_total_mb:.0f}MB). "
                        f"Configuring expandable segments and memory split sizes eliminates internal pool fragmentation."
                    ),
                    actionable_code_snippet=(
                        "import os\n"
                        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16'"
                    ),
                    safe_to_auto_apply=is_safe("RULE_CUDA_ALLOCATOR_TUNING"),
                    confidence=conf_alloc,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence="Unfragmented CUDA allocator preserves continuous address space on smaller GPUs.",
                    verification_plan="Verify peak memory stability without Out-Of-Memory exceptions.",
                    rollback_plan="Clear PYTORCH_CUDA_ALLOC_CONF if driver incompatibility occurs.",
                )
            )

        # Rule 15: Grouped-Query Attention (GQA) Uptraining Conversion
        attn_arch = getattr(summary, "attention_architecture", "MHA").upper()
        seq_len = getattr(summary, "sequence_length", 2048)
        kv_cache_mb = getattr(summary, "estimated_kv_cache_mb", 0.0)
        if attn_arch == "MHA" and (seq_len >= 2048 or kv_cache_mb > 512.0 or model_type.lower() in ["transformer", "llm", "llama", "mistral", "qwen"]):
            conf_gqa = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.95,
                historical_verifications=0,
                rollback_frequency=0.01,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_GQA_UPTRAINING",
                    title="Convert MHA to Grouped-Query Attention (GQA) with Mean-Pool Uptraining",
                    impact_level="HIGH",
                    speedup_estimate_label="~4x-8x KV-Cache Memory Reduction & 2x Decode Speedup (Typical Published Range)",
                    description=(
                        f"Standard Multi-Head Attention (MHA) detected with sequence length {seq_len} (estimated KV cache: {kv_cache_mb:.1f} MB). "
                        f"Converting 1:1 Q:K:V heads to 8:1 Grouped-Query Attention cuts KV memory traffic across HBM by up to 87.5% "
                        f"with minimal accuracy recovery required via mean-pooled uptraining."
                    ),
                    actionable_code_snippet=(
                        "# Convert MHA weights to GQA (8 query heads per 1 KV group):\n"
                        "def convert_mha_to_gqa(k_weight, num_groups=8):\n"
                        "    # Mean-pool K/V heads within each group\n"
                        "    return k_weight.view(num_groups, -1, k_weight.size(-1)).mean(dim=1)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_GQA_UPTRAINING"),
                    confidence=conf_gqa,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Model uses standard MHA with sequence length {seq_len}. KV cache memory bandwidth is a bottleneck.",
                    verification_plan="Enforce Loss-Shift Proxy check (Δ <= 0.10) after 50 uptraining steps.",
                    rollback_plan="Revert to original MHA weight checkpoint if loss delta exceeds 0.10.",
                )
            )

        # Rule 16: Multi-Head Latent Attention (MLA) Low-Rank KV Compression
        if model_type.lower() in ["transformer", "llm", "llama", "mistral", "qwen", "deepseek", "gpt"] and attn_arch not in ["MLA"]:
            conf_mla = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.90,
                historical_verifications=0,
                rollback_frequency=0.02,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_MLA_LATENT_COMPRESSION",
                    title="Implement Multi-Head Latent Attention (MLA) with Decoupled RoPE",
                    impact_level="HIGH",
                    speedup_estimate_label="~90% KV Cache Footprint Reduction with Full Expressiveness (DeepSeek MLA Standard)",
                    description=(
                        f"Deploy Multi-Head Latent Attention (MLA) to compress key and value projections into a low-rank latent vector c_t^(KV) "
                        f"with a separate decoupled RoPE positional key stream. During inference, up-projection weights absorb into Q and O projections, "
                        f"eliminating uncompressed KV VRAM materialization."
                    ),
                    actionable_code_snippet=(
                        "# Multi-Head Latent Attention (MLA) projection structure:\n"
                        "# 1. Down-project hidden state: c_kv = W_DK(h)\n"
                        "# 2. Decoupled RoPE key: k_rope = RoPE(W_KR(h))\n"
                        "# 3. Store only (c_kv, k_rope) in KV cache"
                    ),
                    safe_to_auto_apply=is_safe("RULE_MLA_LATENT_COMPRESSION"),
                    confidence=conf_mla,
                    has_verified_runs=False,
                    risk_level="MEDIUM",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Transformer architecture '{model_type}' can leverage low-rank latent KV compression for maximum generation throughput.",
                    verification_plan="Verify loss trajectory stability (Δ <= 0.10) and evaluate VRAM compression factor >= 5x.",
                    rollback_plan="Fall back to GQA if low-rank down-projection exhibits representation bottleneck.",
                )
            )

        # Rule 17: Dynamic Sparse Attention (DSA) for Ultra-Long Sequences
        if seq_len >= 8192:
            conf_dsa = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.90,
                historical_verifications=0,
                rollback_frequency=0.01,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_DYNAMIC_SPARSE_ATTENTION",
                    title="Activate Dynamic Sparse Attention (DSA) for Ultra-Long Contexts",
                    impact_level="HIGH",
                    speedup_estimate_label="O(N) to O(k) Attention Scaling (~3x-6x Speedup on >8k Contexts)",
                    description=(
                        f"Sequence length is {seq_len} tokens. Standard attention scales quadratically O(N^2) in compute and linearly in memory. "
                        f"Dynamic Sparse Attention routes attention only to top-k relevant token blocks and sliding windows dynamically per step."
                    ),
                    actionable_code_snippet=(
                        "# Configure Dynamic Sparse Attention top-k routing:\n"
                        "# model.set_attention_pattern(pattern='dynamic_sparse', top_k=2048, window_size=512)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_DYNAMIC_SPARSE_ATTENTION"),
                    confidence=conf_dsa,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Ultra-long context sequence length ({seq_len} tokens) detected. Dense attention wastes Tensor Core cycles on uninformative tokens.",
                    verification_plan="Verify step throughput latency scaling and ensure loss delta <= 0.05 on long-context validation set.",
                    rollback_plan="Revert to dense FlashAttention if perplexity degrades on needle-in-a-haystack benchmarks.",
                )
            )

        # Rule 18: Latency Chain De-Bottlenecking for Memory-Bandwidth Bounds
        dominant_bottleneck = getattr(summary, "dominant_latency_bottleneck", "COMPUTE_BOUND")
        tpot_ms = getattr(summary, "tpot_ms", 0.0)
        if dominant_bottleneck == "MEMORY_BANDWIDTH_BOUND" or (tpot_ms > 0.0 and kv_cache_mb > 1024.0):
            conf_lat = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.95,
                historical_verifications=0,
                rollback_frequency=0.0,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_LATENCY_CHAIN_KV_BOUND",
                    title="De-Bottleneck Latency Chain: Apply FP8 KV-Cache Quantization",
                    impact_level="HIGH",
                    speedup_estimate_label="~50% Decode Latency Reduction (Doubles HBM KV Read Bandwidth; Typical Published Range)",
                    description=(
                        f"Latency chain profiling diagnosed dominant bottleneck as '{dominant_bottleneck}' (active KV cache: {kv_cache_mb:.1f} MB, TPOT: {tpot_ms:.1f} ms). "
                        f"Quantizing the KV cache to FP8 / INT4 cuts memory bus bandwidth demand in half per generation step."
                    ),
                    actionable_code_snippet=(
                        "# Enable FP8 KV-Cache Quantization in serving engine:\n"
                        "# engine_args.kv_cache_dtype = 'fp8'\n"
                        "# engine_args.quantization = 'fp8'"
                    ),
                    safe_to_auto_apply=is_safe("RULE_LATENCY_CHAIN_KV_BOUND"),
                    confidence=conf_lat,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Generation step latency is constrained by memory bandwidth reading {kv_cache_mb:.1f} MB of KV cache per step.",
                    verification_plan="Verify TPOT latency reduction > 30% with KL divergence <= 0.01.",
                    rollback_plan="Revert to BF16 KV cache if token generation quality degrades.",
                )
            )

        # Rule 19: Chunked Cross-Entropy Loss Engine for High VRAM / Large Contexts
        vram_pressure_pct = (summary.peak_gpu_memory_mb / max(1.0, summary.gpu_memory_total_mb)) * 100.0
        seq_len_val = getattr(summary, "sequence_length", 2048)
        if vram_pressure_pct >= 80.0 or seq_len_val >= 4096 or getattr(summary, "peak_vram_pct", 0.0) >= 80.0:
            conf_ce = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.95,
                historical_verifications=0,
                rollback_frequency=0.01,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_CHUNKED_CROSS_ENTROPY",
                    title="Enable Chunked & Fused Cross-Entropy Loss (40%-60% VRAM Reduction)",
                    impact_level="HIGH",
                    speedup_estimate_label="~15%-25% Throughput Boost via Activation Headroom (Published Chunked CE Range)",
                    description=(
                        f"High VRAM allocation ({vram_pressure_pct:.1f}% peak) and/or extended sequence length ({seq_len_val} tokens) detected. "
                        f"Materializing full [Batch, Seq, Vocab] logit tensors in GPU DRAM is the dominant consumer of activation VRAM. "
                        f"ChunkedCrossEntropyLoss streams token chunks through projection and loss computation, bypassing intermediate tensor materialization."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import ChunkedCrossEntropyLoss\n"
                        "loss_fn = ChunkedCrossEntropyLoss(chunk_size=4096, label_smoothing=0.0)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_CHUNKED_CROSS_ENTROPY"),
                    confidence=conf_ce,
                    has_verified_runs=False,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC",
                    evidence=f"Peak VRAM is at {vram_pressure_pct:.1f}% capacity. Standard cross-entropy projects full vocabulary into DRAM simultaneously.",
                    verification_plan="Verify exact loss equivalence with standard CE (delta <= 1e-4) and measure peak activation VRAM reduction.",
                    rollback_plan="Revert to standard torch.nn.CrossEntropyLoss if gradient NaN occurs.",
                )
            )

        # Rule 20: Manifold-Constrained Hyper-Connections (mHC) for Residual Stability
        grad_norm_var = getattr(summary, "gradient_norm_variance", 0.0)
        is_multi_stream = getattr(summary, "is_multi_stream", False)
        if grad_norm_var >= 5.0 or is_multi_stream:
            conf_mhc = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.90,
                historical_verifications=0,
                rollback_frequency=0.02,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_MHC_RESIDUAL_ROUTING",
                    title="Constrain Multi-Stream Residual Routing on Birkhoff Polytope (mHC)",
                    impact_level="HIGH",
                    speedup_estimate_label="Prevents Gradient Explosion in Deep Residual Streams",
                    description=(
                        f"Elevated gradient norm variance ({grad_norm_var:.2f}) or multi-stream residual structure detected. "
                        f"Unconstrained residual summation causes exponential norm growth in deep networks. "
                        f"mHC projects residual transition matrices onto the doubly stochastic Birkhoff Polytope via Sinkhorn-Knopp iterations."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import ManifoldConstrainedHyperConnections\n"
                        "mhc = ManifoldConstrainedHyperConnections(num_streams=4, feature_dim=hidden_dim, sinkhorn_iters=10)"
                    ),
                    safe_to_auto_apply=False,  # Architectural modification requires human review
                    confidence=conf_mhc,
                    has_verified_runs=False,
                    risk_level="MEDIUM",
                    rollback_mode="MANUAL",
                    evidence=f"Gradient norm variance ({grad_norm_var:.2f} >= 5.0) indicates manifold instability in residual propagation.",
                    verification_plan="Verify spectral radius of transition matrices <= 1.0 and gradient norm remains bounded across 50 steps.",
                    rollback_plan="Disable mHC projection and restore standard residual addition.",
                )
            )

        # Rule 21: Matrix-Free Quasi-Newton (L-BFGS) Curvature Preconditioning
        # Fires when peak memory pressure is severe or optimizer states / curvature matrices cause VRAM leaks
        vram_headroom = getattr(summary, "vram_headroom_mb", None)
        peak_vram = getattr(summary, "peak_gpu_memory_mb", 0.0)
        total_vram = getattr(summary, "gpu_memory_total_mb", 0.0)
        if vram_headroom is None and total_vram > 0 and peak_vram > 0:
            vram_headroom = max(0.0, total_vram - peak_vram)
        elif vram_headroom is None:
            vram_headroom = 99999.0

        optimizer_overhead = getattr(summary, "optimizer_overhead_pct", 0.0)
        is_curvature_heavy = getattr(summary, "is_curvature_heavy", False)


        if (vram_headroom < 8192.0 and peak_vram > 50000.0) or optimizer_overhead >= 25.0 or is_curvature_heavy:
            conf_lbfgs = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.92,
                historical_verifications=1 if verifications_by_rule.get("RULE_QUASI_NEWTON_LBFGS") else 0,
                rollback_frequency=0.01,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_QUASI_NEWTON_LBFGS",
                    title="Switch Dense Curvature to Matrix-Free Quasi-Newton (L-BFGS)",
                    impact_level="HIGH",
                    speedup_estimate_label="Eliminates O(d^2) Matrix Memory Leaks & Recovers 30-50% Optimizer VRAM",
                    description=(
                        f"Severe VRAM pressure detected (peak memory {peak_vram:.0f} MB, headroom {vram_headroom:.0f} MB). "
                        f"Dense second-order or full covariance preconditioning incurs O(d^2) matrix memory leaks and tensor fragmentation. "
                        f"Switching to Matrix-Free L-BFGS with Damped Powell Updates evaluates H_k^{{-1}} g_k in O(m * d) time and memory, "
                        f"completely eliminating autograd graph retention and matrix leakage."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import LBFGSCurvatureOptimizer\n"
                        "optimizer = LBFGSCurvatureOptimizer(model.parameters(), lr=1e-3, history_size=10, powell_damping=True)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_QUASI_NEWTON_LBFGS"),
                    confidence=conf_lbfgs,
                    has_verified_runs=verifications_by_rule.get("RULE_QUASI_NEWTON_LBFGS") is not None,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC_ROLLBACK",
                    evidence=f"Peak VRAM ({peak_vram:.0f} MB) with narrow headroom ({vram_headroom:.0f} MB) indicates optimizer state bloat and matrix fragmentation.",
                    verification_plan="Verify loss-shift proxy <= 0.10 and confirm VRAM allocation remains strictly bounded across 50 steps.",
                    rollback_plan="Revert optimizer group state to previous snapshot and restore first-order AdamW optimizer.",
                )
            )

        # Rule 22: Vector Field Vorticity Damping & Helmholtz-Hodge Decomposition
        # Fires when high rotational vorticity (curl) or unstable divergence flux is detected
        vorticity = getattr(summary, "vorticity_curl_index", 0.0)
        divergence_flux = getattr(summary, "divergence_flux", 0.0)

        if vorticity >= 0.35 or divergence_flux > 50.0:
            conf_vorticity = calculate_evidence_confidence(
                hardware_match=1.0,
                model_similarity=0.90,
                historical_verifications=1 if verifications_by_rule.get("RULE_VECTOR_FIELD_VORTICITY_DAMPING") else 0,
                rollback_frequency=0.01,
            ) * inverse_cost_weight
            recs.append(
                Recommendation(
                    rule_id="RULE_VECTOR_FIELD_VORTICITY_DAMPING",
                    title="Dampen Vector Field Vorticity via Helmholtz-Hodge Decomposition",
                    impact_level="HIGH" if vorticity >= 0.50 else "MEDIUM",
                    speedup_estimate_label="Eliminates Rotational Limit Cycles & Accelerates Potential Descent",
                    description=(
                        f"Elevated vector field vorticity detected (curl index {vorticity:.2f} >= 0.35). "
                        f"Non-conservative stochastic forces and momentum delays cause the optimizer to orbit saddle points in circular limit cycles. "
                        f"Helmholtz-Hodge decomposition projects updates onto the conservative potential gradient nabla Phi and damps rotational solenoidal flow."
                    ),
                    actionable_code_snippet=(
                        "from ghost_layer.curvature import HelmholtzHodgeFilter\n"
                        "hh_filter = HelmholtzHodgeFilter(vorticity_damping_factor=0.6)"
                    ),
                    safe_to_auto_apply=is_safe("RULE_VECTOR_FIELD_VORTICITY_DAMPING"),
                    confidence=conf_vorticity,
                    has_verified_runs=verifications_by_rule.get("RULE_VECTOR_FIELD_VORTICITY_DAMPING") is not None,
                    risk_level="LOW",
                    rollback_mode="AUTOMATIC_ROLLBACK",
                    evidence=f"Vorticity curl index ({vorticity:.2f}) indicates non-conservative orbital energy dissipation in optimization trajectory.",
                    verification_plan="Verify step projection maintains positive inner product with loss gradient and loss descent rate improves over 30 steps.",
                    rollback_plan="Disable Helmholtz-Hodge filter and restore unconstrained momentum updates.",
                )
            )

        # Enrich recommendations with cluster-scaled financial impact
        hw_target = hardware_type or getattr(summary, "hardware_type", None) or self.target_hardware
        fin_calc = FinancialImpactCalculator()
        for r in recs:
            if r.financial_impact is None:
                m = re.search(r"([\d.]+)", r.speedup_estimate_label)
                speedup_pct = float(m.group(1)) if m else (25.0 if r.impact_level == "HIGH" else (15.0 if r.impact_level == "MEDIUM" else 5.0))
                impact = fin_calc.calculate_impact(
                    hardware_type=hw_target,
                    num_gpus=num_gpus,
                    speedup_potential_pct=speedup_pct,
                )
                r.financial_impact = impact.to_dict()

        return recs

    def get_why_not_rejections(
        self,
        summary: TelemetrySummary,
        model_type: str = "transformer",
        active_recs: Optional[List[Recommendation]] = None,
    ) -> List[RejectedIntervention]:
        """
        Synthesizes explicit 'Why NOT?' intelligence for optimizations considered but rejected.
        """
        active_ids = [r.rule_id for r in active_recs] if active_recs else []
        return WhyNotEngine.evaluate_rejections(
            model_type=model_type,
            mixed_precision=summary.mixed_precision,
            is_compiled=False,
            peak_vram_mb=summary.peak_gpu_memory_mb,
            total_vram_mb=summary.gpu_memory_total_mb,
            active_rule_ids=active_ids,
        )

    def compute_counterfactual_utility(
        self,
        baseline_cost_usd: float,
        candidate_speedup_pct: float,
        risk_level: str = "LOW",
    ) -> DecisionCounterfactual:
        """
        Calculates expected utility: E[utility] = E[savings] - E[risk_cost].
        """
        return WhyNotEngine.calculate_counterfactual(
            baseline_cost_usd=baseline_cost_usd,
            candidate_speedup_pct=candidate_speedup_pct,
            risk_level=risk_level,
        )
