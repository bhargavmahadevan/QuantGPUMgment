"""
DeepSpeed Ghost Hook: Extends GhostWatcherHook with DeepSpeed-specific
telemetry capture for ZeRO optimization stages.

Captures:
- ZeRO stage memory partitioning
- Gradient partitioning overhead
- Pipeline parallel bubble time (if applicable)
- DeepSpeed-specific optimization recommendations
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.telemetry.watcher import MetricSnapshot
from ghost_layer.decision.engine import Recommendation
from ghost_layer.distributed.collective_telemetry import CollectiveTelemetry
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase


@dataclass
class DeepSpeedMetrics:
    """DeepSpeed-specific telemetry for a single step."""
    rank: int
    world_size: int
    zero_stage: int
    partition_memory_mb: float
    gradient_partition_time_ms: float
    optimizer_partition_time_ms: float
    communication_time_ms: float
    compute_time_ms: float
    pipeline_bubble_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DeepSpeedGhostHook(GhostWatcherHook):
    """
    Extended GhostWatcherHook for DeepSpeed training.
    
    Captures ZeRO-stage specific telemetry including partition memory,
    gradient partitioning overhead, and pipeline parallel metrics.
    
    Usage:
        hook = DeepSpeedGhostHook(
            gpu_memory_mb=80000.0,
            world_size=8,
            rank=0,
            zero_stage=2,
        )
        with hook:
            for step, batch in enumerate(dataloader):
                hook.on_step_begin()
                # ... training step ...
                hook.on_deepspeed_step_end(
                    gpu_util_pct=util,
                    gpu_mem_used_mb=mem,
                    loss=loss.item(),
                    partition_memory_mb=part_mem,
                    gradient_partition_time_ms=grad_time,
                )
    """

    def __init__(
        self,
        gpu_memory_mb: float = 80000.0,
        target_hardware: Optional[str] = None,
        knowledge_base: Optional[SharedKnowledgeBase] = None,
        gpu_cost_per_hour: float = 3.50,
        world_size: int = 1,
        rank: int = 0,
        zero_stage: int = 2,
    ):
        super().__init__(
            gpu_memory_mb=gpu_memory_mb,
            target_hardware=target_hardware,
            knowledge_base=knowledge_base,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        self.world_size = world_size
        self.rank = rank
        self.zero_stage = zero_stage
        self.collective_telemetry = CollectiveTelemetry(
            world_size=world_size, rank=rank
        )
        self.deepspeed_metrics: List[DeepSpeedMetrics] = []

    def on_deepspeed_step_end(
        self,
        gpu_util_pct: float,
        gpu_mem_used_mb: float,
        loss: Optional[float] = None,
        dataloader_time_ms: float = 0.0,
        mixed_precision: str = "fp16",
        gradient_checkpointing: bool = False,
        flash_attention: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        partition_memory_mb: float = 0.0,
        gradient_partition_time_ms: float = 0.0,
        optimizer_partition_time_ms: float = 0.0,
        pipeline_bubble_time_ms: float = 0.0,
    ) -> MetricSnapshot:
        """Record a step with DeepSpeed-specific metrics."""
        # Record base telemetry
        snapshot = self.on_step_end(
            gpu_util_pct=gpu_util_pct,
            gpu_mem_used_mb=gpu_mem_used_mb,
            loss=loss,
            dataloader_time_ms=dataloader_time_ms,
            mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing,
            flash_attention=flash_attention,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Record DeepSpeed-specific metrics
        comm_time = gradient_partition_time_ms + optimizer_partition_time_ms
        compute_time = max(0.0, snapshot.step_time_ms - comm_time - pipeline_bubble_time_ms)

        ds_metric = DeepSpeedMetrics(
            rank=self.rank,
            world_size=self.world_size,
            zero_stage=self.zero_stage,
            partition_memory_mb=partition_memory_mb,
            gradient_partition_time_ms=gradient_partition_time_ms,
            optimizer_partition_time_ms=optimizer_partition_time_ms,
            communication_time_ms=comm_time,
            compute_time_ms=compute_time,
            pipeline_bubble_time_ms=pipeline_bubble_time_ms,
        )
        self.deepspeed_metrics.append(ds_metric)

        # Record in collective telemetry
        if gradient_partition_time_ms > 0:
            self.collective_telemetry.record_op("reduce_scatter", gradient_partition_time_ms)
        if optimizer_partition_time_ms > 0:
            self.collective_telemetry.record_op("all_gather", optimizer_partition_time_ms)
        self.collective_telemetry.record_compute(snapshot.step_time_ms, comm_time)

        return snapshot

    def get_deepspeed_recommendations(self) -> List[Recommendation]:
        """Generate DeepSpeed-specific optimization recommendations."""
        recs = []

        if not self.deepspeed_metrics:
            return recs

        avg_comm_time = sum(m.communication_time_ms for m in self.deepspeed_metrics) / len(self.deepspeed_metrics)
        avg_compute_time = sum(m.compute_time_ms for m in self.deepspeed_metrics) / len(self.deepspeed_metrics)
        avg_partition_mem = sum(m.partition_memory_mb for m in self.deepspeed_metrics) / len(self.deepspeed_metrics)
        avg_bubble_time = sum(m.pipeline_bubble_time_ms for m in self.deepspeed_metrics) / len(self.deepspeed_metrics)
        total_step_time = avg_comm_time + avg_compute_time + avg_bubble_time

        # Rule: ZeRO stage upgrade opportunity
        if self.zero_stage < 3 and avg_partition_mem > 0 and self.target_gpu_mb > 0:
            partition_pct = (avg_partition_mem / self.target_gpu_mb) * 100.0
            if partition_pct > 70.0:
                next_stage = self.zero_stage + 1
                recs.append(Recommendation(
                    rule_id="RULE_ZERO_STAGE_UPGRADE",
                    title=f"Upgrade from ZeRO Stage {self.zero_stage} to Stage {next_stage}",
                    impact_level="HIGH",
                    speedup_estimate_label=f"Memory at {partition_pct:.0f}% — Stage {next_stage} enables ~{30*(next_stage-self.zero_stage)}% more memory headroom",
                    description=(
                        f"Current ZeRO Stage {self.zero_stage} partition memory is {avg_partition_mem:.0f}MB "
                        f"({partition_pct:.1f}% of GPU). Stage {next_stage} further partitions "
                        f"{'optimizer states and gradients' if next_stage == 2 else 'parameters, gradients, and optimizer states'} "
                        f"across {self.world_size} ranks."
                    ),
                    actionable_code_snippet=(
                        f'"zero_optimization": {{\n'
                        f'    "stage": {next_stage},\n'
                        f'    "offload_optimizer": {{"device": "cpu"}} // optional for Stage {next_stage}\n'
                        f'}}'
                    ),
                    safe_to_auto_apply=False,
                    confidence=0.50,
                    has_verified_runs=False,
                    risk_level="MEDIUM",
                    evidence=f"ZeRO Stage {self.zero_stage}, partition memory: {avg_partition_mem:.0f}MB ({partition_pct:.1f}%), world_size={self.world_size}.",
                ))

        # Rule: Gradient partitioning overhead
        comm_ratio = avg_comm_time / total_step_time if total_step_time > 0 else 0.0
        if comm_ratio > 0.25:
            recs.append(Recommendation(
                rule_id="RULE_GRADIENT_PARTITIONING",
                title="Optimize DeepSpeed Gradient Partitioning",
                impact_level="MEDIUM",
                speedup_estimate_label=f"Communication overhead: {comm_ratio*100:.0f}%",
                description=(
                    f"Gradient partitioning accounts for {comm_ratio*100:.1f}% of step time. "
                    f"Consider enabling gradient compression, increasing batch size to amortize "
                    f"communication, or using DeepSpeed's overlap_comm option."
                ),
                actionable_code_snippet=(
                    '"zero_optimization": {\n'
                    '    "overlap_comm": true,\n'
                    '    "contiguous_gradients": true,\n'
                    '    "reduce_bucket_size": 5e8\n'
                    '}'
                ),
                safe_to_auto_apply=False,
                confidence=0.50,
                has_verified_runs=False,
                risk_level="LOW",
                evidence=f"Communication ratio: {comm_ratio:.3f} ({avg_comm_time:.1f}ms / {total_step_time:.1f}ms).",
            ))

        # Rule: Pipeline bubble time
        if avg_bubble_time > 0:
            bubble_ratio = avg_bubble_time / total_step_time if total_step_time > 0 else 0.0
            if bubble_ratio > 0.15:
                recs.append(Recommendation(
                    rule_id="RULE_PIPELINE_SCHEDULE",
                    title="Optimize Pipeline Parallel Schedule",
                    impact_level="MEDIUM",
                    speedup_estimate_label=f"Pipeline bubble: {bubble_ratio*100:.0f}% of step time",
                    description=(
                        f"Pipeline bubble time is {avg_bubble_time:.1f}ms ({bubble_ratio*100:.1f}% of step). "
                        f"Consider 1F1B (One Forward One Backward) schedule or increasing micro-batch count "
                        f"to reduce bubble fraction."
                    ),
                    actionable_code_snippet=(
                        '"pipeline": {\n'
                        '    "pipe_partitioned": true,\n'
                        '    "grad_partitioned": true,\n'
                        '    "steps_per_print": 50\n'
                        '}'
                    ),
                    safe_to_auto_apply=False,
                    confidence=0.45,
                    has_verified_runs=False,
                    risk_level="MEDIUM",
                    evidence=f"Pipeline bubble: {avg_bubble_time:.1f}ms ({bubble_ratio*100:.1f}% of step time).",
                ))

        return recs
