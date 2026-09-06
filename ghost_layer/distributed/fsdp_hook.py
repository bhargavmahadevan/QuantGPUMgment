"""
FSDP Ghost Hook: Extends GhostWatcherHook with Fully Sharded Data Parallel
telemetry capture.

Captures FSDP-specific metrics:
- Per-rank shard memory
- All-reduce communication time
- Communication-to-compute ratio
- FSDP-specific optimization recommendations
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.telemetry.watcher import MetricSnapshot
from ghost_layer.decision.engine import Recommendation
from ghost_layer.distributed.collective_telemetry import CollectiveTelemetry
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase


@dataclass
class FSDPMetrics:
    """FSDP-specific telemetry for a single step."""
    rank: int
    world_size: int
    shard_memory_mb: float
    all_reduce_time_ms: float
    all_gather_time_ms: float
    communication_time_ms: float
    compute_time_ms: float
    timestamp: float = field(default_factory=time.time)


class FSDPGhostHook(GhostWatcherHook):
    """
    Extended GhostWatcherHook for FSDP (Fully Sharded Data Parallel) training.
    
    Captures rank-level telemetry and NCCL communication overhead in addition
    to standard GPU utilization and step timing.
    
    Usage:
        hook = FSDPGhostHook(
            gpu_memory_mb=80000.0,
            world_size=8,
            rank=0,
        )
        with hook:
            for step, batch in enumerate(dataloader):
                hook.on_step_begin()
                # ... training step ...
                hook.on_fsdp_step_end(
                    gpu_util_pct=util,
                    gpu_mem_used_mb=mem,
                    loss=loss.item(),
                    shard_memory_mb=shard_mem,
                    all_reduce_time_ms=ar_time,
                    all_gather_time_ms=ag_time,
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
    ):
        super().__init__(
            gpu_memory_mb=gpu_memory_mb,
            target_hardware=target_hardware,
            knowledge_base=knowledge_base,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        self.world_size = world_size
        self.rank = rank
        self.collective_telemetry = CollectiveTelemetry(
            world_size=world_size, rank=rank
        )
        self.fsdp_metrics: List[FSDPMetrics] = []
        self._rank_step_times: Dict[int, List[float]] = {}

    def on_fsdp_step_end(
        self,
        gpu_util_pct: float,
        gpu_mem_used_mb: float,
        loss: Optional[float] = None,
        dataloader_time_ms: float = 0.0,
        mixed_precision: str = "fp32",
        gradient_checkpointing: bool = False,
        flash_attention: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        shard_memory_mb: float = 0.0,
        all_reduce_time_ms: float = 0.0,
        all_gather_time_ms: float = 0.0,
    ) -> MetricSnapshot:
        """Record a step with FSDP-specific metrics."""
        # Record base telemetry via parent
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

        # Record FSDP-specific metrics
        comm_time = all_reduce_time_ms + all_gather_time_ms
        compute_time = max(0.0, snapshot.step_time_ms - comm_time)

        fsdp_metric = FSDPMetrics(
            rank=self.rank,
            world_size=self.world_size,
            shard_memory_mb=shard_memory_mb,
            all_reduce_time_ms=all_reduce_time_ms,
            all_gather_time_ms=all_gather_time_ms,
            communication_time_ms=comm_time,
            compute_time_ms=compute_time,
        )
        self.fsdp_metrics.append(fsdp_metric)

        # Record in collective telemetry
        if all_reduce_time_ms > 0:
            self.collective_telemetry.record_op("all_reduce", all_reduce_time_ms)
        if all_gather_time_ms > 0:
            self.collective_telemetry.record_op("all_gather", all_gather_time_ms)
        self.collective_telemetry.record_compute(snapshot.step_time_ms, comm_time)

        # Track per-rank step times for load balancing analysis
        if self.rank not in self._rank_step_times:
            self._rank_step_times[self.rank] = []
        self._rank_step_times[self.rank].append(snapshot.step_time_ms)

        return snapshot

    def get_fsdp_recommendations(self) -> List[Recommendation]:
        """Generate FSDP-specific optimization recommendations."""
        recs = []

        if not self.fsdp_metrics:
            return recs

        # Calculate averages
        avg_comm_time = sum(m.communication_time_ms for m in self.fsdp_metrics) / len(self.fsdp_metrics)
        avg_compute_time = sum(m.compute_time_ms for m in self.fsdp_metrics) / len(self.fsdp_metrics)
        avg_shard_mem = sum(m.shard_memory_mb for m in self.fsdp_metrics) / len(self.fsdp_metrics)
        total_step_time = avg_comm_time + avg_compute_time
        comm_ratio = avg_comm_time / total_step_time if total_step_time > 0 else 0.0

        # Rule: Communication overhead too high
        if comm_ratio > 0.30:
            recs.append(Recommendation(
                rule_id="RULE_FSDP_COMM_OVERHEAD",
                title="Reduce FSDP Communication Overhead",
                impact_level="HIGH",
                speedup_estimate_label=f"~{comm_ratio*100:.0f}% communication overhead — target <20%",
                description=(
                    f"Communication accounts for {comm_ratio*100:.1f}% of step time "
                    f"({avg_comm_time:.1f}ms comm / {total_step_time:.1f}ms total). "
                    f"Consider gradient accumulation, larger batch sizes, or FSDP HYBRID_SHARD "
                    f"to reduce cross-node communication."
                ),
                actionable_code_snippet=(
                    "from torch.distributed.fsdp import ShardingStrategy\n"
                    "# Use HYBRID_SHARD to limit all-reduce to intra-node:\n"
                    "FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)\n"
                    "# Or increase gradient accumulation:\n"
                    "gradient_accumulation_steps = 4"
                ),
                safe_to_auto_apply=False,
                confidence=0.55,
                has_verified_runs=False,
                risk_level="MEDIUM",
                evidence=f"Measured communication ratio: {comm_ratio:.3f} across {len(self.fsdp_metrics)} steps on {self.world_size} ranks.",
            ))

        # Rule: FSDP mixed precision not enabled
        summary = self.watcher.get_summary()
        if summary.mixed_precision.lower() == "fp32" and self.world_size > 1:
            recs.append(Recommendation(
                rule_id="RULE_FSDP_MIXED_PRECISION",
                title="Enable FSDP-Native Mixed Precision",
                impact_level="HIGH",
                speedup_estimate_label="~40-60% (Published Range for FSDP + BF16)",
                description=(
                    "FSDP supports native mixed precision that reduces both compute "
                    "AND communication volume (sharded parameters in FP16/BF16 transfer "
                    "fewer bytes across NCCL)."
                ),
                actionable_code_snippet=(
                    "from torch.distributed.fsdp import MixedPrecision\n"
                    "mp_policy = MixedPrecision(\n"
                    "    param_dtype=torch.bfloat16,\n"
                    "    reduce_dtype=torch.bfloat16,\n"
                    "    buffer_dtype=torch.bfloat16,\n"
                    ")\n"
                    "FSDP(model, mixed_precision=mp_policy)"
                ),
                safe_to_auto_apply=False,
                confidence=0.55,
                has_verified_runs=False,
                risk_level="MEDIUM",
                evidence=f"Training in FP32 across {self.world_size} FSDP ranks. FSDP MixedPrecision reduces both compute and communication.",
            ))

        # Rule: Activation checkpointing for large shard memory
        if avg_shard_mem > 0 and self.target_gpu_mb > 0:
            shard_usage_pct = (avg_shard_mem / self.target_gpu_mb) * 100.0
            if shard_usage_pct > 75.0 and not summary.gradient_checkpointing:
                recs.append(Recommendation(
                    rule_id="RULE_FSDP_ACTIVATION_CHECKPOINTING",
                    title="Enable FSDP Activation Checkpointing",
                    impact_level="MEDIUM",
                    speedup_estimate_label=f"Shard memory at {shard_usage_pct:.0f}% — enables larger batches",
                    description=(
                        f"Per-rank shard memory is {avg_shard_mem:.0f}MB ({shard_usage_pct:.1f}% of GPU). "
                        f"FSDP activation checkpointing trades recompute for 50-70% activation memory savings."
                    ),
                    actionable_code_snippet=(
                        "from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy\n"
                        "from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (\n"
                        "    checkpoint_wrapper, apply_activation_checkpointing\n"
                        ")\n"
                        "apply_activation_checkpointing(fsdp_model, check_fn=lambda m: isinstance(m, TransformerBlock))"
                    ),
                    safe_to_auto_apply=False,
                    confidence=0.50,
                    has_verified_runs=False,
                    risk_level="LOW",
                    evidence=f"Per-rank shard memory: {avg_shard_mem:.0f}MB / {self.target_gpu_mb:.0f}MB ({shard_usage_pct:.1f}%).",
                ))

        return recs

    def aggregate_across_ranks(self, all_rank_metrics: Dict[int, List[FSDPMetrics]]) -> Dict[str, Any]:
        """
        Aggregate telemetry from all ranks to produce a cluster-wide summary.
        Called on rank 0 after gathering metrics from all ranks.
        """
        all_comm_times = []
        all_compute_times = []
        all_shard_mems = []
        rank_step_counts = {}

        for rank, metrics in all_rank_metrics.items():
            rank_step_counts[rank] = len(metrics)
            for m in metrics:
                all_comm_times.append(m.communication_time_ms)
                all_compute_times.append(m.compute_time_ms)
                all_shard_mems.append(m.shard_memory_mb)

        n = len(all_comm_times)
        if n == 0:
            return {"error": "No metrics to aggregate"}

        return {
            "world_size": self.world_size,
            "total_steps_all_ranks": n,
            "steps_per_rank": rank_step_counts,
            "avg_communication_ms": round(sum(all_comm_times) / n, 3),
            "avg_compute_ms": round(sum(all_compute_times) / n, 3),
            "max_communication_ms": round(max(all_comm_times), 3),
            "avg_shard_memory_mb": round(sum(all_shard_mems) / n, 2),
            "communication_ratio": round(
                sum(all_comm_times) / (sum(all_comm_times) + sum(all_compute_times)), 4
            ) if (sum(all_comm_times) + sum(all_compute_times)) > 0 else 0.0,
        }
