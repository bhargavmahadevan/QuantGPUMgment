"""
Distributed Telemetry & Rank Coordinator:
Handles multi-process DDP / FSDP distributed training environments.
Synchronizes cross-rank telemetry (peak VRAM, mean utilization, max step time)
and guards against redundant/duplicate audit reports across non-zero ranks.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from ghost_layer.telemetry.watcher import MetricSnapshot, TelemetrySummary


def detect_distributed_context() -> Dict[str, Any]:
    """Detect if running inside a PyTorch distributed environment (DDP/FSDP/DeepSpeed)."""
    rank = 0
    world_size = 1
    is_distributed = False

    # 1. Check torch.distributed if imported
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            is_distributed = True
    except Exception:
        pass

    # 2. Check standard environment variables (torchrun / slurm / accelerate)
    if not is_distributed:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            try:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                is_distributed = (world_size > 1)
            except ValueError:
                pass
        elif "LOCAL_RANK" in os.environ:
            try:
                rank = int(os.environ["LOCAL_RANK"])
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                is_distributed = (world_size > 1)
            except ValueError:
                pass

    return {
        "is_distributed": is_distributed,
        "rank": rank,
        "world_size": world_size,
        "is_rank_zero": (rank == 0),
    }


class DistributedTelemetryCoordinator:
    """
    Coordinates telemetry aggregation and audit reporting across distributed ranks.
    """

    def __init__(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        ctx = detect_distributed_context()
        self.rank = ctx["rank"] if rank is None else rank
        self.world_size = ctx["world_size"] if world_size is None else world_size
        self.is_distributed = (self.world_size > 1)

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def aggregate_step_snapshot(self, snapshot: MetricSnapshot) -> MetricSnapshot:
        """
        In distributed mode with active torch.distributed, aggregates global metrics.
        In standalone mode, returns snapshot unchanged.
        """
        if not self.is_distributed:
            return snapshot

        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                # All-reduce max VRAM across all ranks
                vram_t = torch.tensor([snapshot.gpu_memory_used_mb], dtype=torch.float32)
                dist.all_reduce(vram_t, op=dist.ReduceOp.MAX)
                global_max_vram = float(vram_t.item())

                # All-reduce avg GPU util
                util_t = torch.tensor([snapshot.gpu_utilization_pct], dtype=torch.float32)
                dist.all_reduce(util_t, op=dist.ReduceOp.SUM)
                global_avg_util = float(util_t.item()) / self.world_size

                # All-reduce max step time (slowest rank dictates barrier)
                step_t = torch.tensor([snapshot.step_time_ms], dtype=torch.float32)
                dist.all_reduce(step_t, op=dist.ReduceOp.MAX)
                global_max_step_ms = float(step_t.item())

                snapshot.gpu_memory_used_mb = global_max_vram
                snapshot.gpu_utilization_pct = global_avg_util
                snapshot.step_time_ms = global_max_step_ms
        except Exception:
            pass

        return snapshot

    def should_emit_audit_reports(self) -> bool:
        """Only Rank 0 should dispatch external notifications (Slack, Prometheus, Webhooks)."""
        return self.is_rank_zero

    def should_auto_apply_rule(self, rule_id: str) -> bool:
        """
        Determines if an optimization rule should be applied on this specific rank.
        Optimizer, DataLoader, and Compile modifications apply per-rank.
        Global cluster actions coordinate via Rank 0.
        """
        return True
