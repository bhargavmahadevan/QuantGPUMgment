"""
Collective Communication Telemetry: Captures torch.distributed NCCL
metrics for diagnosing communication overhead in distributed training.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class CollectiveOp:
    """A single recorded collective communication operation."""
    op_type: str  # all_reduce, all_gather, reduce_scatter, barrier, broadcast
    duration_ms: float
    tensor_size_bytes: int
    timestamp: float = field(default_factory=time.time)
    rank: int = 0
    world_size: int = 1


@dataclass
class CollectiveSummary:
    """Summary of collective communication overhead for a training window."""
    total_all_reduce_ms: float
    total_all_gather_ms: float
    total_reduce_scatter_ms: float
    total_barrier_ms: float
    total_broadcast_ms: float
    total_communication_ms: float
    total_compute_ms: float
    communication_ratio: float  # comm_time / (comm_time + compute_time)
    num_ops: int
    world_size: int
    avg_bandwidth_gbps: float


class CollectiveTelemetry:
    """
    Captures torch.distributed collective communication metrics.
    
    Hooks into NCCL operations to measure communication overhead and
    compute the compute-to-communication ratio — the key metric for
    distributed training efficiency.
    
    Usage:
        ct = CollectiveTelemetry(world_size=8)
        ct.record_op("all_reduce", duration_ms=2.5, tensor_size_bytes=4_000_000)
        ct.record_compute(step_time_ms=30.0, comm_time_ms=5.0)
        summary = ct.get_summary()
    """

    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        self._ops: List[CollectiveOp] = []
        self._compute_times_ms: List[float] = []
        self._comm_times_ms: List[float] = []

    def record_op(
        self,
        op_type: str,
        duration_ms: float,
        tensor_size_bytes: int = 0,
    ) -> CollectiveOp:
        """Record a single collective communication operation."""
        op = CollectiveOp(
            op_type=op_type,
            duration_ms=duration_ms,
            tensor_size_bytes=tensor_size_bytes,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._ops.append(op)
        return op

    def record_compute(self, step_time_ms: float, comm_time_ms: float):
        """Record compute vs. communication time for a single step."""
        compute_time = max(0.0, step_time_ms - comm_time_ms)
        self._compute_times_ms.append(compute_time)
        self._comm_times_ms.append(comm_time_ms)

    def get_summary(self) -> CollectiveSummary:
        """Generate a summary of collective communication overhead."""
        by_type: Dict[str, float] = {
            "all_reduce": 0.0,
            "all_gather": 0.0,
            "reduce_scatter": 0.0,
            "barrier": 0.0,
            "broadcast": 0.0,
        }

        total_bytes = 0
        total_comm_time = 0.0

        for op in self._ops:
            key = op.op_type.lower()
            if key in by_type:
                by_type[key] += op.duration_ms
            total_comm_time += op.duration_ms
            total_bytes += op.tensor_size_bytes

        total_compute = sum(self._compute_times_ms) if self._compute_times_ms else 0.0
        total_comm_from_steps = sum(self._comm_times_ms) if self._comm_times_ms else total_comm_time

        # Use the more detailed step-level breakdown if available
        effective_comm = total_comm_from_steps if self._comm_times_ms else total_comm_time
        total_time = effective_comm + total_compute

        ratio = effective_comm / total_time if total_time > 0 else 0.0

        # Average bandwidth: total_bytes / total_comm_time_seconds
        avg_bandwidth = 0.0
        if total_comm_time > 0:
            avg_bandwidth = (total_bytes / (total_comm_time / 1000.0)) / 1e9  # GB/s

        return CollectiveSummary(
            total_all_reduce_ms=round(by_type["all_reduce"], 3),
            total_all_gather_ms=round(by_type["all_gather"], 3),
            total_reduce_scatter_ms=round(by_type["reduce_scatter"], 3),
            total_barrier_ms=round(by_type["barrier"], 3),
            total_broadcast_ms=round(by_type["broadcast"], 3),
            total_communication_ms=round(effective_comm, 3),
            total_compute_ms=round(total_compute, 3),
            communication_ratio=round(ratio, 4),
            num_ops=len(self._ops),
            world_size=self.world_size,
            avg_bandwidth_gbps=round(avg_bandwidth, 3),
        )

    def compute_communication_ratio(self) -> float:
        """The key distributed efficiency metric: what fraction of time is spent communicating."""
        summary = self.get_summary()
        return summary.communication_ratio

    def clear(self):
        """Reset all recorded data."""
        self._ops.clear()
        self._compute_times_ms.clear()
        self._comm_times_ms.clear()
