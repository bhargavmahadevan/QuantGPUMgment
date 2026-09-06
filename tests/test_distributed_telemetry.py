"""
GhostLayer Distributed Training Telemetry & Multi-Rank Hooks Test Suite.
Combines tests for:
- Distributed context detection (Standalone vs Multi-Rank env vars)
- DistributedTelemetryCoordinator rank gating & metric snapshot aggregation
- CollectiveTelemetry communication ratio and compute time
- FSDPGhostHook shard memory tracking and communication overhead rules
- DeepSpeedGhostHook ZeRO stage metrics and partition memory rules
"""

import os
import pytest
from ghost_layer.distributed.distributed_telemetry import (
    DistributedTelemetryCoordinator,
    detect_distributed_context,
)
from ghost_layer.distributed import (
    CollectiveTelemetry,
    FSDPGhostHook,
    DeepSpeedGhostHook,
)
from ghost_layer.telemetry.watcher import MetricSnapshot


# ==============================================================================
# 1. Distributed Context & Coordinator Tests
# ==============================================================================

def test_detect_distributed_context_standalone():
    ctx = detect_distributed_context()
    assert ctx["is_rank_zero"] is True


def test_detect_distributed_context_env_vars(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "8")

    ctx = detect_distributed_context()
    assert ctx["is_distributed"] is True
    assert ctx["rank"] == 2
    assert ctx["world_size"] == 8
    assert ctx["is_rank_zero"] is False


def test_distributed_coordinator_rank_gating():
    coord_rank0 = DistributedTelemetryCoordinator(rank=0, world_size=4)
    assert coord_rank0.is_rank_zero is True
    assert coord_rank0.should_emit_audit_reports() is True

    coord_rank2 = DistributedTelemetryCoordinator(rank=2, world_size=4)
    assert coord_rank2.is_rank_zero is False
    assert coord_rank2.should_emit_audit_reports() is False


def test_distributed_coordinator_passthrough_metric_snapshot():
    coord = DistributedTelemetryCoordinator(rank=0, world_size=1)
    snapshot = MetricSnapshot(
        step=1,
        timestamp=0.1,
        gpu_utilization_pct=90.0,
        gpu_memory_used_mb=5000.0,
        gpu_memory_total_mb=16000.0,
        step_time_ms=20.0,
    )
    res = coord.aggregate_step_snapshot(snapshot)
    assert res.gpu_memory_used_mb == 5000.0
    assert res.gpu_utilization_pct == 90.0


# ==============================================================================
# 2. Collective Telemetry & Distributed Hooks (FSDP & DeepSpeed) Tests
# ==============================================================================

def test_collective_telemetry_ratio():
    ct = CollectiveTelemetry(world_size=8, rank=0)
    ct.record_op("all_reduce", duration_ms=10.0, tensor_size_bytes=10_000_000)
    ct.record_op("all_gather", duration_ms=5.0, tensor_size_bytes=5_000_000)
    ct.record_compute(step_time_ms=50.0, comm_time_ms=15.0)

    summary = ct.get_summary()
    assert summary.world_size == 8
    assert summary.total_communication_ms == 15.0
    assert summary.total_compute_ms == 35.0
    assert summary.communication_ratio == pytest.approx(15.0 / 50.0, rel=1e-2)


def test_fsdp_ghost_hook_step():
    hook = FSDPGhostHook(
        gpu_memory_mb=80000.0,
        world_size=8,
        rank=0,
        target_hardware="NVIDIA H100",
    )

    with hook:
        hook.on_step_begin()
        snapshot = hook.on_fsdp_step_end(
            gpu_util_pct=75.0,
            gpu_mem_used_mb=60000.0,
            loss=1.85,
            shard_memory_mb=65000.0,
            all_reduce_time_ms=20.0,
            all_gather_time_ms=10.0,
        )

    assert len(hook.fsdp_metrics) == 1
    recs = hook.get_fsdp_recommendations()
    rule_ids = [r.rule_id for r in recs]
    assert "RULE_FSDP_COMM_OVERHEAD" in rule_ids
    assert "RULE_FSDP_ACTIVATION_CHECKPOINTING" in rule_ids


def test_deepspeed_ghost_hook_step():
    hook = DeepSpeedGhostHook(
        gpu_memory_mb=80000.0,
        world_size=8,
        rank=0,
        zero_stage=1,
        target_hardware="NVIDIA A100",
    )

    with hook:
        hook.on_step_begin()
        hook.on_deepspeed_step_end(
            gpu_util_pct=80.0,
            gpu_mem_used_mb=65000.0,
            loss=2.1,
            partition_memory_mb=62000.0,
            gradient_partition_time_ms=15.0,
            optimizer_partition_time_ms=10.0,
        )

    assert len(hook.deepspeed_metrics) == 1
    recs = hook.get_deepspeed_recommendations()
    rule_ids = [r.rule_id for r in recs]
    assert "RULE_ZERO_STAGE_UPGRADE" in rule_ids
