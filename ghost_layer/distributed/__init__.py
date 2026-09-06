"""
GhostLayer Distributed Training Hooks.

Extends GhostLayer's telemetry capture to multi-GPU and multi-node training:
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed (ZeRO stages)
- Collective communication telemetry (NCCL)
"""

from ghost_layer.distributed.collective_telemetry import CollectiveTelemetry
from ghost_layer.distributed.fsdp_hook import FSDPGhostHook
from ghost_layer.distributed.deepspeed_hook import DeepSpeedGhostHook
from ghost_layer.distributed.signature_embedding import (
    ExecutionSignatureEmbedding,
    ExecutionSignatureRegistry,
    MatchResult,
    cosine_similarity,
)

__all__ = [
    "CollectiveTelemetry",
    "FSDPGhostHook",
    "DeepSpeedGhostHook",
    "ExecutionSignatureEmbedding",
    "ExecutionSignatureRegistry",
    "MatchResult",
    "cosine_similarity",
]
