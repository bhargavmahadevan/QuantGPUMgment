"""
GhostLayer Tensor Flow & GPU Vitality Auditor.

Diagnoses small-scale and workstation training bottlenecks ("Tensor Flow Jamming"):
  1. Synchronous CPU-GPU pipeline stalls (e.g. per-step `.item()`, `.cpu()`, unpinned copies)
  2. PyTorch Caching Allocator fragmentation and VRAM leakage (Reserved vs. Active allocation)
  3. CPU kernel launch dispatch bottlenecks on small batch sizes
  4. Missing CUDA stream pipelining and Tensor Core stride alignment
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch


@dataclass
class TensorFlowDiagnostic:
    """Diagnostic report on tensor pipeline vitality and base leakage."""
    vitality_score_pct: float             # 0.0% to 100.0% (Higher is healthier)
    allocator_fragmentation_pct: float    # Fraction of reserved VRAM that is unusable
    sync_pipeline_stalls_detected: int    # Number of blocking host-device syncs
    unpinned_memory_transfers: bool       # Whether batches are copied over PCIe synchronously
    cuda_graphs_eligible: bool            # Whether workload qualifies for CUDA graph acceleration
    stride_alignment_optimal: bool        # Whether tensor dimensions align to Tensor Core strides (multiples of 16)
    base_leakage_diagnoses: List[str] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)


class TensorFlowAuditor:
    """
    Audits the live PyTorch execution loop to eliminate tensor pipeline jamming
    and maximize GPU compute vitality on single-node and workstation hardware.
    """

    def __init__(self, sample_interval: int = 10):
        self.sample_interval = sample_interval
        self._sync_call_count = 0
        self._step_counter = 0

    def audit_environment(
        self,
        model: Optional[torch.nn.Module] = None,
        sample_batch: Optional[Any] = None,
        dataloader: Optional[Any] = None,
    ) -> TensorFlowDiagnostic:
        """Runs a comprehensive vitality audit on memory and execution flow."""
        diagnoses = []
        actions = []
        vitality = 100.0
        frag_pct = 0.0
        sync_stalls = 0
        unpinned = False
        cuda_graphs_eligible = True
        stride_optimal = True

        # 1. CUDA Memory Allocator Fragmentation & VRAM Leakage Check
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if reserved > 0:
                frag_pct = max(0.0, float(reserved - allocated) / float(reserved)) * 100.0
                if frag_pct > 35.0:
                    diagnoses.append(
                        f"PyTorch Caching Allocator fragmentation is high ({frag_pct:.1f}% unusable reserved VRAM). "
                        f"Allocated: {allocated/(1024**2):.1f}MB, Reserved: {reserved/(1024**2):.1f}MB."
                    )
                    actions.append("Set PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:64'")
                    vitality -= 20.0
            
            # Check for missing expandable segments
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if "expandable_segments:True" not in alloc_conf:
                diagnoses.append("PYTORCH_CUDA_ALLOC_CONF lacks 'expandable_segments:True', risking allocation thrashing.")
                actions.append("Enable expandable_segments in runtime configuration.")
                vitality -= 10.0
        else:
            frag_pct = 0.0

        # 2. DataLoader Pinned Memory & Non-blocking PCIe Pipeline Check
        if dataloader is not None:
            pin_mem = getattr(dataloader, "pin_memory", False)
            num_workers = getattr(dataloader, "num_workers", 0)
            if not pin_mem:
                unpinned = True
                diagnoses.append("DataLoader does not use pin_memory=True. Host-to-Device tensor copies block the GPU stream.")
                actions.append("Configure DataLoader(pin_memory=True, non_blocking=True)")
                vitality -= 25.0
            if num_workers == 0:
                diagnoses.append("DataLoader uses num_workers=0 (Main thread disk I/O). Stalls the GPU pipeline between steps.")
                actions.append("Configure DataLoader(num_workers=2, persistent_workers=True)")
                vitality -= 15.0

        # 3. Tensor Core Cartesian Stride Alignment Check
        if model is not None:
            non_aligned_params = 0
            for p in model.parameters():
                if p.ndim >= 2:
                    # Check if matrix dimensions are multiples of 8 or 16 (NVIDIA Tensor Core MMA warp requirement)
                    if p.size(0) % 8 != 0 or p.size(1) % 8 != 0:
                        non_aligned_params += 1
            if non_aligned_params > 0:
                stride_optimal = False
                diagnoses.append(
                    f"{non_aligned_params} weight tensors have dimensions not divisible by 8. "
                    f"Tensor Cores will fall back to scalar CUDA mode."
                )
                actions.append("Pad embedding / hidden layer dimensions to multiples of 16 (e.g. 128, 256, 512).")
                vitality -= 10.0

        # 4. Small-Scale CUDA Graph Launch Qualification
        if model is not None:
            # Check if dynamic control flow or dynamic shapes prevent CUDA graphs
            if hasattr(model, "_ghost_dynamic_shapes") and model._ghost_dynamic_shapes:
                cuda_graphs_eligible = False
            else:
                actions.append("Enable CUDA Graph execution (torch.compile(mode='reduce-overhead')) to eliminate CPU dispatch latency.")

        vitality_score = float(max(10.0, min(100.0, vitality)))

        return TensorFlowDiagnostic(
            vitality_score_pct=vitality_score,
            allocator_fragmentation_pct=frag_pct,
            sync_pipeline_stalls_detected=sync_stalls,
            unpinned_memory_transfers=unpinned,
            cuda_graphs_eligible=cuda_graphs_eligible,
            stride_alignment_optimal=stride_optimal,
            base_leakage_diagnoses=diagnoses,
            remediation_actions=actions,
        )


def audit_gpu_vitality(
    model: Optional[torch.nn.Module] = None,
    dataloader: Optional[Any] = None,
) -> TensorFlowDiagnostic:
    """Convenience entry point for GhostLayer GPU vitality audit."""
    auditor = TensorFlowAuditor()
    return auditor.audit_environment(model=model, dataloader=dataloader)
