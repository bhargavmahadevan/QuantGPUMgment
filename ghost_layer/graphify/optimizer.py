import copy
from dataclasses import dataclass
from typing import List
from ghost_layer.graphify.graph_builder import ExecutionGraphBuilder
from ghost_layer.graphify.analyzer import GraphAnalyzer

@dataclass
class OptimizationImpactReport:
    original_critical_path_ms: float
    optimized_critical_path_ms: float
    speedup_pct: float
    original_peak_memory_mb: float
    optimized_peak_memory_mb: float
    memory_savings_pct: float
    fused_kernels_count: int
    applied_passes: List[str]

class GraphOptimizer:
    """
    Automated DAG Transformation & Optimization Engine.
    Applies graph rewriting passes (Kernel Fusion, Attention Acceleration, Selective Recomputation, Precision Casting).
    """
    def __init__(self, graph_builder: ExecutionGraphBuilder):
        self.original_builder = graph_builder

    def optimize(
        self,
        enable_kernel_fusion: bool = True,
        enable_flash_attention: bool = True,
        enable_recomputation: bool = True,
        enable_fp16_cast: bool = True,
        enable_matrix_free_epilogue: bool = True
    ) -> tuple[ExecutionGraphBuilder, OptimizationImpactReport]:
        
        # Analyze baseline
        baseline_analyzer = GraphAnalyzer(self.original_builder)
        base_summary = baseline_analyzer.analyze()

        # Clone builder for graph transformations
        opt_builder = ExecutionGraphBuilder(model_name=f"{self.original_builder.model_name}_Optimized")
        opt_builder.nodes = copy.deepcopy(self.original_builder.nodes)
        opt_builder.edges = copy.deepcopy(self.original_builder.edges)

        applied_passes = []
        fused_count = 0

        # Pass 1: Precision Casting (FP32 -> BF16/FP16)
        if enable_fp16_cast:
            applied_passes.append("PrecisionCastingPass (FP32 -> BF16)")
            for edge in opt_builder.edges:
                edge.data_type = "bfloat16"
                edge.data_transfer_mb = round(edge.data_transfer_mb * 0.5, 2)
            for node in opt_builder.nodes.values():
                node.memory_footprint_mb = round(node.memory_footprint_mb * 0.55, 2)

        # Pass 2: FlashAttention Acceleration
        if enable_flash_attention:
            applied_passes.append("AttentionOptimizationPass (FlashAttention-2 Kernel Integration)")
            for node in opt_builder.nodes.values():
                if node.node_type == "ATTENTION" or node.op_type == "attention":
                    node.execution_time_ms = round(node.execution_time_ms * 0.65, 2)
                    node.memory_footprint_mb = round(node.memory_footprint_mb * 0.35, 2)

        # Pass 3: Kernel Fusion (RMSNorm + Activation + Elementwise)
        if enable_kernel_fusion and base_summary.recommended_torch_compile:
            applied_passes.append("KernelFusionPass (Triton / Inductor Fused CUDA Kernels)")
            fusible_ids = base_summary.fusible_subgraph_nodes
            fused_count = len(fusible_ids)
            for nid in fusible_ids:
                if nid in opt_builder.nodes:
                    node = opt_builder.nodes[nid]
                    node.node_type = "FUSED_KERNEL"
                    node.execution_time_ms = round(node.execution_time_ms * 0.72, 2)
                    node.memory_footprint_mb = round(node.memory_footprint_mb * 0.80, 2)

        # Pass 4: Selective Recomputation / Gradient Checkpointing
        if enable_recomputation:
            applied_passes.append("SelectiveRecomputationPass (Activation Recomputation)")
            for nid in base_summary.recomputation_candidate_nodes:
                if nid in opt_builder.nodes:
                    node = opt_builder.nodes[nid]
                    # Memory drops by 70%, small recalculation cost added
                    node.memory_footprint_mb = round(node.memory_footprint_mb * 0.30, 2)
                    node.execution_time_ms = round(node.execution_time_ms * 1.05, 2)

        # Pass 5: Matrix-Free Operator Epilogue Pass (Inspired by GPU Topology Optimization EBE Solvers)
        if enable_matrix_free_epilogue:
            applied_passes.append("MatrixFreeEpiloguePass (Element-by-Element Register-Resident Fusion)")
            for nid, node in opt_builder.nodes.items():
                if node.node_type == "OPERATOR" or "norm" in node.op_type or "linear" in node.op_type:
                    node.execution_time_ms = round(node.execution_time_ms * 0.88, 2)
                    node.memory_footprint_mb = round(node.memory_footprint_mb * 0.80, 2)

        # Analyze optimized graph
        opt_analyzer = GraphAnalyzer(opt_builder)
        opt_summary = opt_analyzer.analyze()

        orig_cp = base_summary.critical_path_length_ms
        opt_cp = opt_summary.critical_path_length_ms
        speedup_pct = round(((orig_cp - opt_cp) / max(orig_cp, 0.1)) * 100.0, 1)

        orig_mem = base_summary.max_memory_node_mb
        opt_mem = opt_summary.max_memory_node_mb
        mem_savings = round(((orig_mem - opt_mem) / max(orig_mem, 0.1)) * 100.0, 1)

        report = OptimizationImpactReport(
            original_critical_path_ms=orig_cp,
            optimized_critical_path_ms=opt_cp,
            speedup_pct=max(speedup_pct, 0.0),
            original_peak_memory_mb=orig_mem,
            optimized_peak_memory_mb=opt_mem,
            memory_savings_pct=max(mem_savings, 0.0),
            fused_kernels_count=fused_count,
            applied_passes=applied_passes
        )

        return opt_builder, report
