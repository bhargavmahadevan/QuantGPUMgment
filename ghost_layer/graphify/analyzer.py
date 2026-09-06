from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from collections import deque
from ghost_layer.graphify.graph_builder import ExecutionGraphBuilder

@dataclass
class GraphBottleneckSummary:
    total_nodes: int
    total_edges: int
    critical_path_length_ms: float
    max_memory_node_id: str
    max_memory_node_mb: float
    fusible_subgraph_nodes: List[str]
    attention_nodes_count: int
    recommended_torch_compile: bool
    # Advanced Graphify Metrics
    critical_path_node_ids: List[str] = field(default_factory=list)
    fusible_clusters_count: int = 0
    estimated_fusion_speedup_pct: float = 0.0
    bandwidth_saved_gbps: float = 0.0
    recomputation_candidate_nodes: List[str] = field(default_factory=list)
    parallelism_split_points: List[str] = field(default_factory=list)
    mermaid_diagram: str = ""

class GraphFusionEngine:
    """
    Identifies fusible operator clusters in computational DAGs (RMSNorm, LayerNorm, SwiGLU, GELU, BiasAdd)
    that can be compiled into unified CUDA/Triton kernels.
    """
    @staticmethod
    def identify_fusible_clusters(graph_builder: ExecutionGraphBuilder) -> List[List[str]]:
        nodes = graph_builder.nodes
        fusible_node_ids = set()

        for nid, node in nodes.items():
            if node.is_fusible or node.node_type == "OPERATOR" or any(k in node.label for k in ["Norm", "MLP", "Activation", "SwiGLU", "GELU", "Residual"]):
                fusible_node_ids.add(nid)

        clusters: List[List[str]] = []
        visited: Set[str] = set()

        topo_order = graph_builder.topological_sort()

        for nid in topo_order:
            if nid in fusible_node_ids and nid not in visited:
                cluster = [nid]
                visited.add(nid)

                # Expand cluster through successors
                queue = deque([nid])
                while queue:
                    curr = queue.popleft()
                    for succ in graph_builder.get_successors(curr):
                        if succ in fusible_node_ids and succ not in visited:
                            visited.add(succ)
                            cluster.append(succ)
                            queue.append(succ)
                clusters.append(cluster)

        return clusters

class GraphAnalyzer:
    """
    Analyzes computational DAGs created by Graphify to identify critical path execution delays,
    fusible operator clusters (`torch.compile` targets), peak VRAM bottlenecks, and parallelism split points.
    """
    def __init__(self, graph_builder: ExecutionGraphBuilder):
        self.graph = graph_builder

    def find_critical_path(self) -> tuple[float, List[str]]:
        """
        Calculates the true critical path (longest path through DAG) using dynamic programming over topological order.
        """
        nodes = self.graph.nodes
        edges = self.graph.edges

        if not nodes:
            return 0.0, []

        topo_order = self.graph.topological_sort()
        dist: Dict[str, float] = {nid: 0.0 for nid in nodes}
        parent: Dict[str, Optional[str]] = {nid: None for nid in nodes}

        # Build adjacency list for O(V+E) edge traversal
        adj: Dict[str, List[Any]] = {nid: [] for nid in nodes}
        for edge in edges:
            adj[edge.source_id].append(edge)

        # Initialize distances with node execution times
        for nid in topo_order:
            dist[nid] += nodes[nid].execution_time_ms

        for u in topo_order:
            for edge in adj[u]:
                v = edge.target_id
                # Edge latency = data_transfer_mb / 100.0 (simulated 100 GB/s bandwidth)
                edge_cost = edge.data_transfer_mb / 100.0
                new_dist = dist[u] + edge_cost + nodes[v].execution_time_ms

                if new_dist > dist[v]:
                    dist[v] = new_dist
                    parent[v] = u

        # Find sink with maximum distance
        max_node = max(topo_order, key=lambda n: dist[n])
        max_dist = dist[max_node]

        # Reconstruct path
        path = []
        curr: Optional[str] = max_node
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()

        # Mark critical path nodes in graph builder
        for nid in nodes:
            nodes[nid].is_critical_path = (nid in path)

        return round(max_dist, 2), path

    def analyze(self) -> GraphBottleneckSummary:
        nodes = self.graph.nodes
        edges = self.graph.edges

        if not nodes:
            return GraphBottleneckSummary(
                total_nodes=0,
                total_edges=0,
                critical_path_length_ms=0.0,
                max_memory_node_id="",
                max_memory_node_mb=0.0,
                fusible_subgraph_nodes=[],
                attention_nodes_count=0,
                recommended_torch_compile=False,
                critical_path_node_ids=[],
                fusible_clusters_count=0,
                estimated_fusion_speedup_pct=0.0,
                bandwidth_saved_gbps=0.0,
                recomputation_candidate_nodes=[],
                parallelism_split_points=[],
                mermaid_diagram=""
            )

        # True Critical Path
        critical_path_ms, critical_path_nodes = self.find_critical_path()

        # Peak Memory Node
        max_mem_node = max(nodes.values(), key=lambda n: n.memory_footprint_mb)

        # Identify Fusible Operators & Clusters
        fusible_clusters = GraphFusionEngine.identify_fusible_clusters(self.graph)
        fusible_nodes = [
            n.node_id for n in nodes.values()
            if n.is_fusible or n.node_type == "OPERATOR" or any(k in n.label for k in ["Norm", "MLP", "Activation", "SwiGLU", "GELU", "Residual"])
        ]

        attention_nodes = [n for n in nodes.values() if n.node_type == "ATTENTION" or n.op_type == "attention"]

        # Recommend torch.compile if there are >= 3 fusible operators
        rec_compile = len(fusible_nodes) >= 3

        # Estimate Speedup % & Bandwidth Saved from Kernel Fusion
        # Fusing elementwise/norm ops eliminates HBM roundtrips: ~25-30% speedup on fusible subgraphs
        fusible_time = sum(nodes[nid].execution_time_ms for nid in fusible_nodes)
        total_time = sum(n.execution_time_ms for n in nodes.values())
        fusion_speedup_pct = round((fusible_time * 0.28 / max(total_time, 0.1)) * 100.0, 1) if rec_compile else 0.0
        bandwidth_saved_gbps = round(len(fusible_clusters) * 45.0, 1)

        # Identify Recomputation Candidates (High memory, non-critical or module nodes)
        recomp_candidates = [
            n.node_id for n in nodes.values()
            if n.memory_footprint_mb > 500.0 and n.node_type in ["MODULE", "ATTENTION"]
        ]

        # Parallelism split points (Attention / MLP layer boundaries)
        parallel_splits = [
            n.node_id for n in nodes.values()
            if "attention" in n.node_id or "mlp" in n.node_id
        ]

        mermaid_diag = self.graph.to_mermaid()

        return GraphBottleneckSummary(
            total_nodes=len(nodes),
            total_edges=len(edges),
            critical_path_length_ms=critical_path_ms,
            max_memory_node_id=max_mem_node.node_id,
            max_memory_node_mb=round(max_mem_node.memory_footprint_mb, 2),
            fusible_subgraph_nodes=fusible_nodes,
            attention_nodes_count=len(attention_nodes),
            recommended_torch_compile=rec_compile,
            critical_path_node_ids=critical_path_nodes,
            fusible_clusters_count=len(fusible_clusters),
            estimated_fusion_speedup_pct=fusion_speedup_pct,
            bandwidth_saved_gbps=bandwidth_saved_gbps,
            recomputation_candidate_nodes=recomp_candidates,
            parallelism_split_points=parallel_splits,
            mermaid_diagram=mermaid_diag
        )
