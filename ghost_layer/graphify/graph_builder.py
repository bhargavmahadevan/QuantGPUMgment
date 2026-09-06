import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from collections import deque

@dataclass(slots=True)
class GraphNode:
    node_id: str
    label: str
    node_type: str  # OPERATOR, MODULE, TENSOR_TRANSFER, ATTENTION, FUSED_KERNEL
    execution_time_ms: float = 0.0
    memory_footprint_mb: float = 0.0
    is_critical_path: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    op_type: str = "general"  # norm, activation, elementwise, gemm, attention, router, ssm
    is_fusible: bool = False
    subgraph_id: Optional[str] = None
    stage_id: int = 0

@dataclass(slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    tensor_shape: Optional[List[int]] = None
    data_transfer_mb: float = 0.0
    data_type: str = "bfloat16"
    is_control_dependency: bool = False

class ExecutionGraphBuilder:
    """
    Builds Directed Acyclic Execution Graphs (DAGs) representing PyTorch model architecture,
    forward/backward pass dataflows, GPU memory usage, critical path bottlenecks, and operator fusion clusters.
    """
    def __init__(self, model_name: str = "PyTorchModel"):
        self.model_name = model_name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._successors: Dict[str, List[str]] = {}
        self._predecessors: Dict[str, List[str]] = {}

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        execution_time_ms: float = 0.0,
        memory_footprint_mb: float = 0.0,
        attributes: Optional[Dict[str, Any]] = None,
        op_type: str = "general",
        is_fusible: bool = False,
        subgraph_id: Optional[str] = None,
        stage_id: int = 0
    ) -> GraphNode:
        node = GraphNode(
            node_id=node_id,
            label=label,
            node_type=node_type,
            execution_time_ms=execution_time_ms,
            memory_footprint_mb=memory_footprint_mb,
            attributes=attributes or {},
            op_type=op_type,
            is_fusible=is_fusible,
            subgraph_id=subgraph_id,
            stage_id=stage_id
        )
        self.nodes[node_id] = node
        if node_id not in self._successors:
            self._successors[node_id] = []
        if node_id not in self._predecessors:
            self._predecessors[node_id] = []
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        tensor_shape: Optional[List[int]] = None,
        data_transfer_mb: float = 0.0,
        data_type: str = "bfloat16",
        is_control_dependency: bool = False
    ) -> GraphEdge:
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            tensor_shape=tensor_shape,
            data_transfer_mb=data_transfer_mb,
            data_type=data_type,
            is_control_dependency=is_control_dependency
        )
        self.edges.append(edge)
        
        if source_id not in self._successors:
            self._successors[source_id] = []
        if target_id not in self._predecessors:
            self._predecessors[target_id] = []
            
        self._successors[source_id].append(target_id)
        self._predecessors[target_id].append(source_id)
        
        return edge

    def build_transformer_dag(
        self,
        num_layers: int = 12,
        hidden_dim: int = 4096,
        sequence_length: int = 2048
    ) -> "ExecutionGraphBuilder":
        """Builds a canonical Transformer computational DAG for bottleneck graph analysis."""
        prev_node_id = "embedding"
        self.add_node(
            "embedding",
            "Token & Positional Embedding",
            "MODULE",
            execution_time_ms=2.5,
            memory_footprint_mb=256.0,
            op_type="gemm",
            is_fusible=False,
            stage_id=0
        )

        for i in range(num_layers):
            layer_attn_id = f"layer_{i}_attention"
            layer_mlp_id = f"layer_{i}_mlp"
            layer_norm_id = f"layer_{i}_norm"

            self.add_node(
                layer_attn_id,
                f"Layer {i} Multi-Head Self-Attention",
                "ATTENTION",
                execution_time_ms=18.5,
                memory_footprint_mb=1200.0,
                attributes={"num_heads": 32, "head_dim": hidden_dim // 32, "seq_len": sequence_length},
                op_type="attention",
                is_fusible=False,
                stage_id=i + 1
            )
            self.add_edge(prev_node_id, layer_attn_id, tensor_shape=[sequence_length, hidden_dim], data_transfer_mb=64.0)

            self.add_node(
                layer_mlp_id,
                f"Layer {i} MLP / SwiGLU FeedForward",
                "MODULE",
                execution_time_ms=12.0,
                memory_footprint_mb=800.0,
                attributes={"hidden_dim": hidden_dim * 4},
                op_type="activation",
                is_fusible=True,
                stage_id=i + 1
            )
            self.add_edge(layer_attn_id, layer_mlp_id, tensor_shape=[sequence_length, hidden_dim], data_transfer_mb=64.0)

            self.add_node(
                layer_norm_id,
                f"Layer {i} RMSNorm & Residual",
                "OPERATOR",
                execution_time_ms=1.5,
                memory_footprint_mb=64.0,
                op_type="norm",
                is_fusible=True,
                stage_id=i + 1
            )
            self.add_edge(layer_mlp_id, layer_norm_id, tensor_shape=[sequence_length, hidden_dim], data_transfer_mb=64.0)
            prev_node_id = layer_norm_id

        self.add_node(
            "head",
            "LM Head Logits Projection",
            "OPERATOR",
            execution_time_ms=5.0,
            memory_footprint_mb=512.0,
            op_type="gemm",
            is_fusible=False,
            stage_id=num_layers + 1
        )
        self.add_edge(prev_node_id, "head", tensor_shape=[sequence_length, 32000], data_transfer_mb=64.0)

        return self

    def build_moe_dag(
        self,
        num_layers: int = 8,
        hidden_dim: int = 4096,
        num_experts: int = 8,
        top_k: int = 2,
        sequence_length: int = 2048
    ) -> "ExecutionGraphBuilder":
        """Builds a Mixture-of-Experts (MoE) computational DAG with expert routing branches."""
        prev_node_id = "embedding"
        self.add_node("embedding", "Token Embedding", "MODULE", execution_time_ms=2.5, memory_footprint_mb=256.0, op_type="gemm")

        for i in range(num_layers):
            attn_id = f"layer_{i}_attention"
            router_id = f"layer_{i}_moe_router"
            combine_id = f"layer_{i}_moe_combine"
            norm_id = f"layer_{i}_norm"

            self.add_node(attn_id, f"Layer {i} Self-Attention", "ATTENTION", execution_time_ms=16.0, memory_footprint_mb=1024.0, op_type="attention")
            self.add_edge(prev_node_id, attn_id, data_transfer_mb=64.0)

            self.add_node(router_id, f"Layer {i} MoE Router (Top-{top_k}/{num_experts})", "OPERATOR", execution_time_ms=1.2, memory_footprint_mb=32.0, op_type="router", is_fusible=True)
            self.add_edge(attn_id, router_id, data_transfer_mb=64.0)

            for e in range(min(num_experts, 4)):
                exp_id = f"layer_{i}_expert_{e}"
                self.add_node(exp_id, f"Layer {i} Expert {e} SwiGLU", "MODULE", execution_time_ms=6.5, memory_footprint_mb=320.0, op_type="activation", is_fusible=True)
                self.add_edge(router_id, exp_id, data_transfer_mb=16.0)
                self.add_edge(exp_id, combine_id, data_transfer_mb=16.0)

            self.add_node(combine_id, f"Layer {i} MoE Gated Weight Combine", "OPERATOR", execution_time_ms=2.0, memory_footprint_mb=64.0, op_type="elementwise", is_fusible=True)
            self.add_node(norm_id, f"Layer {i} RMSNorm", "OPERATOR", execution_time_ms=1.2, memory_footprint_mb=32.0, op_type="norm", is_fusible=True)

            self.add_edge(combine_id, norm_id, data_transfer_mb=64.0)
            prev_node_id = norm_id

        self.add_node("head", "LM Head Projection", "OPERATOR", execution_time_ms=4.5, memory_footprint_mb=512.0, op_type="gemm")
        self.add_edge(prev_node_id, "head", data_transfer_mb=64.0)
        return self

    def build_ssm_dag(
        self,
        num_layers: int = 8,
        d_model: int = 2560,
        d_state: int = 16,
        sequence_length: int = 4096
    ) -> "ExecutionGraphBuilder":
        """Builds a Mamba / State-Space Model (SSM) computational DAG."""
        prev_node_id = "embedding"
        self.add_node("embedding", "Mamba Token Embedding", "MODULE", execution_time_ms=2.0, memory_footprint_mb=128.0, op_type="gemm")

        for i in range(num_layers):
            in_proj_id = f"layer_{i}_in_proj"
            conv_id = f"layer_{i}_conv1d"
            ssm_scan_id = f"layer_{i}_selective_scan"
            out_proj_id = f"layer_{i}_out_proj"
            norm_id = f"layer_{i}_norm"

            self.add_node(in_proj_id, f"Layer {i} Input Linear Expansion", "OPERATOR", execution_time_ms=4.0, memory_footprint_mb=256.0, op_type="gemm")
            self.add_edge(prev_node_id, in_proj_id, data_transfer_mb=32.0)

            self.add_node(conv_id, f"Layer {i} Causal Conv1D (d_state={d_state})", "OPERATOR", execution_time_ms=1.8, memory_footprint_mb=64.0, op_type="elementwise", is_fusible=True)
            self.add_edge(in_proj_id, conv_id, data_transfer_mb=32.0)

            self.add_node(ssm_scan_id, f"Layer {i} Selective SSM Scan Kernel", "OPERATOR", execution_time_ms=8.5, memory_footprint_mb=512.0, op_type="ssm", is_fusible=True)
            self.add_edge(conv_id, ssm_scan_id, data_transfer_mb=32.0)

            self.add_node(out_proj_id, f"Layer {i} Output Linear Contraction", "OPERATOR", execution_time_ms=4.0, memory_footprint_mb=256.0, op_type="gemm")
            self.add_edge(ssm_scan_id, out_proj_id, data_transfer_mb=32.0)

            self.add_node(norm_id, f"Layer {i} RMSNorm & Residual", "OPERATOR", execution_time_ms=1.0, memory_footprint_mb=32.0, op_type="norm", is_fusible=True)
            self.add_edge(out_proj_id, norm_id, data_transfer_mb=32.0)
            prev_node_id = norm_id

        self.add_node("head", "SSM LM Head Projection", "OPERATOR", execution_time_ms=3.5, memory_footprint_mb=256.0, op_type="gemm")
        self.add_edge(prev_node_id, "head", data_transfer_mb=32.0)
        return self

    def build_cnn_dag(
        self,
        num_layers: int = 10,
        channels: int = 32
    ) -> "ExecutionGraphBuilder":
        """Builds a CNN computational DAG."""
        prev_node_id = "input"
        self.add_node("input", "Image Input", "MODULE", execution_time_ms=1.0, memory_footprint_mb=16.0, op_type="elementwise")
        
        for i in range(num_layers // 2):
            conv_id = f"layer_{i}_conv"
            relu_id = f"layer_{i}_relu"
            pool_id = f"layer_{i}_pool"

            self.add_node(conv_id, f"Layer {i} Conv2D", "OPERATOR", execution_time_ms=5.0, memory_footprint_mb=64.0, op_type="gemm")
            self.add_edge(prev_node_id, conv_id, data_transfer_mb=16.0)

            self.add_node(relu_id, f"Layer {i} ReLU", "OPERATOR", execution_time_ms=0.5, memory_footprint_mb=16.0, op_type="activation", is_fusible=True)
            self.add_edge(conv_id, relu_id, data_transfer_mb=16.0)

            self.add_node(pool_id, f"Layer {i} MaxPool", "OPERATOR", execution_time_ms=1.0, memory_footprint_mb=8.0, op_type="elementwise")
            self.add_edge(relu_id, pool_id, data_transfer_mb=16.0)
            
            prev_node_id = pool_id

        self.add_node("head", "Linear Projection", "OPERATOR", execution_time_ms=2.0, memory_footprint_mb=32.0, op_type="gemm")
        self.add_edge(prev_node_id, "head", data_transfer_mb=8.0)
        return self

    def topological_sort(self) -> List[str]:
        """Returns node IDs in topologically sorted execution order."""
        in_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        sorted_nodes = []
        sorted_set = set()

        while queue:
            curr = queue.popleft()
            sorted_nodes.append(curr)
            sorted_set.add(curr)
            
            for edge in self.edges:
                if edge.source_id == curr:
                    in_degree[edge.target_id] -= 1
                    if in_degree[edge.target_id] == 0:
                        queue.append(edge.target_id)

        # Fallback if unvisited nodes remain (e.g. cycle safety)
        if len(sorted_nodes) < len(self.nodes):
            for nid in self.nodes:
                if nid not in sorted_set:
                    sorted_nodes.append(nid)

        return sorted_nodes

    def get_in_degree(self) -> Dict[str, int]:
        return {nid: len(self._predecessors.get(nid, [])) for nid in self.nodes}

    def get_out_degree(self) -> Dict[str, int]:
        return {nid: len(self._successors.get(nid, [])) for nid in self.nodes}

    def get_predecessors(self, node_id: str) -> List[str]:
        return self._predecessors.get(node_id, [])

    def get_successors(self, node_id: str) -> List[str]:
        return self._successors.get(node_id, [])

    def validate_dag(self) -> bool:
        """Verifies that the graph is a valid, acyclic Directed Graph."""
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for succ in self.get_successors(node_id):
                if succ not in visited:
                    if dfs(succ):
                        return True
                elif succ in rec_stack:
                    return True
            rec_stack.remove(node_id)
            return False

        for nid in self.nodes:
            if nid not in visited:
                if dfs(nid):
                    return False  # Contains cycle
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_mermaid(self) -> str:
        """Generates a GitHub-compatible Mermaid flowchart definition."""
        lines = ["flowchart TD", f"    %% Graphify DAG for {self.model_name}"]
        for node in self.nodes.values():
            style_class = ":::critical" if node.is_critical_path else (":::fusible" if node.is_fusible else "")
            lines.append(f'    {node.node_id}["{node.label}<br/>{node.execution_time_ms:.1f}ms | {node.memory_footprint_mb:.0f}MB"]{style_class}')

        for edge in self.edges:
            label = f"{edge.data_transfer_mb:.0f}MB" if edge.data_transfer_mb > 0 else ""
            if label:
                lines.append(f'    {edge.source_id} -->|"{label}"| {edge.target_id}')
            else:
                lines.append(f'    {edge.source_id} --> {edge.target_id}')

        lines.append("    classDef critical fill:#ff4d4d,stroke:#990000,stroke-width:2px,color:#fff;")
        lines.append("    classDef fusible fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff;")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Generates Graphviz DOT representation."""
        lines = [f'digraph "{self.model_name}" {{', '    rankdir=TB;', '    node [shape=box, style=filled, fontname="Helvetica"];']
        for node in self.nodes.values():
            color = "#ffcccc" if node.is_critical_path else ("#ccffcc" if node.is_fusible else "#f0f0f0")
            lines.append(f'    "{node.node_id}" [label="{node.label}\\n{node.execution_time_ms:.1f}ms, {node.memory_footprint_mb:.0f}MB", fillcolor="{color}"];')
        for edge in self.edges:
            lines.append(f'    "{edge.source_id}" -> "{edge.target_id}" [label="{edge.data_transfer_mb:.0f}MB"];')
        lines.append("}")
        return "\n".join(lines)

    def to_html_interactive(self, title: str = "Quant Ghost Layer - Graphify DAG Topology") -> str:
        """Generates a standalone, dark-mode glassmorphism interactive HTML graph visualization."""
        mermaid_code = self.to_mermaid()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title} - {self.model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px 30px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        .header h1 {{ margin: 0 0 8px 0; color: #38bdf8; font-size: 24px; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 14px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .stat-card {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
        }}
        .stat-card .val {{ font-size: 26px; font-weight: bold; color: #38bdf8; }}
        .stat-card .lbl {{ font-size: 12px; color: #64748b; text-transform: uppercase; margin-top: 4px; }}
        .graph-box {{
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 16px;
            padding: 24px;
            overflow-x: auto;
        }}
        pre.mermaid {{ background: transparent !important; }}
    </style>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
</head>
<body>
    <div class="header">
        <h1>Graphify Execution DAG Topology: {self.model_name}</h1>
        <p>Quant Ghost Layer Automated Computational Bottleneck & Kernel Fusion Profiler</p>
    </div>

    <div class="card-grid">
        <div class="stat-card">
            <div class="val">{len(self.nodes)}</div>
            <div class="lbl">Total Execution Nodes</div>
        </div>
        <div class="stat-card">
            <div class="val">{len(self.edges)}</div>
            <div class="lbl">Tensor Transfer Edges</div>
        </div>
        <div class="stat-card">
            <div class="val">{sum(n.execution_time_ms for n in self.nodes.values()):.1f} ms</div>
            <div class="lbl">Total Step Latency</div>
        </div>
        <div class="stat-card">
            <div class="val">{sum(1 for n in self.nodes.values() if n.is_fusible)}</div>
            <div class="lbl">Fusible Operators</div>
        </div>
    </div>

    <div class="graph-box">
        <pre class="mermaid">
{mermaid_code}
        </pre>
    </div>
</body>
</html>
"""
        return html
