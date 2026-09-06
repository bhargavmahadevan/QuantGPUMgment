import React, { useState, useMemo, useRef } from "react";
import { 
  Cpu, 
  Zap, 
  ArrowRight, 
  CheckCircle2
} from "lucide-react";

interface DAGNode {
  id: string;
  name: string;
  category: "embedding" | "attention" | "mlp" | "norm" | "loss" | "communication";
  latencyMs: number;
  memoryMb: number;
  isFusible: boolean;
  fusionGroup?: string;
  gradientNorm?: number;
  inDegree: number;
  outDegree: number;
  layerIndex?: number;
}

interface DAGEdge {
  from: string;
  to: string;
  tensorShape: string;
  bytesTransferred: number;
  isCriticalPath: boolean;
}

const DEFAULT_NODES: DAGNode[] = [
  { id: "emb", name: "Token & Position Embeddings", category: "embedding", latencyMs: 2.4, memoryMb: 256, isFusible: false, inDegree: 0, outDegree: 1 },
  { id: "l0_norm1", name: "L0 Input RMSNorm", category: "norm", latencyMs: 0.8, memoryMb: 32, isFusible: true, fusionGroup: "F0_L0_AttnPrep", layerIndex: 0, inDegree: 1, outDegree: 1 },
  { id: "l0_qkv", name: "L0 QKV Projection + RoPE", category: "attention", latencyMs: 8.2, memoryMb: 640, isFusible: true, fusionGroup: "F0_L0_AttnPrep", layerIndex: 0, inDegree: 1, outDegree: 1 },
  { id: "l0_attn", name: "L0 FlashAttention-2 Kernel", category: "attention", latencyMs: 12.4, memoryMb: 1120, isFusible: false, layerIndex: 0, inDegree: 1, outDegree: 1 },
  { id: "l0_proj", name: "L0 O-Proj + Residual Add", category: "attention", latencyMs: 4.1, memoryMb: 320, isFusible: true, fusionGroup: "F1_L0_ResNorm", layerIndex: 0, inDegree: 1, outDegree: 1 },
  { id: "l0_norm2", name: "L0 Post-Attn RMSNorm", category: "norm", latencyMs: 0.8, memoryMb: 32, isFusible: true, fusionGroup: "F1_L0_ResNorm", layerIndex: 0, inDegree: 1, outDegree: 1 },
  { id: "l0_mlp_gate", name: "L0 SwiGLU Gate & Up-Proj", category: "mlp", latencyMs: 9.6, memoryMb: 840, isFusible: true, fusionGroup: "F2_L0_SwiGLU", layerIndex: 0, inDegree: 1, outDegree: 1 },
  { id: "l0_mlp_down", name: "L0 Down-Proj + Residual Add", category: "mlp", latencyMs: 6.2, memoryMb: 480, isFusible: false, layerIndex: 0, inDegree: 1, outDegree: 1 },
  
  { id: "l1_norm1", name: "L1 Input RMSNorm", category: "norm", latencyMs: 0.8, memoryMb: 32, isFusible: true, fusionGroup: "F3_L1_AttnPrep", layerIndex: 1, inDegree: 1, outDegree: 1 },
  { id: "l1_qkv", name: "L1 QKV Projection + RoPE", category: "attention", latencyMs: 8.3, memoryMb: 640, isFusible: true, fusionGroup: "F3_L1_AttnPrep", layerIndex: 1, inDegree: 1, outDegree: 1 },
  { id: "l1_attn", name: "L1 FlashAttention-2 Kernel", category: "attention", latencyMs: 12.5, memoryMb: 1120, isFusible: false, layerIndex: 1, inDegree: 1, outDegree: 1 },
  { id: "l1_proj", name: "L1 O-Proj + Residual Add", category: "attention", latencyMs: 4.0, memoryMb: 320, isFusible: true, fusionGroup: "F4_L1_ResNorm", layerIndex: 1, inDegree: 1, outDegree: 1 },
  { id: "l1_norm2", name: "L1 Post-Attn RMSNorm", category: "norm", latencyMs: 0.8, memoryMb: 32, isFusible: true, fusionGroup: "F4_L1_ResNorm", layerIndex: 1, inDegree: 1, outDegree: 1 },
  { id: "l1_mlp_gate", name: "L1 SwiGLU Gate & Up-Proj", category: "mlp", latencyMs: 9.7, memoryMb: 840, isFusible: true, fusionGroup: "F5_L1_SwiGLU", layerIndex: 1, inDegree: 1, outDegree: 1 },
  { id: "l1_mlp_down", name: "L1 Down-Proj + Residual Add", category: "mlp", latencyMs: 6.3, memoryMb: 480, isFusible: false, layerIndex: 1, inDegree: 1, outDegree: 1 },

  { id: "final_norm", name: "Final LayerNorm / RMSNorm", category: "norm", latencyMs: 0.9, memoryMb: 48, isFusible: false, inDegree: 1, outDegree: 1 },
  { id: "lm_head", name: "LM Vocabulary Head Projection", category: "mlp", latencyMs: 11.2, memoryMb: 1280, isFusible: false, inDegree: 1, outDegree: 1 },
  { id: "loss_fn", name: "Cross-Entropy Loss & Reduction", category: "loss", latencyMs: 3.5, memoryMb: 128, isFusible: false, inDegree: 1, outDegree: 0 }
];

const DEFAULT_EDGES: DAGEdge[] = [
  { from: "emb", to: "l0_norm1", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l0_norm1", to: "l0_qkv", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l0_qkv", to: "l0_attn", tensorShape: "[4, 32, 4096, 128]", bytesTransferred: 268435456, isCriticalPath: true },
  { from: "l0_attn", to: "l0_proj", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l0_proj", to: "l0_norm2", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l0_norm2", to: "l0_mlp_gate", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l0_mlp_gate", to: "l0_mlp_down", tensorShape: "[4, 4096, 14336]", bytesTransferred: 469762048, isCriticalPath: true },
  { from: "l0_mlp_down", to: "l1_norm1", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },

  { from: "l1_norm1", to: "l1_qkv", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l1_qkv", to: "l1_attn", tensorShape: "[4, 32, 4096, 128]", bytesTransferred: 268435456, isCriticalPath: true },
  { from: "l1_attn", to: "l1_proj", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l1_proj", to: "l1_norm2", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l1_norm2", to: "l1_mlp_gate", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "l1_mlp_gate", to: "l1_mlp_down", tensorShape: "[4, 4096, 14336]", bytesTransferred: 469762048, isCriticalPath: true },
  { from: "l1_mlp_down", to: "final_norm", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },

  { from: "final_norm", to: "lm_head", tensorShape: "[4, 4096, 4096]", bytesTransferred: 134217728, isCriticalPath: true },
  { from: "lm_head", to: "loss_fn", tensorShape: "[4, 4096, 32000]", bytesTransferred: 1048576000, isCriticalPath: true },
];

export function ExecutionDAGViewer() {
  const [selectedNode, setSelectedNode] = useState<DAGNode | null>(DEFAULT_NODES[3]);
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [highlightFusible, setHighlightFusible] = useState<boolean>(true);
  const canvasRef = useRef<HTMLDivElement>(null);

  const totalLatency = useMemo(() => {
    return DEFAULT_NODES.reduce((acc, node) => acc + node.latencyMs, 0);
  }, []);

  const totalFusibleSavings = useMemo(() => {
    const fusibleNodes = DEFAULT_NODES.filter(n => n.isFusible);
    const fusibleLatency = fusibleNodes.reduce((acc, n) => acc + n.latencyMs, 0);
    return (fusibleLatency * 0.28).toFixed(1);
  }, []);

  const filteredNodes = useMemo(() => {
    if (filterCategory === "all") return DEFAULT_NODES;
    return DEFAULT_NODES.filter(n => n.category === filterCategory);
  }, [filterCategory]);

  const categories = [
    { id: "all", label: "All Layers" },
    { id: "attention", label: "Attention & RoPE" },
    { id: "mlp", label: "SwiGLU & FeedForward" },
    { id: "norm", label: "RMSNorm & Residuals" },
    { id: "embedding", label: "Embeddings & Heads" },
  ];

  return (
    <div className="dag-container">
      {/* Top Controls Bar */}
      <div className="dag-controls">
        <div className="dag-controls__left">
          <div className="dag-badge">
            <Cpu size={14} className="text-teal-400" />
            <span>PYTORCH EXECUTION DAG & KERNEL PROFILER</span>
          </div>
          <span className="dag-stat">Step Latency: <strong>{totalLatency.toFixed(1)} ms</strong></span>
          <span className="dag-stat">Fusible Potential: <strong>-{totalFusibleSavings} ms (-18.4%)</strong></span>
        </div>

        <div className="dag-controls__right">
          <div className="category-pills">
            {categories.map(c => (
              <button
                key={c.id}
                className={`pill-btn ${filterCategory === c.id ? "is-active" : ""}`}
                onClick={() => setFilterCategory(c.id)}
              >
                {c.label}
              </button>
            ))}
          </div>

          <button 
            className={`toggle-btn ${highlightFusible ? "is-active" : ""}`}
            onClick={() => setHighlightFusible(!highlightFusible)}
            title="Highlight fusible operator clusters"
          >
            <Zap size={14} />
            <span>Kernel Fusion</span>
          </button>
        </div>
      </div>

      {/* Main Canvas & Inspection Layout */}
      <div className="dag-workspace">
        {/* Node & Edge Flow Graph */}
        <div className="dag-graph-view" ref={canvasRef}>
          <div className="dag-nodes-flow">
            {filteredNodes.map((node, index) => {
              const isSelected = selectedNode?.id === node.id;
              const hasFusionGroup = highlightFusible && node.isFusible;

              return (
                <React.Fragment key={node.id}>
                  <div 
                    className={`dag-node-card dag-node-card--${node.category} ${isSelected ? "is-selected" : ""} ${hasFusionGroup ? "has-fusion" : ""}`}
                    onClick={() => setSelectedNode(node)}
                  >
                    <div className="dag-node-card__header">
                      <span className="dag-node-id">0{index + 1}</span>
                      <span className={`dag-node-tag tag--${node.category}`}>{node.category.toUpperCase()}</span>
                      {node.isFusible && highlightFusible && (
                        <span className="fusion-badge" title={`Fusion Candidate: ${node.fusionGroup}`}>
                          <Zap size={10} /> FUSIBLE
                        </span>
                      )}
                    </div>

                    <div className="dag-node-card__title">
                      {node.name}
                    </div>

                    <div className="dag-node-card__metrics">
                      <div>
                        <small>LATENCY</small>
                        <strong>{node.latencyMs} ms</strong>
                      </div>
                      <div>
                        <small>VRAM</small>
                        <strong>{node.memoryMb} MB</strong>
                      </div>
                      <div>
                        <small>STEP SHARE</small>
                        <strong>{((node.latencyMs / totalLatency) * 100).toFixed(1)}%</strong>
                      </div>
                    </div>

                    {/* Progress Bar for Step Share */}
                    <div className="dag-node-bar">
                      <div 
                        className="dag-node-bar__fill" 
                        style={{ width: `${(node.latencyMs / 15) * 100}%` }}
                      />
                    </div>
                  </div>

                  {index < filteredNodes.length - 1 && (
                    <div className="dag-edge-connector">
                      <div className="edge-line" />
                      <div className="edge-pulse" />
                      <span className="edge-label">
                        {DEFAULT_EDGES[index]?.tensorShape || "[4, 4096, 4096]"}
                      </span>
                      <ArrowRight size={14} className="edge-arrow" />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Node Inspector Detail Panel */}
        {selectedNode && (
          <div className="dag-inspector">
            <div className="dag-inspector__header">
              <div className="inspector-title-group">
                <span className="inspector-eyebrow">NODE PROFILER INSPECTOR</span>
                <h3>{selectedNode.name}</h3>
              </div>
              <span className={`category-tag tag--${selectedNode.category}`}>
                {selectedNode.category}
              </span>
            </div>

            <div className="dag-inspector__metrics-grid">
              <div className="metric-box">
                <span>Kernel Latency</span>
                <strong>{selectedNode.latencyMs} ms</strong>
                <small className="text-emerald-400">Baseline runtime</small>
              </div>
              <div className="metric-box">
                <span>Activation Memory</span>
                <strong>{selectedNode.memoryMb} MB</strong>
                <small className="text-cyan-400">Peak buffer</small>
              </div>
              <div className="metric-box">
                <span>Fusibility Status</span>
                <strong>{selectedNode.isFusible ? "Eligible" : "Isolated"}</strong>
                <small className={selectedNode.isFusible ? "text-amber-400" : "text-gray-400"}>
                  {selectedNode.fusionGroup || "Stand-alone kernel"}
                </small>
              </div>
              <div className="metric-box">
                <span>Critical Path</span>
                <strong>Forward Pass</strong>
                <small className="text-emerald-400">Active gradient sync</small>
              </div>
            </div>

            <div className="dag-inspector__section">
              <h4>Optimization & Fusion Recommendation</h4>
              <div className="recommendation-card">
                {selectedNode.isFusible ? (
                  <>
                    <div className="recommendation-card__title text-amber-300">
                      <Zap size={14} /> JIT Tracing / Triton Kernel Fusion Recommended
                    </div>
                    <p>
                      This operator ({selectedNode.name}) can be fused into the <code>{selectedNode.fusionGroup}</code> composite kernel. 
                      Fusing reduces intermediate global memory roundtrips between DRAM and SRAM by ~{Math.round(selectedNode.memoryMb * 0.7)} MB per step.
                    </p>
                    <div className="recommendation-card__footer">
                      <span>Expected Speedup: <strong>+{(selectedNode.latencyMs * 0.32).toFixed(2)} ms</strong></span>
                      <span>Loss-Shift Risk: <strong>&lt; 0.0001 (Safe)</strong></span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="recommendation-card__title text-teal-300">
                      <CheckCircle2 size={14} /> High-Throughput Dedicated Kernel
                    </div>
                    <p>
                      Operating at peak compute intensity. FlashAttention-2 / GEMM tensor cores saturated at &gt;88% utilization. No kernel fusion required.
                    </p>
                    <div className="recommendation-card__footer">
                      <span>Tensor Core Occupancy: <strong>91.4%</strong></span>
                      <span>Bandwidth: <strong>812 GB/s</strong></span>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="dag-inspector__section">
              <h4>Tensor Dependency Inputs & Outputs</h4>
              <div className="dependency-list">
                <div className="dependency-item">
                  <span className="dep-type">INPUT TENSOR</span>
                  <code>x: [batch=4, seq_len=4096, hidden_dim=4096] (bfloat16)</code>
                </div>
                <div className="dependency-item">
                  <span className="dep-type">OUTPUT TENSOR</span>
                  <code>y: [batch=4, seq_len=4096, hidden_dim=4096] (bfloat16)</code>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
export default ExecutionDAGViewer;
