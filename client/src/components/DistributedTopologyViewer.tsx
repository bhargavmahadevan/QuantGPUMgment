import React, { useState, useMemo } from "react";
import { 
  Network, 
  Server, 
  Cpu, 
  Radio, 
  Activity, 
  Zap, 
  ShieldCheck, 
  Layers, 
  ArrowRightLeft,
  Gauge
} from "lucide-react";

interface GPURank {
  rank: number;
  nodeId: number;
  gpuModel: string;
  vramAllocatedGb: number;
  vramTotalGb: number;
  computeUtilizationPct: number;
  tpGroup: number;
  ppStage: number;
  dpRank: number;
  activeOp?: string;
  bandwidthGbps: number;
}

interface CollectiveMetric {
  name: string;
  type: "all_reduce" | "all_gather" | "reduce_scatter" | "p2p_send_recv";
  avgDurationMs: number;
  sizeMb: number;
  overlapPct: number;
  bottleneckScore: "LOW" | "MODERATE" | "HIGH";
}

const DEFAULT_RANKS: GPURank[] = [
  // Node 0 (GPUs 0-3)
  { rank: 0, nodeId: 0, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 64.2, vramTotalGb: 80.0, computeUtilizationPct: 94.2, tpGroup: 0, ppStage: 0, dpRank: 0, bandwidthGbps: 884.2, activeOp: "AllReduce Ring Sync" },
  { rank: 1, nodeId: 0, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 64.1, vramTotalGb: 80.0, computeUtilizationPct: 93.8, tpGroup: 0, ppStage: 0, dpRank: 1, bandwidthGbps: 882.5, activeOp: "AllReduce Ring Sync" },
  { rank: 2, nodeId: 0, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 63.9, vramTotalGb: 80.0, computeUtilizationPct: 94.5, tpGroup: 1, ppStage: 0, dpRank: 2, bandwidthGbps: 886.1, activeOp: "AllReduce Ring Sync" },
  { rank: 3, nodeId: 0, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 64.4, vramTotalGb: 80.0, computeUtilizationPct: 94.0, tpGroup: 1, ppStage: 0, dpRank: 3, bandwidthGbps: 883.0, activeOp: "AllReduce Ring Sync" },
  
  // Node 1 (GPUs 4-7)
  { rank: 4, nodeId: 1, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 62.8, vramTotalGb: 80.0, computeUtilizationPct: 91.2, tpGroup: 0, ppStage: 1, dpRank: 0, bandwidthGbps: 395.4, activeOp: "InfiniBand Cross-Node P2P" },
  { rank: 5, nodeId: 1, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 62.9, vramTotalGb: 80.0, computeUtilizationPct: 91.5, tpGroup: 0, ppStage: 1, dpRank: 1, bandwidthGbps: 394.8, activeOp: "InfiniBand Cross-Node P2P" },
  { rank: 6, nodeId: 1, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 63.1, vramTotalGb: 80.0, computeUtilizationPct: 92.0, tpGroup: 1, ppStage: 1, dpRank: 2, bandwidthGbps: 396.0, activeOp: "InfiniBand Cross-Node P2P" },
  { rank: 7, nodeId: 1, gpuModel: "NVIDIA H100 SXM5 80GB", vramAllocatedGb: 63.0, vramTotalGb: 80.0, computeUtilizationPct: 91.8, tpGroup: 1, ppStage: 1, dpRank: 3, bandwidthGbps: 395.1, activeOp: "InfiniBand Cross-Node P2P" }
];

const COLLECTIVES_DATA: CollectiveMetric[] = [
  { name: "Gradient AllReduce (TP Ring)", type: "all_reduce", avgDurationMs: 3.12, sizeMb: 512, overlapPct: 82.4, bottleneckScore: "LOW" },
  { name: "FSDP Sharded Parameter AllGather", type: "all_gather", avgDurationMs: 4.85, sizeMb: 1024, overlapPct: 74.1, bottleneckScore: "MODERATE" },
  { name: "Backward Gradient ReduceScatter", type: "reduce_scatter", avgDurationMs: 2.94, sizeMb: 512, overlapPct: 88.6, bottleneckScore: "LOW" },
  { name: "1F1B Pipeline Activation Hand-off", type: "p2p_send_recv", avgDurationMs: 1.45, sizeMb: 256, overlapPct: 95.2, bottleneckScore: "LOW" }
];

export function DistributedTopologyViewer() {
  const [selectedRank, setSelectedRank] = useState<GPURank>(DEFAULT_RANKS[0]);
  const [parallelismMode, setParallelismMode] = useState<"3D" | "FSDP" | "DDP">("3D");
  const [activeTab, setActiveTab] = useState<"topology" | "nccl">("topology");

  const clusterStats = useMemo(() => {
    const totalVramAlloc = DEFAULT_RANKS.reduce((acc, r) => acc + r.vramAllocatedGb, 0);
    const avgUtil = DEFAULT_RANKS.reduce((acc, r) => acc + r.computeUtilizationPct, 0) / DEFAULT_RANKS.length;
    const avgOverlap = COLLECTIVES_DATA.reduce((acc, c) => acc + c.overlapPct, 0) / COLLECTIVES_DATA.length;
    return {
      nodes: 2,
      totalGpus: DEFAULT_RANKS.length,
      totalVramAlloc: totalVramAlloc.toFixed(1),
      avgUtil: avgUtil.toFixed(1),
      avgOverlap: avgOverlap.toFixed(1)
    };
  }, []);

  return (
    <div className="dist-topology-container">
      {/* Top Header Controls */}
      <div className="dist-header">
        <div className="dist-header__left">
          <div className="dist-badge">
            <Network size={14} className="text-cyan-400" />
            <span>NCCL DISTRIBUTED TOPOLOGY & COLLECTIVE ANALYZER</span>
          </div>
          <span className="dist-stat">Cluster Scale: <strong>{clusterStats.nodes} Nodes / {clusterStats.totalGpus} GPUs</strong></span>
          <span className="dist-stat">Comm/Compute Overlap: <strong className="text-emerald-400">{clusterStats.avgOverlap}%</strong></span>
        </div>

        <div className="dist-header__right">
          <div className="parallel-selector">
            <span className="label-dim">Strategy:</span>
            {(["3D", "FSDP", "DDP"] as const).map(mode => (
              <button 
                key={mode}
                className={`parallel-btn ${parallelismMode === mode ? "is-active" : ""}`}
                onClick={() => setParallelismMode(mode)}
              >
                {mode === "3D" ? "3D Hybrid (TP=2, PP=2, DP=2)" : mode === "FSDP" ? "PyTorch FSDP (ZeRO-3)" : "Standard DDP"}
              </button>
            ))}
          </div>

          <div className="view-toggle-group">
            <button 
              className={`tab-btn ${activeTab === "topology" ? "is-active" : ""}`}
              onClick={() => setActiveTab("topology")}
            >
              <Server size={13} /> Interconnects
            </button>
            <button 
              className={`tab-btn ${activeTab === "nccl" ? "is-active" : ""}`}
              onClick={() => setActiveTab("nccl")}
            >
              <Radio size={13} /> NCCL Operations
            </button>
          </div>
        </div>
      </div>

      {/* Main Multi-Node Visualization Layout */}
      <div className="dist-content-grid">
        {/* Nodes and GPUs Grid */}
        <div className="dist-nodes-stage">
          {activeTab === "topology" ? (
            <div className="nodes-wrapper">
              {[0, 1].map(nodeId => {
                const nodeRanks = DEFAULT_RANKS.filter(r => r.nodeId === nodeId);
                const isNode0 = nodeId === 0;

                return (
                  <div key={nodeId} className="node-enclosure">
                    <div className="node-enclosure__header">
                      <div className="node-title">
                        <Server size={16} className="text-teal-400" />
                        <span>COMPUTE NODE {nodeId} (H100-NODE-{nodeId}.cluster.local)</span>
                      </div>
                      <span className="node-interconnect-tag">
                        {isNode0 ? "NVLink 900 GB/s Mesh (Intra-Node)" : "InfiniBand NDR 400 Gbps (Cross-Node Link)"}
                      </span>
                    </div>

                    <div className="node-gpus-grid">
                      {nodeRanks.map(gpu => {
                        const isSelected = selectedRank.rank === gpu.rank;

                        return (
                          <div 
                            key={gpu.rank}
                            className={`gpu-card ${isSelected ? "is-selected" : ""}`}
                            onClick={() => setSelectedRank(gpu)}
                          >
                            <div className="gpu-card__top">
                              <div className="rank-indicator">
                                <Cpu size={14} />
                                <strong>RANK {gpu.rank}</strong>
                              </div>
                              <span className="gpu-tp-pill">TP_{gpu.tpGroup} | PP_{gpu.ppStage}</span>
                            </div>

                            <div className="gpu-card__body">
                              <div className="gpu-metric-row">
                                <small>SM UTILIZATION</small>
                                <strong>{gpu.computeUtilizationPct}%</strong>
                              </div>
                              <div className="gpu-progress-track">
                                <div 
                                  className="gpu-progress-fill" 
                                  style={{ width: `${gpu.computeUtilizationPct}%` }}
                                />
                              </div>

                              <div className="gpu-metric-row mt-2">
                                <small>VRAM ALLOCATION</small>
                                <strong>{gpu.vramAllocatedGb} / {gpu.vramTotalGb} GB</strong>
                              </div>
                              <div className="gpu-progress-track">
                                <div 
                                  className="gpu-progress-fill gpu-progress-fill--cyan" 
                                  style={{ width: `${(gpu.vramAllocatedGb / gpu.vramTotalGb) * 100}%` }}
                                />
                              </div>
                            </div>

                            <div className="gpu-card__footer">
                              <span className="active-op-tag">
                                <Activity size={10} className="animate-pulse text-emerald-400" />
                                {gpu.activeOp}
                              </span>
                              <span className="bandwidth-tag">{gpu.bandwidthGbps} GB/s</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}

              {/* Inter-Node Interconnect Visual Banner */}
              <div className="interconnect-banner">
                <ArrowRightLeft size={16} className="text-cyan-400" />
                <span>CROSS-NODE INTERCONNECT: 8x NDR InfiniBand Links (3.2 Tbps Aggregate Fabric) · Zero Packet Drops</span>
              </div>
            </div>
          ) : (
            <div className="nccl-table-container">
              <div className="nccl-header-row">
                <span>COLLECTIVE OPERATION</span>
                <span>TYPE</span>
                <span>MESSAGE SIZE</span>
                <span>AVG LATENCY</span>
                <span>COMPUTE OVERLAP</span>
                <span>BOTTLENECK RATING</span>
              </div>

              {COLLECTIVES_DATA.map((op, idx) => (
                <div key={idx} className="nccl-data-row">
                  <div className="op-name-cell">
                    <Radio size={13} className="text-teal-400" />
                    <strong>{op.name}</strong>
                  </div>
                  <span className="op-type-badge">{op.type.toUpperCase()}</span>
                  <span className="op-size-cell">{op.sizeMb} MB</span>
                  <span className="op-latency-cell">{op.avgDurationMs} ms</span>
                  <div className="op-overlap-cell">
                    <span>{op.overlapPct}%</span>
                    <div className="overlap-bar">
                      <div className="overlap-bar__fill" style={{ width: `${op.overlapPct}%` }} />
                    </div>
                  </div>
                  <span className={`bottleneck-badge badge--${op.bottleneckScore.toLowerCase()}`}>
                    {op.bottleneckScore}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Selected Rank Inspector */}
        <div className="dist-inspector">
          <div className="dist-inspector__header">
            <span className="inspector-eyebrow">DISTRIBUTED WORKER PROFILE</span>
            <h3>RANK {selectedRank.rank} (GPU {selectedRank.rank % 4} on Node {selectedRank.nodeId})</h3>
            <p className="text-xs text-gray-400">{selectedRank.gpuModel}</p>
          </div>

          <div className="dist-inspector__stats">
            <div className="stat-card">
              <span>Tensor Parallel Group</span>
              <strong>Group {selectedRank.tpGroup} (Size=2)</strong>
              <small className="text-emerald-400">Intra-Node NVLink</small>
            </div>
            <div className="stat-card">
              <span>Pipeline Stage</span>
              <strong>Stage {selectedRank.ppStage} of 2</strong>
              <small className="text-cyan-400">1F1B Schedule</small>
            </div>
            <div className="stat-card">
              <span>Bus Bandwidth</span>
              <strong>{selectedRank.bandwidthGbps} GB/s</strong>
              <small className="text-emerald-400">98.2% Line Rate</small>
            </div>
            <div className="stat-card">
              <span>Pipeline Bubble</span>
              <strong>&lt; 3.8%</strong>
              <small className="text-teal-400">Optimized overlap</small>
            </div>
          </div>

          <div className="dist-inspector__recommendations">
            <h4>Distributed Communication Diagnosis</h4>
            <div className="dist-rec-box">
              <div className="rec-title">
                <Zap size={14} className="text-teal-400" />
                <span>NCCL FastSocket & Ring Partitioning Active</span>
              </div>
              <p>
                Gradient synchronization across TP Group {selectedRank.tpGroup} utilizes direct NVLink memory load/stores without host CPU bounce buffers. 
                InfiniBand RDMA is isolated strictly to PP stage transitions between Node 0 and Node 1.
              </p>
              <div className="rec-footer">
                <span>Inter-node Latency: <strong>1.45 ms</strong></span>
                <span>NCCL Tree Depth: <strong>2</strong></span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
export default DistributedTopologyViewer;
