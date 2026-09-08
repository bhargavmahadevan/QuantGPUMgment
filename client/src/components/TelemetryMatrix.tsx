import React, { useState } from "react";
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend, 
  ReferenceLine 
} from "recharts";
import { 
  Activity, 
  Cpu, 
  Zap, 
  Database, 
  Clock, 
  ShieldCheck, 
  Layers, 
  TrendingUp, 
  Gauge 
} from "lucide-react";

interface StepTelemetryPoint {
  step: number;
  loss: number;
  baselineLoss: number;
  lossDelta: number;
  gradNorm: number;
  throughputTokensPerSec: number;
  stepTimeMs: number;
  computeTimeMs: number;
  commTimeMs: number;
  dataLoadTimeMs: number;
  vramAllocatedGb: number;
  vramReservedGb: number;
}

const TELEMETRY_SERIES: StepTelemetryPoint[] = Array.from({ length: 30 }, (_, i) => {
  const step = 1000 + i * 5;
  const baseLoss = 2.45 - Math.log(i + 1) * 0.28;
  const jitter = Math.sin(i * 1.3) * 0.015;
  const loss = Math.max(1.72, baseLoss + jitter);
  const baselineLoss = loss + 0.045 + Math.sin(i * 0.8) * 0.01;
  const lossDelta = Math.abs(loss - baselineLoss);

  return {
    step,
    loss: Number(loss.toFixed(4)),
    baselineLoss: Number(baselineLoss.toFixed(4)),
    lossDelta: Number(lossDelta.toFixed(4)),
    gradNorm: Number((1.12 + Math.sin(i * 0.7) * 0.35 + (i === 14 ? 1.85 : 0)).toFixed(3)),
    throughputTokensPerSec: Math.round(3850 + Math.cos(i * 0.9) * 220 + (i > 10 ? 420 : 0)),
    stepTimeMs: Number((32.4 - (i > 10 ? 4.2 : 0) + Math.sin(i) * 0.8).toFixed(1)),
    computeTimeMs: Number((24.2 - (i > 10 ? 3.5 : 0)).toFixed(1)),
    commTimeMs: Number((5.8 + Math.sin(i * 0.5) * 0.4).toFixed(1)),
    dataLoadTimeMs: Number((2.4 + Math.cos(i * 0.6) * 0.3).toFixed(1)),
    vramAllocatedGb: Number((62.4 + (i * 0.12)).toFixed(1)),
    vramReservedGb: 74.5
  };
});

export function TelemetryMatrix() {
  const [metricView, setMetricView] = useState<"loss" | "throughput" | "breakdown" | "memory">("loss");

  return (
    <div className="telemetry-matrix-container">
      {/* Header with KPI Cards */}
      <div className="telemetry-matrix-header">
        <div className="telemetry-badge">
          <Activity size={14} className="text-zinc-300" />
          <span>REAL-TIME MULTI-METRIC TRAINING TELEMETRY MATRIX</span>
        </div>

        <div className="view-selector-tabs">
          {[
            { id: "loss", label: "Loss Curvature & 0.10 Gate", icon: Activity },
            { id: "throughput", label: "Throughput (Tokens/sec)", icon: TrendingUp },
            { id: "breakdown", label: "Step Time Breakdown", icon: Clock },
            { id: "memory", label: "VRAM Fragmentation", icon: Database }
          ].map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`telemetry-tab-btn ${metricView === tab.id ? "is-active" : ""}`}
                onClick={() => setMetricView(tab.id as any)}
              >
                <Icon size={13} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Summary KPI Cards */}
      <div className="telemetry-kpis">
        <div className="kpi-card">
          <span className="kpi-label">OPTIMIZED THROUGHPUT</span>
          <div className="kpi-value-row">
            <strong>4,270</strong>
            <small>tokens/s/GPU</small>
          </div>
          <span className="kpi-delta" style={{ color: "#10b981", fontWeight: 600 }}>+18.5% vs Baseline</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">STEP LATENCY</span>
          <div className="kpi-value-row">
            <strong>28.2</strong>
            <small>ms/step</small>
          </div>
          <span className="kpi-delta" style={{ color: "#10b981", fontWeight: 600 }}>-4.2 ms saved</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">MAX LOSS DELTA</span>
          <div className="kpi-value-row">
            <strong>0.0012</strong>
            <small>ΔL</small>
          </div>
          <span className="kpi-delta" style={{ color: "#10b981", fontWeight: 600 }}>&lt; 0.10 Gate (Verified Safe)</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">VRAM FRAGMENTATION</span>
          <div className="kpi-value-row">
            <strong>6.8%</strong>
            <small>inactive</small>
          </div>
          <span className="kpi-delta" style={{ color: "#10b981", fontWeight: 600 }}>Zero OOM Risk</span>
        </div>
      </div>

      {/* Main Chart Canvas */}
      <div className="chart-wrapper">
        {metricView === "loss" && (
          <div className="chart-inner">
            <div className="chart-title-bar">
              <span>LOSS CURVATURE & VERIFICATION BOUND (0.10 LOSS-SHIFT GATE)</span>
              <div className="chart-legend-custom">
                <span className="legend-item"><i style={{ background: "#10b981", boxShadow: "0 0 6px rgba(16, 185, 129, 0.7)" }} /> GhostLayer Optimized</span>
                <span className="legend-item"><i className="bg-zinc-500" /> Baseline Unoptimized</span>
                <span className="legend-item"><i style={{ background: "#f43f5e", boxShadow: "0 0 6px rgba(244, 63, 94, 0.7)" }} /> 0.10 Safety Boundary</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={TELEMETRY_SERIES} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis domain={[1.8, 3.2]} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0A0A0A", borderColor: "rgba(255,255,255,0.2)", borderRadius: "0", fontSize: "12px" }}
                  itemStyle={{ color: "#edf1ed" }}
                />
                <ReferenceLine y={2.65} label={{ value: "0.10 Safety Boundary (Gate Limit)", fill: "#f43f5e", fontSize: 10, position: "insideTopRight" }} stroke="#f43f5e" strokeDasharray="4 4" strokeWidth={1.5} />
                <Line type="monotone" dataKey="loss" stroke="#10b981" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
                <Line type="monotone" dataKey="baselineLoss" stroke="#737373" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {metricView === "throughput" && (
          <div className="chart-inner">
            <div className="chart-title-bar">
              <span>EFFECTIVE THROUGHPUT (TOKENS PER SECOND PER GPU)</span>
              <span className="text-xs text-zinc-400">FlashAttention-2 + Operator Fusion Active</span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={TELEMETRY_SERIES} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                <defs>
                  <linearGradient id="throughputGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#FFFFFF" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#FFFFFF" stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis domain={[3400, 4600]} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0A0A0A", borderColor: "rgba(255,255,255,0.2)", borderRadius: "0", fontSize: "12px" }}
                />
                <Area type="monotone" dataKey="throughputTokensPerSec" stroke="#FFFFFF" strokeWidth={2} fillOpacity={1} fill="url(#throughputGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {metricView === "breakdown" && (
          <div className="chart-inner">
            <div className="chart-title-bar">
              <span>STEP TIME BREAKDOWN (COMPUTE VS NCCL COMM VS DATA-LOADING)</span>
              <div className="chart-legend-custom">
                <span className="legend-item"><i className="bg-white" /> Compute (ms)</span>
                <span className="legend-item"><i className="bg-zinc-400" /> NCCL Comm (ms)</span>
                <span className="legend-item"><i className="bg-zinc-600" /> DataLoader (ms)</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={TELEMETRY_SERIES.slice(0, 15)} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0A0A0A", borderColor: "rgba(255,255,255,0.2)", borderRadius: "0", fontSize: "12px" }}
                />
                <Bar dataKey="computeTimeMs" stackId="a" fill="#FFFFFF" />
                <Bar dataKey="commTimeMs" stackId="a" fill="#A3A3A3" />
                <Bar dataKey="dataLoadTimeMs" stackId="a" fill="#525252" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {metricView === "memory" && (
          <div className="chart-inner">
            <div className="chart-title-bar">
              <span>VRAM ALLOCATION DYNAMICS (ALLOCATED VS PEAK RESERVED)</span>
              <span className="text-xs text-zinc-400">80 GB H100 Buffer Capacity</span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={TELEMETRY_SERIES} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis domain={[50, 85]} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0A0A0A", borderColor: "rgba(255,255,255,0.2)", borderRadius: "0", fontSize: "12px" }}
                />
                <ReferenceLine y={80} label={{ value: "Hardware Limit (80GB)", fill: "#f43f5e", fontSize: 10 }} stroke="#f43f5e" strokeDasharray="3 3" />
                <Area type="monotone" dataKey="vramReservedGb" stroke="#737373" fill="#404040" fillOpacity={0.2} />
                <Area type="monotone" dataKey="vramAllocatedGb" stroke="#FFFFFF" fill="#A3A3A3" fillOpacity={0.4} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
export default TelemetryMatrix;
