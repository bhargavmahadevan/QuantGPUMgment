import React, { useState, useMemo } from "react";
import {
  Calculator,
  DollarSign,
  TrendingUp,
  ShieldCheck,
  Zap,
  Copy,
  Check,
  ArrowUpRight,
  Server,
  Layers,
  Clock,
  Sparkles,
  Percent,
} from "lucide-react";

interface TopologyPreset {
  id: string;
  name: string;
  gpus: number;
  ratePerGpuHour: number;
  description: string;
}

const TOPOLOGY_PRESETS: TopologyPreset[] = [
  {
    id: "8x-h100",
    name: "8x H100 SXM5",
    gpus: 8,
    ratePerGpuHour: 3.50,
    description: "Standard single-node fine-tuning & mid-scale pre-training",
  },
  {
    id: "8x-a100",
    name: "8x A100 80GB",
    gpus: 8,
    ratePerGpuHour: 2.20,
    description: "Workhorse multimodal & embedding model cluster",
  },
  {
    id: "32x-h100",
    name: "32x H100 InfiniBand",
    gpus: 32,
    ratePerGpuHour: 3.50,
    description: "Multi-node FSDP / Megatron training cluster",
  },
  {
    id: "64x-h100",
    name: "64x H100 Quantum-2",
    gpus: 64,
    ratePerGpuHour: 3.50,
    description: "High-throughput frontier LLM pre-training cluster",
  },
  {
    id: "256x-h100",
    name: "256x H100 SuperPOD",
    gpus: 256,
    ratePerGpuHour: 3.50,
    description: "Enterprise foundation model training facility",
  },
];

export default function GpuProfitCalculator() {
  const [selectedPresetId, setSelectedPresetId] = useState<string>("8x-h100");
  const [customGpus, setCustomGpus] = useState<number>(8);
  const [customRate, setCustomRate] = useState<number>(3.50);
  const [monthlyHours, setMonthlyHours] = useState<number>(500);
  const [ioStallPct, setIoStallPct] = useState<number>(18);
  const [precisionWastePct, setPrecisionWastePct] = useState<number>(15);
  const [copied, setCopied] = useState<boolean>(false);

  const isCustom = selectedPresetId === "custom";

  const activePreset = useMemo(() => {
    return TOPOLOGY_PRESETS.find((p) => p.id === selectedPresetId);
  }, [selectedPresetId]);

  const numGpus = isCustom ? customGpus : activePreset ? activePreset.gpus : 8;
  const ratePerGpu = isCustom ? customRate : activePreset ? activePreset.ratePerGpuHour : 3.50;

  // Financial Calculations
  const clusterHourlySpend = useMemo(() => {
    return numGpus * ratePerGpu;
  }, [numGpus, ratePerGpu]);

  const monthlyBaselineSpend = useMemo(() => {
    return clusterHourlySpend * monthlyHours;
  }, [clusterHourlySpend, monthlyHours]);

  const totalWastePct = useMemo(() => {
    return Math.min(50, ioStallPct + precisionWastePct);
  }, [ioStallPct, precisionWastePct]);

  const monthlyWastedSpend = useMemo(() => {
    return monthlyBaselineSpend * (totalWastePct / 100);
  }, [monthlyBaselineSpend, totalWastePct]);

  const monthlyRecoverableSavings = useMemo(() => {
    // GhostLayer typically recovers 75% to 85% of diagnosed waste
    return monthlyWastedSpend * 0.80;
  }, [monthlyWastedSpend]);

  const annualRecoverableSavings = useMemo(() => {
    return monthlyRecoverableSavings * 12;
  }, [monthlyRecoverableSavings]);

  const auditFee = 2500;
  const auditPaybackDays = useMemo(() => {
    const dailySavings = monthlyRecoverableSavings / 30;
    if (dailySavings <= 0) return 999;
    return Math.max(1, Math.round((auditFee / dailySavings) * 10) / 10);
  }, [monthlyRecoverableSavings, auditFee]);

  const auditRoiMultiple = useMemo(() => {
    if (auditFee <= 0) return 0;
    return Math.round((annualRecoverableSavings / auditFee) * 10) / 10;
  }, [annualRecoverableSavings, auditFee]);

  const handleCopyCommand = () => {
    navigator.clipboard.writeText("pip install ghostlayer && ghostlayer audit --steps 100");
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className="gpu-profit-calculator" style={{
      background: "linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(9, 13, 22, 0.98) 100%)",
      border: "1px solid rgba(255, 255, 255, 0.1)",
      borderRadius: "16px",
      padding: "32px",
      color: "#f8fafc",
      boxShadow: "0 20px 50px rgba(0, 0, 0, 0.5)",
      backdropFilter: "blur(16px)",
      margin: "40px 0"
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px", marginBottom: "28px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", color: "#38bdf8", fontWeight: 700, fontSize: "0.85rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            <Calculator size={16} />
            <span>Interactive FinOps Engine</span>
          </div>
          <h2 style={{ fontSize: "1.85rem", fontWeight: 800, margin: "6px 0 0 0", color: "#ffffff" }}>
            GPU Waste & Profit Realization Calculator
          </h2>
          <p style={{ color: "#94a3b8", fontSize: "0.95rem", margin: "6px 0 0 0", maxWidth: "680px" }}>
            Unprofiled PyTorch pipelines typically leak 15% to 35% of GPU compute through DataLoader I/O stalls and unoptimized kernels. Measure your exact dollar recovery below.
          </p>
        </div>

        {/* 1-Click Terminal Audit */}
        <div style={{
          background: "rgba(30, 41, 59, 0.8)",
          border: "1px solid rgba(6, 182, 212, 0.3)",
          borderRadius: "10px",
          padding: "10px 16px",
          display: "flex",
          alignItems: "center",
          gap: "12px"
        }}>
          <code style={{ fontFamily: "monospace", color: "#38bdf8", fontSize: "0.85rem" }}>
            pip install ghostlayer
          </code>
          <button
            onClick={handleCopyCommand}
            style={{
              background: copied ? "rgba(16, 185, 129, 0.2)" : "rgba(6, 182, 212, 0.2)",
              border: `1px solid ${copied ? "#10b981" : "#06b6d4"}`,
              color: copied ? "#34d399" : "#38bdf8",
              padding: "5px 10px",
              borderRadius: "6px",
              cursor: "pointer",
              fontSize: "0.8rem",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              fontWeight: 600,
              transition: "all 0.2s ease"
            }}
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
      </div>

      {/* Cluster Topology Preset Selector */}
      <div style={{ marginBottom: "28px" }}>
        <label style={{ fontSize: "0.85rem", color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: "10px" }}>
          1. Select Cluster Topology
        </label>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "10px" }}>
          {TOPOLOGY_PRESETS.map((p) => {
            const isSelected = selectedPresetId === p.id;
            return (
              <button
                key={p.id}
                onClick={() => setSelectedPresetId(p.id)}
                style={{
                  background: isSelected ? "rgba(6, 182, 212, 0.15)" : "rgba(30, 41, 59, 0.5)",
                  border: `1px solid ${isSelected ? "#06b6d4" : "rgba(255, 255, 255, 0.08)"}`,
                  borderRadius: "8px",
                  padding: "12px",
                  textAlign: "left",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                  color: "#ffffff",
                }}
              >
                <div style={{ fontWeight: 700, fontSize: "0.95rem", color: isSelected ? "#38bdf8" : "#f1f5f9" }}>{p.name}</div>
                <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginTop: "4px" }}>${(p.gpus * p.ratePerGpuHour).toFixed(2)}/hr ({p.gpus}x GPUs)</div>
              </button>
            );
          })}
          <button
            onClick={() => setSelectedPresetId("custom")}
            style={{
              background: isCustom ? "rgba(6, 182, 212, 0.15)" : "rgba(30, 41, 59, 0.5)",
              border: `1px solid ${isCustom ? "#06b6d4" : "rgba(255, 255, 255, 0.08)"}`,
              borderRadius: "8px",
              padding: "12px",
              textAlign: "left",
              cursor: "pointer",
              transition: "all 0.15s ease",
              color: "#ffffff",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: "0.95rem", color: isCustom ? "#38bdf8" : "#f1f5f9" }}>Custom Cluster</div>
            <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginTop: "4px" }}>Specify GPUs & Rate</div>
          </button>
        </div>
      </div>

      {/* Sliders Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "24px", marginBottom: "32px", background: "rgba(15, 23, 42, 0.6)", padding: "20px", borderRadius: "12px", border: "1px solid rgba(255, 255, 255, 0.06)" }}>
        {isCustom && (
          <>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>GPU Count</span>
                <span style={{ fontWeight: 700, color: "#38bdf8" }}>{customGpus} GPUs</span>
              </div>
              <input
                type="range"
                min={1}
                max={512}
                value={customGpus}
                onChange={(e) => setCustomGpus(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#06b6d4" }}
              />
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>Hourly Rate ($/GPU-hr)</span>
                <span style={{ fontWeight: 700, color: "#38bdf8" }}>${customRate.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.35}
                max={8.00}
                step={0.05}
                value={customRate}
                onChange={(e) => setCustomRate(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#06b6d4" }}
              />
            </div>
          </>
        )}

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>Monthly Training Hours</span>
            <span style={{ fontWeight: 700, color: "#f8fafc" }}>{monthlyHours} hrs / month</span>
          </div>
          <input
            type="range"
            min={100}
            max={720}
            step={10}
            value={monthlyHours}
            onChange={(e) => setMonthlyHours(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#06b6d4" }}
          />
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>DataLoader I/O Starvation</span>
            <span style={{ fontWeight: 700, color: "#f87171" }}>{ioStallPct}% idle time</span>
          </div>
          <input
            type="range"
            min={5}
            max={40}
            value={ioStallPct}
            onChange={(e) => setIoStallPct(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#ef4444" }}
          />
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>Suboptimal Kernels / Precision</span>
            <span style={{ fontWeight: 700, color: "#f59e0b" }}>+{precisionWastePct}% latency</span>
          </div>
          <input
            type="range"
            min={0}
            max={30}
            value={precisionWastePct}
            onChange={(e) => setPrecisionWastePct(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#f59e0b" }}
          />
        </div>
      </div>

      {/* KPI Readout Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px", marginBottom: "32px" }}>
        <div style={{ background: "rgba(30, 41, 59, 0.7)", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ fontSize: "0.8rem", color: "#94a3b8", textTransform: "uppercase", fontWeight: 600 }}>Monthly Baseline Spend</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 800, color: "#f8fafc", marginTop: "6px" }}>${monthlyBaselineSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
          <div style={{ fontSize: "0.8rem", color: "#64748b", marginTop: "4px" }}>${clusterHourlySpend.toFixed(2)}/hr across {numGpus} GPUs</div>
        </div>

        <div style={{ background: "rgba(239, 68, 68, 0.08)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ fontSize: "0.8rem", color: "#fca5a5", textTransform: "uppercase", fontWeight: 600 }}>Identified Monthly Waste</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 800, color: "#f87171", marginTop: "6px" }}>${monthlyWastedSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
          <div style={{ fontSize: "0.8rem", color: "#f87171", marginTop: "4px" }}>{totalWastePct}% total compute leakage</div>
        </div>

        <div style={{ background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.3)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ fontSize: "0.8rem", color: "#6ee7b7", textTransform: "uppercase", fontWeight: 600 }}>Recoverable Annual Spend</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 800, color: "#34d399", marginTop: "6px" }}>${annualRecoverableSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
          <div style={{ fontSize: "0.8rem", color: "#34d399", marginTop: "4px" }}>+${monthlyRecoverableSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}/mo net runway</div>
        </div>

        <div style={{ background: "rgba(6, 182, 212, 0.08)", border: "1px solid rgba(6, 182, 212, 0.3)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ fontSize: "0.8rem", color: "#7dd3fc", textTransform: "uppercase", fontWeight: 600 }}>$2,500 Audit Payback</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 800, color: "#38bdf8", marginTop: "6px" }}>{auditPaybackDays} Days</div>
          <div style={{ fontSize: "0.8rem", color: "#38bdf8", marginTop: "4px" }}>{auditRoiMultiple}x 1-Year ROI Multiple</div>
        </div>
      </div>

      {/* 3-Tier Commercial Conversion Matrix */}
      <div style={{ borderTop: "1px solid rgba(255, 255, 255, 0.08)", paddingTop: "28px" }}>
        <div style={{ textAlign: "center", marginBottom: "24px" }}>
          <h3 style={{ fontSize: "1.35rem", fontWeight: 700, margin: 0 }}>GhostLayer Commercial Offerings</h3>
          <p style={{ color: "#94a3b8", fontSize: "0.9rem", margin: "6px 0 0 0" }}>
            Transparent pricing designed for rapid engineering lead discretionary approval without procurement friction.
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "20px" }}>
          {/* Tier 1 */}
          <div style={{
            background: "linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%)",
            border: "1px solid #06b6d4",
            borderRadius: "12px",
            padding: "24px",
            position: "relative",
            boxShadow: "0 10px 30px rgba(6, 182, 212, 0.15)"
          }}>
            <div style={{ position: "absolute", top: "-12px", right: "20px", background: "#06b6d4", color: "#090d16", padding: "3px 10px", borderRadius: "999px", fontSize: "0.75rem", fontWeight: 800 }}>
              MOST POPULAR WEDGE
            </div>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#38bdf8" }}>Tier 1: Pre-Flight Audit</div>
            <div style={{ fontSize: "1.8rem", fontWeight: 800, margin: "10px 0 4px 0" }}>$2,500 <span style={{ fontSize: "0.85rem", fontWeight: 400, color: "#94a3b8" }}>Flat Fee</span></div>
            <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: "0 0 16px 0" }}>
              Under $3k corporate card limit • 48-Hour delivery • Recouped in ~{auditPaybackDays} days
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.85rem", color: "#cbd5e1" }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> 1x to 8x GPU Staging Run Validation</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> DataLoader I/O stall & VRAM headroom audit</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Executive C-level HTML & Markdown Report</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Zero fee if no actionable findings detected</li>
            </ul>
            <a
              href="mailto:solutions@ghostlayer.ai?subject=Book%20Pre-Flight%20Diagnostic%20Audit%20($2,500)"
              style={{
                display: "block",
                textAlign: "center",
                background: "linear-gradient(135deg, #06b6d4, #10b981)",
                color: "#090d16",
                padding: "12px",
                borderRadius: "8px",
                fontWeight: 700,
                textDecoration: "none",
                fontSize: "0.95rem"
              }}
            >
              Book 48-Hour Audit ($2,500)
            </a>
          </div>

          {/* Tier 2 */}
          <div style={{
            background: "rgba(15, 23, 42, 0.8)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            borderRadius: "12px",
            padding: "24px"
          }}>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#34d399" }}>Tier 2: Team Platform SaaS</div>
            <div style={{ fontSize: "1.8rem", fontWeight: 800, margin: "10px 0 4px 0" }}>$499 <span style={{ fontSize: "0.85rem", fontWeight: 400, color: "#94a3b8" }}>/ month</span></div>
            <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: "0 0 16px 0" }}>
              8 to 64 GPUs • Continuous fleet training oversight
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.85rem", color: "#cbd5e1" }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Real-time Slack & Discord stall alerts</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Prometheus FinOps dollar metrics exporter</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Post-mortem decision replay history</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Continuous batch sizing recommendations</li>
            </ul>
            <a
              href="mailto:solutions@ghostlayer.ai?subject=Inquire%20Team%20Platform%20SaaS"
              style={{
                display: "block",
                textAlign: "center",
                background: "rgba(30, 41, 59, 0.9)",
                border: "1px solid rgba(255, 255, 255, 0.15)",
                color: "#f8fafc",
                padding: "12px",
                borderRadius: "8px",
                fontWeight: 600,
                textDecoration: "none",
                fontSize: "0.95rem"
              }}
            >
              Start Team SaaS ($499/mo)
            </a>
          </div>

          {/* Tier 3 */}
          <div style={{
            background: "rgba(15, 23, 42, 0.8)",
            border: "1px solid rgba(168, 85, 247, 0.3)",
            borderRadius: "12px",
            padding: "24px"
          }}>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#c084fc" }}>Tier 3: Enterprise Assurance</div>
            <div style={{ fontSize: "1.8rem", fontWeight: 800, margin: "10px 0 4px 0" }}>$25,000 <span style={{ fontSize: "0.85rem", fontWeight: 400, color: "#94a3b8" }}>/ year</span></div>
            <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: "0 0 16px 0" }}>
              Air-Gapped VPC • Zero Data Exfiltration SLA
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.85rem", color: "#cbd5e1" }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> 100% offline air-gapped cryptographic tokens</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Dedicated custom optimization heuristics</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Cryptographic BaselineLock SLA assurance</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#34d399" /> Dedicated ML Infrastructure Solutions Engineer</li>
            </ul>
            <a
              href="mailto:solutions@ghostlayer.ai?subject=Enterprise%20Compute%20Assurance%20Inquiry"
              style={{
                display: "block",
                textAlign: "center",
                background: "rgba(168, 85, 247, 0.15)",
                border: "1px solid rgba(168, 85, 247, 0.4)",
                color: "#e9d5ff",
                padding: "12px",
                borderRadius: "8px",
                fontWeight: 600,
                textDecoration: "none",
                fontSize: "0.95rem"
              }}
            >
              Inquire Enterprise VPC
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
