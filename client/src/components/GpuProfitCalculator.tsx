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
      background: "#0A0A0A",
      border: "1px solid rgba(255, 255, 255, 0.12)",
      borderRadius: "0",
      padding: "32px",
      color: "#FFFFFF",
      boxShadow: "0 24px 60px rgba(0, 0, 0, 0.6)",
      margin: "40px 0",
      fontFamily: '"DM Mono", monospace'
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px", marginBottom: "28px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", color: "#A3A3A3", fontWeight: 700, fontSize: "0.75rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            <Calculator size={14} />
            <span>Interactive FinOps Engine</span>
          </div>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 400, margin: "6px 0 0 0", color: "#FFFFFF", fontFamily: '"Instrument Serif", Georgia, serif', letterSpacing: "-0.02em" }}>
            GPU Waste & Profit Realization Calculator
          </h2>
          <p style={{ color: "#737373", fontSize: "0.85rem", margin: "6px 0 0 0", maxWidth: "680px", lineHeight: 1.6 }}>
            Unprofiled PyTorch pipelines leak 15% to 35% of GPU compute through DataLoader I/O stalls and unoptimized kernels. Measure your exact dollar recovery below.
          </p>
        </div>

        {/* 1-Click Terminal Audit */}
        <div style={{
          background: "#111111",
          border: "1px solid rgba(255, 255, 255, 0.14)",
          borderRadius: "0",
          padding: "8px 14px",
          display: "flex",
          alignItems: "center",
          gap: "12px"
        }}>
          <code style={{ fontFamily: '"DM Mono", monospace', color: "#D4D4D4", fontSize: "0.8rem" }}>
            pip install ghostlayer
          </code>
          <button
            onClick={handleCopyCommand}
            style={{
              background: copied ? "#FFFFFF" : "rgba(255, 255, 255, 0.08)",
              border: "1px solid rgba(255, 255, 255, 0.25)",
              color: copied ? "#000000" : "#FFFFFF",
              padding: "4px 10px",
              borderRadius: "0",
              cursor: "pointer",
              fontSize: "0.75rem",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              fontWeight: 600,
              transition: "all 0.15s ease"
            }}
          >
            {copied ? <Check size={12} /> : <Copy size={12} />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>

      {/* Cluster Topology Preset Selector */}
      <div style={{ marginBottom: "28px" }}>
        <label style={{ fontSize: "0.75rem", color: "#737373", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", display: "block", marginBottom: "10px" }}>
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
                  background: isSelected ? "#181818" : "#111111",
                  border: `1px solid ${isSelected ? "#FFFFFF" : "rgba(255, 255, 255, 0.1)"}`,
                  borderRadius: "0",
                  padding: "12px",
                  textAlign: "left",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                  color: "#FFFFFF",
                }}
              >
                <div style={{ fontWeight: 700, fontSize: "0.85rem", color: isSelected ? "#FFFFFF" : "#D4D4D4" }}>{p.name}</div>
                <div style={{ fontSize: "0.75rem", color: "#737373", marginTop: "4px" }}>${(p.gpus * p.ratePerGpuHour).toFixed(2)}/hr ({p.gpus}x GPUs)</div>
              </button>
            );
          })}
          <button
            onClick={() => setSelectedPresetId("custom")}
            style={{
              background: isCustom ? "#181818" : "#111111",
              border: `1px solid ${isCustom ? "#FFFFFF" : "rgba(255, 255, 255, 0.1)"}`,
              borderRadius: "0",
              padding: "12px",
              textAlign: "left",
              cursor: "pointer",
              transition: "all 0.15s ease",
              color: "#FFFFFF",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: "0.85rem", color: isCustom ? "#FFFFFF" : "#D4D4D4" }}>Custom Cluster</div>
            <div style={{ fontSize: "0.75rem", color: "#737373", marginTop: "4px" }}>Specify GPUs & Rate</div>
          </button>
        </div>
      </div>

      {/* Sliders Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "24px", marginBottom: "32px", background: "#111111", padding: "20px", borderRadius: "0", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
        {isCustom && (
          <>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ fontSize: "0.78rem", color: "#737373" }}>GPU Count</span>
                <span style={{ fontWeight: 700, color: "#FFFFFF" }}>{customGpus} GPUs</span>
              </div>
              <input
                type="range"
                min={1}
                max={512}
                value={customGpus}
                onChange={(e) => setCustomGpus(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#FFFFFF" }}
              />
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ fontSize: "0.78rem", color: "#737373" }}>Hourly Rate ($/GPU-hr)</span>
                <span style={{ fontWeight: 700, color: "#FFFFFF" }}>${customRate.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.35}
                max={8.00}
                step={0.05}
                value={customRate}
                onChange={(e) => setCustomRate(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#FFFFFF" }}
              />
            </div>
          </>
        )}

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "0.78rem", color: "#737373" }}>Monthly Training Hours</span>
            <span style={{ fontWeight: 700, color: "#FFFFFF" }}>{monthlyHours} hrs / month</span>
          </div>
          <input
            type="range"
            min={100}
            max={720}
            step={10}
            value={monthlyHours}
            onChange={(e) => setMonthlyHours(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#FFFFFF" }}
          />
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "0.78rem", color: "#737373" }}>DataLoader I/O Starvation</span>
            <span style={{ fontWeight: 700, color: "#A3A3A3" }}>{ioStallPct}% idle time</span>
          </div>
          <input
            type="range"
            min={5}
            max={40}
            value={ioStallPct}
            onChange={(e) => setIoStallPct(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#A3A3A3" }}
          />
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "0.78rem", color: "#737373" }}>Suboptimal Kernels / Precision</span>
            <span style={{ fontWeight: 700, color: "#A3A3A3" }}>+{precisionWastePct}% latency</span>
          </div>
          <input
            type="range"
            min={0}
            max={30}
            value={precisionWastePct}
            onChange={(e) => setPrecisionWastePct(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#A3A3A3" }}
          />
        </div>
      </div>

      {/* KPI Readout Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px", marginBottom: "32px" }}>
        <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "0", padding: "18px" }}>
          <div style={{ fontSize: "0.75rem", color: "#737373", textTransform: "uppercase", fontWeight: 600 }}>Monthly Baseline Spend</div>
          <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#FFFFFF", marginTop: "6px" }}>${monthlyBaselineSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
          <div style={{ fontSize: "0.75rem", color: "#737373", marginTop: "4px" }}>${clusterHourlySpend.toFixed(2)}/hr across {numGpus} GPUs</div>
        </div>

        <div style={{ background: "rgba(244, 63, 94, 0.05)", border: "1px solid rgba(244, 63, 94, 0.3)", borderRadius: "0", padding: "18px" }}>
          <div style={{ fontSize: "0.75rem", color: "#fb7185", textTransform: "uppercase", fontWeight: 600 }}>Identified Monthly Waste</div>
          <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#f43f5e", marginTop: "6px" }}>${monthlyWastedSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
          <div>
            <span style={{ display: "inline-flex", padding: "2px 8px", borderRadius: "9999px", background: "rgba(244, 63, 94, 0.15)", color: "#f43f5e", border: "1px solid rgba(244, 63, 94, 0.35)", fontSize: "0.68rem", fontWeight: 700, marginTop: "6px" }}>
              {totalWastePct}% COMPUTE LEAKAGE
            </span>
          </div>
        </div>

        <div style={{ background: "rgba(16, 185, 129, 0.05)", border: "1px solid rgba(16, 185, 129, 0.35)", borderRadius: "0", padding: "18px" }}>
          <div style={{ fontSize: "0.75rem", color: "#34d399", textTransform: "uppercase", fontWeight: 600 }}>Recoverable Annual Spend</div>
          <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#10b981", marginTop: "6px" }}>${annualRecoverableSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
          <div>
            <span style={{ display: "inline-flex", padding: "2px 8px", borderRadius: "9999px", background: "rgba(16, 185, 129, 0.15)", color: "#10b981", border: "1px solid rgba(16, 185, 129, 0.35)", fontSize: "0.68rem", fontWeight: 700, marginTop: "6px" }}>
              +${monthlyRecoverableSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}/mo NET RUNWAY
            </span>
          </div>
        </div>

        <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "0", padding: "18px" }}>
          <div style={{ fontSize: "0.75rem", color: "#737373", textTransform: "uppercase", fontWeight: 600 }}>$2,500 Audit Payback</div>
          <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#FFFFFF", marginTop: "6px" }}>{auditPaybackDays} Days</div>
          <div style={{ fontSize: "0.75rem", color: "#737373", marginTop: "4px" }}>{auditRoiMultiple}x 1-Year ROI Multiple</div>
        </div>
      </div>

      {/* 3-Tier Commercial Conversion Matrix */}
      <div style={{ borderTop: "1px solid rgba(255, 255, 255, 0.08)", paddingTop: "28px" }}>
        <div style={{ textAlign: "center", marginBottom: "24px" }}>
          <h3 style={{ fontSize: "1.35rem", fontWeight: 400, margin: 0, fontFamily: '"Instrument Serif", Georgia, serif', letterSpacing: "-0.01em" }}>GhostLayer Commercial Offerings</h3>
          <p style={{ color: "#737373", fontSize: "0.85rem", margin: "6px 0 0 0" }}>
            Transparent pricing designed for rapid engineering lead discretionary approval without procurement friction.
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "20px" }}>
          {/* Tier 1 */}
          <div style={{
            background: "#111111",
            border: "1px solid #FFFFFF",
            borderRadius: "0",
            padding: "24px",
            position: "relative",
            boxShadow: "0 10px 30px rgba(0, 0, 0, 0.6)"
          }}>
            <div style={{ position: "absolute", top: "-10px", right: "20px", background: "#FFFFFF", color: "#000000", padding: "2px 8px", borderRadius: "0", fontSize: "0.68rem", fontWeight: 700, letterSpacing: "0.06em" }}>
              MOST POPULAR WEDGE
            </div>
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#FFFFFF" }}>Tier 1: Pre-Flight Audit</div>
            <div style={{ fontSize: "1.75rem", fontWeight: 700, margin: "10px 0 4px 0", color: "#FFFFFF" }}>$2,500 <span style={{ fontSize: "0.8rem", fontWeight: 400, color: "#737373" }}>Flat Fee</span></div>
            <p style={{ fontSize: "0.8rem", color: "#737373", margin: "0 0 16px 0" }}>
              Under $3k corporate card limit • 48-Hour delivery • Recouped in ~{auditPaybackDays} days
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.8rem", color: "#D4D4D4" }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> 1x to 8x GPU Staging Run Validation</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> DataLoader I/O stall & VRAM headroom audit</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Executive C-level HTML & Markdown Report</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Zero fee if no actionable findings detected</li>
            </ul>
            <a
              href="mailto:solutions@ghostlayer.ai?subject=Book%20Pre-Flight%20Diagnostic%20Audit%20($2,500)"
              style={{
                display: "block",
                textAlign: "center",
                background: "#FFFFFF",
                color: "#000000",
                padding: "12px",
                borderRadius: "0",
                fontWeight: 700,
                textDecoration: "none",
                fontSize: "0.85rem",
                letterSpacing: "0.04em",
                transition: "background 0.15s ease"
              }}
            >
              Book 48-Hour Audit ($2,500)
            </a>
          </div>

          {/* Tier 2 */}
          <div style={{
            background: "#0D0D0D",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            borderRadius: "0",
            padding: "24px"
          }}>
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#D4D4D4" }}>Tier 2: Team Platform SaaS</div>
            <div style={{ fontSize: "1.75rem", fontWeight: 700, margin: "10px 0 4px 0", color: "#FFFFFF" }}>$499 <span style={{ fontSize: "0.8rem", fontWeight: 400, color: "#737373" }}>/ month</span></div>
            <p style={{ fontSize: "0.8rem", color: "#737373", margin: "0 0 16px 0" }}>
              8 to 64 GPUs • Continuous fleet training oversight
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.8rem", color: "#D4D4D4" }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Real-time Slack & Discord stall alerts</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Prometheus FinOps dollar metrics exporter</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Post-mortem decision replay history</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Continuous batch sizing recommendations</li>
            </ul>
            <a
              href="mailto:solutions@ghostlayer.ai?subject=Inquire%20Team%20Platform%20SaaS"
              style={{
                display: "block",
                textAlign: "center",
                background: "transparent",
                border: "1px solid rgba(255, 255, 255, 0.3)",
                color: "#FFFFFF",
                padding: "12px",
                borderRadius: "0",
                fontWeight: 600,
                textDecoration: "none",
                fontSize: "0.85rem",
                letterSpacing: "0.04em",
                transition: "all 0.15s ease"
              }}
            >
              Start Team SaaS ($499/mo)
            </a>
          </div>

          {/* Tier 3 */}
          <div style={{
            background: "#0D0D0D",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            borderRadius: "0",
            padding: "24px"
          }}>
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#D4D4D4" }}>Tier 3: Enterprise Assurance</div>
            <div style={{ fontSize: "1.75rem", fontWeight: 700, margin: "10px 0 4px 0", color: "#FFFFFF" }}>$25,000 <span style={{ fontSize: "0.8rem", fontWeight: 400, color: "#737373" }}>/ year</span></div>
            <p style={{ fontSize: "0.8rem", color: "#737373", margin: "0 0 16px 0" }}>
              Air-Gapped VPC • Zero Data Exfiltration SLA
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.8rem", color: "#D4D4D4" }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> 100% offline air-gapped cryptographic tokens</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Dedicated custom optimization heuristics</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Cryptographic BaselineLock SLA assurance</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Dedicated ML Infrastructure Solutions Engineer</li>
            </ul>
            <a
              href="mailto:solutions@ghostlayer.ai?subject=Enterprise%20Compute%20Assurance%20Inquiry"
              style={{
                display: "block",
                textAlign: "center",
                background: "transparent",
                border: "1px solid rgba(255, 255, 255, 0.3)",
                color: "#FFFFFF",
                padding: "12px",
                borderRadius: "0",
                fontWeight: 600,
                textDecoration: "none",
                fontSize: "0.85rem",
                letterSpacing: "0.04em",
                transition: "all 0.15s ease"
              }}
            >
              Inquire Enterprise VPC
            </a>
          </div>
        </div>

        {/* FinOps Algorithmic Disclaimer Footnote */}
        <div style={{ marginTop: "28px", padding: "14px 16px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.06)", fontSize: "0.68rem", color: "#737373", lineHeight: "1.6" }}>
          <strong style={{ color: "#A3A3A3" }}>* LEGAL NOTICE &amp; FINOPS DISCLAIMER:</strong> All figures, ROI multiples, and recoverable annual spend figures shown in this calculator are algorithmic heuristic simulations based on user-provided slider inputs and public cloud pricing models. They do not constitute an audited financial assessment, a binding monetary commitment, or a guarantee of cost reduction. GhostLayer assumes zero liability for third-party cloud compute billing variances. Review our <a href="/terms" style={{ color: "#10b981", textDecoration: "underline" }}>Terms of Service</a> and <a href="/legal" style={{ color: "#10b981", textDecoration: "underline" }}>Legal Disclaimers</a>.
        </div>
      </div>
    </div>
  );
}
