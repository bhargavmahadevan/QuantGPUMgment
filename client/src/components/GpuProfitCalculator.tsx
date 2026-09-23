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

export interface TopologyPreset {
  id: string;
  tier: string;
  name: string;
  gpus: number;
  ratePerGpuHour: number;
  workloadProfile: string;
  tokens: string;
  steps: number;
  baseLatencyMs: number;
  optLatencyMs: number;
  latencyReductionPct: number;
  baselineSpendCampaign: number;
  optimizedSpendCampaign: number;
  grossSavingsCampaign: number;
  fee25Campaign: number;
  clientNetSavingsCampaign: number;
  baseDurationHours: number;
  optDurationHours: number;
  defaultCampaignsPerYear: number;
  description: string;
}

export const TOPOLOGY_PRESETS: TopologyPreset[] = [
  {
    id: "tier-0a",
    tier: "Tier 0A",
    name: "1x (RTX A2000 / 4090)",
    gpus: 1,
    ratePerGpuHour: 1.50,
    workloadProfile: "135M–360M params",
    tokens: "50B tokens",
    steps: 95367,
    baseLatencyMs: 1850,
    optLatencyMs: 1110,
    latencyReductionPct: 40.0,
    baselineSpendCampaign: 73.51,
    optimizedSpendCampaign: 44.11,
    grossSavingsCampaign: 29.40,
    fee25Campaign: 7.35,
    clientNetSavingsCampaign: 22.05,
    baseDurationHours: 49.0,
    optDurationHours: 29.4,
    defaultCampaignsPerYear: 24,
    description: "Single Edge Devbox • Physical reference hardware verified on local RTX A2000",
  },
  {
    id: "tier-0b",
    tier: "Tier 0B",
    name: "4x (RTX 4090 / A6000)",
    gpus: 4,
    ratePerGpuHour: 2.50,
    workloadProfile: "1B–1.4B params",
    tokens: "200B tokens",
    steps: 190734,
    baseLatencyMs: 2400,
    optLatencyMs: 1488,
    latencyReductionPct: 38.0,
    baselineSpendCampaign: 1271.56,
    optimizedSpendCampaign: 788.37,
    grossSavingsCampaign: 483.19,
    fee25Campaign: 120.80,
    clientNetSavingsCampaign: 362.39,
    baseDurationHours: 127.2,
    optDurationHours: 78.8,
    defaultCampaignsPerYear: 12,
    description: "Workstation Rig • Multi-GPU lab experimentation and rapid prototyping",
  },
  {
    id: "tier-1a",
    tier: "Tier 1A",
    name: "8x (A100 80GB SXM4)",
    gpus: 8,
    ratePerGpuHour: 3.00,
    workloadProfile: "6B–7B params",
    tokens: "1.0T tokens",
    steps: 476837,
    baseLatencyMs: 3200,
    optLatencyMs: 2080,
    latencyReductionPct: 35.0,
    baselineSpendCampaign: 10172.52,
    optimizedSpendCampaign: 6612.14,
    grossSavingsCampaign: 3560.38,
    fee25Campaign: 890.10,
    clientNetSavingsCampaign: 2670.29,
    baseDurationHours: 423.9,
    optDurationHours: 275.5,
    defaultCampaignsPerYear: 6,
    description: "Single Cloud Node • Workhorse multimodal & embedding model fine-tuning",
  },
  {
    id: "tier-1b",
    tier: "Tier 1B",
    name: "8x (H100 SXM5 80GB)",
    gpus: 8,
    ratePerGpuHour: 4.00,
    workloadProfile: "12B–13B params",
    tokens: "2.0T tokens",
    steps: 476837,
    baseLatencyMs: 3000,
    optLatencyMs: 2010,
    latencyReductionPct: 33.0,
    baselineSpendCampaign: 12715.65,
    optimizedSpendCampaign: 8519.49,
    grossSavingsCampaign: 4196.17,
    fee25Campaign: 1049.04,
    clientNetSavingsCampaign: 3147.12,
    baseDurationHours: 397.4,
    optDurationHours: 266.2,
    defaultCampaignsPerYear: 4,
    description: "High-Throughput H100 Pod • Mid-scale foundation model pretraining",
  },
  {
    id: "tier-2a",
    tier: "Tier 2A",
    name: "32x (H100 InfiniBand)",
    gpus: 32,
    ratePerGpuHour: 3.80,
    workloadProfile: "20B–30B params",
    tokens: "3.0T tokens",
    steps: 715255,
    baseLatencyMs: 3500,
    optLatencyMs: 2520,
    latencyReductionPct: 28.0,
    baselineSpendCampaign: 84559.04,
    optimizedSpendCampaign: 60882.51,
    grossSavingsCampaign: 23676.53,
    fee25Campaign: 5919.13,
    clientNetSavingsCampaign: 17757.40,
    baseDurationHours: 695.4,
    optDurationHours: 500.7,
    defaultCampaignsPerYear: 4,
    description: "Small Distributed Cluster • Multi-node FSDP & Megatron-LM tensor parallel",
  },
  {
    id: "tier-2b",
    tier: "Tier 2B",
    name: "64x (H100 Quantum-2)",
    gpus: 64,
    ratePerGpuHour: 3.75,
    workloadProfile: "65B–70B params",
    tokens: "5.0T tokens",
    steps: 1192092,
    baseLatencyMs: 4000,
    optLatencyMs: 3000,
    latencyReductionPct: 25.0,
    baselineSpendCampaign: 317891.20,
    optimizedSpendCampaign: 238418.40,
    grossSavingsCampaign: 79472.80,
    fee25Campaign: 19868.20,
    clientNetSavingsCampaign: 59604.60,
    baseDurationHours: 1324.5,
    optDurationHours: 993.4,
    defaultCampaignsPerYear: 2,
    description: "Mid-Market Fleet • High-throughput 70B parameter pretraining runs",
  },
  {
    id: "tier-3a",
    tier: "Tier 3A",
    name: "256x (H100 SuperPOD)",
    gpus: 256,
    ratePerGpuHour: 3.50,
    workloadProfile: "176B–180B params",
    tokens: "8.0T tokens",
    steps: 953674,
    baseLatencyMs: 4500,
    optLatencyMs: 3510,
    latencyReductionPct: 22.0,
    baselineSpendCampaign: 1068114.88,
    optimizedSpendCampaign: 833129.61,
    grossSavingsCampaign: 234985.27,
    fee25Campaign: 58746.32,
    clientNetSavingsCampaign: 176238.96,
    baseDurationHours: 1192.1,
    optDurationHours: 929.8,
    defaultCampaignsPerYear: 2,
    description: "Scale-Up Enterprise Pod • Enterprise foundation model pretraining facility",
  },
  {
    id: "tier-4a",
    tier: "Tier 4A",
    name: "512x (H100 SXM5)",
    gpus: 512,
    ratePerGpuHour: 3.40,
    workloadProfile: "754B MoE params",
    tokens: "12.0T tokens",
    steps: 1430511,
    baseLatencyMs: 5200,
    optLatencyMs: 4160,
    latencyReductionPct: 20.0,
    baselineSpendCampaign: 3597004.01,
    optimizedSpendCampaign: 2877603.21,
    grossSavingsCampaign: 719400.80,
    fee25Campaign: 179850.20,
    clientNetSavingsCampaign: 539550.60,
    baseDurationHours: 2066.3,
    optDurationHours: 1653.0,
    defaultCampaignsPerYear: 1,
    description: "Sovereign / Tier-1 AI Lab • Frontier sparse mixture-of-experts training",
  },
  {
    id: "tier-4b",
    tier: "Tier 4B",
    name: "1,024x (H100 / B200)",
    gpus: 1024,
    ratePerGpuHour: 3.20,
    workloadProfile: "1.6T MoE params",
    tokens: "15.0T tokens",
    steps: 894069,
    baseLatencyMs: 6000,
    optLatencyMs: 4920,
    latencyReductionPct: 18.0,
    baselineSpendCampaign: 4882808.83,
    optimizedSpendCampaign: 4003903.24,
    grossSavingsCampaign: 878905.59,
    fee25Campaign: 219726.40,
    clientNetSavingsCampaign: 659179.19,
    baseDurationHours: 1490.1,
    optDurationHours: 1221.9,
    defaultCampaignsPerYear: 1,
    description: "Hyperscaler Mega-Cluster • Massive scale frontier cluster training",
  },
];

export default function GpuProfitCalculator() {
  const [calculatorMode, setCalculatorMode] = useState<"campaign" | "fleet" | "subruns">("campaign");
  const [selectedPresetId, setSelectedPresetId] = useState<string>("tier-2b");
  const [campaignCountMultiplier, setCampaignCountMultiplier] = useState<number>(2);
  const [customGpus, setCustomGpus] = useState<number>(64);
  const [customRate, setCustomRate] = useState<number>(3.75);
  const [customTokens, setCustomTokens] = useState<number>(5.0); // Trillion tokens
  const [customBatchTokensExp, setCustomBatchTokensExp] = useState<number>(22); // 2^22 tokens/batch
  const [customBaseLatencyMs, setCustomBaseLatencyMs] = useState<number>(4000);
  const [customOptLatencyMs, setCustomOptLatencyMs] = useState<number>(3000);
  const [monthlyHours, setMonthlyHours] = useState<number>(500);
  const [ioStallPct, setIoStallPct] = useState<number>(18);
  const [precisionWastePct, setPrecisionWastePct] = useState<number>(15);
  const [copied, setCopied] = useState<boolean>(false);

  const isCustom = selectedPresetId === "custom";

  const activePreset = useMemo(() => {
    return TOPOLOGY_PRESETS.find((p) => p.id === selectedPresetId);
  }, [selectedPresetId]);

  // Derived Tier & Campaign Metrics
  const tierGpus = isCustom ? customGpus : activePreset ? activePreset.gpus : 64;
  const tierRate = isCustom ? customRate : activePreset ? activePreset.ratePerGpuHour : 3.75;
  const tierBaseLatency = isCustom ? customBaseLatencyMs : activePreset ? activePreset.baseLatencyMs : 4000;
  const tierOptLatency = isCustom ? customOptLatencyMs : activePreset ? activePreset.optLatencyMs : 3000;

  // Exact ROICalculator formula:
  // steps = tokens / (2^batchExp)
  const tierSteps = useMemo(() => {
    if (!isCustom && activePreset) return activePreset.steps;
    const tokensTotal = customTokens * 1e12;
    const batchSize = Math.pow(2, customBatchTokensExp);
    return Math.floor(tokensTotal / batchSize);
  }, [isCustom, activePreset, customTokens, customBatchTokensExp]);

  const campaignMetrics = useMemo(() => {
    const stepsToHours = tierSteps / 3600000;
    const gpuHoursMultiplier = stepsToHours * tierGpus;
    const baseGpuHours = (tierBaseLatency * gpuHoursMultiplier);
    const optGpuHours = (tierOptLatency * gpuHoursMultiplier);
    const savedGpuHours = Math.max(0, baseGpuHours - optGpuHours);

    const baseSpend = baseGpuHours * tierRate;
    const optSpend = optGpuHours * tierRate;
    const grossSavings = Math.max(0, baseSpend - optSpend);
    const fee25 = grossSavings * 0.25;
    const clientNet = grossSavings * 0.75;

    const baseWallHours = baseGpuHours / Math.max(1, tierGpus);
    const optWallHours = optGpuHours / Math.max(1, tierGpus);
    const daysSaved = (baseWallHours - optWallHours) / 24;

    const annualGrossSavings = grossSavings * campaignCountMultiplier;
    const annualFee = fee25 * campaignCountMultiplier;
    const annualClientNet = clientNet * campaignCountMultiplier;

    return {
      tierSteps,
      baseGpuHours,
      optGpuHours,
      savedGpuHours,
      baseSpend,
      optSpend,
      grossSavings,
      fee25,
      clientNet,
      baseWallHours,
      optWallHours,
      daysSaved,
      annualGrossSavings,
      annualFee,
      annualClientNet,
    };
  }, [tierSteps, tierGpus, tierRate, tierBaseLatency, tierOptLatency, campaignCountMultiplier]);

  // Continuous Fleet Metrics
  const clusterHourlySpend = useMemo(() => {
    return tierGpus * tierRate;
  }, [tierGpus, tierRate]);

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
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px", marginBottom: "28px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", color: "#A3A3A3", fontWeight: 700, fontSize: "0.75rem", letterSpacing: "0.08em", textTransform: "uppercase", fontFamily: '"DM Mono", monospace' }}>
            <Calculator size={14} />
            <span>Interactive FinOps Engine • Mathematical Specification</span>
          </div>
          <h2 style={{ fontSize: "2.1rem", fontWeight: 400, margin: "8px 0 0 0", color: "#FFFFFF", fontFamily: '"Instrument Serif", Georgia, serif', letterSpacing: "-0.02em" }}>
            GPU Compute Waste & Enterprise Scaling Engine
          </h2>
          <p style={{ color: "#8E8E93", fontSize: "0.88rem", margin: "6px 0 0 0", maxWidth: "780px", lineHeight: 1.6, fontFamily: 'Manrope, sans-serif' }}>
            Direct mathematical projections derived from <code style={{ color: "#E5E5EA", fontFamily: '"DM Mono", monospace' }}>ROICalculator</code> and the 10-tier cluster taxonomy. Model qualifying pretraining campaigns, continuous fleet waste, and multi-stage sub-run multipliers below.
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
              fontFamily: '"DM Mono", monospace',
              transition: "all 0.15s ease"
            }}
          >
            {copied ? <Check size={12} /> : <Copy size={12} />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>

      {/* Mode Navigation Tabs */}
      <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid rgba(255, 255, 255, 0.1)", paddingBottom: "12px", marginBottom: "28px" }}>
        <button
          onClick={() => setCalculatorMode("campaign")}
          style={{
            background: calculatorMode === "campaign" ? "rgba(255, 255, 255, 0.12)" : "transparent",
            border: `1px solid ${calculatorMode === "campaign" ? "#FFFFFF" : "rgba(255, 255, 255, 0.15)"}`,
            color: calculatorMode === "campaign" ? "#FFFFFF" : "#A3A3A3",
            padding: "8px 18px",
            fontSize: "0.8rem",
            fontWeight: 700,
            cursor: "pointer",
            fontFamily: '"DM Mono", monospace',
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <TrendingUp size={14} />
          <span>1. Enterprise Campaign Scaling (Tiers 0A–4B)</span>
        </button>

        <button
          onClick={() => setCalculatorMode("fleet")}
          style={{
            background: calculatorMode === "fleet" ? "rgba(255, 255, 255, 0.12)" : "transparent",
            border: `1px solid ${calculatorMode === "fleet" ? "#FFFFFF" : "rgba(255, 255, 255, 0.15)"}`,
            color: calculatorMode === "fleet" ? "#FFFFFF" : "#A3A3A3",
            padding: "8px 18px",
            fontSize: "0.8rem",
            fontWeight: 700,
            cursor: "pointer",
            fontFamily: '"DM Mono", monospace',
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <Server size={14} />
          <span>2. Continuous Fleet &amp; Pre-Flight Audit</span>
        </button>

        <button
          onClick={() => setCalculatorMode("subruns")}
          style={{
            background: calculatorMode === "subruns" ? "rgba(255, 255, 255, 0.12)" : "transparent",
            border: `1px solid ${calculatorMode === "subruns" ? "#FFFFFF" : "rgba(255, 255, 255, 0.15)"}`,
            color: calculatorMode === "subruns" ? "#FFFFFF" : "#A3A3A3",
            padding: "8px 18px",
            fontSize: "0.8rem",
            fontWeight: 700,
            cursor: "pointer",
            fontFamily: '"DM Mono", monospace',
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <Layers size={14} />
          <span>3. Mega-Cluster Sub-Run Multipliers</span>
        </button>
      </div>

      {/* MODE 1: ENTERPRISE CAMPAIGN SCALING */}
      {calculatorMode === "campaign" && (
        <div>
          {/* Cluster Topology Preset Selector */}
          <div style={{ marginBottom: "24px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
              <label style={{ fontSize: "0.75rem", color: "#8E8E93", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontFamily: '"DM Mono", monospace' }}>
                Select Enterprise Cluster Architecture
              </label>
              <span style={{ fontSize: "0.72rem", color: "#737373", fontFamily: '"DM Mono", monospace' }}>
                Modeled 25% Performance Fee • 75% Client Retained Value
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "8px" }}>
              {TOPOLOGY_PRESETS.map((p) => {
                const isSelected = selectedPresetId === p.id;
                return (
                  <button
                    key={p.id}
                    onClick={() => setSelectedPresetId(p.id)}
                    style={{
                      background: isSelected ? "#1A1A1A" : "#111111",
                      border: `1px solid ${isSelected ? "#FFFFFF" : "rgba(255, 255, 255, 0.1)"}`,
                      padding: "12px",
                      textAlign: "left",
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                      color: "#FFFFFF",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: "0.68rem", color: isSelected ? "#FFFFFF" : "#8E8E93", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>{p.tier}</span>
                      <span style={{ fontSize: "0.65rem", color: "#10b981", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>-{p.latencyReductionPct}%</span>
                    </div>
                    <div style={{ fontWeight: 700, fontSize: "0.82rem", color: isSelected ? "#FFFFFF" : "#D4D4D4", marginTop: "3px", fontFamily: 'Manrope, sans-serif' }}>{p.name}</div>
                    <div style={{ fontSize: "0.72rem", color: "#737373", marginTop: "3px", fontFamily: '"DM Mono", monospace' }}>{p.tokens} • {p.gpus}x GPUs</div>
                  </button>
                );
              })}

              <button
                onClick={() => setSelectedPresetId("custom")}
                style={{
                  background: isCustom ? "#1A1A1A" : "#111111",
                  border: `1px solid ${isCustom ? "#FFFFFF" : "rgba(255, 255, 255, 0.1)"}`,
                  padding: "12px",
                  textAlign: "left",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                  color: "#FFFFFF",
                }}
              >
                <div style={{ fontSize: "0.68rem", color: isCustom ? "#FFFFFF" : "#8E8E93", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Custom</div>
                <div style={{ fontWeight: 700, fontSize: "0.82rem", color: isCustom ? "#FFFFFF" : "#D4D4D4", marginTop: "3px", fontFamily: 'Manrope, sans-serif' }}>Custom Fleet</div>
                <div style={{ fontSize: "0.72rem", color: "#737373", marginTop: "3px", fontFamily: '"DM Mono", monospace' }}>Custom Tokens &amp; GPUs</div>
              </button>
            </div>
          </div>

          {/* Active Preset Workload Banner */}
          <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", padding: "16px 20px", marginBottom: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
            <div>
              <span style={{ fontSize: "0.7rem", color: "#8E8E93", fontWeight: 700, textTransform: "uppercase", fontFamily: '"DM Mono", monospace' }}>Workload Contract: </span>
              <span style={{ fontSize: "0.85rem", color: "#FFFFFF", fontWeight: 600, fontFamily: 'Manrope, sans-serif' }}>
                {isCustom ? `Custom ${customGpus}x GPUs @ $${customRate}/hr • ${customTokens}T Tokens` : `${activePreset?.tier}: ${activePreset?.workloadProfile} (${activePreset?.tokens}, ${activePreset?.steps.toLocaleString()} steps)`}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
              <span style={{ fontSize: "0.75rem", color: "#8E8E93", fontFamily: '"DM Mono", monospace' }}>
                Annual Campaigns:
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                {[1, 2, 4, 6, 12].map((cnt) => (
                  <button
                    key={cnt}
                    onClick={() => setCampaignCountMultiplier(cnt)}
                    style={{
                      background: campaignCountMultiplier === cnt ? "#FFFFFF" : "rgba(255, 255, 255, 0.06)",
                      color: campaignCountMultiplier === cnt ? "#000000" : "#FFFFFF",
                      border: "1px solid rgba(255, 255, 255, 0.2)",
                      padding: "2px 8px",
                      fontSize: "0.72rem",
                      fontWeight: 700,
                      cursor: "pointer",
                      fontFamily: '"DM Mono", monospace',
                    }}
                  >
                    {cnt}x/yr
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Campaign Financial & Wall-Clock KPI Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px", marginBottom: "28px" }}>
            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", padding: "20px" }}>
              <div style={{ fontSize: "0.72rem", color: "#8E8E93", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Qualifying Campaign Spend</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 700, color: "#FFFFFF", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>
                ${campaignMetrics.baseSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                <span style={{ fontSize: "0.85rem", color: "#737373", fontWeight: 400 }}> &rarr; ${campaignMetrics.optSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "#8E8E93", marginTop: "4px", fontFamily: 'Manrope, sans-serif' }}>
                {campaignMetrics.tierSteps.toLocaleString()} steps @ ${(tierGpus * tierRate).toFixed(2)}/hr cluster rate
              </div>
            </div>

            <div style={{ background: "rgba(16, 185, 129, 0.05)", border: "1px solid rgba(16, 185, 129, 0.35)", padding: "20px" }}>
              <div style={{ fontSize: "0.72rem", color: "#34d399", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Gross Modeled Savings</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 700, color: "#10b981", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>
                ${campaignMetrics.grossSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
              </div>
              <div style={{ fontSize: "0.75rem", color: "#34d399", marginTop: "4px", fontFamily: 'Manrope, sans-serif' }}>
                Per qualifying campaign (${campaignMetrics.annualGrossSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}/yr modeled)
              </div>
            </div>

            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", padding: "20px" }}>
              <div style={{ fontSize: "0.72rem", color: "#8E8E93", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Wall-Clock Duration Reduction</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 700, color: "#FFFFFF", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>
                {(campaignMetrics.optWallHours / 24).toFixed(1)} <span style={{ fontSize: "0.85rem", color: "#737373", fontWeight: 400 }}>days (was {(campaignMetrics.baseWallHours / 24).toFixed(1)} d)</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "#10b981", marginTop: "4px", fontFamily: 'Manrope, sans-serif' }}>
                +{campaignMetrics.daysSaved.toFixed(1)} Days Cluster Headroom Unlocked
              </div>
            </div>

            <div style={{ background: "#111111", border: "1px solid #FFFFFF", padding: "20px" }}>
              <div style={{ fontSize: "0.72rem", color: "#A3A3A3", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Value Sharing Breakdown (25/75)</div>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#FFFFFF", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>
                Client: <span style={{ color: "#10b981" }}>${campaignMetrics.clientNet.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</span>
              </div>
              <div style={{ fontSize: "0.82rem", color: "#A3A3A3", marginTop: "4px", fontFamily: '"DM Mono", monospace' }}>
                Fee (25%): ${campaignMetrics.fee25.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MODE 2: CONTINUOUS FLEET & PRE-FLIGHT AUDIT */}
      {calculatorMode === "fleet" && (
        <div>
          {/* Sliders Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "24px", marginBottom: "28px", background: "#111111", padding: "20px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontFamily: '"DM Mono", monospace' }}>
                <span style={{ fontSize: "0.78rem", color: "#8E8E93" }}>Active GPU Fleet Count</span>
                <span style={{ fontWeight: 700, color: "#FFFFFF" }}>{tierGpus} GPUs</span>
              </div>
              <input
                type="range"
                min={1}
                max={1024}
                value={tierGpus}
                onChange={(e) => {
                  setSelectedPresetId("custom");
                  setCustomGpus(Number(e.target.value));
                }}
                style={{ width: "100%", accentColor: "#FFFFFF" }}
              />
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontFamily: '"DM Mono", monospace' }}>
                <span style={{ fontSize: "0.78rem", color: "#8E8E93" }}>Hourly Rate ($/GPU-hr)</span>
                <span style={{ fontWeight: 700, color: "#FFFFFF" }}>${tierRate.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.50}
                max={8.00}
                step={0.05}
                value={tierRate}
                onChange={(e) => {
                  setSelectedPresetId("custom");
                  setCustomRate(Number(e.target.value));
                }}
                style={{ width: "100%", accentColor: "#FFFFFF" }}
              />
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontFamily: '"DM Mono", monospace' }}>
                <span style={{ fontSize: "0.78rem", color: "#8E8E93" }}>Monthly Training Hours</span>
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
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontFamily: '"DM Mono", monospace' }}>
                <span style={{ fontSize: "0.78rem", color: "#8E8E93" }}>DataLoader I/O Starvation</span>
                <span style={{ fontWeight: 700, color: "#f43f5e" }}>{ioStallPct}% idle wait</span>
              </div>
              <input
                type="range"
                min={5}
                max={40}
                value={ioStallPct}
                onChange={(e) => setIoStallPct(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#f43f5e" }}
              />
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontFamily: '"DM Mono", monospace' }}>
                <span style={{ fontSize: "0.78rem", color: "#8E8E93" }}>Suboptimal Precision &amp; Kernels</span>
                <span style={{ fontWeight: 700, color: "#fb7185" }}>+{precisionWastePct}% latency</span>
              </div>
              <input
                type="range"
                min={0}
                max={30}
                value={precisionWastePct}
                onChange={(e) => setPrecisionWastePct(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#fb7185" }}
              />
            </div>
          </div>

          {/* Continuous Fleet KPI Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px", marginBottom: "28px" }}>
            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", padding: "18px" }}>
              <div style={{ fontSize: "0.75rem", color: "#8E8E93", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Monthly Baseline Spend</div>
              <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#FFFFFF", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>${monthlyBaselineSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
              <div style={{ fontSize: "0.75rem", color: "#8E8E93", marginTop: "4px", fontFamily: 'Manrope, sans-serif' }}>${clusterHourlySpend.toFixed(2)}/hr across {tierGpus} GPUs</div>
            </div>

            <div style={{ background: "rgba(244, 63, 94, 0.05)", border: "1px solid rgba(244, 63, 94, 0.3)", padding: "18px" }}>
              <div style={{ fontSize: "0.75rem", color: "#fb7185", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Identified Monthly Leakage</div>
              <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#f43f5e", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>${monthlyWastedSpend.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
              <div>
                <span style={{ display: "inline-flex", padding: "2px 8px", background: "rgba(244, 63, 94, 0.15)", color: "#f43f5e", border: "1px solid rgba(244, 63, 94, 0.35)", fontSize: "0.68rem", fontWeight: 700, marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>
                  {totalWastePct}% COMPUTE LEAKAGE
                </span>
              </div>
            </div>

            <div style={{ background: "rgba(16, 185, 129, 0.05)", border: "1px solid rgba(16, 185, 129, 0.35)", padding: "18px" }}>
              <div style={{ fontSize: "0.75rem", color: "#34d399", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>Recoverable Annual Runway</div>
              <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#10b981", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>${annualRecoverableSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
              <div>
                <span style={{ display: "inline-flex", padding: "2px 8px", background: "rgba(16, 185, 129, 0.15)", color: "#10b981", border: "1px solid rgba(16, 185, 129, 0.35)", fontSize: "0.68rem", fontWeight: 700, marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>
                  +${monthlyRecoverableSavings.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}/mo NET RUNWAY
                </span>
              </div>
            </div>

            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", padding: "18px" }}>
              <div style={{ fontSize: "0.75rem", color: "#8E8E93", textTransform: "uppercase", fontWeight: 700, fontFamily: '"DM Mono", monospace' }}>$2,500 Audit Payback</div>
              <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "#FFFFFF", marginTop: "6px", fontFamily: '"DM Mono", monospace' }}>{auditPaybackDays} Days</div>
              <div style={{ fontSize: "0.75rem", color: "#8E8E93", marginTop: "4px", fontFamily: 'Manrope, sans-serif' }}>{auditRoiMultiple}x 1-Year ROI Multiple</div>
            </div>
          </div>
        </div>
      )}

      {/* MODE 3: SUB-RUN MULTIPLIERS */}
      {calculatorMode === "subruns" && (
        <div style={{ marginBottom: "28px" }}>
          <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.08)", padding: "20px", marginBottom: "20px" }}>
            <h3 style={{ fontSize: "1.35rem", fontWeight: 400, margin: "0 0 8px 0", color: "#FFFFFF", fontFamily: '"Instrument Serif", Georgia, serif' }}>
              Why Enterprise Mega-Clusters Multiply GhostLayer Savings
            </h3>
            <p style={{ fontSize: "0.85rem", color: "#8E8E93", margin: 0, lineHeight: 1.6, fontFamily: 'Manrope, sans-serif' }}>
              In frontier AI labs operating 256 to 1,024+ GPUs (Tiers 3A–4B), clusters do not sit idle between annual pretraining milestones. They execute continuous auxiliary sub-runs across data mixture sweeps, MoE routing ablations, and reasoning post-training that multiply optimization impact across the engineering lifecycle.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "16px" }}>
            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.1)", padding: "18px" }}>
              <div style={{ fontSize: "0.7rem", color: "#10b981", fontWeight: 700, textTransform: "uppercase", fontFamily: '"DM Mono", monospace' }}>64–256 GPUs • 50B–200B Tokens</div>
              <h4 style={{ fontSize: "1.1rem", margin: "6px 0 8px 0", color: "#FFFFFF", fontFamily: '"Instrument Serif", Georgia, serif' }}>1. Data Mixture Sweeps</h4>
              <p style={{ fontSize: "0.8rem", color: "#8E8E93", lineHeight: 1.5, margin: 0, fontFamily: 'Manrope, sans-serif' }}>
                Frequent dataset switches cause worker thread starvation and un-pinned memory buffer delays. GhostLayer surfaces DataLoader stalls before scaling to flagship runs.
              </p>
            </div>

            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.1)", padding: "18px" }}>
              <div style={{ fontSize: "0.7rem", color: "#10b981", fontWeight: 700, textTransform: "uppercase", fontFamily: '"DM Mono", monospace' }}>128–256 GPUs • Small Batch</div>
              <h4 style={{ fontSize: "1.1rem", margin: "6px 0 8px 0", color: "#FFFFFF", fontFamily: '"Instrument Serif", Georgia, serif' }}>2. Architecture &amp; MoE Sweeps</h4>
              <p style={{ fontSize: "0.8rem", color: "#8E8E93", lineHeight: 1.5, margin: 0, fontFamily: 'Manrope, sans-serif' }}>
                Top-2/top-4 expert routing causes all-to-all communication drag. Step-level telemetry isolates inter-node dispatch bottlenecks from raw kernel compute.
              </p>
            </div>

            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.1)", padding: "18px" }}>
              <div style={{ fontSize: "0.7rem", color: "#10b981", fontWeight: 700, textTransform: "uppercase", fontFamily: '"DM Mono", monospace' }}>256–512 GPUs • 4k &rarr; 128k Context</div>
              <h4 style={{ fontSize: "1.1rem", margin: "6px 0 8px 0", color: "#FFFFFF", fontFamily: '"Instrument Serif", Georgia, serif' }}>3. Context Extension Branching</h4>
              <p style={{ fontSize: "0.8rem", color: "#8E8E93", lineHeight: 1.5, margin: 0, fontFamily: 'Manrope, sans-serif' }}>
                High sequence lengths risk OOM spikes. GhostLayer unlocks 40%–60% activation VRAM headroom via chunked cross-entropy and FlashAttention kernel fusion.
              </p>
            </div>

            <div style={{ background: "#111111", border: "1px solid rgba(255, 255, 255, 0.1)", padding: "18px" }}>
              <div style={{ fontSize: "0.7rem", color: "#10b981", fontWeight: 700, textTransform: "uppercase", fontFamily: '"DM Mono", monospace' }}>128–512 GPUs • Dynamic Lengths</div>
              <h4 style={{ fontSize: "1.1rem", margin: "6px 0 8px 0", color: "#FFFFFF", fontFamily: '"Instrument Serif", Georgia, serif' }}>4. Post-Training &amp; RL Reasoning</h4>
              <p style={{ fontSize: "0.8rem", color: "#8E8E93", lineHeight: 1.5, margin: 0, fontFamily: 'Manrope, sans-serif' }}>
                SFT, DPO, and GRPO reasoning rollouts waste compute on padded tokens. Telemetry identifies unpadded ragged tensor packing and prompt caching gains.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Commercial Conversion Matrix */}
      <div style={{ borderTop: "1px solid rgba(255, 255, 255, 0.08)", paddingTop: "28px" }}>
        <div style={{ textAlign: "center", marginBottom: "24px" }}>
          <h3 style={{ fontSize: "1.75rem", fontWeight: 400, margin: 0, fontFamily: '"Instrument Serif", Georgia, serif', letterSpacing: "-0.02em" }}>
            Commercial Deployment Tiers
          </h3>
          <p style={{ color: "#8E8E93", fontSize: "0.85rem", margin: "6px 0 0 0", fontFamily: 'Manrope, sans-serif' }}>
            Transparent pricing designed for engineering lead discretionary approval without procurement friction.
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
            <div style={{ position: "absolute", top: "-10px", right: "20px", background: "#FFFFFF", color: "#000000", padding: "2px 8px", fontSize: "0.68rem", fontWeight: 700, letterSpacing: "0.06em", fontFamily: '"DM Mono", monospace' }}>
              MOST POPULAR WEDGE
            </div>
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#FFFFFF", fontFamily: 'Manrope, sans-serif' }}>Tier 1: Pre-Flight Audit</div>
            <div style={{ fontSize: "1.75rem", fontWeight: 700, margin: "10px 0 4px 0", color: "#FFFFFF", fontFamily: '"DM Mono", monospace' }}>$2,500 <span style={{ fontSize: "0.8rem", fontWeight: 400, color: "#8E8E93" }}>Flat Fee</span></div>
            <p style={{ fontSize: "0.8rem", color: "#8E8E93", margin: "0 0 16px 0", fontFamily: 'Manrope, sans-serif' }}>
              Under $3k corporate card limit • 48-Hour turnaround • Recouped in ~{auditPaybackDays} days
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.8rem", color: "#D4D4D4", fontFamily: 'Manrope, sans-serif' }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> 1x to 8x GPU Staging Run Validation</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> DataLoader I/O stall &amp; VRAM headroom audit</li>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Executive C-level HTML &amp; Markdown Report</li>
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
                fontWeight: 700,
                textDecoration: "none",
                fontSize: "0.85rem",
                letterSpacing: "0.04em",
                fontFamily: '"DM Mono", monospace',
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
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#D4D4D4", fontFamily: 'Manrope, sans-serif' }}>Tier 2: Team Platform SaaS</div>
            <div style={{ fontSize: "1.75rem", fontWeight: 700, margin: "10px 0 4px 0", color: "#FFFFFF", fontFamily: '"DM Mono", monospace' }}>$499 <span style={{ fontSize: "0.8rem", fontWeight: 400, color: "#8E8E93" }}>/ month</span></div>
            <p style={{ fontSize: "0.8rem", color: "#8E8E93", margin: "0 0 16px 0", fontFamily: 'Manrope, sans-serif' }}>
              8 to 64 GPUs • Continuous fleet training oversight
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.8rem", color: "#D4D4D4", fontFamily: 'Manrope, sans-serif' }}>
              <li style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}><Check size={14} color="#10b981" /> Real-time Slack &amp; Discord stall alerts</li>
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
                fontWeight: 600,
                textDecoration: "none",
                fontSize: "0.85rem",
                letterSpacing: "0.04em",
                fontFamily: '"DM Mono", monospace',
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
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#D4D4D4", fontFamily: 'Manrope, sans-serif' }}>Tier 3: Enterprise Assurance</div>
            <div style={{ fontSize: "1.75rem", fontWeight: 700, margin: "10px 0 4px 0", color: "#FFFFFF", fontFamily: '"DM Mono", monospace' }}>$25,000 <span style={{ fontSize: "0.8rem", fontWeight: 400, color: "#8E8E93" }}>/ year</span></div>
            <p style={{ fontSize: "0.8rem", color: "#8E8E93", margin: "0 0 16px 0", fontFamily: 'Manrope, sans-serif' }}>
              Air-Gapped VPC • Zero Data Exfiltration SLA • Performance Sharing
            </p>
            <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", fontSize: "0.8rem", color: "#D4D4D4", fontFamily: 'Manrope, sans-serif' }}>
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
                fontWeight: 600,
                textDecoration: "none",
                fontSize: "0.85rem",
                letterSpacing: "0.04em",
                fontFamily: '"DM Mono", monospace',
                transition: "all 0.15s ease"
              }}
            >
              Inquire Enterprise VPC
            </a>
          </div>
        </div>

        {/* FinOps Algorithmic Disclaimer Footnote */}
        <div style={{ marginTop: "28px", padding: "14px 16px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.06)", fontSize: "0.72rem", color: "#8E8E93", lineHeight: "1.6", fontFamily: 'Manrope, sans-serif' }}>
          <strong style={{ color: "#E5E5EA", fontFamily: '"DM Mono", monospace' }}>* LEGAL NOTICE &amp; EMPIRICAL EVIDENCE BOUNDARY:</strong> Physical execution tested only on 1x local NVIDIA RTX A2000. All cluster-scale figures (Tier 1A to Tier 4B) represent mathematical projections and unit-test formulas (<code style={{ color: "#E5E5EA", fontFamily: '"DM Mono", monospace' }}>ROICalculator</code>), not physically measured multi-node runs. Figures shown are heuristic simulations and do not constitute a binding monetary commitment or guarantee of cost reduction. Review our <a href="/terms" style={{ color: "#10b981", textDecoration: "underline" }}>Terms of Service</a> and <a href="/legal" style={{ color: "#10b981", textDecoration: "underline" }}>Legal Disclaimers</a>.
        </div>
      </div>
    </div>
  );
}

