import { useState } from "react";
import {
  Activity,
  ArrowRight,
  Calculator,
  Code2,
  Cpu,
  FileCheck,
  ShieldCheck,
} from "lucide-react";

interface ProductOverviewProps {
  onOpenCalculator: () => void;
}

export default function ProductOverview({ onOpenCalculator }: ProductOverviewProps) {
  const [activeCodeTab, setActiveCodeTab] = useState<"standard" | "ghostlayer">("ghostlayer");
  const [activeStage, setActiveStage] = useState<number>(0);

  const stages = [
    {
      step: "01",
      name: "Non-Invasive Observation",
      badge: "GhostWatcherHook",
      badgeColor: "#10b981",
      icon: Activity,
      summary: "Captures step-level GPU telemetry without modifying model code or weights.",
      detail:
        "Attaches as a lightweight PyTorch context manager. Ingests SM compute utilization, HBM memory bandwidth, VRAM allocation peaks, DataLoader queue latency, and loss trajectory at every training step with zero overhead.",
      signals: ["Step Time (ms)", "VRAM Allocated (GB)", "DataLoader Latency (ms)", "Loss Trajectory"],
    },
    {
      step: "02",
      name: "Automated Diagnostic Engine",
      badge: "8 Kernel Interventions",
      badgeColor: "#60a5fa",
      icon: Cpu,
      summary: "Identifies compute, memory, and I/O bottlenecks across the execution graph.",
      detail:
        "Analyzes forward and backward execution passes against proven interventions: FlashAttention-2, mixed precision AMP, gradient checkpointing, fused AdamW, DataLoader prefetching, and tensor parallel sharding.",
      signals: ["FlashAttention-2", "FP16 / BF16 AMP", "Activation Checkpointing", "Fused Optimizers"],
    },
    {
      step: "03",
      name: "Loss-Shift Safety Verifier",
      badge: "ΔL ≤ 0.10 Hard Gate",
      badgeColor: "#f59e0b",
      icon: ShieldCheck,
      summary: "Enforces mathematical convergence before committing any optimization.",
      detail:
        "Proposed candidate optimizations must pass the 0.10 loss-shift proxy threshold against the baseline trajectory. If numerical divergence or gradient explosion occurs, GhostLayer revokes the candidate with zero training pipeline stalls.",
      signals: ["Baseline Trajectory", "Optimized Trajectory", "Divergence Delta", "Zero-Stall Rollback"],
    },
    {
      step: "04",
      name: "Immutable Audit Receipts",
      badge: "Verifiable Decision Record",
      badgeColor: "#a855f7",
      icon: FileCheck,
      summary: "Produces verifiable JSON and HTML audit receipts documenting exact dollar recovery.",
      detail:
        "Records the complete lifecycle event sequence: OBSERVE → DIAGNOSE → APPLY → VERIFY → ROLLBACK. Every run produces an auditable receipt with step throughput deltas, VRAM reclaimed, and realized GPU-hour savings.",
      signals: ["Hardware Receipts", "Dollar Payback", "Replay Memory", "Executive Export"],
    },
  ];

  return (
    <section className="product-overview-section" id="product-overview" aria-label="GhostLayer Product Architecture">
      <div className="product-overview-container">
        
        {/* Section Header */}
        <div className="product-overview-header">
          <div className="product-overview-header__copy">
            <p className="orbit-eyebrow">
              <span /> ARCHITECTURAL FOUNDATION &amp; PRODUCT LOOP
            </p>
            <h2 className="product-overview-title">
              How GhostLayer Operates Across Your PyTorch Workloads.
            </h2>
            <p className="product-overview-lede">
              GhostLayer is not a cloud broker or passive telemetry dashboard. It is an active decision and audit layer built as a non-invasive PyTorch context hook that diagnoses execution bottlenecks, enforces mathematical convergence safety, and produces immutable audit records.
            </p>
          </div>
          <div className="product-overview-header__cta">
            <button
              onClick={onOpenCalculator}
              className="product-overview-calc-btn"
              aria-label="Open Interactive FinOps Engine on 3D Globe"
            >
              <Calculator size={15} />
              <span>Open FinOps Engine on Globe</span>
              <ArrowRight size={14} />
            </button>
            <small style={{ color: "#737373", fontFamily: '"DM Mono", monospace', fontSize: "0.62rem" }}>
              Explore GPU ROI modeling on the 3D globe node
            </small>
          </div>
        </div>

        {/* 4-Stage Operational Grid */}
        <div className="product-stages-grid">
          {stages.map((stage, idx) => {
            const Icon = stage.icon;
            const isSelected = activeStage === idx;
            return (
              <div
                key={stage.step}
                className={`product-stage-card ${isSelected ? "is-active" : ""}`}
                onClick={() => setActiveStage(idx)}
              >
                <div className="product-stage-card__top">
                  <span className="product-stage-card__number">{stage.step}</span>
                  <span
                    className="product-stage-card__badge"
                    style={{ borderColor: stage.badgeColor, color: stage.badgeColor }}
                  >
                    {stage.badge}
                  </span>
                </div>
                <div className="product-stage-card__icon-wrap">
                  <Icon size={20} style={{ color: isSelected ? "#FFFFFF" : "#A3A3A3" }} />
                  <h3 className="product-stage-card__title">{stage.name}</h3>
                </div>
                <p className="product-stage-card__summary">{stage.summary}</p>
                <p className="product-stage-card__detail">{stage.detail}</p>
                <div className="product-stage-card__signals">
                  {stage.signals.map((sig) => (
                    <span key={sig} className="product-stage-card__tag">
                      {sig}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Interactive Code & Architecture Ingestion Box */}
        <div className="product-code-showcase">
          <div className="product-code-header">
            <div className="product-code-title">
              <Code2 size={16} style={{ color: "#10b981" }} />
              <span>INTEGRATION SURFACE: NON-INVASIVE PYTORCH CONTEXT HOOK</span>
            </div>
            <div className="product-code-tabs">
              <button
                className={`product-code-tab ${activeCodeTab === "standard" ? "is-active" : ""}`}
                onClick={() => setActiveCodeTab("standard")}
              >
                Standard PyTorch (Unprofiled)
              </button>
              <button
                className={`product-code-tab ${activeCodeTab === "ghostlayer" ? "is-active" : ""}`}
                onClick={() => setActiveCodeTab("ghostlayer")}
              >
                With GhostLayer Hook (1-Line Wrapper)
              </button>
            </div>
          </div>

          <div className="product-code-body">
            {activeCodeTab === "standard" ? (
              <pre className="product-code-block">
                <code>
{`# Standard PyTorch Training Loop — Unprofiled Compute Leaks (15% – 35% GPU Waste)
import torch
from torch.utils.data import DataLoader

model = MyTransformerModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=32)

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        inputs, targets = batch['input'].cuda(), batch['target'].cuda()
        
        # ⚠️ UNWATCHED: Hidden DataLoader I/O stalls (GPU idle 24% of time)
        # ⚠️ UNOPTIMIZED: Standard FP32 attention without FlashAttention-2
        # ⚠️ UNCHECKED: Memory fragmentation causing premature OOM batch ceilings
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()`}
                </code>
              </pre>
            ) : (
              <pre className="product-code-block">
                <code>
{`# GhostLayer Decision Layer — Non-Invasive Step Telemetry & Safe Optimization
import torch
from torch.utils.data import DataLoader
from ghost_layer import GhostWatcher  # ← 1-Line Non-Invasive Integration

model = MyTransformerModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=32)

# Wrap loop in GhostWatcher context — Zero model architecture changes required
with GhostWatcher(run_name="llama3-pretrain", loss_threshold=0.10) as watcher:
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            inputs, targets = batch['input'].cuda(), batch['target'].cuda()
            
            # ✅ TELEMETRY: Ingests step time, VRAM high-water mark, and DataLoader stalls
            # ✅ DIAGNOSTICS: Auto-applies FlashAttention-2 + BF16 Mixed Precision
            # ✅ SAFETY GATE: Validates loss trajectory (ΔL <= 0.10 proxy threshold)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # ✅ AUDIT RECORD: Emits structured JSON & HTML hardware decision receipts`}
                </code>
              </pre>
            )}
          </div>

          {/* Under-code telemetry highlight banner */}
          <div className="product-code-footer">
            <div className="product-code-metric">
              <span className="code-metric-label">Integration Friction</span>
              <span className="code-metric-value">0 Model Modifications</span>
            </div>
            <div className="product-code-metric">
              <span className="code-metric-label">Safety Guarantee</span>
              <span className="code-metric-value">ΔL ≤ 0.10 Hard Gate</span>
            </div>
            <div className="product-code-metric">
              <span className="code-metric-label">Measured Speedup</span>
              <span className="code-metric-value">2.91x Verified on RTX A2000</span>
            </div>
            <div className="product-code-metric">
              <span className="code-metric-label">Decision Audit</span>
              <span className="code-metric-value">Immutable JSON Receipts</span>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}
