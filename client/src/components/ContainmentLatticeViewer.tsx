import React, { useState, useMemo } from "react";
import { 
  ShieldAlert, 
  ShieldCheck, 
  RotateCcw, 
  Flame, 
  Activity, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  DollarSign,
  Clock,
  Layers
} from "lucide-react";

interface LeakEvent {
  step: number;
  lossDelta: number;
  lossValue: number;
  wastedGpuHours: number;
  severity: "NORMAL" | "MINOR" | "MODERATE" | "SEVERE";
  energyDrift: number;
  actionTaken: "CONTINUE" | "ROLLBACK_APPLIED" | "CONTAINED_SHADOW" | "LEARNING_RATE_DAMPENED";
  rootLayer: string;
}

const SIMULATED_LEAK_HISTORY: LeakEvent[] = [
  { step: 1042, lossDelta: -0.0124, lossValue: 2.145, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.002, actionTaken: "CONTINUE", rootLayer: "Stable Normal Step" },
  { step: 1043, lossDelta: -0.0089, lossValue: 2.136, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.001, actionTaken: "CONTINUE", rootLayer: "Stable Normal Step" },
  { step: 1044, lossDelta: 0.0412, lossValue: 2.177, wastedGpuHours: 0.18, severity: "MINOR", energyDrift: 0.045, actionTaken: "CONTAINED_SHADOW", rootLayer: "Layer 14 SwiGLU Up-Proj Gradient Noise" },
  { step: 1045, lossDelta: -0.0115, lossValue: 2.165, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.003, actionTaken: "CONTINUE", rootLayer: "Recovered" },
  { step: 1046, lossDelta: -0.0094, lossValue: 2.156, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.001, actionTaken: "CONTINUE", rootLayer: "Stable Normal Step" },
  { step: 1047, lossDelta: 0.1842, lossValue: 2.340, wastedGpuHours: 1.42, severity: "SEVERE", energyDrift: 0.284, actionTaken: "ROLLBACK_APPLIED", rootLayer: "Layer 22 Multi-Head Attention QKV Overflow" },
  { step: 1048, lossDelta: -0.0185, lossValue: 2.137, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.002, actionTaken: "CONTINUE", rootLayer: "Checkpoint Rollback State Restored" },
  { step: 1049, lossDelta: -0.0078, lossValue: 2.129, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.001, actionTaken: "CONTINUE", rootLayer: "Stable Normal Step" },
  { step: 1050, lossDelta: 0.0765, lossValue: 2.205, wastedGpuHours: 0.54, severity: "MODERATE", energyDrift: 0.089, actionTaken: "LEARNING_RATE_DAMPENED", rootLayer: "Layer 31 RMSNorm Residual Spike" },
  { step: 1051, lossDelta: -0.0142, lossValue: 2.115, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.002, actionTaken: "CONTINUE", rootLayer: "Stable Normal Step" },
  { step: 1052, lossDelta: -0.0110, lossValue: 2.104, wastedGpuHours: 0.0, severity: "NORMAL", energyDrift: 0.001, actionTaken: "CONTINUE", rootLayer: "Stable Normal Step" },
];

export function ContainmentLatticeViewer() {
  const [selectedStep, setSelectedStep] = useState<LeakEvent>(SIMULATED_LEAK_HISTORY[5]); // Step 1047 (Severe)
  const [filterSeverity, setFilterSeverity] = useState<string>("ALL");

  const containmentSummary = useMemo(() => {
    const totalWastedHours = SIMULATED_LEAK_HISTORY.reduce((acc, s) => acc + s.wastedGpuHours, 0);
    const hourlyCost = 3.85; // Standard H100 cloud GPU cost / hour
    const totalWastedCost = totalWastedHours * hourlyCost * 8; // 8x GPU cluster
    const rollbacksTriggered = SIMULATED_LEAK_HISTORY.filter(s => s.actionTaken === "ROLLBACK_APPLIED").length;
    const containedDampens = SIMULATED_LEAK_HISTORY.filter(s => s.actionTaken === "LEARNING_RATE_DAMPENED" || s.actionTaken === "CONTAINED_SHADOW").length;

    return {
      totalWastedHours: totalWastedHours.toFixed(2),
      totalWastedCostUsd: totalWastedCost.toFixed(2),
      containmentSavingsUsd: (totalWastedCost * 1.84).toFixed(2),
      rollbacksTriggered,
      containedDampens,
      leakRatePct: ((3 / SIMULATED_LEAK_HISTORY.length) * 100).toFixed(1)
    };
  }, []);

  const filteredSteps = useMemo(() => {
    if (filterSeverity === "ALL") return SIMULATED_LEAK_HISTORY;
    return SIMULATED_LEAK_HISTORY.filter(s => s.severity === filterSeverity);
  }, [filterSeverity]);

  return (
    <div className="containment-container">
      {/* Top Banner & Control Overview */}
      <div className="containment-header">
        <div className="containment-header__left">
          <div className="containment-badge">
            <ShieldAlert size={14} className="text-zinc-300" />
            <span>VARIANCE CONTAINMENT & ROLLBACK LATTICE</span>
          </div>
          <span className="containment-stat">Leak Prevention: <strong className="text-white">${containmentSummary.containmentSavingsUsd} Saved</strong></span>
          <span className="containment-stat">Wasted Compute Captured: <strong>{containmentSummary.totalWastedHours} GPU-hrs</strong></span>
        </div>

        <div className="containment-header__right">
          <div className="filter-pill-group">
            {["ALL", "SEVERE", "MODERATE", "MINOR", "NORMAL"].map(sev => (
              <button
                key={sev}
                className={`filter-btn ${filterSeverity === sev ? "is-active" : ""}`}
                onClick={() => setFilterSeverity(sev)}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Grid: Step Sequence Lattice + Root-Cause Inspector */}
      <div className="containment-grid">
        {/* Step Sequence Lattice */}
        <div className="lattice-flow-stage">
          <div className="lattice-timeline-header">
            <span>TRAINING STEP SEQUENCE & GRADIENT STABILITY LATTICE</span>
            <small>Red: Divergence Trigger | Yellow: Variance Leak | Green: Stable Step</small>
          </div>

          <div className="step-cards-track">
            {filteredSteps.map((step) => {
              const isSelected = selectedStep.step === step.step;
              const isRegressive = step.lossDelta > 0;

              return (
                <div 
                  key={step.step}
                  className={`step-card step-card--${step.severity.toLowerCase()} ${isSelected ? "is-selected" : ""}`}
                  onClick={() => setSelectedStep(step)}
                >
                  <div className="step-card__top">
                    <span className="step-number">STEP {step.step}</span>
                    <span className={`severity-tag tag--${step.severity.toLowerCase()}`}>
                      {step.severity}
                    </span>
                  </div>

                  <div className="step-card__loss">
                    <small>LOSS VALUE</small>
                    <div className="loss-delta-row">
                      <strong>{step.lossValue.toFixed(3)}</strong>
                      <span className={isRegressive ? "text-zinc-400" : "text-zinc-100"}>
                        {step.lossDelta > 0 ? `+${step.lossDelta.toFixed(4)}` : step.lossDelta.toFixed(4)}
                      </span>
                    </div>
                  </div>

                  <div className="step-card__meta">
                    <div>
                      <small>HAMILTONIAN DRIFT</small>
                      <span>ΔH: {step.energyDrift.toFixed(3)}</span>
                    </div>
                    <div>
                      <small>ACTION</small>
                      <strong className="action-text">{step.actionTaken.replace(/_/g, " ")}</strong>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Loss Trajectory Visual Track */}
          <div className="loss-trajectory-bar">
            <div className="trajectory-title">
              <Activity size={13} className="text-zinc-300" />
              <span>LOSS CURVATURE & THRESHOLD BOUND (0.10 LOSS-SHIFT GATE)</span>
            </div>
            <div className="trajectory-sparkline">
              {SIMULATED_LEAK_HISTORY.map((s, idx) => (
                <div key={idx} className="sparkline-bar-wrapper" title={`Step ${s.step}: Loss ${s.lossValue}`}>
                  <div 
                    className={`sparkline-bar ${s.severity === "SEVERE" ? "bar--severe" : s.severity === "MODERATE" ? "bar--moderate" : s.severity === "MINOR" ? "bar--minor" : "bar--normal"}`}
                    style={{ height: `${Math.max(15, Math.min(100, (s.lossValue - 2.0) * 280))}%` }}
                  />
                  <span className="sparkline-label">{s.step}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Step Inspector & Rollback Root-Cause */}
        <div className="containment-inspector">
          <div className="inspector-top">
            <span className="inspector-eyebrow">STEP {selectedStep.step} DIVERGENCE AUDIT</span>
            <h3>{selectedStep.severity} VARIANCE EVENT</h3>
            <p className="root-layer-text">Root Attribution: <strong>{selectedStep.rootLayer}</strong></p>
          </div>

          <div className="containment-metric-grid">
            <div className="c-metric-card">
              <span>Loss Delta (ΔL)</span>
              <strong className={selectedStep.lossDelta > 0 ? "text-zinc-400" : "text-zinc-100"}>
                {selectedStep.lossDelta > 0 ? `+${selectedStep.lossDelta.toFixed(4)}` : selectedStep.lossDelta.toFixed(4)}
              </strong>
              <small>{selectedStep.lossDelta > 0.10 ? "Exceeded 0.10 Gate" : "Within Policy Limit"}</small>
            </div>
            <div className="c-metric-card">
              <span>Wasted GPU-Hours</span>
              <strong>{selectedStep.wastedGpuHours} hrs</strong>
              <small className="text-zinc-400">Burned Compute</small>
            </div>
            <div className="c-metric-card">
              <span>Symplectic Drift (ΔH)</span>
              <strong>{selectedStep.energyDrift.toFixed(3)}</strong>
              <small className="text-zinc-400">Hamiltonian Invariant</small>
            </div>
            <div className="c-metric-card">
              <span>Containment Action</span>
              <strong className="text-zinc-200">{selectedStep.actionTaken.replace(/_/g, " ")}</strong>
              <small className="text-zinc-400">Zero Model Corruption</small>
            </div>
          </div>

          <div className="containment-resolution-box">
            <h4>Autonomous Containment Rationale</h4>
            {selectedStep.severity === "SEVERE" ? (
              <div className="resolution-card border-rose-500/30 bg-rose-950/20">
                <div className="res-title text-rose-300">
                  <RotateCcw size={14} /> State Restored to Checkpoint at Step {selectedStep.step - 1}
                </div>
                <p>
                  Loss surge (+{selectedStep.lossDelta.toFixed(4)}) exceeded the 0.10 loss-shift safety boundary. GhostLayer automatically revoked 
                  the regressive parameter update, flushed the divergent optimizer momentum buffers, and resumed clean training from step {selectedStep.step - 1}.
                </p>
              </div>
            ) : selectedStep.severity === "MODERATE" ? (
              <div className="resolution-card border-amber-500/30 bg-amber-950/20">
                <div className="res-title text-amber-300">
                  <TrendingDown size={14} /> Adaptive Gradient Dampening Applied
                </div>
                <p>
                  Gradient norm spike detected in {selectedStep.rootLayer}. Learning rate dynamically clipped by 20% for 3 steps, avoiding a full divergence cascade.
                </p>
              </div>
            ) : (
              <div className="resolution-card border-zinc-700 bg-zinc-900/40">
                <div className="res-title text-zinc-200">
                  <CheckCircle size={14} /> Healthy Gradient Descent Trajectory
                </div>
                <p>
                  Parameters within optimal Lyapunov stability bounds. No intervention needed. Training loop operating at peak efficiency.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
export default ContainmentLatticeViewer;
