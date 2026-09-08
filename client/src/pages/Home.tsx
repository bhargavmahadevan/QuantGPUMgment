/**
 * GhostLayer — premium infrastructure landing page.
 * Black/white/graphite only. Globe is the site, not decoration.
 * Motion.dev: UI state + layout. Anime.js: cinematic/data-driven.
 */
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "wouter";
import "@/globe-evidence.css";
import "@/authoritative-content.css";
import "@/ethereal-globe.css";
import {
  ArrowDownRight,
  ArrowUpRight,
  Check,
  CircleAlert,
  Database,
  Gauge,
  Info,
  Layers3,
  Mail,
  Network,
  Orbit,
  Play,
  RotateCcw,
  ScanSearch,
  Send,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
  Calculator,
  Cpu,
  ArrowRight,
} from "lucide-react";
import GlobeNavigator, { DESTINATIONS, type DestinationId } from "@/components/GlobeNavigator";
import ExecutionDAGViewer from "@/components/ExecutionDAGViewer";
import DistributedTopologyViewer from "@/components/DistributedTopologyViewer";
import ContainmentLatticeViewer from "@/components/ContainmentLatticeViewer";
import TelemetryMatrix from "@/components/TelemetryMatrix";
import GpuProfitCalculator from "@/components/GpuProfitCalculator";
import LiveSystemPanel from "@/components/LiveSystemPanel";
import LifecycleFlow, { type LifecyclePhase } from "@/components/LifecycleFlow";
import RoiProofTable from "@/components/RoiProofTable";

const BRAND_MARK = "/manus-storage/ghostlayer-mark_0c552889.png";

type DecisionState = "observing" | "checking" | "verified" | "blocked";

const destinationContent: Record<
  DestinationId,
  { eyebrow: string; title: string; body: string; annotation: string; icon: typeof Info; metrics: { label: string; value: string }[] }
> = {
  info: {
    eyebrow: "Information field / 01",
    title: "A control plane that watches the training loop—not just the cluster.",
    body: "GhostWatcherHook records non-invasive step telemetry from a standard PyTorch loop: GPU utilization, VRAM use, step time, data-loading time, precision settings, and available loss signals. GhostLayer uses that context to turn a symptom into a specific, explainable recommendation.",
    annotation:
      "The implemented loop begins in observation mode; it does not need a pipeline rewrite to record the first decision context.",
    icon: Info,
    metrics: [
      { label: "Integration", value: "PyTorch hook" },
      { label: "Inputs", value: "Step telemetry" },
      { label: "Output", value: "Recommendation" },
    ],
  },
  control: {
    eyebrow: "Decision laboratory / 02",
    title: "Observe → diagnose → safe-apply → verify → roll back when needed.",
    body: "The decision engine can assess mixed precision, data loading, FlashAttention, gradient checkpointing, batch scaling, operator fusion, and compiler options. A candidate that is auto-applied must pass the configured loss-trajectory check; a divergent candidate is revoked and kept as recommendation-only evidence.",
    annotation:
      "The control loop in this site is illustrative. In the source, verification compares baseline and optimized loss trajectories before an outcome is retained.",
    icon: Zap,
    metrics: [
      { label: "Decision surface", value: "Training runtime" },
      { label: "Safety gate", value: "Loss trajectory" },
      { label: "Unsafe outcome", value: "Auto-apply revoked" },
    ],
  },
  memory: {
    eyebrow: "Infrastructure memory / 03",
    title: "Each decision becomes an auditable replay event—not an anecdote.",
    body: "The replay log records OBSERVE, DIAGNOSE, VERIFY, and ROLLBACK events with the step, action, reason, safety result, and throughput outcome. Verified and rejected outcomes can also feed a shared knowledge base, while ROI reporting turns a measured improvement into an audit receipt.",
    annotation:
      "The product treats the rejected path as useful memory too: the reason an intervention was blocked remains available for review.",
    icon: Database,
    metrics: [
      { label: "Replay stages", value: "4 events" },
      { label: "Evidence", value: "Safety + reason" },
      { label: "Economic view", value: "ROI receipt" },
    ],
  },
  contact: {
    eyebrow: "Contact window / 04",
    title: "Start with one training run, one bottleneck, and a measurable baseline.",
    body: "GhostLayer is designed to run inside the client environment and focus on training efficiency rather than acting as a cloud provider. Begin with an actual PyTorch workload, observe without changing it, and define the policy boundary before considering automation or a performance-share conversation.",
    annotation:
      "The project documentation is explicit: the product is not meant to copy a client's model, data, or secrets as part of this workflow.",
    icon: Mail,
    metrics: [
      { label: "Starting point", value: "1 real workload" },
      { label: "Mode", value: "Observe first" },
      { label: "Commercial model", value: "Measured savings" },
    ],
  },
};

function BrandLockup() {
  return (
    <a className="orbit-brand" href="#top" aria-label="GhostLayer home">
      <img src={BRAND_MARK} alt="" />
      <span>
        <strong>GhostLayer</strong>
        <small>Decision intelligence for compute</small>
      </span>
    </a>
  );
}

function DestinationIcon({ id, size = 15 }: { id: DestinationId; size?: number }) {
  const icons = { info: Info, control: Zap, memory: Database, contact: Mail };
  const Icon = icons[id];
  return <Icon size={size} strokeWidth={1.8} />;
}

export default function Home() {
  const [activeDestination, setActiveDestination] = useState<DestinationId | null>(null);
  const [activeTopologyTab, setActiveTopologyTab] = useState<
    "calculator" | "dag" | "dist" | "containment" | "telemetry"
  >("calculator");
  const [navigationStep, setNavigationStep] = useState(0);
  const [decisionState, setDecisionState] = useState<DecisionState>("observing");
  const [isChecking, setIsChecking] = useState(false);
  const [sectionPhase, setSectionPhase] = useState<LifecyclePhase>(null);

  const activeContent = useMemo(
    () => (activeDestination ? destinationContent[activeDestination] : null),
    [activeDestination]
  );

  function focusDestination(id: DestinationId) {
    setActiveDestination(id);
    setNavigationStep((step) => step + 1);
    if (id !== "control") setDecisionState("observing");
  }

  function returnToOrbit() {
    setActiveDestination(null);
    setNavigationStep((step) => step + 1);
    setDecisionState("observing");
  }

  function runPolicyTest(outcome: "verified" | "blocked") {
    if (isChecking) return;
    setIsChecking(true);
    setDecisionState("checking");
    window.setTimeout(() => {
      setDecisionState(outcome);
      setIsChecking(false);
    }, 820);
  }

  return (
    <div className="orbital-site" id="top">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="orbit-header">
        <BrandLockup />
        <nav className="orbit-hotbar" aria-label="Globe destinations">
          {DESTINATIONS.map((destination) => (
            <button
              key={destination.id}
              className={activeDestination === destination.id ? "is-active" : ""}
              onClick={() => focusDestination(destination.id)}
            >
              <DestinationIcon id={destination.id} />
              <span>{destination.label}</span>
            </button>
          ))}
        </nav>
        <button
          className="return-button"
          onClick={returnToOrbit}
          disabled={!activeDestination}
          aria-label="Return to globe overview"
        >
          <Orbit size={15} /> <span>Return to orbit</span>
        </button>
        <button
          className="calculator-header-button"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            background: "transparent",
            border: "1px solid rgba(255,255,255,0.18)",
            color: "#D4D4D4",
            borderRadius: "0",
            padding: "6px 14px",
            fontSize: "0.78rem",
            fontFamily: "\"DM Mono\", monospace",
            fontWeight: 500,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            cursor: "pointer",
            transition: "border-color 160ms, color 160ms",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.6)";
            (e.currentTarget as HTMLElement).style.color = "#FFFFFF";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.18)";
            (e.currentTarget as HTMLElement).style.color = "#D4D4D4";
          }}
          onClick={() => {
            setActiveTopologyTab("calculator");
            document.getElementById("topology-suite")?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          <Calculator size={13} />
          <span>ROI Calculator</span>
        </button>
      </header>

      <main className="orbit-main">
        {/* ── Hero / Globe Stage ─────────────────────────────────────────────── */}
        <section className="orbital-stage" aria-label="GhostLayer globe navigation">
          <div className="orbital-grid" aria-hidden="true" />
          <div className="stage-copy">
            <motion.div
              initial={{ opacity: 0, y: 22 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: [0.23, 1, 0.32, 1] }}
            >
              <p className="orbit-eyebrow">
                <span /> GLOBAL INFRASTRUCTURE DECISION FIELD
              </p>
              <h1>
                Make every
                <br />
                <em>GPU hour</em>
                <br />
                work harder.
              </h1>
              <p>
                GhostLayer continuously discovers, tests, and verifies GPU
                efficiency gains — without touching the training loop until
                safety is confirmed.
              </p>
            </motion.div>

            {/* Hero CTAs */}
            <motion.div
              className="hero-ctas"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, delay: 0.18, ease: [0.23, 1, 0.32, 1] }}
              style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginTop: "36px", pointerEvents: "auto" }}
            >
              <button
                className="cta-primary"
                onClick={() => focusDestination("control")}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "8px",
                  background: "#FFFFFF", color: "#000000",
                  border: "1px solid #FFFFFF", padding: "10px 20px",
                  fontFamily: "\"DM Mono\", monospace", fontSize: "0.72rem",
                  letterSpacing: "0.05em", textTransform: "uppercase", fontWeight: 600,
                  cursor: "pointer", transition: "background 160ms, color 160ms",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "#D4D4D4";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "#FFFFFF";
                }}
              >
                <Play size={12} />
                Watch the system
              </button>
              <button
                className="cta-secondary"
                onClick={() => document.getElementById("lifecycle")?.scrollIntoView({ behavior: "smooth" })}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "8px",
                  background: "transparent", color: "#FFFFFF",
                  border: "1px solid rgba(255,255,255,0.25)", padding: "10px 20px",
                  fontFamily: "\"DM Mono\", monospace", fontSize: "0.72rem",
                  letterSpacing: "0.05em", textTransform: "uppercase",
                  cursor: "pointer", transition: "border-color 160ms",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.7)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.25)";
                }}
              >
                Explore platform
                <ArrowRight size={12} />
              </button>
            </motion.div>

            {/* Live System Panel */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.5 }}
              style={{ pointerEvents: "auto", marginTop: "48px" }}
            >
              <LiveSystemPanel />
            </motion.div>

            <div className="stage-copy__legend">
              <span><i className="legend-dot legend-dot--signal" /> Verified path</span>
              <span><i className="legend-dot legend-dot--risk" /> Constraint / risk</span>
              <span><i className="legend-dot legend-dot--neutral" /> Destination</span>
            </div>
          </div>

          <div className="globe-stage">
            <GlobeNavigator
              activeDestination={activeDestination}
              navigationStep={navigationStep}
              onSelect={focusDestination}
              sectionPhase={sectionPhase}
            />
          </div>

          <div className="stage-instrument" aria-hidden="true">
            <span className="ghost-stack" />
            <div>
              <b>TRAINING DECISION TRACE / ACTIVE</b>
              <small>Observe → diagnose → verify → replay</small>
            </div>
            <div className="instrument-states">
              <span>HOOK: READY</span>
              <span>LOSS GUARD: ARMED</span>
              <span>REPLAY LOG: READY</span>
            </div>
          </div>

          <div className="stage-status" aria-live="polite">
            <span className="status-pulse" />
            <div>
              <strong>
                {activeDestination
                  ? `FOCUSING / ${activeDestination.toUpperCase()}`
                  : "ORBITAL VIEW / READY"}
              </strong>
              <small>
                {activeDestination
                  ? "Camera aligned with destination marker"
                  : "Four destinations are available"}
              </small>
            </div>
          </div>
        </section>

        {/* ── Destination Panel ─────────────────────────────────────────────── */}
        <AnimatePresence mode="wait">
          {activeContent && activeDestination && (
            <motion.section
              className="destination-panel"
              key={activeDestination}
              initial={{ opacity: 0, y: 30, clipPath: "inset(6% 0 0 0)" }}
              animate={{ opacity: 1, y: 0, clipPath: "inset(0% 0 0 0)" }}
              exit={{ opacity: 0, y: -18, clipPath: "inset(0 0 10% 0)" }}
              transition={{ duration: 0.48, ease: [0.23, 1, 0.32, 1] }}
            >
              <div className="panel-rail">
                <span>FOCUSED DESTINATION</span>
                <b>
                  {String(
                    DESTINATIONS.findIndex((d) => d.id === activeDestination) + 1
                  ).padStart(2, "0")}
                </b>
              </div>
              <div className="panel-content">
                <div className="panel-heading">
                  <div className="panel-icon">
                    <activeContent.icon size={20} />
                  </div>
                  <p className="orbit-eyebrow">{activeContent.eyebrow}</p>
                  <h2>{activeContent.title}</h2>
                </div>
                <div className="panel-body">
                  <p className="panel-lede">{activeContent.body}</p>
                  <p className="panel-annotation">
                    <Sparkles size={14} /> {activeContent.annotation}
                  </p>
                  {activeDestination === "control" ? (
                    <div className={`policy-model policy-model--${decisionState}`}>
                      <div className="policy-model__header">
                        <span>POLICY GATE</span>
                        <strong>
                          {decisionState === "observing"
                            ? "Awaiting a test"
                            : decisionState === "checking"
                            ? "Checking the boundary"
                            : decisionState === "verified"
                            ? "Verified and retained"
                            : "Blocked and logged"}
                        </strong>
                      </div>
                      <div className="policy-track">
                        <i /><i /><i />
                      </div>
                      <p>
                        {decisionState === "observing"
                          ? "Run a controlled path to see how the system treats a safe and an unsafe outcome."
                          : decisionState === "checking"
                          ? "The proposed intervention is being measured against its stated safety condition."
                          : decisionState === "verified"
                          ? "The system would keep the intervention, write its context to memory, and mark a verified path."
                          : "The system would keep the workload untouched and preserve the rejected action for audit."}
                      </p>
                      <div className="policy-actions">
                        <button onClick={() => runPolicyTest("verified")} disabled={isChecking}>
                          <Check size={14} /> Test safe path
                        </button>
                        <button onClick={() => runPolicyTest("blocked")} disabled={isChecking}>
                          <CircleAlert size={14} /> Test rejected path
                        </button>
                      </div>
                    </div>
                  ) : activeDestination === "contact" ? (
                    <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
                      <Link className="panel-action panel-action--mail" href="/pilot">
                        <Send size={15} /> Request free pilot audit <ArrowUpRight size={15} />
                      </Link>
                      <a
                        className="panel-action"
                        href="mailto:hello@ghostlayer.ai?subject=GhostLayer%20fleet%20conversation"
                      >
                        <Mail size={15} /> Email directly <ArrowUpRight size={15} />
                      </a>
                    </div>
                  ) : (
                    <button
                      className="panel-action"
                      onClick={() =>
                        document.getElementById("lifecycle")?.scrollIntoView({ behavior: "smooth" })
                      }
                    >
                      <ArrowDownRight size={15} /> Read the field guide
                    </button>
                  )}
                </div>
                <div className="panel-metrics">
                  {activeContent.metrics.map((metric) => (
                    <div key={metric.label}>
                      <span>{metric.label}</span>
                      <strong>{metric.value}</strong>
                    </div>
                  ))}
                </div>
              </div>
              <button
                className="panel-close"
                onClick={returnToOrbit}
                aria-label="Close destination and return to orbit"
              >
                <X size={18} />
              </button>
            </motion.section>
          )}
        </AnimatePresence>

        {/* ── Field Guide ──────────────────────────────────────────────────── */}
        <section className="field-guide" id="field-guide">
          <div className="guide-intro">
            <p className="orbit-eyebrow"><span /> FIELD GUIDE</p>
            <h2>The operating model, traced from one training step to its evidence.</h2>
            <p>
              Use the globe to move between the major parts of the system. GhostLayer
              starts by observing a workload, diagnoses a candidate improvement,
              verifies the safety boundary, and retains both accepted and rejected
              outcomes as operating memory.
            </p>
          </div>
          <div className="guide-route">
            <div className="route-line" aria-hidden="true" />
            {DESTINATIONS.map((destination, index) => {
              const content = destinationContent[destination.id];
              const icons = {
                info: ScanSearch,
                control: ShieldCheck,
                memory: Layers3,
                contact: Mail,
              };
              const Icon = icons[destination.id];
              return (
                <button
                  className={`route-stop route-stop--${destination.tone} ${
                    activeDestination === destination.id ? "is-active" : ""
                  }`}
                  key={destination.id}
                  onClick={() => focusDestination(destination.id)}
                >
                  <span className="route-index">0{index + 1}</span>
                  <Icon size={20} />
                  <div>
                    <b>{destination.label}</b>
                    <small>{content.metrics.map((m) => m.value).join(" · ")}</small>
                  </div>
                  <ArrowUpRight size={15} />
                </button>
              );
            })}
          </div>
        </section>

        {/* ── Lifecycle Flow ───────────────────────────────────────────────── */}
        <div id="lifecycle">
          <LifecycleFlow onPhaseChange={setSectionPhase} />
        </div>

        {/* ── ROI Proof Table ──────────────────────────────────────────────── */}
        <RoiProofTable />

        {/* ── Cluster Topology & Telemetry Suite ──────────────────────────── */}
        <section className="topology-suite-section" id="topology-suite" aria-label="Cluster & Graph Topology Lab">
          <div className="suite-nav-header">
            <div className="suite-title-group">
              <Network size={16} style={{ color: "#737373" }} />
              <h2>CLUSTER GRAPH TOPOLOGY &amp; TELEMETRY LAB</h2>
            </div>
            <nav className="suite-tab-bar" aria-label="Topology Lab Views">
              {[
                { id: "calculator", label: "FinOps Profit Calculator", icon: Calculator },
                { id: "dag", label: "Computational DAG & Fusion", icon: Cpu },
                { id: "dist", label: "Distributed NCCL Fabric", icon: Network },
                { id: "containment", label: "Containment & Rollback", icon: ShieldCheck },
                { id: "telemetry", label: "Telemetry & Loss Gate", icon: Gauge },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    className={`suite-tab ${activeTopologyTab === tab.id ? "is-active" : ""}`}
                    onClick={() => setActiveTopologyTab(tab.id as any)}
                  >
                    <Icon size={13} />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
          <div className="suite-view-content">
            {activeTopologyTab === "calculator" && <GpuProfitCalculator />}
            {activeTopologyTab === "dag" && <ExecutionDAGViewer />}
            {activeTopologyTab === "dist" && <DistributedTopologyViewer />}
            {activeTopologyTab === "containment" && <ContainmentLatticeViewer />}
            {activeTopologyTab === "telemetry" && <TelemetryMatrix />}
          </div>
        </section>

        {/* ── Product Model ───────────────────────────────────────────────── */}
        <section className="product-model" id="product-model">
          <div className="product-model__header">
            <div>
              <p className="orbit-eyebrow"><span /> SOURCE-CONFIRMED OPERATING MODEL</p>
              <h2>What GhostLayer is built to do.</h2>
            </div>
            <p>
              This is the product loop documented in the supplied project source: a PyTorch
              integration surface, a decision engine, correctness verification, replay memory,
              and an ROI audit layer. It is not a generic GPU dashboard and it is not a cloud
              provider.
            </p>
          </div>
          <div className="pillar-list">
            {[
              { index: "01", title: "Observe", detail: "GhostWatcherHook records GPU utilization, VRAM, step timing, data-loading timing, configuration signals, and optional loss context without changing the run first.", icon: ScanSearch },
              { index: "02", title: "Diagnose", detail: "The decision engine and Graphify analysis identify potential bottlenecks and generate recommendations across runtime, memory, kernel, and data-path choices.", icon: Zap },
              { index: "03", title: "Verify", detail: "Correctness verification compares baseline and optimized loss trajectories. When the policy boundary is exceeded, automatic application is revoked.", icon: ShieldCheck },
              { index: "04", title: "Replay & learn", detail: "OBSERVE, DIAGNOSE, VERIFY, and ROLLBACK events preserve the context, action, result, and reason; verified feedback can update the knowledge base.", icon: Database },
              { index: "05", title: "Prove the outcome", detail: "The ROI calculator and report generator connect a verified performance result to GPU-hours, savings, and an auditable written receipt.", icon: Gauge },
            ].map((pillar) => {
              const Icon = pillar.icon;
              return (
                <article className="pillar" key={pillar.index}>
                  <span className="pillar__index">{pillar.index}</span>
                  <div className="pillar__icon"><Icon size={18} /></div>
                  <div><h3>{pillar.title}</h3></div>
                  <p>{pillar.detail}</p>
                  <span className="pillar__trace" />
                </article>
              );
            })}
          </div>
          <div className="evidence-qualification">
            <div>
              <span>PHYSICAL RECEIPT / VERIFIED-SAFE PATH</span>
              <strong>52.2M parameter LLM on an RTX A2000</strong>
              <p>Project audit: 31.27 ms/step baseline to 10.74 ms/step optimized, with a recorded maximum loss delta of 0.0012.</p>
            </div>
            <div className="evidence-qualification__risk">
              <span>PHYSICAL RECEIPT / BLOCKED PATH</span>
              <strong>166.3M high-VRAM workload</strong>
              <p>Project audit: a faster candidate was downgraded when its loss delta reached 0.3845, above the stated 0.10 threshold.</p>
            </div>
            <small>Source: the supplied project's Physical GPU Hardware Evidence Audit. These are project-specific benchmark receipts, not fleet-wide performance guarantees.</small>
          </div>
        </section>

        {/* ── Operating Note ──────────────────────────────────────────────── */}
        <section className="operating-note">
          <div>
            <p className="orbit-eyebrow"><span /> READ THE SIGNAL, NOT JUST THE NUMBER</p>
            <h2>There is a difference between observing a problem and owning the next move.</h2>
          </div>
          <div className="note-systems">
            <div>
              <Gauge size={23} />
              <strong>Observe in context</strong>
              <p>Connect GPU behavior, memory condition, and training signals into one explanation.</p>
            </div>
            <div>
              <ShieldCheck size={23} />
              <strong>Evaluate safely</strong>
              <p>Keep any change inside a stated policy boundary before it becomes a decision.</p>
            </div>
            <div>
              <RotateCcw size={23} />
              <strong>Preserve the proof</strong>
              <p>Record both accepted and rejected outcomes so operations gain memory over time.</p>
            </div>
          </div>
          <div className="instrument-evidence">
            <span className="evidence-margin">INSTRUMENT MODEL</span>
            <div><small>DECISION TRACE</small><strong>Observe → Diagnose → Verify</strong></div>
            <div><small>OBSERVED</small><strong>Training-loop telemetry</strong></div>
            <div><small>SAFETY GATE</small><strong>Loss trajectory check</strong></div>
            <div><small>OUTCOME</small><strong>Replay + ROI receipt</strong></div>
            <span className="ghost-stack ghost-stack--ink" />
          </div>
        </section>
      </main>

      {/* ── Mobile Dock ─────────────────────────────────────────────────── */}
      <nav className="mobile-orbit-dock" aria-label="Mobile globe destinations">
        {DESTINATIONS.map((destination) => (
          <button
            key={destination.id}
            className={activeDestination === destination.id ? "is-active" : ""}
            onClick={() => focusDestination(destination.id)}
          >
            <DestinationIcon id={destination.id} />
            <span>{destination.label === "Contact us" ? "Contact" : destination.label}</span>
          </button>
        ))}
      </nav>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="orbit-footer">
        <BrandLockup />
        <span>© 2026 GhostLayer</span>
        <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
          <Link
            href="/pilot"
            style={{ color: "var(--mist)", display: "inline-flex", alignItems: "center", gap: "4px" }}
          >
            Request Free Audit <ArrowUpRight size={13} />
          </Link>
          <a href="mailto:hello@ghostlayer.ai">
            hello@ghostlayer.ai <ArrowUpRight size={13} />
          </a>
        </div>
      </footer>
    </div>
  );
}
