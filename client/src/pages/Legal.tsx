import React, { useEffect } from "react";
import { Link } from "wouter";
import { ArrowLeft, AlertOctagon, Gauge, ShieldCheck, Scale, Cpu, CheckCircle2, FileText, Info } from "lucide-react";

export default function Legal() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="orbit-page legal-document-page" style={{ minHeight: "100vh", background: "#050505", color: "#F5F5F5", padding: "40px 24px 80px" }}>
      <div style={{ maxWidth: "860px", margin: "0 auto" }}>
        
        {/* Navigation Bar */}
        <nav style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "48px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "20px" }}>
          <Link href="/" style={{ display: "inline-flex", alignItems: "center", gap: "8px", color: "#A3A3A3", textDecoration: "none", fontSize: "13px", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            <ArrowLeft size={16} /> Return to Orbit
          </Link>
          <div style={{ display: "flex", gap: "16px", fontSize: "12px" }}>
            <Link href="/terms" style={{ color: "#737373", textDecoration: "none" }}>Terms of Service</Link>
            <Link href="/privacy" style={{ color: "#737373", textDecoration: "none" }}>Privacy Policy</Link>
            <Link href="/founder" style={{ color: "#10b981", textDecoration: "none" }}>Founder Dossier</Link>
          </div>
        </nav>

        {/* Document Header */}
        <header style={{ marginBottom: "40px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "9999px", padding: "4px 12px", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#D4D4D4", marginBottom: "16px" }}>
            <AlertOctagon size={13} style={{ color: "#f43f5e" }} /> Master Legal &amp; Regulatory Disclosures
          </div>
          <h1 style={{ fontSize: "36px", fontWeight: "700", letterSpacing: "-0.02em", margin: "0 0 12px 0", color: "#FFFFFF" }}>
            Legal Disclaimers &amp; Empirical Transparency
          </h1>
          <p style={{ color: "#737373", fontSize: "14px", margin: 0 }}>
            Comprehensive regulatory notices, empirical testing qualifications, and liability boundaries.
          </p>
        </header>

        {/* Quick Hub Navigation Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px", marginBottom: "40px" }}>
          <Link href="/terms" style={{ textDecoration: "none", background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "20px", borderRadius: "6px", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}>Contractual Terms</span>
              <Scale size={16} style={{ color: "#FFFFFF" }} />
            </div>
            <strong style={{ color: "#FFFFFF", fontSize: "15px", display: "block", marginBottom: "6px" }}>Terms of Service</strong>
            <p style={{ color: "#737373", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>
              AS-IS warranty disclaimers, $100 liability caps, binding arbitration &amp; class action waiver.
            </p>
          </Link>
          <Link href="/privacy" style={{ textDecoration: "none", background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "20px", borderRadius: "6px", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}>Data Security</span>
              <ShieldCheck size={16} style={{ color: "#10b981" }} />
            </div>
            <strong style={{ color: "#FFFFFF", fontSize: "15px", display: "block", marginBottom: "6px" }}>Privacy &amp; Telemetry</strong>
            <p style={{ color: "#737373", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>
              Zero-model-weights ingestion guarantee, scalar profiling limits, and data retention standards.
            </p>
          </Link>
        </div>

        {/* Legal & Empirical Articles */}
        <article style={{ fontSize: "14px", lineHeight: "1.75", color: "#A3A3A3" }}>
          
          {/* Section 1: Empirical Benchmark Disclosures */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              1. Empirical Benchmark Receipts &amp; Real-Hardware Policy
            </h2>
            <p>
              In accordance with GhostLayer's <strong>Zero Fabrication Policy</strong>, all benchmark figures, step timings, and memory profiles displayed on this platform (including the 52.2M LLM test: 31.27 ms/step baseline to 10.74 ms/step optimized) reflect physical measurements performed on an actual <strong>NVIDIA RTX A2000 (6GB VRAM)</strong> workstation under PyTorch 2.x and CUDA 12.x.
            </p>
            <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", padding: "16px", borderRadius: "4px", margin: "16px 0" }}>
              <strong style={{ color: "#FFFFFF", display: "block", marginBottom: "4px" }}>Notice on Benchmarks vs. Fleet-Wide Guarantees:</strong>
              <p style={{ fontSize: "13px", color: "#D4D4D4", margin: 0 }}>
                These benchmark receipts represent historical, isolated physical executions on a single workstation. They are published for transparency and empirical reproducibility, NOT as warranties or guarantees of fleet-wide speedup across distinct GPU architectures (e.g. NVIDIA H100, B200, AMD MI300X), model architectures (e.g. MoE, diffusion, multi-modal), or arbitrary batch configurations.
              </p>
            </div>
          </section>

          {/* Section 2: Safety Boundary & Loss-Shift Proxy */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              2. Loss-Shift Proxy &amp; "Verified Safe" Boundary Limitations
            </h2>
            <p>
              GhostLayer enforces a mathematical loss-shift proxy threshold of <strong>0.10</strong>. When candidate optimizations (such as dynamic kernel fusion, memory chunking, or batch scaling) produce a step loss delta exceeding 0.10, automated application is blocked and marked as rejected.
            </p>
            <div style={{ background: "rgba(244, 63, 94, 0.04)", border: "1px solid rgba(244, 63, 94, 0.2)", padding: "16px", borderRadius: "4px", margin: "16px 0" }}>
              <strong style={{ color: "#fb7185", display: "block", marginBottom: "4px" }}>Legal Scope of "VERIFIED SAFE":</strong>
              <p style={{ fontSize: "13px", color: "#D4D4D4", margin: 0 }}>
                The label <strong>"VERIFIED SAFE"</strong> indicates strictly that the sampled step's loss-shift proxy check did not exceed the configured 0.10 threshold on that specific iteration. <strong>IT DOES NOT CONSTITUTE A GENERAL MATHEMATICAL CLAIM, GUARANTEE OF TOTAL CONVERGENCE, ABSOLUTE NUMERICAL ACCURACY, OR LONG-HORIZON TRAINING STABILITY.</strong> The user remains solely responsible for continuous loss monitoring, checkpointing, and downstream evaluation.
              </p>
            </div>
          </section>

          {/* Section 3: FinOps Calculator Disclaimers */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              3. FinOps Profit Calculator &amp; Financial Simulation Disclaimers
            </h2>
            <p>
              The FinOps Profit Calculator and ROI estimation tools provided on this website are algorithmic mathematical models intended solely for illustrative and exploratory evaluation.
            </p>
            <ul style={{ paddingLeft: "20px" }}>
              <li>Calculations rely on user-supplied variables (GPU fleet size, hourly rental rates, estimated compute efficiency).</li>
              <li>Estimated annual recoveries and compute leakage percentages are heuristic projections based on aggregated academic benchmarks and published cloud pricing tiers (AWS, GCP, CoreWeave, Lambda Labs).</li>
              <li>GhostLayer makes no representation, warranty, or guarantee that Customer will achieve identical or equivalent financial, energy, or operational savings.</li>
              <li>Nothing on this website constitutes investment advice, accounting certification, or a binding financial guarantee.</li>
            </ul>
          </section>

          {/* Section 4: Human-in-the-Loop Mandate */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              4. Mandatory Human-in-the-Loop AI Governance
            </h2>
            <p>
              GhostLayer is engineered under the principle that AI infrastructure operations must remain subject to human sovereignty. Users of GhostLayer automated control policies and rollback mechanisms must ensure:
            </p>
            <ol style={{ paddingLeft: "20px" }}>
              <li>Human systems engineers maintain override capability over all hook actions.</li>
              <li>Automated interventions are audited against structured decision records.</li>
              <li>Production training runs are decoupled from unmonitored automated hyperparameter mutations.</li>
            </ol>
          </section>

          {/* Section 5: Software As-Is & Total Liability Caps */}
          <section style={{ marginBottom: "48px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              5. Comprehensive Liability Disclaimer
            </h2>
            <p>
              UNDER NO CIRCUMSTANCES SHALL GHOSTLAYER, ITS FOUNDER (BHARGAV MAHADEVAN), OR CONTRIBUTORS BE HELD LIABLE FOR LOST TRAINING RUNS, CLOUD BILLING SURGES, HARDWARE DAMAGES, OR LOSS ACCURACY REGRESSIONS ARISING FROM THE EXECUTION OF EXPERIMENTAL PROFILING HOOKS. PLEASE CONSULT OUR FULL <Link href="/terms" style={{ color: "#10b981", textDecoration: "underline" }}>TERMS OF SERVICE</Link> FOR COMPLETE LEGAL CONDITIONS.
            </p>
          </section>

        </article>

        {/* Footer Navigation */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "13px" }}>
          <Link href="/" style={{ color: "#737373", textDecoration: "none" }}>
            ← Back to GhostLayer Orbit
          </Link>
          <div style={{ display: "flex", gap: "20px" }}>
            <Link href="/terms" style={{ color: "#A3A3A3", textDecoration: "none" }}>Terms of Service →</Link>
            <Link href="/privacy" style={{ color: "#A3A3A3", textDecoration: "none" }}>Privacy Policy →</Link>
          </div>
        </div>

      </div>
    </div>
  );
}
