import React, { useEffect } from "react";
import { Link } from "wouter";
import { ArrowLeft, AlertOctagon, Gauge, ShieldCheck, Scale, Cpu, CheckCircle2, FileText, Globe, Lock, Info } from "lucide-react";

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
          <div style={{ display: "flex", gap: "16px", fontSize: "12px", flexWrap: "wrap" }}>
            <Link href="/terms" style={{ color: "#737373", textDecoration: "none" }}>Terms</Link>
            <Link href="/privacy" style={{ color: "#737373", textDecoration: "none" }}>Privacy</Link>
            <Link href="/compliance" style={{ color: "#737373", textDecoration: "none" }}>Compliance</Link>
            <Link href="/security" style={{ color: "#737373", textDecoration: "none" }}>Security</Link>
            <Link href="/dpa" style={{ color: "#737373", textDecoration: "none" }}>DPA</Link>
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
            Comprehensive regulatory notices, empirical testing qualifications, trademark disclaimers, and liability boundaries.
          </p>
        </header>

        {/* Master Legal Hub Grid (5 Companion Policies) */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "14px", marginBottom: "40px" }}>
          <Link href="/terms" style={{ textDecoration: "none", background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "18px", borderRadius: "6px", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}>Contractual</span>
              <Scale size={16} style={{ color: "#FFFFFF" }} />
            </div>
            <strong style={{ color: "#FFFFFF", fontSize: "14px", display: "block", marginBottom: "4px" }}>Terms of Service</strong>
            <p style={{ color: "#737373", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>
              AS-IS warranty disclaimers, $100 liability caps, binding arbitration &amp; class action waiver.
            </p>
          </Link>

          <Link href="/privacy" style={{ textDecoration: "none", background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "18px", borderRadius: "6px", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}>Data Security</span>
              <ShieldCheck size={16} style={{ color: "#10b981" }} />
            </div>
            <strong style={{ color: "#FFFFFF", fontSize: "14px", display: "block", marginBottom: "4px" }}>Privacy &amp; Telemetry</strong>
            <p style={{ color: "#737373", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>
              Zero-model-weights ingestion guarantee, scalar profiling limits, and data retention standards.
            </p>
          </Link>

          <Link href="/compliance" style={{ textDecoration: "none", background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "18px", borderRadius: "6px", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}>Export (EAR)</span>
              <Globe size={16} style={{ color: "#38bdf8" }} />
            </div>
            <strong style={{ color: "#FFFFFF", fontSize: "14px", display: "block", marginBottom: "4px" }}>Acceptable Use &amp; EAR</strong>
            <p style={{ color: "#737373", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>
              Dual-use AI cluster export controls, Commerce Control List adherence, and prohibited workloads.
            </p>
          </Link>

          <Link href="/security" style={{ textDecoration: "none", background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "18px", borderRadius: "6px", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}>VDP &amp; RFC 9116</span>
              <Lock size={16} style={{ color: "#a855f7" }} />
            </div>
            <strong style={{ color: "#FFFFFF", fontSize: "14px", display: "block", marginBottom: "4px" }}>Security &amp; Safe Harbor</strong>
            <p style={{ color: "#737373", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>
              White-hat researcher protection, TLS 1.3 cryptographic boundary, and 24h SLA reporting.
            </p>
          </Link>

          <Link href="/dpa" style={{ textDecoration: "none", background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "18px", borderRadius: "6px", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}>Procurement</span>
              <FileText size={16} style={{ color: "#f59e0b" }} />
            </div>
            <strong style={{ color: "#FFFFFF", fontSize: "14px", display: "block", marginBottom: "4px" }}>DPA &amp; Subprocessors</strong>
            <p style={{ color: "#737373", fontSize: "12px", margin: 0, lineHeight: "1.5" }}>
              GDPR Article 28, SCCs, Technical Measures (TOMs), and active infrastructure registry.
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
              <p style={{ fontSize: "13px", color: "#D4D4D4", margin: "0 0 12px 0" }}>
                These benchmark receipts represent historical, isolated physical executions on a single workstation. They are published for transparency and empirical reproducibility, NOT as warranties or guarantees of fleet-wide speedup across distinct GPU architectures (e.g. NVIDIA H100, B200, AMD MI300X), model architectures (e.g. MoE, diffusion, multi-modal), or arbitrary batch configurations.
              </p>
              <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", fontSize: "12px" }}>
                <a
                  href="/evidence/pythia-70m-audit.html"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: "#10b981", textDecoration: "none", border: "1px solid rgba(16, 185, 129, 0.3)", padding: "6px 10px", borderRadius: "4px", background: "rgba(16, 185, 129, 0.05)", display: "inline-flex", alignItems: "center", gap: "4px" }}
                >
                  Inspect Pythia-70M Live Audit HTML (RTX A2000) ↗
                </a>
                <a
                  href="/evidence/pythia-70m-evidence.json"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: "#38bdf8", textDecoration: "none", border: "1px solid rgba(56, 189, 248, 0.3)", padding: "6px 10px", borderRadius: "4px", background: "rgba(56, 189, 248, 0.05)", display: "inline-flex", alignItems: "center", gap: "4px" }}
                >
                  Download SHA-256 Telemetry JSON ↗
                </a>
                <a
                  href="/evidence/executive-audit.html"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: "#a855f7", textDecoration: "none", border: "1px solid rgba(168, 85, 247, 0.3)", padding: "6px 10px", borderRadius: "4px", background: "rgba(168, 85, 247, 0.05)", display: "inline-flex", alignItems: "center", gap: "4px" }}
                >
                  Inspect Full Executive Diagnostic Receipt ↗
                </a>
              </div>
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
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              5. Comprehensive Liability Disclaimer &amp; $100 Cap
            </h2>
            <p>
              UNDER NO CIRCUMSTANCES SHALL GHOSTLAYER, ITS FOUNDER (BHARGAV MAHADEVAN), OR CONTRIBUTORS BE HELD LIABLE FOR LOST TRAINING RUNS, CLOUD BILLING SURGES, HARDWARE DAMAGES, OR LOSS ACCURACY REGRESSIONS ARISING FROM THE EXECUTION OF EXPERIMENTAL PROFILING HOOKS. PLEASE CONSULT OUR FULL <Link href="/terms" style={{ color: "#10b981", textDecoration: "underline" }}>TERMS OF SERVICE</Link> FOR COMPLETE LEGAL CONDITIONS.
            </p>
          </section>

          {/* Section 6: Third-Party Trademark Legal Notices */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              6. Third-Party Trademarks &amp; Non-Affiliation Disclaimers
            </h2>
            <p>
              All product names, logos, brands, trademarks, and registered trademarks cited on this platform are property of their respective holders:
            </p>
            <ul style={{ paddingLeft: "20px" }}>
              <li><strong>NVIDIA Corporation:</strong> NVIDIA, CUDA, RTX, TensorRT, NVLink, H100, H200, B200, and A2000 are trademarks or registered trademarks of NVIDIA Corporation.</li>
              <li><strong>The Linux Foundation:</strong> PyTorch is a registered trademark of The Linux Foundation.</li>
              <li><strong>Meta Platforms, Inc.:</strong> PyTorch original design marks and FAIR research references are trademarks of Meta Platforms, Inc.</li>
              <li><strong>Cloud Platforms:</strong> AWS, Google Cloud, Microsoft Azure, CoreWeave, and Lambda Labs are trademarks of their respective corporate owners.</li>
            </ul>
            <div style={{ background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "14px", borderRadius: "4px", marginTop: "12px", fontSize: "12px", color: "#A3A3A3" }}>
              <strong>Statement of Non-Affiliation:</strong> Use of these trademarks does not imply any affiliation with, endorsement by, or sponsorship by any trademark holder. GhostLayer is an independent open engineering project and audit layer created by Bhargav Mahadevan.
            </div>
          </section>

          {/* Section 7: Intellectual Property & Trade Secrets */}
          <section style={{ marginBottom: "48px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              7. Intellectual Property &amp; Algorithmic Protection
            </h2>
            <p>
              The GhostLayer containment lattice, heuristic recommendation engine, loss-shift proxy algorithms, structured decision record schemas, and interactive visualization interfaces are protected under United States and international copyright, trade secret, and unfair competition laws. Unauthorized scraping, unauthorized competitive commercial benchmarking, or reverse engineering of binary optimization kernels is strictly prohibited.
            </p>
          </section>

        </article>

        {/* Footer Navigation */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "13px", flexWrap: "wrap", gap: "12px" }}>
          <Link href="/" style={{ color: "#737373", textDecoration: "none" }}>
            ← Back to GhostLayer Orbit
          </Link>
          <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
            <Link href="/terms" style={{ color: "#A3A3A3", textDecoration: "none" }}>Terms →</Link>
            <Link href="/privacy" style={{ color: "#A3A3A3", textDecoration: "none" }}>Privacy →</Link>
            <Link href="/compliance" style={{ color: "#A3A3A3", textDecoration: "none" }}>Compliance →</Link>
            <Link href="/security" style={{ color: "#A3A3A3", textDecoration: "none" }}>Security →</Link>
            <Link href="/dpa" style={{ color: "#A3A3A3", textDecoration: "none" }}>DPA →</Link>
          </div>
        </div>

      </div>
    </div>
  );
}
