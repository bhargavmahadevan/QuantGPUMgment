import React, { useEffect } from "react";
import { Link } from "wouter";
import { ArrowLeft, ShieldAlert, Globe, FileCheck2, Scale, AlertTriangle, Cpu, CheckCircle2 } from "lucide-react";

export default function Compliance() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="orbit-page legal-document-page" style={{ minHeight: "100vh", background: "#050505", color: "#F5F5F5", padding: "40px 24px 80px" }}>
      <div style={{ maxWidth: "860px", margin: "0 auto" }}>
        
        {/* Top Navigation */}
        <nav style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "48px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "20px" }}>
          <Link href="/" style={{ display: "inline-flex", alignItems: "center", gap: "8px", color: "#A3A3A3", textDecoration: "none", fontSize: "13px", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            <ArrowLeft size={16} /> Return to Orbit
          </Link>
          <div style={{ display: "flex", gap: "16px", fontSize: "12px", flexWrap: "wrap" }}>
            <Link href="/terms" style={{ color: "#737373", textDecoration: "none" }}>Terms</Link>
            <Link href="/privacy" style={{ color: "#737373", textDecoration: "none" }}>Privacy</Link>
            <Link href="/security" style={{ color: "#737373", textDecoration: "none" }}>Security</Link>
            <Link href="/dpa" style={{ color: "#737373", textDecoration: "none" }}>DPA</Link>
            <Link href="/legal" style={{ color: "#737373", textDecoration: "none" }}>Disclaimers</Link>
            <Link href="/founder" style={{ color: "#10b981", textDecoration: "none" }}>Founder Dossier</Link>
          </div>
        </nav>

        {/* Document Header */}
        <header style={{ marginBottom: "40px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "9999px", padding: "4px 12px", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#D4D4D4", marginBottom: "16px" }}>
            <Globe size={13} style={{ color: "#10b981" }} /> Acceptable Use &amp; Regulatory Compliance
          </div>
          <h1 style={{ fontSize: "36px", fontWeight: "700", letterSpacing: "-0.02em", margin: "0 0 12px 0", color: "#FFFFFF" }}>
            Acceptable Use &amp; Dual-Use AI Export Policy
          </h1>
          <p style={{ color: "#737373", fontSize: "14px", margin: 0 }}>
            Effective Date: September 10, 2026 · Version 1.2 (Enterprise AI Infrastructure Governance)
          </p>
        </header>

        {/* High-Risk Regulatory Banner */}
        <div style={{ background: "rgba(244, 63, 94, 0.05)", border: "1px solid rgba(244, 63, 94, 0.25)", borderRadius: "6px", padding: "20px", marginBottom: "40px" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
            <ShieldAlert size={22} style={{ color: "#f43f5e", flexShrink: 0, marginTop: "2px" }} />
            <div>
              <strong style={{ color: "#fb7185", fontSize: "14px", display: "block", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Export Administration Regulations (EAR) &amp; Dual-Use AI Notice
              </strong>
              <p style={{ fontSize: "13px", lineHeight: "1.6", color: "#D4D4D4", margin: 0 }}>
                High-performance GPU cluster acceleration, distributed training hooks, and compute optimization software are subject to United States Export Administration Regulations (EAR) and international dual-use export controls. Users of GhostLayer certify that workloads comply with all applicable multilateral export restrictions and OFAC sanctions regimes.
              </p>
            </div>
          </div>
        </div>

        {/* Policy Body */}
        <article style={{ fontSize: "14px", lineHeight: "1.75", color: "#A3A3A3" }}>

          {/* Section 1: Acceptable Use */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              1. Enterprise Acceptable Use Policy (AUP)
            </h2>
            <p>
              GhostLayer is engineered solely for lawful, ethical infrastructure efficiency, telemetry observation, and energy/compute optimization of machine learning training workloads. You agree NOT to use the GhostLayer software, context hooks (`ghost_layer`), simulator, or audit reports to facilitate, optimize, or accelerate any of the following prohibited workloads:
            </p>
            <ul style={{ paddingLeft: "20px" }}>
              <li><strong>Autonomous Kinetic Weapons:</strong> Real-time guidance, lethal targeting, or autonomous deployment of lethal kinetic or drone weapon systems.</li>
              <li><strong>Chemical, Biological, Radiological, or Nuclear (CBRN) Simulation:</strong> Synthesis pathways, dissemination mechanisms, or genetic engineering of hazardous pathogens or regulated chemical toxins.</li>
              <li><strong>Malicious Cyber Intrusion:</strong> Training automated exploit generation models, zero-day discovery engines for unauthorized offensive attacks, or ransomware payloads.</li>
              <li><strong>Non-Consensual Deepfakes or Child Exploitation:</strong> Generation or distribution of non-consensual intimate imagery, synthetic child sexual abuse material (CSAM), or automated biometric surveillance of protected demographics.</li>
              <li><strong>Unauthorized Cryptomining:</strong> Hijacking or redistributing enterprise cluster compute capacity for clandestine cryptocurrency mining operations.</li>
            </ul>
          </section>

          {/* Section 2: U.S. Export Controls */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              2. U.S. Export Administration Regulations (EAR) &amp; Dual-Use Classification
            </h2>
            <p>
              GhostLayer software and its underlying telemetry algorithms operate in proximity to advanced GPU microarchitectures (including NVIDIA H100, H200, B200, AMD MI300X, and custom silicon). In accordance with U.S. Department of Commerce Bureau of Industry and Security (BIS) regulations:
            </p>
            <ol style={{ paddingLeft: "20px" }}>
              <li><strong>Jurisdiction:</strong> Software origin is the United States of America. Export, re-export, or in-country transfer in violation of U.S. law is strictly prohibited.</li>
              <li><strong>Destination Restrictions:</strong> GhostLayer services, code, and pilot audit deliveries may not be exported, downloaded, or transferred to any entity located in U.S. embargoed jurisdictions (including Cuba, Iran, North Korea, Syria, and the Crimea/Donetsk/Luhansk regions of Ukraine).</li>
              <li><strong>Restricted Parties:</strong> GhostLayer shall not be made available to individuals or entities listed on the U.S. Specially Designated Nationals (SDN) List, Denied Persons List, Entity List, or Unverified List maintained by OFAC and BIS.</li>
              <li><strong>Dual-Use Computing Power:</strong> Customers operating clusters exceeding advanced compute thresholds (Total Processing Performance &gt; 4800) represent that they maintain all necessary BIS licenses and end-user export certifications.</li>
            </ol>
          </section>

          {/* Section 3: Cloud Provider Covenants */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              3. Cloud Provider Anti-Circumvention &amp; Multi-Tenant Covenants
            </h2>
            <p>
              When utilizing GhostLayer context hooks on third-party cloud infrastructure (such as Amazon Web Services EC2, Google Cloud Platform, Microsoft Azure, CoreWeave, Lambda Labs, RunPod, or Crusoe Cloud), Customer expressly covenants that:
            </p>
            <ul style={{ paddingLeft: "20px" }}>
              <li>GhostLayer hooks operate strictly within the bounds of Customer's allocated tenant virtual machines or bare-metal instances.</li>
              <li>GhostLayer will not be used to inspect, probe, or intercept noisy-neighbor telemetry, hypervisor host memory, or out-of-band management buses of fellow cloud tenants.</li>
              <li>Customer remains solely responsible for abiding by its cloud service provider's terms of service, billing policies, and compute quotas.</li>
            </ul>
          </section>

          {/* Section 4: Audit Rights & Termination */}
          <section style={{ marginBottom: "48px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              4. Enforcement, Audit Rights &amp; Whistleblower Inquiries
            </h2>
            <p>
              GhostLayer reserves the right to immediately terminate access, revoke pilot evaluation licenses, and decline audit deliveries for any user or organization found to be in breach of this Policy. If you have reason to suspect a compliance or export violation involving GhostLayer software, please report it immediately:
            </p>
            <div style={{ background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.1)", padding: "16px", borderRadius: "4px", marginTop: "12px" }}>
              <strong style={{ color: "#FFFFFF", display: "block", marginBottom: "4px" }}>GhostLayer Regulatory Compliance Office</strong>
              <span>Attn: Export Controls &amp; AI Safety Oversight</span><br />
              <span>Email: <a href="mailto:compliance@ghostlayer.ai" style={{ color: "#10b981", textDecoration: "none" }}>compliance@ghostlayer.ai</a> / <a href="mailto:bhargavmahadevan@gmail.com" style={{ color: "#10b981", textDecoration: "none" }}>bhargavmahadevan@gmail.com</a></span>
            </div>
          </section>

        </article>

        {/* Footer Navigation */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "13px" }}>
          <Link href="/" style={{ color: "#737373", textDecoration: "none" }}>
            ← Back to GhostLayer Orbit
          </Link>
          <div style={{ display: "flex", gap: "20px" }}>
            <Link href="/security" style={{ color: "#A3A3A3", textDecoration: "none" }}>Security Whitepaper →</Link>
            <Link href="/dpa" style={{ color: "#A3A3A3", textDecoration: "none" }}>Data Processing Addendum →</Link>
          </div>
        </div>

      </div>
    </div>
  );
}
