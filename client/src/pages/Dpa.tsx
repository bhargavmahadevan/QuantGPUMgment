import React, { useEffect } from "react";
import { Link } from "wouter";
import { ArrowLeft, FileText, Database, ShieldCheck, Server, Lock, CheckCircle2, UserCheck } from "lucide-react";

export default function Dpa() {
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
            <Link href="/compliance" style={{ color: "#737373", textDecoration: "none" }}>Compliance</Link>
            <Link href="/security" style={{ color: "#737373", textDecoration: "none" }}>Security</Link>
            <Link href="/legal" style={{ color: "#737373", textDecoration: "none" }}>Disclaimers</Link>
            <Link href="/founder" style={{ color: "#10b981", textDecoration: "none" }}>Founder Dossier</Link>
          </div>
        </nav>

        {/* Header */}
        <header style={{ marginBottom: "40px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "9999px", padding: "4px 12px", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#D4D4D4", marginBottom: "16px" }}>
            <FileText size={13} style={{ color: "#10b981" }} /> GDPR Article 28 &amp; CCPA Addendum
          </div>
          <h1 style={{ fontSize: "36px", fontWeight: "700", letterSpacing: "-0.02em", margin: "0 0 12px 0", color: "#FFFFFF" }}>
            Data Processing Addendum (DPA) &amp; Subprocessors Registry
          </h1>
          <p style={{ color: "#737373", fontSize: "14px", margin: 0 }}>
            Standard Contractual Clauses (SCCs), Technical &amp; Organizational Measures (TOMs), and active vendor directory.
          </p>
        </header>

        {/* Notice Banner */}
        <div style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "6px", padding: "20px", marginBottom: "40px" }}>
          <strong style={{ color: "#FFFFFF", fontSize: "13px", display: "block", marginBottom: "4px" }}>
            Scope of Processing Under GhostLayer:
          </strong>
          <p style={{ fontSize: "12px", lineHeight: "1.6", color: "#A3A3A3", margin: 0 }}>
            This Data Processing Addendum governs the processing of numerical telemetry metadata and organizational contact details in connection with the GhostLayer Master Services Agreement. GhostLayer acts as a <strong>Data Processor (or Service Provider)</strong>; Customer remains the <strong>Data Controller</strong>.
          </p>
        </div>

        {/* Body Content */}
        <article style={{ fontSize: "14px", lineHeight: "1.75", color: "#A3A3A3" }}>

          {/* Section 1: Data Subject & Category Scope */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              1. Scope, Subject Matter &amp; Nature of Processing
            </h2>
            <p>
              In performing the Services, GhostLayer processes exclusively two discrete categories of data:
            </p>
            <ol style={{ paddingLeft: "20px" }}>
              <li><strong>Hardware &amp; Execution Telemetry:</strong> Anonymized numerical scalar metrics consisting of GPU utilization percentages, memory bytes allocated/cached, step execution latencies (forward/backward/optimizer in milliseconds), NCCL network collective durations, and relative loss scalar deltas.</li>
              <li><strong>Contact &amp; Administrative Information:</strong> Customer contact name, business email, organization name, and cluster hardware descriptions voluntarily supplied via the pilot intake interface.</li>
            </ol>
            <div style={{ background: "rgba(16, 185, 129, 0.05)", border: "1px solid rgba(16, 185, 129, 0.25)", padding: "14px", borderRadius: "4px", margin: "16px 0" }}>
              <span style={{ color: "#34d399", fontWeight: 600 }}>Strict Exclusion of Customer Datasets &amp; Weights:</span>
              <p style={{ fontSize: "12px", color: "#D4D4D4", margin: "4px 0 0 0" }}>
                GhostLayer's software hooks are technically incapable of processing, parsing, or retaining Customer's model weights, weights checkpoints, input datasets, prompts, or personal data contained in training tokens.
              </p>
            </div>
          </section>

          {/* Section 2: Technical and Organizational Measures (TOMs) */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              2. Technical &amp; Organizational Security Measures (TOMs)
            </h2>
            <ul style={{ paddingLeft: "20px" }}>
              <li><strong>Access Control:</strong> Telemetry endpoints and pilot storage utilize strict principle-of-least-privilege access, multi-factor authentication (MFA), and audit logging.</li>
              <li><strong>Cryptographic Controls:</strong> All external API communications are enforced via TLS 1.3. Persistent decision logs are encrypted at rest with AES-256.</li>
              <li><strong>Pseudonymization &amp; Ephemeral Processing:</strong> Step timing metrics and memory snapshots are held ephemerally in RAM during training runs and aggregated prior to permanent record archiving.</li>
              <li><strong>Disaster Recovery &amp; Redundancy:</strong> Distributed geographic backups with automated failover and daily data integrity validations.</li>
            </ul>
          </section>

          {/* Section 3: Subprocessors Registry */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              3. Authorized Subprocessors Registry
            </h2>
            <p>
              Customer provides general written authorization for GhostLayer to engage the following verified third-party infrastructure subprocessors:
            </p>
            
            <div style={{ overflowX: "auto", margin: "16px 0" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.03)" }}>
                    <th style={{ padding: "10px", color: "#FFFFFF" }}>Subprocessor</th>
                    <th style={{ padding: "10px", color: "#FFFFFF" }}>Purpose</th>
                    <th style={{ padding: "10px", color: "#FFFFFF" }}>Data Processed</th>
                    <th style={{ padding: "10px", color: "#FFFFFF" }}>Location</th>
                    <th style={{ padding: "10px", color: "#FFFFFF" }}>Certifications</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <td style={{ padding: "10px", color: "#FFFFFF", fontWeight: 500 }}>Render Inc.</td>
                    <td style={{ padding: "10px" }}>Cloud Application &amp; API Server Hosting</td>
                    <td style={{ padding: "10px" }}>Pilot audit requests, report tokens, telemetry API</td>
                    <td style={{ padding: "10px" }}>United States</td>
                    <td style={{ padding: "10px", color: "#10b981" }}>SOC 2 Type II, ISO 27001</td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <td style={{ padding: "10px", color: "#FFFFFF", fontWeight: 500 }}>Google Cloud / Firebase</td>
                    <td style={{ padding: "10px" }}>Global Edge CDN &amp; Static Web Delivery</td>
                    <td style={{ padding: "10px" }}>Static web assets, browser cached bundles</td>
                    <td style={{ padding: "10px" }}>Global Edge</td>
                    <td style={{ padding: "10px", color: "#10b981" }}>ISO 27001, SOC 1/2/3</td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <td style={{ padding: "10px", color: "#FFFFFF", fontWeight: 500 }}>Customer GPU Cluster</td>
                    <td style={{ padding: "10px" }}>Local Python Runtime (`ghost_layer`)</td>
                    <td style={{ padding: "10px" }}>In-memory step timings &amp; GPU counters</td>
                    <td style={{ padding: "10px" }}>Customer Premises</td>
                    <td style={{ padding: "10px", color: "#10b981" }}>Air-Gapped / Zero Exfil</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p style={{ fontSize: "12px", color: "#737373" }}>
              GhostLayer will notify customers at least thirty (30) days in advance of onboarding any new infrastructure subprocessor.
            </p>
          </section>

          {/* Section 4: Data Subject Rights & Purge SLA */}
          <section style={{ marginBottom: "48px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              4. Data Subject Rights &amp; 5-Day Purge Protocol
            </h2>
            <p>
              In compliance with GDPR Chapter III and CCPA § 1798.105, Customer or authorized data subjects may demand immediate verification, export, or permanent deletion of all stored records by emailing our Data Protection Officer:
            </p>
            <div style={{ background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.1)", padding: "16px", borderRadius: "4px", marginTop: "12px" }}>
              <strong style={{ color: "#FFFFFF", display: "block", marginBottom: "4px" }}>GhostLayer Data Protection &amp; DPA Officer</strong>
              <span>Attn: Bhargav Mahadevan</span><br />
              <span>DPA Execution: <a href="mailto:dpa@ghostlayer.ai" style={{ color: "#10b981", textDecoration: "none" }}>dpa@ghostlayer.ai</a></span><br />
              <span>Direct: <a href="mailto:bhargavmahadevan@gmail.com" style={{ color: "#10b981", textDecoration: "none" }}>bhargavmahadevan@gmail.com</a></span><br />
              <span>Turnaround Commitment: Execution of deletion within five (5) business days with cryptographically signed confirmation.</span>
            </div>
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
