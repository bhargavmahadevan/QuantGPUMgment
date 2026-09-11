import React, { useEffect } from "react";
import { Link } from "wouter";
import { ArrowLeft, ShieldCheck, Lock, KeyRound, Terminal, CheckCircle2, AlertOctagon, HeartHandshake } from "lucide-react";

export default function Security() {
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
            <Link href="/dpa" style={{ color: "#737373", textDecoration: "none" }}>DPA</Link>
            <Link href="/legal" style={{ color: "#737373", textDecoration: "none" }}>Disclaimers</Link>
            <Link href="/founder" style={{ color: "#10b981", textDecoration: "none" }}>Founder Dossier</Link>
          </div>
        </nav>

        {/* Header */}
        <header style={{ marginBottom: "40px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.3)", borderRadius: "9999px", padding: "4px 12px", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#10b981", marginBottom: "16px" }}>
            <ShieldCheck size={13} /> Institutional Security &amp; Vulnerability Disclosure
          </div>
          <h1 style={{ fontSize: "36px", fontWeight: "700", letterSpacing: "-0.02em", margin: "0 0 12px 0", color: "#FFFFFF" }}>
            Security Architecture &amp; Vulnerability Disclosure (VDP)
          </h1>
          <p style={{ color: "#737373", fontSize: "14px", margin: 0 }}>
            Published RFC 9116 security standard, white-hat safe harbor, and cryptographic telemetry isolation.
          </p>
        </header>

        {/* Safe Harbor Banner */}
        <div style={{ background: "rgba(16, 185, 129, 0.05)", border: "1px solid rgba(16, 185, 129, 0.25)", borderRadius: "6px", padding: "20px", marginBottom: "40px" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
            <HeartHandshake size={22} style={{ color: "#10b981", flexShrink: 0, marginTop: "2px" }} />
            <div>
              <strong style={{ color: "#34d399", fontSize: "14px", display: "block", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                White-Hat Researcher Safe Harbor Commitment
              </strong>
              <p style={{ fontSize: "13px", lineHeight: "1.6", color: "#D4D4D4", margin: 0 }}>
                GhostLayer regards cybersecurity research conducted in good faith as vital to the ecosystem. If you discover a vulnerability and report it responsibly in compliance with this policy, GhostLayer will not initiate legal action against you under the Computer Fraud and Abuse Act (CFAA), the Digital Millennium Copyright Act (DMCA), or state computer crime statutes.
              </p>
            </div>
          </div>
        </div>

        {/* Security Article */}
        <article style={{ fontSize: "14px", lineHeight: "1.75", color: "#A3A3A3" }}>

          {/* Section 1: Zero Knowledge Architecture */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              1. Zero-Knowledge Cryptographic Telemetry Architecture
            </h2>
            <p>
              GhostLayer's runtime hook architecture is specifically designed to eliminate data leakage risks for proprietary enterprise model architectures and confidential training corpuses:
            </p>
            <div style={{ background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.08)", padding: "18px", borderRadius: "6px", margin: "16px 0" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", fontSize: "12px" }}>
                <div>
                  <strong style={{ color: "#10b981", display: "block", marginBottom: "6px" }}>✓ WHAT IS CAPTURED:</strong>
                  <ul style={{ paddingLeft: "16px", margin: 0, color: "#D4D4D4" }}>
                    <li>GPU SM &amp; VRAM hardware occupancy counters</li>
                    <li>Step latency metrics (forward/backward timings in ms)</li>
                    <li>Single floating-point loss scalar deltas (0.10 proxy)</li>
                    <li>Inter-node NCCL collective communication duration</li>
                  </ul>
                </div>
                <div>
                  <strong style={{ color: "#f43f5e", display: "block", marginBottom: "6px" }}>✕ WHAT IS NEVER TOUCHED:</strong>
                  <ul style={{ paddingLeft: "16px", margin: 0, color: "#D4D4D4" }}>
                    <li>Model weights &amp; parameter checkpoint files</li>
                    <li>Training datasets, text tokens, embeddings, or prompts</li>
                    <li>Multi-gigabyte backpropagation gradient buffers</li>
                    <li>Cluster root credentials or SSH private keys</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Section 2: Technical and Operational Controls */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              2. Technical &amp; Operational Security Controls
            </h2>
            <ul style={{ paddingLeft: "20px" }}>
              <li><strong>Zero Inbound Listening Daemons:</strong> GhostLayer's Python hooks do not bind inbound network ports, listen on TCP sockets, or expose open debugging interfaces on training nodes.</li>
              <li><strong>Encryption in Transit:</strong> All communication with the GhostLayer telemetry collector and dashboard occurs over Transport Layer Security (TLS 1.3) with modern cipher suites.</li>
              <li><strong>Encryption at Rest:</strong> Audit decision records and pilot submissions are encrypted at rest using AES-256 standards.</li>
              <li><strong>Least Privilege Isolation:</strong> Context hooks run entirely in the user-space Python process of the training job without requiring root, kernel-level eBPF privileges, or system administrator overrides.</li>
            </ul>
          </section>

          {/* Section 3: Vulnerability Disclosure Policy (VDP) */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              3. Vulnerability Disclosure &amp; Reporting Guidelines (RFC 9116)
            </h2>
            <p>
              To maintain safe harbor protection, security researchers must comply with the following research ground rules:
            </p>
            <ol style={{ paddingLeft: "20px" }}>
              <li><strong>Do No Harm:</strong> Do not access, modify, or destroy customer or enterprise training data, model checkpoints, or telemetry logs.</li>
              <li><strong>No Denial of Service (DoS):</strong> Do not execute resource-exhaustion, distributed denial-of-service, or brute-force testing against active production clusters or API servers.</li>
              <li><strong>Coordinated Disclosure:</strong> Provide GhostLayer a reasonable window of at least ninety (90) days to triage, validate, and deploy a patch before disclosing any vulnerability details publicly.</li>
              <li><strong>Single Channel:</strong> Submit all vulnerability reports directly to our dedicated security contact.</li>
            </ol>
          </section>

          {/* Section 4: SLA & Response Commitments */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              4. Response Commitments &amp; SLAs
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "12px", margin: "16px 0" }}>
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", padding: "14px", borderRadius: "4px" }}>
                <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase" }}>First Response</span>
                <strong style={{ display: "block", color: "#FFFFFF", fontSize: "18px", marginTop: "4px" }}>Within 24 Hours</strong>
                <span style={{ fontSize: "12px", color: "#A3A3A3" }}>Initial human receipt and ticket assignment.</span>
              </div>
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", padding: "14px", borderRadius: "4px" }}>
                <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase" }}>Triage &amp; Severity</span>
                <strong style={{ display: "block", color: "#FFFFFF", fontSize: "18px", marginTop: "4px" }}>Within 72 Hours</strong>
                <span style={{ fontSize: "12px", color: "#A3A3A3" }}>CVSS v3.1 scoring and reproduction verification.</span>
              </div>
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", padding: "14px", borderRadius: "4px" }}>
                <span style={{ fontSize: "11px", color: "#737373", textTransform: "uppercase" }}>Remediation</span>
                <strong style={{ display: "block", color: "#10b981", fontSize: "18px", marginTop: "4px" }}>Priority Patch Cycle</strong>
                <span style={{ fontSize: "12px", color: "#A3A3A3" }}>Hotfix rollout and release notes attribution.</span>
              </div>
            </div>
          </section>

          {/* Section 5: Security Contact & security.txt Info */}
          <section style={{ marginBottom: "48px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              5. Official Security Contact
            </h2>
            <div style={{ background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.1)", padding: "16px", borderRadius: "4px" }}>
              <strong style={{ color: "#FFFFFF", display: "block", marginBottom: "6px" }}>GhostLayer Product Security Incident Response (PSIRT)</strong>
              <div style={{ fontFamily: '"DM Mono", monospace', fontSize: "12px", color: "#D4D4D4", lineHeight: "1.8" }}>
                Contact: <a href="mailto:security@ghostlayer.ai" style={{ color: "#10b981", textDecoration: "none" }}>security@ghostlayer.ai</a><br />
                Direct Lead: <a href="mailto:bhargavmahadevan@gmail.com" style={{ color: "#10b981", textDecoration: "none" }}>bhargavmahadevan@gmail.com</a><br />
                Emergency Hotline: +1 (832) 402-3104<br />
                Preferred Language: English (en-US)<br />
                Policy Canonical: https://ghostlayer.ai/security
              </div>
            </div>
          </section>

        </article>

        {/* Footer Navigation */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "13px" }}>
          <Link href="/" style={{ color: "#737373", textDecoration: "none" }}>
            ← Back to GhostLayer Orbit
          </Link>
          <div style={{ display: "flex", gap: "20px" }}>
            <Link href="/compliance" style={{ color: "#A3A3A3", textDecoration: "none" }}>Compliance &amp; Export →</Link>
            <Link href="/dpa" style={{ color: "#A3A3A3", textDecoration: "none" }}>Data Processing Addendum →</Link>
          </div>
        </div>

      </div>
    </div>
  );
}
