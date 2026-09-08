import React, { useEffect } from "react";
import { Link } from "wouter";
import { ArrowLeft, Shield, Lock, Database, EyeOff, CheckCircle2 } from "lucide-react";

export default function Privacy() {
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
            <Link href="/legal" style={{ color: "#737373", textDecoration: "none" }}>Legal Disclaimers</Link>
            <Link href="/founder" style={{ color: "#10b981", textDecoration: "none" }}>Founder Dossier</Link>
          </div>
        </nav>

        {/* Document Header */}
        <header style={{ marginBottom: "40px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "9999px", padding: "4px 12px", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#D4D4D4", marginBottom: "16px" }}>
            <Shield size={13} style={{ color: "#10b981" }} /> Telemetry Isolation &amp; Privacy Standard
          </div>
          <h1 style={{ fontSize: "36px", fontWeight: "700", letterSpacing: "-0.02em", margin: "0 0 12px 0", color: "#FFFFFF" }}>
            Privacy &amp; Telemetry Data Policy
          </h1>
          <p style={{ color: "#737373", fontSize: "14px", margin: 0 }}>
            Effective Date: September 8, 2026 · Version 2.1 (Enterprise Telemetry Isolation Edition)
          </p>
        </header>

        {/* Core Guarantee Banner */}
        <div style={{ background: "rgba(16, 185, 129, 0.06)", border: "1px solid rgba(16, 185, 129, 0.3)", borderRadius: "6px", padding: "20px", marginBottom: "40px" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
            <EyeOff size={22} style={{ color: "#10b981", flexShrink: 0, marginTop: "2px" }} />
            <div>
              <strong style={{ color: "#34d399", fontSize: "14px", display: "block", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Zero-Model-Weights &amp; Zero-Training-Data Ingestion Guarantee
              </strong>
              <p style={{ fontSize: "13px", lineHeight: "1.6", color: "#D4D4D4", margin: 0 }}>
                GhostLayer is built from the ground up for strict data isolation. Our PyTorch hooks (`ghost_layer`) DO NOT read, capture, serialize, or transmit your model weights, training datasets, raw inputs, token embeddings, prompts, or proprietary loss gradients. Only aggregated numerical hardware metrics and scalar loss deltas leave your execution boundary.
              </p>
            </div>
          </div>
        </div>

        {/* Policy Content */}
        <article style={{ fontSize: "14px", lineHeight: "1.75", color: "#A3A3A3" }}>
          
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              1. What GhostLayer Observes (Numerical Telemetry Only)
            </h2>
            <p>
              When you attach GhostLayer hooks to your PyTorch training loop or interact with our web platform, the system records exclusively non-identifying operational metadata:
            </p>
            <ul style={{ paddingLeft: "20px" }}>
              <li><strong>GPU Hardware Metrics:</strong> GPU core utilization percentage, memory allocated/reserved (VRAM bytes), CUDA device index, compute temperature, and SM activity.</li>
              <li><strong>Execution Step Timing:</strong> Forward pass latency (ms), backward pass latency (ms), optimizer step duration, and data-loader blocking wait time.</li>
              <li><strong>Distributed Interconnect Data:</strong> NCCL collective operation timings (AllReduce, ReduceScatter, AllGather), network bus bandwidth utilization, and packet bottleneck indicators.</li>
              <li><strong>Scalar Loss Trajectory:</strong> Floating-point scalar loss values and step-over-step delta comparisons used exclusively for the 0.10 loss-shift proxy safety gate.</li>
              <li><strong>Platform Contact Info:</strong> Email address, name, company name, and cluster size submitted voluntarily via the pilot audit request form.</li>
            </ul>
          </section>

          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              2. What GhostLayer Strictly Never Observes or Retains
            </h2>
            <ul style={{ paddingLeft: "20px" }}>
              <li><strong>No Model Weights:</strong> GhostLayer never serializes tensor state dictionaries, model parameters, or checkpoint binaries.</li>
              <li><strong>No Dataset Content:</strong> GhostLayer has no visibility into training tokens, text corpora, images, audio, or database records.</li>
              <li><strong>No Gradient Tensors:</strong> Only the single floating-point scalar loss value is compared; multi-gigabyte gradient buffers remain entirely uninspected.</li>
              <li><strong>No Cloud Provider Secret Keys:</strong> GhostLayer does not require or store AWS, GCP, Azure, or SSH private cluster root keys.</li>
            </ul>
          </section>

          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              3. Telemetry Processing &amp; Purpose of Use
            </h2>
            <p>
              All telemetry captured by GhostLayer is used solely to:
            </p>
            <ol style={{ paddingLeft: "20px" }}>
              <li>Provide real-time step profiling, memory headroom analysis, and bottleneck diagnostics.</li>
              <li>Evaluate whether candidate optimization recommendations exceed the 0.10 loss-shift threshold and prevent automated regression.</li>
              <li>Generate structured, tamper-evident audit decision records for your engineering team.</li>
              <li>Compute projected GPU-hour and energy expenditure savings via the FinOps engine.</li>
            </ol>
            <p>
              GhostLayer does not sell, rent, or commercialize your telemetry data to third parties. We do not use Customer telemetry to train foundational AI models.
            </p>
          </section>

          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              4. Data Retention, Security &amp; Deletion
            </h2>
            <p>
              All telemetry logs and pilot audit request submissions are transmitted over TLS 1.3 encrypted connections and stored on access-restricted infrastructure with AES-256 encryption at rest.
            </p>
            <p>
              <strong>Data Deletion:</strong> Customers may request immediate and complete deletion of all associated telemetry logs, decision records, or contact submissions by emailing `bhargavmahadevan@gmail.com` with the subject "Telemetry Purge Request". Purge requests are executed within five (5) business days.
            </p>
          </section>

          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              5. CCPA &amp; GDPR Compliance Rights
            </h2>
            <p>
              Under California Consumer Privacy Act (CCPA), General Data Protection Regulation (GDPR), and applicable privacy frameworks, users have the right to access, rectify, or erase their personal contact details and request confirmation of telemetry isolation.
            </p>
          </section>

          <section style={{ marginBottom: "48px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              6. Data Protection Officer Contact
            </h2>
            <div style={{ background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.1)", padding: "16px", borderRadius: "4px" }}>
              <strong style={{ color: "#FFFFFF", display: "block", marginBottom: "4px" }}>GhostLayer Privacy &amp; Data Security Office</strong>
              <span>Attn: Bhargav Mahadevan</span><br />
              <span>Email: <a href="mailto:bhargavmahadevan@gmail.com" style={{ color: "#10b981", textDecoration: "none" }}>bhargavmahadevan@gmail.com</a> / <a href="mailto:privacy@ghostlayer.ai" style={{ color: "#10b981", textDecoration: "none" }}>privacy@ghostlayer.ai</a></span><br />
              <span>Direct Phone: 832-402-3104</span>
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
            <Link href="/legal" style={{ color: "#A3A3A3", textDecoration: "none" }}>Legal &amp; Empirical Disclaimers →</Link>
          </div>
        </div>

      </div>
    </div>
  );
}
