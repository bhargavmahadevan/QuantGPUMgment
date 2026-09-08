import React, { useState, useEffect } from "react";
import { Link } from "wouter";
import { ShieldCheck, X } from "lucide-react";

export default function LegalConsentBanner() {
  const [acknowledged, setAcknowledged] = useState(true);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("ghostlayer_legal_ack");
      if (!stored) {
        setAcknowledged(false);
      }
    } catch {
      setAcknowledged(false);
    }
  }, []);

  const handleAcknowledge = () => {
    try {
      localStorage.setItem("ghostlayer_legal_ack", "true");
    } catch {}
    setAcknowledged(true);
  };

  if (acknowledged) return null;

  return (
    <aside
      aria-label="Enterprise Legal Notice & Disclaimers"
      style={{
        position: "fixed",
        bottom: "20px",
        right: "20px",
        maxWidth: "440px",
        background: "rgba(10, 10, 10, 0.95)",
        backdropFilter: "blur(12px)",
        border: "1px solid rgba(255, 255, 255, 0.15)",
        boxShadow: "0 12px 36px rgba(0, 0, 0, 0.7)",
        borderRadius: "6px",
        padding: "16px 18px",
        zIndex: 9999,
        fontSize: "12px",
        lineHeight: "1.5",
        color: "#D4D4D4",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        animation: "fadeIn 0.3s ease-out",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <ShieldCheck size={14} style={{ color: "#10b981", flexShrink: 0 }} />
          <strong style={{ color: "#FFFFFF", fontSize: "11px", letterSpacing: "0.06em", textTransform: "uppercase" }}>
            Enterprise Legal &amp; AS-IS Notice
          </strong>
        </div>
        <button
          onClick={handleAcknowledge}
          aria-label="Dismiss legal notice"
          style={{
            background: "none",
            border: "none",
            color: "#737373",
            cursor: "pointer",
            padding: "2px",
            display: "inline-flex",
          }}
        >
          <X size={14} />
        </button>
      </div>

      <p style={{ margin: 0, color: "#A3A3A3" }}>
        GhostLayer telemetry hooks, decision engines, and simulations are provided under strict{" "}
        <strong style={{ color: "#FFFFFF" }}>"AS IS" terms with zero consequential liability</strong>. Physical benchmark receipts reflect isolated historical runs, not fleet-wide guarantees.
      </p>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "10px", marginTop: "2px" }}>
        <div style={{ display: "flex", gap: "10px", fontSize: "11px" }}>
          <Link href="/terms" style={{ color: "#D4D4D4", textDecoration: "underline" }}>
            Terms
          </Link>
          <Link href="/privacy" style={{ color: "#D4D4D4", textDecoration: "underline" }}>
            Privacy
          </Link>
          <Link href="/legal" style={{ color: "#10b981", textDecoration: "underline" }}>
            Disclaimers
          </Link>
        </div>
        <button
          onClick={handleAcknowledge}
          style={{
            background: "#FFFFFF",
            color: "#050505",
            border: "none",
            borderRadius: "4px",
            padding: "4px 12px",
            fontSize: "11px",
            fontWeight: 600,
            cursor: "pointer",
            letterSpacing: "0.04em",
          }}
        >
          Acknowledge
        </button>
      </div>
    </aside>
  );
}
