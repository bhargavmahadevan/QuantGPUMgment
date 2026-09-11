import React, { useState, useEffect } from "react";
import { Link } from "wouter";
import { ShieldCheck, X, Sliders, Lock, CheckCircle2, AlertOctagon } from "lucide-react";

export function openConsentPreferences() {
  window.dispatchEvent(new CustomEvent("open-ghostlayer-consent-modal"));
}

export default function LegalConsentBanner() {
  const [acknowledged, setAcknowledged] = useState(true);
  const [showPreferences, setShowPreferences] = useState(false);
  const [analyticsEnabled, setAnalyticsEnabled] = useState(false);
  const [doNotSellEnabled, setDoNotSellEnabled] = useState(true);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("ghostlayer_legal_ack");
      const storedAnalytics = localStorage.getItem("ghostlayer_pref_analytics");
      const storedDoNotSell = localStorage.getItem("ghostlayer_pref_donotsell");

      if (!stored) {
        setAcknowledged(false);
      }
      if (storedAnalytics === "true") {
        setAnalyticsEnabled(true);
      }
      if (storedDoNotSell === "false") {
        setDoNotSellEnabled(false);
      }
    } catch {
      setAcknowledged(false);
    }

    const handleOpenModal = () => {
      setShowPreferences(true);
    };

    window.addEventListener("open-ghostlayer-consent-modal", handleOpenModal);
    return () => {
      window.removeEventListener("open-ghostlayer-consent-modal", handleOpenModal);
    };
  }, []);

  const handleAcceptAll = () => {
    try {
      localStorage.setItem("ghostlayer_legal_ack", "true");
      localStorage.setItem("ghostlayer_pref_analytics", "true");
      localStorage.setItem("ghostlayer_pref_donotsell", "true");
    } catch {}
    setAnalyticsEnabled(true);
    setAcknowledged(true);
    setShowPreferences(false);
  };

  const handleEssentialOnly = () => {
    try {
      localStorage.setItem("ghostlayer_legal_ack", "true");
      localStorage.setItem("ghostlayer_pref_analytics", "false");
      localStorage.setItem("ghostlayer_pref_donotsell", "true");
    } catch {}
    setAnalyticsEnabled(false);
    setAcknowledged(true);
    setShowPreferences(false);
  };

  const handleSavePreferences = () => {
    try {
      localStorage.setItem("ghostlayer_legal_ack", "true");
      localStorage.setItem("ghostlayer_pref_analytics", analyticsEnabled ? "true" : "false");
      localStorage.setItem("ghostlayer_pref_donotsell", doNotSellEnabled ? "true" : "false");
    } catch {}
    setAcknowledged(true);
    setShowPreferences(false);
  };

  return (
    <>
      {/* ── Banner (Shown until acknowledged) ─────────────────────────────────── */}
      {!acknowledged && (
        <aside
          aria-label="Enterprise Legal Notice & Disclaimers"
          style={{
            position: "fixed",
            bottom: "20px",
            right: "20px",
            maxWidth: "460px",
            background: "rgba(10, 10, 10, 0.96)",
            backdropFilter: "blur(14px)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            boxShadow: "0 16px 40px rgba(0, 0, 0, 0.8)",
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
              <ShieldCheck size={15} style={{ color: "#10b981", flexShrink: 0 }} />
              <strong style={{ color: "#FFFFFF", fontSize: "11px", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                Enterprise Legal &amp; Telemetry Notice
              </strong>
            </div>
            <button
              onClick={handleEssentialOnly}
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

          <p style={{ margin: 0, color: "#A3A3A3", fontSize: "12px" }}>
            GhostLayer operates under strict <strong style={{ color: "#FFFFFF" }}>"AS IS" warranty terms</strong>, a $100 liability cap, and a <strong style={{ color: "#FFFFFF" }}>Zero-Model-Weights Ingestion</strong> standard. We do not inspect weights, tokens, or prompts.
          </p>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px", flexWrap: "wrap", paddingTop: "4px" }}>
            <div style={{ display: "flex", gap: "10px", fontSize: "11px" }}>
              <Link href="/terms" style={{ color: "#D4D4D4", textDecoration: "underline" }}>
                Terms
              </Link>
              <Link href="/privacy" style={{ color: "#D4D4D4", textDecoration: "underline" }}>
                Privacy
              </Link>
              <Link href="/compliance" style={{ color: "#D4D4D4", textDecoration: "underline" }}>
                Export (EAR)
              </Link>
              <Link href="/legal" style={{ color: "#10b981", textDecoration: "underline" }}>
                Disclaimers
              </Link>
            </div>

            <div style={{ display: "flex", gap: "6px" }}>
              <button
                onClick={() => setShowPreferences(true)}
                style={{
                  background: "transparent",
                  color: "#D4D4D4",
                  border: "1px solid rgba(255, 255, 255, 0.2)",
                  borderRadius: "4px",
                  padding: "4px 10px",
                  fontSize: "11px",
                  fontWeight: 500,
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px",
                }}
              >
                <Sliders size={11} /> Preferences
              </button>
              <button
                onClick={handleAcceptAll}
                style={{
                  background: "#FFFFFF",
                  color: "#050505",
                  border: "none",
                  borderRadius: "4px",
                  padding: "4px 12px",
                  fontSize: "11px",
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                Accept All
              </button>
            </div>
          </div>
        </aside>
      )}

      {/* ── Preferences Modal (GDPR / CCPA / Telemetry Manager) ──────────────── */}
      {showPreferences && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="consent-modal-title"
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100vw",
            height: "100vh",
            background: "rgba(0, 0, 0, 0.75)",
            backdropFilter: "blur(6px)",
            zIndex: 10000,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "20px",
          }}
        >
          <div
            style={{
              background: "#0A0A0A",
              border: "1px solid rgba(255, 255, 255, 0.15)",
              borderRadius: "8px",
              maxWidth: "540px",
              width: "100%",
              boxShadow: "0 24px 48px rgba(0, 0, 0, 0.9)",
              color: "#F5F5F5",
              padding: "24px",
              display: "flex",
              flexDirection: "column",
              gap: "20px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <Sliders size={18} style={{ color: "#10b981" }} />
                <h2 id="consent-modal-title" style={{ margin: 0, fontSize: "16px", fontWeight: 600, color: "#FFFFFF" }}>
                  Telemetry &amp; Privacy Preferences
                </h2>
              </div>
              <button
                onClick={() => setShowPreferences(false)}
                aria-label="Close preferences"
                style={{ background: "none", border: "none", color: "#737373", cursor: "pointer", padding: "4px" }}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ margin: 0, fontSize: "12px", lineHeight: "1.6", color: "#A3A3A3" }}>
              GhostLayer provides transparency into telemetry collection. You can manage your preferences below in accordance with GDPR, CCPA/CPRA, and enterprise security policies.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              
              {/* Category 1: Strictly Necessary */}
              <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "6px", padding: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <Lock size={13} style={{ color: "#10b981" }} />
                    <strong style={{ fontSize: "13px", color: "#FFFFFF" }}>Strictly Necessary Telemetry</strong>
                  </div>
                  <span style={{ fontSize: "11px", color: "#10b981", background: "rgba(16, 185, 129, 0.1)", padding: "2px 8px", borderRadius: "9999px" }}>
                    Always Active
                  </span>
                </div>
                <p style={{ fontSize: "11px", color: "#737373", margin: 0, lineHeight: "1.5" }}>
                  Essential step timing (ms), GPU VRAM allocation counters, and loss-shift proxy comparison. Non-identifying and required for cluster safety and optimization audit logging.
                </p>
              </div>

              {/* Category 2: Diagnostic & Lab Performance */}
              <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "6px", padding: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <strong style={{ fontSize: "13px", color: "#FFFFFF" }}>Diagnostic &amp; Lab Performance</strong>
                  <input
                    type="checkbox"
                    checked={analyticsEnabled}
                    onChange={(e) => setAnalyticsEnabled(e.target.checked)}
                    style={{ accentColor: "#10b981", cursor: "pointer", width: "16px", height: "16px" }}
                  />
                </div>
                <p style={{ fontSize: "11px", color: "#737373", margin: 0, lineHeight: "1.5" }}>
                  Anonymous client browser frame-rate diagnostics and lab simulator rendering telemetry to improve platform stability.
                </p>
              </div>

              {/* Category 3: CCPA / CPRA Do Not Sell */}
              <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "6px", padding: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <strong style={{ fontSize: "13px", color: "#FFFFFF" }}>Do Not Sell or Share My Information (CCPA)</strong>
                  <span style={{ fontSize: "11px", color: "#10b981", background: "rgba(16, 185, 129, 0.1)", padding: "2px 8px", borderRadius: "9999px" }}>
                    Enforced by Default
                  </span>
                </div>
                <p style={{ fontSize: "11px", color: "#737373", margin: 0, lineHeight: "1.5" }}>
                  GhostLayer never sells, rents, or monetizes customer contact or telemetry data. This safeguard is hardcoded and permanently active.
                </p>
              </div>

            </div>

            {/* Modal Actions */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid rgba(255, 255, 255, 0.08)", paddingTop: "16px" }}>
              <button
                onClick={handleEssentialOnly}
                style={{
                  background: "transparent",
                  color: "#A3A3A3",
                  border: "1px solid rgba(255, 255, 255, 0.2)",
                  borderRadius: "4px",
                  padding: "6px 14px",
                  fontSize: "12px",
                  cursor: "pointer",
                }}
              >
                Reject Non-Essential
              </button>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  onClick={handleSavePreferences}
                  style={{
                    background: "#FFFFFF",
                    color: "#050505",
                    border: "none",
                    borderRadius: "4px",
                    padding: "6px 16px",
                    fontSize: "12px",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  Save Preferences
                </button>
              </div>
            </div>

          </div>
        </div>
      )}
    </>
  );
}
