import React, { useEffect } from "react";
import { Link } from "wouter";
import { 
  ArrowLeft, 
  ArrowUpRight, 
  Mail, 
  Phone, 
  ShieldCheck, 
  Sparkles, 
  Terminal, 
  CheckCircle2, 
  Send,
  Zap,
  Globe,
  Cpu
} from "lucide-react";

export default function Founder() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div style={{ background: "#050505", minHeight: "100vh", color: "#FFFFFF", fontFamily: '"DM Mono", monospace' }}>
      {/* Header Bar */}
      <header style={{
        height: "76px",
        padding: "0 clamp(20px, 4vw, 48px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
        background: "rgba(0, 0, 0, 0.85)",
        backdropFilter: "blur(20px)",
        position: "sticky",
        top: 0,
        zIndex: 20
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
          <Link href="/" style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            color: "#A3A3A3",
            textDecoration: "none",
            fontSize: "0.75rem",
            textTransform: "uppercase",
            letterSpacing: "0.04em",
            transition: "color 0.15s ease"
          }}>
            <ArrowLeft size={14} /> Back to Orbit
          </Link>
          <span style={{ color: "rgba(255, 255, 255, 0.15)" }}>/</span>
          <span style={{ fontSize: "0.75rem", color: "#10b981", fontWeight: 600 }}>FOUNDER DOSSIER</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <Link href="/pilot" style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            background: "#FFFFFF",
            color: "#000000",
            padding: "8px 16px",
            fontSize: "0.72rem",
            fontWeight: 700,
            textDecoration: "none",
            letterSpacing: "0.04em",
            textTransform: "uppercase"
          }}>
            Request Audit <ArrowUpRight size={13} />
          </Link>
        </div>
      </header>

      {/* Main Founder Stage */}
      <main style={{ maxWidth: "1120px", margin: "0 auto", padding: "clamp(48px, 6vw, 96px) 24px" }}>
        {/* Eyebrow */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px" }}>
          <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#10b981", display: "inline-block", boxShadow: "0 0 8px rgba(16, 185, 129, 0.8)" }} />
          <span style={{ fontSize: "0.68rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "#737373" }}>
            FOUNDER &amp; ARCHITECT / GHOSTLAYER
          </span>
        </div>

        {/* Title Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "48px", alignItems: "start", marginBottom: "64px" }}>
          {/* Portrait Figure */}
          <div>
            <figure style={{
              position: "relative",
              margin: 0,
              overflow: "hidden",
              border: "1px solid rgba(255, 255, 255, 0.15)",
              background: "#0A0A0A",
              borderRadius: "0",
              boxShadow: "0 24px 60px rgba(0, 0, 0, 0.7)"
            }}>
              <img 
                src="/assets/ghostlayer-bhargav-founder.jpg" 
                alt="Portrait of Bhargav Mahadevan"
                style={{
                  width: "100%",
                  height: "auto",
                  minHeight: "440px",
                  maxHeight: "560px",
                  objectFit: "cover",
                  objectPosition: "50% 35%",
                  display: "block",
                  filter: "saturate(0.9) contrast(1.05) brightness(0.95)"
                }}
              />
              <figcaption style={{
                position: "absolute",
                inset: "auto 0 0 0",
                padding: "24px 20px 18px",
                background: "linear-gradient(180deg, transparent 0%, rgba(5, 5, 5, 0.95) 80%)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-end"
              }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: "0.82rem", color: "#FFFFFF", letterSpacing: "0.06em" }}>BHARGAV MAHADEVAN</div>
                  <div style={{ fontSize: "0.62rem", color: "#10b981", marginTop: "2px", letterSpacing: "0.08em" }}>FOUNDER &amp; BUILDER</div>
                </div>
                <span style={{ fontSize: "0.55rem", color: "#737373", letterSpacing: "0.08em" }}>EST. 2026</span>
              </figcaption>
            </figure>

            {/* Quick Contact Box */}
            <div style={{
              marginTop: "20px",
              display: "grid",
              gap: "10px"
            }}>
              <a 
                href="mailto:bhargavmahadevan@gmail.com"
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "12px 16px",
                  background: "#0F0F0F",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  color: "#FFFFFF",
                  textDecoration: "none",
                  fontSize: "0.78rem",
                  transition: "all 0.15s ease"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <Mail size={15} color="#10b981" />
                  <span>bhargavmahadevan@gmail.com</span>
                </div>
                <ArrowUpRight size={14} color="#737373" />
              </a>

              <a 
                href="tel:+18324023104"
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "12px 16px",
                  background: "#0F0F0F",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  color: "#FFFFFF",
                  textDecoration: "none",
                  fontSize: "0.78rem",
                  transition: "all 0.15s ease"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <Phone size={15} color="#10b981" />
                  <span>832-402-3104</span>
                </div>
                <ArrowUpRight size={14} color="#737373" />
              </a>
            </div>
          </div>

          {/* Copy and Vision */}
          <div>
            <h1 style={{
              fontFamily: '"Instrument Serif", Georgia, serif',
              fontSize: "clamp(3.4rem, 6vw, 5.8rem)",
              fontWeight: 400,
              lineHeight: 0.88,
              letterSpacing: "-0.05em",
              margin: "0 0 24px",
              color: "#FFFFFF"
            }}>
              Bhargav<br />
              <em style={{ color: "#F5F5F5" }}>Mahadevan.</em>
            </h1>

            <blockquote style={{
              margin: "0 0 32px",
              padding: "0 0 0 18px",
              borderLeft: "2px solid #10b981",
              fontFamily: '"Instrument Serif", Georgia, serif',
              fontSize: "clamp(1.4rem, 2vw, 1.85rem)",
              fontStyle: "italic",
              lineHeight: 1.28,
              color: "#F0F4F0",
              letterSpacing: "-0.02em"
            }}>
              “Visualization is the first and most important step to boundless momentum.”
            </blockquote>

            <div style={{ color: "#A3A3A3", fontSize: "0.92rem", lineHeight: 1.75, display: "grid", gap: "18px" }}>
              <p>
                GhostLayer started as an idea that turned into an obsession. The project is built around a simple premise: make LLM training cheaper, more effective, and more accessible to more people.
              </p>
              <p>
                I’m building to make AI better and more affordable for everybody, and I’m learning as I go. Machine learning infrastructure has lived in the dark for too long: teams pay thousands of dollars per GPU-hour without visibility into whether memory fragmentation, I/O stalls, or catastrophic parameter spikes are burning their budget.
              </p>
              <p>
                Every feature in GhostLayer is engineered from first principles: deterministic 0.10 loss-shift protection, non-invasive PyTorch context hooks, and mathematical variance containment that preserves hard-earned capital.
              </p>
              <p>
                I’m always open to a real conversation. Whether you’re running an 8-GPU cluster or designing a frontier training superpod, reach out directly.
              </p>
            </div>

            <div style={{ marginTop: "32px", display: "flex", gap: "14px", flexWrap: "wrap" }}>
              <a 
                href="mailto:bhargavmahadevan@gmail.com?subject=GhostLayer%20conversation"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "8px",
                  background: "#10b981",
                  color: "#000000",
                  padding: "12px 22px",
                  fontWeight: 700,
                  fontSize: "0.78rem",
                  textDecoration: "none",
                  letterSpacing: "0.04em",
                  textTransform: "uppercase",
                  boxShadow: "0 0 20px rgba(16, 185, 129, 0.3)"
                }}
              >
                <Send size={14} /> Start a Conversation
              </a>
              <Link 
                href="/pilot"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "8px",
                  background: "transparent",
                  color: "#FFFFFF",
                  border: "1px solid rgba(255, 255, 255, 0.3)",
                  padding: "12px 22px",
                  fontSize: "0.78rem",
                  textDecoration: "none",
                  letterSpacing: "0.04em",
                  textTransform: "uppercase"
                }}
              >
                Request 48-Hour Pilot
              </Link>
            </div>
          </div>
        </div>

        {/* Founder Principles Section */}
        <section style={{
          borderTop: "1px solid rgba(255, 255, 255, 0.1)",
          paddingTop: "56px",
          marginTop: "64px"
        }}>
          <div style={{ marginBottom: "32px" }}>
            <span style={{ fontSize: "0.65rem", color: "#f43f5e", letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 700 }}>
              FOUNDER PRINCIPLES
            </span>
            <h2 style={{
              fontFamily: '"Instrument Serif", Georgia, serif',
              fontSize: "clamp(2.2rem, 3.5vw, 3.4rem)",
              fontWeight: 400,
              margin: "8px 0 0",
              letterSpacing: "-0.04em"
            }}>
              How I build—and lead—at GhostLayer.
            </h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "24px" }}>
            <div style={{
              background: "#0D0D0D",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "24px"
            }}>
              <span style={{ color: "#10b981", fontSize: "0.75rem", fontWeight: 700 }}>01</span>
              <h3 style={{
                fontFamily: '"Instrument Serif", Georgia, serif',
                fontSize: "1.45rem",
                fontWeight: 400,
                margin: "12px 0 8px",
                color: "#FFFFFF"
              }}>
                Build stronger leaders around you.
              </h3>
              <p style={{ color: "#737373", fontSize: "0.82rem", lineHeight: 1.65, margin: 0 }}>
                Give people the context, responsibility, and confidence to lead without you. True scale comes from multiplying high-agency decision makers who own the mission.
              </p>
            </div>

            <div style={{
              background: "#0D0D0D",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "24px"
            }}>
              <span style={{ color: "#10b981", fontSize: "0.75rem", fontWeight: 700 }}>02</span>
              <h3 style={{
                fontFamily: '"Instrument Serif", Georgia, serif',
                fontSize: "1.45rem",
                fontWeight: 400,
                margin: "12px 0 8px",
                color: "#FFFFFF"
              }}>
                Critique should create momentum.
              </h3>
              <p style={{ color: "#737373", fontSize: "0.82rem", lineHeight: 1.65, margin: 0 }}>
                Be direct about what needs work, then make the next improvement clear. Feedback is only as valuable as the velocity it unlocks.
              </p>
            </div>

            <div style={{
              background: "#0D0D0D",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "24px"
            }}>
              <span style={{ color: "#10b981", fontSize: "0.75rem", fontWeight: 700 }}>03</span>
              <h3 style={{
                fontFamily: '"Instrument Serif", Georgia, serif',
                fontSize: "1.45rem",
                fontWeight: 400,
                margin: "12px 0 8px",
                color: "#FFFFFF"
              }}>
                Empirical realism above all.
              </h3>
              <p style={{ color: "#737373", fontSize: "0.82rem", lineHeight: 1.65, margin: 0 }}>
                Never fabricate benchmark numbers, never simulate hardware results as verified truth, and hold every performance optimization to verifiable hardware receipts.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: "1px solid rgba(255, 255, 255, 0.08)",
        padding: "32px clamp(20px, 4vw, 48px)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: "16px",
        fontSize: "0.7rem",
        color: "#737373"
      }}>
        <span>© 2026 GhostLayer. Built by Bhargav Mahadevan.</span>
        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
          <Link href="/" style={{ color: "#A3A3A3", textDecoration: "none" }}>Platform</Link>
          <Link href="/terms" style={{ color: "#737373", textDecoration: "none" }}>Terms</Link>
          <Link href="/privacy" style={{ color: "#737373", textDecoration: "none" }}>Privacy</Link>
          <Link href="/legal" style={{ color: "#737373", textDecoration: "none" }}>Disclaimers</Link>
          <Link href="/pilot" style={{ color: "#10b981", textDecoration: "none" }}>Free Pilot Audit</Link>
          <a href="mailto:bhargavmahadevan@gmail.com" style={{ color: "#A3A3A3", textDecoration: "none" }}>Direct Contact</a>
        </div>
      </footer>
    </div>
  );
}
