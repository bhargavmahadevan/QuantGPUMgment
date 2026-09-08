import React, { useEffect } from "react";
import { Link } from "wouter";
import { ArrowLeft, ShieldAlert, Scale, FileText, CheckCircle2, Lock, AlertTriangle } from "lucide-react";

export default function Terms() {
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
            <Link href="/privacy" style={{ color: "#737373", textDecoration: "none" }}>Privacy Policy</Link>
            <Link href="/legal" style={{ color: "#737373", textDecoration: "none" }}>Legal Disclaimers</Link>
            <Link href="/founder" style={{ color: "#10b981", textDecoration: "none" }}>Founder Dossier</Link>
          </div>
        </nav>

        {/* Document Header */}
        <header style={{ marginBottom: "40px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "9999px", padding: "4px 12px", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#D4D4D4", marginBottom: "16px" }}>
            <Scale size={13} style={{ color: "#10b981" }} /> GhostLayer Master Services &amp; Evaluation Agreement
          </div>
          <h1 style={{ fontSize: "36px", fontWeight: "700", letterSpacing: "-0.02em", margin: "0 0 12px 0", color: "#FFFFFF" }}>
            Terms of Service
          </h1>
          <p style={{ color: "#737373", fontSize: "14px", margin: 0 }}>
            Effective Date: September 8, 2026 · Version 2.4 (Enterprise Production &amp; Evaluation Edition)
          </p>
        </header>

        {/* Critical Legal Warning Banner */}
        <div style={{ background: "rgba(244, 63, 94, 0.08)", border: "1px solid rgba(244, 63, 94, 0.3)", borderRadius: "6px", padding: "20px", marginBottom: "40px" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
            <ShieldAlert size={22} style={{ color: "#f43f5e", flexShrink: 0, marginTop: "2px" }} />
            <div>
              <strong style={{ color: "#fb7185", fontSize: "14px", display: "block", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Important Notice: Mandatory Binding Arbitration &amp; Class Action Waiver
              </strong>
              <p style={{ fontSize: "13px", lineHeight: "1.6", color: "#D4D4D4", margin: 0 }}>
                THESE TERMS CONTAIN A MANDATORY ARBITRATION CLAUSE AND CLASS ACTION WAIVER IN SECTION 11. THEY AFFECT YOUR LEGAL RIGHTS CONCERNING HOW DISPUTES WITH GHOSTLAYER AND ITS FOUNDER ARE RESOLVED. BY ACCESSING OR USING OUR SERVICES, RUNNING TELEMETRY HOOKS, OR EVALUATING THE PLATFORM, YOU EXPRESSLY AGREE TO RESOLVE ALL DISPUTES VIA BINDING INDIVIDUAL ARBITRATION AND WAIVE ANY RIGHT TO A TRIAL BY JURY OR CLASS PROCEEDING.
              </p>
            </div>
          </div>
        </div>

        {/* Legal Text Sections */}
        <article style={{ fontSize: "14px", lineHeight: "1.75", color: "#A3A3A3" }}>
          
          {/* Section 1 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              1. Acceptance of Terms &amp; Scope of Agreement
            </h2>
            <p>
              By accessing, browsing, testing, installing, or executing the GhostLayer software, PyTorch context hooks (`ghost_layer`), telemetry modules, web interfaces, APIs, reports, or benchmark suites (collectively, the "Services"), you ("Customer," "User," or "Licensee") agree to be bound by these Terms of Service ("Terms"). If you represent an entity or organization, you represent and warrant that you possess full legal authority to bind that entity to these Terms.
            </p>
            <p>
              If you do not unconditionally agree to all provisions of these Terms, you are strictly prohibited from installing, testing, or executing the GhostLayer software or accessing this platform.
            </p>
          </section>

          {/* Section 2 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              2. Nature of Software: Decision Support &amp; Observability Only
            </h2>
            <p>
              GhostLayer is an infrastructure telemetry, profiling, and audit layer designed to observe PyTorch training loops, record step-level hardware utilization, evaluate heuristic optimization candidates, and log decision traces.
            </p>
            <p>
              <strong>Not an Automated Replacement for Human ML Engineers:</strong> GhostLayer provides decision support, heuristic recommendations, and safety-gated rollback mechanisms. It is not an autonomous substitute for qualified human machine learning engineers, cluster administrators, or systems architects. Customer agrees to maintain qualified human engineering oversight over all training workloads, hyperparameters, checkpoints, and automated intervention toggles.
            </p>
          </section>

          {/* Section 3 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              3. Customer Data &amp; No-Model-Ingestion Guarantee
            </h2>
            <p>
              <strong>Data Isolation:</strong> GhostLayer expressly does not require, read, ingest, transmit, or store Customer's proprietary model weights, dataset tokens, input prompts, or proprietary forward/backward gradients. GhostLayer captures solely numerical hardware metrics (GPU core utilization %, VRAM bytes allocated, kernel execution step times, inter-node NCCL comm timings, and loss scalar magnitude deltas).
            </p>
            <p>
              <strong>Customer Intellectual Property:</strong> Customer retains 100% exclusive ownership of all machine learning models, training data, weights, checkpoints, algorithms, and intellectual property. GhostLayer claims zero ownership or licensing rights over Customer's model artifacts.
            </p>
          </section>

          {/* Section 4 - DISCLAIMER OF WARRANTIES */}
          <section style={{ marginBottom: "36px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", padding: "24px", borderRadius: "6px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px" }}>
              4. DISCLAIMER OF ALL WARRANTIES ("AS IS" &amp; "AS AVAILABLE")
            </h2>
            <p style={{ color: "#E5E5E5", textTransform: "uppercase", fontSize: "12px", letterSpacing: "0.02em" }}>
              TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE GHOSTLAYER SOFTWARE, HOOKS, ALGORITHMS, DOCUMENTATION, AUDIT TOOLS, FINOPS CALCULATORS, AND SERVICES ARE PROVIDED STRICTLY "AS IS", "WITH ALL FAULTS", AND "AS AVAILABLE", WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS, IMPLIED, STATUTORY, OR OTHERWISE.
            </p>
            <p style={{ color: "#D4D4D4" }}>
              GHOSTLAYER, ITS FOUNDER (BHARGAV MAHADEVAN), AFFILIATES, AND CONTRIBUTORS EXPRESSLY DISCLAIM ALL IMPLIED WARRANTIES, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, AND ACCURACY.
            </p>
            <p style={{ color: "#D4D4D4" }}>
              WITHOUT LIMITING THE GENERALITY OF THE FOREGOING, GHOSTLAYER MAKES NO WARRANTY OR GUARANTEE THAT:
            </p>
            <ul style={{ paddingLeft: "20px", color: "#D4D4D4" }}>
              <li>THE SERVICES WILL PREVENT MODEL TRAINING DIVERGENCE, LOSS SPIKES, OVERFITTING, OR GRADIENT EXPLOSION;</li>
              <li>THE SOFTWARE WILL PREVENT HARDWARE FAULTS, OUT-OF-MEMORY (OOM) CRASHES, HARDWARE OVERHEATING, OR INTERCONNECT FAILURES;</li>
              <li>REDUCED RUNTIME OR FINANCIAL/ENERGY SAVINGS PROJECTED IN ANY CALCULATOR, REPORT, OR BENCHMARK WILL BE ACHIEVED ON CUSTOMER HARDWARE;</li>
              <li>ANY HEURISTIC RECOMMENDATION OR SAFETY PROXY CHECK ("VERIFIED SAFE") ESTABLISHES GENERAL CONVERGENCE OR NUMERICAL VALIDITY BEYOND THE SPECIFIC SAMPLED STEP; OR</li>
              <li>THE SOFTWARE WILL OPERATE UNINTERRUPTED, BUG-FREE, ERROR-FREE, OR COMPATIBLE WITH EVERY PYTORCH, CUDA, TRITON, OR NCCL COMBINATION.</li>
            </ul>
          </section>

          {/* Section 5 - LIMITATION OF LIABILITY */}
          <section style={{ marginBottom: "36px", background: "rgba(244, 63, 94, 0.03)", border: "1px solid rgba(244, 63, 94, 0.2)", padding: "24px", borderRadius: "6px" }}>
            <h2 style={{ fontSize: "18px", color: "#f43f5e", fontWeight: "600", marginBottom: "12px" }}>
              5. LIMITATION OF LIABILITY &amp; WAIVER OF CONSEQUENTIAL DAMAGES
            </h2>
            <p style={{ color: "#E5E5E5", textTransform: "uppercase", fontSize: "12px", letterSpacing: "0.02em" }}>
              TO THE FULLEST EXTENT PERMITTED BY LAW, UNDER NO CIRCUMSTANCES AND UNDER NO LEGAL OR EQUITABLE THEORY (WHETHER IN CONTRACT, TORT, NEGLIGENCE, STRICT LIABILITY, INDEMNITY, OR OTHERWISE) SHALL GHOSTLAYER, ITS FOUNDER (BHARGAV MAHADEVAN), EMPLOYEES, CONTRACTORS, OR LICENSORS BE LIABLE TO CUSTOMER OR ANY THIRD PARTY FOR:
            </p>
            <ol style={{ paddingLeft: "20px", color: "#D4D4D4" }}>
              <li><strong>ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, PUNITIVE, OR EXEMPLARY DAMAGES;</strong></li>
              <li><strong>LOSS OF REVENUE, PROFITS, BUSINESS, DATA, GOODWILL, OR ANTICIPATED SAVINGS;</strong></li>
              <li><strong>DAMAGED, DIVERGENT, OR CORRUPTED MODEL WEIGHTS, CHECKPOINTS, OR LOSS ACCURACY;</strong></li>
              <li><strong>UNRECOVERABLE GPU-HOURS, WASTED COMPUTE SPEND, OVERAGES, OR SURGES ON THIRD-PARTY CLOUD PROVIDERS (INCLUDING AWS, GCP, AZURE, ORACLE, LAMBDA LABS, RUNPOD, CRUSOE, OR COREWEAVE);</strong></li>
              <li><strong>COST OF PROCUREMENT OF SUBSTITUTE COMPUTATIONAL GOODS OR CLOUD CAPACITY; OR</strong></li>
              <li><strong>PHYSICAL GPU HARDWARE DEGRADATION, SILICON FAILURE, THERMAL THROTTLING, OR FIRMWARE CRASHES.</strong></li>
            </ol>
            <p style={{ color: "#E5E5E5", textTransform: "uppercase", fontSize: "12px", letterSpacing: "0.02em", marginTop: "16px" }}>
              <strong>AGGREGATE LIABILITY CAP:</strong> IN NO EVENT SHALL GHOSTLAYER'S TOTAL CUMULATIVE LIABILITY ARISING OUT OF OR RELATED TO THESE TERMS OR THE SERVICES EXCEED THE GREATER OF: (A) THE TOTAL AMOUNT ACTUALLY PAID BY CUSTOMER TO GHOSTLAYER FOR USE OF THE SERVICES IN THE TWELVE (12) WEEKS IMMEDIATELY PRECEDING THE CLAIM, OR (B) EXACTLY ONE HUNDRED UNITED STATES DOLLARS ($100.00 USD).
            </p>
          </section>

          {/* Section 6 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              6. Customer Safeguards &amp; Operational Responsibilities
            </h2>
            <p>
              Customer acknowledges that large-scale deep learning model training carries inherent non-deterministic risks, including loss divergence, numerical instability, gradient overflow, and hardware failure.
            </p>
            <p>
              <strong>Mandatory Customer Precautions:</strong> As a condition of using the Services, Customer agrees that it shall at all times:
            </p>
            <ul style={{ paddingLeft: "20px" }}>
              <li>Maintain frequent, redundant, verified model checkpoints on secure persistent storage independent of any automated optimization hooks;</li>
              <li>Set strict hard spending caps and billing alarms on all third-party cloud GPU provider accounts;</li>
              <li>Verify all loss and validation curves against independent empirical baselines before committing automated changes to production runs; and</li>
              <li>Retain ultimate responsibility for ensuring software execution complies with all applicable export controls, safety regulations, and internal AI governance rules.</li>
            </ul>
          </section>

          {/* Section 7 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              7. Empirical Benchmark Receipts &amp; FinOps Disclosures
            </h2>
            <p>
              All performance figures, millisecond step timings, VRAM memory reductions, and loss deltas published on the GhostLayer platform (including the 52.2M LLM benchmark on NVIDIA RTX A2000) represent historical physical receipts from specific, isolated hardware runs. They do not constitute empirical guarantees that identical speedups or memory reductions will occur on Customer's unique model architecture, batch size, cluster topology, or hardware configuration.
            </p>
            <p>
              All figures generated by the FinOps Profit Calculator are heuristic mathematical estimates for exploratory planning only and do not constitute financial advice, binding commitments, or guarantees of cost reduction.
            </p>
          </section>

          {/* Section 8 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              8. Indemnification by Customer
            </h2>
            <p>
              Customer agrees to defend, indemnify, and hold harmless GhostLayer, its founder (Bhargav Mahadevan), officers, contractors, and agents from and against any and all claims, liabilities, damages, losses, costs, and expenses (including reasonable attorneys' fees) arising out of or in any way connected with: (a) Customer's use or misuse of the Services; (b) Customer's training datasets or model outputs; (c) any violation of these Terms; or (d) any dispute between Customer and its cloud infrastructure providers or downstream users.
            </p>
          </section>

          {/* Section 9 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              9. Intellectual Property &amp; License Restrictions
            </h2>
            <p>
              GhostLayer grants Customer a revocable, non-exclusive, non-transferable, non-sublicensable evaluation license to access the web interface and pilot hooks in accordance with these Terms.
            </p>
            <p>
              Customer shall not: (a) reverse engineer, decompile, or disassemble any proprietary binary hooks or compiled kernels; (b) remove or alter any copyright or trademark notices; (c) use the Services to construct a competing GPU telemetry or optimization service; or (d) publish synthetic or misleading benchmarks purporting to reflect GhostLayer capabilities without prior written verification.
            </p>
          </section>

          {/* Section 10 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              10. Governing Law &amp; Exclusive Jurisdiction
            </h2>
            <p>
              These Terms, and any dispute arising out of or related to them, shall be governed by and construed in accordance with the laws of the State of Delaware and/or the State of Texas, United States of America, without regard to conflict of law principles or the United Nations Convention on Contracts for the International Sale of Goods.
            </p>
          </section>

          {/* Section 11 - ARBITRATION & CLASS ACTION WAIVER */}
          <section style={{ marginBottom: "36px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", padding: "24px", borderRadius: "6px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px" }}>
              11. BINDING ARBITRATION &amp; CLASS ACTION WAIVER
            </h2>
            <p>
              <strong>Informal Resolution:</strong> Prior to initiating any formal legal proceeding, Customer and GhostLayer agree to attempt in good faith to resolve any dispute through direct informal negotiation for a period of at least thirty (30) days by contacting `bhargavmahadevan@gmail.com`.
            </p>
            <p>
              <strong>Binding Arbitration:</strong> If the dispute is not resolved within 30 days, it shall be finally and exclusively settled by binding individual arbitration administered by the American Arbitration Association (AAA) under its Commercial Arbitration Rules before a single neutral arbitrator. The arbitration shall take place in Houston, Texas or Wilmington, Delaware (or remotely via videoconference upon mutual agreement). The arbitrator's award shall be binding and may be entered as a judgment in any court of competent jurisdiction.
            </p>
            <p style={{ color: "#E5E5E5", textTransform: "uppercase", fontSize: "12px", letterSpacing: "0.02em" }}>
              <strong>CLASS ACTION WAIVER:</strong> CUSTOMER AND GHOSTLAYER MUTUALLY AGREE THAT ALL CLAIMS AND DISPUTES MUST BE BROUGHT IN AN INDIVIDUAL CAPACITY ONLY, AND NOT AS A PLAINTIFF, REPRESENTATIVE, OR CLASS MEMBER IN ANY PURPORTED CLASS, COLLECTIVE, REPRESENTATIVE, OR PRIVATE ATTORNEY GENERAL ACTION. THE ARBITRATOR MAY NOT CONSOLIDATE MORE THAN ONE PERSON'S CLAIMS.
            </p>
          </section>

          {/* Section 12 */}
          <section style={{ marginBottom: "36px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              12. Severability, Survival &amp; Entire Agreement
            </h2>
            <p>
              If any provision of these Terms is determined to be unlawful, void, or unenforceable by an arbitrator or court of competent jurisdiction, that provision shall be severed and shall not affect the validity and enforceability of any remaining provisions.
            </p>
            <p>
              Sections 4 (Disclaimer of Warranties), 5 (Limitation of Liability), 7 (Empirical Disclosures), 8 (Indemnification), 10 (Governing Law), and 11 (Arbitration &amp; Class Action Waiver) shall survive any termination or expiration of these Terms.
            </p>
          </section>

          {/* Section 13 */}
          <section style={{ marginBottom: "48px" }}>
            <h2 style={{ fontSize: "18px", color: "#FFFFFF", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "8px" }}>
              13. Contact &amp; Legal Notices
            </h2>
            <p>
              All formal legal notices, claims, or inquiries regarding these Terms should be sent in writing to:
            </p>
            <div style={{ background: "#0A0A0A", border: "1px solid rgba(255,255,255,0.1)", padding: "16px", borderRadius: "4px", marginTop: "12px" }}>
              <strong style={{ color: "#FFFFFF", display: "block", marginBottom: "4px" }}>GhostLayer Legal &amp; Compliance Office</strong>
              <span>Attn: Bhargav Mahadevan, Founder</span><br />
              <span>Email: <a href="mailto:bhargavmahadevan@gmail.com" style={{ color: "#10b981", textDecoration: "none" }}>bhargavmahadevan@gmail.com</a> / <a href="mailto:legal@ghostlayer.ai" style={{ color: "#10b981", textDecoration: "none" }}>legal@ghostlayer.ai</a></span><br />
              <span>Direct: 832-402-3104</span>
            </div>
          </section>

        </article>

        {/* Footer Navigation */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "13px" }}>
          <Link href="/" style={{ color: "#737373", textDecoration: "none" }}>
            ← Back to GhostLayer Orbit
          </Link>
          <div style={{ display: "flex", gap: "20px" }}>
            <Link href="/privacy" style={{ color: "#A3A3A3", textDecoration: "none" }}>Privacy Policy →</Link>
            <Link href="/legal" style={{ color: "#A3A3A3", textDecoration: "none" }}>Legal &amp; Empirical Disclaimers →</Link>
          </div>
        </div>

      </div>
    </div>
  );
}
