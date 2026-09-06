# 💰 12: How We Make Money (Commercial Model)

> **Status:** Active Business Model (Post-Pivot)  
> **Model:** Flat-Fee Diagnostic Audits + Team SaaS Subscriptions  
> **Deprecated Model:** Variable 25% performance-fee splits are *deprecated* due to baseline attribution conflicts and legal/financial liability.

---

### 1. The Commercial Wedge: Productized Pre-Flight Audits ($2,500 Flat Fee)

Instead of arguing over complex baseline attribution or attempting to take 25% of cloud invoices, GhostLayer sells a straightforward, bounded diagnostic service:

- **What the Client Buys:** A 48-hour assisted configuration audit on their staging/pre-training pipeline.
- **Deliverable:** An Executive & Engineering Audit Report detailing exact DataLoader I/O stalls, VRAM allocation fragmentation, pinned memory gaps, and kernel fusion opportunities.
- **Price:** **$2,500 flat fee** per audit engagement.
- **Why $2,500 Specifically (Pricing Rationale):**
  1. **Discretionary Card Limit:** Priced below the typical $3,000–$5,000 corporate credit card threshold for engineering managers/leads, bypassing multi-month procurement committee sign-offs.
  2. **Contractor Benchmark:** Pegged to 1–2 days of specialized ML infrastructure consulting (standard contractor day-rate is $1,500–$2,500/day), delivered via automated telemetry capture.
  3. **High ROI Asymmetry:** On an 8x H100 cluster burning $20k–$40k/month, resolving even a 5% I/O bottleneck recovers $1,000–$2,000/month, amortizing the audit cost in under 60 days.
- **Guarantee:** If we don't identify actionable configuration improvements yielding $\ge 2\times$ the audit fee in annualized compute efficiency, the audit is 100% refunded.

---

### 2. The Recurring Revenue Engine: Team Platform SaaS ($199 – $999 / month)

Once a team completes an initial audit, they convert into a monthly software subscription to ensure ongoing pipeline health across all future training runs:

| Tier | Price | Target Team | Key Features |
| :--- | :--- | :--- | :--- |
| **Developer / OSS** | Free | Single researchers, workstation users | Local PyTorch hook, CLI summary, terminal audit card |
| **Team Starter** | **$199 / mo** | Small AI teams (1–8 GPUs) | Multi-run history, Slack/Discord stall alerts, batch headroom guide |
| **Team Pro** | **$499 / mo** | Growth labs (8–32 GPUs) | Continuous telemetry dashboard, W&B integration, decision replay |
| **Enterprise Fleet** | **$999 / mo** | Large clusters (32+ GPUs) | Air-gapped VPC ingestion, custom advisory rules, team SLA |

---

### 3. Why We Dropped the 25% Performance Fee Model

1. **Attribution Conflict (Plaintiff vs. Judge):** Calculating exact cloud dollar savings requires measuring a counterfactual baseline. Clients and vendors inevitably dispute whether speedups came from software fixes, hardware thermal throttling, or dataset variability.
2. **Uncapped Liability:** Autonomous modification of multi-million dollar training runs introduces catastrophic financial risk if a model diverges.
3. **Sales Cycle Friction:** Enterprise procurement teams reject variable percentage-of-savings billing; they demand predictable, pre-approved software line items ($2,500 audit / fixed monthly SaaS).
