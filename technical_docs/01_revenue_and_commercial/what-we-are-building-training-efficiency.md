# What we're building (the simple version)

**In one sentence:** we're building a quiet background helper that watches an AI model being trained, spots both hardware bottlenecks and algorithmic sample waste, and makes training finish faster and cheaper without changing model quality — backed by structured decision records and verified math.

## Why this makes sense

Training an AI model means running expensive rented computers (GPUs) for hours, days, or sometimes weeks. Most teams use the default settings that came with their training code, and those defaults are almost never tuned well. Independent research found that only about 7 out of 100 companies training AI models are using their GPUs efficiently — meaning the other 93 are paying for computer time they aren't actually using.

Furthermore, in **small-data and few-shot regimes** (fine-tuning on 10k–50k samples or edge workstations), standard first-order algorithms like AdamW waste 35–50% of their training steps oscillating in narrow loss ravines. Switching to **higher-order curvature calculus & manifold constraints** (like Matrix-Free Quasi-Newton L-BFGS with Damped Powell updates, Helmholtz-Hodge vector field vorticity damping, 4-regime Hybrid Muon, DeepSeek-style mHC multi-stream residuals, Chunked Cross-Entropy, Sophia, Shampoo, and Pearlmutter HVP) cuts the total step count required to reach target accuracy while slashing activation and optimizer VRAM.

## What our software actually does

1. **It watches the training job run (`GhostWatcherHook`).** Without touching or changing anything at first — non-invasively capturing step time, GPU utilization, VRAM allocation, divergence flux ($\text{div}(\mathbf{g}) = \text{Tr}(H)$), vorticity curl index, and loss trajectories.
2. **It spots the waste.** Idle time, DataLoader stalls, sub-optimal precision, unpinned memory, logit tensor VRAM explosion, dense matrix memory leaks, and non-conservative rotational limit-cycle orbits.
3. **It produces an actionable Decision Record.** Telling your engineers exactly what to change, the scientific evidence behind it, and the confidence score.
4. **It safely applies optimizations with Consent Control (`AutoApplier`).** Gated by explicit consent levels (`AUDIT_ONLY`, `RECOMMEND_AND_ASK`, `AUTO_APPLY_SAFE`, `AUTO_APPLY_ALL`), with instant automated rollback (`RollbackManager`) if loss divergence is detected.
5. **It guarantees convergence safety with an Adaptive Lagrangian Controller.** Tracking online loss variance and dynamic dual multipliers ($\lambda_{t+1}$) to block divergence without stalling training.
6. **It proves the savings with statistical rigor (`SavingsVerifier`).** Using paired Welch's t-tests, 95% confidence intervals, and p-values — not just illustrative estimates.


## How we get paid

- **Lead Offer:** 48-Hour Pre-Flight Audit ($2,500 flat fee). We run alongside a staging run and deliver an Executive & Engineering GPU Waste Audit. If we find less than 2x our fee in annualized waste, the audit is free.
- **Team SaaS:** $199–$999/month for continuous telemetry monitoring, Slack/Discord alerts, and decision replay across all team runs.
- **Enterprise Compute Assurance:** $15,000–$40,000/year for air-gapped on-premises VPC deployments and custom optimization rules.

## What it's not

- It never touches or copies a client's actual model weights, training data, prompts, or secrets — strict privacy boundary audited by `TelemetryBoundary`.
- It's not a black-box optimizer — every automated change is guarded by loss-shift proxy checks and stateful rollback.

## The one-line pitch

*"We make your AI training run faster and cheaper without changing what your model learns, producing an auditable decision record for every run."*
