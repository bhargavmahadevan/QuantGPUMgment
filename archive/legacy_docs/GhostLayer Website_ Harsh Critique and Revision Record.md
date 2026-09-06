# GhostLayer Website: Harsh Critique and Revision Record

**Assessment date:** August 11, 2026  
**Scope:** Public-facing static website, including the landing page, founder page, technical explainer, interactive calculator, benchmark explorer, and newly added help center.

## Executive verdict

> **This is a visually ambitious prototype, not a credible enterprise or investor-ready website.**

The original site is trying to project institutional certainty before it has earned it. It has enough polish to create a first impression, but its most prominent content—the headline, performance claims, savings figure, security labels, and request form—undermines trust when examined closely. The problem is not a lack of animation, gradients, or interactive panels. The problem is that the website repeatedly presents simulated, limited, unfinished, or non-functional material with the visual confidence of verified production capability.

If this site is used for a serious enterprise buyer, private-equity partner, or technical evaluator today, the likely reaction is not “this is impressive.” It is “what, exactly, is real here?” That is a dangerous outcome for a company selling safety-critical infrastructure intelligence.

| Area | Original score | Why it fails |
|---|---:|---|
| Credibility | 2/10 | The hero claims a 31% reduction on “enterprise fleets,” while the qualifier narrows the result to a single-node RTX A2000 comparison. The benchmark controls also claimed 5 physical and 250 synthetic workloads although the data array rendered 4 and 2, respectively. [1] [2] |
| Conversion | 1/10 | The quote form was a static `alert()` that falsely told visitors a partner would contact them within 24 hours. No request was transmitted, no contact path existed, and no privacy or consent handling existed. [1] [2] |
| Product clarity | 3/10 | The site tries to sell to ML platform teams, enterprise infrastructure teams, CFOs, private-equity operators, and “institutional” buyers simultaneously. It never establishes one clear primary buyer, one primary job-to-be-done, or one proof path. [1] |
| Usability | 4/10 | Ten desktop navigation items, a giant headline, dense jargon, a 64-node interaction, a benchmark table, an ROI calculator, and multiple “moat” arguments all compete before the visitor understands the basic product. [1] |
| Accessibility | 3/10 | Before revision, interactive GPU nodes were clickable `div` elements, dialogs lacked meaningful dialog semantics, there was no global visible keyboard focus treatment, and the animation-heavy experience had no reduced-motion safeguard. [1] [2] |
| Technical readiness | 3/10 | The site is static, external-CDN dependent, and contains no real request-processing service, evidence download flow, analytics instrumentation, consent management, or deployment documentation. This is a demo shell, not a production funnel. [1] [2] |

## The most damaging problems

### 1. The claims hierarchy is upside down

The boldest statement says that GhostLayer reduced training cost by 31% on enterprise fleets, then adds sub-millisecond rollback language. The nearby qualifier narrows the result to a local, single-node A2000 comparison and calls cluster-scale results simulated. That is not a minor copy issue. It is a material credibility problem: the least certain claim gets the largest typography, while the caveat is small and late. [1]

The website should lead with what is genuinely demonstrated, describe the test conditions, and clearly label all projections. A more defensible headline would be something like: **“Evaluate GPU training inefficiencies with shadow-mode evidence before enabling automated changes.”** That is less flashy, but it matches the product’s stated safety story and is far more credible.

### 2. The “proof” controls advertised inventory that did not exist

The interface promised **5** physical-hardware workloads and **250** synthetic projections, but the implementation supplied 4 physical rows and 2 synthetic rows. This is the kind of discrepancy a technical buyer will find in minutes. Once found, every other number becomes suspect. [1] [2]

Do not fake dataset scale in the interface. If you have four physical tests, say four. If you have two projections, say two. If you later build a 250-run reproducible model, publish its methodology, inputs, exclusions, and raw output—not just an oversized number in a filter button.

### 3. The ROI calculator manufactured authority it did not have

The calculator made an illustrative model look like an auditable ledger. Its underlying formula used visitor-selected assumptions, a fixed 20% gain-share assumption, static hardware rates, and hand-appended cents in the “receipt” fields. That is presentation theater, not financial provenance. [2]

The revised calculator now formats estimates consistently without invented cents and labels the receipt as illustrative. That is an improvement, not a cure. The missing work is still substantial: contracted rates, utilization evidence, measured baseline duration, model-quality constraints, workload comparability, and a real commercial proposal must all be handled before any savings estimate is used externally.

### 4. The request flow was actively deceptive

The original form looked like an enterprise inbound workflow but did nothing except show a success alert and close. The text promised a 24-hour follow-up even though no data was collected. That should never ship. It burns trust, creates false user expectations, and makes the website look unfinished the instant someone tests it. [1]

The revised form now explicitly says it is a non-transmitting preview and locally validates required fields. **It must still be connected to an owned, secure CRM or server endpoint before launch.** Do not replace the disclaimer with another fake success state.

### 5. The founder page is a publishing blocker

The founder page contains visible placeholders for the founder name, biography, signature, social links, and contact details. It tells visitors that the page is “prepared for you to share” content. That is not a small cosmetic defect; it announces that the company is unfinished. [3]

Either complete the founder page with real, reviewable facts or remove it from navigation. There is no respectable middle ground.

### 6. The site is over-designed and under-explained

The interface has a strong visual language but too much of it. The visitor meets a sprawling nav, a massive claim, animated telemetry, 64 controls, an evidence table, an architecture section, an incentives section, a calculator, and a quote form before receiving a one-minute explanation of the product. The result is cognitive overload disguised as sophistication. [1]

The better sequence is simple: explain the buyer’s operational problem; state the one real, narrow capability; show one defensible proof item; explain the safety boundary; then present a clear assessment request. Everything else belongs behind “Technical details,” “Evidence,” or “Security review.”

## Changes implemented in this revision

This package deliberately improves honesty, comprehension, and baseline accessibility without pretending to solve the underlying product-evidence problem.

| Change | Implementation | Effect |
|---|---|---|
| Searchable help center | Added a fixed, keyboard-closeable Help drawer with eight practical topics covering evidence, demos, security review, pilot planning, calculator assumptions, and the static form. [4] | Gives visitors immediate context instead of forcing them to infer product boundaries from marketing copy. |
| Evidence labels corrected | Changed benchmark filter labels from 5/250 to the actual rendered 4/2 counts and marked physical results as limited validation. [1] [2] | Removes a readily discoverable credibility error. |
| Calculator honesty improved | Removed fabricated receipt cents, generates an illustrative date, and frames help content around inputs and the 20% gain-share assumption. [2] [4] | Makes the output look like a planning estimate rather than an audit artifact. |
| Static form made truthful | Removed the false submission alert, added form labels, dialog semantics, a preview warning, local validation, focus return, and Escape dismissal. [1] [2] | Stops the site from falsely implying that an inquiry was sent. |
| Basic interaction accessibility | Converted the 64 GPU controls from clickable `div`s to labeled buttons, added visible focus styling, and respected reduced-motion preferences. [2] | Improves keyboard operation and reduces avoidable motion burden. |
| Cache-safe delivery | Versioned the shared stylesheet and scripts in public pages. [1] [3] [4] | Ensures browsers do not silently retain the pre-revision UI. |
| Security claim softened | Replaced an unsubstantiated compliance string with a statement that claims require deployment review. [1] [3] | Reduces the risk of presenting unverified certification as fact. |

## What still must happen before publication

The revised site is better, but it is still **not ready for a serious launch**. The following should be treated as release gates rather than optional polish.

| Priority | Required action | Release criterion |
|---|---|---|
| P0 | Replace or remove every founder placeholder and dead social/contact link. | No bracketed placeholder, `#` social link, or “prepared for you” copy remains. |
| P0 | Connect the evaluation form to an owned, secure endpoint or CRM. | Submission, consent, privacy notice, error state, spam control, and internal notification are tested end-to-end. |
| P0 | Rebuild claims around documented evidence. | Every performance, rollback, security, and compliance claim links to a dated methodology or is removed. |
| P0 | Separate empirical measurements from projections everywhere. | No synthetic output shares prominence, wording, or visual treatment with physical validation. |
| P1 | Choose one primary buyer and one first use case. | Home page can answer “for whom,” “what problem,” “what gets deployed first,” and “what proof exists” in under 30 seconds. |
| P1 | Replace the faux ledger with an explicit estimate workflow. | Calculator inputs, rate-source date, fee model, exclusions, and uncertainty are visible; downloadable estimates include these assumptions. |
| P1 | Add a real security and deployment page. | It describes data flow, permissions, network model, retention, deployment boundary, and only verified attestations. |
| P2 | Reduce the top-level navigation to 4–5 choices. | The landing page has a clear path: Overview, Evidence, Security, How It Works, Request Assessment. |
| P2 | Add independent quality assurance. | Test keyboard navigation, dialog focus containment, mobile layout, browser support, form error states, and analytics events. |

## Final recommendation

The right move is not to keep adding more visual theater. It is to make the site **less impressive-looking and more believable**. Earn the headline through disclosed methods. Earn the enterprise language through a real intake path and security evidence. Earn the “autonomous” claim through a documented operator-controlled rollout model.

Once those foundations are in place, the current visual system can support a strong website. Until then, the main value of the site is as a concept demo—not as proof of a production-ready company.

## References

[1]: [Home page source](index.html)
[2]: [Client-side interaction and calculator source](app.js)
[3]: [Founder page source](founder.html)
[4]: [Shared help center source](help.js)
