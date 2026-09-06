# Cross-Client Learning & Continuous Value: What Was Built and Why

This document explains the mechanism added to address two specific concerns: that a performance-fee
client could copy a one-time config and cancel, and that the "gets smarter with every run" claim
needed to be backed by real, inspectable code rather than asserted.

## What this is NOT

It is not a neural network, and it is not "deep learning." No model is trained. Calling it a "hive
mind" oversells it. Anyone evaluating this technically will see through that framing immediately,
and it would undercut real credibility with exactly the sophisticated technical buyers this product
targets.

## What this actually is

Two small, honest, testable mechanisms:

### 1. A statistical knowledge base with computed confidence (`ghost_layer/kb/knowledge_base.py`)

Every verified (correctness-passed, not-a-regression) optimization run contributes a data point to
a shared table, keyed by a fingerprint of (architecture family, hardware type, model size bucket,
framework version). For each fingerprint, the system tracks:

- A running mean of the measured speedup (Welford's online algorithm)
- A running variance of that speedup
- A **confidence tier** (`NONE` / `LOW` / `MEDIUM` / `HIGH`) computed from both sample count AND
  variance \u2014 ten wildly inconsistent samples do not earn `HIGH` confidence, only ten *consistent*
  ones do

When a new client's exact fingerprint has no data yet, `find_best_match()` progressively broadens
the search (dropping framework version, then model size bucket) until it finds *some* verified data
to generalize from, and reports which match level it used (`EXACT`, `ARCH+HW+SIZE`, `ARCH+HW`, or
`NONE`) so nothing is presented as more specific than it actually is.

**This is the real version of "gets smarter with every run":** each additional verified sample at a
given fingerprint provably narrows the reported confidence interval, and a brand-new client benefits
from the nearest available match instead of starting from zero. That is a legitimate, inspectable
claim. It is also a modest one \u2014 it will not discover an optimization technique nobody has coded
a rule for. Expanding the rule library over time is a separate, ongoing engineering task, not
something this mechanism does automatically.

### 2. Drift / staleness detection (`ghost_layer/continuity/drift_monitor.py`)

This is the actual answer to "why would a client keep paying after the first optimization." A
`DriftMonitor` compares a client's last-verified, currently-applied config against two things that
genuinely change over time:

- **Their own environment.** A PyTorch upgrade, a GPU swap, or the model crossing into a new
  parameter-count bucket all mean the previously-verified config is no longer verified for the
  *current* setup \u2014 re-verification is a legitimate, billable event, not a repeat sales pitch.
- **The shared knowledge base.** If enough other verified clients have since landed on a
  meaningfully better config for the same fingerprint (default threshold: 5 points of speedup, and
  only counted at `MEDIUM`/`HIGH` confidence), that is a real, quantifiable new opportunity the
  client doesn't have access to just by keeping their old config.

A client who copies a config and does nothing else will, over time, genuinely fall behind either
their own environment's verification status or the field's current best-known config \u2014 and the
system can show them exactly why, with numbers, rather than asserting it. That's what makes this a
defensible retention mechanism instead of a marketing claim.

## Honest limitations of this version

- The knowledge base is currently wired into the decision engine for exactly one rule
  (`RULE_GRAPHIFY_TORCH_COMPILE`), because that's the one the example pipelines actually register
  verified learnings against. Extending this to the other five rules requires each of them to be
  registered with the KB the same way, which hasn't been done yet.
- `DriftMonitor` requires a stored `EnvironmentFingerprint` from the client's last engagement to
  compare against \u2014 there is no automatic scheduling or notification system yet; the CLI command
  (`ghost-layer check-staleness`) has to be run, or wired into a cron job / webhook by whoever
  operates this.
- None of this changes the fact that real GPU validation (Section 6 of the prior full-audit report)
  still hasn't happened. This mechanism makes the *business model* more defensible; it doesn't make
  the underlying speedup claims more proven.

## Test coverage

14 new tests in `tests/test_continuous_value.py` cover: parameter-count bucketing, confidence
requiring low variance (not just sample count), rejected samples never polluting the average,
fingerprint-fallback match-level selection, the decision engine correctly using (or not using) a
supplied knowledge base, and all three `DriftMonitor` scenarios (no drift, environment drift, KB
improvement drift) including a check that noise-level KB improvements don't trigger false alarms.
Full suite: 413/413 passing.
