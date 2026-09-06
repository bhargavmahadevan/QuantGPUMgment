# GhostLayer: Semantic Versioning & API Stability Policy

> **Current Version**: `0.2.0`  
> **Specification Standard**: [Semantic Versioning 2.0.0 (SemVer)](https://semver.org/)

---

## 1. Version Format

GhostLayer releases adhere to the `MAJOR.MINOR.PATCH` format:

$$\text{Version} = \text{MAJOR} \,.\, \text{MINOR} \,.\, \text{PATCH}$$

- **MAJOR**: Incompatible API changes, breaking modifications to the PyTorch context hook lifecycle, backward-incompatible schema changes in persisted Knowledge Base artifacts, or removals of public exports.
- **MINOR**: Backward-compatible functionality additions (new optimization rules, new metrics in `TelemetryBoundary`, new exporter integrations like WandB or Prometheus, or adaptive calibration algorithms).
- **PATCH**: Backward-compatible bug fixes, typing hygiene updates, documentation revisions, or performance refinements that preserve behavioral invariants.

---

## 2. Public API Surface Contracts

The following modules and interfaces comprise GhostLayer's guaranteed public surface:

1. **Context Hooks & Lifecycle**:
   - `ghost_layer.hooks.GhostWatcherHook`
   - `ghost_layer.distributed.FSDPGhostHook`, `DeepSpeedGhostHook`
2. **Decision & Verification Pipeline**:
   - `ghost_layer.decision.engine.DecisionEngine` (`evaluate()`)
   - `ghost_layer.verification.verifier.CorrectnessVerifier` (`verify()`)
   - `ghost_layer.rollback.RollbackManager` (`snapshot()`, `rollback()`)
   - `ghost_layer.applier.AutoApplier` (`apply()`)
3. **Telemetry & Privacy Scrubber**:
   - `ghost_layer.telemetry.watcher.TelemetryWatcher`, `MetricSnapshot`, `TelemetrySummary`
   - `ghost_layer.kb.knowledge_base.TelemetryBoundary` (`COLLECTED_FIELDS`, `EXCLUDED_FIELDS`)
4. **State Persistence & Replay**:
   - `ghost_layer.replay.OptimizationReplayLog` (`record_event()`, `generate_audit_trail()`)
   - `ghost_layer.kb.knowledge_base.SharedKnowledgeBase` (`register_learning()`, `register_rule_outcome()`, `get_calibrated_threshold()`)

---

## 3. Deprecation & Breaking Change Policy

- **Notice Period**: Any planned breaking change to hook arguments or telemetry contracts will be marked with a `FutureWarning` deprecation notice across at least **one full minor release** (e.g., deprecated in `0.3.0`, removed in `1.0.0`).
- **Telemetry Schema Invariance**: Existing fields in `MetricSnapshot` and `TelemetrySummary` will not be removed or have their data types altered within a single `MAJOR` release. New fields will always be added with default values to preserve caller backward compatibility.
- **Decision Record Compatibility**: All structured audit receipts and JSON decision records include explicit schema versions (`schema_version: "1.0"`) to allow forward and backward compatibility across parser versions.

---

## 4. Pre-Release Shipping Gate Requirement

Before any version bump is committed or packaged:
1. `python scripts/verify_shipping_hygiene.py` MUST exit with code 0 (zero unimported typing annotations, 100% module importability, 100% test collection).
2. The entire test suite MUST pass with zero failures:
   ```bash
   pytest tests/ ceo_evidence_engine/test_evidence_suite.py
   ```
