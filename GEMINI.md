# Antigravity Global Agent Directives & Operational Rules

> **ALWAYS ACTIVE**: This workspace operates under the unified **gstack Virtual Engineering Team** and **Anti-"Vibe Coding" Disciplined Engineering Harness** (`github/spec-kit`, `mattpocock/skills`, `ziadmomen10/zaude`, `pitimon/8-habit-ai-dev`).
> These directives are mandatory and apply to EVERY interaction.

---

## 1. Always-Active Core Mechanisms

1. **Plan-First Gates:**
   - On any non-trivial code modification or architectural addition, always formulate an explicit implementation plan with touched files, input/output contracts, and failure boundaries.
   - Halt for approval before writing or editing production files.
2. **Test-Driven Development (TDD) Loops (`/tdd`):**
   - **RED:** Write failing unit tests in `tests/` capturing happy paths, edge cases, and error boundaries. Verify failure using `pytest`.
   - **GREEN:** Write the minimal code necessary to make tests pass. Verify green.
   - **REFACTOR:** Clean up, enforce strict type annotations, and ensure 100% pass rate across the full 385+ test suite.
3. **Pre-Implementation Interrogation (`/grill-me`):**
   - On ambiguous prompts or high-risk features, proactively interview the developer on edge cases, state invariants, and failure modes.
4. **Spec-Driven Development (`/spec-kit`):**
   - Treat code as an output generated against living specifications and explicit invariant contracts.
5. **Zero Hallucinated Dependencies & Repo Guardrails:**
   - Never add or install packages (`pip`, `npm`, `pnpm`) without explicit user permission.
   - Restrict edits strictly to targeted functions and files. No unprompted cosmetic refactoring.
6. **Zero Fabrication & Absolute Empirical Realism:**
   - Never fabricate, simulate, or assume benchmark figures, hardware results, or telemetry numbers. State clearly when tests are unexecuted.
7. **Anti-Tautological & Genuine Tests:**
   - Never author circular tests that tailor questions/inputs to suit hardcoded answers. Never pad test counts with synthetic sweeps or mock data loops. Every test must verify genuine behavioral invariants, edge cases, mathematical properties, or error handling.

---

## 2. gstack Software Factory & Sprint Cycle

```
Think (/office-hours)
  → Plan (/plan-ceo-review, /plan-eng-review, /plan-design-review, /autoplan, /spec-kit)
  → Build (/grill-me, /tdd, Bounded Scope)
  → Review (/review, /cso, /freebuff)
  → Test (/qa, /qa-only, /investigate, ghostlayer-qa)
  → Ship (/ship, /land-and-deploy)
  → Reflect (/retro, /learn)
```

- **Boil the Ocean:** Completeness is cheap with AI assistance. Always build the full, robust implementation with comprehensive error handling and tests.
- **Search Before Building:** Standard library → Existing dependencies → First principles.
- **User Sovereignty:** AI models recommend, user decides. Never alter strategic direction without asking.

---

## 3. Reference Files
- Main Directives: [.agents/AGENTS.md](file:///c:/Users/bharg/OneDrive/Documents/bhargav_projects/QuantGPUMgment/.agents/AGENTS.md)
- Anti-Vibe Coding Rules: [.agents/rules/anti-vibe-coding.md](file:///c:/Users/bharg/OneDrive/Documents/bhargav_projects/QuantGPUMgment/.agents/rules/anti-vibe-coding.md)
- Code Quality Standards: [.agents/rules/code-quality.md](file:///c:/Users/bharg/OneDrive/Documents/bhargav_projects/QuantGPUMgment/.agents/rules/code-quality.md)
- GhostLayer Engineering: [.agents/rules/ghostlayer-engineering.md](file:///c:/Users/bharg/OneDrive/Documents/bhargav_projects/QuantGPUMgment/.agents/rules/ghostlayer-engineering.md)
