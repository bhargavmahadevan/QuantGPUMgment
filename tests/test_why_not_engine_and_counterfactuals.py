"""
Unit tests validating Why-NOT Decision Intelligence, Counterfactual Utility Calculus,
The 7-Level Evidence Ladder (E0-E6), and Knowledge Base decay and contamination gating.
"""

import pytest
from ghost_layer.evidence.taxonomy import EvidenceLevel, ExecutionProvenance, EvidenceTag
from ghost_layer.decision.counterfactual import WhyNotEngine, DecisionCounterfactual
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.telemetry.watcher import TelemetrySummary
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, KnowledgeEntry


def test_evidence_ladder_hierarchy():
    """Verify 7-level evidence ladder ranking and provenance formatting."""
    assert EvidenceLevel.E0_THEORETICAL.rank == 0
    assert EvidenceLevel.E3_CONTROLLED_PHYSICAL.rank == 3
    assert EvidenceLevel.E4_REPLICATED_PHYSICAL.rank == 4
    assert EvidenceLevel.E6_INDEPENDENTLY_REPRODUCED.rank == 6

    tag = EvidenceTag(
        level=EvidenceLevel.E4_REPLICATED_PHYSICAL,
        provenance=ExecutionProvenance.REAL_GPU,
        hardware_verified=True,
        experiment_id="EXP-A2000-LLM-001",
    )
    formatted = tag.format_label()
    assert "E4_REPLICATED_PHYSICAL" in formatted
    assert "REAL_GPU" in formatted
    assert "EXP-A2000-LLM-001" in formatted


def test_why_not_engine_rejections():
    """Verify WhyNotEngine produces structured rejection rationale."""
    rejections = WhyNotEngine.evaluate_rejections(
        model_type="transformer",
        mixed_precision="fp32",
        is_compiled=False,
        peak_vram_mb=4000.0,
        total_vram_mb=8000.0,
        active_rule_ids=["RULE_MIXED_PRECISION"],
    )

    assert len(rejections) >= 2
    fp16_rej = next((r for r in rejections if r.rule_id == "RULE_FP16_PRECISION"), None)
    assert fp16_rej is not None
    assert any("dynamic range" in reason.lower() for reason in fp16_rej.rejection_reasons)
    assert "Loss scale collapse" in fp16_rej.risk_factors

    compile_rej = next((r for r in rejections if r.rule_id == "RULE_GRAPHIFY_TORCH_COMPILE"), None)
    assert compile_rej is not None
    assert any("warmup compile overhead" in reason.lower() for reason in compile_rej.rejection_reasons)


def test_counterfactual_utility_calculus():
    """Verify E[Utility] = E[Savings] - E[Risk Cost] arithmetic."""
    cf = WhyNotEngine.calculate_counterfactual(
        baseline_cost_usd=1000.0,
        candidate_speedup_pct=30.0,
        risk_level="MEDIUM",
    )

    assert isinstance(cf, DecisionCounterfactual)
    assert cf.expected_savings_usd == 300.0
    # MEDIUM risk score = 0.08, risk cost = 300 * 0.08 = 24.00
    assert cf.risk_cost_usd == 24.00
    # Utility = 300 - 24 = 276.00
    assert cf.expected_utility_usd == 276.00
    assert cf.expected_cost_usd == 700.00


def test_knowledge_base_decay_weight():
    """Verify half-life exponential time decay formula."""
    entry = KnowledgeEntry(
        entry_id="test",
        architecture_family="transformer",
        hardware_type="rtx_a2000",
        effective_config={},
        throughput_improvement_pct=25.0,
        sample_count=1,
        last_updated_timestamp=1000000.0,
    )

    # At t = t0 (delta = 0), decay weight = 1.0
    w0 = entry.get_decay_weight(half_life_days=30.0, current_time=1000000.0)
    assert w0 == 1.0

    # At t = 30 days later, decay weight should be ~0.50
    t_30d = 1000000.0 + (30.0 * 86400.0)
    w_30d = entry.get_decay_weight(half_life_days=30.0, current_time=t_30d)
    assert abs(w_30d - 0.50) < 0.01

    # At t = 60 days later, decay weight should be ~0.25
    t_60d = 1000000.0 + (60.0 * 86400.0)
    w_60d = entry.get_decay_weight(half_life_days=30.0, current_time=t_60d)
    assert abs(w_60d - 0.25) < 0.01


def test_knowledge_base_rejects_contaminated_run():
    """Verify that contaminated/throttled runs cannot update the Knowledge Base."""
    kb = SharedKnowledgeBase(db_file_path=None)
    initial_count = len(kb.entries)

    result = kb.register_learning(
        architecture_family="llama",
        hardware_type="a100",
        effective_config={"mixed_precision": "bf16"},
        throughput_improvement_pct=40.0,
        was_verified_safe=True,
        is_contaminated=True,  # Contaminated!
    )

    assert result is None
    assert len(kb.entries) == initial_count
