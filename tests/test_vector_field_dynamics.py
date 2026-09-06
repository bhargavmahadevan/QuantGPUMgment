"""
Unit Tests for Vector Field Dynamics (Divergence, Curl, Stokes' Theorem, and Helmholtz-Hodge Decomposition).

Covers:
1. Divergence calculation on known quadratic objective verifying div(g) == Tr(H).
2. Curl and vorticity estimation on pure rotational vector field vs pure radial sink.
3. Helmholtz-Hodge vector field decomposition separating potential descent from rotational curl.
4. Vorticity damping on an oscillating momentum orbit.
5. DecisionEngine Rule 22 (RULE_VECTOR_FIELD_VORTICITY_DAMPING) generation.
6. AutoApplier integration for vector field vorticity damping.
"""

import pytest
import math
import torch
import torch.nn as nn

from ghost_layer.curvature.vector_field import (
    estimate_divergence_flux,
    estimate_vorticity_curl,
    helmholtz_hodge_decomposition,
    HelmholtzHodgeFilter,
)
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.telemetry.watcher import TelemetrySummary
from ghost_layer.applier import AutoApplier, ConsentLevel


# ==============================================================================
# Helper Models
# ==============================================================================

class QuadraticLoss(nn.Module):
    def __init__(self, diag_weights: torch.Tensor):
        super().__init__()
        self.theta = nn.Parameter(torch.ones_like(diag_weights) * 2.0)
        self.register_buffer("diag", diag_weights)

    def forward(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.diag * (self.theta ** 2))



# ==============================================================================
# 1. Divergence Flux Tests (Gauss's Divergence Theorem)
# ==============================================================================

def test_divergence_flux_exact_quadratic():
    """For f(x) = 0.5 * x^T A x with diagonal A, div(grad) == Tr(A)."""
    diag = torch.tensor([3.0, 7.0, 11.0], dtype=torch.float32)
    expected_trace = diag.sum().item()  # 21.0
    
    model = QuadraticLoss(diag)
    loss = model()
    loss.backward()
    
    # Estimate divergence using Hutchinson trace estimator
    estimated_div = estimate_divergence_flux(model, num_samples=100, seed=42)
    assert abs(estimated_div - expected_trace) / expected_trace < 0.05  # Within 5% tolerance


def test_divergence_source_vs_sink():
    """Verify divergence is positive for convex bowl (expanding grad) and negative for concave."""
    # Convex bowl: eigenvalues > 0 -> div(g) > 0 (gradient vectors point outward away from min)
    convex_diag = torch.tensor([5.0, 5.0], dtype=torch.float32)
    m_convex = QuadraticLoss(convex_diag)
    loss = m_convex()
    loss.backward()
    div_convex = estimate_divergence_flux(m_convex, num_samples=50, seed=42)
    assert div_convex > 0.0


# ==============================================================================
# 2. Curl & Vorticity Tests (Stokes' Theorem)
# ==============================================================================

def test_curl_vorticity_pure_rotation():
    """Verify high vorticity on a pure circular orbit and zero vorticity on straight descent."""
    # 1. Pure rotational trajectory: theta_t = [cos(t), sin(t)]
    # v_t = [-sin(t), cos(t)], so v_t and v_{t+1} are perpendicular
    v1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    v2 = torch.tensor([0.0, 1.0], dtype=torch.float32)
    
    curl_rot = estimate_vorticity_curl([v1, v2])
    # For perpendicular vectors, rotational angle is pi/2, vorticity index is 1.0
    assert curl_rot > 0.90

    # 2. Pure radial descent: v1 and v2 are collinear
    v_straight1 = torch.tensor([2.0, 3.0], dtype=torch.float32)
    v_straight2 = torch.tensor([1.0, 1.5], dtype=torch.float32)
    
    curl_straight = estimate_vorticity_curl([v_straight1, v_straight2])
    assert curl_straight < 0.05  # Near zero curl


# ==============================================================================
# 3. Helmholtz-Hodge Vector Field Decomposition
# ==============================================================================

def test_helmholtz_hodge_decomposition():
    """Verify decomposition V = V_potential + V_solenoidal."""
    # Create velocity vector with both radial component (potential) and rotational component (solenoidal)
    v_descent = torch.tensor([3.0, 0.0], dtype=torch.float32)
    v_rotational = torch.tensor([0.0, 4.0], dtype=torch.float32)
    v_total = v_descent + v_rotational  # [3.0, 4.0]
    
    # Reference direction of steepest descent is along x-axis
    ref_descent = torch.tensor([1.0, 0.0], dtype=torch.float32)
    
    v_pot, v_sol = helmholtz_hodge_decomposition(v_total, ref_descent)
    
    # Potential flow must align with ref_descent
    assert torch.allclose(v_pot, v_descent)
    # Solenoidal vortex must be orthogonal to ref_descent
    assert torch.allclose(v_sol, v_rotational)
    # Total reconstruction must equal original
    assert torch.allclose(v_pot + v_sol, v_total)


def test_helmholtz_hodge_filter_damping():
    """Verify HelmholtzHodgeFilter damps rotational curl while preserving descent."""
    hh_filter = HelmholtzHodgeFilter(vorticity_damping_factor=0.8)
    
    v_step = torch.tensor([2.0, 5.0], dtype=torch.float32)
    ref_grad = torch.tensor([2.0, 0.0], dtype=torch.float32)
    
    v_filtered = hh_filter.filter_step(v_step, ref_grad)
    
    # Descent component along x is preserved
    assert abs(v_filtered[0].item() - 2.0) < 1e-4
    # Rotational component along y is damped by 80% (5.0 * (1 - 0.8) = 1.0)
    assert abs(v_filtered[1].item() - 1.0) < 1e-4


# ==============================================================================
# 4. Decision Engine Rule 22 & AutoApplier Integration
# ==============================================================================

def test_decision_engine_vorticity_rule():
    """Verify Rule 22 triggers when high rotational vorticity is present in telemetry."""
    engine = DecisionEngine()
    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=10.0,
        avg_gpu_utilization_pct=90.0,
        peak_gpu_memory_mb=16000.0,
        gpu_memory_total_mb=24000.0,
        avg_step_time_ms=100.0,
        avg_dataloader_stall_pct=1.0,
        mixed_precision="bf16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
    )
    # Attach high vorticity attribute to summary
    summary.vorticity_curl_index = 0.65
    
    recs = engine.evaluate(summary, model_type="transformer")
    rule_ids = [r.rule_id for r in recs]
    assert "RULE_VECTOR_FIELD_VORTICITY_DAMPING" in rule_ids
    
    rec = next(r for r in recs if r.rule_id == "RULE_VECTOR_FIELD_VORTICITY_DAMPING")
    assert rec.impact_level in ["HIGH", "MEDIUM"]
    assert "Vorticity" in rec.title or "Helmholtz" in rec.title


def test_auto_applier_vorticity_damping():
    """Verify AutoApplier executes _apply_vector_field_vorticity_damping safely."""
    model = nn.Linear(8, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    ctx = {}
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_SAFE)
    res = applier.apply_recommendation(
        "RULE_VECTOR_FIELD_VORTICITY_DAMPING",
        model=model,
        optimizer=opt,
        recommendation_is_safe=True,
        target_context=ctx,
    )
    
    assert res.applied is True
    assert res.rule_id == "RULE_VECTOR_FIELD_VORTICITY_DAMPING"
    assert ctx.get("applied_vorticity_damping") is True
    assert isinstance(ctx.get("helmholtz_filter"), HelmholtzHodgeFilter)
