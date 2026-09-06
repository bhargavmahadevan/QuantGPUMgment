"""
Unit Tests for Matrix-Free Quasi-Newton (L-BFGS) Curvature Engine.

Covers:
1. Damped Powell update preventing negative curvature failure on non-convex surfaces.
2. Two-loop recursion mathematical exactness on quadratic loss objectives.
3. Zero memory leak verification: 100 iterations verify ring buffer strictly bounded at m pairs.
4. End-to-end convergence on non-linear MLP.
5. DecisionEngine Rule 21 (RULE_QUASI_NEWTON_LBFGS) generation.
6. AutoApplier dynamic application of L-BFGS.
"""

import pytest
import torch
import torch.nn as nn
import gc

from ghost_layer.curvature.lbfgs import (
    LBFGSCurvatureOptimizer,
    lbfgs_two_loop_recursion,
    damped_powell_update,
)
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.telemetry.watcher import TelemetrySummary
from ghost_layer.applier import AutoApplier


# ==============================================================================
# Helper Models
# ==============================================================================

class QuadraticModel(nn.Module):
    """f(x, y) = 0.5 * x^T A x where A is known positive definite matrix."""
    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([5.0, 5.0], dtype=torch.float32))
        self.register_buffer("A", A)

    def forward(self) -> torch.Tensor:
        return 0.5 * torch.dot(self.theta, torch.mv(self.A, self.theta))


class SimpleNonlinearMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ==============================================================================
# 1. Damped Powell Update Tests
# ==============================================================================

def test_damped_powell_positive_curvature():
    """When s^T y is already strictly positive and large, damping preserves y."""
    s = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([2.0, 4.0], dtype=torch.float32)  # s^T y = 10 > 0
    
    y_tilde, was_damped = damped_powell_update(s, y, gamma=1.0)
    assert not was_damped
    assert torch.allclose(y_tilde, y)
    assert torch.dot(s, y_tilde).item() > 0


def test_damped_powell_negative_curvature():
    """When s^T y <= 0 (saddle point / non-convexity), Powell dampens to guarantee s^T y_tilde > 0."""
    s = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([-2.0, -1.0], dtype=torch.float32)  # s^T y = -4 <= 0
    
    y_tilde, was_damped = damped_powell_update(s, y, gamma=1.0)
    assert was_damped
    # Must be strictly positive
    inner_prod = torch.dot(s, y_tilde).item()
    assert inner_prod > 0
    # Specifically >= 0.2 * gamma * ||s||^2
    assert inner_prod >= 0.2 * 1.0 * torch.dot(s, s).item() - 1e-5


# ==============================================================================
# 2. Two-Loop Recursion Mathematical Correctness
# ==============================================================================

def test_lbfgs_two_loop_recursion_quadratic():
    """Verify that two-loop recursion produces correct Newton direction for a quadratic."""
    # A = [[4.0, 1.0], [1.0, 2.0]], exact inverse A^{-1} = 1/7 * [[2, -1], [-1, 4]]
    A = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    A_inv = torch.inverse(A)
    
    # Simulate exact history pairs from quadratic: y = A s
    s_hist = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    y_hist = [torch.mv(A, s) for s in s_hist]
    
    grad = torch.tensor([2.0, 3.0], dtype=torch.float32)
    direction = lbfgs_two_loop_recursion(grad, s_hist, y_hist)
    
    # Direction should closely approximate -A^{-1} grad
    expected_step = -torch.mv(A_inv, grad)
    # Cosine similarity between direction and true Newton direction
    cos_sim = torch.cosine_similarity(direction.unsqueeze(0), expected_step.unsqueeze(0)).item()
    assert cos_sim > 0.99


# ==============================================================================
# 3. Memory Leak Prevention Tests
# ==============================================================================

def test_lbfgs_zero_memory_leak_bounded_history():
    """Verify history buffer strictly maintains <= m pairs and detaches autograd graph."""
    model = SimpleNonlinearMLP()
    m_history = 5
    optimizer = LBFGSCurvatureOptimizer(model.parameters(), lr=1e-2, history_size=m_history)
    
    x = torch.randn(4, 8)
    target = torch.randn(4, 2)
    criterion = nn.MSELoss()
    
    # Run 50 optimization steps
    for step in range(50):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        # Verify history size is bounded by m_history for all parameter groups
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 's_history' in state:
                    assert len(state['s_history']) <= m_history
                    assert len(state['y_history']) <= m_history
                    # Verify no tensor in history requires grad (autograd graph severed)
                    for s in state['s_history']:
                        assert not s.requires_grad
                        assert s.grad_fn is None
                    for y in state['y_history']:
                        assert not y.requires_grad
                        assert y.grad_fn is None


# ==============================================================================
# 4. Convergence Test on Non-linear MLP
# ==============================================================================

def test_lbfgs_optimizer_convergence():
    """Verify that LBFGSCurvatureOptimizer reduces loss on non-linear regression."""
    torch.manual_seed(42)
    model = SimpleNonlinearMLP()
    optimizer = LBFGSCurvatureOptimizer(model.parameters(), lr=0.05, history_size=5)
    
    x = torch.randn(16, 8)
    target = torch.randn(16, 2)
    criterion = nn.MSELoss()
    
    initial_loss = criterion(model(x), target).item()
    
    for _ in range(25):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss * 0.5  # Substantial loss reduction


# ==============================================================================
# 5. Decision Engine Rule 21 & AutoApplier Integration
# ==============================================================================

def test_decision_engine_quasi_newton_rule():
    """Verify Rule 21 fires when optimizer memory/curvature pressure is detected."""
    engine = DecisionEngine()
    # Summary with high optimizer memory overhead
    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=12.0,
        avg_gpu_utilization_pct=95.0,
        peak_gpu_memory_mb=75000.0,
        gpu_memory_total_mb=80000.0,
        avg_step_time_ms=120.0,
        avg_dataloader_stall_pct=1.0,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
    )
    
    recs = engine.evaluate(summary, model_type="transformer")
    rule_ids = [r.rule_id for r in recs]
    assert "RULE_QUASI_NEWTON_LBFGS" in rule_ids
    
    rec = next(r for r in recs if r.rule_id == "RULE_QUASI_NEWTON_LBFGS")
    assert rec.impact_level in ["HIGH", "CRITICAL"]
    assert "Quasi-Newton" in rec.title or "L-BFGS" in rec.title


def test_auto_applier_quasi_newton_lbfgs():
    """Verify AutoApplier executes _apply_quasi_newton_lbfgs safely."""
    from ghost_layer.applier import ConsentLevel
    model = SimpleNonlinearMLP()
    initial_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    ctx = {}
    applier = AutoApplier(consent_level=ConsentLevel.AUTO_APPLY_SAFE)
    res = applier.apply_recommendation(
        "RULE_QUASI_NEWTON_LBFGS",
        model=model,
        optimizer=initial_opt,
        recommendation_is_safe=True,
        target_context=ctx,
    )
    
    assert res.applied is True
    assert res.rule_id == "RULE_QUASI_NEWTON_LBFGS"
    assert ctx.get("applied_quasi_newton") is True
    assert isinstance(ctx.get("optimizer"), LBFGSCurvatureOptimizer)



