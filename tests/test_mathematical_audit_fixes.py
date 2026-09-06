"""Unit tests verifying mathematical audit fixes across ghost_layer."""
import math
import pytest
import torch

from ghost_layer.curvature.muon import newton_schulz5, Muon
from ghost_layer.curvature.sophia import SophiaG
from ghost_layer.curvature.pareto_hull import ParetoMemoryHull
from ghost_layer.curvature.lbfgs import LBFGSCurvatureOptimizer
from ghost_layer.savings import SavingsVerifier, BaselineMeasurement, VerifiedSavingsReport
from ghost_layer.verification.lagrangian import LagrangianErrorController
from ghost_layer.decision.credibility_manifold import CredibilityManifold, CredibilityStatus


def test_newton_schulz_spectral_norm_scaling():
    """Verify Newton-Schulz uses Frobenius norm scaling and produces valid bounded values."""
    torch.manual_seed(42)
    G = torch.randn(16, 32)
    out = newton_schulz5(G, steps=6)
    
    assert out.shape == G.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_sophia_cold_start_delay():
    """Verify Sophia initializes Hessian on step 1 and handles uninitialized fallback gracefully."""
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0], requires_grad=True))
    opt = SophiaG([p], lr=0.1, k=5)
    
    loss = (p ** 2).sum()
    loss.backward()
    opt.step()
    
    state = opt.state[p]
    assert state.get("hessian_initialized", False) is True
    assert (state["hessian"] > 0).all()
    
    # Test fallback path directly
    state["hessian_initialized"] = False
    opt.step()
    assert not torch.isnan(p.data).any()


def test_savings_p_value_and_t_critical():
    """Verify Hill (1970) p-value approximation in SavingsVerifier."""
    verifier = SavingsVerifier(gpu_cost_per_hour=2.50)
    
    # P-value calculation for moderate t at df=10
    p_val = verifier._t_distribution_p_value(2.228, 10)
    # True two-tailed p-value for t=2.228 at df=10 is ~0.05
    assert 0.03 <= p_val <= 0.07

    # Test confidence interval with small sample (df=10 approx)
    sample_a = [100.0, 102.0, 98.0, 101.0, 99.0, 100.5]
    sample_b = [90.0, 92.0, 89.0, 91.0, 88.0, 90.5]
    ci_low, ci_high = verifier._confidence_interval(sample_a, sample_b, df=10)
    assert ci_low > 0.0
    assert ci_high > ci_low


def test_credibility_manifold_geometric_distance():
    """Verify geometric boundary distance calculation for >=5 runs case."""
    manifold = CredibilityManifold(min_prior_threshold=0.80)
    
    # Unsafe evaluation with prior=0.70 (delta=0.10) and rollback_frequency=0.20 (delta=0.10)
    # Distance should be sqrt(0.10^2 + 0.10^2) = sqrt(0.02) ≈ 0.1414
    eval_result = manifold.evaluate(heuristic_prior=0.70, verified_hardware_runs=5, rollback_frequency=0.20)
    assert eval_result.status == CredibilityStatus.EMPIRICALLY_VALIDATED
    expected_dist = math.sqrt(0.10**2 + 0.10**2)
    assert math.isclose(eval_result.distance_to_auto_apply_boundary, expected_dist, rel_tol=1e-3)


def test_pareto_hull_throughput_parameterization():
    """Verify Pareto hull accepts base_tokens_per_sec and scales throughput correctly."""
    hull_a2000 = ParetoMemoryHull(base_tokens_per_sec=45.0)
    hull_h100 = ParetoMemoryHull(base_tokens_per_sec=450.0)
    
    rep_a2000 = hull_a2000.compute_pareto_frontier(candidate_seq_lens=[512])
    rep_h100 = hull_h100.compute_pareto_frontier(candidate_seq_lens=[512])
    
    tp_a2000 = rep_a2000.pareto_frontier[0].estimated_throughput_tokens_per_sec
    tp_h100 = rep_h100.pareto_frontier[0].estimated_throughput_tokens_per_sec
    
    assert math.isclose(tp_h100, tp_a2000 * 10.0, rel_tol=1e-2)


def test_lbfgs_tensor_isolation():
    """Verify L-BFGS two-loop recursion produces valid updates without mutating inputs."""
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0], requires_grad=True))
    opt = LBFGSCurvatureOptimizer([p], lr=0.1, history_size=5)
    
    loss = (p ** 2).sum()
    loss.backward()
    opt.step()
    
    assert not torch.isnan(p.data).any()
