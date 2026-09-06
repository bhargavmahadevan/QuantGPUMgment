"""
GhostLayer Curvature Optimizers, Lagrangian Loss, & Manifold Connections Test Suite.
Combines tests for:
- Pearlmutter HVP & local curvature diagnostics
- SophiaG second-order stochastic clipped optimizer
- Muon Newton-Schulz matrix polar decomposition optimizer & HybridMuonOptimizer
- Shampoo matrix root tensor preconditioning
- LagrangianErrorController & Welford online variance tracking
- ChunkedCrossEntropyLoss exact loss match & backward gradient fidelity
- Manifold-Constrained Hyper-Connections (mHC) Sinkhorn-Knopp & Perron-Frobenius invariance
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# GhostLayer Curvature & Verification Modules
from ghost_layer.curvature import (
    compute_hvp_pearlmutter,
    estimate_local_curvature,
    SophiaG,
    Muon,
    Shampoo,
    newton_schulz5,
    matrix_power_inverse_root,
    HybridMuonOptimizer,
    create_muon_hybrid_optimizer,
    ChunkedCrossEntropyLoss,
    chunked_cross_entropy,
    sinkhorn_knopp_doubly_stochastic,
    birkhoff_drift_metric,
    mHCResidual,
)
from ghost_layer.verification.lagrangian import LagrangianErrorController, WelfordState
from ghost_layer.verification.verifier import CorrectnessVerifier
from ghost_layer.decision.engine import DecisionEngine
from ghost_layer.telemetry.watcher import TelemetrySummary


# ==============================================================================
# Helper Dummy Models
# ==============================================================================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ==============================================================================
# 1. Pearlmutter HVP & Curvature Diagnostics Tests
# ==============================================================================

def test_pearlmutter_hvp_exact_quadratic():
    A_f32 = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    theta_f32 = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
    v_f32 = [torch.tensor([1.0, 0.0], dtype=torch.float32)]

    def loss_fn_f32():
        return 0.5 * torch.dot(theta_f32, torch.mv(A_f32, theta_f32))

    hvp_f32 = compute_hvp_pearlmutter(loss_fn_f32, [theta_f32], v_f32, epsilon=1e-4)
    expected_f32 = torch.tensor([4.0, 1.0], dtype=torch.float32)
    assert torch.allclose(hvp_f32[0], expected_f32, atol=2e-3)

    A_f64 = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    theta_f64 = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float64))
    v_f64 = [torch.tensor([1.0, 0.0], dtype=torch.float64)]

    def loss_fn_f64():
        return 0.5 * torch.dot(theta_f64, torch.mv(A_f64, theta_f64))

    hvp_f64 = compute_hvp_pearlmutter(loss_fn_f64, [theta_f64], v_f64, epsilon=1e-5)
    expected_f64 = torch.tensor([4.0, 1.0], dtype=torch.float64)
    assert torch.allclose(hvp_f64[0], expected_f64, atol=1e-6)


def test_sophia_hutchinson_ground_truth_quadratic():
    A = torch.tensor([[6.0, 2.0], [2.0, 4.0]], dtype=torch.float32)
    theta = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))

    def loss_fn():
        return 0.5 * torch.dot(theta, torch.mv(A, theta))

    rademacher_states = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, -1.0]),
    ]
    hvp_samples = []
    for u in rademacher_states:
        hvp = compute_hvp_pearlmutter(loss_fn, [theta], [u], epsilon=1e-4)[0]
        hvp_samples.append(u * hvp)

    expected_analytical_diag = torch.tensor([6.0, 4.0])
    mean_estimate = torch.stack(hvp_samples).mean(dim=0)
    assert torch.allclose(mean_estimate, expected_analytical_diag, atol=5e-3)

    torch.manual_seed(42)
    optimizer = SophiaG([theta], lr=1e-3, betas=(0.9, 0.95), hessian_estimator="hvp")

    for step_idx in range(150):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.update_hessian(loss_fn=loss_fn)

    diag_est = optimizer.state[theta]["hessian"]
    assert torch.allclose(diag_est, expected_analytical_diag, rtol=0.15)


def test_muon_optimizer_step():
    torch.manual_seed(42)
    model = SimpleMLP()
    x = torch.randn(16, 16)
    y = torch.randn(16, 1)
    optimizer = Muon(model.parameters(), lr=0.01, momentum=0.9)

    loss_fn = nn.MSELoss()
    initial_loss = loss_fn(model(x), y).item()

    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).item()
    assert final_loss < initial_loss


def test_newton_schulz_polar_orthogonality():
    torch.manual_seed(42)
    G = torch.randn(16, 16)
    U = newton_schulz5(G, steps=6)

    s = torch.linalg.svdvals(U)
    assert (s > 0.5).all()
    assert (s < 1.5).all()
    assert s.mean().item() == pytest.approx(1.0, abs=0.2)


def test_shampoo_optimizer_step():
    torch.manual_seed(42)
    model = nn.Linear(8, 4)
    x = torch.randn(16, 8)
    y = torch.randn(16, 4)
    optimizer = Shampoo(model.parameters(), lr=0.01, momentum=0.9, update_preconditioner_interval=1)

    loss_fn = nn.MSELoss()
    initial_loss = loss_fn(model(x), y).item()

    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).item()
    assert final_loss < initial_loss


def test_shampoo_dimension_guard():
    M_small = torch.randn(16, 16)
    cov_small = M_small @ M_small.T
    root_small = matrix_power_inverse_root(cov_small, root=4)
    assert root_small.shape == (16, 16)

    M_large = torch.eye(32) * 2.0
    root_large = matrix_power_inverse_root(M_large, root=4, max_eigh_dim=16)
    assert root_large.shape == (32, 32)


def test_curvature_decision_engine_rules():
    engine = DecisionEngine()
    summary = TelemetrySummary(
        total_steps=100,
        total_duration_sec=10.0,
        avg_gpu_utilization_pct=85.0,
        peak_gpu_memory_mb=4000.0,
        gpu_memory_total_mb=16000.0,
        avg_step_time_ms=25.0,
        avg_dataloader_stall_pct=2.0,
        mixed_precision="fp16",
        gradient_checkpointing=False,
        flash_attention=True,
        num_workers=4,
        pin_memory=True,
    )

    recs = engine.evaluate(
        summary=summary,
        model_type="transformer",
        inverse_cost_weight=0.55,
    )

    rule_ids = [r.rule_id for r in recs]
    assert "RULE_VARIANCE_CONTAINMENT" in rule_ids
    assert "RULE_MUON_OPTIMIZER" in rule_ids
    assert "RULE_SOPHIA_SECOND_ORDER" in rule_ids
    assert "RULE_SHAMPOO_PRECONDITIONING" in rule_ids
    assert "RULE_PEARLMUTTER_HVP_METAGRADIENT" in rule_ids

    sophia_rec = next(r for r in recs if r.rule_id == "RULE_SOPHIA_SECOND_ORDER")
    assert "0.850" in sophia_rec.description
    assert "HEURISTIC_PRIOR" in sophia_rec.description
    assert "Unvalidated Projection" in sophia_rec.speedup_estimate_label
    assert "0.850 threshold" in sophia_rec.evidence

    shampoo_rec = next(r for r in recs if r.rule_id == "RULE_SHAMPOO_PRECONDITIONING")
    assert "0.600" in shampoo_rec.description
    assert "HEURISTIC_PRIOR" in shampoo_rec.description
    assert "Unvalidated Projection" in shampoo_rec.speedup_estimate_label
    assert "0.600 threshold" in shampoo_rec.evidence

    muon_rec = next(r for r in recs if r.rule_id == "RULE_MUON_OPTIMIZER")
    assert "Unvalidated Projection" in muon_rec.speedup_estimate_label


def test_hybrid_muon_optimizer_split():
    class TransformerToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, 32)
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=True)
            self.norm = nn.LayerNorm(32)
            self.lm_head = nn.Linear(32, 64, bias=False)

        def forward(self, idx):
            x = self.embed_tokens(idx)
            x = self.q_proj(x) + self.v_proj(x)
            x = self.norm(x)
            return self.lm_head(x)

    model = TransformerToyModel()
    hybrid_opt = create_muon_hybrid_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)

    assert hybrid_opt.muon_param_count == 2048
    assert hybrid_opt.adamw_param_count == 4192

    idx = torch.randint(0, 64, (4, 8))
    loss = model(idx).sum()
    loss.backward()
    hybrid_opt.step()


def test_hybrid_muon_custom_naming_and_module_introspection():
    class CustomArchitecture(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_lookup = nn.Embedding(128, 32)
            self.internal_transform = nn.Linear(32, 32, bias=False)
            self.final_dense = nn.Linear(32, 128, bias=False)

        def forward(self, idx):
            x = self.token_lookup(idx)
            x = self.internal_transform(x)
            return self.final_dense(x)

    model = CustomArchitecture()
    opt = HybridMuonOptimizer(model, muon_lr=0.01, adamw_lr=1e-3, vocab_size=128)

    assert opt.muon_param_count == 1024
    assert opt.adamw_param_count == 8192


def test_hybrid_muon_regime_3_snr_and_regime_4_late_stage():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 16, bias=False),
        nn.LayerNorm(16),
        nn.Linear(16, 4, bias=False),
    )
    opt = HybridMuonOptimizer(model, muon_lr=0.01, adamw_lr=1e-3, adaptive_snr_damping=True)

    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    opt.set_late_stage_factor(0.1)
    for group in opt.muon.param_groups:
        assert group["late_stage_factor"] == 0.1

    opt.zero_grad()
    loss2 = model(x).sum()
    loss2.backward()
    opt.step()


# ==============================================================================
# 2. Lagrangian Error Controller & Welford Online Variance Tests
# ==============================================================================

def test_welford_online_statistics():
    welford = WelfordState()
    values = [2.5, 3.0, 3.5, 4.0, 4.5]
    for v in values:
        welford.update(v)

    tensor_vals = torch.tensor(values)
    assert welford.count == 5
    assert pytest.approx(welford.mean, rel=1e-5) == float(tensor_vals.mean())
    assert pytest.approx(welford.variance, rel=1e-5) == float(tensor_vals.var(unbiased=True))
    assert pytest.approx(welford.std, rel=1e-5) == float(tensor_vals.std(unbiased=True))


def test_lagrangian_safe_trajectory():
    ctrl = LagrangianErrorController(tau_base=0.10, eta_lambda=0.05)
    base_losses = [3.0 - 0.05 * i for i in range(30)]
    opt_losses = [3.0 - 0.05 * i + 0.005 for i in range(30)]

    result = ctrl.verify_trajectories(base_losses, opt_losses)
    assert result["is_safe"] is True
    assert result["final_lambda"] == 0.0
    assert result["unsafe_step_count"] == 0


def test_lagrangian_divergence_trip():
    ctrl = LagrangianErrorController(tau_base=0.08, eta_lambda=0.10)
    base_losses = [2.0 - 0.02 * i for i in range(40)]
    opt_losses = [2.0 + 0.15 * i for i in range(40)]

    result = ctrl.verify_trajectories(base_losses, opt_losses)
    assert result["is_safe"] is False
    assert result["final_lambda"] > 0.8
    assert result["unsafe_step_count"] > 0


def test_correctness_verifier_lagrangian_mode():
    verifier = CorrectnessVerifier(max_allowed_loss_delta=0.05, use_lagrangian_dual=True)
    base = [2.5, 2.4, 2.3, 2.2, 2.1]
    opt_safe = [2.505, 2.404, 2.306, 2.203, 2.105]

    res = verifier.verify_trajectories(base, opt_safe, recommendation_id="RULE_CHUNKED_CE")
    assert res.is_safe is True
    assert res.action_taken == "VERIFIED_SAFE"
    assert res.lagrangian_lambda == 0.0


# ==============================================================================
# 3. Chunked Cross-Entropy Loss Tests
# ==============================================================================

def test_chunked_cross_entropy_exact_match_mean():
    torch.manual_seed(42)
    B, S, D, V = 2, 64, 32, 128
    hidden = torch.randn(B, S, D, requires_grad=True)
    lm_head = torch.randn(V, D, requires_grad=True)
    targets = torch.randint(0, V, (B, S))

    full_logits = F.linear(hidden, lm_head)
    standard_loss = F.cross_entropy(full_logits.view(-1, V), targets.view(-1), reduction="mean")

    chunked_loss = chunked_cross_entropy(
        hidden_states=hidden,
        lm_head_weight=lm_head,
        targets=targets,
        chunk_size=16,
        reduction="mean",
    )

    assert torch.allclose(standard_loss, chunked_loss, atol=1e-5)


def test_chunked_cross_entropy_backward_gradient_fidelity():
    torch.manual_seed(42)
    B, S, D, V = 2, 32, 16, 64
    targets = torch.randint(0, V, (B, S))

    h1 = torch.randn(B, S, D, requires_grad=True)
    w1 = torch.randn(V, D, requires_grad=True)
    logits1 = F.linear(h1, w1)
    loss1 = F.cross_entropy(logits1.view(-1, V), targets.view(-1))
    loss1.backward()

    h2 = h1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    loss2 = chunked_cross_entropy(
        hidden_states=h2,
        lm_head_weight=w2,
        targets=targets,
        chunk_size=8,
        reduction="mean",
    )
    loss2.backward()

    assert torch.allclose(loss1, loss2, atol=1e-5)
    assert torch.allclose(h1.grad, h2.grad, atol=1e-5)
    assert torch.allclose(w1.grad, w2.grad, atol=1e-5)


def test_chunked_cross_entropy_with_ignore_index():
    torch.manual_seed(42)
    B, S, D, V = 2, 32, 16, 64
    hidden = torch.randn(B, S, D)
    lm_head = torch.randn(V, D)
    targets = torch.randint(0, V, (B, S))
    targets[0, 5:15] = -100

    std_loss = F.cross_entropy(F.linear(hidden, lm_head).view(-1, V), targets.view(-1), ignore_index=-100)
    chk_loss = chunked_cross_entropy(hidden, lm_head, targets, chunk_size=8, ignore_index=-100)

    assert torch.allclose(std_loss, chk_loss, atol=1e-5)


def test_chunked_cross_entropy_module_wrapper():
    loss_fn = ChunkedCrossEntropyLoss(chunk_size=32, label_smoothing=0.1)
    h = torch.randn(4, 64, 32)
    w = torch.randn(100, 32)
    y = torch.randint(0, 100, (4, 64))

    loss = loss_fn(h, w, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


# ==============================================================================
# 4. Manifold-Constrained Hyper-Connections (mHC) Tests
# ==============================================================================

def test_sinkhorn_knopp_doubly_stochastic_properties():
    torch.manual_seed(42)
    raw_logits = torch.randn(4, 4) * 2.5
    H = sinkhorn_knopp_doubly_stochastic(raw_logits, iters=30)

    assert (H >= 0.0).all()

    row_sums = H.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(4), atol=2e-2)

    col_sums = H.sum(dim=-2)
    assert torch.allclose(col_sums, torch.ones(4), atol=2e-2)


def test_perron_frobenius_spectral_radius():
    torch.manual_seed(42)
    for n in [2, 4, 8]:
        raw = torch.randn(n, n)
        H = sinkhorn_knopp_doubly_stochastic(raw, iters=10)

        eigvals = torch.linalg.eigvals(H)
        max_eig_mag = float(torch.max(torch.abs(eigvals)).item())
        assert max_eig_mag == pytest.approx(1.0, abs=1e-3)


def test_birkhoff_drift_metric():
    torch.manual_seed(42)
    H_clean = sinkhorn_knopp_doubly_stochastic(torch.randn(4, 4), iters=10)
    drift_clean = birkhoff_drift_metric(H_clean)
    assert drift_clean == pytest.approx(0.0, abs=1e-3)

    H_perturbed = H_clean.clone()
    H_perturbed[0, 0] += 0.5
    drift_perturbed = birkhoff_drift_metric(H_perturbed)
    assert drift_perturbed > 0.4


def test_mhc_residual_layer_forward_and_backward():
    torch.manual_seed(42)
    B, S, D, N = 2, 8, 32, 4
    x = torch.randn(B, S, D)

    sublayer = nn.Sequential(
        nn.Linear(D, D * 2),
        nn.ReLU(),
        nn.Linear(D * 2, D),
    )

    mhc = mHCResidual(hidden_dim=D, num_streams=N, sinkhorn_iters=6)

    x_streams = mhc(x, sublayer_fn=sublayer)
    assert x_streams.shape == (B, S, N, D)
    assert mhc.last_birkhoff_drift == pytest.approx(0.0, abs=1e-3)

    x_out = mhc.collapse_to_single_stream(x_streams)
    assert x_out.shape == (B, S, D)

    loss = x_out.sum()
    loss.backward()

    assert mhc.raw_mixing_weights.grad is not None
    assert torch.norm(mhc.raw_mixing_weights.grad) > 0.0
    assert mhc.in_proj.weight.grad is not None
    assert mhc.out_proj.weight.grad is not None


def test_mhc_multi_stream_toy_transformer_convergence():
    torch.manual_seed(42)
    B, S, D, N = 4, 16, 32, 4

    class mHCToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mhc_attn = mHCResidual(hidden_dim=D, num_streams=N)
            self.attn = nn.Linear(D, D, bias=False)
            self.mhc_mlp = mHCResidual(hidden_dim=D, num_streams=N)
            self.mlp = nn.Sequential(
                nn.Linear(D, D * 2),
                nn.GELU(),
                nn.Linear(D * 2, D),
            )

        def forward(self, x):
            h_streams = self.mhc_attn(x, sublayer_fn=self.attn)
            h_streams = self.mhc_mlp(h_streams, sublayer_fn=self.mlp)
            return self.mhc_mlp.collapse_to_single_stream(h_streams)

    model = mHCToyBlock()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x_input = torch.randn(B, S, D)
    target = torch.randn(B, S, D)
    criterion = nn.MSELoss()

    initial_loss = criterion(model(x_input), target).item()

    for _ in range(10):
        optimizer.zero_grad()
        out = model(x_input)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x_input), target).item()
    assert final_loss < initial_loss
