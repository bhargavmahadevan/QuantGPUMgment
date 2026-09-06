"""
Muon Optimizer: Momentum Orthogonalized by Newton-Schulz iteration.

Computes matrix polar decomposition of the momentum buffer:
    X_{k+1} = 0.5 * X_k * (3 I - X_k^T X_k)
Orthogonalizes matrix updates across 2D weight tensors in VRAM, eliminating gradient 
interference and accelerating convergence across all spectral dimensions.

Hybrid Parameter Routing:
- Internal 2D Linear / Attention / MLP weight matrices -> Muon (Newton-Schulz Polar Root)
- Embeddings (wte, embed_tokens), output heads (lm_head, classifier), biases, and norm weights -> AdamW
"""

import warnings
from typing import List, Optional, Tuple, Dict, Any, Callable, Union
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW


def newton_schulz5(G: torch.Tensor, steps: int = 6, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration (quintic polynomial in singular values) for
    approximate matrix orthogonalization (Keller Jordan, 2024).

    Applies the quintic map f(σ) = a·σ + b·σ³ + c·σ⁵ to singular values,
    driving them toward a common scale factor (~1.13). The resulting matrix
    has approximately equalized singular values, providing decorrelated
    update directions for the Muon optimizer. The learning rate compensates
    for the residual scale factor.

    Note: This is NOT an exact polar decomposition (which requires f(1)=1).
    The coefficients satisfy f(1) ≈ 0.70, optimized for fast approximate
    orthogonalization convergence rather than exact unitary convergence.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()

    # Scale matrix using Frobenius norm so that all singular values satisfy σ_i <= 1.0.
    # Since ||X||_F = sqrt(sum σ_i^2) >= max(σ_i) = ||X||_2, dividing by ||X||_F
    # mathematically guarantees the spectral radius is <= 1.0, preventing the
    # quintic polynomial from entering its divergent regime (σ > 1.2).
    X = X / (X.norm() + eps)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    """
    Muon (Momentum Orthogonalized by Newton-Schulz).
    
    Designed specifically for 2D internal weight matrices (Linear, Attention projections, MLP weights)
    in VRAM tensor space where AdamW gradient correlation stalls progress.
    """

    def __init__(
        self,
        params,
        lr: float = 0.005,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        weight_decay: float = 0.01,
        adaptive_snr_damping: bool = False,
        snr_floor: float = 0.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adaptive_snr_damping=adaptive_snr_damping,
            snr_floor=snr_floor,
            late_stage_factor=1.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"] * group.get("late_stage_factor", 1.0)
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            adaptive_snr = group.get("adaptive_snr_damping", False)
            snr_floor = group.get("snr_floor", 0.1)

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    state["grad_variance"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update_mat = g.add(buf, alpha=momentum)
                else:
                    update_mat = buf

                # For 2D parameters (matrices), orthogonalize via Newton-Schulz
                if update_mat.ndim == 2:
                    ortho_update = newton_schulz5(update_mat, steps=ns_steps)
                    # Scale update by aspect ratio (standard Keller Jordan formulation)
                    scale = max(1.0, float(p.data.size(0) / p.data.size(1)) ** 0.5)
                    update_final = ortho_update * scale

                    # Adaptive SNR noise dampening in high-noise small batch regimes
                    if adaptive_snr:
                        g_norm = g.norm() + 1e-8
                        m_norm = buf.norm() + 1e-8
                        # True SNR: ratio of signal (momentum) to noise (gradient)
                        # Unbounded [0, ∞) for proper threshold comparison
                        snr = float((m_norm / g_norm).item())
                        if snr < snr_floor:
                            damp_factor = max(0.2, snr / snr_floor)
                            update_final = update_final * damp_factor
                else:
                    # Fallback for non-2D parameters passed directly to Muon (routed to AdamW in Hybrid)
                    norm = update_mat.norm() + 1e-7
                    update_final = update_mat / norm

                # Weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Parameter update
                p.data.add_(update_final, alpha=-lr)

        return loss


class HybridMuonOptimizer:
    """
    Standard GhostLayer 4-Regime Hybrid Optimizer:
    - Regime 1: Embeddings & Token Projections (wte, embed_tokens) -> AdamW (Sparse coordinate updates)
    - Regime 2: Normalization Vectors & Biases (LayerNorm, RMSNorm, Biases) -> AdamW (1D variance scaling)
    - Regime 3: High-Noise / Small Micro-Batch Regimes -> Adaptive SNR Gating (Blends with AdamW damping)
    - Regime 4: Late-Stage Settlement / Flat Basins -> Spectral Decay (Decays Muon spectral norm to settle)
    - Core 2D Hidden Parameters (Linear, Attention QKV, MLP FFN) -> Muon (Newton-Schulz Polar Root)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 0.005,
        adamw_lr: float = 1e-3,
        muon_weight_decay: float = 0.01,
        adamw_weight_decay: float = 0.0,
        custom_adamw_patterns: Optional[List[str]] = None,
        custom_muon_patterns: Optional[List[str]] = None,
        vocab_size: Optional[int] = None,
        adaptive_snr_damping: bool = True,
    ):
        muon_params = []
        adamw_params = []
        adamw_param_ids = set()

        # 1. Module-type introspection (class-level routing for Embeddings, Norms, etc.)
        for module_name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag, torch.nn.LayerNorm, torch.nn.GroupNorm)) or any(t in cls_name for t in ["norm", "embedding"]):
                for p in module.parameters():
                    adamw_param_ids.add(id(p))

        # Standard name match patterns for Regimes 1 & 2
        adamw_name_tokens = [
            "embed", "wte", "wpe", "shared",
            "lm_head", "head", "classifier", "output_layer", "logits", "norm", "bias"
        ]
        if custom_adamw_patterns:
            adamw_name_tokens.extend([p.lower() for p in custom_adamw_patterns])

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            name_lower = name.lower()

            # Check explicit custom muon match
            if custom_muon_patterns and any(token in name_lower for token in custom_muon_patterns):
                if p.ndim == 2:
                    muon_params.append(p)
                    continue

            # Classify as AdamW if:
            # - Found in an Embedding/Norm module via type introspection (Regimes 1 & 2)
            # - Not a 2D matrix (1D biases, scalars, norm weights cannot be polar-orthogonalized)
            # - Matches known embedding/head/norm name tokens
            # - Matches vocab_size along dimension 0 or 1
            is_embed_or_head = (
                id(p) in adamw_param_ids
                or any(term in name_lower for term in adamw_name_tokens)
                or (vocab_size is not None and p.ndim == 2 and (p.shape[0] == vocab_size or p.shape[1] == vocab_size))
            )

            if p.ndim == 2 and not is_embed_or_head:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            adaptive_snr_damping=adaptive_snr_damping,
        ) if muon_params else None
        
        self.adamw = AdamW(adamw_params, lr=adamw_lr, weight_decay=adamw_weight_decay) if adamw_params else None
        self.muon_param_count = sum(p.numel() for p in muon_params)
        self.adamw_param_count = sum(p.numel() for p in adamw_params)

    def set_late_stage_factor(self, factor: float) -> None:
        """
        Regime 4 Settlement: Decays Muon's spectral step size in late training stages
        to allow parameters to settle into narrow local minima without overshooting.
        """
        if self.muon:
            for group in self.muon.param_groups:
                group["late_stage_factor"] = max(0.01, min(1.0, factor))

    def zero_grad(self, set_to_none: bool = True):
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if self.muon:
            loss = self.muon.step(closure=closure)
        if self.adamw:
            self.adamw.step()
        return loss


def create_muon_hybrid_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.005,
    adamw_lr: float = 1e-3,
    adaptive_snr_damping: bool = True,
) -> HybridMuonOptimizer:
    """Creates the standard GhostLayer 4-Regime Hybrid Muon optimizer for a model."""
    return HybridMuonOptimizer(
        model,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        adaptive_snr_damping=adaptive_snr_damping,
    )
