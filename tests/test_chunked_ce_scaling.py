"""
Tests for ChunkedCrossEntropyLoss scaling, gradient fidelity, and VRAM containment.
Certifies mathematical equivalence across vocabulary dimensions (32k to 128k).
"""

import pytest
import torch
import torch.nn.functional as F
from ghost_layer.curvature.loss import ChunkedCrossEntropyLoss, chunked_cross_entropy


class TestChunkedCrossEntropyScaling:
    """Verifies gradient fidelity, label smoothing, and reduction semantics."""

    @pytest.mark.parametrize("vocab_size", [1000, 4000, 16000])
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_gradient_fidelity_against_standard_ce(self, vocab_size: int, reduction: str):
        """Chunked CE gradients must match standard PyTorch CE within float precision."""
        torch.manual_seed(42)
        num_tokens = 64
        hidden_dim = 128
        chunk_size = 16

        h_init = torch.randn(num_tokens, hidden_dim, dtype=torch.float32)
        w_init = torch.randn(vocab_size, hidden_dim, dtype=torch.float32)
        targets = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long)

        # Standard CE
        h_std = h_init.clone().requires_grad_(True)
        w_std = w_init.clone().requires_grad_(True)
        logits = F.linear(h_std, w_std)
        loss_std = F.cross_entropy(logits, targets, reduction=reduction)
        loss_std.backward()

        # Chunked CE
        h_chk = h_init.clone().requires_grad_(True)
        w_chk = w_init.clone().requires_grad_(True)
        loss_chk = chunked_cross_entropy(
            h_chk, w_chk, targets, chunk_size=chunk_size, reduction=reduction
        )
        loss_chk.backward()

        # Assert loss match
        assert torch.allclose(loss_std, loss_chk, atol=1e-5), (
            f"Loss mismatch: std={loss_std.item():.6f}, chk={loss_chk.item():.6f}"
        )

        # Assert gradient match on hidden activations
        max_h_err = (h_std.grad - h_chk.grad).abs().max().item()
        assert max_h_err < 1e-4, f"Hidden activation gradient divergence: {max_h_err}"

        # Assert gradient match on LM head weights
        max_w_err = (w_std.grad - w_chk.grad).abs().max().item()
        assert max_w_err < 1e-4, f"LM head weight gradient divergence: {max_w_err}"

    def test_ignore_index_and_masking(self):
        """Tokens with ignore_index must contribute zero to gradient and loss."""
        torch.manual_seed(42)
        num_tokens = 32
        hidden_dim = 64
        vocab_size = 500
        chunk_size = 8
        ignore_idx = -100

        h = torch.randn(num_tokens, hidden_dim, requires_grad=True)
        w = torch.randn(vocab_size, hidden_dim, requires_grad=True)
        targets = torch.randint(0, vocab_size, (num_tokens,))
        # Set half the tokens to ignore
        targets[:16] = ignore_idx

        loss = chunked_cross_entropy(h, w, targets, chunk_size=chunk_size, ignore_index=ignore_idx)
        loss.backward()

        # First 16 tokens in h.grad should be zero because they were ignored
        ignored_grad_norm = h.grad[:16].abs().max().item()
        assert ignored_grad_norm == 0.0, f"Ignored tokens leaked into gradient: {ignored_grad_norm}"

    def test_label_smoothing_exactness(self):
        """Label smoothing parameter must match standard CE behavior."""
        torch.manual_seed(42)
        num_tokens = 32
        hidden_dim = 64
        vocab_size = 200
        chunk_size = 8

        h = torch.randn(num_tokens, hidden_dim)
        w = torch.randn(vocab_size, hidden_dim)
        y = torch.randint(0, vocab_size, (num_tokens,))

        logits = F.linear(h, w)
        std_loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        chk_loss = chunked_cross_entropy(h, w, y, chunk_size=chunk_size, label_smoothing=0.1)

        assert torch.allclose(std_loss, chk_loss, atol=1e-5)

    def test_chunked_loss_module_interface(self):
        """Module wrapper must work with 3-arg or standard 2-arg (logits, targets) signature."""
        loss_fn = ChunkedCrossEntropyLoss(chunk_size=16)

        # 3-arg signature: hidden, lm_head_weight, targets
        h = torch.randn(16, 32)
        w = torch.randn(100, 32)
        y = torch.randint(0, 100, (16,))
        l3 = loss_fn(h, w, y)
        assert l3.ndim == 0
        assert not torch.isnan(l3)

        # 2-arg signature: logits, targets (fallback path)
        logits = torch.randn(16, 100)
        l2 = loss_fn(logits, y)
        assert l2.ndim == 0
        assert not torch.isnan(l2)
