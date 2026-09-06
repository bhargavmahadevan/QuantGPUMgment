"""
Tests for extended RuntimeOptimizer functionality:
- DataLoader optimization (pin_memory, persistent_workers, prefetch_factor, auto num_workers)
- ChunkedCrossEntropy optimization and integration
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ghost_layer.runtime_optimizer import RuntimeOptimizer, RuntimeOptimizationResult
from ghost_layer.curvature.loss import ChunkedCrossEntropyLoss


def test_optimize_dataloader_happy_path():
    dataset = TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,)))
    dl = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=False)

    opt = RuntimeOptimizer()
    result = opt.optimize_dataloader(dl, num_workers=2, pin_memory=True, prefetch_factor=2)

    assert result.applied is True
    assert result.optimization_type == "dataloader_config"
    assert result.before_state["num_workers"] == 0
    assert result.before_state["pin_memory"] is False
    assert result.after_state["num_workers"] == 2
    assert result.after_state["pin_memory"] is True
    assert "optimized_dataloader" in result.after_state

    # Verify that the optimized dataloader can iterate cleanly
    opt_dl = result.after_state["optimized_dataloader"]
    batch_count = 0
    for x, y in opt_dl:
        batch_count += 1
        assert x.shape == (4, 8)
    assert batch_count == 8


def test_optimize_dataloader_auto_workers():
    dataset = TensorDataset(torch.randn(16, 4), torch.zeros(16))
    dl = DataLoader(dataset, batch_size=4, num_workers=0)

    opt = RuntimeOptimizer()
    result = opt.optimize_dataloader(dl)

    assert result.applied is True
    assert result.after_state["num_workers"] >= 0
    assert result.after_state["pin_memory"] is True


def test_optimize_dataloader_idempotency():
    dataset = TensorDataset(torch.randn(16, 4), torch.zeros(16))
    dl = DataLoader(dataset, batch_size=4, num_workers=0)

    opt = RuntimeOptimizer()
    res1 = opt.optimize_dataloader(dl, num_workers=2)
    assert res1.applied is True

    # Second apply on the optimized dataloader should be idempotent skip
    opt_dl = res1.after_state["optimized_dataloader"]
    res2 = opt.optimize_dataloader(opt_dl)
    assert res2.applied is False
    assert "Already optimized" in res2.error


def test_optimize_chunked_cross_entropy_basic():
    opt = RuntimeOptimizer()
    loss_fn = nn.CrossEntropyLoss()

    result = opt.optimize_chunked_cross_entropy(loss_fn, chunk_size=32)
    assert result.applied is True
    assert result.optimization_type == "chunked_cross_entropy"
    assert result.before_state["loss_type"] == "CrossEntropyLoss"
    assert result.after_state["loss_type"] == "ChunkedCrossEntropyLoss"
    assert result.after_state["chunk_size"] == 32

    new_loss_fn = result.after_state["optimized_loss"]
    assert isinstance(new_loss_fn, ChunkedCrossEntropyLoss)

    # Test numerical output matches standard CrossEntropyLoss
    torch.manual_seed(42)
    logits = torch.randn(8, 64, requires_grad=True)
    targets = torch.randint(0, 64, (8,))

    std_loss = loss_fn(logits, targets)
    chunked_loss = new_loss_fn(logits, targets)

    assert torch.isclose(std_loss, chunked_loss, atol=1e-5)


def test_optimize_chunked_cross_entropy_idempotency():
    opt = RuntimeOptimizer()
    chunked = ChunkedCrossEntropyLoss(chunk_size=64)

    result = opt.optimize_chunked_cross_entropy(chunked, chunk_size=64)
    assert result.applied is False
    assert "Already chunked" in result.error


def test_apply_all_safe_with_extensions():
    dataset = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    dl = DataLoader(dataset, batch_size=4, num_workers=0)
    model = nn.Linear(4, 2)
    loss_fn = nn.CrossEntropyLoss()

    opt = RuntimeOptimizer()
    results = opt.apply_all_safe(model=model, dataloader=dl, loss_fn=loss_fn)

    opt_types = [r.optimization_type for r in results]
    assert "cuda_memory_config" in opt_types
    assert "cudnn_config" in opt_types
    assert "matmul_precision" in opt_types
    assert "torch_compile" in opt_types
    assert "dataloader_config" in opt_types
    assert "chunked_cross_entropy" in opt_types
