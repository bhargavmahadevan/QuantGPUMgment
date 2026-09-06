"""
Unit tests for Section H: Multi-Scale Gradient Spherical Variance Monitor.
"""

import math
import os
import tempfile
import torch
import torch.nn as nn
import pytest

from ghost_layer.telemetry.spherical_monitor import (
    SphericalGradientMonitor,
    SphericalGradientPoint,
    SphericalVarianceReport,
)


def test_spherical_gradient_coordinate_conversion():
    """Verify conversion of known 3D vectors to radius, polar angle, and azimuth."""
    monitor = SphericalGradientMonitor()
    
    # Vector along +Z axis: r=1, phi=0 (north pole), theta=0
    vec_z = torch.tensor([0.0, 0.0, 2.0])
    pt_z = monitor.record_step(step=1, model_or_grads=vec_z)
    assert math.isclose(pt_z.radius, 2.0, rel_tol=1e-3)
    
    # Vector in XY plane along +X axis: r=3, phi=pi/2 (equator), theta=0
    vec_x = torch.tensor([3.0, 0.0, 0.0])
    pt_x = monitor.record_step(step=2, model_or_grads=vec_x)
    assert math.isclose(pt_x.radius, 3.0, rel_tol=1e-3)


def test_spherical_gradient_outlier_detection():
    """Verify outlier detection flags 3-sigma radius excursions."""
    torch.manual_seed(42)
    monitor = SphericalGradientMonitor(outlier_sigma_threshold=3.0, warmup_steps=10)
    
    # Record 20 stationary steps
    for step in range(1, 21):
        g = torch.randn(64) * 0.5
        monitor.record_step(step=step, model_or_grads=g)
        
    # Step 21: Inject severe gradient spike
    spike_grad = torch.randn(64) * 10.0 + 20.0
    pt_spike = monitor.record_step(step=21, model_or_grads=spike_grad)
    
    assert pt_spike.is_outlier is True
    
    rep = monitor.compute_report()
    assert rep.outlier_count >= 1
    assert 21 in rep.outlier_steps


def test_spherical_gradient_with_model():
    """Verify SphericalGradientMonitor operates directly on nn.Module with backward pass."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    monitor = SphericalGradientMonitor(warmup_steps=5)
    
    loss_fn = nn.MSELoss()
    x = torch.randn(8, 16)
    y = torch.randn(8, 1)
    
    loss = loss_fn(model(x), y)
    loss.backward()
    
    pt = monitor.record_step(step=1, model_or_grads=model)
    assert pt.radius > 0.0
    assert 0.0 <= pt.polar_phi <= math.pi
    assert -math.pi <= pt.azimuth_theta <= math.pi


def test_spherical_gradient_chart_rendering():
    """Verify render_variance_sphere_chart produces a valid non-empty PNG file."""
    torch.manual_seed(42)
    monitor = SphericalGradientMonitor(warmup_steps=5)
    for step in range(1, 15):
        monitor.record_step(step=step, model_or_grads=torch.randn(32))
        
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        out = monitor.render_variance_sphere_chart(tmp_path)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000  # valid image file
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
