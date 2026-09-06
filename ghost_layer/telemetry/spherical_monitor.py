"""
Multi-Scale Gradient Spherical Variance Monitor for GhostLayer (Section H).

Projects high-dimensional parameter gradient trajectories into 3D spherical coordinates:
    r_t = ||g_t||_2
    phi_t = arccos(g_{z,t} / r_t)  (Polar inclination [0, pi])
    theta_t = arctan2(g_{y,t}, g_{x,t}) (Azimuthal angle [-pi, pi])

Tracks running Welford spherical statistics (mu_r, sigma_r) and flags anomalous gradient
outlier excursions (r_t > mu_r + 3 * sigma_r) that signal loss spikes or optimizer divergence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Sequence
import math
import torch
import torch.nn as nn


@dataclass
class SphericalGradientPoint:
    """A single gradient step projected into spherical coordinates."""
    step: int
    radius: float          # r = ||g||_2 (gradient norm)
    polar_phi: float       # phi in [0, pi] (inclination)
    azimuth_theta: float   # theta in [-pi, pi] (azimuth)
    gx: float              # 3D projected x
    gy: float              # 3D projected y
    gz: float              # 3D projected z
    is_outlier: bool       # r > mu_r + 3 * sigma_r


@dataclass
class SphericalVarianceReport:
    """Statistical summary of gradient spherical variance and outlier spikes."""
    total_steps: int
    mean_radius: float
    std_radius: float
    max_radius: float
    outlier_count: int
    outlier_steps: List[int]
    angular_variance_theta: float
    angular_variance_phi: float
    is_trajectory_stable: bool
    points: List[SphericalGradientPoint]


class SphericalGradientMonitor:
    """
    Continuous Multi-Scale Spherical Gradient Variance Tracker (Section H).
    """

    def __init__(
        self,
        outlier_sigma_threshold: float = 3.0,
        warmup_steps: int = 10,
        projection_seed: int = 42,
    ):
        self.sigma_threshold = outlier_sigma_threshold
        self.warmup_steps = warmup_steps
        self.projection_seed = projection_seed

        # Welford running statistics for radius r
        self.count = 0
        self.mean_r = 0.0
        self.M2_r = 0.0

        # Welford for angular coordinates
        self.mean_theta = 0.0
        self.M2_theta = 0.0
        self.mean_phi = 0.0
        self.M2_phi = 0.0

        self.history: List[SphericalGradientPoint] = []
        self._projection_matrix: Optional[torch.Tensor] = None

    def _get_3d_projection(self, grad_flat: torch.Tensor) -> Tuple[float, float, float]:
        """Projects high-dimensional flat gradient vector into deterministic 3D space."""
        d = grad_flat.numel()
        if d < 3:
            gx = float(grad_flat[0].item()) if d > 0 else 0.0
            gy = float(grad_flat[1].item()) if d > 1 else 0.0
            gz = float(grad_flat[2].item()) if d > 2 else 0.0
            return gx, gy, gz

        # Deterministic orthonormal projection matrix for consistency
        if self._projection_matrix is None or self._projection_matrix.shape[1] != d or self._projection_matrix.device != grad_flat.device:
            gen = torch.Generator(device="cpu").manual_seed(self.projection_seed)
            proj = torch.randn(3, d, generator=gen, dtype=grad_flat.dtype)
            # Orthonormalize rows via QR
            proj_q, _ = torch.linalg.qr(proj.T)
            self._projection_matrix = proj_q[:, :3].T.to(grad_flat.device)

        proj_3d = self._projection_matrix @ grad_flat
        return float(proj_3d[0].item()), float(proj_3d[1].item()), float(proj_3d[2].item())

    def record_step(
        self,
        step: int,
        model_or_grads: Any,
    ) -> SphericalGradientPoint:
        """
        Extracts gradients, projects to spherical coordinates, and updates running statistics.
        """
        # Collect flat gradient tensor
        if isinstance(model_or_grads, nn.Module):
            grad_list = [p.grad.flatten() for p in model_or_grads.parameters() if p.grad is not None]
            if not grad_list:
                grad_flat = torch.zeros(3)
            else:
                grad_flat = torch.cat(grad_list)
        elif isinstance(model_or_grads, (list, tuple)):
            grad_list = [g.flatten() for g in model_or_grads if g is not None]
            grad_flat = torch.cat(grad_list) if grad_list else torch.zeros(3)
        elif isinstance(model_or_grads, torch.Tensor):
            grad_flat = model_or_grads.flatten()
        else:
            grad_flat = torch.zeros(3)

        # 1. Total L2 norm (spherical radius r)
        r = float(torch.norm(grad_flat).item())

        # 2. 3D projection for polar/azimuthal angles
        gx, gy, gz = self._get_3d_projection(grad_flat)
        r_3d = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-10

        # Polar angle phi in [0, pi]
        cos_phi = max(-1.0, min(1.0, gz / r_3d))
        phi = math.acos(cos_phi)

        # Azimuthal angle theta in [-pi, pi]
        theta = math.atan2(gy, gx)

        # 3. Check outlier status against running statistics
        std_r = math.sqrt(self.M2_r / (self.count - 1)) if self.count > 1 else 0.0
        is_outlier = (self.count >= self.warmup_steps) and (r > self.mean_r + self.sigma_threshold * std_r)

        # 4. Update running Welford statistics
        self.count += 1
        delta_r = r - self.mean_r
        self.mean_r += delta_r / self.count
        self.M2_r += delta_r * (r - self.mean_r)

        delta_theta = theta - self.mean_theta
        self.mean_theta += delta_theta / self.count
        self.M2_theta += delta_theta * (theta - self.mean_theta)

        delta_phi = phi - self.mean_phi
        self.mean_phi += delta_phi / self.count
        self.M2_phi += delta_phi * (phi - self.mean_phi)

        point = SphericalGradientPoint(
            step=step,
            radius=round(r, 6),
            polar_phi=round(phi, 6),
            azimuth_theta=round(theta, 6),
            gx=round(gx, 6),
            gy=round(gy, 6),
            gz=round(gz, 6),
            is_outlier=is_outlier,
        )
        self.history.append(point)
        return point

    def compute_report(self) -> SphericalVarianceReport:
        """Computes summary diagnostic report over all recorded steps."""
        if not self.history:
            return SphericalVarianceReport(
                total_steps=0,
                mean_radius=0.0,
                std_radius=0.0,
                max_radius=0.0,
                outlier_count=0,
                outlier_steps=[],
                angular_variance_theta=0.0,
                angular_variance_phi=0.0,
                is_trajectory_stable=True,
                points=[],
            )

        std_r = math.sqrt(self.M2_r / (self.count - 1)) if self.count > 1 else 0.0
        var_theta = (self.M2_theta / (self.count - 1)) if self.count > 1 else 0.0
        var_phi = (self.M2_phi / (self.count - 1)) if self.count > 1 else 0.0
        
        outliers = [p.step for p in self.history if p.is_outlier]
        max_r = max(p.radius for p in self.history)
        
        # Trajectory is stable if outlier count <= 5% of total steps
        outlier_pct = len(outliers) / len(self.history)
        is_stable = (outlier_pct <= 0.05)

        return SphericalVarianceReport(
            total_steps=len(self.history),
            mean_radius=round(self.mean_r, 6),
            std_radius=round(std_r, 6),
            max_radius=round(max_r, 6),
            outlier_count=len(outliers),
            outlier_steps=outliers,
            angular_variance_theta=round(var_theta, 6),
            angular_variance_phi=round(var_phi, 6),
            is_trajectory_stable=is_stable,
            points=self.history,
        )

    def render_variance_sphere_chart(self, output_path: str) -> str:
        """
        Renders multi-panel spherical gradient telemetry plot and saves to output_path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        rep = self.compute_report()
        fig = plt.figure(figsize=(16, 6), facecolor="#0B0F19")
        
        # Panel 1: 3D Spherical Coordinate Space
        ax1 = fig.add_subplot(131, projection='3d', facecolor="#0B0F19")
        ax1.set_title("3D Projected Gradient Manifold", color="#F3F4F6", fontsize=11, fontweight="bold", pad=12)
        
        # Wireframe 3-sigma sphere
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 24)
        sphere_r = max(0.1, rep.mean_radius + 3.0 * rep.std_radius)
        x_sphere = sphere_r * np.outer(np.cos(u), np.sin(v))
        y_sphere = sphere_r * np.outer(np.sin(u), np.sin(v))
        z_sphere = sphere_r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, color="#3B82F6", alpha=0.15, linewidth=0.6)

        # Plot normal vs outlier points
        normal_pts = [p for p in self.history if not p.is_outlier]
        outlier_pts = [p for p in self.history if p.is_outlier]
        
        if normal_pts:
            ax1.scatter(
                [p.gx for p in normal_pts],
                [p.gy for p in normal_pts],
                [p.gz for p in normal_pts],
                c="#10B981", s=15, alpha=0.7, label=f"Normal Trajectory (n={len(normal_pts)})"
            )
        if outlier_pts:
            ax1.scatter(
                [p.gx for p in outlier_pts],
                [p.gy for p in outlier_pts],
                [p.gz for p in outlier_pts],
                c="#EF4444", s=50, marker="^", edgecolors="#FFFFFF", label=f"3σ Outlier Spike (n={len(outlier_pts)})"
            )

        ax1.tick_params(colors="#9CA3AF", labelsize=8)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.legend(loc="upper right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)

        # Panel 2: Radial Norm Trajectory vs 3-Sigma Anomaly Boundary
        ax2 = fig.add_subplot(132, facecolor="#111827")
        ax2.set_title(r"Gradient Norm $r_t = ||g_t||_2$ vs. $3\sigma_r$ Bound", color="#F3F4F6", fontsize=11, fontweight="bold", pad=12)
        
        steps = [p.step for p in self.history]
        radii = [p.radius for p in self.history]
        ax2.plot(steps, radii, color="#60A5FA", linewidth=1.5, label=r"Gradient L2 Norm $r_t$")
        ax2.axhline(rep.mean_radius, color="#9CA3AF", linestyle="--", linewidth=1.0, label=rf"Mean $\mu_r={rep.mean_radius:.3f}$")
        ax2.axhline(sphere_r, color="#EF4444", linestyle=":", linewidth=1.4, label=rf"$\mu_r + 3\sigma_r={sphere_r:.3f}$ Boundary")
        
        if outlier_pts:
            ax2.scatter([p.step for p in outlier_pts], [p.radius for p in outlier_pts], c="#EF4444", s=40, zorder=5)

        ax2.set_xlabel("Training Step", color="#9CA3AF", fontsize=9)
        ax2.set_ylabel(r"Gradient Norm $r_t$", color="#9CA3AF", fontsize=9)
        ax2.tick_params(colors="#9CA3AF", labelsize=8)
        ax2.grid(True, color="#374151", alpha=0.4, linestyle="--")
        ax2.legend(loc="upper right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)

        # Panel 3: Angular Phase Dispersion (Theta vs Phi)
        ax3 = fig.add_subplot(133, facecolor="#111827")
        ax3.set_title(r"Angular Direction Distribution $(\theta_t, \phi_t)$", color="#F3F4F6", fontsize=11, fontweight="bold", pad=12)
        
        thetas = [p.azimuth_theta for p in self.history]
        phis = [p.polar_phi for p in self.history]
        sc = ax3.scatter(thetas, phis, c=radii, cmap="viridis", s=20, alpha=0.8)
        cbar = plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label(r"Radius $r_t$", color="#9CA3AF", fontsize=8)
        cbar.ax.tick_params(labelsize=8, colors="#9CA3AF")

        ax3.set_xlabel(r"Azimuth $\theta_t \in [-\pi, \pi]$", color="#9CA3AF", fontsize=9)
        ax3.set_ylabel(r"Polar Inclination $\phi_t \in [0, \pi]$", color="#9CA3AF", fontsize=9)
        ax3.tick_params(colors="#9CA3AF", labelsize=8)
        ax3.grid(True, color="#374151", alpha=0.4, linestyle="--")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path
