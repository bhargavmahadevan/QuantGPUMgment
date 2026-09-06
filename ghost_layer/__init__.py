"""
Ghost Layer: Autonomous AI Training Efficiency Optimization & Decision Layer.
"""

from ghost_layer.telemetry.watcher import TelemetryWatcher, MetricSnapshot, TelemetrySummary
from ghost_layer.telemetry.tensor_auditor import TensorFlowAuditor, audit_gpu_vitality, TensorFlowDiagnostic
from ghost_layer.decision.engine import DecisionEngine, Recommendation
from ghost_layer.verification.verifier import CorrectnessVerifier, VerificationResult
from ghost_layer.verification.lagrangian import LagrangianErrorController, LagrangianStepReport
from ghost_layer.roi.calculator import ROICalculator, ROIAuditReport
from ghost_layer.kb.knowledge_base import SharedKnowledgeBase, TelemetryBoundary
from ghost_layer.hooks import GhostWatcherHook
from ghost_layer.replay import OptimizationReplayLog, ReplayEvent
from ghost_layer.applier import AutoApplier, ConsentLevel, ApplyResult
from ghost_layer.rollback import RollbackManager, ConfigSnapshot, RollbackResult
from ghost_layer.savings import SavingsVerifier, BaselineMeasurement, VerifiedSavingsReport
from ghost_layer.runtime_optimizer import RuntimeOptimizer
from ghost_layer.curvature import (
    compute_hvp_pearlmutter,
    estimate_local_curvature,
    CurvatureDiagnostic,
    SophiaG,
    Muon,
    HybridMuonOptimizer,
    create_muon_hybrid_optimizer,
    Shampoo,
    ChunkedCrossEntropyLoss,
    chunked_cross_entropy,
    mHCResidual,
    sinkhorn_knopp_doubly_stochastic,
    birkhoff_drift_metric,
)
from ghost_layer.distributed import (
    CollectiveTelemetry,
    FSDPGhostHook,
    DeepSpeedGhostHook,
)
from ghost_layer.integrations import (
    SlackNotifier,
    WebhookBridge,
    WandbExporter,
    PrometheusExporter,
)

__version__ = "0.2.0"
__all__ = [
    # Core Infrastructure
    "TelemetryWatcher",
    "MetricSnapshot",
    "TelemetrySummary",
    "TensorFlowAuditor",
    "audit_gpu_vitality",
    "TensorFlowDiagnostic",
    "DecisionEngine",
    "Recommendation",
    "CorrectnessVerifier",
    "VerificationResult",
    "LagrangianErrorController",
    "LagrangianStepReport",
    "ROICalculator",
    "ROIAuditReport",
    "SharedKnowledgeBase",
    "TelemetryBoundary",
    "GhostWatcherHook",
    "OptimizationReplayLog",
    "ReplayEvent",
    # Closed-Loop Execution & Control
    "AutoApplier",
    "ConsentLevel",
    "ApplyResult",
    "RollbackManager",
    "ConfigSnapshot",
    "RollbackResult",
    "SavingsVerifier",
    "BaselineMeasurement",
    "VerifiedSavingsReport",
    "RuntimeOptimizer",
    # Curvature & Low-Data Optimizers
    "compute_hvp_pearlmutter",
    "estimate_local_curvature",
    "CurvatureDiagnostic",
    "SophiaG",
    "Muon",
    "HybridMuonOptimizer",
    "create_muon_hybrid_optimizer",
    "Shampoo",
    "ChunkedCrossEntropyLoss",
    "chunked_cross_entropy",
    "mHCResidual",
    "sinkhorn_knopp_doubly_stochastic",
    "birkhoff_drift_metric",
    # Distributed Hooks
    "CollectiveTelemetry",
    "FSDPGhostHook",
    "DeepSpeedGhostHook",
    # Integrations
    "SlackNotifier",
    "WebhookBridge",
    "WandbExporter",
    "PrometheusExporter",
]
