"""
Prometheus Exporter: Exposes GhostLayer metrics as Prometheus-compatible
gauges and counters for integration with existing monitoring infrastructure.

Uses a simple HTTP server to serve /metrics endpoint in Prometheus text format.
No external dependency on prometheus_client — implemented from scratch.
"""

import time
import threading
from typing import Dict, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler


class MetricStore:
    """Thread-safe in-memory store for Prometheus-format metrics."""

    def __init__(self):
        self._gauges: Dict[str, float] = {}
        self._counters: Dict[str, float] = {}
        self._labels: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric to a specific value."""
        with self._lock:
            key = self._label_key(name, labels)
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._label_key(name, labels)
            self._counters[key] = self._counters.get(key, 0.0) + value
            if labels:
                self._labels[key] = labels

    def format_metrics(self) -> str:
        """Format all metrics in Prometheus text exposition format."""
        lines = []
        with self._lock:
            for key, value in sorted(self._gauges.items()):
                lines.append(f"{key} {value}")
            for key, value in sorted(self._counters.items()):
                lines.append(f"{key} {value}")
        return "\n".join(lines) + "\n"

    def _label_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate a Prometheus-format metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics endpoint."""
    store: Optional[MetricStore] = None

    def do_GET(self):
        if self.path == "/metrics":
            content = self.store.format_metrics() if self.store else ""
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


class PrometheusExporter:
    """
    Exposes GhostLayer metrics as Prometheus gauges and counters.
    
    Usage:
        exporter = PrometheusExporter(port=9090)
        exporter.start()
        
        # During training:
        exporter.record_step(step_time_ms=25.0, gpu_util_pct=85.0, memory_mb=12000.0)
        exporter.record_recommendation("RULE_MIXED_PRECISION")
        
        # Scrape at http://localhost:9090/metrics
        
        exporter.stop()
    """

    def __init__(self, port: int = 9090, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.store = MetricStore()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the Prometheus metrics HTTP server in a background thread."""
        if self._running:
            return

        handler_class = type(
            "_Handler",
            (_MetricsHandler,),
            {"store": self.store},
        )
        self._server = HTTPServer((self.host, self.port), handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self):
        """Stop the metrics server."""
        if self._server and self._running:
            self._server.shutdown()
            self._running = False

    def record_step(
        self,
        step_time_ms: float,
        gpu_util_pct: float,
        memory_mb: float,
        loss: Optional[float] = None,
    ):
        """Record step-level metrics."""
        self.store.set_gauge("ghostlayer_step_time_ms", step_time_ms)
        self.store.set_gauge("ghostlayer_gpu_utilization_pct", gpu_util_pct)
        self.store.set_gauge("ghostlayer_gpu_memory_mb", memory_mb)
        if loss is not None:
            self.store.set_gauge("ghostlayer_training_loss", loss)
        self.store.increment_counter("ghostlayer_steps_total")

    def record_recommendation(self, rule_id: str):
        """Increment recommendation counter."""
        self.store.increment_counter(
            "ghostlayer_recommendations_total",
            labels={"rule_id": rule_id},
        )

    def record_rollback(self, rule_id: str):
        """Increment rollback counter."""
        self.store.increment_counter(
            "ghostlayer_rollbacks_total",
            labels={"rule_id": rule_id},
        )

    def record_apply(self, rule_id: str, success: bool):
        """Record an apply event."""
        status = "success" if success else "failure"
        self.store.increment_counter(
            "ghostlayer_applies_total",
            labels={"rule_id": rule_id, "status": status},
        )

    def record_savings(self, speedup_pct: float, dollar_savings: float):
        """Record verified savings metrics."""
        self.store.set_gauge("ghostlayer_verified_speedup_pct", speedup_pct)
        self.store.set_gauge("ghostlayer_verified_savings_usd", dollar_savings)

    def record_finops(
        self,
        cluster_hourly_spend_usd: float,
        wasted_hourly_spend_usd: float,
        recovered_monthly_savings_usd: float,
        audit_roi_multiplier: float,
        tier_level: int = 1,
    ):
        """Record continuous FinOps and profit metrics for Grafana/Prometheus dashboards."""
        self.store.set_gauge("ghostlayer_cluster_hourly_spend_usd", float(cluster_hourly_spend_usd))
        self.store.set_gauge("ghostlayer_wasted_hourly_spend_usd", float(wasted_hourly_spend_usd))
        self.store.set_gauge("ghostlayer_recovered_monthly_savings_usd", float(recovered_monthly_savings_usd))
        self.store.set_gauge("ghostlayer_audit_roi_multiplier", float(audit_roi_multiplier))
        self.store.set_gauge("ghostlayer_commercial_tier_level", float(tier_level))
