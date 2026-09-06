"""
Generic Webhook Bridge: Sends GhostLayer decision records to any HTTP endpoint
as JSON payloads with configurable headers and retry logic.
"""

import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

try:
    import urllib.request
    import urllib.error
    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    endpoint: str
    payload_type: str
    status_code: int
    success: bool
    attempt: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class WebhookBridge:
    """
    Generic HTTP POST bridge for sending GhostLayer events to any endpoint.
    
    Usage:
        bridge = WebhookBridge(
            endpoint_url="https://api.example.com/ghostlayer/events",
            auth_token="Bearer xxxx",
        )
        bridge.send_recommendation(recommendation)
        bridge.send_decision_record(replay_log.generate_audit_trail())
    """

    def __init__(
        self,
        endpoint_url: str,
        auth_token: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.endpoint_url = endpoint_url
        self.auth_token = auth_token
        self.custom_headers = custom_headers or {}
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.deliveries: List[WebhookDelivery] = []

    def send_recommendation(self, recommendation: Any) -> WebhookDelivery:
        """Send a single recommendation as JSON."""
        payload = {
            "event_type": "recommendation",
            "timestamp": time.time(),
            "data": {
                "rule_id": recommendation.rule_id,
                "title": recommendation.title,
                "impact_level": recommendation.impact_level,
                "confidence": recommendation.confidence,
                "speedup_estimate": recommendation.speedup_estimate_label,
                "description": recommendation.description,
                "safe_to_auto_apply": recommendation.safe_to_auto_apply,
                "evidence": recommendation.evidence,
            },
        }
        return self._send(payload, "recommendation")

    def send_decision_record(self, audit_trail: List[Dict[str, Any]]) -> WebhookDelivery:
        """Send a complete decision record (audit trail) as JSON."""
        payload = {
            "event_type": "decision_record",
            "timestamp": time.time(),
            "data": {"events": audit_trail},
        }
        return self._send(payload, "decision_record")

    def send_verification(self, verification_result: Any) -> WebhookDelivery:
        """Send a verification result as JSON."""
        payload = {
            "event_type": "verification",
            "timestamp": time.time(),
            "data": {
                "is_safe": verification_result.is_safe,
                "max_loss_delta": verification_result.max_loss_delta,
                "relative_mean_loss_shift": verification_result.relative_mean_loss_shift,
                "action_taken": verification_result.action_taken,
                "recommendation_id": verification_result.recommendation_id,
                "reason": verification_result.reason,
            },
        }
        return self._send(payload, "verification")

    def send_custom(self, event_type: str, data: Dict[str, Any]) -> WebhookDelivery:
        """Send a custom event payload."""
        payload = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        return self._send(payload, event_type)

    def _send(self, payload: Dict[str, Any], payload_type: str) -> WebhookDelivery:
        """Send payload with retry logic."""
        if not _HTTP_AVAILABLE:
            delivery = WebhookDelivery(
                endpoint=self.endpoint_url,
                payload_type=payload_type,
                status_code=0,
                success=False,
                attempt=0,
                error="urllib not available",
            )
            self.deliveries.append(delivery)
            return delivery

        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        headers.update(self.custom_headers)

        data = json.dumps(payload).encode("utf-8")

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.endpoint_url,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    delivery = WebhookDelivery(
                        endpoint=self.endpoint_url,
                        payload_type=payload_type,
                        status_code=resp.status,
                        success=(200 <= resp.status < 300),
                        attempt=attempt,
                    )
                    self.deliveries.append(delivery)
                    return delivery
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)  # Exponential backoff
                    continue
                delivery = WebhookDelivery(
                    endpoint=self.endpoint_url,
                    payload_type=payload_type,
                    status_code=0,
                    success=False,
                    attempt=attempt,
                    error=str(e),
                )
                self.deliveries.append(delivery)
                return delivery

        # Shouldn't reach here, but safety fallback
        delivery = WebhookDelivery(
            endpoint=self.endpoint_url,
            payload_type=payload_type,
            status_code=0,
            success=False,
            attempt=self.max_retries,
            error="All retries exhausted",
        )
        self.deliveries.append(delivery)
        return delivery
