"""
Slack Notifier: Sends GhostLayer decision records, drift alerts,
and rollback notifications to Slack via incoming webhooks.
"""

import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

try:
    import urllib.request
    import urllib.error
    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False


@dataclass
class SlackMessage:
    """A structured Slack message with Block Kit formatting."""
    channel: str
    text: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SlackNotifier:
    """
    Sends GhostLayer events to Slack via incoming webhook.
    
    Usage:
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/...")
        notifier.notify_recommendation(recommendation)
        notifier.notify_drift(drift_report)
        notifier.notify_rollback(rule_id, reason)
    """

    def __init__(self, webhook_url: str, channel: str = "#ghost-layer-alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
        self._send_count = 0

    def notify_recommendation(self, recommendation: Any) -> bool:
        """Send a recommendation alert to Slack."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🔍 GhostLayer Recommendation"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Rule:*\n{recommendation.rule_id}"},
                    {"type": "mrkdwn", "text": f"*Impact:*\n{recommendation.impact_level}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{recommendation.confidence:.0%}"},
                    {"type": "mrkdwn", "text": f"*Speedup Est:*\n{recommendation.speedup_estimate_label}"},
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{recommendation.title}*\n{recommendation.description}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```{recommendation.actionable_code_snippet}```"}
            },
        ]

        if hasattr(recommendation, "financial_impact") and recommendation.financial_impact:
            fin = recommendation.financial_impact
            waste_val = fin.get("modeled_monthly_waste_usd", 0.0)
            payback = fin.get("audit_payback_days", 0.0)
            roi_mult = fin.get("audit_roi_multiple", 0.0)
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"💰 *Modeled Waste:* ${waste_val:,.2f}/mo | ⚡ *Audit Payback:* {payback} days ({roi_mult}x ROI)",
                    }
                ]
            })

        return self._send(
            text=f"GhostLayer: {recommendation.title}",
            blocks=blocks,
        )

    def notify_drift(self, drift_report: Any) -> bool:
        """Send a drift/staleness alert to Slack."""
        emoji = "🔴" if drift_report.is_stale else "🟢"
        reasons_text = "\n".join(f"• {r}" for r in drift_report.reasons) if drift_report.reasons else "No drift detected."

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} GhostLayer Drift Monitor"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Status:* {'STALE — re-optimization warranted' if drift_report.is_stale else 'CURRENT — no action needed'}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": reasons_text}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Recommended Action:*\n{drift_report.recommended_action}"}
            },
        ]

        return self._send(
            text=f"GhostLayer Drift: {'STALE' if drift_report.is_stale else 'CURRENT'}",
            blocks=blocks,
        )

    def notify_rollback(self, rule_id: str, reason: str) -> bool:
        """Send a rollback notification to Slack."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "⚠️ GhostLayer Rollback Executed"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Rule:*\n{rule_id}"},
                    {"type": "mrkdwn", "text": f"*Action:*\nROLLBACK_EXECUTED"},
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Reason:*\n{reason}"}
            },
        ]

        return self._send(
            text=f"GhostLayer Rollback: {rule_id}",
            blocks=blocks,
        )

    def notify_savings(self, savings_report: Any) -> bool:
        """Send a verified savings report to Slack."""
        emoji = "✅" if savings_report.is_verified else "⚠️"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} GhostLayer Verified Savings"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Speedup:*\n{savings_report.speedup_pct:.1f}%"},
                    {"type": "mrkdwn", "text": f"*Verified:*\n{savings_report.verification_label}"},
                    {"type": "mrkdwn", "text": f"*p-value:*\n{savings_report.p_value:.4f}"},
                    {"type": "mrkdwn", "text": f"*$/1000 GPU-hrs:*\n${savings_report.dollar_savings_per_1000_gpu_hours:.2f}"},
                ]
            },
        ]

        return self._send(
            text=f"GhostLayer Savings: {savings_report.verification_label}",
            blocks=blocks,
        )

    def _send(self, text: str, blocks: List[Dict[str, Any]]) -> bool:
        """Send a message to Slack via webhook."""
        if not _HTTP_AVAILABLE:
            return False

        payload = {
            "channel": self.channel,
            "text": text,
            "blocks": blocks,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._send_count += 1
                return resp.status == 200
        except Exception:
            return False
