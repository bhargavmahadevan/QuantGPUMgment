"""
GhostLayer Commercial Tiering & Air-Gapped Licensing Engine.
Provides offline, cryptographic license validation for Community, Team SaaS,
and Enterprise Compute Assurance tiers with Zero Data Exfiltration compliance.
"""

import hmac
import hashlib
import json
import base64
import time
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


class GhostTier(str, Enum):
    COMMUNITY = "COMMUNITY"
    TEAM_SAAS = "TEAM_SAAS"
    ENTERPRISE = "ENTERPRISE"

    @property
    def tier_level(self) -> int:
        levels = {
            GhostTier.COMMUNITY: 0,
            GhostTier.TEAM_SAAS: 1,
            GhostTier.ENTERPRISE: 2,
        }
        return levels.get(self, 0)


class CommercialFeature(str, Enum):
    LOCAL_TELEMETRY = "LOCAL_TELEMETRY"
    BASIC_RULES_CHECKLIST = "BASIC_RULES_CHECKLIST"
    SLACK_ALERTS = "SLACK_ALERTS"
    PROMETHEUS_FINOPS = "PROMETHEUS_FINOPS"
    DECISION_REPLAY_HISTORY = "DECISION_REPLAY_HISTORY"
    AIR_GAPPED_VPC = "AIR_GAPPED_VPC"
    CUSTOM_RULES = "CUSTOM_RULES"
    SLA_ASSURANCE = "SLA_ASSURANCE"
    CRYPTOGRAPHIC_BASELINELOCK = "CRYPTOGRAPHIC_BASELINELOCK"


FEATURE_MINIMUM_TIERS: Dict[CommercialFeature, GhostTier] = {
    CommercialFeature.LOCAL_TELEMETRY: GhostTier.COMMUNITY,
    CommercialFeature.BASIC_RULES_CHECKLIST: GhostTier.COMMUNITY,
    CommercialFeature.SLACK_ALERTS: GhostTier.TEAM_SAAS,
    CommercialFeature.PROMETHEUS_FINOPS: GhostTier.TEAM_SAAS,
    CommercialFeature.DECISION_REPLAY_HISTORY: GhostTier.TEAM_SAAS,
    CommercialFeature.AIR_GAPPED_VPC: GhostTier.ENTERPRISE,
    CommercialFeature.CUSTOM_RULES: GhostTier.ENTERPRISE,
    CommercialFeature.SLA_ASSURANCE: GhostTier.ENTERPRISE,
    CommercialFeature.CRYPTOGRAPHIC_BASELINELOCK: GhostTier.ENTERPRISE,
}


class LicenseValidationError(Exception):
    """Raised when an air-gapped license token fails cryptographic or expiration validation."""
    pass


@dataclass
class LicenseToken:
    tenant_id: str
    tier: GhostTier
    max_gpus: int
    issued_at: float
    expires_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "tier": self.tier.value,
            "max_gpus": self.max_gpus,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


class LicenseManager:
    """
    Validates commercial license tokens 100% offline.
    Never transmits telemetry, tokens, or identifiers outside the customer VPC.
    """

    DEFAULT_SECRET = "ghostlayer-airgapped-internal-key-2026"

    def __init__(
        self,
        license_token: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.secret_key = secret_key or self.DEFAULT_SECRET
        if license_token:
            self._payload = self._verify_token(license_token, self.secret_key)
        else:
            self._payload = LicenseToken(
                tenant_id="community-user",
                tier=GhostTier.COMMUNITY,
                max_gpus=8,
                issued_at=time.time(),
                expires_at=time.time() + (3650 * 86400),  # 10 years
            )

    @property
    def current_tier(self) -> GhostTier:
        return self._payload.tier

    @property
    def tenant_id(self) -> str:
        return self._payload.tenant_id

    @property
    def max_gpus(self) -> int:
        return self._payload.max_gpus

    def is_feature_enabled(self, feature: CommercialFeature) -> bool:
        min_tier = FEATURE_MINIMUM_TIERS.get(feature, GhostTier.COMMUNITY)
        return self.current_tier.tier_level >= min_tier.tier_level

    def is_gpu_quota_sufficient(self, num_gpus: int) -> bool:
        return num_gpus <= self.max_gpus

    @classmethod
    def issue_token(
        cls,
        tenant_id: str,
        tier: GhostTier,
        max_gpus: int = 64,
        validity_days: int = 365,
        secret_key: Optional[str] = None,
    ) -> str:
        """Issue an offline, signed HMAC-SHA256 license token."""
        secret = secret_key or cls.DEFAULT_SECRET
        now = time.time()
        expires = now + (validity_days * 86400.0)
        payload = {
            "tenant_id": tenant_id,
            "tier": tier.value,
            "max_gpus": max_gpus,
            "issued_at": now,
            "expires_at": expires,
        }
        json_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        b64_payload = base64.urlsafe_b64encode(json_bytes).decode("utf-8")

        sig = hmac.new(secret.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{b64_payload}.{sig}"

    @classmethod
    def _verify_token(cls, token_str: str, secret_key: str) -> LicenseToken:
        parts = token_str.strip().split(".")
        if len(parts) != 2:
            raise LicenseValidationError("Invalid license token structure.")

        b64_payload, signature = parts[0], parts[1]
        expected_sig = hmac.new(secret_key.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            raise LicenseValidationError("Cryptographic signature mismatch. Token is invalid or tampered.")

        try:
            json_bytes = base64.urlsafe_b64decode(b64_payload.encode("utf-8"))
            data = json.loads(json_bytes.decode("utf-8"))
        except Exception as e:
            raise LicenseValidationError(f"Corrupt token payload: {e}")

        now = time.time()
        if data.get("expires_at", 0) < now:
            raise LicenseValidationError("License token has expired. Please renew Enterprise Compute Assurance.")

        return LicenseToken(
            tenant_id=data.get("tenant_id", "unknown"),
            tier=GhostTier(data.get("tier", "COMMUNITY")),
            max_gpus=int(data.get("max_gpus", 8)),
            issued_at=float(data.get("issued_at", now)),
            expires_at=float(data.get("expires_at", now)),
        )
