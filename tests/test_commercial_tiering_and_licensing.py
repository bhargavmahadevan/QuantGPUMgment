import pytest
import time
from ghost_layer.commercial.tier import (
    GhostTier,
    CommercialFeature,
    LicenseManager,
    LicenseToken,
    LicenseValidationError,
)


def test_ghost_tier_hierarchy():
    assert GhostTier.COMMUNITY.value == "COMMUNITY"
    assert GhostTier.TEAM_SAAS.value == "TEAM_SAAS"
    assert GhostTier.ENTERPRISE.value == "ENTERPRISE"
    assert GhostTier.ENTERPRISE.tier_level > GhostTier.TEAM_SAAS.tier_level > GhostTier.COMMUNITY.tier_level


def test_default_license_manager_is_community():
    mgr = LicenseManager()
    assert mgr.current_tier == GhostTier.COMMUNITY
    assert mgr.is_feature_enabled(CommercialFeature.LOCAL_TELEMETRY) is True
    assert mgr.is_feature_enabled(CommercialFeature.BASIC_RULES_CHECKLIST) is True
    assert mgr.is_feature_enabled(CommercialFeature.SLACK_ALERTS) is False
    assert mgr.is_feature_enabled(CommercialFeature.PROMETHEUS_FINOPS) is False
    assert mgr.is_feature_enabled(CommercialFeature.AIR_GAPPED_VPC) is False


def test_issue_and_verify_valid_team_token():
    secret = "test-enterprise-master-secret-key-42"
    token_str = LicenseManager.issue_token(
        tenant_id="acme-ai-corp",
        tier=GhostTier.TEAM_SAAS,
        max_gpus=32,
        validity_days=30,
        secret_key=secret,
    )
    assert isinstance(token_str, str)
    assert len(token_str) > 20

    mgr = LicenseManager(license_token=token_str, secret_key=secret)
    assert mgr.current_tier == GhostTier.TEAM_SAAS
    assert mgr.tenant_id == "acme-ai-corp"
    assert mgr.max_gpus == 32
    assert mgr.is_feature_enabled(CommercialFeature.SLACK_ALERTS) is True
    assert mgr.is_feature_enabled(CommercialFeature.PROMETHEUS_FINOPS) is True
    assert mgr.is_feature_enabled(CommercialFeature.AIR_GAPPED_VPC) is False


def test_issue_and_verify_enterprise_token():
    secret = "test-enterprise-master-secret-key-42"
    token_str = LicenseManager.issue_token(
        tenant_id="quant-trading-desk-alpha",
        tier=GhostTier.ENTERPRISE,
        max_gpus=1024,
        validity_days=365,
        secret_key=secret,
    )

    mgr = LicenseManager(license_token=token_str, secret_key=secret)
    assert mgr.current_tier == GhostTier.ENTERPRISE
    assert mgr.is_feature_enabled(CommercialFeature.AIR_GAPPED_VPC) is True
    assert mgr.is_feature_enabled(CommercialFeature.CUSTOM_RULES) is True
    assert mgr.is_feature_enabled(CommercialFeature.SLA_ASSURANCE) is True
    assert mgr.is_gpu_quota_sufficient(512) is True
    assert mgr.is_gpu_quota_sufficient(2048) is False


def test_tampered_token_rejected():
    secret = "test-enterprise-master-secret-key-42"
    token_str = LicenseManager.issue_token(
        tenant_id="victim-corp",
        tier=GhostTier.TEAM_SAAS,
        max_gpus=16,
        validity_days=30,
        secret_key=secret,
    )
    # Tamper with token payload
    tampered = token_str[:-4] + ("ABCD" if not token_str.endswith("ABCD") else "WXYZ")
    with pytest.raises(LicenseValidationError):
        LicenseManager(license_token=tampered, secret_key=secret)


def test_expired_token_rejected():
    secret = "test-enterprise-master-secret-key-42"
    # Negative validity days => already expired
    token_str = LicenseManager.issue_token(
        tenant_id="expired-org",
        tier=GhostTier.ENTERPRISE,
        max_gpus=64,
        validity_days=-1,
        secret_key=secret,
    )
    with pytest.raises(LicenseValidationError, match="expired"):
        LicenseManager(license_token=token_str, secret_key=secret)
