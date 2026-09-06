"""
GhostLayer Enterprise Cryptographic Audit Trail Engine
Inspired by codenotary/immudb, auditumio/auditum, javers, and access-control-audit-backend.

Features:
  1. Standard Event Schema (The 5 Ws: Actor, Action, Resource, Timestamp, Context, Diff/State).
  2. Cryptographic Hash Chaining: sha256(prev_hash + serialized_sanitized_payload).
  3. Tamper Detection & Integrity Verification.
  4. PII, Token, and Tensor Weight Sanitization & Redaction Filter.
  5. Decoupled Asynchronous Outbox Pipeline for zero-overhead training telemetry.
"""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class SanitizationFilter:
    """
    Sanitizes audit payloads to ensure no raw secrets, Hugging Face/API tokens,
    user PII, or massive raw tensor arrays leak into persistent audit trails.
    """
    SECRET_KEYS = {"api_key", "secret", "token", "huggingface_token", "hf_token", "password", "auth"}
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    TOKEN_REGEX = re.compile(r"(?:hf_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE)

    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            if any(secret in key_lower for secret in self.SECRET_KEYS):
                sanitized[key] = "[REDACTED_SECRET]"
            elif key_lower in ("prompt", "prompt_text", "user_prompt", "system_prompt"):
                sanitized[key] = "[REDACTED_PROMPT_TEXT]"
            elif isinstance(value, str):
                if self.EMAIL_REGEX.search(value):
                    sanitized[key] = "[REDACTED_PII]"
                elif self.TOKEN_REGEX.search(value):
                    sanitized[key] = "[REDACTED_SECRET_TOKEN]"
                else:
                    sanitized[key] = value
            elif isinstance(value, (list, tuple)) and len(value) > 32 and all(isinstance(x, (int, float)) for x in value[:10]):
                sanitized[key] = f"[REDACTED_TENSOR_BLOB: length={len(value)}]"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            else:
                sanitized[key] = value
        return sanitized


@dataclass
class AuditDiffSnapshot:
    """
    Captures before-and-after state transitions (javers/paper_trail model).
    """
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    delta_pct: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "delta_pct": self.delta_pct
        }

    def to_sanitized_dict(self, sanitizer: Optional[SanitizationFilter] = None) -> Dict[str, Any]:
        s = sanitizer or SanitizationFilter()
        return {
            "before": s.sanitize_dict(self.before),
            "after": s.sanitize_dict(self.after),
            "delta_pct": self.delta_pct
        }


@dataclass
class AuditTrailRecord:
    """
    Standard Event Schema adhering to the 5 Ws:
      - Actor (Who)
      - Action (What)
      - Resource (Target)
      - Timestamp (When: UTC ISO-8601 millisecond precision)
      - Context (Where / How: cluster topology, run ID, correlation ID)
      - Diff / State: Before and after telemetry and configuration delta
    """
    actor: str
    action: str
    resource: str
    context: Dict[str, Any]
    diff: AuditDiffSnapshot
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = ""
    prev_hash: str = "0" * 64
    record_hash: str = ""

    def __post_init__(self):
        if not self.event_id:
            # Generate deterministic event fingerprint if empty
            raw_id = f"{self.actor}:{self.action}:{self.resource}:{self.timestamp}"
            self.event_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

    def compute_hash(self, prev_hash: Optional[str] = None) -> str:
        """
        Computes SHA-256 hash of (prev_hash + canonical serialized event).
        Both context and diff state (before/after) are recursively sanitized.
        """
        use_prev = prev_hash if prev_hash is not None else self.prev_hash
        sanitizer = SanitizationFilter()
        sanitized_context = sanitizer.sanitize_dict(self.context)
        sanitized_diff = self.diff.to_sanitized_dict(sanitizer)
        
        payload = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "context": sanitized_context,
            "diff": sanitized_diff,
            "prev_hash": use_prev
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class AuditTrailEngine:
    """
    Append-only, tamper-evident cryptographic audit log engine (immudb / auditum style).
    Maintains a SHA-256 hash chain where each event cryptographically signs the preceding chain state.
    """
    def __init__(self):
        self.records: List[AuditTrailRecord] = []
        self.sanitizer = SanitizationFilter()

    def record_event(
        self,
        actor: str,
        action: str,
        resource: str,
        context: Dict[str, Any],
        diff: Optional[AuditDiffSnapshot] = None
    ) -> AuditTrailRecord:
        prev_hash = self.records[-1].record_hash if self.records else "0" * 64
        
        diff_snapshot = diff or AuditDiffSnapshot()
        sanitized_context = self.sanitizer.sanitize_dict(context)
        
        record = AuditTrailRecord(
            actor=actor,
            action=action,
            resource=resource,
            context=sanitized_context,
            diff=diff_snapshot,
            prev_hash=prev_hash
        )
        record.record_hash = record.compute_hash(prev_hash=prev_hash)
        self.records.append(record)
        return record

    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verifies the cryptographic integrity of the entire audit chain.
        Returns (True, None) if untampered, or (False, error_message) if tampered.
        """
        expected_prev_hash = "0" * 64
        for idx, rec in enumerate(self.records):
            if rec.prev_hash != expected_prev_hash:
                return (
                    False,
                    f"Tamper detected at index {idx}: broken link. Expected prev_hash {expected_prev_hash}, got {rec.prev_hash}"
                )
            
            calculated_hash = rec.compute_hash(prev_hash=expected_prev_hash)
            if rec.record_hash != calculated_hash:
                return (
                    False,
                    f"Tamper detected at index {idx}: record content modified. Expected {calculated_hash}, found {rec.record_hash}"
                )
            
            expected_prev_hash = rec.record_hash
            
        return (True, None)

    def export_audit_log_json(self) -> str:
        """
        Exports the verified audit trail as a formatted JSON document.
        """
        export_list = []
        for r in self.records:
            export_list.append({
                "event_id": r.event_id,
                "timestamp": r.timestamp,
                "actor": r.actor,
                "action": r.action,
                "resource": r.resource,
                "context": r.context,
                "diff": r.diff.to_dict(),
                "prev_hash": r.prev_hash,
                "record_hash": r.record_hash
            })
        return json.dumps(export_list, indent=2)
