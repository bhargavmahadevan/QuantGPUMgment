import json
import os
import hashlib
import math
import time
import enum
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple


class OutcomeStatus(enum.Enum):
    """
    Explicit outcome taxonomy preventing semantic contamination in Knowledge Base priors.
    Only VERIFIED_SAFE and VERIFIED_NON_INFERIOR outcomes contribute to speedup averages.
    """
    MEASURED = "MEASURED"
    VERIFIED_SAFE = "VERIFIED_SAFE"
    VERIFIED_NON_INFERIOR = "VERIFIED_NON_INFERIOR"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    SIMULATED = "SIMULATED"
    REFERENCE_ONLY = "REFERENCE_ONLY"


@dataclass
class TelemetryField:
    """A single telemetry field with its purpose and privacy classification."""
    field_name: str
    data_type: str
    purpose: str
    privacy_class: str  # "safe", "anonymized", "excluded"
    example_value: str


class TelemetryBoundary:
    """
    Explicit contract of what telemetry GhostLayer collects and what it excludes.
    
    This serves as a machine-readable privacy manifest that enterprise buyers
    and compliance teams can audit.
    """
    # Fields GhostLayer DOES collect (all are operational metrics, no PII or IP)
    COLLECTED_FIELDS: List[TelemetryField] = [
        TelemetryField("gpu_utilization_pct", "float", "GPU compute utilization per step", "safe", "85.2"),
        TelemetryField("gpu_memory_used_mb", "float", "GPU VRAM allocation per step", "safe", "12450.0"),
        TelemetryField("step_time_ms", "float", "Wall-clock time per training step", "safe", "31.4"),
        TelemetryField("loss", "float", "Training loss value (scalar only)", "safe", "2.341"),
        TelemetryField("data_loading_time_ms", "float", "DataLoader I/O stall time", "safe", "5.2"),
        TelemetryField("mixed_precision", "str", "Active precision mode", "safe", "fp32"),
        TelemetryField("gradient_checkpointing", "bool", "Whether grad checkpointing is enabled", "safe", "False"),
        TelemetryField("flash_attention", "bool", "Whether FlashAttention is enabled", "safe", "False"),
        TelemetryField("num_workers", "int", "DataLoader worker count", "safe", "0"),
        TelemetryField("architecture_family", "str", "Model architecture category (anonymized)", "anonymized", "transformer"),
        TelemetryField("hardware_type", "str", "GPU model name", "safe", "NVIDIA RTX A2000"),
        TelemetryField("param_count_bucket", "str", "Parameter count range bucket (not exact count)", "anonymized", "medium(100M-1B)"),
        TelemetryField("attention_architecture", "str", "Attention mechanism type (MHA, GQA, MLA, DSA)", "safe", "GQA"),
        TelemetryField("kv_cache_mb", "float", "Estimated active KV cache memory footprint", "safe", "512.0"),
        TelemetryField("ttft_ms", "float", "Prefill time-to-first-token latency", "safe", "45.2"),
        TelemetryField("tpot_ms", "float", "Decode time-per-output-token latency", "safe", "12.8"),
        TelemetryField("dominant_latency_bottleneck", "str", "Diagnosed latency bottleneck category", "safe", "MEMORY_BANDWIDTH_BOUND"),
    ]

    # Fields GhostLayer NEVER collects
    EXCLUDED_FIELDS: List[TelemetryField] = [
        TelemetryField("model_weights", "tensor", "Model parameters / state_dict", "excluded", "N/A"),
        TelemetryField("training_data", "any", "Training dataset contents or samples", "excluded", "N/A"),
        TelemetryField("dataset_path", "str", "File paths to training data", "excluded", "N/A"),
        TelemetryField("model_outputs", "tensor", "Model predictions or logits", "excluded", "N/A"),
        TelemetryField("gradients", "tensor", "Individual gradient values", "excluded", "N/A"),
        TelemetryField("prompts", "str", "Training prompts or text data", "excluded", "N/A"),
        TelemetryField("tokens", "list", "Tokenized input sequences", "excluded", "N/A"),
        TelemetryField("client_name", "str", "Client identity information", "excluded", "N/A"),
        TelemetryField("user_id", "str", "User identification", "excluded", "N/A"),
        TelemetryField("ip_address", "str", "Network addresses", "excluded", "N/A"),
        TelemetryField("api_keys", "str", "Authentication credentials", "excluded", "N/A"),
        TelemetryField("file_paths", "str", "Local filesystem paths", "excluded", "N/A"),
    ]

    @classmethod
    def audit(cls) -> Dict[str, Any]:
        """Generate a complete telemetry boundary audit report."""
        return {
            "boundary_version": "1.0",
            "collected": [asdict(f) for f in cls.COLLECTED_FIELDS],
            "excluded": [asdict(f) for f in cls.EXCLUDED_FIELDS],
            "enforcement": {
                "privacy_scrubber": "PrivacyScrubber.sanitize() applied to all KB entries",
                "forbidden_keys": list(PrivacyScrubber._FORBIDDEN_KEYS),
                "path_detection": "Any string containing '/' or '\\' or 'C:' is excluded",
            },
            "summary": {
                "total_collected": len(cls.COLLECTED_FIELDS),
                "total_excluded": len(cls.EXCLUDED_FIELDS),
                "pii_collected": False,
                "model_weights_collected": False,
                "training_data_collected": False,
            },
        }

    @classmethod
    def format_audit_markdown(cls) -> str:
        """Format the telemetry boundary as a readable markdown report."""
        lines = ["# GhostLayer Telemetry Boundary Audit\n"]
        lines.append("## Data We Collect\n")
        lines.append("| Field | Type | Purpose | Privacy |")
        lines.append("|:---|:---|:---|:---|")
        for f in cls.COLLECTED_FIELDS:
            lines.append(f"| `{f.field_name}` | {f.data_type} | {f.purpose} | {f.privacy_class} |")
        lines.append("\n## Data We NEVER Collect\n")
        lines.append("| Field | Type | Why Excluded | Privacy |")
        lines.append("|:---|:---|:---|:---|")
        for f in cls.EXCLUDED_FIELDS:
            lines.append(f"| `{f.field_name}` | {f.data_type} | {f.purpose} | {f.privacy_class} |")
        return "\n".join(lines)



@dataclass
class KnowledgeEntry:
    entry_id: str
    architecture_family: str
    hardware_type: str
    effective_config: Dict[str, Any]
    throughput_improvement_pct: float
    sample_count: int
    rejected_sample_count: int = 0
    param_count_bucket: str = ""
    framework_version: str = ""
    _m2: float = 0.0
    provenance_note: str = ""
    verifier_status: str = "UNVERIFIED"
    outcome_status: str = "MEASURED"
    last_updated_timestamp: float = field(default_factory=time.time)

    def get_decay_weight(self, half_life_days: float = 30.0, current_time: Optional[float] = None) -> float:
        """
        Calculates time-decay weight: w = exp(-lambda * delta_t_days).
        Lambda = ln(2) / half_life_days.
        """
        now = current_time if current_time is not None else time.time()
        delta_days = max(0.0, (now - self.last_updated_timestamp) / 86400.0)
        decay_constant = math.log(2.0) / max(0.1, half_life_days)
        return round(math.exp(-decay_constant * delta_days), 4)

    @property
    def variance(self) -> float:
        if self.sample_count < 2:
            return 0.0
        return round(self._m2 / (self.sample_count - 1), 4)

    @property
    def std_dev(self) -> float:
        return round(self.variance ** 0.5, 3)

    @property
    def confidence_tier(self) -> str:
        if self.sample_count == 0:
            return "NONE"
        if self.sample_count < 3:
            return "LOW"
        if self.sample_count < 10:
            return "MEDIUM" if self.std_dev < 15.0 else "LOW"
        return "HIGH" if self.std_dev < 10.0 else "MEDIUM"

    @property
    def surrogate_posterior(self) -> Dict[str, Any]:
        return SurrogateOptimizationModel.init_posterior()


class PrivacyScrubber:
    _FORBIDDEN_KEYS = {
        "weights", "state_dict", "data_path", "dataset", "dataset_name",
        "prompt", "tokens", "client_name", "user_id", "ip_address", "file_path"
    }

    @staticmethod
    def sanitize(raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for k, v in raw_metadata.items():
            if k.lower() in PrivacyScrubber._FORBIDDEN_KEYS:
                continue
            if isinstance(v, str) and ("/" in v or "\\" in v or "C:" in v):
                continue
            sanitized[k] = v
        return sanitized


def param_count_to_bucket(param_count: Optional[int]) -> str:
    if param_count is None:
        return ""
    if param_count < 10_000_000:
        return "tiny(<10M)"
    if param_count < 100_000_000:
        return "small(10M-100M)"
    if param_count < 1_000_000_000:
        return "medium(100M-1B)"
    if param_count < 10_000_000_000:
        return "large(1B-10B)"
    return "xlarge(10B+)"


class SurrogateOptimizationModel:
    @staticmethod
    def init_posterior() -> Dict[str, Any]:
        return {
            "micro_batch_multiplier": {"1.0": 1.0, "2.0": 1.0, "4.0": 1.0},
            "inductor_mode": {"default": 1.0, "reduce-overhead": 1.0, "max-autotune": 1.0},
            "prefetch_factor": {"2": 1.0, "4": 1.0, "8": 1.0},
            "triton_epilogue_fusion": {"true": 1.0, "false": 1.0}
        }

    @staticmethod
    def update_posterior(posterior: Dict[str, Any], effective_config: Dict[str, Any], throughput_improvement_pct: float, was_verified_safe: bool = True) -> Dict[str, Any]:
        updated = {k: dict(v) for k, v in posterior.items()}
        weight = max(1.0, throughput_improvement_pct / 5.0) if was_verified_safe else 0.5
        for param, val in effective_config.items():
            str_val = str(val).lower()
            if param in updated:
                if str_val not in updated[param]:
                    updated[param][str_val] = 1.0
                updated[param][str_val] += weight
        return updated

    @staticmethod
    def predict_optimal_config(posterior: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for param, values in posterior.items():
            best_val = max(values.items(), key=lambda x: x[1])[0]
            if best_val in ("true", "false"):
                result[param] = (best_val == "true")
            else:
                try:
                    result[param] = float(best_val) if "." in best_val else int(best_val)
                except ValueError:
                    result[param] = best_val
        return result


class SharedKnowledgeBase:
    def __init__(self, db_file_path: Optional[str] = None):
        self.db_file_path = db_file_path or "ghost_knowledge_base.json"
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.rule_calibrations: Dict[str, Dict[str, Any]] = {}
        self.load()

    def _generate_key(self, architecture: str, hardware: str, param_bucket: str = "", framework_version: str = "") -> str:
        raw = f"{architecture.lower().strip()}:{hardware.lower().strip()}:{param_bucket.lower().strip()}:{framework_version.lower().strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def load(self):
        if self.db_file_path and os.path.exists(self.db_file_path):
            try:
                with open(self.db_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "_rule_calibrations" in data:
                        self.rule_calibrations = data.pop("_rule_calibrations")
                    valid_keys = {
                        "entry_id", "architecture_family", "hardware_type",
                        "effective_config", "throughput_improvement_pct",
                        "sample_count", "rejected_sample_count",
                        "param_count_bucket", "framework_version", "_m2",
                        "provenance_note", "verifier_status", "outcome_status",
                        "last_updated_timestamp"
                    }
                    for k, v in data.items():
                        v.setdefault("param_count_bucket", "")
                        v.setdefault("framework_version", "")
                        v.setdefault("_m2", 0.0)
                        v.setdefault("provenance_note", "")
                        v.setdefault("verifier_status", "UNVERIFIED")
                        v.setdefault("outcome_status", "MEASURED")
                        filtered = {k2: v2 for k2, v2 in v.items() if k2 in valid_keys}
                        self.entries[k] = KnowledgeEntry(**filtered)
            except Exception:
                self.entries = {}
                self.rule_calibrations = {}

    def save(self):
        if self.db_file_path:
            out = {k: asdict(v) for k, v in self.entries.items()}
            if self.rule_calibrations:
                out["_rule_calibrations"] = self.rule_calibrations
            with open(self.db_file_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    def register_rule_outcome(
        self,
        rule_id: str,
        was_verified_safe: bool,
        realized_speedup_pct: float = 0.0,
        loss_shift: float = 0.0,
        outcome_status: Optional[str] = None,
    ) -> None:
        """Register empirical outcome for a specific optimization rule to close the feedback loop."""
        if rule_id not in self.rule_calibrations:
            self.rule_calibrations[rule_id] = {
                "verified_safe_count": 0,
                "rejected_count": 0,
                "total_samples": 0,
                "mean_speedup_pct": 0.0,
                "mean_loss_shift": 0.0,
                "last_updated": time.time(),
                "last_outcome_status": "MEASURED",
            }
        rec = self.rule_calibrations[rule_id]
        n = rec["total_samples"] + 1
        rec["total_samples"] = n
        status = outcome_status or ("VERIFIED_SAFE" if was_verified_safe else "REJECTED")
        rec["last_outcome_status"] = status
        if was_verified_safe:
            rec["verified_safe_count"] += 1
            rec["mean_speedup_pct"] = round(
                rec["mean_speedup_pct"] + (realized_speedup_pct - rec["mean_speedup_pct"]) / n, 2
            )
        else:
            rec["rejected_count"] += 1
        rec["mean_loss_shift"] = round(
            rec["mean_loss_shift"] + (loss_shift - rec["mean_loss_shift"]) / n, 4
        )
        rec["last_updated"] = time.time()
        self.save()

    def get_calibrated_threshold(
        self,
        rule_id: str,
        default_threshold: float,
        min_bound: float,
        max_bound: float,
    ) -> Tuple[float, str]:
        """
        Empirically closed-loop threshold adaptation:
        Dynamically adjusts decision thresholds based on accumulated verified outcomes vs rejections.

        - If verified safe win-rate is high (>=80%) with n>=2, sensitivity window expands (shifts toward max_bound).
        - If win-rate is low (<70%) or rejections occur, sensitivity window contracts (shifts toward min_bound).
        - If no samples exist, returns default_threshold labeled as 'HEURISTIC_PRIOR'.
        """
        rec = self.rule_calibrations.get(rule_id)
        if not rec or rec["total_samples"] == 0:
            return default_threshold, "HEURISTIC_PRIOR"

        total = rec["total_samples"]
        safe = rec["verified_safe_count"]
        win_rate = safe / total

        if total >= 2:
            if win_rate >= 0.80:
                factor = min(1.0, (total / 5.0) * (win_rate - 0.5))
                expansion = (max_bound - default_threshold) * factor
                calibrated = min(max_bound, default_threshold + expansion)
                return round(calibrated, 3), f"EMPIRICALLY_ADAPTED (win_rate={win_rate:.1%}, n={total})"
            elif win_rate < 0.70:
                factor = min(1.0, (1.0 - win_rate) * 1.5)
                contraction = (default_threshold - min_bound) * factor
                calibrated = max(min_bound, default_threshold - contraction)
                return round(calibrated, 3), f"EMPIRICALLY_ADAPTED_CONSERVATIVE (win_rate={win_rate:.1%}, rejections={rec['rejected_count']})"

        return default_threshold, f"EMPIRICALLY_VERIFIED (n={total}, neutral)"

    def register_learning(
        self,
        architecture_family: str,
        hardware_type: str,
        effective_config: Dict[str, Any],
        throughput_improvement_pct: float,
        was_verified_safe: bool = True,
        param_count_bucket: str = "",
        framework_version: str = "",
        is_contaminated: bool = False,
        outcome_status: Optional[Any] = None,
        provenance_note: str = "",
    ) -> Optional[KnowledgeEntry]:
        if is_contaminated:
            # Block contaminated/throttled runs from poisoning the Knowledge Base priors
            return None

        # Resolve outcome status string
        if outcome_status is not None:
            status_str = outcome_status.value if hasattr(outcome_status, "value") else str(outcome_status)
        else:
            status_str = "VERIFIED_SAFE" if was_verified_safe else "REJECTED"

        # Only VERIFIED_SAFE and VERIFIED_NON_INFERIOR runs count as verified safe samples
        is_safe_sample = status_str in ("VERIFIED_SAFE", "VERIFIED_NON_INFERIOR") and was_verified_safe

        safe_config = PrivacyScrubber.sanitize(effective_config)
        key = self._generate_key(architecture_family, hardware_type, param_count_bucket, framework_version)

        if key in self.entries:
            existing = self.entries[key]
            existing.outcome_status = status_str
            if provenance_note:
                existing.provenance_note = provenance_note
            if is_safe_sample:
                n = existing.sample_count + 1
                delta = throughput_improvement_pct - existing.throughput_improvement_pct
                new_avg = existing.throughput_improvement_pct + delta / n
                delta2 = throughput_improvement_pct - new_avg
                existing._m2 += delta * delta2
                existing.throughput_improvement_pct = round(new_avg, 2)
                existing.sample_count = n
                existing.verifier_status = "VERIFIED_SAFE"
            else:
                existing.rejected_sample_count += 1
                existing.verifier_status = status_str
            existing.effective_config.update(safe_config)
            existing.last_updated_timestamp = time.time()
            entry = existing
        else:
            entry = KnowledgeEntry(
                entry_id=key,
                architecture_family=architecture_family,
                hardware_type=hardware_type,
                effective_config=safe_config,
                throughput_improvement_pct=round(throughput_improvement_pct, 2) if is_safe_sample else 0.0,
                sample_count=1 if is_safe_sample else 0,
                rejected_sample_count=0 if is_safe_sample else 1,
                param_count_bucket=param_count_bucket,
                framework_version=framework_version,
                provenance_note=provenance_note,
                verifier_status="VERIFIED_SAFE" if is_safe_sample else status_str,
                outcome_status=status_str,
            )
            self.entries[key] = entry

        self.save()
        return entry

    def query(self, architecture_family: str, hardware_type: str,
              param_count_bucket: str = "", framework_version: str = "") -> Optional[KnowledgeEntry]:
        key = self._generate_key(architecture_family, hardware_type, param_count_bucket, framework_version)
        return self.entries.get(key)

    def find_best_match(
        self,
        architecture_family: str,
        hardware_type: str,
        param_count_bucket: str = "",
        framework_version: str = "",
    ) -> Tuple[Optional[KnowledgeEntry], str]:
        candidates = [
            (architecture_family, hardware_type, param_count_bucket, framework_version, "EXACT"),
            (architecture_family, hardware_type, param_count_bucket, "", "ARCH+HW+SIZE"),
            (architecture_family, hardware_type, "", "", "ARCH+HW"),
        ]
        for arch, hw, bucket, fw, level in candidates:
            key = self._generate_key(arch, hw, bucket, fw)
            entry = self.entries.get(key)
            if entry is not None and entry.sample_count > 0:
                return entry, level
        return None, "NONE"

    def predict_hive_mind_config(self, architecture_family: str, hardware_type: str) -> Dict[str, Any]:
        entry, match_level = self.find_best_match(architecture_family, hardware_type)
        if entry is not None and entry.effective_config:
            return entry.effective_config
        return {
            "micro_batch_multiplier": 2.0,
            "inductor_mode": "reduce-overhead",
            "prefetch_factor": 4,
            "cuda_alloc_conf": "max_split_size_mb:128",
            "triton_epilogue_fusion": True
        }
