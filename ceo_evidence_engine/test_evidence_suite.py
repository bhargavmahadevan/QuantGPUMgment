"""
CEO Evidence Engine — Automated Pytest Verification Suite
Tests benchmark execution, JSON evidence generation, report formatting, A/B verification logic,
SHA-256 cryptographic audit hashing, projection model provenance, and HF Cloud Inference benchmark.
"""

import os
import json
import pytest
from ceo_evidence_engine.benchmark_runner import run_benchmark, get_hardware_telemetry
from ceo_evidence_engine.report_generator import generate_markdown_report
from ceo_evidence_engine.shadow_audit_collector import ShadowAuditCollector
from ceo_evidence_engine.split_fleet_verifier import run_split_fleet_verification
from ceo_evidence_engine.real_hardware_benchmarks import execute_model_benchmark, load_hardware_reference_database, load_projection_assumptions
from ceo_evidence_engine.hf_cloud_inference_client import run_cloud_inference_benchmark

def test_hardware_telemetry():
    """Verify hardware telemetry format."""
    telemetry = get_hardware_telemetry()
    assert "timestamp" in telemetry
    assert "run_id" in telemetry
    assert "gpu_name" in telemetry

def test_hardware_reference_database_loading():
    """Verify loading hardware_reference_database.json."""
    db = load_hardware_reference_database()
    assert "database_metadata" in db
    assert "workloads" in db
    assert "llama3-8b" in db["workloads"]
    assert "provenance" in db["workloads"]["llama3-8b"]["hardware_references"][0]

def test_projection_assumptions_loading():
    """Verify loading projection_model_assumptions.json."""
    assumptions = load_projection_assumptions()
    assert "projection_methods" in assumptions
    assert "analytical_kernel_estimator" in assumptions["projection_methods"]
    assert assumptions["projection_methods"]["analytical_kernel_estimator"]["default_confidence_score"] == 0.88

def test_run_benchmark_sha256_hashing():
    """Verify benchmark runner produces valid JSON artifact with 64-character SHA-256 audit hash."""
    res = run_benchmark(workload="llama3-8b", mode="shadow", steps=10)
    assert res["metadata"]["model_key"] == "llama3-8b"
    sha256_hash = res["metadata"]["reproducibility"]["sha256_audit_hash"]
    assert len(sha256_hash) == 64
    assert all(c in "0123456789abcdef" for c in sha256_hash)

def test_report_generator(tmp_path):
    """Verify HTML audit card generation from real hardware benchmark artifact in sample_outputs/."""
    from ceo_evidence_engine.real_hardware_benchmarks import generate_html_audit_card
    res = run_benchmark(workload="llama3-8b", mode="shadow", steps=10)
    html_path = generate_html_audit_card(res)
    assert os.path.exists(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "GhostLayer Hardware Audit Card" in content
        assert "SHA-256 CRYPTOGRAPHIC AUDIT HASH" in content

def test_shadow_audit_collector():
    """Verify Shadow Audit Collector metrics."""
    collector = ShadowAuditCollector(cluster_name="Test Cluster")
    for _ in range(50):
        collector.observe_step(step_latency_sec=0.030, vram_allocated_mb=6000, vram_reserved_mb=8000)
    report = collector.generate_shadow_report(active_gpus=32)
    assert report["observation_duration_hours"] == 48.0
    assert report["financial_attribution"]["projected_net_monthly_savings"] > 0

def test_split_fleet_verification():
    """Verify Split-Fleet A/B experiment calculation."""
    res = run_split_fleet_verification(total_gpus=64, control_gpus=16, active_gpus=48, steps=100)
    assert res["verification_summary"]["latency_reduction_pct"] > 0
    assert "CFO-AUDIT-" in res["verification_summary"]["cfo_audit_token"]

def test_real_hardware_benchmarks_provenance():
    """Verify hardware benchmarks attach SHA-256 hashes and projection provenance assumptions."""
    res = execute_model_benchmark("llama3-8b")
    assert res["metadata"]["model_key"] == "llama3-8b"
    sha256_hash = res["metadata"]["reproducibility"]["sha256_audit_hash"]
    assert len(sha256_hash) == 64
    
    result_entry = res["results"][0]
    assert result_entry["mode"] in ["LIVE_MEASURED_RUN", "PUBLISHED_REFERENCE_BASELINE"]
    
    if result_entry["mode"] == "PUBLISHED_REFERENCE_BASELINE":
        proj = result_entry["ghostlayer_projected_outcome"]["projection_provenance"]
        assert proj["confidence_score"] == 0.88
        assert "methodology" in proj
        assert len(proj["assumptions"]) > 0

def test_hf_cloud_inference_benchmark():
    """Verify Hugging Face cloud inference benchmark module."""
    res = run_cloud_inference_benchmark(
        prompt="Can you please let us know more details about your ",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        provider="hf-inference"
    )

    assert res is not None
    assert "mode" in res
    assert "metrics" in res
    assert res["metrics"]["throughput_tokens_per_sec"] > 0
    assert len(res["reproducibility"]["sha256_audit_hash"]) == 64

def test_end_to_end_runtime_trace():
    """Verify full end-to-end runtime trace from telemetry -> KB -> decision -> replay -> verification -> report."""
    from ceo_evidence_engine.e2e_runtime_trace_verifier import run_end_to_end_runtime_trace
    assert run_end_to_end_runtime_trace() is True


def test_hardware_target_detection():
    """Verify detect_hardware_target() returns a structurally valid detection report."""
    from ceo_evidence_engine.real_hardware_benchmarks import detect_hardware_target
    report = detect_hardware_target()

    assert "recommended_target" in report
    assert report["recommended_target"] in ("cuda", "mps", "cpu")
    assert "execution_mode" in report
    assert report["execution_mode"] in ("LIVE_MEASURED_RUN", "PUBLISHED_REFERENCE_BASELINE")
    assert isinstance(report["notes"], list)
    assert len(report["notes"]) > 0
    assert isinstance(report["cuda"]["available"], bool)
    assert isinstance(report["mps"]["available"], bool)


def test_html_audit_card_export(tmp_path):
    """Verify generate_html_audit_card() creates a valid HTML file containing the SHA-256 hash."""
    from ceo_evidence_engine.real_hardware_benchmarks import execute_model_benchmark, generate_html_audit_card
    import os

    artifact = execute_model_benchmark("llama3-8b")
    sha256_hash = artifact["metadata"]["reproducibility"]["sha256_audit_hash"]

    html_path = generate_html_audit_card(artifact)
    assert os.path.exists(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "GhostLayer Hardware Audit Card" in content
    assert sha256_hash in content
    assert "SHA-256 CRYPTOGRAPHIC AUDIT HASH" in content
    assert "DOCTYPE html" in content


def test_sha256_hash_reproducibility():
    """Verify that re-computing the SHA-256 hash from the artifact payload always matches the stored value."""
    import json, hashlib
    from ceo_evidence_engine.real_hardware_benchmarks import execute_model_benchmark

    artifact = execute_model_benchmark("llama3-8b")
    stored_hash = artifact["metadata"]["reproducibility"]["sha256_audit_hash"]

    raw_payload = {
        "timestamp": artifact["metadata"]["timestamp"],
        "model_key": artifact["metadata"]["model_key"],
        "results": artifact["results"]
    }
    recomputed = hashlib.sha256(
        json.dumps(raw_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    assert recomputed == stored_hash, (
        f"SHA-256 hash mismatch: stored={stored_hash}, recomputed={recomputed}"
    )


