"""
CEO Evidence Engine — Hugging Face Cloud Inference Benchmark Client
Executes remote cloud inference benchmarking via huggingface_hub.InferenceClient.
Supports text_generation and chat_completion (conversational) endpoints with automatic provider fallback.
"""

import os
import sys
import time
import json
import uuid
import hashlib
import argparse
from datetime import datetime, timezone

try:
    from huggingface_hub import InferenceClient
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

def run_cloud_inference_benchmark(
    prompt="Can you please let us know more details about your ",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="featherless-ai",
    max_new_tokens=100
):
    """
    Executes a cloud inference benchmark call using Hugging Face InferenceClient.
    Auto-detects whether the provider expects text-generation or conversational chat completion.
    """
    print(f"\n========================================================")
    print(f"  GHOSTLAYER HF CLOUD INFERENCE BENCHMARK")
    print(f"========================================================")
    print(f"Model ID  : {model}")
    print(f"Provider  : {provider}")
    print(f"Prompt    : '{prompt}'")
    print(f"--------------------------------------------------------\n")
    
    hf_token = os.environ.get("HF_TOKEN")
    
    if not HAS_HF_HUB:
        print("[ERROR] huggingface_hub package is not installed. Install via `pip install huggingface_hub`.")
        return None

    if not hf_token:
        print("[WARNING] HF_TOKEN environment variable is not set.")
        print("  Set your token in terminal: $env:HF_TOKEN='your_hf_token' (PowerShell).")
        print("  Executing in Simulated Cloud Mode for local evidence validation...\n")
        
        start_time = time.time()
        time.sleep(0.05)
        generated_text = prompt + "autonomous AI infrastructure intelligence platform designed to optimize enterprise GPU fleets."
        end_time = time.time()
        
        elapsed_sec = end_time - start_time
        tokens_generated = len(generated_text.split()) * 1.3
        throughput_tok_s = round(tokens_generated / max(0.001, elapsed_sec), 2)
        
        evidence_data = {
            "benchmark_category": "HOSTED_INFERENCE_LATENCY",
            "training_efficiency_benchmark": False,
            "note": "Measures hosted inference token latency via Hugging Face API; distinct from training-step GPU telemetry.",
            "mode": "SIMULATED_CLOUD_INFERENCE_RUN (Offline Simulation - requires HF_TOKEN)",
            "provider": provider,
            "cost_tier": "FREE_SIMULATION",
            "model": model,
            "prompt": prompt,
            "generated_text": generated_text,
            "metrics": {
                "ttft_ms": 50.0,
                "end_to_end_latency_sec": round(elapsed_sec, 4),
                "generated_tokens": round(tokens_generated, 1),
                "throughput_tokens_per_sec": throughput_tok_s
            },
            "reproducibility": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": f"EXP-CLOUD-{uuid.uuid4().hex[:8].upper()}",
                "sha256_audit_hash": hashlib.sha256(f"{model}:{provider}:{throughput_tok_s}".encode("utf-8")).hexdigest()
            }
        }
    else:
        print(f"-> Executing Live Cloud Inference Request via HF InferenceClient ({provider})...")
        try:
            client = InferenceClient(
                provider=provider,
                api_key=hf_token
            )
            
            start_time = time.time()
            try:
                # 1. Try text_generation first
                result = client.text_generation(
                    prompt,
                    model=model,
                    max_new_tokens=max_new_tokens
                )
            except Exception as task_err:
                # 2. Fall back to conversational chat_completion if task is conversational
                if "conversational" in str(task_err).lower():
                    messages = [{"role": "user", "content": prompt}]
                    chat_resp = client.chat_completion(
                        messages,
                        model=model,
                        max_tokens=max_new_tokens
                    )
                    result = chat_resp.choices[0].message.content
                else:
                    raise task_err
                    
            end_time = time.time()
            
            elapsed_sec = end_time - start_time
            tokens_generated = len(result.split()) * 1.3
            throughput_tok_s = round(tokens_generated / max(0.001, elapsed_sec), 2)
            
            evidence_data = {
                "benchmark_category": "HOSTED_INFERENCE_LATENCY",
                "training_efficiency_benchmark": False,
                "note": "Measures hosted inference token latency via Hugging Face API; distinct from training-step GPU telemetry.",
                "mode": "LIVE_CLOUD_INFERENCE_RUN",
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "generated_text": result,
                "metrics": {
                    "end_to_end_latency_sec": round(elapsed_sec, 4),
                    "generated_tokens": round(tokens_generated, 1),
                    "throughput_tokens_per_sec": throughput_tok_s
                },
                "reproducibility": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "experiment_id": f"EXP-CLOUD-{uuid.uuid4().hex[:8].upper()}",
                    "sha256_audit_hash": hashlib.sha256(f"{model}:{provider}:{result}".encode("utf-8")).hexdigest()
                }
            }
            print(f"  [SUCCESS] Live Cloud Inference Completed in {elapsed_sec:.3f}s")
            print(f"  [METRICS] Throughput: {throughput_tok_s} tokens/sec")
            clean_res = result.replace('\n', ' ')
            print(f"  [OUTPUT] Response Text: '{clean_res[:100]}...'\n")
            
        except Exception as e:
            print(f"[ERROR] Cloud inference API call failed: {e}")
            return None

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"cloud_inference_{uuid.uuid4().hex[:8]}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"[SAVED] Cloud Evidence Artifact saved to: {out_file}")

    # Sync to canonical .ghostlayer/ store alongside calibration.json
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ghostlayer_dir = os.path.join(root_dir, ".ghostlayer")
    os.makedirs(ghostlayer_dir, exist_ok=True)
    ghostlayer_file = os.path.join(ghostlayer_dir, "inference_benchmark.json")
    with open(ghostlayer_file, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"[SAVED] Synced to GhostLayer evidence store: {ghostlayer_file}\n")
    return evidence_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Cloud Inference Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model ID")
    parser.add_argument("--provider", type=str, default="featherless-ai", help="Inference provider (featherless-ai, together, etc.)")
    parser.add_argument("--prompt", type=str, default="Can you please let us know more details about your ", help="Prompt string")
    args = parser.parse_args()
    
    run_cloud_inference_benchmark(prompt=args.prompt, model=args.model, provider=args.provider)
