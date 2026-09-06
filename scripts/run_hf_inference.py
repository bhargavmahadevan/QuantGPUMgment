"""
Hugging Face Cloud Inference Benchmark Runner.

Executes remote cloud inference benchmarking via huggingface_hub.InferenceClient.
Saves structured inference latency receipts to .ghostlayer/inference_benchmark.json.

NOTE: This measures hosted API inference latency, distinct from training-step GPU telemetry.
"""

import sys
import os
import argparse
from pathlib import Path

# Add repository root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from ceo_evidence_engine.hf_cloud_inference_client import run_cloud_inference_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="GhostLayer Hugging Face Cloud Inference Benchmark Runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face Model ID (default: meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="featherless-ai",
        help="Inference provider (featherless-ai, together, etc.)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Can you please let us know more details about your ",
        help="Inference prompt string",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()

    print("\n[GhostLayer] Running Hugging Face Cloud Inference Benchmark...")
    print(f"  Model    : {args.model}")
    print(f"  Provider : {args.provider}")
    print(f"  Prompt   : '{args.prompt}'")
    
    result = run_cloud_inference_benchmark(
        prompt=args.prompt,
        model=args.model,
        provider=args.provider,
        max_new_tokens=args.max_new_tokens,
    )

    if result:
        print(f"[Done] Mode: {result.get('mode')}")
        print(f"[Done] Output saved to: .ghostlayer/inference_benchmark.json")


if __name__ == "__main__":
    main()
