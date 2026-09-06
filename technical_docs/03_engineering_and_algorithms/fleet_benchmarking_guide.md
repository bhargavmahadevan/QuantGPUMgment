# Fleet Benchmarking Guide

To generate 50+ verifications across different hardware architectures (e.g. NVIDIA A100s, H100s, L40s) without fabricating data, GhostLayer includes a distributed runner: `examples/run_fleet_benchmarks.py`.

This script uses `asyncio` and secure SSH tunnels to dispatch the Heavy LLM Benchmark to a fleet of remote nodes simultaneously. Once the benchmarks complete, the script retrieves the official receipts and automatically aggregates the telemetry back into your central `ghost_knowledge_base.json`.

## Prerequisites

1. **Remote GPU Nodes**: Provision a cluster of nodes (via AWS, Lambda Labs, RunPod, etc.).
2. **SSH Keys**: Ensure you have a valid `.pem` or `ed25519` key to securely access the nodes.
3. **GhostLayer Installation**: Your remote nodes must have GhostLayer installed (or the script should be configured to clone the repository as a setup step).

## Configuration

Open `examples/run_fleet_benchmarks.py` and populate the `cluster_nodes` array with your actual node IPs, users, and paths to SSH keys:

```python
cluster_nodes = [
    {"host": "198.51.100.10", "user": "ubuntu", "pem": "~/.ssh/a100.pem", "dir": "~/QuantGPUMgment"},
    {"host": "198.51.100.11", "user": "ubuntu", "pem": "~/.ssh/h100.pem", "dir": "~/QuantGPUMgment"},
    # Add all 50 nodes here...
]
```

## Execution

Run the script from your local control machine:

```bash
python examples/run_fleet_benchmarks.py
```

### What Happens?
1. **Dispatch**: The script sends a command to all 50 nodes concurrently to run `python examples/run_heavy_llama_benchmark.py --model_id meta-llama/Llama-3.2-1B`.
2. **Execution**: The nodes physically execute the heavy training loops on their tensor cores.
3. **Retrieval**: Upon completion, the runner uses `scp` to pull the generated `real_training_validation_heavy_llama_report.md` and the updated local `ghost_knowledge_base.json` from each node into a local `fleet_reports/` directory.
4. **Aggregation**: The script mathematically merges the Welford variance accumulators and sample counts from all 50 nodes into your local `ghost_knowledge_base.json`.

You will then possess cryptographically authentic, non-fabricated hardware evidence across 50 nodes to prove your optimization claims to enterprise clients.
