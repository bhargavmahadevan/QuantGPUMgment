# CEO Evidence Engine — Sample Outputs

This directory contains outputs from the `ceo_evidence_engine` report-generation pipeline.

## Current Status

**No verified GPU evidence exists in this directory.** The `cloud_inference_*.json` files are simulation/inference projection outputs (not measured GPU benchmarks).

All previously stored `evidence_run_RUN-*.json`, `evidence_report_RUN-*.md`, and `hardware_evidence_*.json` files have been moved to [`_QUARANTINED_CPU_ONLY_RUNS/`](./_QUARANTINED_CPU_ONLY_RUNS/README.md) because they were generated on a CPU-only PyTorch install (`cuda_available: false`) and contain fabricated GPU performance figures. See the quarantine README for details.

## To Generate Real Evidence

Run the benchmark runner on a machine with a CUDA-capable PyTorch install:

```bash
python -m ceo_evidence_engine.benchmark_runner --workload <model> --mode shadow --steps <N>
```

Verify `cuda_available: true` in the output JSON before citing any number.
