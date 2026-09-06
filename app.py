"""
Ghost Layer Unified Master Application Entrypoint.

Usage:
  python app.py [command]

Commands:
  benchmark-genuine : Run empirical benchmark across genuine published Hugging Face training logs.
  compensate        : Test active closed-loop S-plane damping across training regimes.
  frontier          : Run frontier-scale cluster telemetry & dual-revenue ROI projections.
  audit             : Run efficiency and configuration audit.
  graphify          : Run computational DAG analysis & kernel fusion optimizer.
  kb                : Inspect anonymized shared knowledge base entries.
"""
import sys
import os

# Ensure package is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ghost_layer.app import main

if __name__ == "__main__":
    main()
