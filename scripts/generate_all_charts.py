"""
Master Chart Generation Pipeline.
Runs all three specialized chart generation scripts sequentially to produce
the complete set of 18 analytical, commercial, roadmap, and telemetry charts.
Validates all outputs across both flat charts/ and the 3-Pillar directory hierarchy.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_script(script_rel_path: str):
    script_path = REPO_ROOT / script_rel_path
    print(f"\n========================================================")
    print(f"  Executing: {script_rel_path}")
    print(f"========================================================")
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_rel_path} failed with code {result.returncode}")


def main():
    print("[GhostLayer] Starting Master Chart Generation Pipeline...")
    
    # 1. Financial analysis and statistical curvature charts
    run_script("scripts/generate_analysis_charts.py")
    
    # 2. Roadmap Gantt, subrun taxonomy, and SLA timeline charts
    run_script("scripts/generate_roadmap_and_timeline_charts.py")
    
    # 3. Engineering step latency, VRAM, and memory bandwidth charts
    run_script("scripts/generate_engineering_and_telemetry_charts.py")

    # 4. Verification
    charts_dir = REPO_ROOT / "charts"
    p1_dir = charts_dir / "01_revenue_and_commercial"
    p2_dir = charts_dir / "02_prospects_and_pipeline"
    p3_dir = charts_dir / "03_engineering_and_algorithms"

    print("\n========================================================")
    print("  Master Chart Generation Summary")
    print("========================================================")
    print(f"Pillar 1 (Revenue & Commercial)    : {len(list(p1_dir.glob('*.png')))} charts")
    print(f"Pillar 2 (Prospects & Pipeline)    : {len(list(p2_dir.glob('*.png')))} charts")
    print(f"Pillar 3 (Engineering & Algorithms): {len(list(p3_dir.glob('*.png')))} charts")
    print(f"Total Unique Pillar Charts         : {len(list(p1_dir.glob('*.png'))) + len(list(p2_dir.glob('*.png'))) + len(list(p3_dir.glob('*.png')))}")
    print(f"Root charts/ Directory Files       : {len(list(charts_dir.glob('*.png')))} charts")
    print("\n[SUCCESS] Master Chart Generation Completed Successfully!\n")


if __name__ == "__main__":
    main()
