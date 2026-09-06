"""
Automated Test Inventory Generator.

Inspects the active test suite via pytest collection, generates the canonical
TEST_INVENTORY.json, and eliminates hardcoded test counts across documentation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def collect_pytest_inventory(repo_root: Path) -> Dict[str, Any]:
    """Runs pytest collection and parses the exact test cases."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)

    if proc.returncode != 0 and "no tests collected" not in proc.stdout:
        print(f"Warning: pytest collection exited with code {proc.returncode}")

    test_ids: List[str] = []
    file_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {
        "unit_tests": 0,
        "evidence_engine": 0,
        "closed_loop_and_control": 0,
        "benchmarking_and_calibration": 0,
        "hardware_telemetry": 0,
    }

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("warning"):
            continue
        # Standard pytest line looks like: tests/test_foo.py: 12 or test node IDs
        # When called with -q --collect-only, lines list file and count or node IDs
        if "::" in line:
            test_ids.append(line)
            file_part = line.split("::")[0].replace("\\", "/")
            file_counts[file_part] = file_counts.get(file_part, 0) + 1
        elif ":" in line and not line.startswith("ERROR"):
            parts = line.split(":")
            if len(parts) == 2 and parts[1].strip().isdigit():
                file_part = parts[0].strip().replace("\\", "/")
                count = int(parts[1].strip())
                file_counts[file_part] = count

    total_tests = sum(file_counts.values()) if file_counts else len(test_ids)

    # Categorize
    for file_name, count in file_counts.items():
        if "ceo_evidence" in file_name:
            category_counts["evidence_engine"] += count
        elif "closed_loop" in file_name or "control" in file_name:
            category_counts["closed_loop_and_control"] += count
        elif "telemetry" in file_name or "probe" in file_name:
            category_counts["hardware_telemetry"] += count
        else:
            category_counts["unit_tests"] += count

    now_iso = datetime.now(timezone.utc).isoformat()

    inventory = {
        "total_active_tests": total_tests,
        "generated_at_utc": now_iso,
        "timestamp": time.time(),
        "pytest_exit_code": proc.returncode,
        "category_breakdown": category_counts,
        "file_breakdown": file_counts,
    }

    return inventory


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    inventory = collect_pytest_inventory(repo_root)

    out_file = repo_root / "TEST_INVENTORY.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print(f"[Ghost Layer] TEST_INVENTORY.json generated successfully.")
    print(f"  Total Active Tests Collected: {inventory['total_active_tests']}")
    print(f"  Timestamp: {inventory['generated_at_utc']}")


if __name__ == "__main__":
    main()
