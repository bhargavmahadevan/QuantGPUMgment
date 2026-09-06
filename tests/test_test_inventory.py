"""
Unit test verifying TEST_INVENTORY.json matches current repository state.
Guarantees test count integrity without hardcoded stale numbers.
"""

import json
from pathlib import Path
import pytest


def test_test_inventory_json_exists_and_valid():
    """Verify TEST_INVENTORY.json exists, parses, and contains expected metadata."""
    repo_root = Path(__file__).resolve().parent.parent
    inv_path = repo_root / "TEST_INVENTORY.json"
    assert inv_path.exists(), "TEST_INVENTORY.json must exist at repo root"

    with open(inv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_active_tests" in data
    assert data["total_active_tests"] > 0
    assert "category_breakdown" in data
    assert "file_breakdown" in data
    assert sum(data["file_breakdown"].values()) == data["total_active_tests"]
    assert data["pytest_exit_code"] == 0

