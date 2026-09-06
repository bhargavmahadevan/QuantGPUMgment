"""
Telemetry Persistence & Session Checkpointing:
Provides append-only JSONL write-ahead logs for telemetry snapshots, enabling
crash recovery, state preservation across preemptions/spot instances, and run resumption.
"""

import os
import json
import time
from dataclasses import asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

from ghost_layer.telemetry.watcher import MetricSnapshot


class TelemetryPersistence:
    """
    Manages disk-backed append-only telemetry storage.
    Snapshots are flushed to `.ghostlayer/sessions/{session_id}.jsonl`.
    """

    def __init__(self, storage_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.storage_dir = Path(storage_dir or ".ghostlayer/sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.file_path = self.storage_dir / f"{self.session_id}.jsonl"

    def append_snapshot(self, snapshot: MetricSnapshot):
        """Append a single snapshot as a JSON line and flush immediately."""
        data = asdict(snapshot)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def load_snapshots(self) -> List[MetricSnapshot]:
        """Load all snapshots from the current session file."""
        if not self.file_path.exists():
            return []

        snapshots: List[MetricSnapshot] = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        snapshots.append(MetricSnapshot(**d))
                    except Exception:
                        continue
        return snapshots

    @classmethod
    def load_from_file(cls, file_path: str) -> List[MetricSnapshot]:
        """Load snapshots directly from any given jsonl file path."""
        p = Path(file_path)
        if not p.exists():
            return []
        snapshots: List[MetricSnapshot] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        snapshots.append(MetricSnapshot(**d))
                    except Exception:
                        continue
        return snapshots

    def list_available_sessions(self) -> List[str]:
        """List all available session IDs in the storage directory."""
        if not self.storage_dir.exists():
            return []
        return [f.stem for f in self.storage_dir.glob("*.jsonl")]
