"""Append-only JSONL run log. One file per run, keyed by alert id and run id.

Every event carries ``alert_id``, ``run_id``, ``seq`` (monotonic within the run) and
``ts`` (wall clock). A run must be replayable from its log alone: the first record is
``run_start`` (model, prompt version, package hash, full package), and every
subsequent model turn, tool call, tool result, cost and latency is its own record.

The file is opened in append mode and every record is written with a single
``write`` of one line, so a crash loses at most the in-flight record. Nothing here
rewrites or truncates.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "results" / "agent" / "logs"


def new_run_id() -> str:
    """Run id: UTC timestamp plus pid, sortable and unique enough for one box."""
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"Z-{os.getpid()}"


def package_sha256(package: dict) -> str:
    """Stable digest of the evidence package as the model saw it."""
    blob = json.dumps(package, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RunLog:
    """Append-only writer for one agent run."""

    def __init__(self, alert_id: str, run_id: str, log_dir: Path | None = None):
        self.alert_id = alert_id
        self.run_id = run_id
        self.dir = Path(log_dir) if log_dir else DEFAULT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"alert_{alert_id}__run_{run_id}.jsonl"
        if self.path.exists():
            raise FileExistsError(f"refusing to append to existing run log: {self.path}")
        self._seq = 0

    def _now(self) -> str:
        return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")

    def record(self, event: str, **fields: Any) -> dict:
        """Append one record. Returns it so callers can reuse computed values."""
        rec = {
            "seq": self._seq,
            "ts": self._now(),
            "alert_id": self.alert_id,
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        line = json.dumps(rec, sort_keys=True, default=str) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._seq += 1
        return rec


def read_log(path: Path) -> list[dict]:
    """Load a run log. Used by tests and the replay check."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
