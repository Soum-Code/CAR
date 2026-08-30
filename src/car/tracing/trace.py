"""JSONL trace writer.

Named `tracing` rather than `logging` so it cannot shadow the standard library
module inside this package.

Every decision the system makes is appended here: score, threshold, influence,
gate outcome, exploration flag, verdict, budget state. Two reasons this is not
optional.

1. Reproducibility. A results table with no trace behind it cannot be audited,
   and "we cannot reconstruct how that number arose" is fatal in a thesis
   defence.
2. The analysis pipeline reads traces, not models. Once the GPU has written
   them, every experiment and ablation runs on CPU against these files.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from car.types import Trajectory


class TraceWriter:
    """Append-only JSONL writer for trajectories."""

    def __init__(self, path: str | Path, *, manifest: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self.n_written = 0
        if manifest is not None:
            self.write_manifest(manifest)

    def write_manifest(self, config: dict) -> None:
        """Record the exact configuration a run was produced under.

        `config_hash` is what makes test-set leakage detectable: if the hash of
        the config that produced a calibration threshold differs from the one
        in the test run, something was retuned between them.
        """
        record = {
            "_type": "manifest",
            "timestamp": datetime.now(UTC).isoformat(),
            "config": config,
            "config_hash": config_hash(config),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def write(self, traj: Trajectory) -> None:
        self._fh.write(traj.model_dump_json() + "\n")
        self.n_written += 1
        # Flushing per trajectory costs throughput but means a run killed by a
        # preemptible cloud instance keeps everything up to that point.
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def read_traces(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL trace, skipping manifest records."""
    out: list[Trajectory] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("_type") == "manifest":
            continue
        out.append(Trajectory.model_validate(obj))
    return out


def read_manifests(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("_type") == "manifest":
            out.append(obj)
    return out


def flatten_steps(trajectories: list[Trajectory]) -> list[dict]:
    """Explode trajectories into one row per step, for analysis in pandas."""
    rows = []
    for t in trajectories:
        for r in t.steps:
            rows.append(
                {
                    "example_id": t.example_id,
                    "step_id": r.step.step_id,
                    "score": r.score,
                    "threshold": r.threshold,
                    "gate_value": r.gate_value,
                    "influence": r.influence,
                    "decision": r.decision.value,
                    "forced_exploration": r.forced_exploration,
                    "verdict": r.verdict.value if r.verdict else None,
                    "label": r.label,
                    "n_dependencies": len(r.step.dependency_ids),
                    "final_correct": t.correct,
                    **r.features.model_dump(),
                }
            )
    return rows
