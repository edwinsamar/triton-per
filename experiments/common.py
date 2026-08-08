"""Run-artifact collection: every experiment writes a self-describing run dir.

    results/runs/<name>/
        config.json    all hyperparameters + seed + git SHA + host + GPU
        env.txt        python/torch/triton/torchrl versions, pip freeze, nvidia-smi
        metrics.csv    periodic: env_step, episodic return, wall-clock, sps
        final.json     summary written on clean exit

A paper claiming reproducibility points at: (repo commit, config.json, seed) ->
rerun bit-similar experiment.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa
        return f"unavailable: {e}"


class RunLogger:
    def __init__(self, run_dir: str | Path, config: dict):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()

        config = dict(config)
        config["git_sha"] = _cmd(["git", "rev-parse", "HEAD"])
        config["git_dirty"] = bool(_cmd(["git", "status", "--porcelain"]))
        config["hostname"] = platform.node()
        config["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

        lines = [
            f"python {sys.version}",
            f"torch {torch.__version__} (cuda {torch.version.cuda})",
        ]
        for mod in ("triton", "torchrl", "tensordict", "gymnasium", "numpy"):
            try:
                m = __import__(mod)
                lines.append(f"{mod} {m.__version__}")
            except Exception:
                lines.append(f"{mod} not installed")
        lines.append("--- nvidia-smi ---")
        lines.append(_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv"]))
        lines.append("--- pip freeze ---")
        lines.append(_cmd([sys.executable, "-m", "pip", "freeze"]))
        (self.dir / "env.txt").write_text("\n".join(lines))

        self._csv = open(self.dir / "metrics.csv", "w", newline="")
        self._writer = None

    def log(self, **row):
        row = {"wall_s": round(time.perf_counter() - self.t0, 2), **row}
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv.flush()

    def finish(self, **summary):
        summary["total_wall_s"] = round(time.perf_counter() - self.t0, 2)
        (self.dir / "final.json").write_text(json.dumps(summary, indent=2))
        self._csv.close()


def seed_everything(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
