"""Aggregate the SAC/REDQ array into paper-ready tables.

Reads results/runs/*/ (config.json + metrics.csv + final.json), groups by
(env, utd, backend), and reports:
  - wall-clock (mean over seeds) + speedup of every backend vs cpu-sumtree
  - final return mean +- std over seeds (learning-equivalence evidence)
  - per-pair curve divergence: mean |return_a - return_b| over logged steps,
    in units of the pooled across-seed std (values ~<1 => curves overlap
    within seed noise)

Writes e2e_summary.csv, the input to the paper's number-generation script.

  python experiments/analyze.py --runs results/runs --out results/e2e_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(root: Path):
    rows = []
    for d in sorted(root.iterdir()):
        cfg_f, fin_f, met_f = d / "config.json", d / "final.json", d / "metrics.csv"
        if not (cfg_f.exists() and fin_f.exists()):
            continue
        cfg, fin = json.loads(cfg_f.read_text()), json.loads(fin_f.read_text())
        met = pd.read_csv(met_f) if met_f.exists() else None
        rows.append({
            "env": cfg["env_id"], "backend": cfg["backend"], "utd": cfg["utd"],
            "seed": cfg["seed"], "steps": cfg["total_steps"],
            "wall_s": fin["total_wall_s"],
            "final_return": fin.get("final_return_mean10"),
            "curve": met,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/e2e_summary.csv")
    ap.add_argument("--min-steps", type=int, default=100_000)
    args = ap.parse_args()

    runs = [r for r in load_runs(Path(args.runs)) if r["steps"] >= args.min_steps]
    if not runs:
        print("no completed full-length runs found")
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"} for r in runs])

    print("=== wall-clock (s), mean over seeds ===")
    wall = df.pivot_table(index=["env", "utd"], columns="backend", values="wall_s")
    print(wall.round(0).to_string(), "\n")

    out_rows = []
    if "cpu-sumtree" in wall.columns:
        for backend in wall.columns:
            if backend == "cpu-sumtree":
                continue
            sp = wall["cpu-sumtree"] / wall[backend]
            print(f"=== speedup vs cpu-sumtree: {backend} ===")
            print(sp.round(2).to_string(), "\n")
            for (env, utd), v in sp.items():
                out_rows.append({"env": env, "utd": utd, "backend": backend,
                                 "speedup": round(v, 3)})

    print("=== final return, mean +- std over seeds ===")
    fr = df.groupby(["env", "utd", "backend"]).final_return.agg(["mean", "std"])
    print(fr.round(1).to_string(), "\n")

    print("=== curve divergence (pairs vs cpu-sumtree, in pooled-std units) ===")
    for (env, utd), grp in pd.DataFrame(runs).groupby(["env", "utd"]):
        backends = grp.backend.unique()
        if "cpu-sumtree" not in backends:
            continue
        curves = {}
        for b in backends:
            cs = [r["curve"] for _, r in grp[grp.backend == b].iterrows()
                  if r["curve"] is not None]
            n = min(len(c) for c in cs)
            curves[b] = np.stack([c.return_mean10.values[:n] for c in cs])
        ref = curves["cpu-sumtree"]
        pooled_std = np.mean([c.std(axis=0).mean() for c in curves.values()])
        for b, c in curves.items():
            if b == "cpu-sumtree":
                continue
            n = min(ref.shape[1], c.shape[1])
            div = np.abs(ref[:, :n].mean(0) - c[:, :n].mean(0)).mean()
            units = div / pooled_std if pooled_std > 0 else float("nan")
            print(f"{env} utd={utd} {b}: {div:.1f} return-units"
                  f" = {units:.2f} pooled-std  {'OK' if units < 1 else 'CHECK'}")

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
