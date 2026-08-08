"""Microbenchmark: sample + priority-update round trip, per backend.

Writes one CSV row per (backend, capacity, batch_size, obs_dim) with mean/p50/
p95 latency and derived samples/sec. Run on a GPU node:

  python bench/bench_sampling.py --out results/micro_a100.csv
  python bench/bench_sampling.py --backends triton,torchrl-cuda --capacities 1000000

Backends: triton, triton-unfused, torch (naive GPU cumsum+searchsorted),
cpu-sumtree, torchrl-cpu, torchrl-cuda.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch


def make_buffer(backend: str, capacity: int, fields: dict, device: str):
    if backend in ("triton", "triton-unfused", "torch"):
        from triton_per.buffer import PERBuffer
        return PERBuffer(capacity, fields, device=device, backend=backend)
    if backend == "cpu-sumtree":
        from triton_per.baselines.cpu_sumtree import CpuSumTreePER
        return CpuSumTreePER(capacity, fields, device=device)
    if backend == "torchrl-cpu":
        from triton_per.baselines.torchrl_wrapper import TorchRLPER
        return TorchRLPER(capacity, fields, device=device, sampler_device="cpu")
    if backend == "torchrl-cuda":
        from triton_per.baselines.torchrl_wrapper import TorchRLPER
        return TorchRLPER(capacity, fields, device=device, sampler_device=device)
    raise ValueError(backend)


def bench_one(backend, capacity, batch_size, obs_dim, device, iters, warmup):
    fields = {"obs": obs_dim, "action": 8, "reward": 1, "next_obs": obs_dim, "done": 1}
    buf = make_buffer(backend, capacity, fields, device)

    fill = min(capacity, 1_000_000)
    chunk = 100_000
    for i in range(0, fill, chunk):
        n = min(chunk, fill - i)
        buf.add({k: torch.randn(n, d, device=device) for k, d in fields.items()})
    # realistic spread of priorities
    idx = torch.randint(0, buf.size, (fill,), device=device)
    buf.update_priorities(idx, torch.rand(fill, device=device) * 4 + 1e-3)

    td = torch.rand(batch_size, device=device)
    lat = []
    for it in range(warmup + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, indices, _ = buf.sample(batch_size)
        buf.update_priorities(indices, td)
        if device == "cuda":
            torch.cuda.synchronize()
        if it >= warmup:
            lat.append(time.perf_counter() - t0)

    lat = torch.tensor(lat) * 1e3  # ms
    return {
        "backend": backend,
        "capacity": capacity,
        "batch_size": batch_size,
        "obs_dim": obs_dim,
        "mean_ms": lat.mean().item(),
        "p50_ms": lat.median().item(),
        "p95_ms": lat.quantile(0.95).item(),
        "samples_per_sec": batch_size / (lat.mean().item() / 1e3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="triton,triton-unfused,torch,cpu-sumtree,torchrl-cpu,torchrl-cuda")
    ap.add_argument("--capacities", default="100000,1000000,10000000")
    ap.add_argument("--batch-sizes", default="256,1024,4096")
    ap.add_argument("--obs-dims", default="32,256")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/micro.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in args.backends.split(","):
        for cap in map(int, args.capacities.split(",")):
            if backend == "cpu-sumtree" and cap > 1_000_000:
                continue  # host-side fill is prohibitively slow at this size
            for bs in map(int, args.batch_sizes.split(",")):
                for od in map(int, args.obs_dims.split(",")):
                    try:
                        r = bench_one(backend, cap, bs, od, args.device,
                                      args.iters, args.warmup)
                    except Exception as e:  # noqa: record and continue the sweep
                        print(f"SKIP {backend} cap={cap} bs={bs} od={od}: {e}")
                        continue
                    rows.append(r)
                    print({k: (f"{v:.3f}" if isinstance(v, float) else v)
                           for k, v in r.items()})

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
