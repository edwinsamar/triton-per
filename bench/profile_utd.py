"""What fraction of a high-UTD training step is replay sampling + priority
updates?

Simulates the update phase of SAC (MLP actor + 2 critics, realistic sizes) with
a real buffer backend, at UTD ratios 1..32, and reports the replay share of
step wall-clock. Also exports a torch.profiler chrome trace that makes the
GPU-idle-during-CPU-sampling gap visible.

  python bench/profile_utd.py --backend cpu-sumtree --utd 32 --trace results/trace.json
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from bench_sampling import make_buffer  # same dir


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu-sumtree")
    ap.add_argument("--utd", type=int, default=20)
    ap.add_argument("--obs-dim", type=int, default=32)
    ap.add_argument("--act-dim", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--capacity", type=int, default=1_000_000)
    ap.add_argument("--num-q", type=int, default=2, help="critic count (10 = REDQ-style)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trace", default=None, help="chrome trace output path")
    ap.add_argument("--csv-out", default=None, help="append result row to this CSV")
    args = ap.parse_args()

    dev = args.device
    fields = {"obs": args.obs_dim, "action": args.act_dim, "reward": 1,
              "next_obs": args.obs_dim, "done": 1}
    buf = make_buffer(args.backend, args.capacity, fields, dev)
    n = min(args.capacity, 200_000)
    buf.add({k: torch.randn(n, d, device=dev) for k, d in fields.items()})

    actor = mlp(args.obs_dim, args.act_dim).to(dev)
    qs = [mlp(args.obs_dim + args.act_dim, 1).to(dev) for _ in range(args.num_q)]
    params = [p for q in qs for p in q.parameters()] + list(actor.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)

    def grad_step(batch, idx, w):
        obs, act = batch["obs"], batch["action"]
        qin = torch.cat([obs, act], dim=-1)
        r = batch["reward"].squeeze(-1)
        tds = [(q(qin).squeeze(-1) - r) for q in qs]
        loss = sum((w.squeeze(-1) * t.pow(2)).mean() for t in tds)
        loss = loss + actor(obs).pow(2).mean() * 1e-3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return tds[0].detach().abs()

    replay_t, learn_t = 0.0, 0.0
    prof = None
    if args.trace:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=20),
        )
        prof.start()

    for step in range(args.steps):
        for _ in range(args.utd):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch, idx, w = buf.sample(args.batch_size)
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            td = grad_step(batch, idx, w)
            if dev == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            buf.update_priorities(idx, td)
            if dev == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            if step >= 10:
                replay_t += (t1 - t0) + (t3 - t2)
                learn_t += t2 - t1
        if prof:
            prof.step()

    if prof:
        prof.stop()
        prof.export_chrome_trace(args.trace)
        print(f"trace -> {args.trace}")

    share = replay_t / (replay_t + learn_t)
    print(f"backend={args.backend} utd={args.utd} batch={args.batch_size}")
    print(f"replay {replay_t:.2f}s | learn {learn_t:.2f}s | replay share = {share:.1%}")
    if args.csv_out:
        import csv
        from pathlib import Path

        p = Path(args.csv_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with p.open("a", newline="") as f:
            wtr = csv.writer(f)
            if new:
                wtr.writerow(["backend", "utd", "batch_size", "num_q",
                              "replay_s", "learn_s", "replay_share"])
            wtr.writerow([args.backend, args.utd, args.batch_size, args.num_q,
                          round(replay_t, 3), round(learn_t, 3), round(share, 4)])


if __name__ == "__main__":
    main()
