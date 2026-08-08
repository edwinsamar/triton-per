"""Count CUDA kernel launches per sample()+update_priorities() call, per
backend. Attributes sampler overhead to dispatch count versus kernel time.

  python bench/launch_counts.py
"""

import torch

from bench_sampling import make_buffer


def count_launches(backend, capacity=1_000_000, batch=1024, obs_dim=32):
    fields = {"obs": obs_dim, "action": 8, "reward": 1, "next_obs": obs_dim, "done": 1}
    buf = make_buffer(backend, capacity, fields, "cuda")
    n = min(capacity, 200_000)
    chunk = 100_000
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        buf.add({k: torch.randn(m, d, device="cuda") for k, d in fields.items()})
    td = torch.rand(batch, device="cuda")
    for _ in range(10):  # warmup incl. JIT/autotune
        _, idx, _ = buf.sample(batch)
        buf.update_priorities(idx, td)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(10):
            _, idx, _ = buf.sample(batch)
            buf.update_priorities(idx, td)
        torch.cuda.synchronize()

    kernels = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
    n_launch = sum(e.count for e in kernels) / 10
    t_total = sum(e.self_device_time_total for e in kernels) / 10
    print(f"{backend}: {n_launch:.0f} kernel launches / round trip, "
          f"{t_total/1000:.3f}ms device time")
    return n_launch


if __name__ == "__main__":
    for b in ("triton", "torch", "torchrl-cuda"):
        try:
            count_launches(b)
        except Exception as e:
            print(f"{b}: FAILED {e}")
