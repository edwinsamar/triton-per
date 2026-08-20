# triton-per: Fused Prioritized Replay Sampling in OpenAI Triton

GPU-resident prioritized experience replay (PER) where **sampling, transition
gather, and importance-sampling weights run in one fused Triton kernel**,
eliminating the CPU sum-tree round trip that stalls high update-to-data (UTD)
off-policy RL training.

> **Paper artifact:** all numbers reported in the accompanying paper,
> *Fused GPU Sampling for Prioritized Replay: Removing the Bottleneck in
> High-UTD Reinforcement Learning* (under review), were produced with the
> code released as [`v0.1.0`](../../releases/tag/v0.1.0).

## Design

- Priorities in a **two-level structure**: flat `float32[capacity]` priority
  array + per-block (1024) partial sums. Sampling = binary search over the
  block-sum cumsum, then an in-kernel `tl.cumsum` inside one block. Priority
  updates are hardware atomics (`atomic_xchg` + `atomic_add` on block sums);
  no host-side tree, correct under duplicate indices.
- Transitions stored **packed** (`float32[capacity, row_dim]`); the fused
  kernel writes the sampled batch directly, fields come back as zero-copy views.
- Interchangeable backends: `triton` (fused), `triton-unfused` (kernel sample +
  PyTorch gather), `torch` (naive GPU cumsum + searchsorted). Baselines under
  `src/triton_per/baselines/`: classic CPU SumTree, torchrl adapters (CPU and
  CUDA priority trees).

## Install and test

Requires an NVIDIA/AMD GPU for the Triton backends; the `torch` backend and
its tests run on any device.

```bash
pip install -e ".[dev,rl]"
python -m pytest tests/ -v          # chi-square distribution + semantics tests
```

## Usage

```python
import torch
from triton_per import PERBuffer

buf = PERBuffer(1_000_000, fields={"obs": 32, "action": 8, "reward": 1,
                                   "next_obs": 32, "done": 1}, device="cuda")
buf.add({...})                                # dict of [n, dim] tensors
batch, idx, w = buf.sample(256, beta=0.4)     # fused sample+gather+weights
buf.update_priorities(idx, td_errors)         # atomic priority update
```

## Benchmarks

```bash
python bench/bench_sampling.py --out results/micro.csv   # backend sweep
python bench/profile_utd.py --backend cpu-sumtree --utd 32 --trace results/trace.json
```

`bench/profile_utd.py` measures the replay share of a SAC-style training step
and exports a chrome trace (open in https://ui.perfetto.dev). Slurm templates
for V100/A100/H100 nodes are under `slurm/`.

## Layout

```
src/triton_per/kernels/    sampling.py (unfused), fused.py, update.py
src/triton_per/buffer.py   PERBuffer (all backends)
src/triton_per/baselines/  CPU SumTree, torchrl adapters
tests/                     distributional correctness (chi-square), semantics
bench/                     microbenchmark sweep, UTD-share profiler
slurm/                     sbatch templates (V100/A100/H100)
```
