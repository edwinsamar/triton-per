"""GPU-resident prioritized replay buffer.

Storage is a single packed float32[capacity, row_dim] tensor; fields are views
into row slices, so sampling returns a dict of zero-copy views of the gathered
batch. Priorities live in a two-level structure (flat priorities + per-block
partial sums) that the Triton kernels sample from and update.

Backends:
  "triton"        fused sample+gather+weights kernel (requires CUDA/ROCm GPU)
  "triton-unfused" sampling kernel + PyTorch gather (ablation)
  "torch"         full cumsum + searchsorted, any device (baseline & CPU tests)
"""

from __future__ import annotations

import math

import torch

BLOCK_P = 1024  # priorities per block; keep in sync across kernels


class PERBuffer:
    def __init__(
        self,
        capacity: int,
        fields: dict[str, int],  # name -> flattened per-transition dim
        device: str | torch.device = "cuda",
        alpha: float = 0.6,
        eps: float = 1e-6,
        backend: str = "triton",
    ):
        assert backend in ("triton", "triton-unfused", "torch")
        self.capacity = capacity
        self.fields = dict(fields)
        self.device = torch.device(device)
        self.alpha = alpha
        self.eps = eps
        self.backend = backend

        self.row_dim = sum(fields.values())
        self._offsets: dict[str, tuple[int, int]] = {}
        o = 0
        for name, d in fields.items():
            self._offsets[name] = (o, o + d)
            o += d

        self.num_blocks = math.ceil(capacity / BLOCK_P)
        self.capacity_padded = self.num_blocks * BLOCK_P
        self.log2_blocks = max(1, math.ceil(math.log2(self.num_blocks))) if self.num_blocks > 1 else 0

        self.storage = torch.zeros(capacity, self.row_dim, device=self.device)
        self.priorities = torch.zeros(self.capacity_padded, device=self.device)
        self.block_sums = torch.zeros(self.num_blocks, device=self.device)

        self.size = 0
        self._cursor = 0
        # device scalar (already ^alpha space) so updates never sync the host
        self._max_priority = torch.ones((), device=self.device)
        # atomic delta-updates drift in fp32; refresh block sums exactly
        # every K update calls to bound the error (O(capacity), amortized)
        self._updates_since_refresh = 0
        self._refresh_every = 512

    # ------------------------------------------------------------------ add
    def add(self, batch: dict[str, torch.Tensor]) -> None:
        """Insert a batch of transitions (new entries get max priority)."""
        n = next(iter(batch.values())).shape[0]
        idx = (torch.arange(n, device=self.device) + self._cursor) % self.capacity
        for name, (a, b) in self._offsets.items():
            self.storage[idx, a:b] = batch[name].reshape(n, -1).to(self.storage.dtype)

        self.priorities[idx] = self._max_priority
        touched = torch.unique(idx // BLOCK_P)
        self.block_sums[touched] = (
            self.priorities.view(self.num_blocks, BLOCK_P)[touched].sum(dim=1)
        )
        self._cursor = int((self._cursor + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    # --------------------------------------------------------------- sample
    def sample(self, batch_size: int, beta: float = 0.4):
        """Returns (fields: dict of views, indices, normalized IS weights)."""
        assert self.size > 0
        rand = torch.rand(batch_size, device=self.device)

        if self.backend == "torch":
            idx, w, out = self._sample_torch(rand, beta)
        else:
            idx, w, out = self._sample_triton(rand, beta, batch_size)

        w = w / w.max()
        views = {name: out[:, a:b] for name, (a, b) in self._offsets.items()}
        return views, idx, w

    def _sample_torch(self, rand, beta):
        p = self.priorities[: self.capacity]
        csum = torch.cumsum(p, dim=0)
        total = csum[-1]
        idx = torch.searchsorted(csum, rand * total, right=True)
        idx = idx.clamp_(max=self.size - 1)
        prob = p[idx] / total
        w = (self.size * prob) ** (-beta)
        out = self.storage[idx]
        return idx, w, out

    def _sample_triton(self, rand, beta, batch_size):
        from .kernels.fused import fused_sample_kernel
        from .kernels.sampling import sample_kernel

        # no host sync anywhere: the kernels read the total on-device
        block_cumsum = torch.cumsum(self.block_sums, dim=0)
        idx = torch.empty(batch_size, dtype=torch.int64, device=self.device)
        w = torch.empty(batch_size, device=self.device)

        if self.backend == "triton-unfused":
            sample_kernel[(batch_size,)](
                block_cumsum, self.priorities, rand, idx, w,
                self.num_blocks, self.size, beta,
                LOG2_BLOCKS=self.log2_blocks, BLOCK_P=BLOCK_P,
            )
            out = self.storage[idx]  # separate gather, the ablation target
        else:
            out = torch.empty(batch_size, self.row_dim, device=self.device)
            fused_sample_kernel[(batch_size,)](
                block_cumsum, self.priorities, rand, self.storage, out, idx, w,
                self.num_blocks, self.size, beta, self.row_dim,
                LOG2_BLOCKS=self.log2_blocks, BLOCK_P=BLOCK_P, BLOCK_D=256,
            )
        return idx, w, out

    # --------------------------------------------------------------- update
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        td = td_errors.detach().reshape(-1).abs()
        self._max_priority = torch.maximum(
            self._max_priority, (td.max() + self.eps) ** self.alpha
        )
        if self.backend == "torch":
            new_p = (td + self.eps) ** self.alpha
            self.priorities[indices] = new_p
            touched = torch.unique(indices // BLOCK_P)
            self.block_sums[touched] = (
                self.priorities.view(self.num_blocks, BLOCK_P)[touched].sum(dim=1)
            )
            return

        from .kernels.update import update_kernel

        n = indices.numel()
        grid = (math.ceil(n / 256),)
        update_kernel[grid](
            self.priorities, self.block_sums, indices, td,
            n, self.eps, self.alpha,
            BLOCK_P=BLOCK_P, BLOCK_B=256,
        )
        self._updates_since_refresh += 1
        if self._updates_since_refresh >= self._refresh_every:
            self._updates_since_refresh = 0
            torch.sum(
                self.priorities.view(self.num_blocks, BLOCK_P), dim=1,
                out=self.block_sums,
            )
