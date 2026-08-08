"""Thin adapter around torchrl's PrioritizedSampler so the benchmark harness
can drive it with the same interface as PERBuffer.

Covers both the CPU-tree variant and the CUDA-tree variant (PrioritizedSampler
with device="cuda", which requires torchrl's C++/CUDA extension).
"""

from __future__ import annotations

import torch


class TorchRLPER:
    def __init__(self, capacity, fields, device="cuda", alpha=0.6, eps=1e-6,
                 sampler_device="cpu", **_):
        from torchrl.data.replay_buffers import (  # noqa: deferred import
            LazyTensorStorage, PrioritizedSampler, ReplayBuffer,
        )
        from tensordict import TensorDict

        self._TensorDict = TensorDict
        self.device = torch.device(device)
        self.fields = dict(fields)
        self.eps = eps
        self.capacity = capacity
        try:
            sampler = PrioritizedSampler(
                max_capacity=capacity, alpha=alpha, beta=0.4, eps=eps,
                device=sampler_device,
            )
        except TypeError:
            # older torchrl without device kwarg -> CPU tree only
            if str(sampler_device) != "cpu":
                raise
            sampler = PrioritizedSampler(
                max_capacity=capacity, alpha=alpha, beta=0.4, eps=eps,
            )
        self.rb = ReplayBuffer(
            storage=LazyTensorStorage(capacity, device=self.device),
            sampler=sampler,
            batch_size=None,
        )
        self.size = 0

    def add(self, batch):
        n = next(iter(batch.values())).shape[0]
        td = self._TensorDict(
            {k: v.reshape(n, -1).to(self.device) for k, v in batch.items()},
            batch_size=[n],
        )
        self.rb.extend(td)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, beta=0.4):
        td, info = self.rb.sample(batch_size, return_info=True)
        views = {k: td[k] for k in self.fields}
        # torchrl renamed "_weight" -> "priority_weight" (0.13)
        w = info.get("priority_weight", info.get("_weight"))
        return views, info["index"], w

    def update_priorities(self, indices, td_errors):
        self.rb.update_priority(indices, td_errors.detach().reshape(-1).abs() + self.eps)
