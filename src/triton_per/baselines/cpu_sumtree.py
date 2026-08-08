"""Classic CPU SumTree PER (Dopamine-style numpy sum tree): sampling walks
the tree per-element on the host and the batch crosses PCIe both ways.
"""

from __future__ import annotations

import numpy as np
import torch


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[1])

    def update(self, idx: int, p: float) -> None:
        i = idx + self.capacity
        delta = p - self.tree[i]
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def find(self, u: float) -> int:
        i = 1
        while i < self.capacity:
            left = 2 * i
            if u <= self.tree[left]:
                i = left
            else:
                u -= self.tree[left]
                i = left + 1
        return i - self.capacity

    def get(self, idx: int) -> float:
        return float(self.tree[idx + self.capacity])


class CpuSumTreePER:
    """Same interface as triton_per.buffer.PERBuffer, storage on `device`
    (typically CUDA) but priorities and sampling on CPU: the standard setup
    the fused kernels replace."""

    def __init__(self, capacity, fields, device="cuda", alpha=0.6, eps=1e-6, **_):
        self.capacity = capacity
        self.fields = dict(fields)
        self.device = torch.device(device)
        self.alpha = alpha
        self.eps = eps
        self.row_dim = sum(fields.values())
        self._offsets = {}
        o = 0
        for name, d in fields.items():
            self._offsets[name] = (o, o + d)
            o += d
        self.storage = torch.zeros(capacity, self.row_dim, device=self.device)
        self.tree = SumTree(capacity)
        self.size = 0
        self._cursor = 0
        self._max_priority = 1.0

    def add(self, batch):
        n = next(iter(batch.values())).shape[0]
        for k in range(n):
            i = (self._cursor + k) % self.capacity
            self.tree.update(i, self._max_priority)
        idx = (torch.arange(n, device=self.device) + self._cursor) % self.capacity
        for name, (a, b) in self._offsets.items():
            self.storage[idx, a:b] = batch[name].reshape(n, -1).to(self.storage.dtype)
        self._cursor = (self._cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, beta=0.4):
        total = self.tree.total()
        us = np.random.random(batch_size) * total
        idx_np = np.empty(batch_size, dtype=np.int64)
        probs = np.empty(batch_size, dtype=np.float64)
        for j, u in enumerate(us):
            i = min(self.tree.find(u), self.size - 1)
            idx_np[j] = i
            probs[j] = self.tree.get(i) / total
        w = (self.size * probs) ** (-beta)
        w = torch.as_tensor(w / w.max(), dtype=torch.float32, device=self.device)
        idx = torch.as_tensor(idx_np, device=self.device)  # PCIe hop
        out = self.storage[idx]
        views = {name: out[:, a:b] for name, (a, b) in self._offsets.items()}
        return views, idx, w

    def update_priorities(self, indices, td_errors):
        td = td_errors.detach().reshape(-1).abs().cpu().numpy()  # PCIe hop
        idx = indices.cpu().numpy()
        for i, t in zip(idx, td):
            p = (float(t) + self.eps) ** self.alpha
            self.tree.update(int(i), p)
            self._max_priority = max(self._max_priority, p)
