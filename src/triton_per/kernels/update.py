"""Priority update kernel.

Applies p_i <- (|td_i| + eps)^alpha for a batch of indices and keeps the
per-block partial sums consistent. Duplicate indices within a batch are handled
correctly: atomic_xchg serializes the swaps, so the sum of (new - old) deltas
telescopes to a consistent block sum regardless of ordering. No host-side
locks are involved, only hardware atomics.
"""

import triton
import triton.language as tl


@triton.jit
def update_kernel(
    priorities_ptr,   # float32[capacity_padded]
    block_sums_ptr,   # float32[num_blocks]
    indices_ptr,      # int64[B]
    td_ptr,           # float32[B] TD errors (sign irrelevant)
    n,
    eps,
    alpha,
    BLOCK_P: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    m = offs < n

    idx = tl.load(indices_ptr + offs, mask=m, other=0)
    td = tl.load(td_ptr + offs, mask=m, other=0.0)
    new_p = tl.exp(alpha * tl.log(tl.abs(td) + eps))

    old_p = tl.atomic_xchg(priorities_ptr + idx, new_p, mask=m)
    tl.atomic_add(block_sums_ptr + idx // BLOCK_P, new_p - old_p, mask=m)
