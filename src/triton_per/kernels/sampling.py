"""Unfused prioritized sampling kernel (ablation variant).

Two-level inverse-transform sampling:
  level 1: binary search over the inclusive cumsum of per-block priority sums
  level 2: tl.cumsum over the BLOCK_P priorities inside the chosen block

One program per sample. Returns indices and unnormalized IS weights; the
transition gather is done by the caller with plain PyTorch indexing, which is
exactly what the fused kernel eliminates. Kept for the ablation.
"""

import triton
import triton.language as tl


@triton.jit
def _find_index(
    block_cumsum_ptr,  # float32[num_blocks], inclusive cumsum of block sums
    priorities_ptr,    # float32[capacity_padded]
    u,                 # scalar float in [0, total)
    num_blocks,
    size,              # number of valid entries currently in the buffer
    LOG2_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Shared search routine: returns (index, priority_at_index)."""
    # ---- level 1: branchless binary search for first block with cumsum > u
    pos = 0
    for k in tl.static_range(LOG2_BLOCKS):
        step = 1 << (LOG2_BLOCKS - 1 - k)
        npos = pos + step
        c = tl.load(
            block_cumsum_ptr + npos - 1,
            mask=npos <= num_blocks,
            other=float("inf"),
        )
        pos = tl.where(c <= u, npos, pos)

    prev = tl.load(block_cumsum_ptr + pos - 1, mask=pos > 0, other=0.0)
    residual = u - prev

    # ---- level 2: cumsum inside the block
    offs = tl.arange(0, BLOCK_P)
    pr = tl.load(priorities_ptr + pos * BLOCK_P + offs)
    csum = tl.cumsum(pr, axis=0)
    local = tl.sum((csum <= residual).to(tl.int32), axis=0)
    local = tl.minimum(local, BLOCK_P - 1)

    idx = pos * BLOCK_P + local
    # fp round-off can land in zero-priority padding; clamp to valid range
    idx = tl.minimum(idx, size - 1)

    # load p at the CLAMPED index: every valid slot has p >= eps^alpha > 0,
    # so the IS weight stays finite even on rounding-edge samples
    p = tl.load(priorities_ptr + idx)
    return idx, p


@triton.jit
def sample_kernel(
    block_cumsum_ptr,
    priorities_ptr,
    rand_ptr,       # float32[B] uniform in [0, 1)
    indices_ptr,    # int64[B] out
    weights_ptr,    # float32[B] out, unnormalized (N * P(i))^(-beta)
    num_blocks,
    size,
    beta,
    LOG2_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    j = tl.program_id(0)
    # total priority read on-device: keeps the host sample path sync-free
    total = tl.load(block_cumsum_ptr + num_blocks - 1)
    u = tl.load(rand_ptr + j) * total
    idx, p = _find_index(
        block_cumsum_ptr, priorities_ptr, u, num_blocks, size,
        LOG2_BLOCKS=LOG2_BLOCKS, BLOCK_P=BLOCK_P,
    )
    tl.store(indices_ptr + j, idx.to(tl.int64))
    # w = (size * p / total)^(-beta); p > 0 by construction of the search
    w = tl.exp(-beta * tl.log(size.to(tl.float32) * p / total))
    tl.store(weights_ptr + j, w)
