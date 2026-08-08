"""Fused kernel: priority sampling, transition gather, and IS weights in one
kernel.

Compared to the unfused variant (kernels/sampling.py + PyTorch fancy indexing),
this saves the intermediate index materialization, the separate gather kernel
launch(es), and one full read of the index tensor. Transitions are stored as
packed rows (float32[capacity, row_dim]); field views are sliced by the caller.
"""

import triton
import triton.language as tl

from .sampling import _find_index


@triton.jit
def fused_sample_kernel(
    block_cumsum_ptr,
    priorities_ptr,
    rand_ptr,        # float32[B] uniform in [0, 1)
    storage_ptr,     # float32[capacity, row_dim] packed transitions
    out_ptr,         # float32[B, row_dim] sampled batch
    indices_ptr,     # int64[B] out (needed later for the priority update)
    weights_ptr,     # float32[B] out, unnormalized (N * P(i))^(-beta)
    num_blocks,
    size,
    beta,
    row_dim,
    LOG2_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
    w = tl.exp(-beta * tl.log(size * p / total))
    tl.store(weights_ptr + j, w)

    # ---- fused gather: copy storage[idx, :] -> out[j, :]
    src = storage_ptr + idx.to(tl.int64) * row_dim
    dst = out_ptr + j.to(tl.int64) * row_dim
    for d0 in range(0, row_dim, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        m = offs < row_dim
        vals = tl.load(src + offs, mask=m)
        tl.store(dst + offs, vals, mask=m)
