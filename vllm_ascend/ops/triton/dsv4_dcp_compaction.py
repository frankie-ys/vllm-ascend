# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact DSV4 rank-owned local KV indices without materializing holes or sorting."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_compact_gather_kernel
from vllm_ascend.ops.triton.triton_utils import get_element

COMPACTION_BLOCK_SIZE = 128


@triton.jit(do_not_specialize=["topk_count"])
def _compact_owned_chunks(
    owned_ptr,
    local_ptr,
    chunk_out_ptr,
    chunk_count_ptr,
    out_ptr,
    topk_count,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < topk_count
    owned = tl.load(owned_ptr + row * topk_count + offsets, mask=in_bounds, other=0).to(tl.int32)
    local = tl.load(local_ptr + row * topk_count + offsets, mask=in_bounds, other=0)
    local = local.to(chunk_out_ptr.dtype.element_ty)

    # Match the existing A2 SFA compactor: prefill each disjoint output
    # region here, before a separate launch merges the compacted chunks.
    # This avoids both cross-program races and the known tl.cumsum issues.
    tl.store(out_ptr + row * topk_count + offsets, -1, mask=in_bounds)
    chunk_out = chunk_out_ptr + (row * NUM_CHUNKS + chunk) * BLOCK
    count = 0
    for position in range(BLOCK):
        if get_element(owned, (position,)) != 0:
            tl.store(chunk_out + count, get_element(local, (position,)))
            count += 1
    tl.store(chunk_count_ptr + row * NUM_CHUNKS + chunk, count)


def compact_owned_indices(owned: torch.Tensor, local: torch.Tensor, output: torch.Tensor) -> None:
    """Write owned local indices in input order, with an all-(-1) tail.

    The DSV4 caller supplies contiguous, equally sized owned/local/output
    tensors. ``owned`` already includes sequence validity and rank ownership;
    ``output`` supplies the original selection's shape and integer dtype.
    No device value is read by the host, including during ACL graph capture.
    """
    topk = output.shape[-1]
    rows = output.numel() // topk
    block = min(COMPACTION_BLOCK_SIZE, next_power_of_2(topk))
    chunks = triton.cdiv(topk, block)
    chunk_out = torch.empty((rows * chunks, block), dtype=output.dtype, device=output.device)
    counts = torch.empty((rows, chunks), dtype=torch.int32, device=output.device)
    _compact_owned_chunks[(chunks, rows)](
        owned,
        local,
        chunk_out,
        counts,
        output,
        topk,
        BLOCK=block,
        NUM_CHUNKS=chunks,
        multibuffer=False,
    )
    remap_sparse_indices_compact_gather_kernel[(rows,)](
        chunk_out,
        counts,
        output,
        topk,
        BLOCK=block,
        NUM_CHUNKS=chunks,
        multibuffer=False,
    )
