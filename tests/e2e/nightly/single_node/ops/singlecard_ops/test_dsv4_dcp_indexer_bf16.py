# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native A2/A3 BF16 indexer scores, causal selection, and graph replay."""

import pytest
import torch
import torch_npu

from vllm_ascend.attention.context_parallel.dsa_dcp import merge_dsa_dcp_indexer_topk
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.fixture(autouse=True)
def require_arch22():
    soc = torch_npu.npu.get_soc_version()
    if not (220 <= soc <= 225 or 250 <= soc <= 256):
        pytest.skip("BF16 score selection requires the A2/A3 indexer kernel")


def local_length(raw_end, size, rank):
    # Enumerate complete groups, independently of the device count formula.
    return sum((j * 4 // 128) % size == rank for j in range(max(raw_end, 0) // 4))


def make_inputs(lengths, query_lengths, size, rank, strided=False):
    storage = 128
    capacity = max(local_length(length, size, rank) for length in lengths)
    columns = max((capacity + storage - 1) // storage, 1)
    blocks = len(lengths) * columns
    step = 2 if strided else 1
    key = torch.ones((blocks * step, storage, 1, 128), dtype=torch.int8, device="npu")[::step]
    scale_pool = torch.empty((blocks * step, storage, 1), dtype=torch.float16, device="npu")
    key_scale = scale_pool[::step]
    # Non-monotonic scales expose an index/value pairing error. Adjacent
    # FP16 scales also collapse into BF16 ties around the selection cutoff.
    scale_cpu = (1 + ((torch.arange(blocks * storage) * 37 + rank * 19) % 1021) / 2048).half()
    key_scale.copy_(scale_cpu.view(blocks, storage, 1))
    rows = sum(query_lengths)
    row_weights = (torch.arange(rows) % 4 + 1).half()
    query = torch.ones((rows, 64, 128), dtype=torch.int8, device="npu")
    weights = row_weights[:, None].expand(rows, 64).contiguous().npu()
    query_scale = torch.ones_like(weights)
    cu_q = [0]
    for length in query_lengths:
        cu_q.append(cu_q[-1] + length)
    inputs = dict(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        topk=512,
        quant_mode=2,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device="npu"),
        seqused_k=torch.empty(len(lengths), dtype=torch.int32, device="npu"),
        cmp_residual_k=torch.empty(len(lengths), dtype=torch.int32, device="npu"),
        block_table=torch.arange(blocks, dtype=torch.int32, device="npu").view(len(lengths), columns),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        dcp_context=torch.tensor([size, rank, 128, *lengths], dtype=torch.int32, device="npu"),
    )
    update_lengths(inputs, lengths, size, rank)
    return inputs, scale_cpu.view(len(lengths), columns * storage), row_weights


def update_lengths(inputs, lengths, size, rank):
    inputs["dcp_context"][3:].copy_(torch.tensor(lengths, dtype=torch.int32))
    inputs["seqused_k"].copy_(torch.tensor([local_length(length, size, rank) for length in lengths]))
    residual = [length % 4 if length and ((length - 1) // 128) % size == rank else 0 for length in lengths]
    inputs["cmp_residual_k"].copy_(torch.tensor(residual))


def run_indexer(inputs, query_lengths, return_value=1):
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        64,
        1,
        128,
        512,
        2,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        seqused_k=inputs["seqused_k"],
        cmp_residual_k=inputs["cmp_residual_k"],
        batch_size=len(query_lengths),
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=inputs["block_table"].shape[1] * 128,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        dcp_context=inputs["dcp_context"],
    )
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(**inputs, metadata=metadata, return_value=return_value)


def check_output(indices, values, scales, weights, lengths, query_lengths, size, rank, return_value=1):
    assert indices.dtype == torch.int32
    assert values.dtype == torch.bfloat16
    assert values.shape == (indices.shape if return_value else (0,))
    indices = indices.cpu().squeeze(1).long()
    values = values.cpu().squeeze(1) if return_value else values.cpu()
    row = 0
    for request, (length, query_length) in enumerate(zip(lengths, query_lengths)):
        for q in range(query_length):
            visible = local_length(length - query_length + q + 1, size, rank)
            valid = indices[row] >= 0
            selected = indices[row][valid]
            assert selected.numel() == min(visible, 512)
            assert selected.unique().numel() == selected.numel()
            assert (selected < visible).all()
            # Q=K=1: ReLU(dot/1024)=1/8 exactly, so 64 heads
            # weighted by w give the FP32 final score 8*w*k_scale.
            scores = 8 * weights[row].float() * scales[request, :visible].float()
            if return_value:
                scores = scores.bfloat16().float()
                assert torch.isneginf(values[row][~valid]).all()
                torch.testing.assert_close(values[row][valid].float(), scores[selected], rtol=0, atol=0)
            if selected.numel():
                cutoff = scores.topk(selected.numel()).values[-1]
                assert (scores[selected] >= cutoff).all()
                mandatory = torch.where(scores > cutoff)[0]
                assert torch.isin(mandatory, selected).all()
            row += 1


@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    ("lengths", "query_lengths"),
    [
        ((8191, 129, 0), (8, 4, 0)),
        ((32768, 1029, 0), (8, 4, 0)),
        ((262144, 129, 0), (8, 1, 0)),
        ((8192, 1029, 0), (1024, 129, 0)),
    ],
)
@torch.inference_mode()
def test_dcp_indexer_native_bf16_scores(rank, strided, lengths, query_lengths):
    inputs, scales, weights = make_inputs(lengths, query_lengths, 8, rank, strided)
    indices, values = run_indexer(inputs, query_lengths)
    check_output(indices, values, scales, weights, lengths, query_lengths, 8, rank)


@pytest.mark.parametrize("return_value", [0, 1])
@torch.inference_mode()
def test_indexer_score_mode_without_dcp(return_value):
    lengths, query_lengths = (16384,), (8,)
    inputs, scales, weights = make_inputs(lengths, query_lengths, 1, 0)
    indices, values = run_indexer(inputs, query_lengths, return_value)
    check_output(indices, values, scales, weights, lengths, query_lengths, 1, 0, return_value)


@pytest.mark.parametrize("rank", [0, 7])
@torch.inference_mode()
def test_dcp_indexer_bf16_graph_replay(rank):
    lengths, query_lengths = (131200,), (8,)
    inputs, scales, weights = make_inputs(lengths, query_lengths, 8, rank)
    run_indexer(inputs, query_lengths)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        indices, values = run_indexer(inputs, query_lengths)
    for length in (131071, 131072, 131073, 131200):
        update_lengths(inputs, (length,), 8, rank)
        graph.replay()
        check_output(indices, values, scales, weights, (length,), query_lengths, 8, rank)


def test_dcp_indexer_bf16_meta():
    query = torch.empty((8, 64, 128), dtype=torch.int8, device="meta")
    key = torch.empty((4, 128, 1, 128), dtype=torch.int8, device="meta")
    weights = torch.empty((8, 64), dtype=torch.float16, device="meta")
    scales = torch.empty((4, 128, 1), dtype=torch.float16, device="meta")
    for return_value in (0, 1):
        indices, values = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query,
            key,
            weights,
            weights,
            scales,
            512,
            2,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=return_value,
        )
        assert indices.shape == (8, 1, 512)
        assert indices.dtype == torch.int32
        assert values.shape == ((8, 1, 512) if return_value else (0,))
        assert values.dtype == torch.bfloat16


@torch.inference_mode()
def test_dcp_indexer_bf16_global_merge():
    lengths, query_lengths = (32768,), (8,)
    rows, size, topk = 8, 8, 512
    payloads, local_outputs = [], []
    golden = torch.empty((rows, lengths[0] // 4), dtype=torch.float32)
    masks = []
    for rank in range(size):
        inputs, scales, weights = make_inputs(lengths, query_lengths, size, rank)
        indices, values = run_indexer(inputs, query_lengths)
        check_output(indices, values, scales, weights, lengths, query_lengths, size, rank)
        local_outputs.append((indices, values))
        local_idx = indices.cpu().squeeze(1).long()
        # Independent inverse of 128-raw-token striping for C4 entries.
        global_idx = (local_idx // 32 * size + rank) * 32 + local_idx % 32
        visible = torch.tensor([local_length(lengths[0] - rows + q + 1, size, rank) for q in range(rows)])
        valid = (local_idx >= 0) & (local_idx < visible[:, None])
        masks.append(valid.npu())
        packed = torch.cat(
            [
                global_idx.int().masked_fill(~valid, -1),
                values.cpu().squeeze(1).masked_fill(~valid, float("-inf")).contiguous().view(torch.int32),
            ],
            dim=-1,
        )
        payloads.append(packed)
        local_pos = torch.arange(scales.shape[1])
        global_pos = (local_pos // 32 * size + rank) * 32 + local_pos % 32
        golden[:, global_pos] = (8 * weights.float()[:, None] * scales[0].float()).bfloat16().float()

    class GatheredCandidates:
        world_size = size

        def __init__(self):
            self.payload = torch.stack(payloads).npu()

        def all_gather(self, tensor, dim):
            assert tensor.dtype == torch.int32
            assert tensor.shape == (1, rows, topk * 3 // 2)
            assert tensor.numel() * tensor.element_size() == rows * topk * 6
            assert dim == 0
            return self.payload

    group = GatheredCandidates()

    def run_merge():
        return merge_dsa_dcp_indexer_topk(
            *local_outputs[0],
            valid_mask=masks[0],
            dcp_group=group,
            dcp_rank=0,
            interleave_size=128,
            compress_ratio=4,
            storage_block_size=128,
        )

    merged = run_merge().cpu().squeeze(1).long()
    for row in range(rows):
        visible = (lengths[0] - rows + row + 1) // 4
        scores = golden[row, :visible]
        selected = merged[row]
        assert selected.unique().numel() == topk
        assert ((selected >= 0) & (selected < visible)).all()
        cutoff = scores.topk(topk).values[-1]
        assert (scores[selected] >= cutoff).all()
        assert torch.isin(torch.where(scores > cutoff)[0], selected).all()

    # Capture the actual unpack + merge, then change only the fixed wire
    # buffer. Empty candidate rows must remain -1 on graph replay too.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        graph_output = run_merge()
    original_payload = group.payload.clone()
    invalid_scores = torch.full((size, rows, topk), float("-inf"), dtype=torch.bfloat16, device="npu")
    group.payload[..., topk:].copy_(invalid_scores.view(torch.int32))
    graph.replay()
    assert (graph_output == -1).all()
    group.payload.copy_(original_payload)
    graph.replay()
    torch.testing.assert_close(graph_output.cpu().squeeze(1).long(), merged, rtol=0, atol=0)
