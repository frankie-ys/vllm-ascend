# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check DCP compaction on NPU, including refreshed inputs under ACL replay."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut._tools.test_dsv4_dcp_compaction import make_case, reference_remap
from vllm_ascend.attention.context_parallel.dsa_dcp import remap_dsa_dcp_topk_indices
from vllm_ascend.ops.triton.dsv4_dcp_compaction import compact_owned_indices


def compact_case(topk):
    counts = [0, 1, 127, 128, 129, 255, 256, 257, topk - 1, topk]
    positions = torch.arange(topk)
    owned = torch.stack([positions * 37 % topk < count for count in counts])
    local = ((1 << 27) + 1 + torch.arange(len(counts) * topk)).reshape(len(counts), topk)
    expected = torch.full(local.shape, -1, dtype=torch.int32)
    for row, count in enumerate(counts):
        expected[row, :count] = local[row, owned[row]]
    return owned, local, expected


@pytest.mark.parametrize("topk", [512, 1024])
@torch.inference_mode()
def test_owned_and_local_compact_without_holes(topk):
    owned, local, expected = compact_case(topk)
    output = torch.empty(expected.shape, dtype=expected.dtype, device="npu")
    compact_owned_indices(owned.npu(), local.npu(), output)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("topk", [512, 1024])
@torch.inference_mode()
def test_compactor_repeated_graph_replay_refreshes_owned_and_local(topk):
    owned, local, expected = compact_case(topk)
    device_owned, device_local = owned.npu(), local.npu()
    output = torch.empty(expected.shape, dtype=expected.dtype, device="npu")
    for _ in range(3):
        compact_owned_indices(device_owned, device_local, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        compact_owned_indices(device_owned, device_local, output)

    for _ in range(32):
        graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    device_owned.zero_()
    for _ in range(32):
        graph.replay()
    torch.testing.assert_close(output.cpu(), torch.full_like(expected, -1), rtol=0, atol=0)

    device_local.add_(17)
    device_owned.copy_(owned)
    for _ in range(32):
        graph.replay()
    torch.testing.assert_close(output.cpu(), torch.where(expected >= 0, expected + 17, -1), rtol=0, atol=0)

    device_owned.fill_(True)
    for _ in range(32):
        graph.replay()
    torch.testing.assert_close(output.cpu(), (local + 17).int(), rtol=0, atol=0)


@pytest.mark.parametrize("topk", [512, 1024])
@pytest.mark.parametrize("size,rank", [(4, 0), (4, 3), (8, 0), (8, 7)])
@torch.inference_mode()
def test_compaction_matches_scalar_reference(topk, size, rank):
    indices, lengths, requests = make_case(topk, size, rank)
    expected = reference_remap(indices, lengths.tolist(), requests, size, rank)
    actual = remap_dsa_dcp_topk_indices(
        indices.npu(),
        dcp_size=size,
        dcp_rank=rank,
        interleave_size=128,
        compress_ratio=4,
        storage_block_size=128,
        global_cmp_seq_lens=lengths.npu(),
        req_indices=requests.npu(),
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("topk", [512, 1024])
@torch.inference_mode()
def test_compaction_graph_refreshes_selection_and_lengths(topk):
    indices, lengths, requests = make_case(topk, 4, 0)
    expected = reference_remap(indices, lengths.tolist(), requests, 4, 0)
    device_indices, device_lengths, device_requests = indices.npu(), lengths.npu(), requests.npu()

    def run():
        return remap_dsa_dcp_topk_indices(
            device_indices,
            dcp_size=4,
            dcp_rank=0,
            interleave_size=128,
            compress_ratio=4,
            storage_block_size=128,
            global_cmp_seq_lens=device_lengths,
            req_indices=device_requests,
        )

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()

    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    device_indices.fill_(-1)
    graph.replay()
    torch.testing.assert_close(output.cpu(), torch.full_like(indices, -1), rtol=0, atol=0)

    device_indices.copy_(indices)
    device_lengths.zero_()
    graph.replay()
    torch.testing.assert_close(output.cpu(), torch.full_like(indices, -1), rtol=0, atol=0)

    device_lengths.copy_(lengths)
    device_requests.fill_(0)
    graph.replay()
    torch.testing.assert_close(output.cpu(), torch.full_like(indices, -1), rtol=0, atol=0)

    device_requests.copy_(requests)
    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
