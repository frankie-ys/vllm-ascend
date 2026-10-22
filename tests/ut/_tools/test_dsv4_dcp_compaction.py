# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check remap/compactor wiring on CPU; NPU tests execute the Triton kernels."""

import ast
import random
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def cpu_compact_owned_indices(owned, local, output):
    # CPU oracle substituted at the device-kernel boundary, not a production
    # fallback. The NPU suite exercises the real two-kernel compactor.
    rows = output.reshape_as(local)
    rows.fill_(-1)
    for row in range(local.shape[0]):
        selected = local[row, owned[row]]
        rows[row, : selected.numel()] = selected


def load_remap():
    source = REPO_ROOT / "vllm_ascend/attention/context_parallel/dsa_dcp.py"
    names = {"_compressed_stripe_len", "dsa_dcp_global_cmp_to_local", "remap_dsa_dcp_topk_indices"}
    functions = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch, "compact_owned_indices": cpu_compact_owned_indices}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["remap_dsa_dcp_topk_indices"]


def reference_remap(indices, lengths, request_ids, size, rank):
    # C4 / interleave=128: each rank owns a stripe of 32 compressed entries.
    # Walk the selection in its original order, without sorting or float casts.
    topk = indices.shape[-1]
    output = []
    for row, request in zip(indices.reshape(-1, topk).tolist(), request_ids.tolist()):
        local = []
        for index in row:
            stripe, offset = divmod(index, 32)
            if 0 <= index < lengths[request] and stripe % size == rank:
                local.append(stripe // size * 32 + offset)
        output.append(local + [-1] * (topk - len(local)))
    return torch.tensor(output, dtype=indices.dtype).reshape(indices.shape)


def make_case(topk, size, rank):
    generator = random.Random(512 + topk + size + rank)
    lengths = torch.tensor([0, 1, 33, 257, 65536], dtype=torch.int32)
    request_ids = torch.tensor([0, 1, 2, 3, 4, 4, 4, 4], dtype=torch.int64)
    rows = [generator.sample(range(65540), topk) for _ in range(8)]
    for row in rows:
        row[:10] = [-1, 0, 31, 32, 33, 127, 128, 129, 65535, 65536]
    # Explicit empty, full and alternating ownership rows exercise both tails.
    rows[4] = [-1] * topk
    owned = [(position // 32 * size + rank) * 32 + position % 32 for position in range(topk)]
    rows[5] = owned[::-1]
    rows[6] = [owned[i] if i % 2 else -1 for i in range(topk)]
    rows[7] = [owned[-1]] + [-1] * (topk - 1)
    indices = torch.tensor(rows, dtype=torch.int32).unsqueeze(1)
    return indices, lengths, request_ids


class TestDCPCompaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.remap = staticmethod(load_remap())

    def check_case(self, indices, lengths, requests, size, rank):
        actual = self.remap(
            indices,
            dcp_size=size,
            dcp_rank=rank,
            interleave_size=128,
            compress_ratio=4,
            storage_block_size=128,
            global_cmp_seq_lens=lengths,
            req_indices=requests,
        )
        expected = reference_remap(indices, lengths.tolist(), requests, size, rank)
        self.assertEqual(actual.shape, indices.shape)
        self.assertEqual(actual.dtype, indices.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mixed_requests_and_owned_prefix(self):
        for topk in (512, 1024):
            for size in (4, 8):
                for rank in range(size):
                    with self.subTest(topk=topk, size=size, rank=rank):
                        self.check_case(*make_case(topk, size, rank), size, rank)

    def test_noncontiguous_int64_indices(self):
        for topk in (512, 1024):
            indices, lengths, requests = make_case(topk, 4, 3)
            storage = torch.empty((*indices.shape[:-1], topk * 2), dtype=torch.int64)
            storage[..., ::2] = indices
            self.check_case(storage[..., ::2], lengths, requests, 4, 3)

    def test_large_kv_indices_keep_low_bits(self):
        # Neither the mapping nor the fused compactor may round KV indices
        # through FP32, including the scalar extraction in the Triton kernel.
        first = (1 << 27) + 1
        indices = torch.full((1, 1, 512), -1, dtype=torch.int32)
        indices[0, 0, :4] = torch.tensor([first, first + 32, first + 2, first + 128])
        self.check_case(indices, torch.tensor([1 << 28], dtype=torch.int32), torch.tensor([0]), 4, 0)


if __name__ == "__main__":
    unittest.main()
