# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production candidate wire format on CPU, without vLLM/NPU imports."""

import ast
import struct
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_merge():
    source = REPO_ROOT / "vllm_ascend/attention/context_parallel/dsa_dcp.py"
    names = {"_compressed_stripe_len", "dsa_dcp_local_cmp_to_global", "merge_dsa_dcp_indexer_topk"}
    functions = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["merge_dsa_dcp_indexer_topk"]


def reference_packet(indices, scores, mask, size, rank):
    # Encode bytes independently of Torch's dtype reinterpretation.
    rows, _, topk = indices.shape
    packets, candidates = [], []
    for row in range(rows):
        ids, bits, valid_candidates = [], [], []
        for slot in range(topk):
            local = indices[row, 0, slot].item()
            valid = local >= 0 and mask[row, slot].item()
            global_id = (local // 32 * size + rank) * 32 + local % 32 if valid else -1
            score = scores[row, 0, slot].item() if valid else float("-inf")
            ids.append(global_id)
            bits.append(struct.pack("<f", score)[2:])
            if valid:
                valid_candidates.append((global_id, score))
        words = [struct.unpack("<i", bits[i] + bits[i + 1])[0] for i in range(0, topk, 2)]
        packets.append(ids + words)
        candidates.append(valid_candidates)
    return torch.tensor(packets, dtype=torch.int32), candidates


class GatheredCandidates:
    def __init__(self, test, payload, rank):
        self.world_size = payload.shape[0]
        self.test, self.payload, self.rank = test, payload, rank
        self.calls = 0

    def all_gather(self, tensor, dim):
        self.calls += 1
        self.test.assertEqual(dim, 0)
        self.test.assertEqual(tensor.dtype, torch.int32)
        self.test.assertEqual(tensor.shape, self.payload[self.rank : self.rank + 1].shape)
        self.test.assertTrue(tensor.is_contiguous())
        torch.testing.assert_close(tensor, self.payload[self.rank : self.rank + 1], rtol=0, atol=0)
        return self.payload


class TestPackedCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merge = staticmethod(load_merge())

    def check_merge(self, indices, scores, masks):
        size, rows, _, topk = indices.shape
        encoded = [reference_packet(indices[r], scores[r], masks[r], size, r) for r in range(size)]
        payload = torch.stack([packet for packet, _ in encoded])
        self.assertEqual(payload.shape, (size, rows, topk * 3 // 2))
        self.assertEqual(payload.numel() * payload.element_size(), size * rows * topk * 6)
        for rank in (0, size - 1):
            group = GatheredCandidates(self, payload, rank)
            actual = self.merge(
                indices[rank],
                scores[rank],
                valid_mask=masks[rank],
                dcp_group=group,
                dcp_rank=rank,
                interleave_size=128,
                compress_ratio=4,
                storage_block_size=128,
            )
            self.assertEqual(group.calls, 1)
            self.assertEqual(actual.shape, (rows, 1, topk))
            self.assertEqual(actual.dtype, torch.int32)
            for row in range(rows):
                # Python's stable sort is an independent oracle for the
                # rank-major tie policy, including partially filled rows.
                candidates = [entry for _, per_row in encoded for entry in per_row[row]]
                candidates.sort(key=lambda entry: -entry[1])
                expected = [index for index, _ in candidates[:topk]]
                expected += [-1] * (topk - len(expected))
                self.assertEqual(actual[row, 0].tolist(), expected)

    def test_partial_and_empty_shards_keep_fixed_payload(self):
        topk, size = 512, 8
        counts = (0, 1, 100, 255, 256, 511, 512)
        rows = len(counts)
        indices = torch.arange(topk, dtype=torch.int32).view(1, 1, 1, topk).expand(size, rows, 1, topk).clone()
        scores = torch.empty((size, rows, 1, topk), dtype=torch.bfloat16)
        masks = torch.empty((size, rows, topk), dtype=torch.bool)
        for rank in range(size):
            for row, count in enumerate(counts):
                if row == 2 and rank == 0:
                    count = 0
                masks[rank, row] = torch.arange(topk) < count
                scores[rank, row, 0] = ((torch.arange(topk) * 13 % 97) / 8 + rank / 32).bfloat16()
                scores[rank, row, 0, count:] = 10000  # invalid canaries must lose
        self.check_merge(indices, scores, masks)

    def test_negative_index_is_masked_even_if_length_mask_is_true(self):
        indices = torch.full((8, 1, 1, 512), -1, dtype=torch.int32)
        scores = torch.full(indices.shape, 10000, dtype=torch.bfloat16)
        masks = torch.ones((8, 1, 512), dtype=torch.bool)
        self.check_merge(indices, scores, masks)

    def test_indices_above_fp32_exact_integer_range_remain_exact(self):
        indices = torch.full((8, 1, 1, 512), -1, dtype=torch.int32)
        indices[..., 0] = (1 << 21) + 1  # global position exceeds 2**24
        scores = torch.ones(indices.shape, dtype=torch.bfloat16)
        masks = torch.ones((8, 1, 512), dtype=torch.bool)
        self.check_merge(indices, scores, masks)

    def test_signed_zero_and_negative_score_bits_remain_exact(self):
        indices = torch.full((8, 1, 1, 512), -1, dtype=torch.int32)
        indices[..., :4] = torch.arange(4, dtype=torch.int32)
        scores = torch.zeros(indices.shape, dtype=torch.bfloat16)
        scores[..., :4] = torch.tensor([0.0, -0.0, -1.5, -2.0], dtype=torch.bfloat16)
        masks = torch.ones((8, 1, 512), dtype=torch.bool)
        self.check_merge(indices, scores, masks)


if __name__ == "__main__":
    unittest.main()
