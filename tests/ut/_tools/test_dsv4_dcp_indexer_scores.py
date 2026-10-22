# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the native-score consumer without loading Torch or an NPU runtime."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_selector(operator):
    source = REPO_ROOT / "vllm_ascend/models/deepseek_v4/indexer.py"
    tree = ast.parse(source.read_text())
    indexer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendIndexerOps")
    method = next(node for node in indexer.body if isinstance(node, ast.FunctionDef) and node.name == "select_topk")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": SimpleNamespace(
            ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_quant_lightning_indexer_v2=operator))
        ),
        "wait_for_device_metadata": Mock(),
        "DeviceMetadataStage": SimpleNamespace(INDEXER=object()),
        "_QLI_RETURN_VALUE_UNSUPPORTED": False,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["select_topk"]


class TestNativeIndexerScores(unittest.TestCase):
    def setUp(self):
        self.indices, self.scores = object(), object()
        self.operator = Mock(return_value=(self.indices, self.scores))
        self.select_topk = load_selector(self.operator)
        self.owner = SimpleNamespace(
            index_topk=512,
            device_operator=SimpleNamespace(
                prepare_dsa_indexer_weights=lambda value: value,
                prepare_dsa_indexer_query_scale=lambda value: value,
                prepare_dsa_indexer_key_scale=lambda value: value,
                get_dsa_indexer_quant_mode=lambda: 2,
            ),
        )
        self.metadata = SimpleNamespace(
            **{
                key: object()
                for key in (
                    "qli_metadata",
                    "qli_cu_seqlens_q",
                    "qli_seqused_k",
                    "qli_cmp_residual_k",
                    "block_table",
                    "dcp_causal_context",
                )
            }
        )

    def call_selector(self, return_values):
        return self.select_topk(
            self.owner,
            object(),
            object(),
            object(),
            object(),
            object(),
            self.metadata,
            return_values=return_values,
        )

    def test_dcp_consumes_native_score_tensor(self):
        indices, scores = self.call_selector(True)
        self.assertIs(indices, self.indices)
        self.assertIs(scores, self.scores)
        self.operator.assert_called_once()
        self.assertEqual(self.operator.call_args.kwargs["return_value"], 1)
        self.assertIs(self.operator.call_args.kwargs["dcp_context"], self.metadata.dcp_causal_context)

    def test_indices_only_call_does_not_request_scores(self):
        self.assertIs(self.call_selector(False), self.indices)
        self.operator.assert_called_once()
        self.assertEqual(self.operator.call_args.kwargs["return_value"], 0)

    def test_unsupported_native_score_output_is_not_silently_recomputed(self):
        self.operator.side_effect = RuntimeError("return_value=1 unsupported")
        with self.assertRaisesRegex(RuntimeError, "return_value=1 unsupported"):
            self.call_selector(True)
        self.operator.assert_called_once()


if __name__ == "__main__":
    unittest.main()
