#!/usr/bin/env python
"""Runtime checks for CoT-VLA action-path paper alignment.

This script imports the real project utilities and uses torch. It is intended
for the server VILA-U environment, where torch is available, and complements
``scripts/check_action_paper_alignment.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vila_u.constants import ACTION_NUM_BINS
from vila_u.data.libero_dataset_v2 import LiberoGoalDataset
from vila_u.train.train_action_prediction_main import select_action_slot_token_ids
from vila_u.utils.action_tokenizer import (
    build_typed_action_slot_token_ids,
    compute_percentile_action_bin_edges,
    discretize_actions,
    select_action_token_ids,
    undiscretize_action_bins,
)
from vila_u.utils.hybrid_attention import build_hybrid_attention_mask
from vila_u.utils.libero_image import rotate_libero_image_180


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


class DummyTokenizer:
    def __init__(self, vocab_size: int = 32000, added_visual_tokens: int = 4):
        self.vocab_size = vocab_size
        self.unk_token_id = 0
        self.all_special_ids = [0, 1, 2]
        self._vocab = {f"tok_{idx}": idx for idx in range(vocab_size + added_visual_tokens)}
        self._added_vocab = {
            "<im_start>": vocab_size,
            "<im_end>": vocab_size + 1,
            "<vi_start>": vocab_size + 2,
            "<vi_end>": vocab_size + 3,
        }

    def get_vocab(self):
        return self._vocab

    def get_added_vocab(self):
        return self._added_vocab


def check_openvla_token_ids() -> CheckResult:
    tokenizer = DummyTokenizer()
    action_token_ids = select_action_token_ids(tokenizer, num_bins=ACTION_NUM_BINS)
    expected = list(range(31744, 32000))
    ok = action_token_ids == expected
    return CheckResult(
        "runtime OpenVLA-style action ids",
        ok,
        f"got {action_token_ids[0]}..{action_token_ids[-1]}, expected 31744..31999",
    )


def check_typed_slot_pattern() -> CheckResult:
    tokenizer = DummyTokenizer()
    action_token_ids = select_action_token_ids(tokenizer, num_bins=ACTION_NUM_BINS)
    slot_ids = select_action_slot_token_ids(tokenizer, action_token_ids)
    slots = build_typed_action_slot_token_ids(slot_ids, action_chunk_size=2, action_dim=7)
    expected_step = torch.tensor(
        [31741, 31741, 31741, 31742, 31742, 31742, 31743],
        dtype=torch.long,
    )
    ok = slot_ids == {"x": 31741, "theta": 31742, "gripper": 31743}
    ok = ok and torch.equal(slots[:7].cpu(), expected_step)
    ok = ok and torch.equal(slots[7:].cpu(), expected_step)
    ok = ok and not set(slot_ids.values()).intersection(action_token_ids)
    return CheckResult(
        "runtime typed action slots",
        ok,
        f"slot_ids={slot_ids}; pattern={slots[:7].tolist()}",
    )


def check_percentile_discretization() -> CheckResult:
    actions = torch.tensor(
        [
            [-2.0, -0.5, 0.0, 0.2, -0.2, 0.4, -1.0],
            [-1.0, 0.0, 0.1, 0.1, -0.1, 0.3, -1.0],
            [0.0, 0.5, 0.2, 0.0, 0.0, 0.2, 1.0],
            [1.0, 1.0, 0.3, -0.1, 0.1, 0.1, 1.0],
            [2.0, 1.5, 0.4, -0.2, 0.2, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    edges = compute_percentile_action_bin_edges(actions, low_percentile=1.0, high_percentile=99.0)
    bins = discretize_actions(actions, bin_edges=edges)
    restored = undiscretize_action_bins(bins, bin_edges=edges)
    ok = tuple(edges.shape) == (7, ACTION_NUM_BINS + 1)
    ok = ok and bins.min().item() >= 0 and bins.max().item() < ACTION_NUM_BINS
    ok = ok and tuple(restored.shape) == tuple(actions.shape)
    return CheckResult(
        "runtime percentile binning",
        ok,
        f"edges_shape={tuple(edges.shape)}, bins_range={int(bins.min())}..{int(bins.max())}",
    )


def check_hybrid_mask() -> CheckResult:
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    hybrid = build_hybrid_attention_mask(attention_mask, num_action_tokens=2, dtype=torch.float32)
    matrix = hybrid[0, 0]
    ok = matrix.shape == (6, 6)
    ok = ok and matrix[1, 2].item() < 0
    ok = ok and matrix[2, 1].item() == 0
    ok = ok and torch.all(matrix[3, :5] == 0).item()
    ok = ok and torch.all(matrix[4, :5] == 0).item()
    ok = ok and matrix[0, 5].item() < 0
    return CheckResult(
        "runtime hybrid attention mask",
        ok,
        "non-action rows causal; final action rows full-attend over valid tokens",
    )


def check_rotation() -> CheckResult:
    image = np.arange(2 * 3 * 1, dtype=np.uint8).reshape(2, 3, 1)
    rotated = rotate_libero_image_180(image)
    expected = image[::-1, ::-1]
    tensor = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3)
    rotated_tensor = rotate_libero_image_180(tensor)
    expected_tensor = torch.flip(tensor, dims=(-2, -1))
    ok = np.array_equal(rotated, expected)
    ok = ok and torch.equal(rotated_tensor, expected_tensor)
    return CheckResult(
        "runtime LIBERO 180-degree rotation",
        ok,
        "checks HWC numpy and CHW torch rotations",
    )


def check_noop_filter() -> CheckResult:
    dataset = object.__new__(LiberoGoalDataset)
    dataset.pause_threshold = 0.01
    dataset.gripper_pause_threshold = 1e-6
    no_op = np.zeros(7, dtype=np.float32)
    gripper_change = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    arm_motion = np.array([0.02, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    ok = dataset._is_pause(no_op, no_op)
    ok = ok and not dataset._is_pause(gripper_change, no_op)
    ok = ok and not dataset._is_pause(arm_motion, no_op)
    return CheckResult(
        "runtime no-op filtering",
        ok,
        "only filters low translation/rotation with unchanged gripper",
    )


def check_checkpoint_config(config_path: Path) -> list[CheckResult]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    action_token_ids = config.get("action_token_ids")
    slot_ids = config.get("action_slot_token_ids")
    bin_edges = config.get("action_bin_edges")

    token_ok = (
        isinstance(action_token_ids, list)
        and action_token_ids == list(range(31744, 32000))
    )
    slot_ok = slot_ids == {"x": 31741, "theta": 31742, "gripper": 31743}
    bin_ok = (
        config.get("use_action_percentile_bins") is True
        and isinstance(bin_edges, list)
        and len(bin_edges) == 7
        and all(len(row) == ACTION_NUM_BINS + 1 for row in bin_edges)
    )
    hybrid_ok = config.get("use_hybrid_attention") is True

    return [
        CheckResult("checkpoint action ids", token_ok, "expects exactly 31744..31999"),
        CheckResult("checkpoint typed slot ids", slot_ok, "expects 31741/31742/31743"),
        CheckResult("checkpoint percentile edges", bin_ok, "expects 7 x 257 bin edges"),
        CheckResult("checkpoint hybrid attention", hybrid_ok, "expects use_hybrid_attention=True"),
    ]


def print_results(results: list[CheckResult]) -> bool:
    all_passed = True
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.name}")
        print(f"       {result.detail}")
        all_passed = all_passed and result.passed
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run torch-backed CoT-VLA action alignment checks.")
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    args = parser.parse_args()

    results = [
        check_openvla_token_ids(),
        check_typed_slot_pattern(),
        check_percentile_discretization(),
        check_hybrid_mask(),
        check_rotation(),
        check_noop_filter(),
    ]
    if args.checkpoint_config is not None:
        results.extend(check_checkpoint_config(args.checkpoint_config))

    print("=" * 80)
    print("CoT-VLA action paper-alignment runtime checks")
    print("=" * 80)
    all_passed = print_results(results)
    print("=" * 80)
    print("All runtime checks passed" if all_passed else "Some runtime checks failed")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
