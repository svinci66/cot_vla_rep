#!/usr/bin/env python
"""Static/action-config checks for CoT-VLA action-path paper alignment.

This script intentionally avoids importing torch or loading the 7B model. It
checks source-level invariants that should hold before training/evaluating an
action checkpoint:

1. OpenVLA-style action token reuse from the base tokenizer vocab tail.
2. 256 action bins with uniform bins by default and optional percentile bins.
3. Typed parallel action slots for x/theta/gripper dimensions.
4. Hybrid attention with full attention over the action block.
5. LIBERO/OpenVLA 180-degree image rotation in train and inference paths.
6. No-op filtering that preserves gripper-only actions.
7. Optional gripper close/transition CE reweighting for action-only ablations.

Optionally pass ``--checkpoint-config path/to/config.json`` to verify that a
trained checkpoint persisted the expected action config.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def has_all(text: str, patterns: list[str]) -> bool:
    return all(pattern in text for pattern in patterns)


def check_action_constants() -> CheckResult:
    constants = read_repo_file("vila_u/constants.py")
    ok = has_all(
        constants,
        [
            "ACTION_DIM = 7",
            "ACTION_CHUNK_SIZE = 10",
            "ACTION_NUM_BINS = 256",
        ],
    )
    return CheckResult(
        "action constants",
        ok,
        "expects 7-DoF actions, chunk size 10, and 256 bins",
    )


def check_openvla_style_action_tokens() -> CheckResult:
    text = read_repo_file("vila_u/utils/action_tokenizer.py")
    ok = has_all(
        text,
        [
            "base_vocab_size = getattr(tokenizer, \"vocab_size\", None)",
            "base_start = int(base_vocab_size) - num_bins",
            "base_tail = list(range(base_start, int(base_vocab_size)))",
            "return base_tail",
        ],
    )
    return CheckResult(
        "OpenVLA-style action token reuse",
        ok,
        "expects tokenizer.vocab_size tail, not len(tokenizer)/model vocab tail",
    )


def check_percentile_bins_defaults() -> CheckResult:
    config = read_repo_file("vila_u/model/configuration_vila_u.py")
    train = read_repo_file("vila_u/train/train_action_prediction_main.py")
    script = read_repo_file("scripts/train/train_action_prediction.sh")
    ok = has_all(
        config,
        [
            "use_action_percentile_bins = kwargs.pop(\"use_action_percentile_bins\", False)",
            "action_bin_low_percentile = kwargs.pop(\"action_bin_low_percentile\", 1.0)",
            "action_bin_high_percentile = kwargs.pop(\"action_bin_high_percentile\", 99.0)",
        ],
    )
    ok = ok and has_all(
        train,
        [
            "compute_percentile_action_bin_edges(",
            "model.config.action_bin_edges = action_bin_edges.cpu().tolist()",
        ],
    )
    ok = ok and has_all(
        script,
        [
            "USE_ACTION_PERCENTILE_BINS=${USE_ACTION_PERCENTILE_BINS:-False}",
            "ACTION_BIN_LOW_PERCENTILE=${ACTION_BIN_LOW_PERCENTILE:-1.0}",
            "ACTION_BIN_HIGH_PERCENTILE=${ACTION_BIN_HIGH_PERCENTILE:-99.0}",
        ],
    )
    return CheckResult(
        "OpenVLA-style uniform action bins by default",
        ok,
        "expects percentile bins disabled by default while keeping optional percentile support",
    )


def check_typed_action_slots() -> CheckResult:
    tokenizer = read_repo_file("vila_u/utils/action_tokenizer.py")
    train = read_repo_file("vila_u/train/train_action_prediction_main.py")
    model = read_repo_file("vila_u/model/vila_u_arch.py")
    ok = has_all(
        tokenizer,
        [
            "def build_typed_action_slot_token_ids(",
            "required = (\"x\", \"theta\", \"gripper\")",
            "per_step = [x_token_id] * min(3, action_dim)",
            "per_step.extend([theta_token_id] * min(3, action_dim - 3))",
            "per_step.extend([gripper_token_id] * (action_dim - 6))",
        ],
    )
    ok = ok and has_all(
        train,
        [
            "def select_action_slot_token_ids(",
            "\"x\": min(action_token_set) - 3",
            "\"theta\": min(action_token_set) - 2",
            "\"gripper\": min(action_token_set) - 1",
            "model.config.action_slot_token_ids = action_slot_token_ids",
        ],
    )
    ok = ok and "build_typed_action_slot_token_ids(" in model
    return CheckResult(
        "typed x/theta/gripper action slots",
        ok,
        "expects separate slot ids for translation, rotation, and gripper groups",
    )


def check_hybrid_attention() -> CheckResult:
    mask = read_repo_file("vila_u/utils/hybrid_attention.py")
    train = read_repo_file("vila_u/train/train_action_prediction_main.py")
    model = read_repo_file("vila_u/model/vila_u_arch.py")
    ok = has_all(
        mask,
        [
            "allowed[:valid_len, :valid_len] = causal",
            "action_start = valid_len - action_len",
            "allowed[action_start:valid_len, :valid_len] = True",
        ],
    )
    ok = ok and "attention_mask=hybrid_attention_mask" in train
    ok = ok and "attention_mask=hybrid_attention_mask" in model
    return CheckResult(
        "hybrid attention action block",
        ok,
        "expects causal non-action region and full-attention action rows",
    )


def check_libero_rotation() -> CheckResult:
    util = read_repo_file("vila_u/utils/libero_image.py")
    dataset = read_repo_file("vila_u/data/libero_dataset_v2.py")
    legacy_dataset = read_repo_file("vila_u/data/libero_dataset.py")
    model = read_repo_file("vila_u/model/vila_u_arch.py")
    ok = has_all(
        util,
        [
            "def rotate_libero_image_180(",
            "ROTATE_180",
            "np.flip(image, axis=(0, 1))",
        ],
    )
    ok = ok and "rgb = rotate_libero_image_180(rgb)" in dataset
    ok = ok and "obs_rgb = rotate_libero_image_180(obs_rgb)" in legacy_dataset
    ok = ok and "image = rotate_libero_image_180(image)" in model
    ok = ok and "subgoal_image = rotate_libero_image_180(subgoal_image)" in model
    return CheckResult(
        "LIBERO 180-degree image rotation",
        ok,
        "expects raw LIBERO images rotated in training and inference paths",
    )


def check_noop_filtering() -> CheckResult:
    dataset = read_repo_file("vila_u/data/libero_dataset_v2.py")
    tests = read_repo_file("tests/test_libero_dataset_v2_filters.py")
    ok = has_all(
        dataset,
        [
            "position_norm = np.linalg.norm(action[:3])",
            "rotation_norm = np.linalg.norm(action[3:6])",
            "gripper_delta = abs(float(action[6]) - float(previous_action[6]))",
            "gripper_delta < self.gripper_pause_threshold",
        ],
    )
    ok = ok and "assert not dataset._is_pause(gripper_action, no_op_action)" in tests
    return CheckResult(
        "no-op filtering preserves gripper-only actions",
        ok,
        "expects low arm motion AND unchanged gripper before filtering",
    )


def check_action_token_reweighting() -> CheckResult:
    train = read_repo_file("vila_u/train/train_action_prediction_main.py")
    script = read_repo_file("scripts/train/train_action_prediction.sh")
    wrapper = read_repo_file("scripts/train_action_only_full_8gpu_fixed_lr.sh")
    ok = has_all(
        train,
        [
            "xyz_loss_weight: float = field(",
            "gripper_close_loss_weight: float = field(",
            "gripper_transition_loss_weight: float = field(",
            "reduction=\"none\"",
            "per_token_xyz_mask[:, : min(3, action_dim)] = True",
            "token_weights = torch.where(xyz_mask, xyz_weights, token_weights)",
            "close_mask = gripper_values < 0",
            "transition_mask[:, 0]",
            "torch.abs(gripper_values[:, 1:] - gripper_values[:, :-1]) > 1e-6",
            "torch.maximum(per_step_weights, close_weights)",
            "torch.maximum(per_step_weights, transition_weights)",
            "token_weights.sum().clamp_min(1.0)",
            "config.xyz_loss_weight = action_args.xyz_loss_weight",
            "config.gripper_close_loss_weight = action_args.gripper_close_loss_weight",
            "config.gripper_transition_loss_weight = action_args.gripper_transition_loss_weight",
        ],
    )
    ok = ok and has_all(
        script,
        [
            "XYZ_LOSS_WEIGHT=${XYZ_LOSS_WEIGHT:-1.0}",
            "GRIPPER_CLOSE_LOSS_WEIGHT=${GRIPPER_CLOSE_LOSS_WEIGHT:-1.0}",
            "GRIPPER_TRANSITION_LOSS_WEIGHT=${GRIPPER_TRANSITION_LOSS_WEIGHT:-1.0}",
            "--xyz_loss_weight \"$XYZ_LOSS_WEIGHT\"",
            "--gripper_close_loss_weight \"$GRIPPER_CLOSE_LOSS_WEIGHT\"",
            "--gripper_transition_loss_weight \"$GRIPPER_TRANSITION_LOSS_WEIGHT\"",
        ],
    )
    ok = ok and has_all(
        wrapper,
        [
            "GRIPPER_CLOSE_LOSS_WEIGHT=${GRIPPER_CLOSE_LOSS_WEIGHT:-2.0}",
            "GRIPPER_TRANSITION_LOSS_WEIGHT=${GRIPPER_TRANSITION_LOSS_WEIGHT:-4.0}",
        ],
    )
    return CheckResult(
        "action token loss reweighting",
        ok,
        "expects optional xyz and gripper max/override CE weights",
    )


def check_visual_vocab_not_extended_for_action_slots() -> CheckResult:
    train = read_repo_file("vila_u/train/train_action_prediction_main.py")
    slot_fn = train[train.find("def select_action_slot_token_ids(") : train.find("def select_action_slot_token_id(")]
    ok = "tokenizer.add_tokens" not in slot_fn and "resize_token_embeddings" not in slot_fn
    return CheckResult(
        "typed slots do not extend VILA-U vocab",
        ok,
        "expects no new action slot tokens because VILA-U uses vocab tail for visual specials",
    )


def check_checkpoint_config(config_path: Path) -> list[CheckResult]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    results = []

    action_token_ids = config.get("action_token_ids")
    if isinstance(action_token_ids, list) and len(action_token_ids) == 256:
        expected_start = action_token_ids[-1] - 255
        contiguous = action_token_ids == list(range(expected_start, expected_start + 256))
        no_visual_special_collision = max(action_token_ids) < 32000
        ok = contiguous and no_visual_special_collision
    else:
        ok = False
    results.append(
        CheckResult(
            "checkpoint action_token_ids",
            ok,
            "expects 256 contiguous base-vocab ids below VILA-U visual specials",
        )
    )

    slot_ids = config.get("action_slot_token_ids")
    if isinstance(slot_ids, dict) and isinstance(action_token_ids, list) and action_token_ids:
        expected = {
            "x": min(action_token_ids) - 3,
            "theta": min(action_token_ids) - 2,
            "gripper": min(action_token_ids) - 1,
        }
        ok = slot_ids == expected
    else:
        ok = False
    results.append(
        CheckResult(
            "checkpoint action_slot_token_ids",
            ok,
            "expects x/theta/gripper ids immediately before action bin ids",
        )
    )

    results.append(
        CheckResult(
            "checkpoint uniform bins",
            config.get("use_action_percentile_bins") is False
            and config.get("action_bin_edges") is None,
            "expects uniform bins by default with no persisted percentile edges",
        )
    )

    results.append(
        CheckResult(
            "checkpoint hybrid attention",
            config.get("use_hybrid_attention") is True,
            "expects hybrid attention enabled for action block",
        )
    )

    return results


def print_results(results: list[CheckResult]) -> bool:
    all_passed = True
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.name}")
        print(f"       {result.detail}")
        all_passed = all_passed and result.passed
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CoT-VLA action paper-alignment invariants.")
    parser.add_argument(
        "--checkpoint-config",
        type=Path,
        default=None,
        help="Optional checkpoint config.json to verify persisted action settings.",
    )
    args = parser.parse_args()

    checks: list[Callable[[], CheckResult]] = [
        check_action_constants,
        check_openvla_style_action_tokens,
        check_percentile_bins_defaults,
        check_typed_action_slots,
        check_visual_vocab_not_extended_for_action_slots,
        check_hybrid_attention,
        check_libero_rotation,
        check_noop_filtering,
        check_action_token_reweighting,
    ]

    print("=" * 80)
    print("CoT-VLA action paper-alignment static checks")
    print("=" * 80)
    results = [check() for check in checks]

    if args.checkpoint_config is not None:
        print()
        print("=" * 80)
        print(f"Checkpoint config checks: {args.checkpoint_config}")
        print("=" * 80)
        results.extend(check_checkpoint_config(args.checkpoint_config))

    all_passed = print_results(results)
    print("=" * 80)
    print("All checks passed" if all_passed else "Some checks failed")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
