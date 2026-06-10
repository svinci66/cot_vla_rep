#!/usr/bin/env python
"""Run the standard action-only offline evaluation suite.

The suite runs three diagnostics with paper-aligned action-training defaults:

1. Full-task strict offline evaluation.
2. Per-task/per-dimension strict offline evaluation.
3. Semantic sensitivity evaluation by instruction swapping.

Only ``--model-path`` is required. Other defaults match the current LIBERO
Goal server setup used for action-only checkpoints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = "/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"


def safe_name(path: str) -> str:
    name = Path(path.rstrip("/")).name or "checkpoint"
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def run_command(label: str, cmd: list[str]) -> None:
    print("=" * 80, flush=True)
    print(label, flush=True)
    print("=" * 80, flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full action-only offline evaluation suite.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory to evaluate.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="outputs/offline_action_eval")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--semantic-batch-size", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=1000000)
    parser.add_argument("--max-samples-per-task", type=int, default=1000000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--semantic-num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-demos-per-task", type=int, default=5)
    parser.add_argument("--samples-per-demo", type=int, default=5)
    parser.add_argument("--remove-pause-intervals", default="True", choices=("True", "False", "true", "false"))
    parser.add_argument("--pause-threshold", type=float, default=0.01)
    parser.add_argument("--gripper-pause-threshold", type=float, default=1e-6)
    parser.add_argument("--recompute-action-bin-edges", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-per-task", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()

    model_name = safe_name(args.output_prefix or args.model_path)
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    full_json = output_dir / f"full_alltasks_{args.max_demos_per_task}demo.json"
    per_task_json = output_dir / f"per_task_dim_{args.max_demos_per_task}demo.json"
    semantic_json = output_dir / f"semantic_{args.max_demos_per_task}demo_{args.samples_per_demo}sample.json"
    summary_json = output_dir / "summary.json"

    common = [
        "--model-path",
        args.model_path,
        "--data-root",
        args.data_root,
        "--device",
        args.device,
        "--model-dtype",
        args.model_dtype,
        "--image-size",
        str(args.image_size),
        "--remove-pause-intervals",
        args.remove_pause_intervals,
        "--pause-threshold",
        str(args.pause_threshold),
        "--gripper-pause-threshold",
        str(args.gripper_pause_threshold),
        "--max-demos-per-task",
        str(args.max_demos_per_task),
    ]

    if not args.skip_full:
        cmd = [
            sys.executable,
            "scripts/eval_action_training_distribution_offline.py",
            *common,
            "--batch-size",
            str(args.batch_size),
            "--max-samples",
            str(args.max_samples),
            "--num-workers",
            str(args.num_workers),
            "--output-json",
            str(full_json),
        ]
        if args.recompute_action_bin_edges:
            cmd.append("--recompute-action-bin-edges")
        run_command("[1/3] Full-task strict offline evaluation", cmd)

    if not args.skip_per_task:
        cmd = [
            sys.executable,
            "scripts/eval_action_per_task_dim_offline.py",
            *common,
            "--batch-size",
            str(args.batch_size),
            "--max-samples-per-task",
            str(args.max_samples_per_task),
            "--num-workers",
            str(args.num_workers),
            "--output-json",
            str(per_task_json),
        ]
        if args.recompute_action_bin_edges:
            cmd.append("--recompute-action-bin-edges")
        run_command("[2/3] Per-task/per-dimension offline evaluation", cmd)

    if not args.skip_semantic:
        cmd = [
            sys.executable,
            "scripts/check_action_semantic_sensitivity.py",
            *common,
            "--batch-size",
            str(args.semantic_batch_size),
            "--samples-per-demo",
            str(args.samples_per_demo),
            "--num-workers",
            str(args.semantic_num_workers),
            "--output-json",
            str(semantic_json),
        ]
        if args.recompute_action_bin_edges:
            cmd.append("--recompute-action-bin-edges")
        run_command("[3/3] Semantic sensitivity evaluation", cmd)

    full_payload = load_json(full_json)
    per_task_payload = load_json(per_task_json)
    semantic_payload = load_json(semantic_json)
    summary = {
        "model_path": args.model_path,
        "data_root": args.data_root,
        "outputs": {
            "output_dir": str(output_dir),
            "full": str(full_json),
            "per_task": str(per_task_json),
            "semantic": str(semantic_json),
            "summary": str(summary_json),
        },
        "key_metrics": {
            "full_token_accuracy": nested_get(full_payload, "summary", "token_accuracy")
            or nested_get(full_payload, "token_accuracy"),
            "full_chunk_exact_match": nested_get(full_payload, "summary", "chunk_exact_match")
            or nested_get(full_payload, "chunk_exact_match"),
            "full_gripper_close_recall": nested_get(full_payload, "summary", "gripper_close_recall")
            or nested_get(full_payload, "gripper_close_recall"),
            "full_gripper_transition_change_recall": nested_get(
                full_payload,
                "summary",
                "gripper_transition_change_recall",
            )
            or nested_get(full_payload, "gripper_transition_change_recall"),
            "full_gripper_pred_open_rate": nested_get(full_payload, "summary", "gripper_pred_open_rate")
            or nested_get(full_payload, "gripper_pred_open_rate"),
            "per_task_mean_token_accuracy": nested_get(per_task_payload, "mean_token_accuracy"),
            "per_task_mean_gripper_sign_accuracy": nested_get(per_task_payload, "mean_gripper_sign_accuracy"),
            "per_task_mean_gripper_close_recall": nested_get(
                per_task_payload,
                "mean_gripper_close_recall",
            ),
            "per_task_mean_gripper_transition_change_recall": nested_get(
                per_task_payload,
                "mean_gripper_transition_change_recall",
            ),
            "semantic_correct_best_rate": nested_get(semantic_payload, "summary", "correct_best_rate"),
            "semantic_mean_correct_rank": nested_get(semantic_payload, "summary", "mean_correct_rank"),
            "semantic_correct_wrong_mae_gap": nested_get(
                semantic_payload,
                "summary",
                "mean_correct_wrong_mae_gap",
            ),
            "semantic_gripper_sign_diff_rate_vs_correct": nested_get(
                semantic_payload,
                "summary",
                "mean_gripper_sign_diff_rate_vs_correct",
            ),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Evaluation suite summary")
    print("=" * 80)
    for key, value in summary["key_metrics"].items():
        print(f"{key}: {value}")
    print(f"wrote {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
