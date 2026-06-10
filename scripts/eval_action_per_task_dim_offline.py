#!/usr/bin/env python
"""Per-task/per-dimension strict offline action evaluation.

This script wraps ``eval_action_training_distribution_offline.py`` and runs it
once per LIBERO HDF5 task file. It is intended for action-only diagnostics such
as per-task MAE, per-dimension token accuracy, and gripper close/transition
recall without changing the strict training-distribution eval path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def natural_key(value: str):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "summary" in payload:
        return payload["summary"]
    return payload


def mean_present(per_task: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in per_task if item.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict offline action eval per LIBERO task file.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples-per-task", type=int, default=1000000)
    parser.add_argument("--max-demos-per-task", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--task-file-pattern", default=None)
    parser.add_argument("--remove-pause-intervals", default="True")
    parser.add_argument("--pause-threshold", type=float, default=0.01)
    parser.add_argument("--gripper-pause-threshold", type=float, default=1e-6)
    parser.add_argument("--recompute-action-bin-edges", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    task_files = sorted(
        (path.name for path in data_root.glob("*.hdf5")),
        key=natural_key,
    )
    if args.task_file_pattern:
        task_files = [name for name in task_files if args.task_file_pattern in name]
    if not task_files:
        raise FileNotFoundError(f"No task files found under {data_root}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_output_dir = output_path.with_suffix("")
    task_output_dir.mkdir(parents=True, exist_ok=True)

    per_task = []
    script = Path(__file__).with_name("eval_action_training_distribution_offline.py")
    for task_file in task_files:
        task_json = task_output_dir / f"{Path(task_file).stem}.json"
        cmd = [
            sys.executable,
            str(script),
            "--model-path",
            args.model_path,
            "--data-root",
            args.data_root,
            "--device",
            args.device,
            "--model-dtype",
            args.model_dtype,
            "--batch-size",
            str(args.batch_size),
            "--max-samples",
            str(args.max_samples_per_task),
            "--num-workers",
            str(args.num_workers),
            "--image-size",
            str(args.image_size),
            "--remove-pause-intervals",
            args.remove_pause_intervals,
            "--pause-threshold",
            str(args.pause_threshold),
            "--gripper-pause-threshold",
            str(args.gripper_pause_threshold),
            "--task-file",
            task_file,
            "--max-demos-per-task",
            str(args.max_demos_per_task),
            "--output-json",
            str(task_json),
        ]
        if args.recompute_action_bin_edges:
            cmd.append("--recompute-action-bin-edges")
        print("=" * 80)
        print(f"Evaluating task: {task_file}")
        subprocess.run(cmd, check=True)
        summary = load_summary(task_json)
        summary["task_file"] = task_file
        summary["task_output_json"] = str(task_json)
        per_task.append(summary)

    aggregate = {
        "model_path": args.model_path,
        "data_root": args.data_root,
        "num_tasks": len(per_task),
        "max_demos_per_task": args.max_demos_per_task,
        "max_samples_per_task": args.max_samples_per_task,
        "mean_mae": sum(item["mae"] for item in per_task) / len(per_task),
        "mean_token_accuracy": sum(item["token_accuracy"] for item in per_task) / len(per_task),
        "mean_gripper_sign_accuracy": mean_present(per_task, "gripper_sign_accuracy"),
        "mean_gripper_close_recall": mean_present(per_task, "gripper_close_recall"),
        "mean_gripper_transition_change_recall": mean_present(
            per_task,
            "gripper_transition_change_recall",
        ),
        "mean_gripper_pred_open_rate": mean_present(per_task, "gripper_pred_open_rate"),
        "per_task": per_task,
    }
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"wrote {output_path}")
    print(f"mean_mae = {aggregate['mean_mae']}")
    print(f"mean_token_accuracy = {aggregate['mean_token_accuracy']}")
    print(f"mean_gripper_sign_accuracy = {aggregate['mean_gripper_sign_accuracy']}")
    print(f"mean_gripper_close_recall = {aggregate['mean_gripper_close_recall']}")
    print(f"mean_gripper_transition_change_recall = {aggregate['mean_gripper_transition_change_recall']}")


if __name__ == "__main__":
    main()
