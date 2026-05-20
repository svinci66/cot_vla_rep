#!/usr/bin/env python
"""Offline action evaluation for Phase 3 discrete-action checkpoints.

Loads LIBERO HDF5 demonstrations, calls model.predict_action(), and reports
continuous-action MAE/MSE without environment rollout.
"""

import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from vila_u.model.builder import load_pretrained_model
from vila_u.train.utils import get_checkpoint_path
from vila_u.utils.action_tokenizer import discretize_actions, normalize_action_bin_edges


def natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_dir():
        checkpoint_path, _ = get_checkpoint_path(str(path))
        if checkpoint_path is not None:
            return checkpoint_path
        return str(path)

    if not path.exists() and path.name.startswith("tmp-checkpoint-"):
        alt = path.with_name(path.name.replace("tmp-checkpoint-", "checkpoint-", 1))
        if alt.is_dir():
            return str(alt)

    raise FileNotFoundError(f"Model path does not exist: {path}")


def iter_libero_samples(
    data_root: str,
    action_chunk_size: int,
    stride: int,
    task_file: str | None = None,
    task_file_pattern: str | None = None,
    max_task_files: int | None = None,
    max_demos_per_task: int | None = None,
):
    hdf5_files = sorted(
        (filename for filename in os.listdir(data_root) if filename.endswith(".hdf5")),
        key=natural_key,
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_root}")
    if task_file is not None:
        requested = os.path.basename(task_file)
        hdf5_files = [filename for filename in hdf5_files if filename == requested]
        if not hdf5_files:
            raise FileNotFoundError(f"Task file {requested!r} not found under {data_root}")
    if task_file_pattern is not None:
        pattern = re.compile(task_file_pattern)
        hdf5_files = [
            filename
            for filename in hdf5_files
            if task_file_pattern in filename or pattern.search(filename)
        ]
        if not hdf5_files:
            raise FileNotFoundError(
                f"No HDF5 task files under {data_root} matched pattern {task_file_pattern!r}"
            )
    if max_task_files is not None:
        hdf5_files = hdf5_files[:max_task_files]

    for filename in hdf5_files:
        data_file = os.path.join(data_root, filename)
        with h5py.File(data_file, "r") as h5_file:
            instruction = json.loads(h5_file["data"].attrs["problem_info"])["language_instruction"]
            demo_names = sorted(h5_file["data"].keys(), key=natural_key)
            if max_demos_per_task is not None:
                demo_names = demo_names[:max_demos_per_task]
            for demo_name in demo_names:
                demo = h5_file["data"][demo_name]
                actions = demo["actions"][:]
                num_frames = len(actions)
                max_start = max(0, num_frames - action_chunk_size)
                for timestep in range(0, max_start + 1, stride):
                    yield {
                        "data_file": data_file,
                        "demo_name": demo_name,
                        "instruction": instruction,
                        "timestep": timestep,
                        "image": demo["obs/agentview_rgb"][timestep],
                        "action_labels": np.clip(
                            actions[timestep : timestep + action_chunk_size],
                            -1.0,
                            1.0,
                        ),
                    }


def main():
    parser = argparse.ArgumentParser(description="Offline eval for Phase 3 action prediction checkpoints.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument(
        "--data-root",
        default="/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal",
        help="Path to LIBERO dataset root.",
    )
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1, help="Timestep stride while scanning demos.")
    parser.add_argument("--task-file", default=None, help="Exact HDF5 task filename to evaluate.")
    parser.add_argument("--task-file-pattern", default=None, help="Substring or regex for task files.")
    parser.add_argument("--max-task-files", type=int, default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--save-records",
        action="store_true",
        help="Include per-sample predictions and labels in the JSON output.",
    )
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model_path)
    _, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        device=args.device,
    )
    model.eval()

    action_chunk_size = int(model.config.action_chunk_size)
    action_dim = int(model.config.action_dim)

    print("=" * 72)
    print("Phase 3 Offline Action Evaluation")
    print("=" * 72)
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Max samples: {args.max_samples}")
    print(f"Stride: {args.stride}")
    print(f"Task file: {args.task_file}")
    print(f"Task file pattern: {args.task_file_pattern}")
    print(f"Max task files: {args.max_task_files}")
    print(f"Max demos per task: {args.max_demos_per_task}")
    print(f"Action shape: ({action_chunk_size}, {action_dim})")
    print(f"use_discrete_action_prediction = {getattr(model.config, 'use_discrete_action_prediction', None)}")
    print(f"use_hybrid_attention = {getattr(model.config, 'use_hybrid_attention', None)}")
    print(f"use_action_percentile_bins = {getattr(model.config, 'use_action_percentile_bins', None)}")
    action_bin_edges = normalize_action_bin_edges(
        getattr(model.config, "action_bin_edges", None),
        device="cpu",
    )
    print(f"action_bin_edges = {'present' if action_bin_edges is not None else 'None'}")
    print()

    maes = []
    mses = []
    first_step_maes = []
    per_dim_abs_errors = []
    per_horizon_abs_errors = []
    pred_mins = []
    pred_maxes = []
    finite_flags = []
    motion_maes = []
    motion_mses = []
    gripper_maes = []
    gripper_mses = []
    gripper_sign_matches = []
    token_accuracies = []
    first_step_token_accuracies = []
    first_step_exact_matches = []
    chunk_exact_matches = []
    per_dim_token_accuracies = []
    gripper_token_accuracies = []
    file_stats = {}
    records = []

    sample_iter = iter_libero_samples(
        data_root=args.data_root,
        action_chunk_size=action_chunk_size,
        stride=max(1, args.stride),
        task_file=args.task_file,
        task_file_pattern=args.task_file_pattern,
        max_task_files=args.max_task_files,
        max_demos_per_task=args.max_demos_per_task,
    )
    for sample_idx, sample in enumerate(tqdm(sample_iter, total=args.max_samples)):
        if sample_idx >= args.max_samples:
            break

        with torch.no_grad():
            pred_actions = model.predict_action(
                image=sample["image"],
                instruction=sample["instruction"],
                image_processor=image_processor,
            )
            pred_actions = pred_actions.detach().cpu().float().numpy()

        gt_actions = sample["action_labels"].astype(np.float32)
        abs_error = np.abs(pred_actions - gt_actions)
        sq_error = np.square(pred_actions - gt_actions)
        mae = float(abs_error.mean())
        mse = float(sq_error.mean())
        first_step_mae = float(abs_error[0].mean())
        finite = bool(np.isfinite(pred_actions).all())
        motion_abs_error = abs_error[:, :6] if action_dim > 1 else abs_error
        motion_sq_error = sq_error[:, :6] if action_dim > 1 else sq_error
        gripper_abs_error = abs_error[:, 6] if action_dim > 6 else None
        gripper_sq_error = sq_error[:, 6] if action_dim > 6 else None
        motion_mae = float(motion_abs_error.mean())
        motion_mse = float(motion_sq_error.mean())
        gripper_mae = float(gripper_abs_error.mean()) if gripper_abs_error is not None else None
        gripper_mse = float(gripper_sq_error.mean()) if gripper_sq_error is not None else None
        gripper_sign_match = (
            float((np.sign(pred_actions[:, 6]) == np.sign(gt_actions[:, 6])).mean())
            if action_dim > 6
            else None
        )
        pred_bins = discretize_actions(
            torch.from_numpy(pred_actions),
            bin_edges=action_bin_edges,
        ).numpy()
        gt_bins = discretize_actions(
            torch.from_numpy(gt_actions),
            bin_edges=action_bin_edges,
        ).numpy()
        token_matches = pred_bins == gt_bins
        token_accuracy = float(token_matches.mean())
        first_step_token_accuracy = float(token_matches[0].mean())
        first_step_exact_match = float(token_matches[0].all())
        chunk_exact_match = float(token_matches.all())
        per_dim_token_accuracy = token_matches.mean(axis=0).astype(float)
        gripper_token_accuracy = float(token_matches[:, 6].mean()) if action_dim > 6 else None

        maes.append(mae)
        mses.append(mse)
        first_step_maes.append(first_step_mae)
        per_dim_abs_errors.append(abs_error.mean(axis=0))
        per_horizon_abs_errors.append(abs_error.mean(axis=1))
        pred_mins.append(float(np.min(pred_actions)))
        pred_maxes.append(float(np.max(pred_actions)))
        finite_flags.append(finite)
        motion_maes.append(motion_mae)
        motion_mses.append(motion_mse)
        if gripper_mae is not None:
            gripper_maes.append(gripper_mae)
            gripper_mses.append(gripper_mse)
            gripper_sign_matches.append(gripper_sign_match)
        token_accuracies.append(token_accuracy)
        first_step_token_accuracies.append(first_step_token_accuracy)
        first_step_exact_matches.append(first_step_exact_match)
        chunk_exact_matches.append(chunk_exact_match)
        per_dim_token_accuracies.append(per_dim_token_accuracy)
        if gripper_token_accuracy is not None:
            gripper_token_accuracies.append(gripper_token_accuracy)

        file_key = os.path.basename(sample["data_file"])
        if file_key not in file_stats:
            file_stats[file_key] = {
                "num_samples": 0,
                "mae": [],
                "mse": [],
                "motion_mae": [],
                "gripper_mae": [],
                "token_accuracy": [],
                "first_step_token_accuracy": [],
                "first_step_exact_match": [],
                "chunk_exact_match": [],
                "gripper_token_accuracy": [],
            }
        file_stats[file_key]["num_samples"] += 1
        file_stats[file_key]["mae"].append(mae)
        file_stats[file_key]["mse"].append(mse)
        file_stats[file_key]["motion_mae"].append(motion_mae)
        if gripper_mae is not None:
            file_stats[file_key]["gripper_mae"].append(gripper_mae)
        file_stats[file_key]["token_accuracy"].append(token_accuracy)
        file_stats[file_key]["first_step_token_accuracy"].append(first_step_token_accuracy)
        file_stats[file_key]["first_step_exact_match"].append(first_step_exact_match)
        file_stats[file_key]["chunk_exact_match"].append(chunk_exact_match)
        if gripper_token_accuracy is not None:
            file_stats[file_key]["gripper_token_accuracy"].append(gripper_token_accuracy)

        record = {
            "file": sample["data_file"],
            "demo": sample["demo_name"],
            "timestep": sample["timestep"],
            "mae": mae,
            "mse": mse,
            "first_step_mae": first_step_mae,
            "motion_mae": motion_mae,
            "motion_mse": motion_mse,
            "gripper_mae": gripper_mae,
            "gripper_mse": gripper_mse,
            "gripper_sign_match": gripper_sign_match,
            "token_accuracy": token_accuracy,
            "first_step_token_accuracy": first_step_token_accuracy,
            "first_step_exact_match": first_step_exact_match,
            "chunk_exact_match": chunk_exact_match,
            "per_dim_token_accuracy": per_dim_token_accuracy.tolist(),
            "gripper_token_accuracy": gripper_token_accuracy,
            "pred_min": pred_mins[-1],
            "pred_max": pred_maxes[-1],
            "finite": finite,
        }
        if args.save_records:
            record["pred_actions"] = pred_actions.tolist()
            record["gt_actions"] = gt_actions.tolist()
        records.append(record)

    file_summary = {
        file_key: {
            "num_samples": stats["num_samples"],
            "mae": float(np.mean(stats["mae"])),
            "mse": float(np.mean(stats["mse"])),
            "motion_mae": float(np.mean(stats["motion_mae"])),
            "gripper_mae": float(np.mean(stats["gripper_mae"])) if stats["gripper_mae"] else None,
            "token_accuracy": float(np.mean(stats["token_accuracy"])),
            "first_step_token_accuracy": float(np.mean(stats["first_step_token_accuracy"])),
            "first_step_exact_match": float(np.mean(stats["first_step_exact_match"])),
            "chunk_exact_match": float(np.mean(stats["chunk_exact_match"])),
            "gripper_token_accuracy": (
                float(np.mean(stats["gripper_token_accuracy"]))
                if stats["gripper_token_accuracy"]
                else None
            ),
        }
        for file_key, stats in sorted(file_stats.items())
    }

    summary = {
        "model_path": resolved_model_path,
        "data_root": args.data_root,
        "num_samples": len(maes),
        "stride": args.stride,
        "task_file": args.task_file,
        "task_file_pattern": args.task_file_pattern,
        "max_task_files": args.max_task_files,
        "max_demos_per_task": args.max_demos_per_task,
        "use_action_percentile_bins": getattr(model.config, "use_action_percentile_bins", None),
        "has_action_bin_edges": action_bin_edges is not None,
        "mae": float(np.mean(maes)) if maes else None,
        "mse": float(np.mean(mses)) if mses else None,
        "first_step_mae": float(np.mean(first_step_maes)) if first_step_maes else None,
        "motion_mae": float(np.mean(motion_maes)) if motion_maes else None,
        "motion_mse": float(np.mean(motion_mses)) if motion_mses else None,
        "gripper_mae": float(np.mean(gripper_maes)) if gripper_maes else None,
        "gripper_mse": float(np.mean(gripper_mses)) if gripper_mses else None,
        "gripper_sign_match": float(np.mean(gripper_sign_matches)) if gripper_sign_matches else None,
        "token_accuracy": float(np.mean(token_accuracies)) if token_accuracies else None,
        "first_step_token_accuracy": (
            float(np.mean(first_step_token_accuracies))
            if first_step_token_accuracies
            else None
        ),
        "first_step_exact_match": (
            float(np.mean(first_step_exact_matches))
            if first_step_exact_matches
            else None
        ),
        "chunk_exact_match": (
            float(np.mean(chunk_exact_matches))
            if chunk_exact_matches
            else None
        ),
        "per_dim_token_accuracy": (
            np.mean(per_dim_token_accuracies, axis=0).astype(float).tolist()
            if per_dim_token_accuracies
            else None
        ),
        "gripper_token_accuracy": (
            float(np.mean(gripper_token_accuracies))
            if gripper_token_accuracies
            else None
        ),
        "per_dim_mae": np.mean(per_dim_abs_errors, axis=0).astype(float).tolist() if per_dim_abs_errors else None,
        "per_horizon_mae": np.mean(per_horizon_abs_errors, axis=0).astype(float).tolist() if per_horizon_abs_errors else None,
        "pred_min": float(np.min(pred_mins)) if pred_mins else None,
        "pred_max": float(np.max(pred_maxes)) if pred_maxes else None,
        "finite_rate": float(np.mean(finite_flags)) if finite_flags else None,
        "files": file_summary,
        "records": records,
    }

    print()
    print("Summary")
    print(f"  num_samples = {summary['num_samples']}")
    print(f"  mae = {summary['mae']}")
    print(f"  mse = {summary['mse']}")
    print(f"  first_step_mae = {summary['first_step_mae']}")
    print(f"  motion_mae = {summary['motion_mae']}")
    print(f"  motion_mse = {summary['motion_mse']}")
    print(f"  gripper_mae = {summary['gripper_mae']}")
    print(f"  gripper_mse = {summary['gripper_mse']}")
    print(f"  gripper_sign_match = {summary['gripper_sign_match']}")
    print(f"  token_accuracy = {summary['token_accuracy']}")
    print(f"  first_step_token_accuracy = {summary['first_step_token_accuracy']}")
    print(f"  first_step_exact_match = {summary['first_step_exact_match']}")
    print(f"  chunk_exact_match = {summary['chunk_exact_match']}")
    print(f"  per_dim_token_accuracy = {summary['per_dim_token_accuracy']}")
    print(f"  gripper_token_accuracy = {summary['gripper_token_accuracy']}")
    print(f"  per_dim_mae = {summary['per_dim_mae']}")
    print(f"  per_horizon_mae = {summary['per_horizon_mae']}")
    print(f"  pred_min/max = {summary['pred_min']}/{summary['pred_max']}")
    print(f"  finite_rate = {summary['finite_rate']}")
    if summary["files"]:
        print("  per_file_mae:")
        for file_key, stats in summary["files"].items():
            print(
                f"    {file_key}: n={stats['num_samples']} "
                f"mae={stats['mae']:.6f} "
                f"motion_mae={stats['motion_mae']:.6f} "
                f"gripper_mae={stats['gripper_mae']} "
                f"token_acc={stats['token_accuracy']:.6f} "
                f"first_exact={stats['first_step_exact_match']:.6f} "
                f"chunk_exact={stats['chunk_exact_match']:.6f} "
                f"gripper_token_acc={stats['gripper_token_accuracy']}"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
