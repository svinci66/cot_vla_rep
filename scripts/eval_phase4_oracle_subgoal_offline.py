#!/usr/bin/env python
"""Offline evaluation for the Phase 4 oracle-subgoal simplified baseline.

Evaluates action prediction on LIBERO HDF5 demonstrations by using a future
frame from the same trajectory as the oracle subgoal image. This measures the
simplified Visual CoT action path without requiring environment rollout or a
separate CoT dataset.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from vila_u.model.builder import load_pretrained_model
from vila_u.train.utils import get_checkpoint_path


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


def iter_libero_samples(data_root: str, action_chunk_size: int, subgoal_offset: int):
    hdf5_files = sorted(
        filename for filename in os.listdir(data_root) if filename.endswith(".hdf5")
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_root}")

    for filename in hdf5_files:
        data_file = os.path.join(data_root, filename)
        with h5py.File(data_file, "r") as h5_file:
            instruction = json.loads(h5_file["data"].attrs["problem_info"])["language_instruction"]
            for demo_name in sorted(h5_file["data"].keys()):
                demo = h5_file["data"][demo_name]
                actions = demo["actions"][:]
                num_frames = len(actions)
                max_start = max(0, num_frames - action_chunk_size)
                for timestep in range(max_start):
                    subgoal_timestep = min(timestep + subgoal_offset, num_frames - 1)
                    yield {
                        "data_file": data_file,
                        "demo_name": demo_name,
                        "instruction": instruction,
                        "timestep": timestep,
                        "subgoal_timestep": subgoal_timestep,
                        "image": demo["obs/agentview_rgb"][timestep],
                        "subgoal_image": demo["obs/agentview_rgb"][subgoal_timestep],
                        "action_labels": np.clip(
                            actions[timestep : timestep + action_chunk_size],
                            -1.0,
                            1.0,
                        ),
                    }


def main():
    parser = argparse.ArgumentParser(description="Offline eval for Phase 4 subgoal-conditioned baseline.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument(
        "--data-root",
        default="/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal",
        help="Path to LIBERO dataset root.",
    )
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--subgoal-offset", type=int, default=10)
    parser.add_argument("--mode", choices=("oracle", "generated"), default="oracle")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--output-json", default=None)
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
    print("Phase 4 Subgoal-Conditioned Offline Evaluation")
    print("=" * 72)
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Max samples: {args.max_samples}")
    print(f"Subgoal offset: {args.subgoal_offset}")
    print(f"Mode: {args.mode}")
    print(f"Action shape: ({action_chunk_size}, {action_dim})")
    print()

    maes = []
    mses = []
    first_step_maes = []
    records = []

    sample_iter = iter_libero_samples(
        data_root=args.data_root,
        action_chunk_size=action_chunk_size,
        subgoal_offset=args.subgoal_offset,
    )
    for sample_idx, sample in enumerate(tqdm(sample_iter, total=args.max_samples)):
        if sample_idx >= args.max_samples:
            break

        with torch.no_grad():
            if args.mode == "oracle":
                pred_actions = model.predict_action(
                    image=sample["image"],
                    instruction=sample["instruction"],
                    image_processor=image_processor,
                    subgoal_image=sample["subgoal_image"],
                )
            else:
                pred_actions = model.predict_action_with_generated_subgoal(
                    image=sample["image"],
                    instruction=sample["instruction"],
                    image_processor=image_processor,
                    cfg=args.cfg,
                )
            pred_actions = pred_actions.detach().cpu().float().numpy()

        gt_actions = sample["action_labels"].astype(np.float32)
        abs_error = np.abs(pred_actions - gt_actions)
        sq_error = np.square(pred_actions - gt_actions)
        mae = float(abs_error.mean())
        mse = float(sq_error.mean())
        first_step_mae = float(abs_error[0].mean())
        maes.append(mae)
        mses.append(mse)
        first_step_maes.append(first_step_mae)
        records.append(
            {
                "file": sample["data_file"],
                "demo": sample["demo_name"],
                "timestep": sample["timestep"],
                "subgoal_timestep": sample["subgoal_timestep"],
                "mae": mae,
                "mse": mse,
                "first_step_mae": first_step_mae,
            }
        )

    summary = {
        "model_path": resolved_model_path,
        "data_root": args.data_root,
        "num_samples": len(maes),
        "subgoal_offset": args.subgoal_offset,
        "mode": args.mode,
        "cfg": args.cfg,
        "mae": float(np.mean(maes)) if maes else None,
        "mse": float(np.mean(mses)) if mses else None,
        "first_step_mae": float(np.mean(first_step_maes)) if first_step_maes else None,
        "records": records,
    }

    print()
    print("Summary")
    print(f"  num_samples = {summary['num_samples']}")
    print(f"  mae = {summary['mae']}")
    print(f"  mse = {summary['mse']}")
    print(f"  first_step_mae = {summary['first_step_mae']}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
