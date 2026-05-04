#!/usr/bin/env python
"""Phase 4 oracle-subgoal simplified pipeline check.

This script loads one LIBERO HDF5 sample, samples a future frame as the oracle
subgoal image, and calls model.predict_action(..., subgoal_image=...). It is an
offline sanity entrypoint for the simplified Visual CoT implementation; it does
not construct a separate CoT dataset.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import torch

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


def load_one_oracle_sample(
    data_root: str,
    file_index: int,
    demo_index: int,
    timestep: int,
    subgoal_offset: int,
):
    hdf5_files = sorted(
        filename for filename in os.listdir(data_root) if filename.endswith(".hdf5")
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_root}")

    file_index = min(file_index, len(hdf5_files) - 1)
    data_file = os.path.join(data_root, hdf5_files[file_index])

    with h5py.File(data_file, "r") as h5_file:
        instruction = json.loads(h5_file["data"].attrs["problem_info"])["language_instruction"]
        demo_names = sorted(h5_file["data"].keys())
        demo_index = min(demo_index, len(demo_names) - 1)
        demo_name = demo_names[demo_index]
        demo = h5_file["data"][demo_name]
        num_frames = int(demo.attrs.get("num_samples", len(demo["actions"])))
        timestep = min(timestep, num_frames - 1)
        subgoal_timestep = min(timestep + subgoal_offset, num_frames - 1)
        image = demo["obs/agentview_rgb"][timestep]
        subgoal_image = demo["obs/agentview_rgb"][subgoal_timestep]

    return {
        "data_file": data_file,
        "demo_name": demo_name,
        "instruction": instruction,
        "timestep": timestep,
        "subgoal_timestep": subgoal_timestep,
        "image": image,
        "subgoal_image": subgoal_image,
    }


def main():
    parser = argparse.ArgumentParser(description="Check Phase 4 subgoal-conditioned inference.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument(
        "--data-root",
        default="/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal",
        help="Path to LIBERO dataset root.",
    )
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--subgoal-offset", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=("oracle", "generated"),
        default="oracle",
        help="Use GT future-frame subgoal or generated visual CoT subgoal.",
    )
    parser.add_argument("--cfg", type=float, default=3.0, help="Classifier-free guidance for generated subgoals.")
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model_path)
    sample = load_one_oracle_sample(
        data_root=args.data_root,
        file_index=args.file_index,
        demo_index=args.demo_index,
        timestep=args.timestep,
        subgoal_offset=args.subgoal_offset,
    )

    print("=" * 72)
    print("Phase 4 Subgoal-Conditioned Inference Check")
    print("=" * 72)
    print(f"Requested model path: {args.model_path}")
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Sample file: {sample['data_file']}")
    print(f"Demo: {sample['demo_name']}")
    print(f"Timestep: {sample['timestep']}")
    print(f"Subgoal timestep: {sample['subgoal_timestep']}")
    print(f"Mode: {args.mode}")
    print()

    _, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        device=args.device,
    )
    model.eval()

    print("[1/2] Config check")
    print(f"  use_discrete_action_prediction = {getattr(model.config, 'use_discrete_action_prediction', None)}")
    print(f"  use_hybrid_attention = {getattr(model.config, 'use_hybrid_attention', None)}")
    print(f"  use_visual_cot = {getattr(model.config, 'use_visual_cot', None)}")
    print(f"  action_chunk_size = {getattr(model.config, 'action_chunk_size', None)}")
    print(f"  action_dim = {getattr(model.config, 'action_dim', None)}")
    print()

    print("[2/2] Subgoal-conditioned predict_action")
    with torch.no_grad():
        if args.mode == "oracle":
            actions = model.predict_action(
                image=sample["image"],
                instruction=sample["instruction"],
                image_processor=image_processor,
                subgoal_image=sample["subgoal_image"],
            )
        else:
            actions, generated_subgoal, generated_codes = model.predict_action_with_generated_subgoal(
                image=sample["image"],
                instruction=sample["instruction"],
                image_processor=image_processor,
                cfg=args.cfg,
                return_subgoal=True,
            )
            print(f"  generated subgoal image shape = {tuple(generated_subgoal.shape)}")
            print(f"  generated subgoal code shape = {tuple(generated_codes.shape)}")

    print(f"  predicted action shape = {tuple(actions.shape)}")
    print(f"  finite = {bool(torch.isfinite(actions).all().item())}")
    print(f"  min/max = {float(actions.min()):.4f}/{float(actions.max()):.4f}")
    expected_shape = (model.config.action_chunk_size, model.config.action_dim)
    assert tuple(actions.shape) == expected_shape, (tuple(actions.shape), expected_shape)
    assert torch.isfinite(actions).all(), "Predicted actions contain NaN or Inf"
    print("  ✓ Phase 4 subgoal-conditioned inference path is callable")
    print()
    print("=" * 72)
    print("Phase 4 subgoal-conditioned inference check passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
