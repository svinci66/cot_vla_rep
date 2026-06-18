#!/usr/bin/env python
"""Visualize Phase 4 oracle and generated Visual CoT subgoals.

The script samples LIBERO HDF5 frames, optionally runs generated-subgoal
inference, and writes observation / oracle subgoal / generated subgoal panels
plus metadata. It is intended for pre-training and training-time diagnostics;
it does not require environment rollout.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from vila_u.model.builder import load_pretrained_model
from vila_u.train.utils import get_checkpoint_path
from vila_u.utils.libero_image import rotate_libero_image_180


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


def iter_libero_samples(data_root: str, subgoal_offset: int):
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
                num_frames = len(demo["actions"])
                for timestep in range(num_frames):
                    subgoal_timestep = min(timestep + subgoal_offset, num_frames - 1)
                    yield {
                        "data_file": data_file,
                        "demo_name": demo_name,
                        "instruction": instruction,
                        "timestep": timestep,
                        "subgoal_timestep": subgoal_timestep,
                        "image": demo["obs/agentview_rgb"][timestep],
                        "oracle_subgoal": demo["obs/agentview_rgb"][subgoal_timestep],
                        "selection": "sequential",
                        "subgoal_offset_mode": "raw_frame",
                    }


def is_noop_action(
    action: np.ndarray,
    previous_action: np.ndarray | None,
    pause_threshold: float,
    gripper_pause_threshold: float,
) -> bool:
    position_norm = np.linalg.norm(action[:3])
    rotation_norm = np.linalg.norm(action[3:6])
    if action.shape[0] > 6 and previous_action is not None and previous_action.shape[0] > 6:
        gripper_delta = abs(float(action[6]) - float(previous_action[6]))
    else:
        gripper_delta = abs(float(action[6])) if action.shape[0] > 6 else 0.0
    return (
        position_norm < pause_threshold
        and rotation_norm < pause_threshold
        and gripper_delta < gripper_pause_threshold
    )


def compute_non_pause_indices(
    actions: np.ndarray,
    pause_threshold: float,
    gripper_pause_threshold: float,
) -> list[int]:
    indices = []
    previous_action = None
    for timestep, action in enumerate(actions):
        if not is_noop_action(action, previous_action, pause_threshold, gripper_pause_threshold):
            indices.append(timestep)
        previous_action = action
    return indices


def image_mae(image: np.ndarray, oracle: np.ndarray) -> float:
    image = rotate_libero_image_180(image).astype(np.float32) / 255.0
    oracle = rotate_libero_image_180(oracle).astype(np.float32) / 255.0
    return float(np.mean(np.abs(image - oracle)))


def collect_high_delta_samples(
    data_root: str,
    subgoal_offset: int,
    max_samples: int,
    action_chunk_size: int,
    pause_threshold: float,
    gripper_pause_threshold: float,
) -> list[dict]:
    samples = []
    hdf5_files = sorted(
        filename for filename in os.listdir(data_root) if filename.endswith(".hdf5")
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_root}")

    for filename in tqdm(hdf5_files, desc="scoring high-delta samples"):
        data_file = os.path.join(data_root, filename)
        with h5py.File(data_file, "r") as h5_file:
            instruction = json.loads(h5_file["data"].attrs["problem_info"])["language_instruction"]
            for demo_name in sorted(h5_file["data"].keys()):
                demo = h5_file["data"][demo_name]
                actions = demo["actions"][:]
                non_pause_indices = compute_non_pause_indices(
                    actions,
                    pause_threshold=pause_threshold,
                    gripper_pause_threshold=gripper_pause_threshold,
                )
                max_filtered_start = len(non_pause_indices) - action_chunk_size + 1
                if max_filtered_start <= 0:
                    continue

                for filtered_t in range(max_filtered_start):
                    raw_t = int(non_pause_indices[filtered_t])
                    target_filtered_t = min(
                        filtered_t + subgoal_offset,
                        len(non_pause_indices) - 1,
                    )
                    subgoal_timestep = int(non_pause_indices[target_filtered_t])
                    image = demo["obs/agentview_rgb"][raw_t]
                    oracle_subgoal = demo["obs/agentview_rgb"][subgoal_timestep]
                    delta = image_mae(image, oracle_subgoal)
                    samples.append(
                        {
                            "data_file": data_file,
                            "demo_name": demo_name,
                            "instruction": instruction,
                            "timestep": raw_t,
                            "filtered_timestep": filtered_t,
                            "subgoal_timestep": subgoal_timestep,
                            "image": image,
                            "oracle_subgoal": oracle_subgoal,
                            "delta_mae": delta,
                            "selection": "high-delta",
                            "subgoal_offset_mode": "filtered_action_step",
                        }
                    )

    samples = sorted(samples, key=lambda sample: sample["delta_mae"], reverse=True)
    return samples[:max_samples]


def image_from_array(array: np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(array, torch.Tensor):
        tensor = array.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.float().numpy()

    if array.dtype != np.uint8:
        if array.max() <= 1.0 and array.min() >= 0.0:
            array = array * 255.0
        elif array.min() < 0.0:
            array = (array + 1.0) * 127.5
        array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


def make_panel(images: list[tuple[str, Image.Image]], width: int = 256) -> Image.Image:
    label_height = 28
    resized = []
    for label, image in images:
        cur = image.resize((width, width))
        resized.append((label, cur))

    panel = Image.new("RGB", (width * len(resized), width + label_height), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, image) in enumerate(resized):
        x = idx * width
        panel.paste(image, (x, label_height))
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
    return panel


def save_sample_outputs(output_dir: Path, sample_idx: int, sample: dict, images: list[tuple[str, Image.Image]], metadata: dict):
    sample_dir = output_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for label, image in images:
        image.save(sample_dir / f"{label}.png")

    panel = make_panel(images)
    panel.save(sample_dir / "panel.png")

    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Visualize Phase 4 Visual CoT subgoals.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument(
        "--data-root",
        default="/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal",
        help="Path to LIBERO dataset root.",
    )
    parser.add_argument("--output-dir", default="outputs/phase4_subgoal_visualizations")
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--subgoal-offset", type=int, default=10)
    parser.add_argument("--selection", choices=("sequential", "high-delta"), default="sequential")
    parser.add_argument("--action-chunk-size", type=int, default=10)
    parser.add_argument("--pause-threshold", type=float, default=0.01)
    parser.add_argument("--gripper-pause-threshold", type=float, default=1e-6)
    parser.add_argument("--mode", choices=("oracle", "generated", "both"), default="both")
    parser.add_argument("--cfg", type=float, default=3.0)
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model_path)
    _, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        device=args.device,
    )
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "model_path": resolved_model_path,
        "data_root": args.data_root,
        "mode": args.mode,
        "cfg": args.cfg,
        "subgoal_offset": args.subgoal_offset,
        "selection": args.selection,
        "action_chunk_size": args.action_chunk_size,
        "pause_threshold": args.pause_threshold,
        "gripper_pause_threshold": args.gripper_pause_threshold,
        "max_samples": args.max_samples,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    if args.selection == "high-delta":
        samples = collect_high_delta_samples(
            data_root=args.data_root,
            subgoal_offset=args.subgoal_offset,
            max_samples=args.start_index + args.max_samples,
            action_chunk_size=args.action_chunk_size,
            pause_threshold=args.pause_threshold,
            gripper_pause_threshold=args.gripper_pause_threshold,
        )
        sample_iter = iter(samples)
        total = len(samples)
    else:
        sample_iter = iter_libero_samples(args.data_root, args.subgoal_offset)
        total = args.start_index + args.max_samples

    saved = 0
    for sample_idx, sample in enumerate(tqdm(sample_iter, total=total)):
        if sample_idx < args.start_index:
            continue
        if saved >= args.max_samples:
            break

        images = [
            ("observation", image_from_array(sample["image"])),
        ]
        if args.mode in {"oracle", "both"}:
            images.append(("oracle_subgoal", image_from_array(sample["oracle_subgoal"])))

        metadata = {
            "file": sample["data_file"],
            "demo": sample["demo_name"],
            "instruction": sample["instruction"],
            "timestep": sample["timestep"],
            "subgoal_timestep": sample["subgoal_timestep"],
            "selection": sample["selection"],
            "subgoal_offset_mode": sample["subgoal_offset_mode"],
        }
        if "delta_mae" in sample:
            metadata["delta_mae"] = sample["delta_mae"]
        if "filtered_timestep" in sample:
            metadata["filtered_timestep"] = sample["filtered_timestep"]

        if args.mode in {"generated", "both"}:
            with torch.no_grad():
                actions, generated_subgoal, generated_codes = model.predict_action_with_generated_subgoal(
                    image=sample["image"],
                    instruction=sample["instruction"],
                    image_processor=image_processor,
                    cfg=args.cfg,
                    return_subgoal=True,
                )
            images.append(("generated_subgoal", image_from_array(generated_subgoal)))
            metadata.update(
                {
                    "generated_subgoal_shape": list(generated_subgoal.shape),
                    "generated_code_shape": list(generated_codes.shape),
                    "predicted_action_shape": list(actions.shape),
                    "predicted_action_min": float(actions.min().item()),
                    "predicted_action_max": float(actions.max().item()),
                    "predicted_action_finite": bool(torch.isfinite(actions).all().item()),
                }
            )

        save_sample_outputs(output_dir, saved, sample, images, metadata)
        saved += 1

    print(f"Wrote {saved} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
