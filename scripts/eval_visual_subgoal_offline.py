#!/usr/bin/env python
"""Offline evaluation for generated Visual CoT subgoal images.

This evaluates the subgoal generator on demonstration trajectories where the
expert future frame is known. It answers a narrower question than online
rollout: does the generated subgoal move closer to the expert future image than
the current observation does?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
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


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


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
    indices: list[int] = []
    previous_action = None
    for timestep, action in enumerate(actions):
        if not is_noop_action(action, previous_action, pause_threshold, gripper_pause_threshold):
            indices.append(timestep)
        previous_action = action
    return indices


def list_hdf5_files(
    data_root: str,
    task_file: str | None,
    task_file_pattern: str | None,
    max_task_files: int | None,
) -> list[str]:
    if task_file:
        files = [task_file]
    else:
        files = sorted(filename for filename in os.listdir(data_root) if filename.endswith(".hdf5"))
        if task_file_pattern:
            files = [filename for filename in files if task_file_pattern in filename]
    if max_task_files is not None:
        files = files[: max(0, max_task_files)]
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_root}")
    return [str(Path(data_root) / filename) for filename in files]


def iter_samples(args: argparse.Namespace):
    files = list_hdf5_files(
        args.data_root,
        args.task_file,
        args.task_file_pattern,
        args.max_task_files,
    )
    for data_file in files:
        with h5py.File(data_file, "r") as h5_file:
            instruction = json.loads(h5_file["data"].attrs["problem_info"])["language_instruction"]
            demo_names = sorted(h5_file["data"].keys())
            if args.max_demos_per_task is not None:
                demo_names = demo_names[: max(0, args.max_demos_per_task)]

            for demo_name in demo_names:
                demo = h5_file["data"][demo_name]
                actions = demo["actions"][:]
                num_frames = len(actions)
                if args.subgoal_offset_mode == "filtered_action_step":
                    non_pause_indices = compute_non_pause_indices(
                        actions,
                        pause_threshold=args.pause_threshold,
                        gripper_pause_threshold=args.gripper_pause_threshold,
                    )
                    if len(non_pause_indices) <= 1:
                        continue
                    max_filtered_start = max(0, len(non_pause_indices) - 1)
                    if args.max_samples_per_demo is not None:
                        max_filtered_start = min(max_filtered_start, args.max_samples_per_demo)
                    for filtered_t in range(max_filtered_start):
                        raw_t = int(non_pause_indices[filtered_t])
                        target_filtered_t = min(filtered_t + args.subgoal_offset, len(non_pause_indices) - 1)
                        subgoal_t = int(non_pause_indices[target_filtered_t])
                        yield {
                            "data_file": data_file,
                            "task_file": Path(data_file).name,
                            "demo_name": demo_name,
                            "instruction": instruction,
                            "timestep": raw_t,
                            "filtered_timestep": filtered_t,
                            "subgoal_timestep": subgoal_t,
                            "image": demo["obs/agentview_rgb"][raw_t],
                            "future_image": demo["obs/agentview_rgb"][subgoal_t],
                            "subgoal_offset_mode": "filtered_action_step",
                        }
                else:
                    max_start = max(0, num_frames - 1)
                    if args.max_samples_per_demo is not None:
                        max_start = min(max_start, args.max_samples_per_demo)
                    for raw_t in range(max_start):
                        subgoal_t = min(raw_t + args.subgoal_offset, num_frames - 1)
                        yield {
                            "data_file": data_file,
                            "task_file": Path(data_file).name,
                            "demo_name": demo_name,
                            "instruction": instruction,
                            "timestep": raw_t,
                            "filtered_timestep": None,
                            "subgoal_timestep": subgoal_t,
                            "image": demo["obs/agentview_rgb"][raw_t],
                            "future_image": demo["obs/agentview_rgb"][subgoal_t],
                            "subgoal_offset_mode": "raw_frame",
                        }


def to_libero_uint8_image(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        tensor = array.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.float().numpy()

    if array.dtype != np.uint8:
        if float(np.nanmax(array)) <= 1.0 and float(np.nanmin(array)) >= 0.0:
            array = array * 255.0
        elif float(np.nanmin(array)) < 0.0:
            array = (array + 1.0) * 127.5
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 4:
        array = array[0]
    return np.ascontiguousarray(array)


def resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).convert("RGB").resize((size, size)))


def image_metrics(current: np.ndarray, generated: np.ndarray, future: np.ndarray, image_size: int) -> dict[str, float]:
    current = resize_uint8(current, image_size).astype(np.float32) / 255.0
    generated = resize_uint8(generated, image_size).astype(np.float32) / 255.0
    future = resize_uint8(future, image_size).astype(np.float32) / 255.0

    current_to_future_mae = float(np.mean(np.abs(current - future)))
    generated_to_future_mae = float(np.mean(np.abs(generated - future)))
    generated_to_current_mae = float(np.mean(np.abs(generated - current)))
    current_to_future_mse = float(np.mean((current - future) ** 2))
    generated_to_future_mse = float(np.mean((generated - future) ** 2))

    return {
        "current_to_future_mae": current_to_future_mae,
        "generated_to_future_mae": generated_to_future_mae,
        "generated_to_current_mae": generated_to_current_mae,
        "future_improvement_mae": current_to_future_mae - generated_to_future_mae,
        "current_to_future_mse": current_to_future_mse,
        "generated_to_future_mse": generated_to_future_mse,
        "future_improvement_mse": current_to_future_mse - generated_to_future_mse,
    }


def image_to_model_tensor(image: np.ndarray, image_processor: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rotated = rotate_libero_image_180(image)
    pil_image = Image.fromarray(rotated.astype(np.uint8)).convert("RGB")
    return image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)


@torch.inference_mode()
def encode_gt_codes(model, image_processor: Any, future_image: np.ndarray) -> torch.Tensor:
    core_model = model.module if hasattr(model, "module") else model
    vision_model = core_model.vision_tower.vision_tower
    vision_param = next(core_model.vision_tower.parameters())
    image_tensor = image_to_model_tensor(
        future_image,
        image_processor=image_processor,
        device=vision_param.device,
        dtype=vision_param.dtype,
    )
    gt_codes, _ = vision_model.rqvaesiglip.encode_image(image_tensor)
    return gt_codes.reshape(gt_codes.shape[0], -1, gt_codes.shape[-1]).long()


def code_metrics(generated_codes: torch.Tensor, gt_codes: torch.Tensor) -> dict[str, Any]:
    generated = generated_codes.detach().cpu().long()
    gt = gt_codes.detach().cpu().long()
    if generated.ndim == 4:
        generated = generated.reshape(generated.shape[0], -1, generated.shape[-1])
    if gt.ndim == 4:
        gt = gt.reshape(gt.shape[0], -1, gt.shape[-1])

    seq_len = min(generated.shape[1], gt.shape[1])
    depth = min(generated.shape[2], gt.shape[2])
    generated = generated[:, :seq_len, :depth]
    gt = gt[:, :seq_len, :depth]
    matches = generated.eq(gt)
    per_depth = matches.float().mean(dim=(0, 1)).tolist()
    per_token = matches.float().mean(dim=-1)
    return {
        "visual_token_accuracy": float(matches.float().mean().item()),
        "visual_token_accuracy_per_codebook": [float(x) for x in per_depth],
        "visual_patch_all_codebooks_accuracy": float(per_token.eq(1.0).float().mean().item()),
        "visual_code_seq_len": int(seq_len),
        "visual_code_depth": int(depth),
    }


def make_panel(images: list[tuple[str, np.ndarray]], width: int = 256) -> Image.Image:
    label_height = 28
    panel = Image.new("RGB", (width * len(images), width + label_height), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, image) in enumerate(images):
        cur = Image.fromarray(image).convert("RGB").resize((width, width))
        x = idx * width
        panel.paste(cur, (x, label_height))
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
    return panel


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "visual_token_accuracy",
        "visual_patch_all_codebooks_accuracy",
        "current_to_future_mae",
        "generated_to_future_mae",
        "generated_to_current_mae",
        "future_improvement_mae",
        "current_to_future_mse",
        "generated_to_future_mse",
        "future_improvement_mse",
    ]
    summary: dict[str, Any] = {"num_samples": len(records)}
    for key in metric_keys:
        values = [float(record[key]) for record in records if key in record and record[key] is not None]
        if values:
            summary[key] = float(np.mean(values))

    per_codebook = [
        record.get("visual_token_accuracy_per_codebook")
        for record in records
        if record.get("visual_token_accuracy_per_codebook") is not None
    ]
    if per_codebook:
        max_depth = max(len(values) for values in per_codebook)
        summary["visual_token_accuracy_per_codebook"] = [
            float(np.mean([values[idx] for values in per_codebook if idx < len(values)]))
            for idx in range(max_depth)
        ]

    future_positive = [
        float(record["future_improvement_mae"]) > 0
        for record in records
        if "future_improvement_mae" in record
    ]
    if future_positive:
        summary["future_improvement_positive_rate"] = float(np.mean(future_positive))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline generated-subgoal evaluation on LIBERO demos.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument("--data-root", required=True, help="LIBERO dataset root.")
    parser.add_argument("--output-dir", default="outputs/offline_visual_subgoal_eval")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto.")
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-samples-per-demo", type=int, default=None)
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--task-file-pattern", default=None)
    parser.add_argument("--max-task-files", type=int, default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--selection", choices=("sequential", "high-delta"), default="sequential")
    parser.add_argument("--subgoal-offset", type=int, default=10)
    parser.add_argument(
        "--subgoal-offset-mode",
        choices=("filtered_action_step", "raw_frame"),
        default="filtered_action_step",
    )
    parser.add_argument("--pause-threshold", type=float, default=0.01)
    parser.add_argument("--gripper-pause-threshold", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--save-panels", type=bool_arg, default=True)
    parser.add_argument("--save-images", type=bool_arg, default=False)
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved_model_path = resolve_model_path(args.model_path)
    load_device = "cuda" if args.device == "auto" else args.device
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        model_dtype=dtype_map[args.model_dtype],
        device=load_device,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model.llm.pad_token_id = tokenizer.pad_token_id

    output_dir = Path(args.output_dir)
    panels_dir = output_dir / "panels"
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_panels:
        panels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model_path,
        "data_root": args.data_root,
        "selection": args.selection,
        "subgoal_offset": args.subgoal_offset,
        "subgoal_offset_mode": args.subgoal_offset_mode,
        "pause_threshold": args.pause_threshold,
        "gripper_pause_threshold": args.gripper_pause_threshold,
        "cfg": args.cfg,
        "max_samples": args.max_samples,
        "task_file": args.task_file,
        "task_file_pattern": args.task_file_pattern,
        "max_task_files": args.max_task_files,
        "max_demos_per_task": args.max_demos_per_task,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    candidate_samples = []
    sample_iter = iter_samples(args)
    if args.selection == "high-delta":
        for sample in tqdm(sample_iter, desc="scoring high-delta samples"):
            metrics = image_metrics(
                rotate_libero_image_180(sample["image"]),
                rotate_libero_image_180(sample["image"]),
                rotate_libero_image_180(sample["future_image"]),
                image_size=args.image_size,
            )
            sample["delta_mae"] = metrics["current_to_future_mae"]
            candidate_samples.append(sample)
        candidate_samples.sort(key=lambda item: item["delta_mae"], reverse=True)
        samples = candidate_samples[: args.max_samples]
    else:
        samples = []
        for sample in sample_iter:
            samples.append(sample)
            if len(samples) >= args.max_samples:
                break

    records: list[dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as records_file:
        for sample_idx, sample in enumerate(tqdm(samples, desc="evaluating subgoals")):
            with torch.inference_mode():
                subgoal_embeds, generated_codes, generated_subgoal = model.generate_visual_cot_subgoal(
                    image=sample["image"],
                    instruction=sample["instruction"],
                    image_processor=image_processor,
                    cfg=args.cfg,
                    return_image=True,
                )
                del subgoal_embeds
                gt_codes = encode_gt_codes(model, image_processor, sample["future_image"])

            current_rgb = to_libero_uint8_image(rotate_libero_image_180(sample["image"]))
            future_rgb = to_libero_uint8_image(rotate_libero_image_180(sample["future_image"]))
            # generate_visual_cot_subgoal returns debug images in LIBERO RGB space.
            # Rotate it back to the model/training view before comparing with
            # current/future frames, which are also shown in the rotated view.
            generated_rgb = to_libero_uint8_image(rotate_libero_image_180(generated_subgoal))

            img_metrics = image_metrics(
                current=current_rgb,
                generated=generated_rgb,
                future=future_rgb,
                image_size=args.image_size,
            )
            tok_metrics = code_metrics(generated_codes, gt_codes)
            record = {
                "sample_index": sample_idx,
                "task_file": sample["task_file"],
                "data_file": sample["data_file"],
                "demo_name": sample["demo_name"],
                "instruction": sample["instruction"],
                "timestep": sample["timestep"],
                "filtered_timestep": sample["filtered_timestep"],
                "subgoal_timestep": sample["subgoal_timestep"],
                "selection": args.selection,
                "subgoal_offset": args.subgoal_offset,
                "subgoal_offset_mode": sample["subgoal_offset_mode"],
                "generated_code_shape": list(generated_codes.shape),
                "gt_code_shape": list(gt_codes.shape),
                **tok_metrics,
                **img_metrics,
            }
            if "delta_mae" in sample:
                record["delta_mae"] = float(sample["delta_mae"])

            if args.save_panels:
                panel = make_panel(
                    [
                        ("current", current_rgb),
                        ("generated", generated_rgb),
                        ("gt_future", future_rgb),
                    ],
                    width=args.image_size,
                )
                panel_path = panels_dir / (
                    f"{sample_idx:04d}_{Path(sample['task_file']).stem}_"
                    f"{sample['demo_name']}_t{sample['timestep']:04d}_"
                    f"f{sample['subgoal_timestep']:04d}.png"
                )
                panel.save(panel_path)
                record["panel_path"] = str(panel_path)

            if args.save_images:
                sample_image_dir = images_dir / f"sample_{sample_idx:04d}"
                sample_image_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(current_rgb).save(sample_image_dir / "current.png")
                Image.fromarray(generated_rgb).save(sample_image_dir / "generated.png")
                Image.fromarray(future_rgb).save(sample_image_dir / "gt_future.png")
                record["image_dir"] = str(sample_image_dir)

            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_file.flush()

    summary = summarize(records)
    summary.update(run_metadata)
    summary["records_path"] = str(records_path)
    if args.save_panels:
        summary["panels_dir"] = str(panels_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
