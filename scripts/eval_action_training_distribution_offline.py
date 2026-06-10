#!/usr/bin/env python
"""Offline action eval that matches the action-training data path.

This script intentionally evaluates through ``LiberoGoalDataset`` and
``DiscreteActionPredictionDataCollator`` instead of scanning raw HDF5 timesteps.
That keeps evaluation aligned with training-time preprocessing:

- LIBERO 180-degree image rotation
- no-op/pause filtering, including gripper-change preservation
- action chunks sampled from ``non_pause_indices``
- the same prompt/action-slot layout and hybrid-attention forward pass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vila_u.constants import ACTION_NUM_BINS
from vila_u.data.libero_dataset_v2 import LiberoGoalDataset
from vila_u.model.builder import load_pretrained_model
from vila_u.train.train_action_prediction_main import (
    ActionPredictionArguments,
    DiscreteActionPredictionDataCollator,
    _compute_dataset_action_bin_edges_local,
    select_action_slot_token_ids,
)
from vila_u.train.utils import get_checkpoint_path
from vila_u.utils.action_tokenizer import (
    compute_selected_token_logits,
    normalize_action_bin_edges,
    select_action_token_ids,
    token_ids_to_bins,
    undiscretize_action_bins,
)
from vila_u.utils.hybrid_attention import (
    build_action_token_position_mask,
    build_hybrid_attention_mask,
)


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


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def infer_device(model, requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    return next(model.parameters()).device


def summarize_float(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


@torch.inference_mode()
def predict_training_batch(model, batch: dict[str, torch.Tensor]):
    core_model = model
    while hasattr(core_model, "module"):
        core_model = core_model.module

    action_token_count = int(core_model.config.action_chunk_size) * int(core_model.config.action_dim)
    (
        _,
        position_ids,
        mm_attention_mask,
        past_key_values,
        inputs_embeds,
        mm_labels,
    ) = core_model.prepare_inputs_labels_for_multimodal(
        input_ids=batch["input_ids"],
        position_ids=None,
        attention_mask=batch["attention_mask"],
        past_key_values=None,
        labels=batch["labels"],
        images=batch["images"],
    )

    if getattr(core_model.config, "use_hybrid_attention", False):
        attention_mask = build_hybrid_attention_mask(
            mm_attention_mask,
            num_action_tokens=action_token_count,
            dtype=inputs_embeds.dtype,
        )
    else:
        attention_mask = mm_attention_mask

    outputs = core_model.llm.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        seqlens_in_batch=None,
    )

    action_position_mask = build_action_token_position_mask(
        mm_attention_mask,
        num_action_tokens=action_token_count,
    )
    batch_size = outputs.last_hidden_state.shape[0]
    action_hidden_states = outputs.last_hidden_state[action_position_mask].view(
        batch_size,
        action_token_count,
        outputs.last_hidden_state.size(-1),
    )
    action_labels = mm_labels[:, :, 0][action_position_mask].view(
        batch_size,
        action_token_count,
    )
    action_logits = compute_selected_token_logits(
        action_hidden_states,
        core_model.llm.lm_head,
        core_model.config.action_token_ids,
    )
    pred_bins = action_logits.argmax(dim=-1).view(
        batch_size,
        int(core_model.config.action_chunk_size),
        int(core_model.config.action_dim),
    )
    gt_bins = token_ids_to_bins(
        action_labels,
        core_model.config.action_token_ids,
    ).view_as(pred_bins)
    return pred_bins, gt_bins


def main():
    parser = argparse.ArgumentParser(
        description="Strict offline eval on the same distribution as action training."
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument("--data-root", required=True, help="LIBERO dataset root.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto.")
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--remove-pause-intervals", type=bool_arg, default=True)
    parser.add_argument("--pause-threshold", type=float, default=0.01)
    parser.add_argument("--gripper-pause-threshold", type=float, default=1e-6)
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--task-file-pattern", default=None)
    parser.add_argument("--max-task-files", type=int, default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument(
        "--recompute-action-bin-edges",
        action="store_true",
        help="Recompute percentile bins from the selected eval dataset if checkpoint has no saved edges.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--save-records", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved_model_path = resolve_model_path(args.model_path)
    load_device = "cuda" if args.device == "auto" else args.device
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=resolved_model_path,
        model_dtype=dtype_map[args.model_dtype],
        device=load_device,
    )
    model.eval()
    device = infer_device(model, args.device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = int(args.model_max_length or context_len)

    action_chunk_size = int(model.config.action_chunk_size)
    action_dim = int(model.config.action_dim)
    action_token_ids = getattr(model.config, "action_token_ids", None)
    if action_token_ids is None:
        action_token_ids = select_action_token_ids(tokenizer, ACTION_NUM_BINS)
        model.config.action_token_ids = action_token_ids
    action_slot_token_ids = getattr(model.config, "action_slot_token_ids", None)
    if action_slot_token_ids is None:
        action_slot_token_ids = select_action_slot_token_ids(tokenizer, action_token_ids, model=model)
        model.config.action_slot_token_ids = action_slot_token_ids
    if not getattr(model.config, "use_hybrid_attention", False):
        raise ValueError(
            "Strict training-distribution eval currently supports the paper-aligned "
            "hybrid-attention action path only."
        )

    dataset = LiberoGoalDataset(
        data_root=args.data_root,
        image_processor=image_processor,
        tokenizer=tokenizer,
        action_chunk_size=action_chunk_size,
        image_size=args.image_size,
        remove_pause_intervals=args.remove_pause_intervals,
        pause_threshold=args.pause_threshold,
        include_subgoal_image=False,
        max_task_files=args.max_task_files,
        max_demos_per_task=args.max_demos_per_task,
        task_file=args.task_file,
        task_file_pattern=args.task_file_pattern,
        gripper_pause_threshold=args.gripper_pause_threshold,
    )
    eval_count = min(int(args.max_samples), len(dataset))
    if eval_count <= 0:
        raise ValueError("No samples available for evaluation")
    eval_dataset = Subset(dataset, range(eval_count))

    action_bin_edges = normalize_action_bin_edges(
        getattr(model.config, "action_bin_edges", None),
        device="cpu",
    )
    if action_bin_edges is None and args.recompute_action_bin_edges:
        action_args = ActionPredictionArguments(
            data_root=args.data_root,
            action_chunk_size=action_chunk_size,
            action_dim=action_dim,
            image_size=args.image_size,
            remove_pause_intervals=args.remove_pause_intervals,
            pause_threshold=args.pause_threshold,
            max_task_files=args.max_task_files,
            max_demos_per_task=args.max_demos_per_task,
            task_file=args.task_file,
            task_file_pattern=args.task_file_pattern,
            gripper_pause_threshold=args.gripper_pause_threshold,
        )
        action_bin_edges = _compute_dataset_action_bin_edges_local(dataset, action_args)
    model.config.action_bin_edges = action_bin_edges.cpu().tolist() if action_bin_edges is not None else None

    collator = DiscreteActionPredictionDataCollator(
        tokenizer=tokenizer,
        model_max_length=int(args.model_max_length or context_len),
        mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
        action_token_ids=action_token_ids,
        action_slot_token_ids=action_slot_token_ids,
        action_chunk_size=action_chunk_size,
        action_dim=action_dim,
        use_hybrid_attention=bool(getattr(model.config, "use_hybrid_attention", False)),
        action_bin_edges=action_bin_edges,
    )

    def collate_with_continuous_actions(batch):
        output = collator(batch)
        output["action_labels"] = torch.stack([item["action_labels"] for item in batch])
        return output

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_with_continuous_actions,
    )

    print("=" * 72)
    print("Strict Action Offline Evaluation")
    print("=" * 72)
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Eval samples: {eval_count}/{len(dataset)}")
    print(f"Task file: {args.task_file}")
    print(f"Task file pattern: {args.task_file_pattern}")
    print(f"Max task files: {args.max_task_files}")
    print(f"Max demos per task: {args.max_demos_per_task}")
    print(f"Remove pause intervals: {args.remove_pause_intervals}")
    print(f"Pause thresholds: motion={args.pause_threshold}, gripper={args.gripper_pause_threshold}")
    print(f"Action shape: ({action_chunk_size}, {action_dim})")
    print(f"use_hybrid_attention = {getattr(model.config, 'use_hybrid_attention', None)}")
    print(f"action_bin_edges = {'present' if action_bin_edges is not None else 'None'}")
    print()

    maes = []
    mses = []
    first_step_maes = []
    per_dim_abs_errors = []
    per_horizon_abs_errors = []
    token_accuracies = []
    first_step_token_accuracies = []
    first_step_exact_matches = []
    chunk_exact_matches = []
    per_dim_token_accuracies = []
    gripper_maes = []
    gripper_mses = []
    gripper_sign_matches = []
    gripper_token_accuracies = []
    gripper_close_recalls = []
    gripper_transition_change_recalls = []
    gripper_pred_open_rates = []
    pred_mins = []
    pred_maxes = []
    finite_flags = []
    records = []

    processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch = move_batch_to_device(batch, device)
        pred_bins, gt_bins = predict_training_batch(model, batch)
        pred_actions = undiscretize_action_bins(
            pred_bins,
            num_bins=ACTION_NUM_BINS,
            bin_edges=action_bin_edges.to(pred_bins.device) if action_bin_edges is not None else None,
        ).float()
        gt_actions = batch["action_labels"].float()

        abs_error = (pred_actions - gt_actions).abs()
        sq_error = (pred_actions - gt_actions).square()
        token_matches = pred_bins.eq(gt_bins)
        batch_size = gt_actions.shape[0]

        maes.extend(abs_error.mean(dim=(1, 2)).detach().cpu().tolist())
        mses.extend(sq_error.mean(dim=(1, 2)).detach().cpu().tolist())
        first_step_maes.extend(abs_error[:, 0, :].mean(dim=1).detach().cpu().tolist())
        per_dim_abs_errors.extend(abs_error.mean(dim=1).detach().cpu().numpy())
        per_horizon_abs_errors.extend(abs_error.mean(dim=2).detach().cpu().numpy())
        token_accuracies.extend(token_matches.float().mean(dim=(1, 2)).detach().cpu().tolist())
        first_step_token_accuracies.extend(token_matches[:, 0, :].float().mean(dim=1).detach().cpu().tolist())
        first_step_exact_matches.extend(token_matches[:, 0, :].all(dim=1).float().detach().cpu().tolist())
        chunk_exact_matches.extend(token_matches.flatten(1).all(dim=1).float().detach().cpu().tolist())
        per_dim_token_accuracies.extend(token_matches.float().mean(dim=1).detach().cpu().numpy())
        pred_mins.extend(pred_actions.amin(dim=(1, 2)).detach().cpu().tolist())
        pred_maxes.extend(pred_actions.amax(dim=(1, 2)).detach().cpu().tolist())
        finite_flags.extend(torch.isfinite(pred_actions).flatten(1).all(dim=1).detach().cpu().tolist())

        if action_dim > 6:
            pred_gripper = pred_actions[:, :, 6]
            gt_gripper = gt_actions[:, :, 6]
            gripper_maes.extend(abs_error[:, :, 6].mean(dim=1).detach().cpu().tolist())
            gripper_mses.extend(sq_error[:, :, 6].mean(dim=1).detach().cpu().tolist())
            gripper_sign_matches.extend(
                torch.sign(pred_gripper).eq(torch.sign(gt_gripper))
                .float()
                .mean(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
            gripper_token_accuracies.extend(
                token_matches[:, :, 6].float().mean(dim=1).detach().cpu().tolist()
            )
            pred_close = pred_gripper < 0
            gt_close = gt_gripper < 0
            close_counts = gt_close.float().sum(dim=1)
            close_hits = (pred_close & gt_close).float().sum(dim=1)
            close_recall = torch.where(
                close_counts > 0,
                close_hits / close_counts.clamp_min(1.0),
                torch.full_like(close_counts, float("nan")),
            )
            gripper_close_recalls.extend(
                close_recall[torch.isfinite(close_recall)].detach().cpu().tolist()
            )
            pred_open = pred_gripper > 0
            gripper_pred_open_rates.extend(pred_open.float().mean(dim=1).detach().cpu().tolist())
            if pred_gripper.shape[1] > 1:
                pred_transition = torch.sign(pred_gripper[:, 1:]).ne(torch.sign(pred_gripper[:, :-1]))
                gt_transition = torch.sign(gt_gripper[:, 1:]).ne(torch.sign(gt_gripper[:, :-1]))
                transition_counts = gt_transition.float().sum(dim=1)
                transition_hits = (pred_transition & gt_transition).float().sum(dim=1)
                transition_recall = torch.where(
                    transition_counts > 0,
                    transition_hits / transition_counts.clamp_min(1.0),
                    torch.full_like(transition_counts, float("nan")),
                )
                gripper_transition_change_recalls.extend(
                    transition_recall[torch.isfinite(transition_recall)].detach().cpu().tolist()
                )

        if args.save_records:
            sample_start = processed
            pred_np = pred_actions.detach().cpu().numpy()
            gt_np = gt_actions.detach().cpu().numpy()
            pred_bins_np = pred_bins.detach().cpu().numpy()
            gt_bins_np = gt_bins.detach().cpu().numpy()
            for item_idx in range(batch_size):
                sample = dataset.samples[sample_start + item_idx]
                records.append(
                    {
                        "index": sample_start + item_idx,
                        "file": sample["file"],
                        "demo": sample["demo"],
                        "timestep": int(sample["timestep"]),
                        "filtered_timestep": int(sample["filtered_timestep"]),
                        "prediction": pred_np[item_idx],
                        "ground_truth": gt_np[item_idx],
                        "pred_bins": pred_bins_np[item_idx],
                        "gt_bins": gt_bins_np[item_idx],
                    }
                )
        processed += batch_size
        _ = batch_idx

    summary = {
        "num_samples": len(maes),
        "mae": summarize_float(maes),
        "mse": summarize_float(mses),
        "first_step_mae": summarize_float(first_step_maes),
        "per_dim_mae": np.mean(np.asarray(per_dim_abs_errors), axis=0).tolist(),
        "per_horizon_mae": np.mean(np.asarray(per_horizon_abs_errors), axis=0).tolist(),
        "token_accuracy": summarize_float(token_accuracies),
        "first_step_token_accuracy": summarize_float(first_step_token_accuracies),
        "first_step_exact_match": summarize_float(first_step_exact_matches),
        "chunk_exact_match": summarize_float(chunk_exact_matches),
        "per_dim_token_accuracy": np.mean(np.asarray(per_dim_token_accuracies), axis=0).tolist(),
        "gripper_mae": summarize_float(gripper_maes),
        "gripper_mse": summarize_float(gripper_mses),
        "gripper_sign_accuracy": summarize_float(gripper_sign_matches),
        "gripper_token_accuracy": summarize_float(gripper_token_accuracies),
        "gripper_close_recall": summarize_float(gripper_close_recalls),
        "gripper_transition_change_recall": summarize_float(gripper_transition_change_recalls),
        "gripper_pred_open_rate": summarize_float(gripper_pred_open_rates),
        "pred_min": float(np.min(pred_mins)),
        "pred_max": float(np.max(pred_maxes)),
        "finite_rate": summarize_float([float(flag) for flag in finite_flags]),
        "resolved_model_path": resolved_model_path,
        "data_root": args.data_root,
        "task_file": args.task_file,
        "task_file_pattern": args.task_file_pattern,
        "max_task_files": args.max_task_files,
        "max_demos_per_task": args.max_demos_per_task,
        "remove_pause_intervals": args.remove_pause_intervals,
        "pause_threshold": args.pause_threshold,
        "gripper_pause_threshold": args.gripper_pause_threshold,
        "action_bin_edges_present": action_bin_edges is not None,
    }
    output = {"summary": summary}
    if args.save_records:
        output["records"] = records

    print("Summary")
    for key, value in summary.items():
        print(f"  {key} = {value}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, default=to_jsonable)
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
