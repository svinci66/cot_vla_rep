#!/usr/bin/env python
"""Check whether action predictions are sensitive to LIBERO task language.

The diagnostic keeps the image fixed and swaps in every LIBERO-Goal instruction.
It reports whether the correct instruction ranks best against the ground-truth
action chunk, and how much predicted actions change across instructions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.eval_action_training_distribution_offline import (
    bool_arg,
    infer_device,
    move_batch_to_device,
    predict_training_batch,
    resolve_model_path,
    to_jsonable,
)
from vila_u.constants import ACTION_NUM_BINS
from vila_u.data.libero_dataset_v2 import LiberoGoalDataset
from vila_u.model.builder import load_pretrained_model
from vila_u.train.train_action_prediction_main import (
    ActionPredictionArguments,
    DiscreteActionPredictionDataCollator,
    _compute_dataset_action_bin_edges_local,
    select_action_slot_token_ids,
)
from vila_u.utils.action_tokenizer import (
    normalize_action_bin_edges,
    select_action_token_ids,
    undiscretize_action_bins,
)


LIBERO_GOAL_TASKS = [
    ("open_the_middle_drawer_of_the_cabinet_demo.hdf5", "open the middle drawer of the cabinet"),
    ("put_the_bowl_on_the_stove_demo.hdf5", "put the bowl on the stove"),
    ("put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5", "put the wine bottle on top of the cabinet"),
    ("open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5", "open the top drawer and put the bowl inside"),
    ("put_the_bowl_on_top_of_the_cabinet_demo.hdf5", "put the bowl on top of the cabinet"),
    ("push_the_plate_to_the_front_of_the_stove_demo.hdf5", "push the plate to the front of the stove"),
    ("put_the_cream_cheese_in_the_bowl_demo.hdf5", "put the cream cheese in the bowl"),
    ("turn_on_the_stove_demo.hdf5", "turn on the stove"),
    ("put_the_bowl_on_the_plate_demo.hdf5", "put the bowl on the plate"),
    ("put_the_wine_bottle_on_the_rack_demo.hdf5", "put the wine bottle on the rack"),
]


def natural_key(value: str):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def task_id_from_file(path: str) -> int | None:
    name = Path(path).name
    for task_id, (task_file, _) in enumerate(LIBERO_GOAL_TASKS):
        if name == task_file:
            return task_id
    return None


def select_balanced_indices(dataset: LiberoGoalDataset, samples_per_demo: int) -> list[int]:
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        task_id = task_id_from_file(sample["file"])
        if task_id is None:
            continue
        buckets[(task_id, sample["demo"])].append(idx)

    selected = []
    for key in sorted(buckets):
        indices = buckets[key]
        count = min(samples_per_demo, len(indices))
        if count <= 0:
            continue
        positions = np.linspace(0, len(indices) - 1, count, dtype=int)
        selected.extend(indices[int(pos)] for pos in positions)
    return selected


def summarize(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_int(values: list[int]) -> float | None:
    return float(np.mean(values)) if values else None


def clone_with_instruction(item: dict[str, Any], instruction: str) -> dict[str, Any]:
    cloned = dict(item)
    cloned["instructions"] = instruction
    return cloned


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal LIBERO-Goal action semantic sensitivity check.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=5)
    parser.add_argument("--samples-per-demo", type=int, default=5)
    parser.add_argument("--remove-pause-intervals", type=bool_arg, default=True)
    parser.add_argument("--pause-threshold", type=float, default=0.01)
    parser.add_argument("--gripper-pause-threshold", type=float, default=1e-6)
    parser.add_argument("--recompute-action-bin-edges", action="store_true")
    parser.add_argument("--save-examples", type=int, default=20)
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
        raise ValueError("Semantic sensitivity check requires hybrid-attention action checkpoints.")

    dataset = LiberoGoalDataset(
        data_root=args.data_root,
        image_processor=image_processor,
        tokenizer=tokenizer,
        action_chunk_size=action_chunk_size,
        image_size=args.image_size,
        remove_pause_intervals=args.remove_pause_intervals,
        pause_threshold=args.pause_threshold,
        include_subgoal_image=False,
        max_task_files=None,
        max_demos_per_task=args.max_demos_per_task,
        task_file=None,
        task_file_pattern=None,
        gripper_pause_threshold=args.gripper_pause_threshold,
    )
    selected_indices = select_balanced_indices(dataset, args.samples_per_demo)
    if not selected_indices:
        raise ValueError("No LIBERO-Goal samples selected for semantic sensitivity check")

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
            max_demos_per_task=args.max_demos_per_task,
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

    base_subset = Subset(dataset, selected_indices)
    base_loader = DataLoader(
        base_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=lambda batch: batch,
    )

    print("=" * 72)
    print("Action Semantic Sensitivity Check")
    print("=" * 72)
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Selected base samples: {len(selected_indices)}")
    print(f"Instructions per sample: {len(LIBERO_GOAL_TASKS)}")
    print(f"Total predictions: {len(selected_indices) * len(LIBERO_GOAL_TASKS)}")
    print(f"Action bin edges: {'present' if action_bin_edges is not None else 'None'}")

    overall = {
        "correct_best": [],
        "correct_rank": [],
        "correct_mae": [],
        "wrong_mae_mean": [],
        "correct_wrong_mae_gap": [],
        "token_diff_rate": [],
        "action_l2_diff": [],
        "first_step_l2_diff": [],
        "gripper_sign_diff_rate": [],
    }
    per_task: dict[int, dict[str, list[float] | list[int]]] = {
        task_id: defaultdict(list) for task_id in range(len(LIBERO_GOAL_TASKS))
    }
    examples = []

    for base_counter, batch_items in enumerate(tqdm(base_loader, total=len(base_subset))):
        base_item = batch_items[0]
        original_sample = dataset.samples[selected_indices[base_counter]]
        correct_task_id = task_id_from_file(original_sample["file"])
        if correct_task_id is None:
            continue

        variant_items = [
            clone_with_instruction(base_item, instruction)
            for _, instruction in LIBERO_GOAL_TASKS
        ]
        batch = collator(variant_items)
        batch["action_labels"] = torch.stack([item["action_labels"] for item in variant_items])
        batch = move_batch_to_device(batch, device)

        pred_bins, gt_bins = predict_training_batch(model, batch)
        pred_actions = undiscretize_action_bins(
            pred_bins,
            num_bins=ACTION_NUM_BINS,
            bin_edges=action_bin_edges.to(pred_bins.device) if action_bin_edges is not None else None,
        ).float()
        gt_actions = batch["action_labels"].float()
        gt_action = gt_actions[0]
        gt_bin = gt_bins[0]

        abs_error = (pred_actions - gt_action.unsqueeze(0)).abs()
        mae_by_instruction = abs_error.mean(dim=(1, 2))
        token_acc_by_instruction = pred_bins.eq(gt_bin.unsqueeze(0)).float().mean(dim=(1, 2))
        order = torch.argsort(mae_by_instruction, stable=True)
        rank = int((order == int(correct_task_id)).nonzero(as_tuple=False)[0].item() + 1)
        correct_mae = float(mae_by_instruction[correct_task_id].detach().cpu())
        wrong_mask = torch.ones(len(LIBERO_GOAL_TASKS), dtype=torch.bool, device=mae_by_instruction.device)
        wrong_mask[correct_task_id] = False
        wrong_mae_mean = float(mae_by_instruction[wrong_mask].mean().detach().cpu())
        correct_wrong_gap = wrong_mae_mean - correct_mae

        correct_pred_bins = pred_bins[correct_task_id]
        correct_pred_actions = pred_actions[correct_task_id]
        token_diff = pred_bins.ne(correct_pred_bins.unsqueeze(0)).float().mean(dim=(1, 2))
        action_l2 = torch.linalg.vector_norm(
            (pred_actions - correct_pred_actions.unsqueeze(0)).flatten(1),
            dim=1,
        )
        first_step_l2 = torch.linalg.vector_norm(
            pred_actions[:, 0, :] - correct_pred_actions[0].unsqueeze(0),
            dim=1,
        )
        if action_dim > 6:
            gripper_sign_diff = torch.sign(pred_actions[:, :, 6]).ne(
                torch.sign(correct_pred_actions[:, 6]).unsqueeze(0)
            ).float().mean(dim=1)
        else:
            gripper_sign_diff = torch.zeros_like(token_diff)

        wrong_token_diff = float(token_diff[wrong_mask].mean().detach().cpu())
        wrong_action_l2 = float(action_l2[wrong_mask].mean().detach().cpu())
        wrong_first_step_l2 = float(first_step_l2[wrong_mask].mean().detach().cpu())
        wrong_gripper_sign_diff = float(gripper_sign_diff[wrong_mask].mean().detach().cpu())
        correct_best = int(rank == 1)

        values = {
            "correct_best": correct_best,
            "correct_rank": rank,
            "correct_mae": correct_mae,
            "wrong_mae_mean": wrong_mae_mean,
            "correct_wrong_mae_gap": correct_wrong_gap,
            "token_diff_rate": wrong_token_diff,
            "action_l2_diff": wrong_action_l2,
            "first_step_l2_diff": wrong_first_step_l2,
            "gripper_sign_diff_rate": wrong_gripper_sign_diff,
        }
        for key, value in values.items():
            overall[key].append(value)
            per_task[correct_task_id][key].append(value)

        if len(examples) < args.save_examples:
            examples.append(
                {
                    "sample_index": selected_indices[base_counter],
                    "task_id": correct_task_id,
                    "task_file": Path(original_sample["file"]).name,
                    "demo": original_sample["demo"],
                    "timestep": int(original_sample["timestep"]),
                    "filtered_timestep": int(original_sample["filtered_timestep"]),
                    "correct_rank": rank,
                    "correct_mae": correct_mae,
                    "wrong_mae_mean": wrong_mae_mean,
                    "mae_by_instruction": mae_by_instruction.detach().cpu(),
                    "token_acc_by_instruction": token_acc_by_instruction.detach().cpu(),
                    "pred_bins_by_instruction": pred_bins.detach().cpu(),
                }
            )

    overall_summary = {
        "num_base_samples": len(overall["correct_rank"]),
        "num_instructions": len(LIBERO_GOAL_TASKS),
        "num_predictions": len(overall["correct_rank"]) * len(LIBERO_GOAL_TASKS),
        "correct_best_rate": summarize_int(overall["correct_best"]),
        "mean_correct_rank": summarize_int(overall["correct_rank"]),
        "mean_correct_mae": summarize(overall["correct_mae"]),
        "mean_wrong_mae": summarize(overall["wrong_mae_mean"]),
        "mean_correct_wrong_mae_gap": summarize(overall["correct_wrong_mae_gap"]),
        "mean_token_diff_rate_vs_correct": summarize(overall["token_diff_rate"]),
        "mean_action_l2_diff_vs_correct": summarize(overall["action_l2_diff"]),
        "mean_first_step_l2_diff_vs_correct": summarize(overall["first_step_l2_diff"]),
        "mean_gripper_sign_diff_rate_vs_correct": summarize(overall["gripper_sign_diff_rate"]),
    }
    per_task_summary = []
    for task_id, (_, instruction) in enumerate(LIBERO_GOAL_TASKS):
        stats = per_task[task_id]
        per_task_summary.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "num_base_samples": len(stats["correct_rank"]),
                "correct_best_rate": summarize_int(stats["correct_best"]),
                "mean_correct_rank": summarize_int(stats["correct_rank"]),
                "mean_correct_mae": summarize(stats["correct_mae"]),
                "mean_wrong_mae": summarize(stats["wrong_mae_mean"]),
                "mean_correct_wrong_mae_gap": summarize(stats["correct_wrong_mae_gap"]),
                "mean_token_diff_rate_vs_correct": summarize(stats["token_diff_rate"]),
                "mean_action_l2_diff_vs_correct": summarize(stats["action_l2_diff"]),
                "mean_gripper_sign_diff_rate_vs_correct": summarize(stats["gripper_sign_diff_rate"]),
            }
        )

    output = {
        "summary": overall_summary,
        "per_task": per_task_summary,
        "examples": examples,
        "resolved_model_path": resolved_model_path,
        "data_root": args.data_root,
        "max_demos_per_task": args.max_demos_per_task,
        "samples_per_demo": args.samples_per_demo,
        "remove_pause_intervals": args.remove_pause_intervals,
        "pause_threshold": args.pause_threshold,
        "gripper_pause_threshold": args.gripper_pause_threshold,
        "action_bin_edges_present": action_bin_edges is not None,
    }

    print("Summary")
    for key, value in overall_summary.items():
        print(f"  {key} = {value}")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=to_jsonable), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
