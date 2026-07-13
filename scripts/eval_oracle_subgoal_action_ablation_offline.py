#!/usr/bin/env python
"""Offline ablation for oracle-subgoal action conditioning.

Evaluates the same selected LIBERO samples under five input modes:

- no_subgoal: current observation + instruction only
- current_copy: current observation copied into the subgoal token block
- oracle_t10: correct GT filtered t+10 image
- same_task_wrong_t10: t+10 image from the same task but another demo
- cross_task_wrong_t10: t+10 image from a different task
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_action_training_distribution_offline import (  # noqa: E402
    bool_arg,
    infer_device,
    move_batch_to_device,
    resolve_model_path,
    to_jsonable,
)
from vila_u.constants import ACTION_NUM_BINS  # noqa: E402
from vila_u.data.libero_dataset_v2 import LiberoGoalDataset  # noqa: E402
from vila_u.model.builder import load_pretrained_model  # noqa: E402
from vila_u.train.train_action_prediction_main import (  # noqa: E402
    ActionPredictionArguments,
    DiscreteActionPredictionDataCollator,
    _compute_dataset_action_bin_edges_local,
    insert_subgoal_embeds_before_action_block,
    select_action_slot_token_ids,
)
from vila_u.utils.action_tokenizer import (  # noqa: E402
    compute_selected_token_logits,
    normalize_action_bin_edges,
    select_action_token_ids,
    token_ids_to_bins,
    undiscretize_action_bins,
)
from vila_u.utils.hybrid_attention import (  # noqa: E402
    build_action_token_position_mask,
    build_hybrid_attention_mask,
)


VALID_MODES = (
    "no_subgoal",
    "current_copy",
    "oracle_t10",
    "same_task_wrong_t10",
    "cross_task_wrong_t10",
)
MODE_ALIASES = {
    "oracle": "oracle_t10",
    "wrong": "cross_task_wrong_t10",
    "wrong_t10": "cross_task_wrong_t10",
}
NEGATIVE_MODES = {"same_task_wrong_t10", "cross_task_wrong_t10"}
LOWER_IS_BETTER = {"mae", "mse", "first_step_mae", "gripper_mae", "gripper_mse"}


def parse_modes(value: str) -> list[str]:
    requested_modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [
        mode
        for mode in requested_modes
        if mode not in VALID_MODES and mode not in MODE_ALIASES
    ]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown mode(s): {unknown}. Valid modes: {', '.join(VALID_MODES)}"
        )
    if not requested_modes:
        raise argparse.ArgumentTypeError("At least one mode is required")

    modes = []
    for requested_mode in requested_modes:
        mode = MODE_ALIASES.get(requested_mode, requested_mode)
        if mode not in modes:
            modes.append(mode)
    return modes


def select_balanced_sample_indices(
    samples: list[dict[str, Any]],
    max_samples_per_task: int,
    max_samples: int | None = None,
) -> list[int]:
    if max_samples_per_task <= 0:
        raise ValueError("max_samples_per_task must be positive")
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive when provided")
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_task[sample["file"]].append(index)
    selected_indices = []
    for task_file in sorted(indices_by_task):
        selected_indices.extend(indices_by_task[task_file][:max_samples_per_task])
    if max_samples is not None:
        selected_indices = selected_indices[: int(max_samples)]
    return selected_indices


def select_negative_subgoal_indices(
    source_samples: list[dict[str, Any]],
    source_indices: list[int],
    negative_samples: list[dict[str, Any]],
    selection: str,
    seed: int,
    same_task_min_filtered_distance: int,
) -> list[int]:
    """Select deterministic same-task or cross-task negative subgoals."""
    if same_task_min_filtered_distance < 0:
        raise ValueError("same_task_min_filtered_distance must be non-negative")
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(negative_samples):
        indices_by_task[sample["file"]].append(index)

    selected_negative_indices = []
    for source_index in source_indices:
        source = source_samples[source_index]
        if selection == "same_task":
            candidates = [
                index
                for index in indices_by_task.get(source["file"], [])
                if negative_samples[index]["demo"] != source["demo"]
            ]
            distant_candidates = [
                index
                for index in candidates
                if abs(
                    int(negative_samples[index]["filtered_timestep"])
                    - int(source["filtered_timestep"])
                )
                >= int(same_task_min_filtered_distance)
            ]
            candidates = distant_candidates or candidates
        elif selection == "cross_task":
            candidate_task_files = sorted(
                task_file
                for task_file in indices_by_task
                if task_file != source["file"] and indices_by_task[task_file]
            )
            if not candidate_task_files:
                candidates = []
            else:
                selection_key = int(source_index) + int(seed)
                task_file = candidate_task_files[selection_key % len(candidate_task_files)]
                task_candidates = indices_by_task[task_file]
                candidates = [
                    task_candidates[
                        (selection_key // len(candidate_task_files)) % len(task_candidates)
                    ]
                ]
        else:
            raise ValueError(f"Unsupported negative selection: {selection}")

        if not candidates:
            raise ValueError(
                f"{selection} negative requires an eligible candidate for "
                f"{source['file']}::{source['demo']}"
            )
        candidate_position = (int(source_index) + int(seed)) % len(candidates)
        selected_negative_indices.append(candidates[candidate_position])
    return selected_negative_indices


def clone_item(item: dict[str, Any]) -> dict[str, Any]:
    return dict(item)


def without_subgoal(item: dict[str, Any]) -> dict[str, Any]:
    cloned = clone_item(item)
    cloned.pop("subgoal_images", None)
    cloned.pop("subgoal_timestep", None)
    cloned.pop("subgoal_filtered_timestep", None)
    return cloned


def with_current_copy(item: dict[str, Any]) -> dict[str, Any]:
    cloned = clone_item(item)
    cloned["subgoal_images"] = item["observations"]
    cloned["subgoal_timestep"] = item.get("timestep", -1)
    cloned["subgoal_filtered_timestep"] = item.get("filtered_timestep", -1)
    return cloned


def with_negative_subgoal(item: dict[str, Any], negative_item: dict[str, Any]) -> dict[str, Any]:
    cloned = clone_item(item)
    cloned["subgoal_images"] = negative_item["subgoal_images"]
    cloned["subgoal_timestep"] = negative_item["subgoal_timestep"]
    cloned["subgoal_filtered_timestep"] = negative_item["subgoal_filtered_timestep"]
    return cloned


@torch.inference_mode()
def predict_training_batch_with_optional_subgoal(model, batch: dict[str, torch.Tensor]):
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

    if batch.get("subgoal_images") is not None:
        subgoal_embeds, _ = core_model.encode_images(
            batch["subgoal_images"],
            image_ids=None,
        )
        inputs_embeds, mm_labels, mm_attention_mask, position_ids, _ = (
            insert_subgoal_embeds_before_action_block(
                inputs_embeds=inputs_embeds,
                labels=mm_labels,
                attention_mask=mm_attention_mask,
                position_ids=position_ids,
                subgoal_embeds=subgoal_embeds,
                num_action_tokens=action_token_count,
            )
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
    action_labels = mm_labels[:, :, 0][action_position_mask].view(batch_size, action_token_count)
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


def validate_prediction_outputs(
    mode: str,
    pred_bins: torch.Tensor,
    pred_actions: torch.Tensor,
    batch_size: int,
    action_chunk_size: int,
    action_dim: int,
) -> None:
    expected_shape = (batch_size, action_chunk_size, action_dim)
    if tuple(pred_bins.shape) != expected_shape:
        raise ValueError(
            f"Mode {mode} returned pred_bins shape {tuple(pred_bins.shape)}, "
            f"expected {expected_shape}"
        )
    if tuple(pred_actions.shape) != expected_shape:
        raise ValueError(
            f"Mode {mode} returned actions shape {tuple(pred_actions.shape)}, "
            f"expected {expected_shape}"
        )
    if not torch.isfinite(pred_actions).all():
        raise ValueError(f"Mode {mode} produced non-finite action predictions")


def new_metric_store() -> dict[str, list[Any]]:
    return defaultdict(list)


def extend_metrics(
    store: dict[str, list[Any]],
    pred_bins: torch.Tensor,
    gt_bins: torch.Tensor,
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
) -> None:
    abs_error = (pred_actions - gt_actions).abs()
    sq_error = (pred_actions - gt_actions).square()
    token_matches = pred_bins.eq(gt_bins)

    store["mae"].extend(abs_error.mean(dim=(1, 2)).detach().cpu().tolist())
    store["mse"].extend(sq_error.mean(dim=(1, 2)).detach().cpu().tolist())
    store["first_step_mae"].extend(abs_error[:, 0, :].mean(dim=1).detach().cpu().tolist())
    store["per_dim_mae"].extend(abs_error.mean(dim=1).detach().cpu().numpy())
    store["per_horizon_mae"].extend(abs_error.mean(dim=2).detach().cpu().numpy())
    store["token_accuracy"].extend(token_matches.float().mean(dim=(1, 2)).detach().cpu().tolist())
    store["first_step_token_accuracy"].extend(
        token_matches[:, 0, :].float().mean(dim=1).detach().cpu().tolist()
    )
    store["first_step_exact_match"].extend(
        token_matches[:, 0, :].all(dim=1).float().detach().cpu().tolist()
    )
    store["chunk_exact_match"].extend(
        token_matches.flatten(1).all(dim=1).float().detach().cpu().tolist()
    )
    store["per_dim_token_accuracy"].extend(token_matches.float().mean(dim=1).detach().cpu().numpy())
    store["pred_min"].extend(pred_actions.amin(dim=(1, 2)).detach().cpu().tolist())
    store["pred_max"].extend(pred_actions.amax(dim=(1, 2)).detach().cpu().tolist())
    store["finite_rate"].extend(
        torch.isfinite(pred_actions).flatten(1).all(dim=1).float().detach().cpu().tolist()
    )

    if gt_actions.shape[-1] > 6:
        pred_gripper = pred_actions[:, :, 6]
        gt_gripper = gt_actions[:, :, 6]
        store["gripper_mae"].extend(abs_error[:, :, 6].mean(dim=1).detach().cpu().tolist())
        store["gripper_mse"].extend(sq_error[:, :, 6].mean(dim=1).detach().cpu().tolist())
        store["gripper_sign_accuracy"].extend(
            torch.sign(pred_gripper).eq(torch.sign(gt_gripper))
            .float()
            .mean(dim=1)
            .detach()
            .cpu()
            .tolist()
        )
        store["gripper_token_accuracy"].extend(
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
        store["gripper_close_recall"].extend(
            close_recall[torch.isfinite(close_recall)].detach().cpu().tolist()
        )
        store["gripper_pred_open_rate"].extend((pred_gripper > 0).float().mean(dim=1).detach().cpu().tolist())
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
            store["gripper_transition_change_recall"].extend(
                transition_recall[torch.isfinite(transition_recall)].detach().cpu().tolist()
            )


def mean_or_none(values: list[Any]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def summarize_metrics(store: dict[str, list[Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, values in store.items():
        if not values:
            summary[key] = None
        elif key == "pred_min":
            summary[key] = float(np.min(values))
        elif key == "pred_max":
            summary[key] = float(np.max(values))
        elif key in {"per_dim_mae", "per_horizon_mae", "per_dim_token_accuracy"}:
            summary[key] = np.mean(np.asarray(values), axis=0).tolist()
        else:
            summary[key] = mean_or_none(values)
    summary["num_samples"] = len(store.get("mae", []))
    return summary


def build_mode_items(
    mode: str,
    batch_items: list[dict[str, Any]],
    same_task_negative_items: list[dict[str, Any]],
    cross_task_negative_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if mode == "no_subgoal":
        return [without_subgoal(item) for item in batch_items]
    if mode == "current_copy":
        return [with_current_copy(item) for item in batch_items]
    if mode == "oracle_t10":
        return [clone_item(item) for item in batch_items]
    if mode == "same_task_wrong_t10":
        if len(same_task_negative_items) != len(batch_items):
            raise ValueError(
                "same-task mode requires one negative subgoal per evaluated sample"
            )
        return [
            with_negative_subgoal(item, negative_item)
            for item, negative_item in zip(batch_items, same_task_negative_items)
        ]
    if mode == "cross_task_wrong_t10":
        if len(cross_task_negative_items) != len(batch_items):
            raise ValueError(
                "cross-task mode requires one negative subgoal per evaluated sample"
            )
        return [
            with_negative_subgoal(item, negative_item)
            for item, negative_item in zip(batch_items, cross_task_negative_items)
        ]
    raise ValueError(f"Unsupported mode: {mode}")


def comparison_summary(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    pairs = [
        ("oracle_t10", "no_subgoal"),
        ("oracle_t10", "current_copy"),
        ("oracle_t10", "same_task_wrong_t10"),
        ("oracle_t10", "cross_task_wrong_t10"),
        ("current_copy", "no_subgoal"),
    ]
    metric_names = sorted({key for summary in summaries.values() for key in summary.keys()})
    for left, right in pairs:
        if left not in summaries or right not in summaries:
            continue
        key = f"{left}_minus_{right}"
        comparisons[key] = {}
        for metric in metric_names:
            left_value = summaries[left].get(metric)
            right_value = summaries[right].get(metric)
            if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
                continue
            comparisons[key][metric] = float(left_value - right_value)
            if metric in LOWER_IS_BETTER:
                comparisons[key][f"{metric}_improved"] = left_value < right_value
            elif metric.endswith("accuracy") or metric.endswith("recall") or metric.endswith("rate"):
                comparisons[key][f"{metric}_improved"] = left_value > right_value
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-subgoal action ablation eval.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument("--data-root", required=True, help="LIBERO dataset root.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto.")
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-task", type=int, default=500)
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
    parser.add_argument("--demo-start-index", type=int, default=40)
    parser.add_argument("--demo-end-index", type=int, default=50)
    parser.add_argument("--subgoal-offset", type=int, default=10)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=list(VALID_MODES),
        help="Comma-separated experiment-1 modes (legacy oracle/wrong aliases are accepted).",
    )
    parser.add_argument("--negative-seed", type=int, default=0)
    parser.add_argument("--same-task-min-filtered-distance", type=int, default=20)
    parser.add_argument("--recompute-action-bin-edges", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--save-records", action="store_true")
    args = parser.parse_args()
    if args.subgoal_offset != 10:
        parser.error("Experiment-1 t10 modes require --subgoal-offset 10")

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
        raise ValueError("Oracle-subgoal ablation requires hybrid-attention action checkpoints.")

    dataset = LiberoGoalDataset(
        data_root=args.data_root,
        image_processor=image_processor,
        tokenizer=tokenizer,
        action_chunk_size=action_chunk_size,
        image_size=args.image_size,
        remove_pause_intervals=args.remove_pause_intervals,
        pause_threshold=args.pause_threshold,
        include_subgoal_image=True,
        subgoal_min_offset=args.subgoal_offset,
        subgoal_max_offset=args.subgoal_offset,
        subgoal_sampling_strategy="fixed",
        max_task_files=args.max_task_files,
        max_demos_per_task=args.max_demos_per_task,
        demo_start_index=args.demo_start_index,
        demo_end_index=args.demo_end_index,
        task_file=args.task_file,
        task_file_pattern=args.task_file_pattern,
        gripper_pause_threshold=args.gripper_pause_threshold,
    )
    selected_indices = select_balanced_sample_indices(
        dataset.samples,
        max_samples_per_task=args.max_samples_per_task,
        max_samples=args.max_samples,
    )
    if not selected_indices:
        raise ValueError("No samples available for evaluation")
    eval_count = len(selected_indices)

    negative_dataset = None
    same_task_negative_indices = []
    cross_task_negative_indices = []
    if any(mode in NEGATIVE_MODES for mode in args.modes):
        negative_dataset = LiberoGoalDataset(
            data_root=args.data_root,
            image_processor=image_processor,
            tokenizer=tokenizer,
            action_chunk_size=action_chunk_size,
            image_size=args.image_size,
            remove_pause_intervals=args.remove_pause_intervals,
            pause_threshold=args.pause_threshold,
            include_subgoal_image=True,
            subgoal_min_offset=args.subgoal_offset,
            subgoal_max_offset=args.subgoal_offset,
            subgoal_sampling_strategy="fixed",
            max_demos_per_task=args.max_demos_per_task,
            demo_start_index=args.demo_start_index,
            demo_end_index=args.demo_end_index,
            gripper_pause_threshold=args.gripper_pause_threshold,
        )
    if "same_task_wrong_t10" in args.modes:
        if negative_dataset is None:
            raise AssertionError("Negative dataset was not initialized")
        same_task_negative_indices = select_negative_subgoal_indices(
            dataset.samples,
            selected_indices,
            negative_dataset.samples,
            selection="same_task",
            seed=args.negative_seed,
            same_task_min_filtered_distance=args.same_task_min_filtered_distance,
        )
    if "cross_task_wrong_t10" in args.modes:
        if negative_dataset is None:
            raise AssertionError("Negative dataset was not initialized")
        cross_task_negative_indices = select_negative_subgoal_indices(
            dataset.samples,
            selected_indices,
            negative_dataset.samples,
            selection="cross_task",
            seed=args.negative_seed,
            same_task_min_filtered_distance=args.same_task_min_filtered_distance,
        )

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
            demo_start_index=args.demo_start_index,
            demo_end_index=args.demo_end_index,
            task_file=args.task_file,
            task_file_pattern=args.task_file_pattern,
            gripper_pause_threshold=args.gripper_pause_threshold,
        )
        action_bin_edges = _compute_dataset_action_bin_edges_local(dataset, action_args)
    if action_bin_edges is None:
        raise ValueError(
            "Experiment-1 evaluation requires action_bin_edges from the checkpoint; "
            "use --recompute-action-bin-edges only for explicit diagnostics"
        )
    expected_bin_shape = (action_dim, ACTION_NUM_BINS + 1)
    if tuple(action_bin_edges.shape) != expected_bin_shape:
        raise ValueError(
            f"Expected action_bin_edges shape {expected_bin_shape}, "
            f"got {tuple(action_bin_edges.shape)}"
        )
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

    dataloader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=lambda batch: batch,
    )

    print("=" * 72)
    print("Oracle Subgoal Action Ablation")
    print("=" * 72)
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Eval samples: {eval_count}/{len(dataset)}")
    print(f"Samples per task: up to {args.max_samples_per_task}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Demo range: [{args.demo_start_index}, {args.demo_end_index})")
    print(f"Subgoal offset: {args.subgoal_offset}")
    print(f"Negative seed: {args.negative_seed}")
    print(f"Same-task minimum filtered distance: {args.same_task_min_filtered_distance}")
    print(f"Task file: {args.task_file}")
    print(f"Max demos per task: {args.max_demos_per_task}")
    print(f"Action bin edges: {'present' if action_bin_edges is not None else 'None'}")
    print()

    metrics = {mode: new_metric_store() for mode in args.modes}
    records = []
    processed = 0
    for batch_items in tqdm(dataloader, total=len(dataloader)):
        cur_indices = selected_indices[processed : processed + len(batch_items)]
        same_negative_batch_indices = same_task_negative_indices[
            processed : processed + len(batch_items)
        ]
        cross_negative_batch_indices = cross_task_negative_indices[
            processed : processed + len(batch_items)
        ]
        same_negative_items = (
            [negative_dataset[index] for index in same_negative_batch_indices]
            if same_negative_batch_indices
            else []
        )
        cross_negative_items = (
            [negative_dataset[index] for index in cross_negative_batch_indices]
            if cross_negative_batch_indices
            else []
        )
        expected_action_labels = torch.stack(
            [item["action_labels"] for item in batch_items]
        )

        for mode in args.modes:
            mode_items = build_mode_items(
                mode,
                batch_items,
                same_negative_items,
                cross_negative_items,
            )
            if mode == "current_copy":
                for source_item, mode_item in zip(batch_items, mode_items):
                    if not torch.equal(
                        source_item["observations"],
                        mode_item["subgoal_images"],
                    ):
                        raise AssertionError("current_copy must reuse observations exactly")
            batch = collator(mode_items)
            if not torch.equal(batch["action_labels"], expected_action_labels):
                raise AssertionError(f"Action labels changed in mode {mode}")
            batch = move_batch_to_device(batch, device)
            pred_bins, gt_bins = predict_training_batch_with_optional_subgoal(model, batch)
            pred_actions = undiscretize_action_bins(
                pred_bins,
                num_bins=ACTION_NUM_BINS,
                bin_edges=action_bin_edges.to(pred_bins.device),
            ).float()
            validate_prediction_outputs(
                mode,
                pred_bins,
                pred_actions,
                batch_size=len(batch_items),
                action_chunk_size=action_chunk_size,
                action_dim=action_dim,
            )
            gt_actions = batch["action_labels"].float()
            extend_metrics(metrics[mode], pred_bins, gt_bins, pred_actions, gt_actions)

            if args.save_records:
                pred_np = pred_actions.detach().cpu().numpy()
                gt_np = gt_actions.detach().cpu().numpy()
                pred_bins_np = pred_bins.detach().cpu().numpy()
                gt_bins_np = gt_bins.detach().cpu().numpy()
                subgoal_timesteps = batch.get("subgoal_timesteps")
                subgoal_timesteps_np = (
                    subgoal_timesteps.detach().cpu().numpy()
                    if subgoal_timesteps is not None
                    else [None] * len(cur_indices)
                )
                subgoal_filtered_timesteps = batch.get("subgoal_filtered_timesteps")
                subgoal_filtered_timesteps_np = (
                    subgoal_filtered_timesteps.detach().cpu().numpy()
                    if subgoal_filtered_timesteps is not None
                    else [None] * len(cur_indices)
                )
                for item_offset, sample_index in enumerate(cur_indices):
                    sample = dataset.samples[sample_index]
                    record = {
                        "mode": mode,
                        "index": sample_index,
                        "file": sample["file"],
                        "demo": sample["demo"],
                        "timestep": int(sample["timestep"]),
                        "filtered_timestep": int(sample["filtered_timestep"]),
                        "subgoal_timestep": (
                            None
                            if subgoal_timesteps_np[item_offset] is None
                            else int(subgoal_timesteps_np[item_offset])
                        ),
                        "subgoal_filtered_timestep": (
                            None
                            if subgoal_filtered_timesteps_np[item_offset] is None
                            else int(subgoal_filtered_timesteps_np[item_offset])
                        ),
                        "prediction": pred_np[item_offset],
                        "ground_truth": gt_np[item_offset],
                        "pred_bins": pred_bins_np[item_offset],
                        "gt_bins": gt_bins_np[item_offset],
                    }
                    if mode in NEGATIVE_MODES:
                        if mode == "same_task_wrong_t10":
                            negative_index = same_negative_batch_indices[item_offset]
                            negative_item = same_negative_items[item_offset]
                            negative_selection = "same_task"
                        else:
                            negative_index = cross_negative_batch_indices[item_offset]
                            negative_item = cross_negative_items[item_offset]
                            negative_selection = "cross_task"
                        negative_sample = negative_dataset.samples[negative_index]
                        record.update(
                            {
                                "negative_file": negative_sample["file"],
                                "negative_demo": negative_sample["demo"],
                                "negative_timestep": int(negative_item["subgoal_timestep"]),
                                "negative_filtered_timestep": int(
                                    negative_item["subgoal_filtered_timestep"]
                                ),
                                "negative_selection": negative_selection,
                            }
                        )
                    records.append(record)
        processed += len(batch_items)

    summaries = {mode: summarize_metrics(store) for mode, store in metrics.items()}
    comparisons = comparison_summary(summaries)
    output = {
        "summary": summaries,
        "comparisons": comparisons,
        "config": {
            "resolved_model_path": resolved_model_path,
            "data_root": args.data_root,
            "task_file": args.task_file,
            "task_file_pattern": args.task_file_pattern,
            "max_task_files": args.max_task_files,
            "max_demos_per_task": args.max_demos_per_task,
            "max_samples": args.max_samples,
            "max_samples_per_task": args.max_samples_per_task,
            "modes": list(args.modes),
            "demo_start_index": args.demo_start_index,
            "demo_end_index": args.demo_end_index,
            "subgoal_offset": args.subgoal_offset,
            "negative_seed": args.negative_seed,
            "same_task_min_filtered_distance": args.same_task_min_filtered_distance,
            "remove_pause_intervals": args.remove_pause_intervals,
            "pause_threshold": args.pause_threshold,
            "gripper_pause_threshold": args.gripper_pause_threshold,
            "action_bin_edges_present": action_bin_edges is not None,
        },
    }
    if args.save_records:
        output["records"] = records

    print("Summary")
    key_metrics = [
        "first_step_mae",
        "token_accuracy",
        "first_step_token_accuracy",
        "gripper_sign_accuracy",
        "gripper_close_recall",
        "gripper_transition_change_recall",
    ]
    for mode in args.modes:
        print(f"  [{mode}]")
        for metric in key_metrics:
            print(f"    {metric} = {summaries[mode].get(metric)}")
    if comparisons:
        print("Comparisons")
        for name, values in comparisons.items():
            print(f"  [{name}]")
            for metric in key_metrics:
                if metric in values:
                    print(f"    {metric}_delta = {values[metric]}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, default=to_jsonable)
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
