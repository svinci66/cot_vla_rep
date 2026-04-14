#!/usr/bin/env python
"""
Compare predicted actions against ground-truth actions for one LIBERO sample.

Outputs:
- Ground-truth continuous actions
- Predicted continuous actions
- Ground-truth action tokens
- Predicted action tokens
- Per-step and overall L1 errors
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import torch

from vila_u.constants import (
    ACTION_NUM_BINS,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from vila_u.model.builder import load_pretrained_model
from vila_u.train.utils import get_checkpoint_path
from vila_u.utils.action_tokenizer import (
    actions_to_token_ids,
    bins_to_token_ids,
    compute_selected_token_logits,
)
from vila_u.utils.hybrid_attention import build_hybrid_attention_mask
from vila_u.utils.tokenizer import tokenize_conversation


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_dir():
        checkpoint_path, _ = get_checkpoint_path(str(path))
        if checkpoint_path is not None:
            return checkpoint_path
        return str(path)

    raise FileNotFoundError(f"Model path does not exist: {path}")


def load_one_sample(data_root: str, file_index: int, demo_index: int, timestep: int, chunk_size: int):
    hdf5_files = sorted(
        filename for filename in os.listdir(data_root) if filename.endswith(".hdf5")
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_root}")

    file_index = min(file_index, len(hdf5_files) - 1)
    data_file = os.path.join(data_root, hdf5_files[file_index])

    with h5py.File(data_file, "r") as f:
        instruction = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
        demo_names = sorted(f["data"].keys())
        demo_index = min(demo_index, len(demo_names) - 1)
        demo_name = demo_names[demo_index]
        image = f["data"][demo_name]["obs/agentview_rgb"][timestep]
        actions = f["data"][demo_name]["actions"][timestep : timestep + chunk_size]

    return {
        "data_file": data_file,
        "demo_name": demo_name,
        "instruction": instruction,
        "image": image,
        "actions": torch.from_numpy(actions).float(),
    }


def build_prompt(model, instruction: str) -> str:
    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    return f"{image_token}\n{instruction}"


def main():
    parser = argparse.ArgumentParser(description="Compare predicted and GT actions for one sample.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--data-root",
        default="/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=0)
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        device=args.device,
    )
    model.eval()

    sample = load_one_sample(
        data_root=args.data_root,
        file_index=args.file_index,
        demo_index=args.demo_index,
        timestep=args.timestep,
        chunk_size=model.config.action_chunk_size,
    )

    gt_actions = sample["actions"].clamp(-1.0, 1.0)
    pred_actions = model.predict_action(
        image=sample["image"],
        instruction=sample["instruction"],
        image_processor=image_processor,
    ).cpu()

    gt_action_tokens = actions_to_token_ids(
        gt_actions.view(-1),
        model.config.action_token_ids,
        num_bins=ACTION_NUM_BINS,
    ).view(model.config.action_chunk_size, model.config.action_dim)

    prompt = build_prompt(model, sample["instruction"])
    input_ids = tokenize_conversation(
        [{"from": "human", "value": prompt}],
        tokenizer,
        add_generation_prompt=True,
    ).unsqueeze(0).to(args.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    image_tensor = image_processor.preprocess(sample["image"], return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(next(model.parameters()).device)

    num_action_tokens = model.config.action_chunk_size * model.config.action_dim
    if getattr(model.config, "use_hybrid_attention", False):
        action_slot_token_id = model.config.action_slot_token_id
        action_slots = torch.full(
            (1, num_action_tokens),
            fill_value=action_slot_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        full_input_ids = torch.cat([input_ids, action_slots], dim=1)
        full_attention_mask = full_input_ids.ne(tokenizer.pad_token_id)
        (
            _,
            position_ids,
            mm_attention_mask,
            past_key_values,
            inputs_embeds,
            _,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=full_input_ids,
            position_ids=None,
            attention_mask=full_attention_mask,
            past_key_values=None,
            labels=None,
            images=image_tensor,
        )
        hybrid_attention_mask = build_hybrid_attention_mask(
            mm_attention_mask,
            num_action_tokens=num_action_tokens,
            dtype=inputs_embeds.dtype,
        )
        outputs = model.llm.model(
            input_ids=None,
            attention_mask=hybrid_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            seqlens_in_batch=None,
        )
        action_hidden_states = outputs.last_hidden_state[:, -num_action_tokens:, :]
        action_logits = compute_selected_token_logits(
            action_hidden_states,
            model.llm.lm_head,
            model.config.action_token_ids,
        )
        pred_bins = torch.argmax(action_logits, dim=-1)
        pred_action_tokens = bins_to_token_ids(pred_bins, model.config.action_token_ids).view(
            model.config.action_chunk_size,
            model.config.action_dim,
        ).cpu()
    else:
        raise RuntimeError("This comparison script is intended for Phase 3 hybrid attention checkpoints.")

    abs_diff = (pred_actions - gt_actions).abs()
    step_l1 = abs_diff.mean(dim=-1)
    overall_l1 = float(abs_diff.mean())

    print("=" * 100)
    print("Phase 3 Action Comparison")
    print("=" * 100)
    print(f"Model path: {resolved_model_path}")
    print(f"Sample file: {sample['data_file']}")
    print(f"Demo: {sample['demo_name']}")
    print(f"Timestep: {args.timestep}")
    print(f"Instruction: {sample['instruction']}")
    print()

    print("Ground-truth continuous actions:")
    print(gt_actions)
    print()
    print("Predicted continuous actions:")
    print(pred_actions)
    print()
    print("Absolute diff:")
    print(abs_diff)
    print()
    print("Per-step mean L1:")
    print(step_l1)
    print(f"Overall mean L1: {overall_l1:.6f}")
    print()

    print("Ground-truth action tokens:")
    print(gt_action_tokens)
    print()
    print("Predicted action tokens:")
    print(pred_action_tokens)
    print()
    print("Token exact match ratio:")
    token_match = (gt_action_tokens == pred_action_tokens).float().mean()
    print(f"{float(token_match):.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
