#!/usr/bin/env python
"""
Phase 2 discrete-action inference validation.

Checks:
1. Config-level validation
2. Single-sample predict_action() validation
3. Token-level constrained generation validation
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import torch

from vila_u.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from vila_u.model.builder import load_pretrained_model
from vila_u.train.utils import get_checkpoint_path
from vila_u.utils.action_tokenizer import (
    AllowedActionTokensLogitsProcessor,
    bins_to_token_ids,
    compute_selected_token_logits,
    token_ids_to_actions,
)
from vila_u.utils.hybrid_attention import build_hybrid_attention_mask
from vila_u.utils.tokenizer import tokenize_conversation


def load_one_libero_sample(data_root: str, file_index: int, demo_index: int, timestep: int):
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

    return {
        "data_file": data_file,
        "demo_name": demo_name,
        "instruction": instruction,
        "image": image,
    }


def build_prompt(model, instruction: str) -> str:
    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    return f"{image_token}\n{instruction}"


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

    raise FileNotFoundError(
        f"Model path does not exist: {path}. "
        "Pass a final model directory, an output directory containing checkpoint-* folders, "
        "or a concrete checkpoint directory."
    )


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 2 discrete-action inference.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained Phase 2 checkpoint/output directory.",
    )
    parser.add_argument(
        "--data-root",
        default="/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal",
        help="Path to LIBERO dataset root.",
    )
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--file-index", type=int, default=0, help="Which hdf5 file to sample.")
    parser.add_argument("--demo-index", type=int, default=0, help="Which demo in the file to sample.")
    parser.add_argument("--timestep", type=int, default=0, help="Which timestep to sample.")
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model_path)

    sample = load_one_libero_sample(
        data_root=args.data_root,
        file_index=args.file_index,
        demo_index=args.demo_index,
        timestep=args.timestep,
    )

    print("=" * 70)
    print("Phase 2 Discrete Inference Validation")
    print("=" * 70)
    print(f"Requested model path: {args.model_path}")
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Sample file: {sample['data_file']}")
    print(f"Demo: {sample['demo_name']}")
    print(f"Timestep: {args.timestep}")
    print()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        device=args.device,
    )
    model.eval()

    print("[1/3] Config check")
    print(f"  use_discrete_action_prediction = {getattr(model.config, 'use_discrete_action_prediction', None)}")
    action_token_ids = getattr(model.config, "action_token_ids", None)
    print(f"  num_action_tokens = {0 if action_token_ids is None else len(action_token_ids)}")
    print(f"  action_chunk_size = {model.config.action_chunk_size}")
    print(f"  action_dim = {model.config.action_dim}")
    assert getattr(model.config, "use_discrete_action_prediction", False), "Model is not in discrete-action mode"
    assert action_token_ids is not None and len(action_token_ids) == 256, "Expected 256 action token ids"
    print("  ✓ Config looks correct")
    print()

    print("[2/3] predict_action() check")
    actions = model.predict_action(
        image=sample["image"],
        instruction=sample["instruction"],
        image_processor=image_processor,
    )
    print(f"  predict_action shape = {tuple(actions.shape)}")
    print(f"  value range = [{float(actions.min()):.4f}, {float(actions.max()):.4f}]")
    print(f"  has_nan = {bool(torch.isnan(actions).any())}")
    assert actions.shape == (model.config.action_chunk_size, model.config.action_dim)
    assert not torch.isnan(actions).any()
    print("  ✓ predict_action() returned a valid action chunk")
    print()

    print("[3/3] Token-level generation check")
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
        action_slot_token_id = getattr(model.config, "action_slot_token_id", None)
        assert action_slot_token_id is not None, "Hybrid attention requires action_slot_token_id"

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
            action_token_ids,
        )
        predicted_bins = torch.argmax(action_logits, dim=-1)
        generated_action_ids = bins_to_token_ids(predicted_bins, action_token_ids)
    else:
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=num_action_tokens,
            use_cache=True,
            logits_processor=[AllowedActionTokensLogitsProcessor(action_token_ids)],
        )
        generated_action_ids = output_ids[:, -num_action_tokens:]
    allowed_ids = set(action_token_ids)
    all_valid = all(int(token_id) in allowed_ids for token_id in generated_action_ids.view(-1).tolist())
    decoded_actions = token_ids_to_actions(
        generated_action_ids,
        action_token_ids,
    ).view(model.config.action_chunk_size, model.config.action_dim)
    max_diff = float((decoded_actions - actions).abs().max())

    print(f"  generated token shape = {tuple(generated_action_ids.shape)}")
    print(f"  all generated ids are action tokens = {all_valid}")
    print(f"  decoded action shape = {tuple(decoded_actions.shape)}")
    print(f"  max diff vs predict_action() = {max_diff:.6f}")
    assert all_valid, "Found non-action token in constrained generation output"
    assert decoded_actions.shape == actions.shape
    print("  ✓ Constrained generation only emits action tokens")
    print()

    print("=" * 70)
    print("Phase 2 discrete inference validation passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
