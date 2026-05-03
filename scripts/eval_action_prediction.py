#!/usr/bin/env python
"""
VILA-U Action Prediction Offline Evaluation
基于 train_action_prediction_main.py 的评测脚本
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vila_u.data.libero_dataset import LiberoGoalDataset
from vila_u.model import VILAULlamaModel
from vila_u.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    ACTION_NUM_BINS,
)
from vila_u.utils.tokenizer import tokenize_conversation
from vila_u.utils.action_tokenizer import (
    actions_to_token_ids,
    token_ids_to_actions,
    compute_selected_token_logits,
    select_action_token_ids,
)
from vila_u.utils.hybrid_attention import build_hybrid_attention_mask


def collate_fn_discrete(batch, tokenizer, mm_use_im_start_end, action_token_ids, action_slot_token_id, num_action_tokens):
    """离散动作预测的collate函数"""
    images = torch.stack([item["observation"] for item in batch])
    actions = torch.stack([item["action_labels"] for item in batch])

    image_token = DEFAULT_IMAGE_TOKEN
    if mm_use_im_start_end:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    input_id_list = []
    label_list = []

    for item, action in zip(batch, actions):
        # 构建prompt
        prompt_ids = tokenize_conversation(
            [{"from": "human", "value": f"{image_token}\n{item['instruction']}"}],
            tokenizer,
            add_generation_prompt=True,
        )

        # 转换动作为token IDs
        action_token_ids_flat = actions_to_token_ids(
            action.view(-1),
            action_token_ids,
            num_bins=ACTION_NUM_BINS,
        )

        # 构建输入：prompt + action slots
        action_input_ids = torch.full_like(action_token_ids_flat, action_slot_token_id)
        full_input_ids = torch.cat([prompt_ids, action_input_ids])

        # 构建标签：忽略prompt部分，只计算action部分的loss
        labels = torch.cat([
            torch.full_like(prompt_ids, -100),
            action_token_ids_flat,
        ])

        input_id_list.append(full_input_ids)
        label_list.append(labels)

    # Padding
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        label_list, batch_first=True, padding_value=-100
    )
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images,
        "action_labels": actions,
    }


def evaluate(args):
    print("=" * 70)
    print("VILA-U Action Prediction Evaluation")
    print("=" * 70)

    device = torch.device(args.device)

    # 1. 加载模型
    print("\n[1/3] Loading model...")
    model = VILAULlamaModel.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    model.eval()
    print(f"  ✓ Model loaded from {args.checkpoint_path}")

    # 获取action token IDs
    action_token_ids = select_action_token_ids(tokenizer, ACTION_NUM_BINS)
    action_slot_token_id = tokenizer.convert_tokens_to_ids("<action>")
    num_action_tokens = args.action_chunk_size * args.action_dim

    # 2. 加载数据集
    print("\n[2/3] Loading dataset...")
    dataset = LiberoGoalDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        action_chunk_size=args.action_chunk_size,
    )

    if args.num_samples is not None:
        num_samples = min(args.num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn_discrete(
            batch, tokenizer, True, action_token_ids, action_slot_token_id, num_action_tokens
        ),
    )

    print(f"  ✓ Dataset loaded: {len(dataset)} samples")

    # 3. 评估
    print(f"\n[3/3] Evaluating...")

    all_token_accuracy = []
    all_action_mse = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)
            true_actions = batch["action_labels"].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                output_hidden_states=True,
                return_dict=True,
            )

            # 提取action token位置的logits
            action_mask = labels != -100
            if not action_mask.any():
                continue

            hidden_states = outputs.hidden_states[-1]
            action_hidden = hidden_states[action_mask.unsqueeze(-1).expand_as(hidden_states)].view(-1, hidden_states.size(-1))

            # 计算action token logits
            action_logits = compute_selected_token_logits(
                action_hidden.unsqueeze(0),
                model.get_llm().lm_head,
                action_token_ids,
            )

            # 预测的token IDs
            pred_bins = torch.argmax(action_logits, dim=-1).view(-1)
            true_token_ids = labels[action_mask]

            # Token准确率
            token_acc = (pred_bins == token_ids_to_bins(true_token_ids, action_token_ids, ACTION_NUM_BINS)).float().mean()
            all_token_accuracy.append(token_acc.item())

            # 转换为连续动作并计算MSE
            pred_actions = token_ids_to_actions(pred_bins, action_token_ids, ACTION_NUM_BINS)
            pred_actions = pred_actions.view(true_actions.shape)

            mse = torch.nn.functional.mse_loss(pred_actions, true_actions)
            all_action_mse.append(mse.item())

    # 4. 输出结果
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    mean_token_acc = np.mean(all_token_accuracy)
    mean_mse = np.mean(all_action_mse)

    print(f"\nToken Accuracy: {mean_token_acc*100:.2f}%")
    print(f"Action MSE: {mean_mse:.6f}")
    print(f"Action RMSE: {np.sqrt(mean_mse):.6f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained checkpoint")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to LIBERO dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
