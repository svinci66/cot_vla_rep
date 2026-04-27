#!/usr/bin/env python
"""
离线评估脚本 - 完全基于训练代码的数据处理流程
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from vila_u.constants import ACTION_DIM, ACTION_CHUNK_SIZE, DEFAULT_IMAGE_TOKEN
from vila_u.model.builder import load_pretrained_model
from vila_u.data.libero_dataset import LiberoGoalDataset, collate_fn


def evaluate_offline(
    model_path: str,
    checkpoint_path: str,
    data_path: str,
    batch_size: int = 8,
    num_samples: int = 100,
    device: str = "cuda",
):
    print("=" * 70)
    print("VILA-U Offline Evaluation")
    print("=" * 70)

    # 1. 加载模型（完全按照训练代码）
    print("\n[1/3] Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        device_map=device,
    )

    # 加载checkpoint
    if os.path.isdir(checkpoint_path):
        print(f"  Loading from checkpoint directory: {checkpoint_path}")
        llm_path = os.path.join(checkpoint_path, 'llm')
        if os.path.exists(llm_path):
            print(f"  Loading LLM from {llm_path}")
            llm_state = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16
            ).state_dict()
            model.get_llm().load_state_dict(llm_state, strict=False)

        mm_proj_path = os.path.join(checkpoint_path, 'mm_projector', 'pytorch_model.bin')
        if os.path.exists(mm_proj_path):
            print(f"  Loading mm_projector from {mm_proj_path}")
            model.get_mm_projector().load_state_dict(
                torch.load(mm_proj_path, map_location=device),
                strict=False
            )

    model.eval()
    print(f"  ✓ Model loaded")

    # 2. 加载数据集（完全按照训练代码）
    print("\n[2/3] Loading dataset...")
    dataset = LiberoGoalDataset(
        data_root=data_path,
        image_size=256,
        action_chunk_size=ACTION_CHUNK_SIZE,
    )

    if num_samples is not None:
        num_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"  ✓ Dataset loaded: {len(dataset)} samples")

    # 3. 评估（完全按照训练代码的compute_loss）
    print(f"\n[3/3] Evaluating...")

    all_l1_errors = []
    all_mse_errors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            observations = batch['observations'].to(device)  # [B, 3, 256, 256]
            instructions = batch['instructions']  # List[str]
            action_labels = batch['action_labels'].to(device)  # [B, 10, 7]

            # 1. 构建输入文本（添加图像占位符）
            prompts = [f"{DEFAULT_IMAGE_TOKEN}\n{inst}" for inst in instructions]

            # 2. Tokenize 文本
            inputs = model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # 3. 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=observations,
                output_hidden_states=True,
                return_dict=True,
            )

            # 4. 获取隐层状态
            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]

            # 5. 预测动作
            action_pred = model.predict_actions(hidden_states)  # [B, 10, 7]

            # 6. 计算误差
            l1_error = nn.functional.l1_loss(action_pred, action_labels, reduction='none')
            l1_error = l1_error.mean(dim=[1, 2])  # [B]

            mse_error = nn.functional.mse_loss(action_pred, action_labels, reduction='none')
            mse_error = mse_error.mean(dim=[1, 2])  # [B]

            all_l1_errors.extend(l1_error.cpu().numpy())
            all_mse_errors.extend(mse_error.cpu().numpy())

    # 4. 输出结果
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    mean_l1 = np.mean(all_l1_errors)
    mean_mse = np.mean(all_mse_errors)
    rmse = np.sqrt(mean_mse)

    print(f"\nMean L1 Error (MAE): {mean_l1:.6f}")
    print(f"Mean Squared Error (MSE): {mean_mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

    print("\n" + "=" * 70)

    return {
        'l1_error': mean_l1,
        'mse': mean_mse,
        'rmse': rmse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    evaluate_offline(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
