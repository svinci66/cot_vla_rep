#!/usr/bin/env python
"""
简化的离线评估脚本 - 评估动作预测准确率
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from vila_u.data.libero_dataset import LiberoGoalDataset, collate_fn
from vila_u.model.builder import load_pretrained_model
from vila_u.mm_utils import process_images
from transformers import AutoModelForCausalLM


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

    # 1. 加载模型
    print("\n[1/3] Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        device=device,
    )

    # 加载checkpoint
    import os
    if os.path.isdir(checkpoint_path):
        print(f"  Loading from checkpoint directory: {checkpoint_path}")
        llm_path = os.path.join(checkpoint_path, 'llm')
        if os.path.exists(llm_path):
            print(f"  Loading LLM from {llm_path}")
            llm_state = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16).state_dict()
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

    # 2. 加载数据集
    print("\n[2/3] Loading dataset...")
    dataset = LiberoGoalDataset(
        data_root=data_path,
        image_size=256,
        action_chunk_size=10,
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

    # 3. 评估
    print(f"\n[3/3] Evaluating...")

    all_mse = []
    all_mae = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            observations = batch['observations'].to(device)  # [B, 3, 256, 256]
            instructions = batch['instructions']
            true_actions = batch['action_labels'].to(device)  # [B, chunk_size, 7]

            B = observations.shape[0]

            # 处理图像 - 直接使用image_processor
            images = image_processor.preprocess(observations, return_tensors='pt')['pixel_values']
            images = images.to(dtype=model.dtype, device=device)

            # 构建输入文本
            prompts = [f"Instruction: {inst}\nAction:" for inst in instructions]
            input_ids = [tokenizer(p, return_tensors='pt').input_ids[0] for p in prompts]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

            # 前向传播获取hidden states
            outputs = model(
                input_ids=input_ids,
                images=images,
                return_dict=True,
            )

            # 预测动作
            try:
                pred_actions = model.predict_actions(outputs.hidden_states[-1])  # [B, chunk_size, 7]
            except:
                # 如果是离散模式，跳过
                print("  Skipping: model uses discrete action prediction")
                continue

            # 计算误差
            mse = ((pred_actions - true_actions) ** 2).mean(dim=[1, 2])  # [B]
            mae = (pred_actions - true_actions).abs().mean(dim=[1, 2])  # [B]

            all_mse.extend(mse.cpu().numpy())
            all_mae.extend(mae.cpu().numpy())

    # 4. 输出结果
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    if all_mse:
        print(f"\nMean Squared Error (MSE): {np.mean(all_mse):.6f}")
        print(f"Mean Absolute Error (MAE): {np.mean(all_mae):.6f}")
        print(f"Root MSE (RMSE): {np.sqrt(np.mean(all_mse)):.6f}")
    else:
        print("\nNo results (model may use discrete actions)")

    print("\n" + "=" * 70)


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
