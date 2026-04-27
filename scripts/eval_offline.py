#!/usr/bin/env python
"""
离线评估脚本 - 在测试集上评估动作预测准确率
不需要仿真环境，直接计算预测动作与真实动作的误差
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from vila_u.data.libero_dataset import LiberoGoalDataset
from vila_u.model.builder import load_pretrained_model
from vila_u.constants import ACTION_DIM


def evaluate_offline(
    model_path: str,
    checkpoint_path: str,
    data_path: str,
    batch_size: int = 8,
    num_samples: int = 100,
    device: str = "cuda",
):
    """
    在测试集上评估动作预测准确率

    Args:
        model_path: VILA-U 模型路径
        checkpoint_path: 动作预测头检查点路径
        data_path: LIBERO 数据集路径
        batch_size: 批次大小
        num_samples: 评估样本数量（None表示全部）
        device: 设备
    """
    print("=" * 70)
    print("VILA-U Offline Evaluation (Test Set)")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/3] Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        device=device,
    )

    # 加载动作预测头
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print(f"  ✓ Model loaded from {model_path}")
    print(f"  ✓ Checkpoint loaded from {checkpoint_path}")

    # 2. 加载测试集
    print("\n[2/3] Loading test dataset...")
    dataset = LiberoGoalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="test",  # 使用测试集
    )

    if num_samples is not None:
        num_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,  # 使用自定义collate
    )

    print(f"  ✓ Test dataset loaded: {len(dataset)} samples")

    # 3. 评估
    print(f"\n[3/3] Evaluating on {len(dataset)} samples...")

    all_errors = []
    all_errors_per_dim = [[] for _ in range(ACTION_DIM)]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for sample in batch:
                # 准备输入
                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                images = sample['images'].unsqueeze(0).to(device)
                labels = sample['labels'].to(device)  # [seq_len]

                # 找到动作token的位置
                action_mask = labels != -100
                if not action_mask.any():
                    continue

                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    images=images,
                    return_dict=True,
                )

                # 提取动作预测
                logits = outputs.logits[0]  # [seq_len, vocab_size]
                pred_action_tokens = logits[action_mask].argmax(dim=-1)  # [num_actions]
                true_action_tokens = labels[action_mask]  # [num_actions]

                # 计算误差（token级别）
                error = (pred_action_tokens != true_action_tokens).float().mean().item()
                all_errors.append(error)

                # 按维度统计
                for i in range(min(ACTION_DIM, len(pred_action_tokens))):
                    if i < len(pred_action_tokens):
                        dim_error = (pred_action_tokens[i] != true_action_tokens[i]).float().item()
                        all_errors_per_dim[i].append(dim_error)

    # 4. 计算指标
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    mean_error = np.mean(all_errors)
    accuracy = 1.0 - mean_error

    print(f"\nOverall Metrics:")
    print(f"  Token Accuracy: {accuracy*100:.2f}%")
    print(f"  Token Error Rate: {mean_error*100:.2f}%")

    print(f"\nPer-Dimension Accuracy:")
    for i, errors in enumerate(all_errors_per_dim):
        if errors:
            dim_accuracy = 1.0 - np.mean(errors)
            print(f"  Dimension {i}: {dim_accuracy*100:.2f}%")

    print("\n" + "=" * 70)

    return {
        'accuracy': accuracy,
        'error_rate': mean_error,
        'per_dim_accuracy': [1.0 - np.mean(e) for e in all_errors_per_dim if e],
    }


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation on test set")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to VILA-U model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to action head checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to LIBERO dataset")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

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
