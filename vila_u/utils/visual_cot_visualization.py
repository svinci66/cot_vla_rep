"""
Phase 4: Visual CoT-VLA Visualization Utilities
用于可视化生成的子目标图像
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional
import matplotlib.pyplot as plt


def save_subgoal_comparison(
    observation: torch.Tensor,
    generated_subgoal: torch.Tensor,
    gt_subgoal: torch.Tensor,
    save_path: str,
    instruction: Optional[str] = None,
):
    """
    保存观测图像、生成的子目标和 GT 子目标的对比图

    Args:
        observation: [3, H, W] 观测图像，值域 [0, 255]
        generated_subgoal: [3, H, W] 生成的子目标图像，值域 [0, 255]
        gt_subgoal: [3, H, W] GT 子目标图像，值域 [0, 255]
        save_path: 保存路径
        instruction: 可选的文本指令
    """
    # 转换为 numpy 格式 [H, W, 3]
    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            if img.shape[0] == 3:  # [3, H, W]
                img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    obs_np = to_numpy(observation)
    gen_np = to_numpy(generated_subgoal)
    gt_np = to_numpy(gt_subgoal)

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(obs_np)
    axes[0].set_title("Observation", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(gen_np)
    axes[1].set_title("Generated Subgoal", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(gt_np)
    axes[2].set_title("Ground Truth Subgoal", fontsize=14)
    axes[2].axis("off")

    if instruction:
        fig.suptitle(f"Instruction: {instruction}", fontsize=12, y=0.98)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_subgoal_grid(
    observations: torch.Tensor,
    generated_subgoals: torch.Tensor,
    gt_subgoals: torch.Tensor,
    save_path: str,
    num_samples: int = 8,
):
    """
    保存多个样本的子目标对比网格图

    Args:
        observations: [B, 3, H, W] 观测图像
        generated_subgoals: [B, 3, H, W] 生成的子目标图像
        gt_subgoals: [B, 3, H, W] GT 子目标图像
        save_path: 保存路径
        num_samples: 显示的样本数量
    """
    num_samples = min(num_samples, observations.shape[0])

    # 转换为 numpy
    def to_numpy(imgs):
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
            if imgs.shape[1] == 3:  # [B, 3, H, W]
                imgs = np.transpose(imgs, (0, 2, 3, 1))  # [B, H, W, 3]
        imgs = np.clip(imgs, 0, 255).astype(np.uint8)
        return imgs

    obs_np = to_numpy(observations[:num_samples])
    gen_np = to_numpy(generated_subgoals[:num_samples])
    gt_np = to_numpy(gt_subgoals[:num_samples])

    # 创建网格图
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        axes[i, 0].imshow(obs_np[i])
        axes[i, 0].set_title(f"Sample {i+1}: Observation", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gen_np[i])
        axes[i, 1].set_title(f"Sample {i+1}: Generated", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gt_np[i])
        axes[i, 2].set_title(f"Sample {i+1}: Ground Truth", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    反归一化图像（从 ImageNet 归一化恢复到 [0, 255]）

    Args:
        image: [3, H, W] 归一化后的图像

    Returns:
        denorm_image: [3, H, W] 反归一化后的图像，值域 [0, 255]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)

    denorm_image = image * std + mean
    denorm_image = denorm_image * 255.0
    denorm_image = torch.clamp(denorm_image, 0, 255)

    return denorm_image


class VisualCoTVisualizer:
    """
    Visual CoT 可视化工具类

    用于训练和评估过程中的可视化
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

    def visualize_training_batch(
        self,
        model,
        batch,
        step: int,
        num_samples: int = 4,
    ):
        """
        可视化训练批次中的子目标生成结果

        Args:
            model: 模型
            batch: 训练批次数据
            step: 当前训练步数
            num_samples: 可视化的样本数量
        """
        observations = batch["images"][:num_samples]
        subgoal_images = batch["subgoal_images"][:num_samples]
        input_ids = batch["input_ids"][:num_samples]
        attention_mask = batch["attention_mask"][:num_samples]

        # 生成子目标 tokens
        with torch.no_grad():
            # 找到 prompt 部分
            from vila_u.constants import DEFAULT_SUBGOAL_TOKEN
            subgoal_token_id = model.tokenizer.convert_tokens_to_ids(DEFAULT_SUBGOAL_TOKEN)

            prompt_lengths = []
            for i in range(num_samples):
                subgoal_positions = (input_ids[i] == subgoal_token_id).nonzero(as_tuple=True)[0]
                if len(subgoal_positions) > 0:
                    prompt_lengths.append(subgoal_positions[0].item())
                else:
                    prompt_lengths.append(attention_mask[i].sum().item())

            max_prompt_len = max(prompt_lengths)
            prompt_input_ids = input_ids[:, :max_prompt_len]
            prompt_attention_mask = attention_mask[:, :max_prompt_len]

            # 生成子目标
            generated_subgoal_ids = model.generate_subgoal_tokens(
                input_ids=prompt_input_ids,
                images=observations,
                attention_mask=prompt_attention_mask,
                max_new_tokens=1024,
                do_sample=False,
            )

            # 解码为图像
            generated_subgoals = model.decode_subgoal_tokens(generated_subgoal_ids)

        # 反归一化观测图像和 GT 子目标
        observations_denorm = torch.stack([denormalize_image(obs) for obs in observations])
        subgoals_denorm = torch.stack([denormalize_image(sg) for sg in subgoal_images])

        # 保存对比图
        save_path = os.path.join(self.vis_dir, f"step_{step:06d}.png")
        save_subgoal_grid(
            observations_denorm,
            generated_subgoals,
            subgoals_denorm,
            save_path,
            num_samples=num_samples,
        )

        print(f"[Visualization] Saved training visualization to {save_path}")

    def visualize_evaluation_samples(
        self,
        model,
        eval_dataset,
        num_samples: int = 50,
    ):
        """
        评估时批量可视化子目标生成结果

        Args:
            model: 模型
            eval_dataset: 评估数据集
            num_samples: 可视化的样本数量
        """
        eval_vis_dir = os.path.join(self.vis_dir, "evaluation")
        os.makedirs(eval_vis_dir, exist_ok=True)

        model.eval()

        for i in range(min(num_samples, len(eval_dataset))):
            sample = eval_dataset[i]

            observation = sample["observations"].unsqueeze(0).to(model.device)
            instruction = sample["instructions"]
            gt_subgoal = sample["subgoal_images"]

            # 生成子目标和动作
            with torch.no_grad():
                try:
                    actions, generated_subgoal = model.generate_with_visual_cot(
                        observation=observation.squeeze(0),
                        instruction=instruction,
                        do_sample=False,
                    )
                except Exception as e:
                    print(f"[Warning] Failed to generate for sample {i}: {e}")
                    continue

            # 反归一化
            obs_denorm = denormalize_image(observation.squeeze(0))
            gt_subgoal_denorm = denormalize_image(gt_subgoal)

            # 保存单个样本的对比图
            sample_dir = os.path.join(eval_vis_dir, f"sample_{i:04d}")
            os.makedirs(sample_dir, exist_ok=True)

            save_path = os.path.join(sample_dir, "comparison.png")
            save_subgoal_comparison(
                obs_denorm,
                generated_subgoal,
                gt_subgoal_denorm,
                save_path,
                instruction=instruction,
            )

            # 保存元数据
            import json
            metadata = {
                "sample_id": i,
                "instruction": instruction,
                "predicted_actions": actions.cpu().numpy().tolist(),
            }
            with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"[Visualization] Saved {num_samples} evaluation samples to {eval_vis_dir}")
