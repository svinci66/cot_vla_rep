"""
VILA-U 动作预测训练脚本

训练 VILA-U 的动作预测头，使其能够从观察图像和语言指令预测机器人动作。

训练策略：
- 冻结视觉编码器 (vision_tower)
- 训练 LLM + 多模态投影器 + 动作预测头
- 使用 L1 Loss 进行动作回归
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from vila_u.constants import ACTION_DIM, ACTION_CHUNK_SIZE
from vila_u.model.builder import load_pretrained_model
from vila_u.data.libero_dataset import LiberoGoalDataset, collate_fn


class ActionPredictionTrainer:
    """VILA-U 动作预测训练器"""

    def __init__(
        self,
        model_path: str,
        data_root: str,
        output_dir: str,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        use_wandb: bool = False,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.data_root = data_root
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_wandb = use_wandb
        self.device = device

        os.makedirs(output_dir, exist_ok=True)

        # 初始化 wandb
        if use_wandb:
            wandb.init(
                project="vila-u-action-prediction",
                config={
                    "model_path": model_path,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                }
            )

    def load_model(self):
        """加载 VILA-U 模型并启用动作预测"""
        print(f"Loading model from {self.model_path}...")

        # 加载预训练模型
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=self.model_path,
            device_map=self.device,
        )

        # 启用动作预测
        if not hasattr(model, 'action_head'):
            # 如果模型没有动作头，需要重新初始化配置
            from vila_u.model.configuration_vila_u import VILAUConfig
            config = model.config
            config.use_action_prediction = True
            config.action_dim = ACTION_DIM
            config.action_chunk_size = ACTION_CHUNK_SIZE

            # 重新初始化模型以添加动作头
            model.init_vlm(config)

        # 冻结视觉编码器
        vision_tower = model.get_vision_tower()
        for param in vision_tower.parameters():
            param.requires_grad = False

        print(f"✓ Model loaded with action prediction enabled")
        print(f"  - Vision tower: frozen")
        print(f"  - LLM: trainable")
        print(f"  - Action head: trainable")

        return model, tokenizer, image_processor

    def create_dataloader(self):
        """创建数据加载器"""
        print(f"Loading dataset from {self.data_root}...")

        dataset = LiberoGoalDataset(
            data_root=self.data_root,
            image_size=256,
            action_chunk_size=ACTION_CHUNK_SIZE,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataloader

    def setup_optimizer(self, model):
        """设置优化器和学习率调度器"""
        # 只优化可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # 余弦退火学习率
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.1,
        )

        print(f"✓ Optimizer configured")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        return optimizer, scheduler

    def compute_loss(self, model, batch):
        """计算训练损失"""
        observations = batch['observations'].to(self.device)  # [B, 3, 256, 256]
        instructions = batch['instructions']  # List[str]
        action_labels = batch['action_labels'].to(self.device)  # [B, 10, 7]

        # 1. 构建输入文本（添加图像占位符）
        from vila_u.constants import DEFAULT_IMAGE_TOKEN

        # 为每个指令添加图像 token
        prompts = [f"{DEFAULT_IMAGE_TOKEN}\n{inst}" for inst in instructions]

        # 2. Tokenize 文本
        input_ids = model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).input_ids.to(self.device)

        # 3. 前向传播
        outputs = model(
            input_ids=input_ids,
            images=observations,
            output_hidden_states=True,
            return_dict=True,
        )

        # 4. 获取隐层状态
        hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]

        # 5. 预测动作
        action_pred = model.predict_actions(hidden_states)  # [B, 10, 7]

        # 6. 计算 L1 损失
        loss = nn.functional.l1_loss(action_pred, action_labels)

        return loss, action_pred

    def train_epoch(self, model, dataloader, optimizer, epoch):
        """训练一个 epoch"""
        model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            loss, action_pred = self.compute_loss(model, batch)

            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
            })

            # 记录到 wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item() * self.gradient_accumulation_steps,
                    'train/epoch': epoch,
                    'train/step': epoch * len(dataloader) + batch_idx,
                })

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, model, epoch, loss):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}.pt")

        # 只保存动作头和可训练的 LLM 参数
        state_dict = {
            'epoch': epoch,
            'loss': loss,
            'action_head': model.action_head.state_dict() if hasattr(model, 'action_head') else None,
            # 'llm': model.llm.state_dict(),  # 可选：保存完整 LLM
        }

        torch.save(state_dict, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # 保存最佳模型
        best_path = os.path.join(self.output_dir, "best_model.pt")
        if not os.path.exists(best_path) or loss < self.best_loss:
            torch.save(state_dict, best_path)
            self.best_loss = loss
            print(f"✓ Best model updated: {best_path}")

    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("VILA-U Action Prediction Training")
        print("=" * 60)

        # 加载模型
        model, tokenizer, image_processor = self.load_model()

        # 创建数据加载器
        dataloader = self.create_dataloader()

        # 设置优化器
        optimizer, scheduler = self.setup_optimizer(model)

        # 训练循环
        self.best_loss = float('inf')

        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")

            # 训练一个 epoch
            avg_loss = self.train_epoch(model, dataloader, optimizer, epoch)

            # 更新学习率
            scheduler.step()

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  - Average Loss: {avg_loss:.4f}")
            print(f"  - Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # 保存检查点
            self.save_checkpoint(model, epoch, avg_loss)

            # 记录到 wandb
            if self.use_wandb:
                wandb.log({
                    'epoch/loss': avg_loss,
                    'epoch/lr': scheduler.get_last_lr()[0],
                    'epoch/num': epoch,
                })

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train VILA-U action prediction")

    # 模型和数据
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to pretrained VILA-U model")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to LIBERO Goal dataset directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")

    # 其他
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # 创建训练器
    trainer = ActionPredictionTrainer(
        model_path=args.model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb,
        device=args.device,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
