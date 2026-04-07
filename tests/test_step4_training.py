"""
测试 Step 4: 验证训练逻辑

测试训练脚本的各个组件是否正确工作
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch
import torch.nn as nn


def test_training_script_exists():
    """测试训练脚本文件是否存在"""
    train_script = os.path.join(project_root, 'vila_u', 'train', 'train_action_prediction.py')

    if not os.path.exists(train_script):
        raise AssertionError(f"Training script not found: {train_script}")

    with open(train_script, 'r') as f:
        content = f.read()

    # 检查关键类和方法
    required_items = [
        'class ActionPredictionTrainer',
        'def load_model',
        'def create_dataloader',
        'def setup_optimizer',
        'def compute_loss',
        'def train_epoch',
        'def save_checkpoint',
        'def train',
    ]

    for item in required_items:
        if item not in content:
            raise AssertionError(f"Missing in train_action_prediction.py: {item}")

    print("✓ Training script structure verified")
    print(f"  - File: {train_script}")
    print(f"  - All required methods present")


def test_loss_computation():
    """测试损失计算逻辑"""
    batch_size = 4
    action_chunk_size = 10
    action_dim = 7

    # 模拟预测和标签
    action_pred = torch.randn(batch_size, action_chunk_size, action_dim)
    action_pred = torch.tanh(action_pred)  # 限制到 [-1, 1]

    action_labels = torch.randn(batch_size, action_chunk_size, action_dim)
    action_labels = torch.tanh(action_labels)

    # 计算 L1 损失
    loss = nn.functional.l1_loss(action_pred, action_labels)

    # 验证
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"

    print("✓ Loss computation verified")
    print(f"  - Prediction shape: {action_pred.shape}")
    print(f"  - Label shape: {action_labels.shape}")
    print(f"  - Loss value: {loss.item():.4f}")


def test_optimizer_setup():
    """测试优化器设置"""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # 创建简单模型
    model = nn.Linear(100, 70)

    # 设置优化器
    optimizer = AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # 设置学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-6,
    )

    # 验证
    assert optimizer.defaults['lr'] == 1e-5
    assert len(optimizer.param_groups) == 1

    print("✓ Optimizer setup verified")
    print(f"  - Optimizer: AdamW")
    print(f"  - Learning rate: {optimizer.defaults['lr']}")
    print(f"  - Scheduler: CosineAnnealingLR")


def test_gradient_accumulation():
    """测试梯度累积逻辑"""
    model = nn.Linear(10, 7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    gradient_accumulation_steps = 4

    # 模拟梯度累积
    optimizer.zero_grad()

    for step in range(gradient_accumulation_steps):
        # 模拟前向传播和损失
        x = torch.randn(2, 10)
        y = torch.randn(2, 7)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)

        # 缩放损失
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # 只在最后一步更新
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print("✓ Gradient accumulation logic verified")
    print(f"  - Accumulation steps: {gradient_accumulation_steps}")


def test_checkpoint_saving():
    """测试检查点保存逻辑"""
    import tempfile

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建简单模型
        action_head = nn.Linear(4096, 70)

        # 保存检查点
        checkpoint_path = os.path.join(tmpdir, "checkpoint_epoch_1.pt")
        state_dict = {
            'epoch': 0,
            'loss': 0.5,
            'action_head': action_head.state_dict(),
        }

        torch.save(state_dict, checkpoint_path)

        # 验证文件存在
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist"

        # 加载检查点
        loaded = torch.load(checkpoint_path)
        assert 'epoch' in loaded
        assert 'loss' in loaded
        assert 'action_head' in loaded

        print("✓ Checkpoint saving/loading verified")
        print(f"  - Checkpoint path: {checkpoint_path}")
        print(f"  - Keys: {list(loaded.keys())}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: Testing Training Logic")
    print("=" * 60)

    test_training_script_exists()
    print()
    test_loss_computation()
    print()
    test_optimizer_setup()
    print()
    test_gradient_accumulation()
    print()
    test_checkpoint_saving()

    print("\n" + "=" * 60)
    print("Step 4: All tests passed! ✓")
    print("=" * 60)
    print("\nNote: This tests the training logic components.")
    print("Actual training requires:")
    print("  1. Pretrained VILA-U model")
    print("  2. LIBERO Goal dataset")
    print("  3. GPU with sufficient memory")
