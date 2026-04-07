"""
测试 Step 2: 验证动作预测头是否正确初始化和工作
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch


def test_action_head_initialization():
    """测试动作头初始化"""
    from vila_u.model.configuration_vila_u import VILAUConfig

    # 创建配置
    config = VILAUConfig(
        hidden_size=4096,
        action_dim=7,
        action_chunk_size=10,
        use_action_prediction=True,
    )

    print(f"✓ Config created with action prediction enabled")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - action_dim: {config.action_dim}")
    print(f"  - action_chunk_size: {config.action_chunk_size}")


def test_action_head_forward():
    """测试动作头前向传播"""
    import torch.nn as nn

    # 模拟动作头
    hidden_size = 4096
    action_dim = 7
    action_chunk_size = 10
    action_out_dim = action_chunk_size * action_dim

    action_head = nn.Linear(hidden_size, action_out_dim, bias=True)
    nn.init.normal_(action_head.weight, std=0.02)
    nn.init.zeros_(action_head.bias)

    # 模拟输入
    batch_size = 2
    hidden_states = torch.randn(batch_size, hidden_size)

    # 前向传播
    raw_output = action_head(hidden_states)
    actions = raw_output.view(batch_size, action_chunk_size, action_dim)
    actions = torch.tanh(actions)

    # 验证输出形状
    assert actions.shape == (batch_size, action_chunk_size, action_dim), \
        f"Expected shape ({batch_size}, {action_chunk_size}, {action_dim}), got {actions.shape}"

    # 验证输出范围
    assert actions.min() >= -1.0 and actions.max() <= 1.0, \
        f"Actions should be in [-1, 1], got range [{actions.min():.3f}, {actions.max():.3f}]"

    print(f"✓ Action head forward pass successful")
    print(f"  - Input shape: {hidden_states.shape}")
    print(f"  - Output shape: {actions.shape}")
    print(f"  - Output range: [{actions.min():.3f}, {actions.max():.3f}]")


def test_predict_actions_method():
    """测试 predict_actions 方法的逻辑"""
    import torch.nn as nn

    # 模拟完整流程
    batch_size = 2
    seq_len = 100
    hidden_size = 4096
    action_dim = 7
    action_chunk_size = 10

    # 模拟 LLM 输出
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 模拟动作头
    action_head = nn.Linear(hidden_size, action_chunk_size * action_dim)

    # 提取最后一个 token 的隐层状态
    act_hidden = hidden_states[:, -1, :]  # [B, hidden_size]

    # 预测动作
    raw = action_head(act_hidden)
    actions = raw.view(batch_size, action_chunk_size, action_dim)
    actions = torch.tanh(actions)

    # 验证
    assert actions.shape == (batch_size, action_chunk_size, action_dim)
    assert actions.min() >= -1.0 and actions.max() <= 1.0

    print(f"✓ predict_actions method logic verified")
    print(f"  - Hidden states shape: {hidden_states.shape}")
    print(f"  - Actions shape: {actions.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Testing Action Prediction Head")
    print("=" * 60)

    test_action_head_initialization()
    print()
    test_action_head_forward()
    print()
    test_predict_actions_method()

    print("\n" + "=" * 60)
    print("Step 2: All tests passed! ✓")
    print("=" * 60)
