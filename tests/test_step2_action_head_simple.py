"""
测试 Step 2: 验证动作预测头代码（简化版）
不导入完整的 vila_u 模块，直接读取文件内容验证
"""
import os
import torch
import torch.nn as nn


def test_action_head_in_file():
    """测试 vila_u_arch.py 中是否添加了动作头代码"""
    arch_file = os.path.join(os.path.dirname(__file__), '..', 'vila_u', 'model', 'vila_u_arch.py')

    with open(arch_file, 'r') as f:
        content = f.read()

    # 检查动作头初始化代码
    required_code = [
        'use_action_prediction',
        'self.action_head',
        'nn.Linear',
        'action_out_dim',
    ]

    for code in required_code:
        if code not in content:
            raise AssertionError(f"Missing code in vila_u_arch.py: {code}")

    # 检查 predict_actions 方法
    if 'def predict_actions' not in content:
        raise AssertionError("Missing predict_actions method in vila_u_arch.py")

    print("✓ Action head code added correctly in vila_u_arch.py")


def test_action_head_logic():
    """测试动作头的逻辑（独立测试）"""
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
    seq_len = 100
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 提取最后一个 token
    act_hidden = hidden_states[:, -1, :]

    # 预测动作
    raw = action_head(act_hidden)
    actions = raw.view(batch_size, action_chunk_size, action_dim)
    actions = torch.tanh(actions)

    # 验证
    assert actions.shape == (batch_size, action_chunk_size, action_dim), \
        f"Expected shape ({batch_size}, {action_chunk_size}, {action_dim}), got {actions.shape}"
    assert actions.min() >= -1.0 and actions.max() <= 1.0, \
        f"Actions should be in [-1, 1], got range [{actions.min():.3f}, {actions.max():.3f}]"

    print(f"✓ Action head logic verified")
    print(f"  - Input shape: {hidden_states.shape}")
    print(f"  - Output shape: {actions.shape}")
    print(f"  - Output range: [{actions.min():.3f}, {actions.max():.3f}]")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Testing Action Prediction Head (Simplified)")
    print("=" * 60)

    test_action_head_in_file()
    print()
    test_action_head_logic()

    print("\n" + "=" * 60)
    print("Step 2: All tests passed! ✓")
    print("=" * 60)
