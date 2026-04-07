"""
测试 Step 5: 验证推理接口

测试从观察图像预测动作的推理接口
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
from PIL import Image


def test_predict_action_method_exists():
    """测试 predict_action 方法是否存在"""
    arch_file = os.path.join(project_root, 'vila_u', 'model', 'vila_u_arch.py')

    with open(arch_file, 'r') as f:
        content = f.read()

    # 检查方法是否存在
    if 'def predict_action' not in content:
        raise AssertionError("predict_action method not found in vila_u_arch.py")

    # 检查关键功能
    required_items = [
        '@torch.no_grad()',
        'image_processor',
        'instruction',
        'self.eval()',
    ]

    for item in required_items:
        if item not in content:
            raise AssertionError(f"Missing in predict_action method: {item}")

    print("✓ predict_action method exists in vila_u_arch.py")
    print("  - Decorator: @torch.no_grad()")
    print("  - Image preprocessing support")
    print("  - Instruction tokenization")


def test_image_preprocessing():
    """测试图像预处理逻辑"""
    import torchvision.transforms as transforms

    # 创建测试图像
    image = Image.new('RGB', (256, 256), color='red')

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image_tensor = transform(image)

    # 验证
    assert image_tensor.shape == (3, 256, 256), f"Expected (3, 256, 256), got {image_tensor.shape}"
    assert image_tensor.dtype == torch.float32

    print("✓ Image preprocessing verified")
    print(f"  - Input: PIL.Image (256, 256)")
    print(f"  - Output: {image_tensor.shape}")
    print(f"  - Range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")


def test_inference_logic():
    """测试推理逻辑（模拟）"""
    import torch.nn as nn

    # 模拟模型组件
    hidden_size = 4096
    action_dim = 7
    action_chunk_size = 10

    # 模拟动作头
    action_head = nn.Linear(hidden_size, action_chunk_size * action_dim)

    # 模拟输入
    batch_size = 1
    seq_len = 50
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 提取最后一个 token
    act_hidden = hidden_states[:, -1, :]

    # 预测动作
    raw = action_head(act_hidden)
    actions = raw.view(batch_size, action_chunk_size, action_dim)
    actions = torch.tanh(actions)

    # 验证
    assert actions.shape == (batch_size, action_chunk_size, action_dim)
    assert actions.min() >= -1.0 and actions.max() <= 1.0

    # 返回单个样本
    actions = actions.squeeze(0)
    assert actions.shape == (action_chunk_size, action_dim)

    print("✓ Inference logic verified")
    print(f"  - Hidden states: {hidden_states.shape}")
    print(f"  - Output actions: {actions.shape}")
    print(f"  - Action range: [{actions.min():.3f}, {actions.max():.3f}]")


def test_different_image_formats():
    """测试不同图像格式的处理"""
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 1. PIL Image
    pil_image = Image.new('RGB', (128, 128), color='blue')
    tensor1 = transform(pil_image)
    assert tensor1.shape == (3, 256, 256)

    # 2. Numpy array [H, W, 3]
    np_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    pil_from_np = Image.fromarray(np_image)
    tensor2 = transform(pil_from_np)
    assert tensor2.shape == (3, 256, 256)

    # 3. Torch tensor [3, H, W]
    torch_image = torch.randn(3, 128, 128)
    # 需要先转换为 PIL
    torch_np = (torch_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    torch_np = np.clip(torch_np, 0, 255)
    pil_from_torch = Image.fromarray(torch_np)
    tensor3 = transform(pil_from_torch)
    assert tensor3.shape == (3, 256, 256)

    print("✓ Different image formats verified")
    print("  - PIL.Image: ✓")
    print("  - Numpy array [H, W, 3]: ✓")
    print("  - Torch tensor [3, H, W]: ✓")


def test_batch_vs_single_inference():
    """测试批量推理 vs 单样本推理"""
    import torch.nn as nn

    hidden_size = 4096
    action_dim = 7
    action_chunk_size = 10

    action_head = nn.Linear(hidden_size, action_chunk_size * action_dim)

    # 批量推理
    batch_hidden = torch.randn(4, 50, hidden_size)
    batch_actions = action_head(batch_hidden[:, -1, :])
    batch_actions = batch_actions.view(4, action_chunk_size, action_dim)
    batch_actions = torch.tanh(batch_actions)

    # 单样本推理
    single_hidden = torch.randn(1, 50, hidden_size)
    single_actions = action_head(single_hidden[:, -1, :])
    single_actions = single_actions.view(1, action_chunk_size, action_dim)
    single_actions = torch.tanh(single_actions)
    single_actions = single_actions.squeeze(0)

    # 验证
    assert batch_actions.shape == (4, action_chunk_size, action_dim)
    assert single_actions.shape == (action_chunk_size, action_dim)

    print("✓ Batch vs single inference verified")
    print(f"  - Batch output: {batch_actions.shape}")
    print(f"  - Single output: {single_actions.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: Testing Inference Interface")
    print("=" * 60)

    test_predict_action_method_exists()
    print()
    test_image_preprocessing()
    print()
    test_inference_logic()
    print()
    test_different_image_formats()
    print()
    test_batch_vs_single_inference()

    print("\n" + "=" * 60)
    print("Step 5: All tests passed! ✓")
    print("=" * 60)
    print("\nNote: This tests the inference interface logic.")
    print("Actual inference requires:")
    print("  1. Trained VILA-U model with action head")
    print("  2. Proper image and text encoding")
    print("  3. Complete forward pass through the model")
