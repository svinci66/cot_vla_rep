"""
测试 VILA-U 模型加载和前向传播

验证：
1. 模型加载
2. 动作预测头初始化
3. 前向传播
4. 动作预测
"""

import sys
import os
sys.path.insert(0, '/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep')

import torch
import numpy as np
from PIL import Image

# 模型路径
MODEL_PATH = "/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"

print("=" * 70)
print("Testing VILA-U Model Loading and Forward Pass")
print("=" * 70)

# 1. 测试模型加载
print("\n[1/5] Loading VILA-U model...")
try:
    from vila_u.model.builder import load_pretrained_model
    from vila_u.constants import ACTION_DIM, ACTION_CHUNK_SIZE

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        device_map="cuda",
    )

    print(f"✓ Model loaded successfully!")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Context length: {context_len}")
    print(f"  - Has vision_tower: {hasattr(model, 'get_vision_tower')}")
    print(f"  - Has tokenizer: {tokenizer is not None}")
    print(f"  - Has image_processor: {image_processor is not None}")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 测试动作预测头初始化
print("\n[2/5] Initializing action prediction head...")
try:
    # 启用动作预测
    model.config.use_action_prediction = True
    model.config.action_dim = ACTION_DIM
    model.config.action_chunk_size = ACTION_CHUNK_SIZE

    # 初始化动作头
    model.init_vlm(model.config)

    print(f"✓ Action head initialized!")
    print(f"  - Has action_head: {hasattr(model, 'action_head')}")
    if hasattr(model, 'action_head'):
        print(f"  - Action head type: {type(model.action_head).__name__}")
        print(f"  - Action head parameters: {sum(p.numel() for p in model.action_head.parameters()):,}")

except Exception as e:
    print(f"✗ Error initializing action head: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. 测试图像预处理
print("\n[3/5] Testing image preprocessing...")
try:
    # 创建测试图像
    test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))

    # 预处理
    image_tensor = image_processor.preprocess(test_image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to('cuda', dtype=model.dtype)

    print(f"✓ Image preprocessing successful!")
    print(f"  - Input image size: {test_image.size}")
    print(f"  - Processed tensor shape: {image_tensor.shape}")
    print(f"  - Tensor dtype: {image_tensor.dtype}")
    print(f"  - Tensor device: {image_tensor.device}")

except Exception as e:
    print(f"✗ Error in image preprocessing: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. 测试前向传播
print("\n[4/5] Testing forward pass...")
try:
    from vila_u.constants import DEFAULT_IMAGE_TOKEN

    # 构建输入
    instruction = "pick up the bowl"
    prompt = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"

    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to('cuda')

    print(f"  - Prompt: {prompt}")
    print(f"  - Input IDs shape: {input_ids.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            output_hidden_states=True,
            return_dict=True,
        )

    print(f"✓ Forward pass successful!")
    print(f"  - Output type: {type(outputs)}")
    print(f"  - Has hidden_states: {hasattr(outputs, 'hidden_states')}")
    if hasattr(outputs, 'hidden_states'):
        print(f"  - Number of layers: {len(outputs.hidden_states)}")
        print(f"  - Last hidden state shape: {outputs.hidden_states[-1].shape}")

except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 测试动作预测
print("\n[5/5] Testing action prediction...")
try:
    # 使用 predict_action 方法
    test_image_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    actions = model.predict_action(
        image=test_image_np,
        instruction=instruction,
        image_processor=image_processor,
    )

    print(f"✓ Action prediction successful!")
    print(f"  - Actions shape: {actions.shape}")
    print(f"  - Expected shape: ({ACTION_CHUNK_SIZE}, {ACTION_DIM})")
    print(f"  - Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  - Actions dtype: {actions.dtype}")

    # 验证形状
    assert actions.shape == (ACTION_CHUNK_SIZE, ACTION_DIM), \
        f"Expected shape ({ACTION_CHUNK_SIZE}, {ACTION_DIM}), got {actions.shape}"

    # 验证范围
    assert actions.min() >= -1.0 and actions.max() <= 1.0, \
        f"Actions should be in [-1, 1], got [{actions.min():.3f}, {actions.max():.3f}]"

    print(f"\n✓ All validations passed!")

except Exception as e:
    print(f"✗ Error in action prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
print("\nThe model is ready for:")
print("  1. Training with train_action_prediction.py")
print("  2. Inference with predict_action()")
print("  3. Trajectory generation in LIBERO")
