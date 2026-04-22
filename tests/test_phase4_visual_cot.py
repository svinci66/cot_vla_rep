"""
Phase 4: Visual CoT-VLA 测试脚本

测试 Visual Chain-of-Thought 推理功能：
1. 子目标图像生成
2. 基于子目标的动作预测
3. 完整的 CoT 推理流程
"""

import torch
import numpy as np
from PIL import Image

def test_phase4_visual_cot():
    """测试 Phase 4 Visual CoT 功能"""
    print("=" * 60)
    print("Phase 4: Visual CoT-VLA Test")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] Loading model...")
    from vila_u.model.builder import load_pretrained_model

    model_path = "mit-han-lab/vila-u-7b-256"  # 替换为实际路径

    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # 2. 启用 Visual CoT
    print("\n[2/4] Enabling Visual CoT...")
    model.config.use_visual_cot = True
    model.config.use_discrete_action_prediction = True
    model.config.use_hybrid_attention = True
    model.config.subgoal_horizon_low = 4
    model.config.subgoal_horizon_high = 16
    print("✓ Visual CoT enabled")
    print(f"  - use_visual_cot: {model.config.use_visual_cot}")
    print(f"  - subgoal_horizon: [{model.config.subgoal_horizon_low}, {model.config.subgoal_horizon_high}]")

    # 3. 测试子目标生成
    print("\n[3/4] Testing subgoal generation...")

    # 创建测试图像
    test_image = torch.rand(3, 256, 256)  # Random image
    test_instruction = "Pick up the red cube and place it in the box"

    try:
        if hasattr(model, 'generate_subgoal_image'):
            subgoal_image = model.generate_subgoal_image(
                observation=test_image,
                instruction=test_instruction,
                cfg=3.0
            )
            print("✓ Subgoal generation successful")
            print(f"  - Subgoal image shape: {subgoal_image.shape}")
        else:
            print("✗ generate_subgoal_image method not found")
            return False
    except Exception as e:
        print(f"✗ Subgoal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 测试完整 CoT 推理
    print("\n[4/4] Testing full CoT inference...")

    try:
        if hasattr(model, 'predict_action_with_visual_cot'):
            actions = model.predict_action_with_visual_cot(
                observation=test_image,
                instruction=test_instruction,
                subgoal_image=None,  # 自动生成
                cfg=3.0
            )
            print("✓ CoT inference successful")
            print(f"  - Action shape: {actions.shape}")
            print(f"  - Expected shape: [{model.config.action_chunk_size}, {model.config.action_dim}]")

            # 验证输出形状
            expected_shape = (model.config.action_chunk_size, model.config.action_dim)
            if actions.shape == expected_shape:
                print("✓ Action shape is correct")
            else:
                print(f"✗ Action shape mismatch: {actions.shape} vs {expected_shape}")
                return False

            # 验证动作值范围
            if torch.all(actions >= -1.0) and torch.all(actions <= 1.0):
                print("✓ Action values in valid range [-1, 1]")
            else:
                print(f"✗ Action values out of range: min={actions.min()}, max={actions.max()}")
                return False

        else:
            print("✗ predict_action_with_visual_cot method not found")
            return False
    except Exception as e:
        print(f"✗ CoT inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 测试损失计算
    print("\n[5/5] Testing CoT loss computation...")

    try:
        if hasattr(model, 'compute_cot_vla_loss'):
            batch_size = 2
            observation_images = torch.rand(batch_size, 3, 256, 256).cuda()
            subgoal_images = torch.rand(batch_size, 3, 256, 256).cuda()
            instructions = [test_instruction] * batch_size
            action_labels = torch.rand(batch_size, model.config.action_chunk_size, model.config.action_dim).cuda()

            # 如果是离散模式，需要action_token_ids
            action_token_ids = None
            if model.config.use_discrete_action_prediction:
                num_action_tokens = model.config.action_chunk_size * model.config.action_dim
                action_token_ids = torch.randint(0, 256, (batch_size, num_action_tokens)).cuda()

            loss_dict = model.compute_cot_vla_loss(
                observation_images=observation_images,
                subgoal_images=subgoal_images,
                instructions=instructions,
                action_labels=action_labels,
                action_token_ids=action_token_ids,
            )

            print("✓ Loss computation successful")
            print(f"  - Total loss: {loss_dict['total_loss'].item():.4f}")
            print(f"  - Visual loss: {loss_dict['visual_loss'].item():.4f}")
            print(f"  - Action loss: {loss_dict['action_loss'].item():.4f}")
        else:
            print("✗ compute_cot_vla_loss method not found")
            return False
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All Phase 4 tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = test_phase4_visual_cot()
    sys.exit(0 if success else 1)
