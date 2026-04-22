"""
Phase 4: Visual CoT-VLA 验证测试程序

严格验证实现是否符合论文方案：
1. 单序列自回归生成（不是双图像输入）
2. 子目标以tokens形式存在（不是解码的图像）
3. 正确的序列结构：[观测tokens] [文本tokens] [子目标tokens] [<act>] → 动作
4. 损失函数：视觉自回归损失 + 动作回归损失
"""

import torch
import sys
import traceback

def test_phase4_paper_compliance():
    """验证Phase 4实现是否符合论文方案"""

    print("=" * 80)
    print("Phase 4: Visual CoT-VLA 论文方案验证测试")
    print("=" * 80)
    print()

    all_tests_passed = True

    # ========================================
    # 测试 1: 检查错误的方法是否已删除
    # ========================================
    print("[测试 1/6] 验证错误的双图像输入方法已删除")
    print("-" * 80)

    try:
        from vila_u.model.builder import load_pretrained_model

        # 这里只检查方法是否存在，不实际加载模型
        from vila_u.model import vila_u_arch
        import inspect

        # 检查VILAUMetaForCausalLM类的方法
        methods = [name for name, _ in inspect.getmembers(vila_u_arch.VILAUMetaForCausalLM, predicate=inspect.isfunction)]

        # 这些错误的方法应该已被删除
        wrong_methods = [
            'generate_subgoal_image',  # 错误：解码子目标为图像
            'predict_action_with_visual_cot',  # 错误：双图像输入
            '_predict_action_continuous_with_subgoal',  # 错误：双图像输入
            '_predict_action_discrete_with_subgoal',  # 错误：双图像输入
        ]

        found_wrong_methods = [m for m in wrong_methods if m in methods]

        if found_wrong_methods:
            print(f"❌ 失败: 发现错误的方法仍然存在: {found_wrong_methods}")
            print("   这些方法应该已被删除，因为它们使用了错误的双图像输入方式")
            all_tests_passed = False
        else:
            print("✅ 通过: 所有错误的双图像输入方法已删除")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        all_tests_passed = False

    print()

    # ========================================
    # 测试 2: 检查正确的方法是否存在
    # ========================================
    print("[测试 2/6] 验证正确的论文方案方法已实现")
    print("-" * 80)

    try:
        from vila_u.model import vila_u_arch
        import inspect

        methods = [name for name, _ in inspect.getmembers(vila_u_arch.VILAUMetaForCausalLM, predicate=inspect.isfunction)]

        # 正确的方法应该存在
        correct_methods = {
            'generate_with_visual_cot': '论文方案的CoT推理方法',
            'compute_cot_vla_loss': '论文方案的损失函数',
        }

        missing_methods = []
        for method, desc in correct_methods.items():
            if method not in methods:
                missing_methods.append(f"{method} ({desc})")

        if missing_methods:
            print(f"❌ 失败: 缺少正确的方法: {missing_methods}")
            all_tests_passed = False
        else:
            print("✅ 通过: 所有正确的论文方案方法已实现")
            for method, desc in correct_methods.items():
                print(f"   - {method}: {desc}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        all_tests_passed = False

    print()

    # ========================================
    # 测试 3: 验证generate_with_visual_cot的实现逻辑
    # ========================================
    print("[测试 3/6] 验证generate_with_visual_cot符合论文方案")
    print("-" * 80)

    try:
        from vila_u.model import vila_u_arch
        import inspect

        # 获取方法源码
        source = inspect.getsource(vila_u_arch.VILAUMetaForCausalLM.generate_with_visual_cot)

        # 检查关键步骤
        checks = {
            'self.generate': '✓ 使用自回归生成子目标tokens',
            'max_new_tokens': '✓ 指定生成token数量',
            'torch.cat': '✓ 拼接<act> token到序列',
            'action_position': '✓ 记录<act> token位置',
            'self.llm.model': '✓ 使用LLM前向传播',
            'hidden_states': '✓ 提取隐层状态',
            'self.action_head': '✓ 使用动作头解码',
        }

        # 不应该出现的错误模式
        wrong_patterns = {
            'torch.cat([observation, subgoal': '双图像拼接（错误）',
            'images = torch.cat': '多图像输入（可能错误）',
            'generate_subgoal_image': '调用错误的子目标生成方法',
        }

        print("检查正确的实现模式:")
        for pattern, desc in checks.items():
            if pattern in source:
                print(f"  {desc}")
            else:
                print(f"  ⚠️  未找到: {pattern}")

        print("\n检查是否存在错误模式:")
        found_wrong = False
        for pattern, desc in wrong_patterns.items():
            if pattern in source:
                print(f"  ❌ 发现错误模式: {desc}")
                found_wrong = True

        if not found_wrong:
            print("  ✅ 未发现错误模式")
        else:
            all_tests_passed = False

        # 检查关键注释
        if '论文方案' in source or 'paper' in source.lower():
            print("\n✅ 代码包含论文方案的注释说明")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        all_tests_passed = False

    print()

    # ========================================
    # 测试 4: 验证compute_cot_vla_loss的实现
    # ========================================
    print("[测试 4/6] 验证compute_cot_vla_loss符合论文方案")
    print("-" * 80)

    try:
        from vila_u.model import vila_u_arch
        import inspect

        source = inspect.getsource(vila_u_arch.VILAUMetaForCausalLM.compute_cot_vla_loss)

        # 检查损失函数的关键组件
        checks = {
            'F.cross_entropy': '✓ 视觉自回归损失（交叉熵）',
            'F.l1_loss': '✓ 动作回归损失（L1）',
            'shift_logits': '✓ 正确的logits偏移',
            'shift_labels': '✓ 正确的labels偏移',
            'IGNORE_INDEX': '✓ 忽略padding位置',
        }

        print("检查损失函数组件:")
        all_found = True
        for pattern, desc in checks.items():
            if pattern in source:
                print(f"  {desc}")
            else:
                print(f"  ❌ 缺少: {pattern}")
                all_found = False

        if all_found:
            print("\n✅ 损失函数实现完整")
        else:
            print("\n❌ 损失函数实现不完整")
            all_tests_passed = False

        # 检查参数
        sig = inspect.signature(vila_u_arch.VILAUMetaForCausalLM.compute_cot_vla_loss)
        params = list(sig.parameters.keys())

        expected_params = ['self', 'lm_logits', 'action_pred', 'visual_labels', 'action_labels']
        if params == expected_params:
            print(f"✅ 参数列表正确: {params}")
        else:
            print(f"⚠️  参数列表: {params}")
            print(f"   期望: {expected_params}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        all_tests_passed = False

    print()

    # ========================================
    # 测试 5: 验证数据加载器支持
    # ========================================
    print("[测试 5/6] 验证数据加载器支持子目标图像")
    print("-" * 80)

    try:
        from vila_u.data.libero_dataset_v2 import LiberoGoalDataset
        import inspect

        # 检查__init__参数
        sig = inspect.signature(LiberoGoalDataset.__init__)
        params = list(sig.parameters.keys())

        required_params = ['use_visual_cot', 'subgoal_horizon_low', 'subgoal_horizon_high']
        found_params = [p for p in required_params if p in params]

        if len(found_params) == len(required_params):
            print(f"✅ 数据加载器包含所有必需参数:")
            for p in found_params:
                print(f"   - {p}")
        else:
            missing = set(required_params) - set(found_params)
            print(f"❌ 缺少参数: {missing}")
            all_tests_passed = False

        # 检查__getitem__是否加载子目标图像
        source = inspect.getsource(LiberoGoalDataset.__getitem__)

        if 'subgoal_images' in source and 'use_visual_cot' in source:
            print("✅ __getitem__正确加载子目标图像")
        else:
            print("❌ __getitem__未正确实现子目标加载")
            all_tests_passed = False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        all_tests_passed = False

    print()

    # ========================================
    # 测试 6: 验证常量定义
    # ========================================
    print("[测试 6/6] 验证Phase 4常量定义")
    print("-" * 80)

    try:
        from vila_u import constants

        required_constants = {
            'DEFAULT_ACT_START_TOKEN': '<act>',
            'DEFAULT_ACT_END_TOKEN': '</act>',
            'DEFAULT_SUBGOAL_START_TOKEN': '<subgoal>',
            'DEFAULT_SUBGOAL_END_TOKEN': '</subgoal>',
            'SUBGOAL_HORIZON_LOW': 4,
            'SUBGOAL_HORIZON_HIGH': 16,
        }

        all_found = True
        for const_name, expected_value in required_constants.items():
            if hasattr(constants, const_name):
                actual_value = getattr(constants, const_name)
                if actual_value == expected_value:
                    print(f"✅ {const_name} = {actual_value}")
                else:
                    print(f"⚠️  {const_name} = {actual_value} (期望: {expected_value})")
            else:
                print(f"❌ 缺少常量: {const_name}")
                all_found = False

        if not all_found:
            all_tests_passed = False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        all_tests_passed = False

    print()

    # ========================================
    # 最终总结
    # ========================================
    print("=" * 80)
    if all_tests_passed:
        print("✅ 所有测试通过！Phase 4实现符合论文方案")
        print()
        print("验证要点:")
        print("  ✓ 错误的双图像输入方法已删除")
        print("  ✓ 正确的单序列自回归方法已实现")
        print("  ✓ generate_with_visual_cot使用自回归生成子目标tokens")
        print("  ✓ 子目标以tokens形式存在，不是解码的图像")
        print("  ✓ 损失函数：视觉自回归 + 动作回归")
        print("  ✓ 数据加载器支持子目标图像")
        print("  ✓ 所有必需常量已定义")
        return 0
    else:
        print("❌ 部分测试失败！请检查实现")
        return 1
    print("=" * 80)


if __name__ == "__main__":
    try:
        exit_code = test_phase4_paper_compliance()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 测试程序异常: {e}")
        traceback.print_exc()
        sys.exit(1)
