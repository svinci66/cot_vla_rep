"""
测试 Step 1: 验证常量和配置是否正确添加（简化版）
不导入完整的 vila_u 模块，直接读取文件内容验证
"""
import os
import re

def test_constants():
    """测试常量定义"""
    # 读取 constants.py 文件
    constants_file = os.path.join(os.path.dirname(__file__), '..', 'vila_u', 'constants.py')

    with open(constants_file, 'r') as f:
        content = f.read()

    # 检查是否包含所需的常量
    required_constants = [
        'ACTION_DIM = 7',
        'ACTION_CHUNK_SIZE = 10',
        'ACTION_HORIZON = 10',
        'ACTION_MIN = -1.0',
        'ACTION_MAX = 1.0',
    ]

    for const in required_constants:
        if const not in content:
            raise AssertionError(f"Missing constant: {const}")

    print("✓ All constants defined correctly in constants.py")


def test_config():
    """测试配置字段"""
    # 读取 configuration_vila_u.py 文件
    config_file = os.path.join(os.path.dirname(__file__), '..', 'vila_u', 'model', 'configuration_vila_u.py')

    with open(config_file, 'r') as f:
        content = f.read()

    # 检查是否包含所需的配置字段
    required_configs = [
        'self.action_dim',
        'self.action_chunk_size',
        'self.use_action_prediction',
    ]

    for config in required_configs:
        if config not in content:
            raise AssertionError(f"Missing config field: {config}")

    print("✓ Configuration fields added correctly in configuration_vila_u.py")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Testing Constants and Configuration (Simplified)")
    print("=" * 60)

    test_constants()
    test_config()

    print("\n" + "=" * 60)
    print("Step 1: All tests passed! ✓")
    print("=" * 60)
