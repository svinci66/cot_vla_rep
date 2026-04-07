"""
测试 Step 1: 验证常量和配置是否正确添加
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def test_constants():
    """测试常量定义"""
    from vila_u.constants import (
        ACTION_DIM,
        ACTION_CHUNK_SIZE,
        ACTION_HORIZON,
        ACTION_MIN,
        ACTION_MAX,
    )

    assert ACTION_DIM == 7, "ACTION_DIM should be 7"
    assert ACTION_CHUNK_SIZE == 10, "ACTION_CHUNK_SIZE should be 10"
    assert ACTION_HORIZON == 10, "ACTION_HORIZON should be 10"
    assert ACTION_MIN == -1.0, "ACTION_MIN should be -1.0"
    assert ACTION_MAX == 1.0, "ACTION_MAX should be 1.0"

    print("✓ All constants defined correctly")


def test_config():
    """测试配置字段"""
    from vila_u.model.configuration_vila_u import VILAUConfig

    config = VILAUConfig(
        action_dim=7,
        action_chunk_size=10,
        use_action_prediction=True,
    )

    assert config.action_dim == 7, "action_dim should be 7"
    assert config.action_chunk_size == 10, "action_chunk_size should be 10"
    assert config.use_action_prediction == True, "use_action_prediction should be True"

    print("✓ Configuration fields added correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Testing Constants and Configuration")
    print("=" * 60)

    test_constants()
    test_config()

    print("\n" + "=" * 60)
    print("Step 1: All tests passed! ✓")
    print("=" * 60)
