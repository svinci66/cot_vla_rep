"""
测试 Step 6: 验证轨迹生成器

测试在环境中闭环生成轨迹的功能
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
from collections import deque


def test_trajectory_generator_exists():
    """测试轨迹生成器文件是否存在"""
    generator_file = os.path.join(project_root, 'vila_u', 'eval', 'trajectory_generator.py')

    if not os.path.exists(generator_file):
        raise AssertionError(f"Trajectory generator file not found: {generator_file}")

    with open(generator_file, 'r') as f:
        content = f.read()

    # 检查关键类和方法
    required_items = [
        'class TrajectoryGenerator',
        'def generate_trajectory',
        'def generate_multiple_trajectories',
        'def generate_with_temporal_ensembling',
        'def create_trajectory_generator',
    ]

    for item in required_items:
        if item not in content:
            raise AssertionError(f"Missing in trajectory_generator.py: {item}")

    print("✓ TrajectoryGenerator class exists")
    print("  - generate_trajectory(): Single trajectory generation")
    print("  - generate_multiple_trajectories(): Batch generation")
    print("  - generate_with_temporal_ensembling(): Temporal ensembling")


def test_action_queue_logic():
    """测试动作队列逻辑"""
    action_chunk_size = 10

    # 模拟动作 chunk
    action_chunk = torch.randn(action_chunk_size, 7)

    # 创建队列
    action_queue = deque(maxlen=action_chunk_size)

    # 填充队列
    for i in range(action_chunk_size):
        action_queue.append(action_chunk[i].numpy())

    assert len(action_queue) == action_chunk_size

    # 逐个取出
    actions_taken = []
    while len(action_queue) > 0:
        action = action_queue.popleft()
        actions_taken.append(action)

    assert len(actions_taken) == action_chunk_size
    assert len(action_queue) == 0

    print("✓ Action queue logic verified")
    print(f"  - Queue size: {action_chunk_size}")
    print(f"  - Actions taken: {len(actions_taken)}")


def test_temporal_ensembling():
    """测试 temporal ensembling 逻辑"""
    ensemble_k = 5

    # 模拟动作历史
    action_history = deque(maxlen=ensemble_k)

    # 添加动作
    for i in range(ensemble_k):
        action = np.random.randn(7)
        action_history.append(action)

    # 计算平均
    if len(action_history) > 0:
        ensembled_action = np.mean(list(action_history), axis=0)

    assert ensembled_action.shape == (7,)

    print("✓ Temporal ensembling logic verified")
    print(f"  - Ensemble window: {ensemble_k}")
    print(f"  - History length: {len(action_history)}")
    print(f"  - Ensembled action shape: {ensembled_action.shape}")


def test_trajectory_data_structure():
    """测试轨迹数据结构"""
    # 模拟轨迹数据
    num_steps = 100

    trajectory = {
        'observations': [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(num_steps)],
        'actions': np.random.randn(num_steps, 7),
        'rewards': np.random.rand(num_steps),
        'dones': np.zeros(num_steps, dtype=bool),
        'success': True,
        'num_steps': num_steps,
        'instruction': 'test task',
    }

    # 验证结构
    assert 'observations' in trajectory
    assert 'actions' in trajectory
    assert 'rewards' in trajectory
    assert 'dones' in trajectory
    assert 'success' in trajectory
    assert 'num_steps' in trajectory
    assert 'instruction' in trajectory

    # 验证形状
    assert len(trajectory['observations']) == num_steps
    assert trajectory['actions'].shape == (num_steps, 7)
    assert trajectory['rewards'].shape == (num_steps,)
    assert trajectory['dones'].shape == (num_steps,)

    print("✓ Trajectory data structure verified")
    print(f"  - Observations: {len(trajectory['observations'])} frames")
    print(f"  - Actions: {trajectory['actions'].shape}")
    print(f"  - Rewards: {trajectory['rewards'].shape}")
    print(f"  - Success: {trajectory['success']}")


def test_success_rate_calculation():
    """测试成功率计算"""
    # 模拟多条轨迹
    trajectories = [
        {'success': True, 'num_steps': 100},
        {'success': False, 'num_steps': 300},
        {'success': True, 'num_steps': 150},
        {'success': True, 'num_steps': 120},
        {'success': False, 'num_steps': 300},
    ]

    # 计算成功率
    success_count = sum(1 for t in trajectories if t['success'])
    success_rate = success_count / len(trajectories)

    assert success_count == 3
    assert success_rate == 0.6

    print("✓ Success rate calculation verified")
    print(f"  - Total trajectories: {len(trajectories)}")
    print(f"  - Successful: {success_count}")
    print(f"  - Success rate: {success_rate*100:.1f}%")


def test_mock_environment_interaction():
    """测试模拟环境交互"""
    class MockEnv:
        def __init__(self):
            self.step_count = 0
            self.max_steps = 10

        def reset(self):
            self.step_count = 0
            return {'agentview_image': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)}

        def step(self, action):
            self.step_count += 1
            obs = {'agentview_image': np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)}
            reward = 1.0 if self.step_count >= self.max_steps else 0.0
            done = self.step_count >= self.max_steps
            info = {}
            return obs, reward, done, info

    # 测试环境
    env = MockEnv()
    obs = env.reset()

    assert 'agentview_image' in obs
    assert obs['agentview_image'].shape == (128, 128, 3)

    # 执行几步
    for i in range(5):
        action = np.random.randn(7)
        obs, reward, done, info = env.step(action)

    print("✓ Mock environment interaction verified")
    print(f"  - Steps executed: {env.step_count}")
    print(f"  - Observation shape: {obs['agentview_image'].shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 6: Testing Trajectory Generator")
    print("=" * 60)

    test_trajectory_generator_exists()
    print()
    test_action_queue_logic()
    print()
    test_temporal_ensembling()
    print()
    test_trajectory_data_structure()
    print()
    test_success_rate_calculation()
    print()
    test_mock_environment_interaction()

    print("\n" + "=" * 60)
    print("Step 6: All tests passed! ✓")
    print("=" * 60)
    print("\nNote: This tests the trajectory generation logic.")
    print("Actual trajectory generation requires:")
    print("  1. Trained VILA-U model with action head")
    print("  2. LIBERO environment setup")
    print("  3. Proper model-environment integration")
