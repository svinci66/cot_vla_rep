"""
测试 Step 7: 验证 LIBERO 格式保存

测试将轨迹保存为 LIBERO 兼容的 HDF5 格式
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import tempfile
import h5py
import json
import numpy as np


def test_libero_saver_exists():
    """测试 LIBERO 保存器文件是否存在"""
    saver_file = os.path.join(project_root, 'vila_u', 'utils', 'libero_saver.py')

    if not os.path.exists(saver_file):
        raise AssertionError(f"LIBERO saver file not found: {saver_file}")

    with open(saver_file, 'r') as f:
        content = f.read()

    # 检查关键类和方法
    required_items = [
        'class LiberoSaver',
        'def save_trajectory',
        'def save_multiple_trajectories',
        'def append_trajectory',
        'def verify_libero_format',
        'def convert_trajectory_to_libero',
    ]

    for item in required_items:
        if item not in content:
            raise AssertionError(f"Missing in libero_saver.py: {item}")

    print("✓ LiberoSaver class exists")
    print("  - save_trajectory(): Save single trajectory")
    print("  - save_multiple_trajectories(): Save multiple trajectories")
    print("  - append_trajectory(): Append to existing file")
    print("  - verify_libero_format(): Validate format")


def test_save_single_trajectory():
    """测试保存单条轨迹"""
    from vila_u.utils.libero_saver import LiberoSaver

    # 创建模拟轨迹
    num_steps = 50
    trajectory = {
        'observations': [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(num_steps)],
        'actions': np.random.randn(num_steps, 7).astype(np.float32),
        'rewards': np.random.rand(num_steps).astype(np.float32),
        'dones': np.zeros(num_steps, dtype=bool),
        'instruction': 'test task',
    }

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        saver = LiberoSaver(tmp_path)
        saver.save_trajectory(trajectory)

        # 验证文件存在
        assert os.path.exists(tmp_path), "Output file should exist"

        # 读取并验证
        with h5py.File(tmp_path, 'r') as f:
            assert 'data' in f
            assert 'demo_0' in f['data']

            demo = f['data']['demo_0']
            assert demo.attrs['num_samples'] == num_steps
            assert demo['actions'].shape == (num_steps, 7)
            assert demo['rewards'].shape == (num_steps,)
            assert demo['obs']['agentview_rgb'].shape == (num_steps, 128, 128, 3)

        print("✓ Single trajectory saved and verified")
        print(f"  - File: {tmp_path}")
        print(f"  - Samples: {num_steps}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_save_multiple_trajectories():
    """测试保存多条轨迹"""
    from vila_u.utils.libero_saver import LiberoSaver

    # 创建多条轨迹
    trajectories = []
    for i in range(3):
        num_steps = 40 + i * 10
        trajectory = {
            'observations': [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(num_steps)],
            'actions': np.random.randn(num_steps, 7).astype(np.float32),
            'rewards': np.random.rand(num_steps).astype(np.float32),
            'dones': np.zeros(num_steps, dtype=bool),
            'instruction': f'test task {i}',
        }
        trajectories.append(trajectory)

    # 保存
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        saver = LiberoSaver(tmp_path)
        saver.save_multiple_trajectories(trajectories)

        # 验证
        with h5py.File(tmp_path, 'r') as f:
            assert 'data' in f
            data_group = f['data']

            # 检查所有 demo
            for i in range(3):
                demo_name = f'demo_{i}'
                assert demo_name in data_group
                demo = data_group[demo_name]
                expected_steps = 40 + i * 10
                assert demo.attrs['num_samples'] == expected_steps

        print("✓ Multiple trajectories saved and verified")
        print(f"  - Trajectories: {len(trajectories)}")
        print(f"  - Demos: demo_0, demo_1, demo_2")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_libero_format_validation():
    """测试 LIBERO 格式验证"""
    from vila_u.utils.libero_saver import LiberoSaver, verify_libero_format

    # 创建有效的 LIBERO 文件
    num_steps = 30
    trajectory = {
        'observations': [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(num_steps)],
        'actions': np.random.randn(num_steps, 7).astype(np.float32),
        'rewards': np.random.rand(num_steps).astype(np.float32),
        'dones': np.zeros(num_steps, dtype=bool),
        'instruction': 'test task',
    }

    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        saver = LiberoSaver(tmp_path)
        saver.save_trajectory(trajectory)

        # 验证格式
        is_valid = verify_libero_format(tmp_path)
        assert is_valid, "File should be valid LIBERO format"

        print("✓ LIBERO format validation passed")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_metadata_structure():
    """测试元数据结构"""
    from vila_u.utils.libero_saver import LiberoSaver

    num_steps = 20
    trajectory = {
        'observations': [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(num_steps)],
        'actions': np.random.randn(num_steps, 7).astype(np.float32),
        'rewards': np.random.rand(num_steps).astype(np.float32),
        'dones': np.zeros(num_steps, dtype=bool),
        'instruction': 'pick up the bowl',
    }

    problem_info = {
        'language_instruction': 'pick up the bowl',
        'task_id': 'test_task_001',
    }

    env_args = {
        'env_name': 'libero_goal',
        'type': 1,
        'env_kwargs': {'camera_heights': 128, 'camera_widths': 128},
    }

    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        saver = LiberoSaver(tmp_path)
        saver.save_trajectory(trajectory, env_args=env_args, problem_info=problem_info)

        # 验证元数据
        with h5py.File(tmp_path, 'r') as f:
            data_group = f['data']

            # 检查 problem_info
            problem_info_str = data_group.attrs['problem_info']
            problem_info_loaded = json.loads(problem_info_str)
            assert problem_info_loaded['language_instruction'] == 'pick up the bowl'

            # 检查 env_args
            env_args_str = data_group.attrs['env_args']
            env_args_loaded = json.loads(env_args_str)
            assert env_args_loaded['env_name'] == 'libero_goal'

        print("✓ Metadata structure verified")
        print(f"  - problem_info: {problem_info_loaded}")
        print(f"  - env_args: {env_args_loaded}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_append_trajectory():
    """测试追加轨迹"""
    from vila_u.utils.libero_saver import LiberoSaver

    # 创建初始文件
    num_steps = 30
    trajectory1 = {
        'observations': [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(num_steps)],
        'actions': np.random.randn(num_steps, 7).astype(np.float32),
        'rewards': np.random.rand(num_steps).astype(np.float32),
        'dones': np.zeros(num_steps, dtype=bool),
        'instruction': 'task 1',
    }

    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        saver = LiberoSaver(tmp_path)
        saver.save_trajectory(trajectory1)

        # 追加第二条轨迹
        trajectory2 = {
            'observations': [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(40)],
            'actions': np.random.randn(40, 7).astype(np.float32),
            'rewards': np.random.rand(40).astype(np.float32),
            'dones': np.zeros(40, dtype=bool),
            'instruction': 'task 2',
        }
        saver.append_trajectory(trajectory2)

        # 验证
        with h5py.File(tmp_path, 'r') as f:
            data_group = f['data']
            assert 'demo_0' in data_group
            assert 'demo_1' in data_group

            assert data_group['demo_0'].attrs['num_samples'] == 30
            assert data_group['demo_1'].attrs['num_samples'] == 40

        print("✓ Trajectory append verified")
        print("  - Initial: demo_0 (30 samples)")
        print("  - Appended: demo_1 (40 samples)")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Step 7: Testing LIBERO Format Saver")
    print("=" * 60)

    test_libero_saver_exists()
    print()
    test_save_single_trajectory()
    print()
    test_save_multiple_trajectories()
    print()
    test_libero_format_validation()
    print()
    test_metadata_structure()
    print()
    test_append_trajectory()

    print("\n" + "=" * 60)
    print("Step 7: All tests passed! ✓")
    print("=" * 60)
    print("\nNote: This tests the LIBERO format saving logic.")
    print("Generated files are compatible with LIBERO dataset format.")
