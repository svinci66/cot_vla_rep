"""
测试 Step 3: 验证 LIBERO 数据加载器
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader


def test_dataset_creation():
    """测试数据集创建（不需要实际数据）"""
    print("Testing dataset creation logic...")

    # 这里只测试类定义，不加载实际数据
    from vila_u.data.libero_dataset import LiberoGoalDataset

    print("✓ LiberoGoalDataset class imported successfully")


def test_dataset_with_mock_data():
    """使用模拟数据测试数据集"""
    import tempfile
    import h5py
    import json
    import numpy as np
    from vila_u.data.libero_dataset import LiberoGoalDataset, collate_fn

    # 创建临时目录和模拟数据
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟 HDF5 文件
        mock_file = f"{tmpdir}/test_task_demo.hdf5"

        with h5py.File(mock_file, 'w') as f:
            data_group = f.create_group('data')

            # 添加元数据
            problem_info = {'language_instruction': 'test task'}
            data_group.attrs['problem_info'] = json.dumps(problem_info)

            # 创建一个演示
            demo = data_group.create_group('demo_0')
            demo.attrs['num_samples'] = 50

            # 创建模拟数据
            demo.create_dataset('actions', data=np.random.randn(50, 7))

            obs_group = demo.create_group('obs')
            obs_group.create_dataset('agentview_rgb',
                data=np.random.randint(0, 255, (50, 128, 128, 3), dtype=np.uint8))

        # 测试数据集加载
        dataset = LiberoGoalDataset(
            data_root=tmpdir,
            image_size=256,
            action_chunk_size=10,
        )

        print(f"✓ Dataset created with {len(dataset)} samples")

        # 测试单个样本
        sample = dataset[0]
        assert 'observation' in sample
        assert 'instruction' in sample
        assert 'action_labels' in sample
        assert sample['observation'].shape == (3, 256, 256)
        assert sample['action_labels'].shape == (10, 7)

        print(f"✓ Sample structure verified")
        print(f"  - observation shape: {sample['observation'].shape}")
        print(f"  - instruction: {sample['instruction']}")
        print(f"  - action_labels shape: {sample['action_labels'].shape}")

        # 测试 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
        )

        batch = next(iter(dataloader))
        assert batch['observations'].shape == (2, 3, 256, 256)
        assert len(batch['instructions']) == 2
        assert batch['action_labels'].shape == (2, 10, 7)

        print(f"✓ DataLoader working correctly")
        print(f"  - batch observations shape: {batch['observations'].shape}")
        print(f"  - batch action_labels shape: {batch['action_labels'].shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Testing LIBERO Data Loader")
    print("=" * 60)

    test_dataset_creation()
    print()
    test_dataset_with_mock_data()

    print("\n" + "=" * 60)
    print("Step 3: All tests passed! ✓")
    print("=" * 60)
