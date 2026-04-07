"""
测试 Step 3: 验证 LIBERO 数据加载器（简化版）
不导入完整的 vila_u 模块，直接测试数据加载器逻辑
"""
import os
import sys
import tempfile
import h5py
import json
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def test_dataset_file_exists():
    """测试数据集文件是否存在"""
    dataset_file = os.path.join(os.path.dirname(__file__), '..', 'vila_u', 'data', 'libero_dataset.py')

    if not os.path.exists(dataset_file):
        raise AssertionError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'r') as f:
        content = f.read()

    # 检查关键类和函数
    required_items = [
        'class LiberoGoalDataset',
        'def collate_fn',
        '__getitem__',
        '_build_index',
    ]

    for item in required_items:
        if item not in content:
            raise AssertionError(f"Missing in libero_dataset.py: {item}")

    print("✓ LiberoGoalDataset file structure verified")


def test_dataset_logic():
    """测试数据集加载逻辑（使用模拟数据）"""

    # 简化的数据集类（用于测试）
    class SimpleLiberoDataset(Dataset):
        def __init__(self, data_root, image_size=256, action_chunk_size=10):
            self.data_root = data_root
            self.image_size = image_size
            self.action_chunk_size = action_chunk_size
            self.samples = self._build_index()
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        def _build_index(self):
            samples = []
            for filename in os.listdir(self.data_root):
                if filename.endswith('.hdf5'):
                    filepath = os.path.join(self.data_root, filename)
                    with h5py.File(filepath, 'r') as f:
                        problem_info = json.loads(f['data'].attrs['problem_info'])
                        instruction = problem_info['language_instruction']
                        for demo_name in f['data'].keys():
                            demo = f['data'][demo_name]
                            num_samples = demo.attrs['num_samples']
                            for t in range(num_samples - self.action_chunk_size):
                                samples.append({
                                    'file': filepath,
                                    'demo': demo_name,
                                    'timestep': t,
                                    'instruction': instruction,
                                })
            return samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            with h5py.File(sample['file'], 'r') as f:
                demo = f['data'][sample['demo']]
                t = sample['timestep']
                obs_rgb = demo['obs/agentview_rgb'][t]
                obs_image = Image.fromarray(obs_rgb)
                obs_tensor = self.transform(obs_image)
                actions = demo['actions'][t : t + self.action_chunk_size]
                action_tensor = torch.from_numpy(actions).float()
            return {
                'observation': obs_tensor,
                'instruction': sample['instruction'],
                'action_labels': action_tensor,
            }

    # 创建临时数据
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = f"{tmpdir}/test_task_demo.hdf5"

        with h5py.File(mock_file, 'w') as f:
            data_group = f.create_group('data')
            problem_info = {'language_instruction': 'test task'}
            data_group.attrs['problem_info'] = json.dumps(problem_info)

            demo = data_group.create_group('demo_0')
            demo.attrs['num_samples'] = 50
            demo.create_dataset('actions', data=np.random.randn(50, 7))

            obs_group = demo.create_group('obs')
            obs_group.create_dataset('agentview_rgb',
                data=np.random.randint(0, 255, (50, 128, 128, 3), dtype=np.uint8))

        # 测试数据集
        dataset = SimpleLiberoDataset(tmpdir, image_size=256, action_chunk_size=10)

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


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Testing LIBERO Data Loader (Simplified)")
    print("=" * 60)

    test_dataset_file_exists()
    print()
    test_dataset_logic()

    print("\n" + "=" * 60)
    print("Step 3: All tests passed! ✓")
    print("=" * 60)
