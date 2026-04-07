"""
LIBERO Goal 数据集加载器
用于训练 VILA-U 的动作预测能力
"""
import os
import json
import random
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class LiberoGoalDataset(Dataset):
    """
    LIBERO Goal 数据集加载器

    每条样本返回：
        - observation: 当前帧图像 [3, H, W]
        - instruction: 任务文本指令 str
        - action_labels: 接下来 ACTION_CHUNK_SIZE 步的动作 [chunk_size, 7]
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        action_chunk_size: int = 10,
        transform=None,
    ):
        """
        Args:
            data_root: LIBERO 数据集根目录
            image_size: 图像尺寸
            action_chunk_size: 动作 chunk 大小
            transform: 图像变换
        """
        self.data_root = data_root
        self.image_size = image_size
        self.action_chunk_size = action_chunk_size

        # 默认图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        # 加载数据集索引
        self.samples = self._build_index()

        print(f"[LiberoGoalDataset] Loaded {len(self.samples)} samples from {data_root}")

    def _build_index(self):
        """构建数据集索引"""
        samples = []

        # 遍历所有 .hdf5 文件
        for filename in os.listdir(self.data_root):
            if not filename.endswith('.hdf5'):
                continue

            filepath = os.path.join(self.data_root, filename)

            with h5py.File(filepath, 'r') as f:
                # 获取语言指令
                problem_info = json.loads(f['data'].attrs['problem_info'])
                instruction = problem_info['language_instruction']

                # 遍历所有演示
                for demo_name in f['data'].keys():
                    demo = f['data'][demo_name]
                    num_samples = demo.attrs['num_samples']

                    # 为每个有效的起始位置创建一个样本
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

            # 加载观察图像
            obs_rgb = demo['obs/agentview_rgb'][t]  # [H, W, 3]
            obs_image = Image.fromarray(obs_rgb)
            obs_tensor = self.transform(obs_image)  # [3, H, W]

            # 加载动作序列
            actions = demo['actions'][t : t + self.action_chunk_size]  # [chunk, 7]
            action_tensor = torch.from_numpy(actions).float()

        return {
            'observation': obs_tensor,
            'instruction': sample['instruction'],
            'action_labels': action_tensor,
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    observations = torch.stack([item['observation'] for item in batch])
    instructions = [item['instruction'] for item in batch]
    action_labels = torch.stack([item['action_labels'] for item in batch])

    return {
        'observations': observations,
        'instructions': instructions,
        'action_labels': action_labels,
    }
