"""
LIBERO Goal 数据集加载器 - Phase 4: Visual CoT-VLA
支持加载观测图像、子目标图像和动作序列
"""
import os
import json
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class LiberoCoTDataset(Dataset):
    """
    LIBERO Goal 数据集加载器 - Phase 4 版本

    每条样本返回：
        - observations: 当前帧图像 [3, H, W]
        - instructions: 任务文本指令 str
        - subgoal_images: 子目标图像 [3, H, W]（未来某一帧）
        - action_labels: 接下来 ACTION_CHUNK_SIZE 步的动作 [chunk_size, 7]
    """

    def __init__(
        self,
        data_root: str,
        tokenizer,
        image_size: int = 256,
        action_chunk_size: int = 10,
        subgoal_horizon: int = 5,
        transform=None,
        remove_pause_intervals: bool = True,
        pause_threshold: float = 0.01,
    ):
        """
        Args:
            data_root: LIBERO 数据集根目录
            tokenizer: tokenizer（用于兼容性）
            image_size: 图像尺寸
            action_chunk_size: 动作 chunk 大小
            subgoal_horizon: 子目标的时间跨度（未来第几帧作为子目标）
            transform: 图像变换
            remove_pause_intervals: 是否移除暂停区间
            pause_threshold: 暂停检测阈值
        """
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.action_chunk_size = action_chunk_size
        self.subgoal_horizon = subgoal_horizon
        self.remove_pause_intervals = remove_pause_intervals
        self.pause_threshold = pause_threshold

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

        print(f"[LiberoCoTDataset] Loaded {len(self.samples)} samples from {data_root}")
        print(f"[LiberoCoTDataset] Subgoal horizon: {subgoal_horizon} steps")

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

                    # 过滤暂停区间
                    if self.remove_pause_intervals:
                        actions = demo['actions'][:]
                        action_norms = np.linalg.norm(actions, axis=-1)
                        valid_indices = np.where(action_norms > self.pause_threshold)[0]
                    else:
                        valid_indices = np.arange(num_samples)

                    # 为每个有效的起始位置创建一个样本
                    # 需要确保有足够的未来帧用于子目标和动作
                    min_future_steps = max(self.subgoal_horizon, self.action_chunk_size)
                    for t in valid_indices:
                        if t + min_future_steps < num_samples:
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

            # 1. 加载观察图像（当前帧）
            obs_rgb = demo['obs/agentview_rgb'][t]  # [H, W, 3]
            obs_image = Image.fromarray(obs_rgb)
            obs_tensor = self.transform(obs_image)  # [3, H, W]

            # 2. 加载子目标图像（未来第 subgoal_horizon 帧）
            subgoal_t = t + self.subgoal_horizon
            subgoal_rgb = demo['obs/agentview_rgb'][subgoal_t]  # [H, W, 3]
            subgoal_image = Image.fromarray(subgoal_rgb)
            subgoal_tensor = self.transform(subgoal_image)  # [3, H, W]

            # 3. 加载动作序列
            actions = demo['actions'][t : t + self.action_chunk_size]  # [chunk, 7]
            action_tensor = torch.from_numpy(actions).float()

        return {
            'observations': obs_tensor,
            'instructions': sample['instruction'],
            'subgoal_images': subgoal_tensor,
            'action_labels': action_tensor,
        }
