"""
LIBERO 格式保存器

将生成的轨迹保存为 LIBERO 兼容的 HDF5 格式
"""

import h5py
import json
import numpy as np
import os
from typing import Dict, List, Optional


class LiberoSaver:
    """
    将轨迹保存为 LIBERO HDF5 格式
    """

    def __init__(self, output_path: str):
        """
        Args:
            output_path: 输出 HDF5 文件路径
        """
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    def save_trajectory(
        self,
        trajectory: Dict,
        env_args: Optional[Dict] = None,
        problem_info: Optional[Dict] = None,
    ):
        """
        保存单条轨迹到 HDF5 文件

        Args:
            trajectory: 轨迹数据字典
                - observations: List[np.ndarray] - RGB 图像序列
                - actions: np.ndarray [T, 7] - 动作序列
                - rewards: np.ndarray [T] - 奖励序列
                - dones: np.ndarray [T] - 终止标志
                - instruction: str - 语言指令
            env_args: 环境参数（可选）
            problem_info: 任务信息（可选）
        """
        with h5py.File(self.output_path, 'w') as f:
            # 创建 data 组
            data_group = f.create_group('data')

            # 设置元数据
            if problem_info is None:
                problem_info = {
                    'language_instruction': trajectory.get('instruction', 'unknown task')
                }
            data_group.attrs['problem_info'] = json.dumps(problem_info)

            if env_args is None:
                env_args = {
                    'env_name': 'libero_goal',
                    'type': 1,
                    'env_kwargs': {}
                }
            data_group.attrs['env_args'] = json.dumps(env_args)

            # 保存演示数据
            demo_name = 'demo_0'
            demo_group = data_group.create_group(demo_name)

            num_samples = len(trajectory['actions'])
            demo_group.attrs['num_samples'] = num_samples

            # 保存动作
            demo_group.create_dataset('actions', data=trajectory['actions'], compression='gzip')

            # 保存奖励
            demo_group.create_dataset('rewards', data=trajectory['rewards'], compression='gzip')

            # 保存 dones
            demo_group.create_dataset('dones', data=trajectory['dones'], compression='gzip')

            # 保存观察
            obs_group = demo_group.create_group('obs')

            # 将观察列表转换为数组
            observations = np.array(trajectory['observations'])  # [T, H, W, 3]

            # 保存 RGB 图像
            obs_group.create_dataset('agentview_rgb', data=observations, compression='gzip')

            # 如果有其他观察模态，也保存
            # 例如：机器人状态、深度图等

        print(f"✓ Trajectory saved to {self.output_path}")
        print(f"  - Samples: {num_samples}")
        print(f"  - Instruction: {trajectory.get('instruction', 'N/A')}")

    def save_multiple_trajectories(
        self,
        trajectories: List[Dict],
        env_args: Optional[Dict] = None,
        problem_info: Optional[Dict] = None,
    ):
        """
        保存多条轨迹到 HDF5 文件

        Args:
            trajectories: 轨迹列表
            env_args: 环境参数（可选）
            problem_info: 任务信息（可选）
        """
        with h5py.File(self.output_path, 'w') as f:
            # 创建 data 组
            data_group = f.create_group('data')

            # 设置元数据（使用第一条轨迹的信息）
            if problem_info is None:
                problem_info = {
                    'language_instruction': trajectories[0].get('instruction', 'unknown task')
                }
            data_group.attrs['problem_info'] = json.dumps(problem_info)

            if env_args is None:
                env_args = {
                    'env_name': 'libero_goal',
                    'type': 1,
                    'env_kwargs': {}
                }
            data_group.attrs['env_args'] = json.dumps(env_args)

            # 保存每条轨迹
            for idx, trajectory in enumerate(trajectories):
                demo_name = f'demo_{idx}'
                demo_group = data_group.create_group(demo_name)

                num_samples = len(trajectory['actions'])
                demo_group.attrs['num_samples'] = num_samples

                # 保存动作
                demo_group.create_dataset('actions', data=trajectory['actions'], compression='gzip')

                # 保存奖励
                demo_group.create_dataset('rewards', data=trajectory['rewards'], compression='gzip')

                # 保存 dones
                demo_group.create_dataset('dones', data=trajectory['dones'], compression='gzip')

                # 保存观察
                obs_group = demo_group.create_group('obs')
                observations = np.array(trajectory['observations'])
                obs_group.create_dataset('agentview_rgb', data=observations, compression='gzip')

        print(f"✓ {len(trajectories)} trajectories saved to {self.output_path}")

    def append_trajectory(
        self,
        trajectory: Dict,
    ):
        """
        向已有的 HDF5 文件追加轨迹

        Args:
            trajectory: 轨迹数据字典
        """
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"File not found: {self.output_path}")

        with h5py.File(self.output_path, 'a') as f:
            data_group = f['data']

            # 找到下一个 demo 编号
            existing_demos = list(data_group.keys())
            demo_indices = [int(name.split('_')[1]) for name in existing_demos if name.startswith('demo_')]
            next_idx = max(demo_indices) + 1 if demo_indices else 0

            # 创建新的 demo
            demo_name = f'demo_{next_idx}'
            demo_group = data_group.create_group(demo_name)

            num_samples = len(trajectory['actions'])
            demo_group.attrs['num_samples'] = num_samples

            # 保存数据
            demo_group.create_dataset('actions', data=trajectory['actions'], compression='gzip')
            demo_group.create_dataset('rewards', data=trajectory['rewards'], compression='gzip')
            demo_group.create_dataset('dones', data=trajectory['dones'], compression='gzip')

            obs_group = demo_group.create_group('obs')
            observations = np.array(trajectory['observations'])
            obs_group.create_dataset('agentview_rgb', data=observations, compression='gzip')

        print(f"✓ Trajectory appended as {demo_name}")


def verify_libero_format(file_path: str) -> bool:
    """
    验证 HDF5 文件是否符合 LIBERO 格式

    Args:
        file_path: HDF5 文件路径

    Returns:
        valid: 是否有效
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 检查必需的组和属性
            if 'data' not in f:
                print("✗ Missing 'data' group")
                return False

            data_group = f['data']

            # 检查元数据
            if 'problem_info' not in data_group.attrs:
                print("✗ Missing 'problem_info' attribute")
                return False

            if 'env_args' not in data_group.attrs:
                print("✗ Missing 'env_args' attribute")
                return False

            # 检查至少有一个 demo
            demos = [key for key in data_group.keys() if key.startswith('demo_')]
            if len(demos) == 0:
                print("✗ No demonstrations found")
                return False

            # 检查第一个 demo 的结构
            demo = data_group[demos[0]]

            required_datasets = ['actions', 'rewards', 'dones']
            for dataset in required_datasets:
                if dataset not in demo:
                    print(f"✗ Missing dataset: {dataset}")
                    return False

            if 'obs' not in demo:
                print("✗ Missing 'obs' group")
                return False

            obs_group = demo['obs']
            if 'agentview_rgb' not in obs_group:
                print("✗ Missing 'agentview_rgb' in observations")
                return False

            print(f"✓ Valid LIBERO format")
            print(f"  - Demonstrations: {len(demos)}")
            print(f"  - Samples in demo_0: {demo.attrs.get('num_samples', 'N/A')}")

            return True

    except Exception as e:
        print(f"✗ Error validating file: {e}")
        return False


def convert_trajectory_to_libero(
    trajectory: Dict,
    output_path: str,
    env_args: Optional[Dict] = None,
    problem_info: Optional[Dict] = None,
):
    """
    便捷函数：将轨迹转换并保存为 LIBERO 格式

    Args:
        trajectory: 轨迹数据
        output_path: 输出路径
        env_args: 环境参数
        problem_info: 任务信息
    """
    saver = LiberoSaver(output_path)
    saver.save_trajectory(trajectory, env_args, problem_info)

    # 验证
    if verify_libero_format(output_path):
        print(f"✓ Successfully saved and verified: {output_path}")
    else:
        print(f"✗ Validation failed: {output_path}")
