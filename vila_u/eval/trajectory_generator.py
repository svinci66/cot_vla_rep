"""
VILA-U 轨迹生成器

在 LIBERO 环境中闭环生成完整的机器人执行轨迹
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class TrajectoryGenerator:
    """
    轨迹生成器：使用 VILA-U 模型在 LIBERO 环境中生成动作轨迹
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        action_chunk_size: int = 10,
        max_steps: int = 300,
        device: str = "cuda",
    ):
        """
        Args:
            model: VILA-U 模型（带动作预测头）
            tokenizer: 文本 tokenizer
            image_processor: 图像预处理器
            action_chunk_size: 动作 chunk 大小
            max_steps: 最大执行步数
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.action_chunk_size = action_chunk_size
        self.max_steps = max_steps
        self.device = device

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def generate_trajectory(
        self,
        env,
        instruction: str,
        camera_name: str = "agentview_image",
        verbose: bool = True,
    ) -> Dict:
        """
        在环境中生成完整轨迹

        Args:
            env: LIBERO 环境实例
            instruction: 任务语言指令
            camera_name: 相机名称
            verbose: 是否打印详细信息

        Returns:
            trajectory: 包含完整轨迹数据的字典
                - observations: List[np.ndarray] - RGB 图像序列
                - actions: np.ndarray [T, 7] - 执行的动作序列
                - rewards: np.ndarray [T] - 奖励序列
                - dones: np.ndarray [T] - 终止标志
                - success: bool - 是否成功完成任务
                - num_steps: int - 执行步数
        """
        if verbose:
            print(f"Generating trajectory for: {instruction}")

        # 重置环境
        obs = env.reset()

        # 初始化轨迹存储
        observations = []
        actions_executed = []
        rewards = []
        dones = []

        # 动作队列（用于 temporal ensembling）
        action_queue = deque(maxlen=self.action_chunk_size)

        step = 0
        done = False
        success = False

        while step < self.max_steps and not done:
            # 1. 获取当前观察
            current_obs = obs[camera_name]  # [H, W, 3]
            observations.append(current_obs.copy())

            # 2. 预测动作 chunk
            if len(action_queue) == 0:
                # 队列为空，需要预测新的动作 chunk
                action_chunk = self.model.predict_action(
                    image=current_obs,
                    instruction=instruction,
                    image_processor=self.image_processor,
                )  # [chunk_size, 7]

                # 将动作加入队列
                for i in range(self.action_chunk_size):
                    action_queue.append(action_chunk[i].cpu().numpy())

            # 3. 从队列中取出下一个动作
            action = action_queue.popleft()  # [7]

            # 4. 在环境中执行动作
            obs, reward, done, info = env.step(action)

            # 5. 记录数据
            actions_executed.append(action)
            rewards.append(reward)
            dones.append(done)

            step += 1

            if verbose and step % 50 == 0:
                print(f"  Step {step}/{self.max_steps}, Reward: {reward:.3f}")

            # 检查是否成功
            if done and reward > 0:
                success = True
                break

        # 转换为 numpy 数组
        actions_array = np.array(actions_executed)  # [T, 7]
        rewards_array = np.array(rewards)  # [T]
        dones_array = np.array(dones)  # [T]

        trajectory = {
            'observations': observations,
            'actions': actions_array,
            'rewards': rewards_array,
            'dones': dones_array,
            'success': success,
            'num_steps': step,
            'instruction': instruction,
        }

        if verbose:
            print(f"  Trajectory completed: {step} steps, Success: {success}")

        return trajectory

    def generate_multiple_trajectories(
        self,
        env,
        instruction: str,
        num_trajectories: int = 10,
        camera_name: str = "agentview_image",
        verbose: bool = True,
    ) -> List[Dict]:
        """
        生成多条轨迹（用于评估）

        Args:
            env: LIBERO 环境实例
            instruction: 任务语言指令
            num_trajectories: 生成轨迹数量
            camera_name: 相机名称
            verbose: 是否打印详细信息

        Returns:
            trajectories: 轨迹列表
        """
        trajectories = []
        success_count = 0

        for i in range(num_trajectories):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Trajectory {i+1}/{num_trajectories}")
                print(f"{'='*60}")

            trajectory = self.generate_trajectory(
                env=env,
                instruction=instruction,
                camera_name=camera_name,
                verbose=verbose,
            )

            trajectories.append(trajectory)

            if trajectory['success']:
                success_count += 1

        # 计算成功率
        success_rate = success_count / num_trajectories

        if verbose:
            print(f"\n{'='*60}")
            print(f"Summary: {success_count}/{num_trajectories} successful")
            print(f"Success Rate: {success_rate*100:.1f}%")
            print(f"{'='*60}")

        return trajectories

    def generate_with_temporal_ensembling(
        self,
        env,
        instruction: str,
        camera_name: str = "agentview_image",
        ensemble_k: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        使用 temporal ensembling 生成轨迹

        在每个时间步，使用过去 k 个预测的平均值作为执行动作

        Args:
            env: LIBERO 环境实例
            instruction: 任务语言指令
            camera_name: 相机名称
            ensemble_k: ensemble 窗口大小
            verbose: 是否打印详细信息

        Returns:
            trajectory: 轨迹数据
        """
        if verbose:
            print(f"Generating trajectory with temporal ensembling (k={ensemble_k})")

        # 重置环境
        obs = env.reset()

        # 初始化
        observations = []
        actions_executed = []
        rewards = []
        dones = []

        # 动作历史（用于 ensembling）
        action_history = deque(maxlen=ensemble_k)

        step = 0
        done = False
        success = False

        while step < self.max_steps and not done:
            # 获取观察
            current_obs = obs[camera_name]
            observations.append(current_obs.copy())

            # 预测动作
            action_chunk = self.model.predict_action(
                image=current_obs,
                instruction=instruction,
                image_processor=self.image_processor,
            )  # [chunk_size, 7]

            # 取第一个动作
            predicted_action = action_chunk[0].cpu().numpy()

            # 加入历史
            action_history.append(predicted_action)

            # Ensemble：取平均
            if len(action_history) > 0:
                action = np.mean(list(action_history), axis=0)
            else:
                action = predicted_action

            # 执行动作
            obs, reward, done, info = env.step(action)

            # 记录
            actions_executed.append(action)
            rewards.append(reward)
            dones.append(done)

            step += 1

            if verbose and step % 50 == 0:
                print(f"  Step {step}/{self.max_steps}, Reward: {reward:.3f}")

            if done and reward > 0:
                success = True
                break

        trajectory = {
            'observations': observations,
            'actions': np.array(actions_executed),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'success': success,
            'num_steps': step,
            'instruction': instruction,
        }

        if verbose:
            print(f"  Trajectory completed: {step} steps, Success: {success}")

        return trajectory


def create_trajectory_generator(
    model_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> TrajectoryGenerator:
    """
    创建轨迹生成器的便捷函数

    Args:
        model_path: VILA-U 模型路径
        checkpoint_path: 动作预测头检查点路径（可选）
        device: 设备

    Returns:
        generator: TrajectoryGenerator 实例
    """
    from vila_u.model.builder import load_pretrained_model

    # 加载模型
    model, tokenizer, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="vila-u",
        device_map=device,
    )

    # 加载动作预测头检查点
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'action_head' in checkpoint:
            model.action_head.load_state_dict(checkpoint['action_head'])
            print(f"Loaded action head from {checkpoint_path}")

    # 创建生成器
    generator = TrajectoryGenerator(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
    )

    return generator
