#!/usr/bin/env python
"""
LIBERO 端到端评估脚本

在 LIBERO 环境中评估训练好的 VILA-U 模型
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from vila_u.eval.trajectory_generator import create_trajectory_generator
from vila_u.utils.libero_saver import LiberoSaver


def evaluate_on_libero(
    model_path: str,
    checkpoint_path: str,
    benchmark_name: str = "libero_goal",
    task_id: int = 0,
    num_episodes: int = 10,
    output_dir: str = "./eval_results",
    save_trajectories: bool = True,
    device: str = "cuda",
):
    """
    在 LIBERO 环境中评估模型

    Args:
        model_path: VILA-U 模型路径
        checkpoint_path: 动作预测头检查点路径
        benchmark_name: LIBERO benchmark 名称
        task_id: 任务 ID
        num_episodes: 评估轨迹数量
        output_dir: 输出目录
        save_trajectories: 是否保存轨迹
        device: 设备
    """
    print("=" * 70)
    print("VILA-U LIBERO Evaluation")
    print("=" * 70)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 LIBERO 环境
    print("\n[1/5] Loading LIBERO environment...")
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        # 获取 benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[benchmark_name]()

        # 获取任务
        task = benchmark_instance.get_task(task_id)
        task_name = task.name
        instruction = task.language

        print(f"  ✓ Benchmark: {benchmark_name}")
        print(f"  ✓ Task {task_id}: {task_name}")
        print(f"  ✓ Instruction: {instruction}")

        # 创建环境
        bddl_file = benchmark_instance.get_task_bddl_file_path(task_id)
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)

        print(f"  ✓ Environment created")

    except ImportError:
        print("  ✗ LIBERO not installed. Please install LIBERO first.")
        print("    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
        print("    cd LIBERO && pip install -e .")
        return

    # 2. 加载模型
    print("\n[2/5] Loading VILA-U model...")
    generator = create_trajectory_generator(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print(f"  ✓ Model loaded from {model_path}")
    print(f"  ✓ Checkpoint loaded from {checkpoint_path}")

    # 3. 生成轨迹
    print(f"\n[3/5] Generating {num_episodes} trajectories...")
    trajectories = generator.generate_multiple_trajectories(
        env=env,
        instruction=instruction,
        num_trajectories=num_episodes,
        camera_name="agentview_image",
        verbose=True,
    )

    # 4. 计算评估指标
    print("\n[4/5] Computing evaluation metrics...")

    success_count = sum(1 for t in trajectories if t['success'])
    success_rate = success_count / num_episodes

    avg_steps = np.mean([t['num_steps'] for t in trajectories])
    avg_reward = np.mean([np.sum(t['rewards']) for t in trajectories])

    successful_trajectories = [t for t in trajectories if t['success']]
    if successful_trajectories:
        avg_steps_success = np.mean([t['num_steps'] for t in successful_trajectories])
    else:
        avg_steps_success = 0

    metrics = {
        'benchmark': benchmark_name,
        'task_id': task_id,
        'task_name': task_name,
        'instruction': instruction,
        'num_episodes': num_episodes,
        'success_count': success_count,
        'success_rate': success_rate,
        'avg_steps': float(avg_steps),
        'avg_steps_success': float(avg_steps_success),
        'avg_reward': float(avg_reward),
    }

    print(f"  ✓ Success Rate: {success_rate*100:.1f}% ({success_count}/{num_episodes})")
    print(f"  ✓ Average Steps: {avg_steps:.1f}")
    print(f"  ✓ Average Steps (Success): {avg_steps_success:.1f}")
    print(f"  ✓ Average Reward: {avg_reward:.3f}")

    # 5. 保存结果
    print("\n[5/5] Saving results...")

    # 保存评估指标
    metrics_path = os.path.join(output_dir, f"{task_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")

    # 保存轨迹
    if save_trajectories:
        trajectories_path = os.path.join(output_dir, f"{task_name}_trajectories.hdf5")
        saver = LiberoSaver(trajectories_path)

        env_args_dict = {
            'env_name': benchmark_name,
            'type': 1,
            'env_kwargs': env_args,
        }

        problem_info = {
            'language_instruction': instruction,
            'task_name': task_name,
        }

        saver.save_multiple_trajectories(
            trajectories,
            env_args=env_args_dict,
            problem_info=problem_info,
        )
        print(f"  ✓ Trajectories saved to {trajectories_path}")

    # 关闭环境
    env.close()

    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)

    return metrics


def evaluate_full_benchmark(
    model_path: str,
    checkpoint_path: str,
    benchmark_name: str = "libero_goal",
    num_episodes_per_task: int = 10,
    output_dir: str = "./eval_results",
    device: str = "cuda",
):
    """
    评估整个 benchmark 的所有任务

    Args:
        model_path: VILA-U 模型路径
        checkpoint_path: 动作预测头检查点路径
        benchmark_name: LIBERO benchmark 名称
        num_episodes_per_task: 每个任务的评估轨迹数量
        output_dir: 输出目录
        device: 设备
    """
    print("=" * 70)
    print(f"Evaluating Full Benchmark: {benchmark_name}")
    print("=" * 70)

    # 加载 benchmark
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[benchmark_name]()
    num_tasks = benchmark_instance.get_num_tasks()

    print(f"\nTotal tasks: {num_tasks}")
    print(f"Episodes per task: {num_episodes_per_task}")

    # 评估每个任务
    all_metrics = []

    for task_id in range(num_tasks):
        print(f"\n{'='*70}")
        print(f"Task {task_id+1}/{num_tasks}")
        print(f"{'='*70}")

        metrics = evaluate_on_libero(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            benchmark_name=benchmark_name,
            task_id=task_id,
            num_episodes=num_episodes_per_task,
            output_dir=output_dir,
            save_trajectories=True,
            device=device,
        )

        all_metrics.append(metrics)

    # 计算总体统计
    overall_success_rate = np.mean([m['success_rate'] for m in all_metrics])
    overall_avg_steps = np.mean([m['avg_steps'] for m in all_metrics])

    print("\n" + "=" * 70)
    print("Overall Results")
    print("=" * 70)
    print(f"Overall Success Rate: {overall_success_rate*100:.1f}%")
    print(f"Overall Average Steps: {overall_avg_steps:.1f}")

    # 保存总体结果
    summary_path = os.path.join(output_dir, f"{benchmark_name}_summary.json")
    summary = {
        'benchmark': benchmark_name,
        'num_tasks': num_tasks,
        'num_episodes_per_task': num_episodes_per_task,
        'overall_success_rate': float(overall_success_rate),
        'overall_avg_steps': float(overall_avg_steps),
        'per_task_metrics': all_metrics,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VILA-U on LIBERO")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to VILA-U model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to action head checkpoint")

    # 评估参数
    parser.add_argument("--benchmark", type=str, default="libero_goal",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                       help="LIBERO benchmark name")
    parser.add_argument("--task_id", type=int, default=None,
                       help="Specific task ID to evaluate (if None, evaluate all tasks)")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes per task")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Output directory")
    parser.add_argument("--save_trajectories", action="store_true",
                       help="Save generated trajectories")

    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # 评估
    if args.task_id is not None:
        # 评估单个任务
        evaluate_on_libero(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint_path,
            benchmark_name=args.benchmark,
            task_id=args.task_id,
            num_episodes=args.num_episodes,
            output_dir=args.output_dir,
            save_trajectories=args.save_trajectories,
            device=args.device,
        )
    else:
        # 评估整个 benchmark
        evaluate_full_benchmark(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint_path,
            benchmark_name=args.benchmark,
            num_episodes_per_task=args.num_episodes,
            output_dir=args.output_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
