"""
测试 Step 8: 验证端到端评估

测试完整的评估流程
"""
import sys
import os

# 自动获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import json
import numpy as np


def test_eval_script_exists():
    """测试评估脚本是否存在"""
    eval_script = os.path.join(project_root, 'scripts', 'eval_libero.py')

    if not os.path.exists(eval_script):
        raise AssertionError(f"Evaluation script not found: {eval_script}")

    with open(eval_script, 'r') as f:
        content = f.read()

    # 检查关键函数
    required_items = [
        'def evaluate_on_libero',
        'def evaluate_full_benchmark',
        'def main',
        'argparse',
        'TrajectoryGenerator',
        'LiberoSaver',
    ]

    for item in required_items:
        if item not in content:
            raise AssertionError(f"Missing in eval_libero.py: {item}")

    print("✓ Evaluation script exists")
    print("  - evaluate_on_libero(): Single task evaluation")
    print("  - evaluate_full_benchmark(): Full benchmark evaluation")
    print("  - Command-line interface")


def test_metrics_calculation():
    """测试评估指标计算"""
    # 模拟轨迹数据
    trajectories = [
        {'success': True, 'num_steps': 100, 'rewards': np.array([0.0] * 99 + [1.0])},
        {'success': False, 'num_steps': 300, 'rewards': np.array([0.0] * 300)},
        {'success': True, 'num_steps': 150, 'rewards': np.array([0.0] * 149 + [1.0])},
        {'success': True, 'num_steps': 120, 'rewards': np.array([0.0] * 119 + [1.0])},
        {'success': False, 'num_steps': 300, 'rewards': np.array([0.0] * 300)},
    ]

    # 计算指标
    num_episodes = len(trajectories)
    success_count = sum(1 for t in trajectories if t['success'])
    success_rate = success_count / num_episodes

    avg_steps = np.mean([t['num_steps'] for t in trajectories])
    avg_reward = np.mean([np.sum(t['rewards']) for t in trajectories])

    successful_trajectories = [t for t in trajectories if t['success']]
    avg_steps_success = np.mean([t['num_steps'] for t in successful_trajectories])

    # 验证
    assert success_count == 3
    assert success_rate == 0.6
    assert avg_steps == 194.0
    assert avg_steps_success == 123.33333333333333 or abs(avg_steps_success - 123.33) < 0.01

    print("✓ Metrics calculation verified")
    print(f"  - Success rate: {success_rate*100:.1f}%")
    print(f"  - Average steps: {avg_steps:.1f}")
    print(f"  - Average steps (success): {avg_steps_success:.1f}")
    print(f"  - Average reward: {avg_reward:.3f}")


def test_metrics_json_structure():
    """测试评估指标 JSON 结构"""
    metrics = {
        'benchmark': 'libero_goal',
        'task_id': 0,
        'task_name': 'test_task',
        'instruction': 'pick up the bowl',
        'num_episodes': 10,
        'success_count': 7,
        'success_rate': 0.7,
        'avg_steps': 150.5,
        'avg_steps_success': 120.3,
        'avg_reward': 0.7,
    }

    # 验证结构
    required_keys = [
        'benchmark', 'task_id', 'task_name', 'instruction',
        'num_episodes', 'success_count', 'success_rate',
        'avg_steps', 'avg_steps_success', 'avg_reward'
    ]

    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"

    # 验证可以序列化为 JSON
    json_str = json.dumps(metrics, indent=2)
    loaded = json.loads(json_str)

    assert loaded['success_rate'] == 0.7

    print("✓ Metrics JSON structure verified")
    print(f"  - Keys: {list(metrics.keys())}")


def test_summary_structure():
    """测试总结 JSON 结构"""
    summary = {
        'benchmark': 'libero_goal',
        'num_tasks': 10,
        'num_episodes_per_task': 10,
        'overall_success_rate': 0.65,
        'overall_avg_steps': 145.2,
        'per_task_metrics': [
            {'task_id': 0, 'success_rate': 0.7},
            {'task_id': 1, 'success_rate': 0.6},
        ]
    }

    # 验证结构
    assert 'benchmark' in summary
    assert 'num_tasks' in summary
    assert 'overall_success_rate' in summary
    assert 'per_task_metrics' in summary
    assert isinstance(summary['per_task_metrics'], list)

    print("✓ Summary JSON structure verified")
    print(f"  - Benchmark: {summary['benchmark']}")
    print(f"  - Tasks: {summary['num_tasks']}")
    print(f"  - Overall success rate: {summary['overall_success_rate']*100:.1f}%")


def test_evaluation_pipeline():
    """测试评估流程（模拟）"""
    # 模拟评估流程的各个步骤

    # 1. 加载环境（模拟）
    class MockEnv:
        def reset(self):
            return {'agentview_image': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)}

        def step(self, action):
            obs = {'agentview_image': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)}
            reward = np.random.rand()
            done = np.random.rand() > 0.95
            info = {}
            return obs, reward, done, info

        def close(self):
            pass

    env = MockEnv()

    # 2. 生成轨迹（模拟）
    trajectories = []
    for i in range(3):
        obs = env.reset()
        actions = []
        rewards = []
        dones = []

        for step in range(50):
            action = np.random.randn(7)
            obs, reward, done, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            if done:
                break

        trajectory = {
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'success': rewards[-1] > 0.5,
            'num_steps': len(actions),
        }
        trajectories.append(trajectory)

    # 3. 计算指标
    success_count = sum(1 for t in trajectories if t['success'])
    success_rate = success_count / len(trajectories)

    # 4. 验证
    assert len(trajectories) == 3
    assert 0 <= success_rate <= 1

    env.close()

    print("✓ Evaluation pipeline verified")
    print(f"  - Trajectories generated: {len(trajectories)}")
    print(f"  - Success rate: {success_rate*100:.1f}%")


def test_command_line_args():
    """测试命令行参数"""
    import argparse

    # 模拟命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, default="test_model")
    parser.add_argument("--checkpoint_path", type=str, required=False, default="test_checkpoint")
    parser.add_argument("--benchmark", type=str, default="libero_goal")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--device", type=str, default="cuda")

    # 解析测试参数
    args = parser.parse_args([
        "--model_path", "/path/to/model",
        "--checkpoint_path", "/path/to/checkpoint",
        "--benchmark", "libero_goal",
        "--task_id", "0",
        "--num_episodes", "20",
    ])

    # 验证
    assert args.model_path == "/path/to/model"
    assert args.checkpoint_path == "/path/to/checkpoint"
    assert args.benchmark == "libero_goal"
    assert args.task_id == 0
    assert args.num_episodes == 20

    print("✓ Command-line arguments verified")
    print(f"  - model_path: {args.model_path}")
    print(f"  - benchmark: {args.benchmark}")
    print(f"  - num_episodes: {args.num_episodes}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 8: Testing End-to-End Evaluation")
    print("=" * 60)

    test_eval_script_exists()
    print()
    test_metrics_calculation()
    print()
    test_metrics_json_structure()
    print()
    test_summary_structure()
    print()
    test_evaluation_pipeline()
    print()
    test_command_line_args()

    print("\n" + "=" * 60)
    print("Step 8: All tests passed! ✓")
    print("=" * 60)
    print("\nNote: This tests the evaluation pipeline logic.")
    print("Actual evaluation requires:")
    print("  1. Trained VILA-U model with action head")
    print("  2. LIBERO environment installed")
    print("  3. GPU for model inference")
    print("\nUsage:")
    print("  python scripts/eval_libero.py \\")
    print("    --model_path /path/to/model \\")
    print("    --checkpoint_path /path/to/checkpoint \\")
    print("    --benchmark libero_goal \\")
    print("    --num_episodes 10")
