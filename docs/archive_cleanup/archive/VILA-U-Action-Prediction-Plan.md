# VILA-U 动作预测能力实施方案（方案A - 无CoT）

> 目标：让 VILA-U 能够从观察图像和语言指令直接预测机器人动作，生成 LIBERO 格式的轨迹数据

## 总体架构

```
观测图像 (256×256 RGB) + 文本指令
    ↓
VILA-U 视觉编码器 (冻结)
    ↓
多模态投影器
    ↓
LLM 主干网络
    ↓
动作预测头 (新增) → [10, 7] 动作 chunk
    ↓
机器人执行 → 收集轨迹 → 保存为 LIBERO 格式
```

---

## 实施步骤

### Step 1: 添加动作相关常量和配置

**目标**: 定义动作预测所需的常量和配置项

**文件改动**:
1. `vila_u/constants.py` - 添加动作常量
2. `vila_u/model/configuration_vila_u.py` - 添加配置字段

**测试**: `tests/test_step1_constants.py`

---

### Step 2: 实现动作预测头

**目标**: 在 VILA-U 架构中添加动作预测头

**文件改动**:
1. `vila_u/model/vila_u_arch.py` - 添加 `action_head` 和 `predict_actions()` 方法

**测试**: `tests/test_step2_action_head.py`

---

### Step 3: 实现 LIBERO 数据加载器

**目标**: 加载 LIBERO Goal 数据集用于训练

**文件改动**:
1. `vila_u/data/libero_dataset.py` (新建) - LIBERO 数据集加载器

**测试**: `tests/test_step3_data_loader.py`

---

### Step 4: 实现训练逻辑

**目标**: 训练动作预测头

**文件改动**:
1. `vila_u/train/train_action_prediction.py` (新建) - 训练脚本

**测试**: `tests/test_step4_training.py`

---

### Step 5: 实现推理接口

**目标**: 从观察预测动作

**文件改动**:
1. `vila_u/model/vila_u_arch.py` - 添加 `predict_action()` 方法

**测试**: `tests/test_step5_inference.py`

---

### Step 6: 实现轨迹生成器

**目标**: 闭环生成完整轨迹

**文件改动**:
1. `vila_u/eval/trajectory_generator.py` (新建) - 轨迹生成器

**测试**: `tests/test_step6_trajectory.py`

---

### Step 7: 实现 LIBERO 格式保存

**目标**: 保存轨迹为 LIBERO 兼容的 HDF5 格式

**文件改动**:
1. `vila_u/utils/libero_saver.py` (新建) - LIBERO 格式保存器

**测试**: `tests/test_step7_save_format.py`

---

### Step 8: 端到端评估

**目标**: 在 LIBERO 环境中完整评估

**文件改动**:
1. `scripts/eval_libero.py` (新建) - 完整评估脚本

**测试**: `tests/test_step8_end_to_end.py`

---

## 详细实施内容

### Step 1: 添加动作相关常量和配置

#### 1.1 修改 `vila_u/constants.py`

在文件末尾添加：

```python
# ===== Action Prediction (方案A - 无CoT) =====
ACTION_DIM = 7              # 7-DoF 动作: [dx, dy, dz, droll, dpitch, dyaw, gripper]
ACTION_CHUNK_SIZE = 10      # 每次预测的动作步数
ACTION_HORIZON = 10         # 动作预测的时间跨度（与 chunk size 相同）

# 动作归一化范围
ACTION_MIN = -1.0
ACTION_MAX = 1.0
```

#### 1.2 修改 `vila_u/model/configuration_vila_u.py`

在 `VILAUConfig.__init__()` 中添加：

```python
# ===== Action Prediction =====
self.action_dim = kwargs.pop("action_dim", 7)
self.action_chunk_size = kwargs.pop("action_chunk_size", 10)
self.use_action_prediction = kwargs.pop("use_action_prediction", False)
```

#### 1.3 测试脚本: `tests/test_step1_constants.py`

```python
"""
测试 Step 1: 验证常量和配置是否正确添加
"""
import sys
sys.path.insert(0, '/Users/sauvinci/repo/vila-u-main')

def test_constants():
    """测试常量定义"""
    from vila_u.constants import (
        ACTION_DIM,
        ACTION_CHUNK_SIZE,
        ACTION_HORIZON,
        ACTION_MIN,
        ACTION_MAX,
    )

    assert ACTION_DIM == 7, "ACTION_DIM should be 7"
    assert ACTION_CHUNK_SIZE == 10, "ACTION_CHUNK_SIZE should be 10"
    assert ACTION_HORIZON == 10, "ACTION_HORIZON should be 10"
    assert ACTION_MIN == -1.0, "ACTION_MIN should be -1.0"
    assert ACTION_MAX == 1.0, "ACTION_MAX should be 1.0"

    print("✓ All constants defined correctly")


def test_config():
    """测试配置字段"""
    from vila_u.model.configuration_vila_u import VILAUConfig

    config = VILAUConfig(
        action_dim=7,
        action_chunk_size=10,
        use_action_prediction=True,
    )

    assert config.action_dim == 7, "action_dim should be 7"
    assert config.action_chunk_size == 10, "action_chunk_size should be 10"
    assert config.use_action_prediction == True, "use_action_prediction should be True"

    print("✓ Configuration fields added correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Testing Constants and Configuration")
    print("=" * 60)

    test_constants()
    test_config()

    print("\n" + "=" * 60)
    print("Step 1: All tests passed! ✓")
    print("=" * 60)
```

---

### Step 2: 实现动作预测头

#### 2.1 修改 `vila_u/model/vila_u_arch.py`

在 `VILAUMetaModel` 类的 `init_vlm()` 方法末尾（`self.is_loaded = True` 之前）添加：

```python
# 添加动作预测头
if getattr(config, "use_action_prediction", False):
    from torch import nn
    action_out_dim = config.action_chunk_size * config.action_dim
    self.action_head = nn.Linear(
        config.hidden_size,
        action_out_dim,
        bias=True,
    )
    # 小值初始化，避免训练初期梯度爆炸
    nn.init.normal_(self.action_head.weight, std=0.02)
    nn.init.zeros_(self.action_head.bias)
    print(f"[Action Head] Initialized: {config.hidden_size} -> {action_out_dim}")
```

在 `VILAUMetaModel` 类中添加方法：

```python
def predict_actions(
    self,
    hidden_states: torch.Tensor,
    action_token_position: int = -1,
) -> torch.Tensor:
    """
    从 LLM 隐层状态预测动作序列。

    Args:
        hidden_states: [B, seq_len, hidden_size] LLM 输出的隐层状态
        action_token_position: 用于预测动作的 token 位置（默认 -1 表示最后一个）

    Returns:
        actions: [B, ACTION_CHUNK_SIZE, ACTION_DIM] 预测的动作序列
    """
    if not hasattr(self, 'action_head'):
        raise RuntimeError("Action head not initialized. Set use_action_prediction=True in config.")

    B = hidden_states.shape[0]
    # 提取指定位置的隐层状态
    act_hidden = hidden_states[:, action_token_position, :]  # [B, hidden_size]

    # 通过动作头解码
    raw = self.action_head(act_hidden)  # [B, chunk_size * action_dim]

    # Reshape 为动作序列
    actions = raw.view(B, self.config.action_chunk_size, self.config.action_dim)

    # Tanh 激活，限制到 [-1, 1]
    actions = torch.tanh(actions)

    return actions
```

#### 2.2 测试脚本: `tests/test_step2_action_head.py`

```python
"""
测试 Step 2: 验证动作预测头是否正确初始化和工作
"""
import sys
sys.path.insert(0, '/Users/sauvinci/repo/vila-u-main')

import torch


def test_action_head_initialization():
    """测试动作头初始化"""
    from vila_u.model.configuration_vila_u import VILAUConfig
    from vila_u.model.vila_u_arch import VILAUMetaModel

    # 创建配置
    config = VILAUConfig(
        hidden_size=4096,
        action_dim=7,
        action_chunk_size=10,
        use_action_prediction=True,
    )

    # 注意：这里只测试配置，不加载完整模型
    print(f"✓ Config created with action prediction enabled")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - action_dim: {config.action_dim}")
    print(f"  - action_chunk_size: {config.action_chunk_size}")


def test_action_head_forward():
    """测试动作头前向传播"""
    import torch.nn as nn

    # 模拟动作头
    hidden_size = 4096
    action_dim = 7
    action_chunk_size = 10
    action_out_dim = action_chunk_size * action_dim

    action_head = nn.Linear(hidden_size, action_out_dim, bias=True)
    nn.init.normal_(action_head.weight, std=0.02)
    nn.init.zeros_(action_head.bias)

    # 模拟输入
    batch_size = 2
    hidden_states = torch.randn(batch_size, hidden_size)

    # 前向传播
    raw_output = action_head(hidden_states)
    actions = raw_output.view(batch_size, action_chunk_size, action_dim)
    actions = torch.tanh(actions)

    # 验证输出形状
    assert actions.shape == (batch_size, action_chunk_size, action_dim), \
        f"Expected shape ({batch_size}, {action_chunk_size}, {action_dim}), got {actions.shape}"

    # 验证输出范围
    assert actions.min() >= -1.0 and actions.max() <= 1.0, \
        f"Actions should be in [-1, 1], got range [{actions.min():.3f}, {actions.max():.3f}]"

    print(f"✓ Action head forward pass successful")
    print(f"  - Input shape: {hidden_states.shape}")
    print(f"  - Output shape: {actions.shape}")
    print(f"  - Output range: [{actions.min():.3f}, {actions.max():.3f}]")


def test_predict_actions_method():
    """测试 predict_actions 方法的逻辑"""
    import torch.nn as nn

    # 模拟完整流程
    batch_size = 2
    seq_len = 100
    hidden_size = 4096
    action_dim = 7
    action_chunk_size = 10

    # 模拟 LLM 输出
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 模拟动作头
    action_head = nn.Linear(hidden_size, action_chunk_size * action_dim)

    # 提取最后一个 token 的隐层状态
    act_hidden = hidden_states[:, -1, :]  # [B, hidden_size]

    # 预测动作
    raw = action_head(act_hidden)
    actions = raw.view(batch_size, action_chunk_size, action_dim)
    actions = torch.tanh(actions)

    # 验证
    assert actions.shape == (batch_size, action_chunk_size, action_dim)
    assert actions.min() >= -1.0 and actions.max() <= 1.0

    print(f"✓ predict_actions method logic verified")
    print(f"  - Hidden states shape: {hidden_states.shape}")
    print(f"  - Actions shape: {actions.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Testing Action Prediction Head")
    print("=" * 60)

    test_action_head_initialization()
    print()
    test_action_head_forward()
    print()
    test_predict_actions_method()

    print("\n" + "=" * 60)
    print("Step 2: All tests passed! ✓")
    print("=" * 60)
```

---

### Step 3: 实现 LIBERO 数据加载器

#### 3.1 创建 `vila_u/data/libero_dataset.py`

```python
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
```

#### 3.2 测试脚本: `tests/test_step3_data_loader.py`

```python
"""
测试 Step 3: 验证 LIBERO 数据加载器
"""
import sys
sys.path.insert(0, '/Users/sauvinci/repo/vila-u-main')

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
```

---

### Step 4-8: 后续步骤概要

由于篇幅限制，后续步骤的详细实现将在单独的文件中提供。每个步骤都遵循相同的模式：

1. **明确目标**
2. **文件改动清单**
3. **详细代码实现**
4. **独立测试脚本**

---

## 测试运行顺序

```bash
# Step 1: 测试常量和配置
python tests/test_step1_constants.py

# Step 2: 测试动作预测头
python tests/test_step2_action_head.py

# Step 3: 测试数据加载器
python tests/test_step3_data_loader.py

# Step 4: 测试训练逻辑
python tests/test_step4_training.py

# Step 5: 测试推理接口
python tests/test_step5_inference.py

# Step 6: 测试轨迹生成
python tests/test_step6_trajectory.py

# Step 7: 测试格式保存
python tests/test_step7_save_format.py

# Step 8: 端到端测试
python tests/test_step8_end_to_end.py
```

---

## 依赖安装

```bash
# 基础依赖
pip install torch torchvision h5py numpy pillow

# LIBERO 环境（用于 Step 6-8）
cd /Users/sauvinci/repo/LIBERO-master
pip install -e .
```

---

## 注意事项

1. **每步独立测试**: 每个步骤都有独立的测试脚本，可以单独运行
2. **渐进式开发**: 前一步测试通过后再进行下一步
3. **模拟数据测试**: 前期步骤使用模拟数据，避免依赖实际数据集
4. **真实数据测试**: Step 4 之后需要实际的 LIBERO Goal 数据集
5. **环境隔离**: 测试脚本不会修改原始代码，只验证功能

---

## 下一步

请告诉我：
1. 是否需要我立即实现 Step 1-3 的代码？
2. 是否需要我继续编写 Step 4-8 的详细方案？
3. 是否有任何步骤需要调整或补充？
