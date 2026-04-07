# VILA-U 动作预测能力实施指南

## 概述

本项目为 VILA-U 添加机器人动作预测能力，使其能够：
1. 从观察图像和语言指令预测机器人动作
2. 生成 LIBERO 格式的轨迹数据
3. 在 LIBERO 环境中进行闭环评估

## 项目结构

```
vila-u-main/
├── VILA-U-Action-Prediction-Plan.md    # 详细实施方案
├── IMPLEMENTATION_STATUS.md             # 实施进度跟踪
├── README_ACTION_PREDICTION.md          # 本文件
├── tests/                               # 测试脚本
│   ├── test_step1_constants.py
│   ├── test_step2_action_head.py
│   └── test_step3_data_loader.py
├── libero_goal_reader.py                # LIBERO 数据读取器
└── vila_u/                              # 待修改的源代码
    ├── constants.py                     # Step 1: 添加常量
    ├── model/
    │   ├── configuration_vila_u.py      # Step 1: 添加配置
    │   └── vila_u_arch.py               # Step 2: 添加动作头
    └── data/
        └── libero_dataset.py            # Step 3: 数据加载器 (新建)
```

## 快速开始

### 1. 查看实施方案

```bash
# 查看完整的分步实施方案
cat VILA-U-Action-Prediction-Plan.md

# 查看当前进度
cat IMPLEMENTATION_STATUS.md
```

### 2. 实施 Step 1: 添加常量和配置

#### 2.1 修改 `vila_u/constants.py`

在文件末尾添加：

```python
# ===== Action Prediction (方案A - 无CoT) =====
ACTION_DIM = 7              # 7-DoF 动作
ACTION_CHUNK_SIZE = 10      # 每次预测的动作步数
ACTION_HORIZON = 10         # 动作预测的时间跨度
ACTION_MIN = -1.0           # 动作归一化最小值
ACTION_MAX = 1.0            # 动作归一化最大值
```

#### 2.2 修改 `vila_u/model/configuration_vila_u.py`

在 `VILAUConfig.__init__()` 中添加：

```python
# ===== Action Prediction =====
self.action_dim = kwargs.pop("action_dim", 7)
self.action_chunk_size = kwargs.pop("action_chunk_size", 10)
self.use_action_prediction = kwargs.pop("use_action_prediction", False)
```

#### 2.3 运行测试

```bash
python tests/test_step1_constants.py
```

预期输出：
```
============================================================
Step 1: Testing Constants and Configuration
============================================================
✓ All constants defined correctly
✓ Configuration fields added correctly

============================================================
Step 1: All tests passed! ✓
============================================================
```

### 3. 实施 Step 2: 添加动作预测头

详见 `VILA-U-Action-Prediction-Plan.md` 中的 Step 2 部分。

### 4. 实施 Step 3: 添加数据加载器

详见 `VILA-U-Action-Prediction-Plan.md` 中的 Step 3 部分。

## 测试策略

### 单元测试
每个步骤都有独立的测试脚本，使用模拟数据验证功能：

```bash
python tests/test_step1_constants.py    # 测试常量和配置
python tests/test_step2_action_head.py  # 测试动作预测头
python tests/test_step3_data_loader.py  # 测试数据加载器
```

### 集成测试
Step 4 之后需要实际的 LIBERO Goal 数据集：

```bash
# 下载 LIBERO Goal 数据集
cd /Users/sauvinci/repo/LIBERO-master
python benchmark_scripts/download_libero_datasets.py --datasets libero_goal

# 运行集成测试
python tests/test_step4_training.py
```

## 依赖安装

```bash
# 基础依赖
pip install torch torchvision h5py numpy pillow

# LIBERO 环境（用于 Step 6-8）
cd /Users/sauvinci/repo/LIBERO-master
pip install -e .
```

## 常见问题

### Q1: 为什么不直接使用 CoT-VLA？
A: 方案 A 是简化版本，先实现基础的动作预测能力。CoT-VLA 的视觉链式推理可以在后续添加。

### Q2: 测试脚本为什么使用模拟数据？
A: 前期测试不依赖实际数据集，可以快速验证代码逻辑。实际数据集在 Step 4 训练时才需要。

### Q3: 如何验证实施是否正确？
A: 每个步骤都有对应的测试脚本，测试通过即表示该步骤实施正确。

### Q4: 动作预测头的输出范围是什么？
A: 使用 `tanh` 激活函数，输出范围为 `[-1, 1]`，与 LIBERO 数据集的动作归一化范围一致。

## 下一步

1. **实施 Step 1-3**: 添加基础组件和测试
2. **准备数据**: 下载 LIBERO Goal 数据集
3. **实施 Step 4**: 实现训练逻辑
4. **训练模型**: 在 LIBERO 数据上训练动作预测头
5. **实施 Step 5-8**: 实现推理和评估

## 参考文档

- `VILA-U-Action-Prediction-Plan.md` - 完整实施方案
- `LIBERO_INTEGRATION_CN.md` - LIBERO 数据集集成说明
- `README_LIBERO.md` - LIBERO 数据读取器文档
- `CoT-VLA-modification-plan.md` - 原始 CoT-VLA 改造方案

## 联系方式

如有问题，请参考：
- VILA-U 项目: https://github.com/mit-han-lab/vila-u
- LIBERO 项目: https://github.com/Lifelong-Robot-Learning/LIBERO
