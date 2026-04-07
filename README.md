# VILA-U Action Prediction

为 VILA-U 视觉语言模型添加机器人动作预测能力，支持在 LIBERO Goal benchmark 上训练和评估。

## 🚀 快速开始

```bash
# 1. 测试模型加载
python tests/test_model_loading.py

# 2. 准备 LIBERO Goal 数据集
cd /path/to/LIBERO-master
python benchmark_scripts/download_libero_datasets.py --datasets libero_goal --use-huggingface

# 3. 开始训练
python -m vila_u.train.train_action_prediction \
    --model_path /path/to/vila-u-7b-256 \
    --data_root /path/to/libero_goal \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --num_epochs 10
```

详细说明请查看 [QUICK_START.md](QUICK_START.md) 和 [READY_TO_TRAIN.md](READY_TO_TRAIN.md)

## 📚 文档

### 核心文档
- [QUICK_START.md](QUICK_START.md) - 快速开始指南
- [READY_TO_TRAIN.md](READY_TO_TRAIN.md) - 训练就绪指南
- [docs/](docs/) - 完整文档目录

### 技术文档
- [动作生成方法对比](docs/technical/ACTION_GENERATION_COMPARISON.md) - 与 OpenVLA、RT-2、Octo 等项目对比
- [自回归 vs 直接回归](docs/technical/AUTOREGRESSIVE_VS_DIRECT_REGRESSION.md) - 深入技术解析

### 项目报告
- [项目修改总结](docs/reports/PROJECT_MODIFICATIONS_SUMMARY.md) - 完整的代码修改说明
- [最终报告](docs/reports/FINAL_REPORT.md) - 项目完整报告

## 🎯 核心功能

### 动作预测架构
```
输入: RGB图像 [3, 256, 256] + 语言指令
  ↓
VILA-U 视觉语言编码器
  ↓
动作预测头 (Linear + Tanh)
  ↓
输出: 动作序列 [10, 7] (10步预测，7-DoF)
```

### 主要特点
- ✅ 连续值直接回归（快速推理，~50ms）
- ✅ Action Chunking（一次预测10步）
- ✅ 完整的训练和评估流程
- ✅ LIBERO Goal 数据集集成
- ✅ 轨迹生成和保存

## 📊 代码结构

```
vila_u/
├── model/
│   ├── vila_u_arch.py          # 动作预测头和推理接口
│   └── configuration_vila_u.py  # 模型配置
├── data/
│   └── libero_dataset.py        # LIBERO 数据加载器
├── train/
│   └── train_action_prediction.py  # 训练逻辑
├── eval/
│   └── trajectory_generator.py  # 轨迹生成器
└── utils/
    └── libero_saver.py          # 轨迹保存

scripts/
└── eval_libero.py               # 评估脚本

tests/
├── test_model_loading.py        # 模型加载测试
└── test_step*.py                # 各步骤测试
```

## 🔧 核心修改

### 1. 模型架构 (vila_u_arch.py)
- 添加动作预测头初始化
- 实现 `predict_actions()` 方法
- 实现 `predict_action()` 推理接口

### 2. 训练逻辑 (train_action_prediction.py)
- 完整的训练循环
- L1 损失计算
- 检查点保存和 WandB 集成

### 3. 数据加载 (libero_dataset.py)
- LIBERO Goal HDF5 数据集加载
- 图像预处理和动作提取

### 4. 评估工具
- 轨迹生成器（闭环执行）
- LIBERO 格式保存
- 成功率统计

详细修改说明请查看 [PROJECT_MODIFICATIONS_SUMMARY.md](docs/reports/PROJECT_MODIFICATIONS_SUMMARY.md)

## 📈 性能特点

| 特性 | VILA-U (当前) | OpenVLA | RT-2 | Octo |
|------|--------------|---------|------|------|
| 推理速度 | ~50ms | ~700ms | ~1500ms | ~2000ms |
| 动作表示 | 连续值 | 离散token | 离散token | 连续值 |
| 生成方式 | 直接回归 | 自回归 | 自回归 | Diffusion |
| Action Chunk | 10步 | 1步 | 1步 | 4-16步 |

## 🧪 测试

```bash
# 运行所有测试
./run_tests.sh

# 或单独运行
python tests/test_model_loading.py
python tests/test_step1_constants.py
python tests/test_step2_action_head.py
# ...
```

## 📝 引用

如果使用本项目，请引用：

```bibtex
@software{vila_u_action_prediction,
  title={VILA-U Action Prediction},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo/vila-u-action-prediction}
}
```

## 📄 许可证

本项目基于 VILA-U 开发，遵循相应的开源许可证。

## 🙏 致谢

- VILA-U 团队提供的预训练模型
- LIBERO 团队提供的 benchmark
- OpenVLA、RT-2、Octo 等项目的启发
