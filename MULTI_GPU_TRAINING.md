# 8卡训练配置指南

## 概述

训练脚本已支持多卡训练。使用8卡可以显著提升训练速度。

## 使用方法

### 方法1：直接设置环境变量（推荐）

```bash
# 8卡训练，每卡batch=20
SINGLE_GPU_MODE=False \
NUM_GPUS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=160 \
bash scripts/train/train_action_prediction.sh
```

### 方法2：修改脚本默认值

编辑 `scripts/train/train_action_prediction.sh`：

```bash
# 修改这些行
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-False}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-160}  # 8卡 × 20/卡
```

然后直接运行：
```bash
bash scripts/train/train_action_prediction.sh
```

## Batch Size 计算

### 公式
```
BATCH_SIZE = NUM_GPUS × per_gpu_batch × ACC_STEP
```

### 推荐配置

#### 配置1：保持单卡batch size（推荐）
```bash
NUM_GPUS=8
per_gpu_batch=20
ACC_STEP=2
BATCH_SIZE=320  # 8 × 20 × 2
```

**优势**：
- 每卡显存占用与单卡相同
- 有效batch size = 320（非常大）
- 训练速度：~8倍提升

#### 配置2：减小per_gpu_batch（更安全）
```bash
NUM_GPUS=8
per_gpu_batch=16
ACC_STEP=2
BATCH_SIZE=256  # 8 × 16 × 2
```

**优势**：
- 显存占用更小
- 更不容易OOM
- 训练速度：~8倍提升

#### 配置3：不使用梯度累积
```bash
NUM_GPUS=8
per_gpu_batch=20
ACC_STEP=1
BATCH_SIZE=160  # 8 × 20 × 1
```

**优势**：
- 最简单的配置
- 有效batch size = 160
- 训练速度：~8倍提升

## 完整的8卡训练命令

### 基础8卡训练
```bash
SINGLE_GPU_MODE=False \
NUM_GPUS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=160 \
ACC_STEP=1 \
bash scripts/train/train_action_prediction.sh
```

### 8卡 + torch.compile
```bash
SINGLE_GPU_MODE=False \
NUM_GPUS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=160 \
ACC_STEP=1 \
USE_TORCH_COMPILE=True \
bash scripts/train/train_action_prediction.sh
```

### 8卡 + DeepSpeed ZeRO-2（节省显存）
```bash
SINGLE_GPU_MODE=False \
NUM_GPUS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=160 \
ACC_STEP=1 \
USE_DEEPSPEED=True \
bash scripts/train/train_action_prediction.sh
```

## 性能预期

### 单卡训练
- 时间：~2小时/epoch
- 吞吐量：0.67 steps/sec
- 有效batch size：40

### 8卡训练（理想情况）
- 时间：**~15分钟/epoch**（8倍加速）
- 吞吐量：5.36 steps/sec
- 有效batch size：160

### 8卡训练（实际情况，考虑通信开销）
- 时间：**~18-20分钟/epoch**（6-7倍加速）
- 吞吐量：4.5-5.0 steps/sec
- 有效batch size：160

## 注意事项

### 1. Batch Size 必须能整除
```
BATCH_SIZE 必须能被 (NUM_GPUS × ACC_STEP) 整除
```

**正确示例**：
- BATCH_SIZE=160, NUM_GPUS=8, ACC_STEP=1 ✓ (160 / 8 / 1 = 20)
- BATCH_SIZE=320, NUM_GPUS=8, ACC_STEP=2 ✓ (320 / 8 / 2 = 20)

**错误示例**：
- BATCH_SIZE=100, NUM_GPUS=8, ACC_STEP=1 ✗ (100 / 8 = 12.5)

### 2. 学习率调整

使用更大的batch size时，可能需要调整学习率：

```bash
# 线性缩放规则：batch size增加N倍，学习率也增加N倍
# 单卡：batch=40, lr=1e-5
# 8卡：batch=160 (4倍), lr=4e-5

LEARNING_RATE=4e-5 \
BATCH_SIZE=160 \
NUM_GPUS=8 \
bash scripts/train/train_action_prediction.sh
```

**或者使用warmup**（推荐）：
```bash
# 保持学习率不变，增加warmup
LEARNING_RATE=1e-5 \
WARMUP_RATIO=0.1 \
BATCH_SIZE=160 \
NUM_GPUS=8 \
bash scripts/train/train_action_prediction.sh
```

### 3. DataLoader Workers

8卡训练时，每个GPU都有独立的DataLoader：

```bash
# 总workers = NUM_GPUS × DATALOADER_NUM_WORKERS
# 8卡 × 4 workers = 32个进程

# 如果CPU核心不够，减少workers
DATALOADER_NUM_WORKERS=2 \
NUM_GPUS=8 \
bash scripts/train/train_action_prediction.sh
```

### 4. 显存占用

每张卡的显存占用与单卡训练相同（假设per_gpu_batch相同）。

**如果遇到OOM**：
```bash
# 减小per_gpu_batch
BATCH_SIZE=128 \  # 8 × 16 × 1
NUM_GPUS=8 \
bash scripts/train/train_action_prediction.sh

# 或启用gradient checkpointing
GRADIENT_CHECKPOINTING=True \
BATCH_SIZE=160 \
NUM_GPUS=8 \
bash scripts/train/train_action_prediction.sh
```

## 监控和调试

### 检查GPU使用情况
```bash
# 在训练时，另开一个终端
watch -n 1 nvidia-smi
```

应该看到所有8张卡都在使用。

### 检查训练日志
```bash
# 查看是否正确使用8卡
# 日志中应该显示：
# GPUs per Node: 8
# Single GPU Mode: False
```

### 性能分析
```bash
# 启用性能监控
ENABLE_PROFILING=True \
NUM_GPUS=8 \
BATCH_SIZE=160 \
bash scripts/train/train_action_prediction.sh
```

## 故障排查

### 问题1：NCCL错误
```
NCCL error: unhandled system error
```

**解决**：
```bash
# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果没有InfiniBand
export NCCL_P2P_DISABLE=1  # 如果P2P通信有问题
```

### 问题2：进程卡住
```
Waiting for all processes to finish...
```

**解决**：
```bash
# 检查是否有进程残留
ps aux | grep python | grep train

# 杀掉残留进程
pkill -f train_action_prediction

# 重新启动
```

### 问题3：速度没有提升

**可能原因**：
- 数据加载成为瓶颈 → 增加workers
- 通信开销过大 → 使用DeepSpeed
- Batch size太小 → 增大batch size

## 推荐配置总结

### 快速开始（最简单）
```bash
SINGLE_GPU_MODE=False \
NUM_GPUS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=160 \
bash scripts/train/train_action_prediction.sh
```

### 最佳性能（推荐）
```bash
SINGLE_GPU_MODE=False \
NUM_GPUS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=160 \
DATALOADER_NUM_WORKERS=2 \
USE_TORCH_COMPILE=True \
bash scripts/train/train_action_prediction.sh
```

### 显存优化（如果OOM）
```bash
SINGLE_GPU_MODE=False \
NUM_GPUS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=128 \
GRADIENT_CHECKPOINTING=True \
USE_DEEPSPEED=True \
bash scripts/train/train_action_prediction.sh
```

## 预期训练时间

假设数据集有 58,598 samples，batch_size=160：

- **Steps per epoch**: 58598 / 160 ≈ 366 steps
- **单卡时间**: 366 steps × 1.5s/step ≈ 9分钟/epoch
- **8卡时间**: 366 steps × 0.2s/step ≈ **1.2分钟/epoch**

**完整训练（10 epochs）**：
- 单卡：~90分钟
- 8卡：**~12分钟**

太快了！建议增加训练数据或epochs。
