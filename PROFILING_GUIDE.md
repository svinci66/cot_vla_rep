# Performance Profiling Guide

## 概述

我们添加了详细的性能分析工具来定位训练瓶颈。有两种使用方式：

---

## 方式1：独立性能分析脚本（推荐用于首次分析）

### 使用方法

```bash
# 完整分析（推荐）
python scripts/profile_training.py \
    --model_path /path/to/vila-u-7b-256 \
    --data_root /path/to/libero_goal \
    --profile_all

# 只分析数据加载
python scripts/profile_training.py \
    --model_path /path/to/vila-u-7b-256 \
    --data_root /path/to/libero_goal \
    --profile_data

# 只分析模型前向传播
python scripts/profile_training.py \
    --model_path /path/to/vila-u-7b-256 \
    --data_root /path/to/libero_goal \
    --profile_forward

# 只分析完整训练步骤
python scripts/profile_training.py \
    --model_path /path/to/vila-u-7b-256 \
    --data_root /path/to/libero_goal \
    --profile_training

# 只分析注意力mask构建
python scripts/profile_training.py \
    --model_path /path/to/vila-u-7b-256 \
    --data_root /path/to/libero_goal \
    --profile_attention
```

### 输出示例

```
================================================================================
PROFILING DATA LOADING
================================================================================
Dataset size: 12450 samples

--- Testing with 0 workers ---
  Average batch time: 145.23ms
  Throughput: 137.7 samples/sec

--- Testing with 4 workers ---
  Average batch time: 52.18ms
  Throughput: 383.3 samples/sec

--- Testing with 16 workers ---
  Average batch time: 28.45ms
  Throughput: 703.2 samples/sec

================================================================================
Training Performance Profile
================================================================================
Operation                      Count   Total(s)   Mean(ms)        %
--------------------------------------------------------------------------------
single_sample_load                 5      0.725     145.00    100.0%
--------------------------------------------------------------------------------
TOTAL                                     0.725                100.0%
================================================================================

================================================================================
PROFILING HYBRID ATTENTION MASK CONSTRUCTION
================================================================================
Profiling 100 mask constructions...
  Batch size: 20
  Sequence length: 512
  Action tokens: 70

Throughput: 2341.5 masks/sec
Per-batch overhead: 0.43ms

================================================================================
PROFILING MODEL FORWARD PASS
================================================================================
Profiling 10 forward passes...

Operation                      Count   Total(s)   Mean(ms)        %
--------------------------------------------------------------------------------
2_model_forward                   10      3.245     324.50    100.0%
1_prepare_inputs                  10      0.000       0.00      0.0%
--------------------------------------------------------------------------------
TOTAL                                     3.245                100.0%
================================================================================

================================================================================
PROFILING TRAINING STEP (Forward + Backward + Optimizer)
================================================================================
Profiling 10 training steps...

Operation                      Count   Total(s)   Mean(ms)        %
--------------------------------------------------------------------------------
1_forward                         10      3.521     352.10     45.2%
2_backward                        10      3.012     301.20     38.7%
3_optimizer_step                  10      1.123     112.30     14.4%
4_zero_grad                       10      0.132      13.20      1.7%
--------------------------------------------------------------------------------
TOTAL                                     7.788                100.0%
================================================================================
```

---

## 方式2：集成到训练中（用于持续监控）

### 使用方法

在训练脚本中启用性能分析：

```bash
# 启用性能分析（每50步输出一次）
ENABLE_PROFILING=True bash scripts/train/train_action_prediction.sh
```

### 输出示例

训练过程中每50步会输出：

```
================================================================================
Performance Summary (steps 1-50)
================================================================================
Average step time: 456.23ms
Throughput: 2.19 steps/sec

Component                      Avg(ms)        %
--------------------------------------------------------------------------------
training_step                   456.23    100.0%
================================================================================

================================================================================
Performance Summary (steps 51-100)
================================================================================
Average step time: 442.15ms
Throughput: 2.26 steps/sec

Component                      Avg(ms)        %
--------------------------------------------------------------------------------
training_step                   442.15    100.0%
================================================================================
```

---

## 如何解读结果

### 1. 数据加载分析

**关注指标**：
- `Average batch time`: 每个batch的加载时间
- `Throughput`: 每秒处理的样本数

**优化建议**：
- 如果 `num_workers=0` 很慢，增加workers
- 如果 `num_workers=16` 比 `num_workers=8` 慢，说明workers过多
- 理想情况：数据加载时间 < 模型前向时间

### 2. 模型前向传播分析

**关注指标**：
- `model_forward`: 总前向时间
- 如果有细分，查看哪个部分最慢

**常见瓶颈**：
- 视觉编码器（rqvaesiglip + rqtransformer）
- LLM前向传播
- 注意力计算

### 3. 训练步骤分析

**关注指标**：
- `forward`: 前向传播时间（应占 40-50%）
- `backward`: 反向传播时间（应占 30-40%）
- `optimizer_step`: 优化器更新（应占 10-20%）

**异常情况**：
- 如果 `backward` > `forward`：可能有梯度计算问题
- 如果 `optimizer_step` > 20%：可能需要优化优化器配置
- 如果 `zero_grad` > 5%：异常，需要检查

### 4. 注意力mask分析

**关注指标**：
- `Per-batch overhead`: 每个batch构建mask的时间
- 理想值：< 1ms

**如果 > 5ms**：
- 说明mask构建是瓶颈
- 需要进一步优化向量化

---

## 常见性能瓶颈及解决方案

### 瓶颈1：数据加载慢

**症状**：
- `Average batch time` > 100ms
- GPU利用率低（< 80%）

**解决方案**：
```bash
# 增加workers
DATALOADER_NUM_WORKERS=16 bash scripts/train/train_action_prediction.sh

# 启用预取
DATALOADER_PREFETCH_FACTOR=4 bash scripts/train/train_action_prediction.sh

# 预处理数据集（一次性工作）
python scripts/preprocess_libero_data.py --data_root /path/to/data --output_dir /path/to/preprocessed
```

### 瓶颈2：前向传播慢

**症状**：
- `forward` 时间 > 500ms
- 训练速度慢

**解决方案**：
```bash
# 启用gradient checkpointing（节省显存，略微增加计算）
GRADIENT_CHECKPOINTING=True bash scripts/train/train_action_prediction.sh

# 减小batch size
BATCH_SIZE=16 bash scripts/train/train_action_prediction.sh

# 使用混合精度（已启用bf16）
# 检查是否正确使用
```

### 瓶颈3：反向传播慢

**症状**：
- `backward` 时间 > `forward` 时间
- 显存占用高

**解决方案**：
```bash
# 启用gradient checkpointing
GRADIENT_CHECKPOINTING=True bash scripts/train/train_action_prediction.sh

# 减小batch size
BATCH_SIZE=16 ACC_STEP=3 bash scripts/train/train_action_prediction.sh
```

### 瓶颈4：优化器更新慢

**症状**：
- `optimizer_step` > 20% 总时间

**解决方案**：
- 检查是否有不必要的参数在优化
- 考虑使用更高效的优化器（如 AdamW with fused=True）

---

## 性能目标

基于当前优化，目标性能指标：

| 指标 | 目标值 | 当前值 |
|------|--------|--------|
| 数据加载时间 | < 30ms/batch | ~28ms (16 workers) |
| 前向传播时间 | 300-400ms | 待测 |
| 反向传播时间 | 250-350ms | 待测 |
| 优化器更新 | < 100ms | 待测 |
| **总训练步骤** | **< 800ms** | **待测** |
| **Throughput** | **> 1.25 steps/sec** | **待测** |

---

## 下一步行动

1. **运行完整性能分析**：
   ```bash
   python scripts/profile_training.py \
       --model_path /path/to/model \
       --data_root /path/to/data \
       --profile_all
   ```

2. **分析结果**：
   - 找出占用时间 > 10% 的操作
   - 对比目标值

3. **针对性优化**：
   - 根据瓶颈选择对应的优化方案
   - 重新测试验证效果

4. **启动训练**：
   ```bash
   # 启用性能监控
   ENABLE_PROFILING=True bash scripts/train/train_action_prediction.sh
   ```

5. **持续监控**：
   - 观察每50步的性能报告
   - 确保训练速度稳定
   - 记录最终的训练时间

---

## 故障排查

### 问题：profiling脚本报错

**解决**：
```bash
# 确保所有依赖已安装
pip install torch transformers h5py pillow

# 检查路径是否正确
ls /path/to/model
ls /path/to/data
```

### 问题：性能分析输出为空

**解决**：
- 检查是否有CUDA
- 确保模型和数据可以正常加载
- 查看是否有错误信息

### 问题：训练中看不到性能报告

**解决**：
```bash
# 确保启用了profiling
export ENABLE_PROFILING=True
bash scripts/train/train_action_prediction.sh

# 或者直接在命令中设置
ENABLE_PROFILING=True bash scripts/train/train_action_prediction.sh
```

---

## 参考

- `vila_u/utils/profiling.py` - 详细的性能分析工具
- `vila_u/utils/quick_profiling.py` - 轻量级性能监控
- `scripts/profile_training.py` - 独立性能分析脚本
