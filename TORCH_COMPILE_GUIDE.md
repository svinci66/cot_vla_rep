# torch.compile 使用指南

## 概述

torch.compile 是 PyTorch 2.0+ 的新特性，可以显著加速训练（预期提升 20-40%）。

## 要求

- PyTorch >= 2.0
- CUDA 支持

## 使用方法

### 方法1：启用 torch.compile（推荐）

```bash
# 使用默认模式
USE_TORCH_COMPILE=True bash scripts/train/train_action_prediction.sh
```

### 方法2：指定编译模式

torch.compile 支持不同的编译模式：

```bash
# default 模式（平衡速度和编译时间）
USE_TORCH_COMPILE=True TORCH_COMPILE_MODE=default bash scripts/train/train_action_prediction.sh

# reduce-overhead 模式（最大化速度，编译时间更长）
USE_TORCH_COMPILE=True TORCH_COMPILE_MODE=reduce-overhead bash scripts/train/train_action_prediction.sh

# max-autotune 模式（最激进优化，编译时间最长）
USE_TORCH_COMPILE=True TORCH_COMPILE_MODE=max-autotune bash scripts/train/train_action_prediction.sh
```

## 编译模式对比

| 模式 | 编译时间 | 运行速度 | 推荐场景 |
|------|---------|---------|---------|
| `default` | 中等 | 快 | 一般训练（推荐） |
| `reduce-overhead` | 长 | 更快 | 长时间训练 |
| `max-autotune` | 很长 | 最快 | 生产环境 |

## 预期效果

### 首次运行
- **编译时间**：5-15分钟（取决于模式）
- 第一个 epoch 会比较慢
- 编译完成后会缓存，后续运行直接使用

### 后续运行
- **加速效果**：20-40%
- 前向传播：~780ms → ~470-620ms
- 总训练时间：~2h/epoch → ~1.2-1.6h/epoch

## 注意事项

### 1. 首次编译很慢
```
================================================================================
Compiling model with torch.compile...
This will take a few minutes on first run but speeds up training significantly.
================================================================================
```
**不要中断！** 编译完成后会自动开始训练。

### 2. 显存占用
torch.compile 可能会增加显存占用（约 10-20%）。如果遇到 OOM：

```bash
# 减小 batch size
BATCH_SIZE=16 USE_TORCH_COMPILE=True bash scripts/train/train_action_prediction.sh
```

### 3. 兼容性
某些自定义操作可能不兼容 torch.compile。如果遇到错误：

```bash
# 禁用 torch.compile
USE_TORCH_COMPILE=False bash scripts/train/train_action_prediction.sh
```

### 4. 调试模式
如果需要调试，建议关闭 torch.compile：

```bash
# 调试时关闭
USE_TORCH_COMPILE=False bash scripts/train/train_action_prediction.sh
```

## 检查 PyTorch 版本

在服务器上检查是否支持 torch.compile：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Has compile: {hasattr(torch, \"compile\")}')"
```

输出示例：
```
PyTorch: 2.1.0
Has compile: True  ← 支持
```

如果显示 `Has compile: False`，说明 PyTorch 版本 < 2.0，需要升级：

```bash
# 升级 PyTorch（根据你的 CUDA 版本）
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 性能对比

### 不使用 torch.compile
```
Forward: 782ms
Backward: ~600ms
Optimizer: ~100ms
Total: ~1482ms/step
Throughput: 0.67 steps/sec
```

### 使用 torch.compile (default)
```
Forward: ~470ms (↓40%)
Backward: ~450ms (↓25%)
Optimizer: ~100ms
Total: ~1020ms/step
Throughput: 0.98 steps/sec
```

### 使用 torch.compile (reduce-overhead)
```
Forward: ~390ms (↓50%)
Backward: ~420ms (↓30%)
Optimizer: ~100ms
Total: ~910ms/step
Throughput: 1.10 steps/sec
```

## 故障排查

### 问题1：编译失败

```
⚠ Warning: torch.compile failed: ...
Continuing without compilation...
```

**原因**：可能是自定义操作不兼容

**解决**：训练会继续，但不使用编译加速。检查错误信息，可能需要修改代码。

### 问题2：编译时间过长

**正常现象**。首次编译需要 5-15 分钟。可以：
- 使用 `default` 模式（更快）
- 耐心等待（只需编译一次）

### 问题3：OOM

```
torch.OutOfMemoryError: CUDA out of memory
```

**解决**：
```bash
# 减小 batch size
BATCH_SIZE=16 USE_TORCH_COMPILE=True bash scripts/train/train_action_prediction.sh

# 或禁用 torch.compile
USE_TORCH_COMPILE=False bash scripts/train/train_action_prediction.sh
```

## 推荐配置

### 快速测试（首次尝试）
```bash
USE_TORCH_COMPILE=True TORCH_COMPILE_MODE=default bash scripts/train/train_action_prediction.sh
```

### 生产训练（长时间训练）
```bash
USE_TORCH_COMPILE=True TORCH_COMPILE_MODE=reduce-overhead bash scripts/train/train_action_prediction.sh
```

### 最大性能（有充足时间）
```bash
USE_TORCH_COMPILE=True TORCH_COMPILE_MODE=max-autotune bash scripts/train/train_action_prediction.sh
```

## 总结

✅ **推荐使用 torch.compile**，特别是：
- 长时间训练（多个 epoch）
- PyTorch >= 2.0
- 有充足的显存

⚠️ **不推荐使用**，如果：
- 快速调试
- 显存紧张
- PyTorch < 2.0
