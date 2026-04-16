# 训练速度优化更改说明

## 📊 优化目标
- **当前速度**: ~3小时/epoch
- **优化目标**: 1.5-2小时/epoch（提升40-50%）

## ✅ 已完成的优化（立即生效）

### 1. 增大Batch Size和梯度累积
**修改文件**:
- `scripts/train/train_action_prediction.sh`
- `scripts/train/train_action_prediction_l40.sh`

**更改内容**:
```bash
# 之前
BATCH_SIZE=16 (标准脚本) / 8 (L40脚本)
ACC_STEP=1
有效batch = 16 / 8

# 现在
BATCH_SIZE=32
ACC_STEP=2
有效batch = 64
```

**预期提升**: 15-20%
**原理**: 更大的batch size提高GPU利用率，减少梯度更新次数

---

### 2. 增加DataLoader Workers
**更改内容**:
```bash
# 之前
DATALOADER_NUM_WORKERS=8 (标准) / 4 (L40)

# 现在
DATALOADER_NUM_WORKERS=16
```

**预期提升**: 10-15%
**原理**: 更多workers并行加载数据，减少GPU等待时间

---

## 🎯 使用方法

### 直接运行（使用新的默认配置）
```bash
bash scripts/train/train_action_prediction.sh
# 或
bash scripts/train/train_action_prediction_l40.sh
```

### 自定义配置（如果显存不够）
```bash
# 如果遇到OOM（显存不足），可以降低batch size
BATCH_SIZE=16 ACC_STEP=4 bash scripts/train/train_action_prediction.sh
# 有效batch仍然是64，但每个GPU batch更小

# 或者进一步降低
BATCH_SIZE=8 ACC_STEP=8 bash scripts/train/train_action_prediction.sh
```

---

## 📈 预期效果

### 优化前
- Batch size: 16 (标准) / 8 (L40)
- Workers: 8 / 4
- 速度: ~3小时/epoch

### 优化后
- Batch size: 32 (有效64)
- Workers: 16
- **预期速度: 1.8-2.2小时/epoch**

---

## 🔍 监控指标

训练时注意观察：
1. **GPU利用率**: 应该接近100%
2. **DataLoader时间**: 在日志中查看数据加载是否成为瓶颈
3. **显存使用**: 确保不会OOM

```bash
# 监控GPU使用
watch -n 1 nvidia-smi
```

---

## 🚀 后续优化计划

### 短期（需要安装依赖）
1. **启用Flash Attention 2**
   - 需要: `pip install flash-attn --no-build-isolation`
   - 修改: `ATTN_IMPLEMENTATION=flash_attention_2`
   - 预期提升: 20-30%

### 中期（需要开发）
2. **缓存冻结的视觉特征**
   - 预先计算rqvaesiglip的输出
   - 预期提升: 30-40%
   - 需要编写预计算脚本

---

## ⚠️ 注意事项

1. **显存监控**: 如果遇到OOM，降低BATCH_SIZE但保持有效batch不变
2. **学习率**: 更大的batch可能需要调整学习率（当前保持1e-5）
3. **收敛性**: 观察loss曲线，确保优化后收敛正常

---

## 📝 回滚方法

如果遇到问题，可以恢复原配置：
```bash
BATCH_SIZE=16 ACC_STEP=1 DATALOADER_NUM_WORKERS=8 bash scripts/train/train_action_prediction.sh
```
