# VILA-U Action Prediction Training - 重构版本

## 🎯 重构内容

基于 VILA-U 的 `scripts/train/sft.sh` 重构了训练代码，主要改进：

### 1. 训练框架对齐
- ✅ 使用 VILA-U 的标准训练流程 (`train.py` + `train_mem.py`)
- ✅ 支持 DeepSpeed ZeRO-2/3
- ✅ 支持多节点分布式训练
- ✅ 支持梯度累积和混合精度训练

### 2. 数据预处理增强
- ✅ **移除暂停区间** (pause intervals)：过滤掉低运动片段
- ✅ **标准化图像分辨率**：统一调整为 256×256
- ✅ 使用 VILA-U 的 `image_processor` 进行预处理
- ✅ 动作归一化到 [-1, 1]

### 3. 新增文件

#### 训练脚本
- `scripts/train/train_action_prediction.sh` - 训练启动脚本
- `vila_u/train/train_action_prediction_mem.py` - 内存优化入口
- `vila_u/train/train_action_prediction_main.py` - 主训练逻辑

#### 数据加载器
- `vila_u/data/libero_dataset_v2.py` - 增强版数据加载器

---

## 📋 新功能详解

### 1. 移除暂停区间 (Pause Removal)

**原理**：
```python
def _is_pause(self, action: np.ndarray) -> bool:
    """检测动作是否为暂停"""
    position_norm = np.linalg.norm(action[:3])  # 位置变化
    rotation_norm = np.linalg.norm(action[3:6])  # 旋转变化

    # 如果位置和旋转变化都很小，认为是暂停
    return (position_norm < 0.01 and rotation_norm < 0.01)
```

**效果**：
- 过滤掉机器人静止或微小抖动的片段
- 提高训练数据质量
- 减少无效样本

**统计信息**：
```python
# 计算数据集中的暂停比例
stats = compute_dataset_statistics(data_root)
print(f"Pause ratio: {stats['pause_ratio']:.2%}")
```

### 2. 图像标准化 (256×256)

**实现**：
```python
# 使用 VILA-U 的 image_processor
obs_tensor = self.image_processor.preprocess(
    obs_image,
    return_tensors='pt',
    do_resize=True,
    size={'height': 256, 'width': 256},
)['pixel_values'].squeeze(0)  # [3, 256, 256]
```

**优势**：
- 与 VILA-U 预训练时的图像尺寸一致
- 使用相同的归一化参数
- 保证特征提取的有效性

---

## 🚀 使用方法

### 方法 1: 使用训练脚本（推荐）

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep

# 设置环境变量
export MODEL_PATH="/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"
export DATA_ROOT="/path/to/libero_goal"
export OUTPUT_DIR="./checkpoints/vila-u-action-prediction"
export BATCH_SIZE=32
export ACC_STEP=4
export NUM_EPOCHS=10
export LEARNING_RATE=1e-5

# 运行训练
bash scripts/train/train_action_prediction.sh
```

### 方法 2: 直接使用 torchrun

```bash
torchrun --nproc_per_node=8 --master_port=25001 \
    vila_u/train/train_action_prediction_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/share/1919650160032350208/sj/vila-u/vila-u-7b-256 \
    --data_root /path/to/libero_goal \
    --version v1 \
    --mm_projector mlp2x_gelu \
    --tune_mm_projector True \
    --tune_language_model True \
    --tune_vision_tower False \
    --image_size 256 \
    --bf16 True \
    --output_dir ./checkpoints/vila-u-action-prediction \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --save_steps 500 \
    --logging_steps 10 \
    --report_to wandb \
    --action_chunk_size 10 \
    --action_dim 7 \
    --remove_pause_intervals True \
    --pause_threshold 0.01
```

---

## ⚙️ 配置参数

### 模型参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name_or_path` | - | VILA-U 模型路径 |
| `tune_language_model` | True | 是否微调 LLM |
| `tune_mm_projector` | True | 是否微调多模态投影器 |
| `tune_vision_tower` | False | 是否微调视觉编码器（建议冻结） |

### 数据参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_root` | - | LIBERO Goal 数据集路径 |
| `image_size` | 256 | 图像分辨率 |
| `action_chunk_size` | 10 | 动作序列长度 |
| `action_dim` | 7 | 动作维度 (7-DoF) |
| `remove_pause_intervals` | True | 是否移除暂停区间 |
| `pause_threshold` | 0.01 | 暂停检测阈值 |

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_epochs` | 10 | 训练轮数 |
| `per_device_train_batch_size` | 4 | 每设备批大小 |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `learning_rate` | 1e-5 | 学习率 |
| `warmup_ratio` | 0.03 | 预热比例 |
| `save_steps` | 500 | 保存间隔 |

---

## 📊 与旧版本对比

| 特性 | 旧版本 | 新版本 (重构) |
|------|--------|--------------|
| 训练框架 | 自定义循环 | VILA-U 标准框架 |
| 分布式训练 | 不支持 | ✅ 支持多节点 |
| DeepSpeed | 不支持 | ✅ 支持 ZeRO-2/3 |
| 图像预处理 | ImageNet 归一化 | ✅ VILA-U image_processor |
| 暂停移除 | 不支持 | ✅ 支持 |
| 图像分辨率 | 可变 | ✅ 统一 256×256 |
| 梯度检查点 | 不支持 | ✅ 支持 |
| WandB 集成 | 基础 | ✅ 完整集成 |

---

## 🔍 数据集统计

运行以下代码查看数据集统计信息：

```python
from vila_u.data.libero_dataset_v2 import compute_dataset_statistics

stats = compute_dataset_statistics('/path/to/libero_goal')

print(f"Total samples: {stats['total_samples']}")
print(f"Total pauses: {stats['total_pauses']}")
print(f"Pause ratio: {stats['pause_ratio']:.2%}")
print(f"Action norm: {stats['action_norm_mean']:.4f} ± {stats['action_norm_std']:.4f}")
print(f"Action norm range: [{stats['action_norm_min']:.4f}, {stats['action_norm_max']:.4f}]")
```

---

## ⚠️ 注意事项

### 1. DeepSpeed 配置
确保 `scripts/zero2.json` 存在：
```json
{
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2
    },
    "bf16": {
        "enabled": true
    }
}
```

### 2. 环境依赖
```bash
# 确保安装了必要的包
pip install deepspeed
pip install wandb
pip install flash-attn --no-build-isolation
```

### 3. 内存优化
如果遇到 OOM：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing`
- 使用 DeepSpeed ZeRO-3

---

## 📈 预期效果

### 训练速度
- 单卡 (A100 80GB): ~2-3 samples/sec
- 8卡 (A100 80GB): ~16-24 samples/sec

### 内存占用
- 单卡: ~40GB (batch_size=4, gradient_checkpointing=True)
- 使用 ZeRO-2: ~25GB per GPU

### 收敛情况
- L1 Loss 应该在前几个 epoch 快速下降
- 预期最终 loss < 0.1

---

## 🐛 故障排查

### 问题 1: 找不到 DeepSpeed 配置
```bash
# 创建 zero2.json
mkdir -p scripts
cat > scripts/zero2.json << 'EOF'
{
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2
    },
    "bf16": {
        "enabled": true
    }
}
EOF
```

### 问题 2: 数据加载失败
```bash
# 检查数据集格式
python -c "
import h5py
f = h5py.File('/path/to/libero_goal/task_0.hdf5', 'r')
print('Keys:', list(f.keys()))
print('Demo keys:', list(f['data'].keys()))
demo = f['data']['demo_0']
print('Obs keys:', list(demo['obs'].keys()))
print('Image shape:', demo['obs/agentview_rgb'].shape)
print('Actions shape:', demo['actions'].shape)
"
```

### 问题 3: 暂停移除导致样本过少
```bash
# 调整暂停阈值
--pause_threshold 0.005  # 更严格（移除更多）
--pause_threshold 0.02   # 更宽松（保留更多）

# 或者禁用暂停移除
--remove_pause_intervals False
```

---

## ✅ 总结

重构后的训练代码：
1. ✅ 完全对齐 VILA-U 的训练框架
2. ✅ 支持大规模分布式训练
3. ✅ 增强的数据预处理（暂停移除 + 图像标准化）
4. ✅ 更好的内存效率和训练速度
5. ✅ 完整的 WandB 集成和日志记录

现在可以开始训练了！🚀
