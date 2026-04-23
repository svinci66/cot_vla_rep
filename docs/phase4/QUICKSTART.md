# Phase 4: Visual CoT-VLA 快速开始

## 实现完成情况

✅ **所有核心功能已实现**

### 已完成的任务

1. ✅ 注册 Phase 4 特殊 tokens (`<subgoal>`, `<act>`)
2. ✅ 实现子目标图像编码/解码功能
3. ✅ 创建 LiberoCoTDataset（支持子目标图像）
4. ✅ 实现自回归生成子目标 tokens
5. ✅ 实现视觉损失计算
6. ✅ 实现联合损失训练（视觉 + 动作）
7. ✅ 实现完整推理接口
8. ✅ 实现训练和评估可视化工具

## 快速开始

### 1. 准备环境

```bash
# 确保已安装依赖
pip install torch transformers accelerate wandb matplotlib pillow
```

### 2. 准备数据

确保 LIBERO Goal 数据集路径正确：

```bash
DATA_ROOT="/path/to/libero_goal"
```

数据集要求：
- 每个轨迹至少有 15 帧（action_chunk_size=10 + subgoal_horizon=5）
- 包含 `obs/agentview_rgb` 和 `actions` 数据

### 3. 修改训练脚本

编辑 `scripts/train_phase4.sh`：

```bash
# 修改模型路径
MODEL_PATH="/path/to/VILA-U-Llama3-8B"

# 修改数据路径
DATA_ROOT="/path/to/libero_goal"

# 修改输出路径
OUTPUT_DIR="./checkpoints/phase4-visual-cot"
```

### 4. 启动训练

```bash
bash scripts/train_phase4.sh
```

### 5. 监控训练

在 WandB 中查看：
- `train/visual_loss`: 子目标生成损失
- `train/action_loss`: 动作预测损失
- `train/total_loss`: 总损失

### 6. 可视化检查

训练过程中会自动保存可视化结果到：
```
checkpoints/phase4-visual-cot/visualizations/
├── step_000000.png
├── step_000500.png
└── ...
```

## 推理示例

```python
import torch
from vila_u.model import VILAULlamaModel
from PIL import Image

# 加载训练好的模型
model = VILAULlamaModel.from_pretrained("./checkpoints/phase4-visual-cot")
model.eval()
model.cuda()

# 加载观测图像
observation = Image.open("observation.png")

# 使用 Visual CoT 生成动作
actions, subgoal_image = model.generate_with_visual_cot(
    observation=observation,
    instruction="pick up the red block",
    do_sample=False,
)

# 保存生成的子目标图像
from torchvision.utils import save_image
save_image(subgoal_image / 255.0, "generated_subgoal.png")

print(f"Predicted actions shape: {actions.shape}")  # [10, 7]
print(f"Subgoal image shape: {subgoal_image.shape}")  # [3, 256, 256]
```

## 关键参数说明

### 训练参数

- `--model_max_length 1536`: 序列长度，需要容纳 1024 个子目标 tokens
- `--subgoal_horizon 5`: 子目标时间跨度（未来第 5 帧作为子目标）
- `--visual_loss_weight 1.0`: 视觉损失权重
- `--action_loss_weight 1.0`: 动作损失权重

### 数据参数

- `--action_chunk_size 10`: 动作序列长度
- `--action_dim 7`: 动作维度（7-DoF）
- `--image_size 256`: 图像尺寸
- `--remove_pause_intervals True`: 移除暂停区间

## 训练流程详解

### 第 1 步：自回归生成子目标

```python
# 输入：观测图像 + 文本指令
generated_subgoal_ids = model.generate_subgoal_tokens(
    input_ids=prompt_input_ids,
    images=observations,
    max_new_tokens=1024,
)
```

### 第 2 步：计算视觉损失

```python
# 编码 GT 子目标
gt_subgoal_token_ids = model.encode_subgoal_image(gt_subgoal_images)

# 计算交叉熵损失
visual_loss = F.cross_entropy(subgoal_logits, gt_subgoal_token_ids)
```

### 第 3 步：预测动作

```python
# 基于生成的子目标预测动作
action_outputs = model.predict_actions(generated_subgoal_ids)
action_loss = action_outputs.loss
```

### 第 4 步：联合训练

```python
total_loss = visual_loss_weight * visual_loss + action_loss_weight * action_loss
```

## 预期结果

### 训练指标

- **Visual Loss**: 应该从初始值（~8-10）逐渐下降到 2-3
- **Action Loss**: 应该从初始值（~5-7）逐渐下降到 1-2
- **Total Loss**: 应该稳定下降

### 子目标质量

- 生成的子目标图像应该与 GT 在语义上一致
- 图像应该清晰可辨，反映任务的中间状态
- PSNR 应该 > 20dB（待测试）

### 动作预测

- 成功率应该 > Phase 3 基线 + 10%（待测试）
- 推理速度应该 < 200ms（待测试）

## 故障排查

### 问题 1：显存不足

**解决方案**：
- 减小 batch size：`--per_device_train_batch_size 2`
- 增加梯度累积：`--gradient_accumulation_steps 8`
- 启用梯度检查点：`--gradient_checkpointing True`

### 问题 2：视觉损失不下降

**可能原因**：
- 子目标 horizon 太大，预测难度高
- 视觉损失权重太小

**解决方案**：
- 减小 subgoal_horizon：`--subgoal_horizon 3`
- 增加视觉损失权重：`--visual_loss_weight 2.0`

### 问题 3：生成的子目标图像模糊

**可能原因**：
- 训练不充分
- 学习率太大

**解决方案**：
- 增加训练轮数：`--num_train_epochs 20`
- 降低学习率：`--learning_rate 1e-5`

## 下一步

1. **在小数据集上测试**：验证训练流程是否正常
2. **调整超参数**：优化损失权重和学习率
3. **完整训练**：在完整 LIBERO 数据集上训练
4. **评估性能**：计算成功率、PSNR、SSIM 等指标
5. **分析失败案例**：通过子目标图像分析失败原因

## 文档

- 详细实现文档：`docs/phase4/PHASE4_IMPLEMENTATION.md`
- 需求文档：`docs/phase4/PHASE4_REQUIREMENTS.md`

## 提交历史

```
c101838 Phase 4: Complete Visual CoT-VLA implementation
3fd627a Phase 4: Add Visual CoT-VLA basic implementation
```

## 联系

如有问题，请查看：
- 实现文档：`docs/phase4/PHASE4_IMPLEMENTATION.md`
- 测试脚本：`scripts/test_phase4.py`
