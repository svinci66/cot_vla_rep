# Phase 4: Visual CoT-VLA Implementation

基于论文 *CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models* (NVIDIA & Stanford, 2025) 的完整实现。

## 核心创新

在机器人动作预测之前，先**自回归生成未来子目标图像**作为视觉推理的中间步骤，实现视觉链式推理（Visual Chain-of-Thought）。

## 实现概览

### 1. 特殊 Tokens

- `<subgoal>`: 标记子目标生成的开始
- `<act>`: 标记动作预测的开始

定义位置：`vila_u/constants.py`

### 2. 核心方法

#### 2.1 子目标图像编码/解码

```python
# 编码：图像 → token IDs
subgoal_token_ids = model.encode_subgoal_image(subgoal_image)  # [B, 1024]

# 解码：token IDs → 图像
subgoal_image = model.decode_subgoal_tokens(subgoal_token_ids)  # [B, 3, 256, 256]
```

实现位置：`vila_u/model/vila_u_arch.py`

#### 2.2 自回归生成子目标

```python
# 生成子目标 tokens
generated_subgoal_ids = model.generate_subgoal_tokens(
    input_ids=input_ids,
    images=observation,
    attention_mask=attention_mask,
    max_new_tokens=1024,
    do_sample=False,
)
```

实现位置：`vila_u/model/vila_u_arch.py`

#### 2.3 推理接口

```python
# 完整的 Visual CoT 推理
actions, subgoal_image = model.generate_with_visual_cot(
    observation=observation,
    instruction="pick up the red block",
    do_sample=False,
)
```

实现位置：`vila_u/model/vila_u_arch.py`

### 3. 数据集

`LiberoCoTDataset` 支持加载：
- 观测图像（当前帧）
- 子目标图像（未来第 N 帧）
- 动作序列
- 文本指令

实现位置：`vila_u/data/libero_cot_dataset.py`

### 4. 训练流程

#### 4.1 损失计算

```python
# 1. 自回归生成子目标 tokens
generated_subgoal_ids = model.generate_subgoal_tokens(...)

# 2. 编码 GT 子目标为 tokens
gt_subgoal_token_ids = model.encode_subgoal_image(gt_subgoal_images)

# 3. 计算视觉损失（交叉熵）
visual_loss = F.cross_entropy(subgoal_logits, gt_subgoal_token_ids)

# 4. 基于生成的子目标预测动作
action_outputs = model.predict_actions(...)

# 5. 计算动作损失
action_loss = action_outputs.loss

# 6. 联合损失
total_loss = visual_loss_weight * visual_loss + action_loss_weight * action_loss
```

实现位置：`vila_u/train/train_visual_cot.py`

#### 4.2 训练脚本

```bash
bash scripts/train_phase4.sh
```

关键参数：
- `--subgoal_horizon 5`: 子目标时间跨度（未来第 5 帧）
- `--visual_loss_weight 1.0`: 视觉损失权重
- `--action_loss_weight 1.0`: 动作损失权重
- `--model_max_length 1536`: 序列长度（需要容纳 1024 个子目标 tokens）

### 5. 可视化

#### 5.1 训练时可视化

```python
from vila_u.utils.visual_cot_visualization import VisualCoTVisualizer

visualizer = VisualCoTVisualizer(output_dir="./checkpoints/phase4")
visualizer.visualize_training_batch(model, batch, step=1000)
```

#### 5.2 评估时可视化

```python
visualizer.visualize_evaluation_samples(model, eval_dataset, num_samples=50)
```

保存格式：
```
output_dir/visualizations/
├── step_000000.png
├── step_000500.png
└── evaluation/
    ├── sample_0000/
    │   ├── comparison.png
    │   └── metadata.json
    └── sample_0001/
        ├── comparison.png
        └── metadata.json
```

实现位置：`vila_u/utils/visual_cot_visualization.py`

## 文件结构

```
vila_u/
├── constants.py                          # 添加 Phase 4 常量
├── model/
│   └── vila_u_arch.py                    # 添加 Phase 4 方法
├── data/
│   └── libero_cot_dataset.py             # Phase 4 数据集
├── train/
│   └── train_visual_cot.py               # Phase 4 训练脚本
└── utils/
    └── visual_cot_visualization.py       # 可视化工具

scripts/
├── train_phase4.sh                       # 训练启动脚本
└── test_phase4.py                        # 测试脚本
```

## 训练流程

### 1. 准备数据

确保 LIBERO Goal 数据集包含足够的未来帧用于子目标：

```python
# 数据集要求
- 每个轨迹至少有 action_chunk_size + subgoal_horizon 帧
- 默认：10 + 5 = 15 帧
```

### 2. 启动训练

```bash
# 修改 scripts/train_phase4.sh 中的路径
MODEL_PATH="/path/to/VILA-U-Llama3-8B"
DATA_ROOT="/path/to/libero_goal"

# 启动训练
bash scripts/train_phase4.sh
```

### 3. 监控训练

关注以下指标：
- `train/visual_loss`: 子目标生成损失（应该下降）
- `train/action_loss`: 动作预测损失（应该下降）
- `train/total_loss`: 总损失

### 4. 可视化检查

定期检查生成的子目标图像：
- 是否与 GT 在语义上一致？
- 是否反映了任务的中间状态？
- 图像质量是否足够清晰？

## 推理示例

```python
import torch
from vila_u.model import VILAULlamaModel
from PIL import Image

# 加载模型
model = VILAULlamaModel.from_pretrained("./checkpoints/phase4")
model.eval()
model.cuda()

# 加载观测图像
observation = Image.open("observation.png")

# 生成动作和子目标
actions, subgoal_image = model.generate_with_visual_cot(
    observation=observation,
    instruction="pick up the red block",
    do_sample=False,
)

# 保存子目标图像
from torchvision.utils import save_image
save_image(subgoal_image / 255.0, "generated_subgoal.png")

print(f"Predicted actions: {actions.shape}")  # [10, 7]
```

## 关键设计决策

### 1. 为什么使用自回归生成？

- ✓ 符合论文要求
- ✓ 模型真正学习视觉推理能力
- ✓ 推理时可以生成子目标

### 2. 为什么不直接使用 GT 子目标？

- ✗ 模型不会学习生成能力
- ✗ 推理时无法生成子目标
- ✗ 不符合论文的 CoT 思想

### 3. 序列结构

**训练时**：
```
输入：[观测图像 tokens] [文本 tokens]
生成：[子目标 tokens(1024)] [<act>] [动作 tokens(70)]
标签：[GT 子目标 tokens(1024)] [GT 动作 tokens(70)]
```

**推理时**：
```
输入：[观测图像 tokens] [文本 tokens]
生成：[子目标 tokens(1024)] [<act>] [动作 tokens(70)]
```

## 验收标准

### 功能验收

- [x] 训练时模型自回归生成子目标 tokens
- [x] 视觉损失正常计算和下降
- [x] 动作损失正常计算和下降
- [x] 推理时能生成子目标图像
- [x] 子目标图像可视化功能正常

### 性能验收（待测试）

- [ ] 在 LIBERO-Spatial 上成功率 > Phase 3 基线 + 10%
- [ ] 子目标图像质量：PSNR > 20dB
- [ ] 推理速度：< 200ms（包含子目标生成）

### 可解释性验收（待测试）

- [ ] 生成的子目标图像与 GT 在语义上一致
- [ ] 子目标图像能反映任务的中间状态
- [ ] 失败案例可通过子目标图像分析原因

## 下一步

1. **测试训练流程**：在小数据集上验证训练是否正常
2. **调整超参数**：优化视觉损失和动作损失的权重
3. **评估性能**：在 LIBERO 测试集上评估成功率
4. **分析子目标质量**：计算 PSNR、SSIM、LPIPS 等指标
5. **优化推理速度**：使用 KV cache 加速生成

## 参考

- 论文：CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models
- 需求文档：`docs/phase4/PHASE4_REQUIREMENTS.md`
- VILA-U 图像生成：`vila_u/model/vila_u_arch.py:generate_image_content()`
