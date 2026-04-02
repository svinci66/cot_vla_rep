# VILA-U → CoT-VLA 改造方案

> 基于论文：*CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models*（NVIDIA & Stanford, 2025）

## 概述

CoT-VLA 以 VILA-U 为基座，在机器人动作预测之前先**自回归生成未来子目标图像**作为视觉推理中间步骤，再根据子目标生成动作序列。核心创新点：

1. **视觉链式推理（Visual CoT）**：先生成 n 步后的子目标图像，再预测动作
2. **混合注意力机制**：图像/文本生成用因果注意力，动作预测用全注意力
3. **两阶段训练**：预训练（机器人演示 + 无动作视频） → 下游任务微调

---

## 推理时序

```
观测图像 + 文本指令
    ↓
自回归生成子目标图像 tokens（因果注意力，复用 VILA-U 现有生成能力）
    ↓
生成 <act> token，用全注意力解码动作 chunk（新增）
    ↓
机器人执行 ACTION_CHUNK_SIZE 步动作
    ↓
获取新观测，闭环循环
```

---

## 文件改动清单

```
vila_u/
├── constants.py                        ← 新增动作相关常量和特殊 token
├── model/
│   ├── configuration_vila_u.py         ← 新增动作配置项
│   ├── vila_u_arch.py                  ← 新增动作头、推理流程、训练损失
│   └── language_model/
│       └── vila_u_llama.py             ← 新增混合注意力 mask 逻辑
├── data/
│   └── robot_dataset.py               ← 新建：机器人数据集加载
└── cli/
    └── eval.py                         ← 新增机器人评测逻辑
```

---

## 改动一：`vila_u/constants.py`

在文件末尾追加动作相关常量：

```python
# ===== CoT-VLA 新增 =====
ACTION_TOKEN_INDEX = -201
DEFAULT_ACT_START_TOKEN  = "<act>"
DEFAULT_ACT_END_TOKEN    = "</act>"
DEFAULT_SUBGOAL_START_TOKEN = "<subgoal>"
DEFAULT_SUBGOAL_END_TOKEN   = "</subgoal>"

ACTION_DIM         = 7    # 7-DoF 末端执行器控制（Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper）
ACTION_CHUNK_SIZE  = 10   # 每次预测的动作步数
SUBGOAL_HORIZON_LOW  = 4  # 子目标采样下界（帧数）
SUBGOAL_HORIZON_HIGH = 16 # 子目标采样上界（帧数）
```

---

## 改动二：`vila_u/model/configuration_vila_u.py`

在 `VILAUConfig.__init__()` 中新增字段：

```python
# ===== CoT-VLA 新增 =====
self.action_dim        = kwargs.pop("action_dim", 7)
self.action_chunk_size = kwargs.pop("action_chunk_size", 10)
self.use_visual_cot    = kwargs.pop("use_visual_cot", False)
self.subgoal_horizon_low  = kwargs.pop("subgoal_horizon_low", 4)
self.subgoal_horizon_high = kwargs.pop("subgoal_horizon_high", 16)
```

---

## 改动三：`vila_u/model/language_model/vila_u_llama.py`

### 3.1 新增混合注意力 mask 构建函数

在文件顶部 import 区域后、类定义前插入：

```python
def build_hybrid_attention_mask(
    seq_len: int,
    action_token_positions: list,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    构建混合注意力掩码：
    - 默认使用因果掩码（下三角）用于图像/文本 token
    - action_token_positions 中的行改为全注意力（该行全 True）

    返回 additive mask，shape: [1, 1, seq_len, seq_len]
    0.0 表示可见，-inf 表示遮蔽
    """
    # 因果掩码
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    # 动作 token 位置改为全注意力
    for pos in action_token_positions:
        causal[pos, :] = True
    # 转为 additive mask
    additive = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    additive.masked_fill_(~causal, float("-inf"))
    return additive.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
```

### 3.2 修改 `forward()` 方法签名

在 `VILAULlamaModel.forward()` 参数列表中追加：

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    # ... 原有参数 ...
    action_token_positions: Optional[list] = None,  # 新增
    **kwargs,
):
    # 若有动作 token 位置，覆盖 attention_mask
    if action_token_positions is not None:
        seq_len = input_ids.shape[1]
        attention_mask = build_hybrid_attention_mask(
            seq_len,
            action_token_positions,
            device=input_ids.device,
            dtype=self.dtype,
        )
    # 其余逻辑不变
    ...
```

---

## 改动四：`vila_u/model/vila_u_arch.py`

### 4.1 在 `init_vlm()` 末尾新增动作预测头

```python
# 在 self.is_loaded = True 之前插入
if getattr(config, "use_visual_cot", False):
    from torch import nn
    action_out_dim = config.action_chunk_size * config.action_dim  # 默认 10×7=70
    self.action_head = nn.Linear(
        config.hidden_size,   # LLM 隐层维度，如 4096
        action_out_dim,
        bias=True,
    )
    # 初始化为小值，避免训练初期梯度爆炸
    nn.init.normal_(self.action_head.weight, std=0.02)
    nn.init.zeros_(self.action_head.bias)
```

### 4.2 新增动作预测方法

在 `VILAUMetaModel` 类中插入：

```python
def predict_actions(
    self,
    hidden_states: torch.Tensor,
    action_positions: torch.Tensor,
) -> torch.Tensor:
    """
    从 LLM 隐层状态解码动作序列。

    Args:
        hidden_states:    [B, seq_len, hidden_size]
        action_positions: [B] 每个样本 <act> token 在序列中的位置
    Returns:
        actions: [B, ACTION_CHUNK_SIZE, ACTION_DIM]
    """
    B = hidden_states.shape[0]
    act_hidden = hidden_states[torch.arange(B), action_positions]  # [B, hidden_size]
    raw = self.action_head(act_hidden)                             # [B, chunk*dim]
    return raw.view(B, self.config.action_chunk_size, self.config.action_dim)
```

### 4.3 新增训练损失计算方法

```python
def compute_cot_vla_loss(
    self,
    lm_logits: torch.Tensor,
    action_pred: torch.Tensor,
    visual_labels: torch.Tensor,
    action_labels: torch.Tensor,
) -> torch.Tensor:
    """
    总损失 = 视觉自回归损失 + 动作回归损失

    Args:
        lm_logits:     [B, seq_len, vocab_size]  LLM 输出 logits
        action_pred:   [B, chunk_size, action_dim]  动作预测值
        visual_labels: [B, seq_len]  图像/文本 token 标签（IGNORE_INDEX 处不计算）
        action_labels: [B, chunk_size, action_dim]  真实动作（归一化到 [-1, 1]）
    """
    import torch.nn.functional as F

    # 视觉/文本自回归损失（原 VILA-U 损失不变）
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = visual_labels[..., 1:].contiguous()
    loss_visual = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )

    # 动作回归损失（L1，对异常值更鲁棒）
    loss_action = F.l1_loss(action_pred, action_labels)

    return loss_visual + loss_action
```

### 4.4 新增闭环推理方法

```python
@torch.inference_mode()
def generate_with_visual_cot(
    self,
    observation: torch.Tensor,
    instruction: str,
    subgoal_horizon: int = 8,
) -> torch.Tensor:
    """
    CoT-VLA 闭环推理：先生成子目标图像，再预测动作。

    Args:
        observation:      当前观测图像，[1, 3, H, W]，值域 [0, 1]
        instruction:      自然语言任务指令
        subgoal_horizon:  预测未来第几帧作为子目标
    Returns:
        actions: [1, ACTION_CHUNK_SIZE, ACTION_DIM]
    """
    # Step 1: 编码观测图像和文本指令（复用现有逻辑）
    input_ids, pixel_values = self._prepare_robot_inputs(observation, instruction)

    # Step 2: 自回归生成子目标图像 tokens（因果注意力，复用现有 generate_image 逻辑）
    # 每张 256×256 图像编码为 16×16×4 tokens（残差深度 4）
    subgoal_token_len = 16 * 16 * 4
    subgoal_ids = self.generate(
        input_ids,
        pixel_values=pixel_values,
        max_new_tokens=subgoal_token_len,
        do_sample=False,
    )

    # Step 3: 拼接子目标序列，在末尾添加 <act> token
    act_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_ACT_START_TOKEN)
    act_token = torch.tensor([[act_token_id]], device=subgoal_ids.device)
    full_seq = torch.cat([subgoal_ids, act_token], dim=1)
    action_position = full_seq.shape[1] - 1  # <act> token 的位置

    # Step 4: 使用全注意力解码动作（覆盖 attention_mask）
    outputs = self.llm(
        full_seq,
        action_token_positions=[action_position],
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]  # 取最后一层

    # Step 5: 动作头解码
    actions = self.predict_actions(
        hidden_states,
        torch.tensor([action_position], device=full_seq.device),
    )
    return actions  # [1, ACTION_CHUNK_SIZE, ACTION_DIM]
```

---

## 改动五：`vila_u/data/robot_dataset.py`（新建）

```python
"""
机器人数据集加载器，支持：
- Open X-Embodiment (OpenX) 格式的机器人演示数据
- 无动作视频数据（EPIC-KITCHENS / Something-Something V2）
"""
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from vila_u.constants import (
    ACTION_DIM, ACTION_CHUNK_SIZE,
    SUBGOAL_HORIZON_LOW, SUBGOAL_HORIZON_HIGH,
)


class RobotDemonstrationDataset(Dataset):
    """
    加载机器人演示数据。
    每条样本返回：
      - observation:    当前帧图像 tensor [3, H, W]
      - instruction:    任务文本指令 str
      - subgoal_image:  未来第 n 帧图像 tensor [3, H, W]
      - action_labels:  接下来 ACTION_CHUNK_SIZE 步的动作 [chunk, ACTION_DIM]，归一化到 [-1, 1]
    """

    def __init__(self, data_root: str, image_size: int = 256, transform=None):
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.episodes = self._load_episode_index()

    def _load_episode_index(self):
        # TODO: 遍历 data_root，加载 OpenX RLDS 格式或自定义 JSON 索引
        raise NotImplementedError("请实现具体数据集索引加载逻辑")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        t = random.randint(0, len(ep["frames"]) - ACTION_CHUNK_SIZE - SUBGOAL_HORIZON_HIGH)
        n = random.randint(SUBGOAL_HORIZON_LOW, SUBGOAL_HORIZON_HIGH)

        obs    = self._load_image(ep["frames"][t])
        subgoal = self._load_image(ep["frames"][t + n])
        actions = torch.tensor(
            ep["actions"][t : t + ACTION_CHUNK_SIZE], dtype=torch.float32
        )  # [chunk, ACTION_DIM]

        return {
            "observation":   obs,
            "instruction":   ep["instruction"],
            "subgoal_image": subgoal,
            "action_labels": actions,
        }

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        if self.transform:
            return self.transform(img)
        import torchvision.transforms.functional as TF
        return TF.to_tensor(img)  # [3, H, W], 值域 [0, 1]


class ActionlessVideoDataset(Dataset):
    """
    加载无动作视频数据（EPIC-KITCHENS / Something-Something V2）。
    仅用于训练视觉推理（子目标生成），不包含动作标签。
    每条样本返回：
      - observation:    某帧图像 tensor [3, H, W]
      - instruction:    视频描述文本 str
      - subgoal_image:  未来第 n 帧图像 tensor [3, H, W]
    """

    def __init__(self, data_root: str, image_size: int = 256, transform=None):
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.clips = self._load_clip_index()

    def _load_clip_index(self):
        # TODO: 加载视频帧索引和文本描述
        raise NotImplementedError("请实现具体视频数据集索引加载逻辑")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        t = random.randint(0, len(clip["frames"]) - SUBGOAL_HORIZON_HIGH - 1)
        n = random.randint(SUBGOAL_HORIZON_LOW, SUBGOAL_HORIZON_HIGH)
        obs     = self._load_image(clip["frames"][t])
        subgoal = self._load_image(clip["frames"][t + n])
        return {
            "observation":   obs,
            "instruction":   clip["description"],
            "subgoal_image": subgoal,
            "action_labels": None,  # 无动作标签
        }

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        if self.transform:
            return self.transform(img)
        import torchvision.transforms.functional as TF
        return TF.to_tensor(img)
```

---

## 改动六：训练脚本（新建 `scripts/train/train_cot_vla.sh`）

```bash
#!/bin/bash
# 阶段一：预训练（机器人演示 + 无动作视频）
# 冻结 vision_tower，训练 llm + mm_projector + depth_transformer + action_head
python -m vila_u.cli.train \
  --model_path mit-han-lab/vila-u-7b-256 \
  --stage pretrain \
  --data_config configs/data/openx+epic+sthsth.json \
  --freeze_modules vision_tower \
  --use_visual_cot true \
  --action_chunk_size 10 \
  --action_dim 7 \
  --subgoal_horizon_low 4 \
  --subgoal_horizon_high 16 \
  --image_size 256 \
  --output_dir checkpoints/cot_vla_pretrain

# 阶段二：下游任务微调
python -m vila_u.cli.train \
  --model_path checkpoints/cot_vla_pretrain \
  --stage finetune \
  --data_config configs/data/task_specific.json \
  --use_visual_cot true \
  --output_dir checkpoints/cot_vla_finetune
```

---

## 优先级与工作量汇总

| 优先级 | 文件 | 改动内容 | 工作量 |
|--------|------|----------|--------|
| P0 | `constants.py` | 新增动作常量和特殊 token | 低 |
| P0 | `configuration_vila_u.py` | 新增动作配置字段 | 低 |
| P0 | `vila_u_arch.py` | 动作预测头 + 训练损失 | 中 |
| P0 | `vila_u_llama.py` | 混合注意力 mask | 中 |
| P1 | `vila_u_arch.py` | 闭环推理方法 `generate_with_visual_cot` | 中 |
| P1 | `data/robot_dataset.py` | 机器人数据集加载器 | 中 |
| P2 | `scripts/train/train_cot_vla.sh` | 训练脚本 | 低 |
| P2 | `cli/eval.py` | 机器人评测逻辑（LIBERO/BridgeV2） | 高 |

---

## 注意事项

1. **视觉 tower 保持冻结**：训练时只更新 `llm`、`mm_projector`、`depth_transformer` 和新增的 `action_head`。
2. **动作归一化**：训练前需将原始动作值归一化到 `[-1, 1]`，推理后反归一化再发给机器人。
3. **特殊 token 注册**：`<act>`、`<subgoal>` 需通过 `tokenizer.add_special_tokens()` 注册，并在模型 embedding 层 `resize_token_embeddings()` 后再训练。
4. **子目标图像分辨率**：与 VILA-U 保持一致，使用 `256×256`，编码为 `16×16×4` tokens。
5. **action chunking 的不连续性**：chunk 边界处可能出现动作跳变，可在执行层加时序平滑（低通滤波）缓解，无需改模型。
