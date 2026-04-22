# Phase 4 修正说明

## 问题发现

在初始实现中，我错误地使用了**双图像输入**的方式，这与论文方案不符。

### ❌ 错误的实现

```python
# 错误：使用两个独立的图像作为输入
images = torch.cat([observation, subgoal_image], dim=0)  # [2, 3, H, W]
prompt = f"{image_token}\nObservation.\n{image_token}\nSubgoal.\n{instruction}"
```

这种方式将观测图像和子目标图像作为两个独立的视觉输入，不符合论文的自回归生成方案。

## 论文的正确方案

根据 `docs/archive/CoT-VLA-modification-plan.md`，正确的推理流程是：

### ✅ 正确的实现

**序列结构**：
```
[观测图像tokens] [文本tokens] [生成的子目标图像tokens] [<act> token] → 动作预测
```

**推理步骤**：
1. 编码观测图像和文本指令
2. **自回归生成**子目标图像tokens（1024个tokens，因果注意力）
3. 拼接 `<act>` token
4. 使用全注意力解码动作（<act> token可以attend到所有之前的tokens）

**关键点**：
- 子目标是**生成的tokens**，不是独立的图像输入
- 整个过程是**单序列自回归**
- 动作预测基于整个序列的上下文，包括生成的子目标tokens

## 修正内容

### 1. 删除错误的方法

删除了以下错误实现：
- `generate_subgoal_image()` - 错误地将子目标解码为图像
- `predict_action_with_visual_cot()` - 错误的双图像输入
- `_predict_action_continuous_with_subgoal()` - 错误的双图像输入
- `_predict_action_discrete_with_subgoal()` - 错误的双图像输入
- 错误的 `compute_cot_vla_loss()` - 基于双图像输入的损失

### 2. 正确的实现

#### `generate_with_visual_cot()` - 论文方案

```python
@torch.inference_mode()
def generate_with_visual_cot(
    self,
    observation: torch.Tensor,
    instruction: str,
    subgoal_horizon: int = 8,
) -> torch.Tensor:
    """
    CoT-VLA 闭环推理：先生成子目标图像tokens，再预测动作。

    论文方案：
    1. 编码观测图像和文本指令
    2. 自回归生成子目标图像tokens（因果注意力）
    3. 拼接<act> token
    4. 使用全注意力解码动作
    """
    # Step 1: 编码观测图像和文本指令
    prompt = f"{image_token}\n{instruction}"
    input_ids = tokenize_conversation(...)

    # Step 2: 自回归生成子目标图像tokens
    subgoal_token_len = 16 * 16 * 4  # 1024 tokens
    subgoal_ids = self.generate(
        input_ids=input_ids,
        images=observation,
        max_new_tokens=subgoal_token_len,
        do_sample=False,
    )

    # Step 3: 拼接<act> token
    act_token_id = self.tokenizer.convert_tokens_to_ids("<act>")
    full_seq = torch.cat([subgoal_ids, act_token], dim=1)
    action_position = full_seq.shape[1] - 1

    # Step 4: 使用全注意力解码动作
    outputs = self.llm.model(
        input_ids=full_seq,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]

    # Step 5: 动作头解码
    act_hidden = hidden_states[:, action_position, :]
    raw = self.action_head(act_hidden)
    actions = raw.view(chunk_size, action_dim)
    actions = torch.tanh(actions)

    return actions
```

#### `compute_cot_vla_loss()` - 论文方案

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

    论文方案：
    - 视觉损失：子目标图像tokens的交叉熵损失
    - 动作损失：动作值的L1损失
    """
    # 视觉/文本自回归损失
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = visual_labels[..., 1:].contiguous()
    loss_visual = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )

    # 动作回归损失（L1）
    loss_action = F.l1_loss(action_pred, action_labels)

    return loss_visual + loss_action
```

### 3. 数据加载器

数据加载器的实现是正确的，它返回：
- `observations`: 当前观测图像
- `subgoal_images`: 未来n帧的子目标图像（用于训练时编码成tokens作为标签）
- `action_labels`: 动作标签

在训练时，子目标图像会被编码成tokens，作为视觉自回归损失的标签。

## 与论文的对应关系

| 论文组件 | 修正后的实现 | 对应代码 |
|---------|-------------|---------|
| 观测图像编码 | ✅ | `input_ids, images=observation` |
| 自回归生成子目标tokens | ✅ | `self.generate(max_new_tokens=1024)` |
| 拼接<act> token | ✅ | `torch.cat([subgoal_ids, act_token])` |
| 全注意力解码动作 | ✅ | `llm.model(full_seq)` |
| 视觉自回归损失 | ✅ | `F.cross_entropy(shift_logits, shift_labels)` |
| 动作回归损失 | ✅ | `F.l1_loss(action_pred, action_labels)` |

## 代码统计

**删除的代码**: ~400行（错误实现）
**新增的代码**: ~160行（正确实现）
**净减少**: ~240行

## 关键改进

1. **单序列自回归**: 符合论文方案，整个推理过程是一个连续的序列生成
2. **子目标tokens**: 子目标以tokens形式存在于序列中，不是独立的图像输入
3. **简化实现**: 删除了复杂的双图像处理逻辑
4. **正确的损失**: 视觉自回归损失 + 动作回归损失

## 下一步工作

1. **注册特殊tokens**:
   ```python
   tokenizer.add_special_tokens({
       'additional_special_tokens': ['<act>', '</act>', '<subgoal>', '</subgoal>']
   })
   model.resize_token_embeddings(len(tokenizer))
   ```

2. **训练时的序列构建**: 需要实现训练时如何构建包含子目标tokens的序列

3. **测试验证**: 更新测试脚本以验证修正后的实现

## 参考文档

- 论文方案: `docs/archive/CoT-VLA-modification-plan.md`
- 阶段规划: `docs/roadmaps/COT_VLA_PHASED_TASK_LIST.md`

---

**修正日期**: 2026-04-22
**修正原因**: 初始实现使用了错误的双图像输入方式，不符合论文的自回归生成方案
**修正结果**: 严格按照论文方案实现单序列自回归生成
