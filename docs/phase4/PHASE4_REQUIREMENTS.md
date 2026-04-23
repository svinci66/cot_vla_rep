# Phase 4: Visual CoT-VLA 完整需求文档

> 基于论文：*CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models* (NVIDIA & Stanford, 2025)

---

## 1. 项目背景

### 1.1 核心创新

CoT-VLA在机器人动作预测之前，先**自回归生成未来子目标图像**作为视觉推理的中间步骤，实现视觉链式推理（Visual Chain-of-Thought）。

### 1.2 关键优势

- **可解释性**：生成的子目标图像可视化了机器人的"思考过程"
- **长期规划**：通过预测未来状态，提升复杂任务的成功率
- **泛化能力**：视觉推理能力可迁移到新场景

---

## 2. 核心需求

### 2.1 推理流程（必须实现）

```
观测图像 + 文本指令
    ↓
【阶段1：视觉推理】
自回归生成子目标图像tokens（1024个tokens，因果注意力）
    ↓
【阶段2：动作预测】
拼接 <act> token，使用全注意力解码动作
    ↓
输出：动作序列 [10, 7]
```

**关键点**：
- ✅ 子目标必须由模型**自回归生成**，不能使用ground truth
- ✅ 生成过程使用**因果注意力**（causal attention）
- ✅ 动作预测使用**全注意力**（full attention）

### 2.2 训练流程（必须实现）

```
输入：
- 观测图像 [B, 3, 256, 256]
- 文本指令 [B, seq_len]
- GT子目标图像 [B, 3, 256, 256]（仅用于计算损失）
- GT动作 [B, 10, 7]

训练步骤：
1. 编码观测图像 → 图像tokens
2. 编码文本指令 → 文本tokens
3. 模型自回归生成子目标tokens（1024个）
4. 编码GT子目标图像 → GT子目标tokens（用于计算损失）
5. 基于生成的子目标tokens预测动作
6. 计算联合损失：
   - 视觉损失：生成的子目标tokens vs GT子目标tokens（交叉熵）
   - 动作损失：预测的动作 vs GT动作（L1损失）
```

**关键点**：
- ❌ 不能直接将GT子目标tokens拼接到输入序列
- ✅ 必须让模型自己生成子目标tokens
- ✅ GT子目标仅用于计算损失，不参与前向传播

### 2.3 序列结构

**训练时**：
```
输入序列：[观测图像tokens] [文本tokens]
生成序列：[子目标tokens(1024)] [<act>] [动作tokens(70)]
标签序列：[GT子目标tokens(1024)] [GT动作tokens(70)]
```

**推理时**：
```
输入序列：[观测图像tokens] [文本tokens]
生成序列：[子目标tokens(1024)] [<act>]
动作预测：基于生成的子目标tokens → [10, 7]
```

---

## 3. 重要需求：子目标图像可视化

### 3.1 需求描述

**必须支持将生成的子目标tokens解码为可视化图像**，用于：
- 验证模型是否学会了视觉推理
- 调试训练过程
- 展示给用户理解机器人的"思考"
- 论文中的可视化结果

### 3.2 技术可行性

✅ **VILA-U已支持图像解码功能**：

```python
# 发现的关键代码：
# 1. 编码图像 → tokens
code, z_q = self.vision_tower.vision_tower.rqvaesiglip.encode_image(image)
# code: [B, 16, 16, 4] - 子目标tokens

# 2. 解码 tokens → 图像
decoded_image = self.vision_tower.vision_tower.rqvaesiglip.decode(z_q)
# decoded_image: [B, 3, 256, 256] - 重建的图像

# 3. 现有的图像生成方法
response = model.generate_image_content(prompt="a robot picking up a cup", cfg=3.0)
# response: [B, 3, 256, 256] - 生成的图像
```

### 3.3 实现要求

**必须实现以下功能**：

1. **训练时可视化**：
   ```python
   def visualize_subgoal_during_training(
       generated_subgoal_tokens: torch.Tensor,  # [B, 1024]
       gt_subgoal_image: torch.Tensor,          # [B, 3, 256, 256]
       save_path: str
   ):
       """
       将生成的子目标tokens解码为图像，与GT对比保存
       """
       # 1. 将flat tokens reshape为 [B, 16, 16, 4]
       # 2. 通过quantizer获取embeddings
       # 3. 使用rqvaesiglip.decode()解码为图像
       # 4. 保存对比图：[观测图像 | 生成的子目标 | GT子目标]
   ```

2. **推理时可视化**：
   ```python
   def generate_with_visual_cot_and_visualize(
       observation: torch.Tensor,
       instruction: str,
       save_path: str
   ) -> Tuple[torch.Tensor, torch.Tensor]:
       """
       生成动作并保存子目标图像

       Returns:
           actions: [1, 10, 7]
           subgoal_image: [1, 3, 256, 256] - 解码后的子目标图像
       """
   ```

3. **评估时批量可视化**：
   ```python
   def evaluate_with_visualization(
       model,
       eval_dataset,
       output_dir: str,
       num_samples: int = 50
   ):
       """
       评估时保存子目标图像样本

       保存格式：
       output_dir/
         ├── sample_0/
         │   ├── observation.png
         │   ├── generated_subgoal.png
         │   ├── gt_subgoal.png
         │   └── metadata.json
         ├── sample_1/
         ...
       """
   ```

---

## 4. 技术实现细节

### 4.1 子目标tokens的表示

**VILA-U的RQ-VAE编码**：
- 输入图像：[B, 3, 256, 256]
- 编码输出：code [B, 16, 16, 4] + z_q [B, 16, 16, embed_dim]
- Flatten后：[B, 1024] tokens

**关键理解**：
- `code`：离散的codebook索引，用于自回归生成
- `z_q`：量化后的embeddings，用于解码为图像

### 4.2 生成到解码的流程

```python
# 训练/推理时生成子目标tokens
subgoal_token_ids = model.generate(
    input_ids=input_ids,
    images=observation,
    max_new_tokens=1024,
    do_sample=False,
    use_cache=True
)  # [B, 1024] - 离散token IDs

# 将token IDs转换为embeddings
subgoal_embeds = model.vision_tower.vision_tower.rqtransformer.embed_with_model_aux(
    subgoal_token_ids,
    model.vision_tower.vision_tower.rqvaesiglip
)  # [B, 1024, depth, embed_dim]

# 累积depth维度（RQ-VAE的残差特性）
subgoal_embeds = torch.cumsum(subgoal_embeds, dim=-2)[:, :, -1, :]  # [B, 1024, embed_dim]

# Reshape为2D特征图
subgoal_embeds = subgoal_embeds.reshape(B, 16, 16, -1)  # [B, 16, 16, embed_dim]

# 解码为图像
subgoal_image = model.vision_tower.vision_tower.rqvaesiglip.decode(subgoal_embeds)
# [B, 3, 256, 256], 值域[-1, 1]

# 转换为可视化格式
subgoal_image = subgoal_image.add_(1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
```

### 4.3 训练损失计算

```python
def compute_cot_vla_loss(
    model,
    observation: torch.Tensor,      # [B, 3, 256, 256]
    instruction_ids: torch.Tensor,  # [B, seq_len]
    gt_subgoal_image: torch.Tensor, # [B, 3, 256, 256]
    gt_actions: torch.Tensor,       # [B, 10, 7]
) -> torch.Tensor:
    """
    Phase 4训练损失（正确实现）
    """
    # 1. 编码观测图像和文本
    input_ids, images = prepare_inputs(observation, instruction_ids)

    # 2. 模型自回归生成子目标tokens
    with torch.no_grad():
        generated_subgoal_ids = model.generate(
            input_ids=input_ids,
            images=images,
            max_new_tokens=1024,
            do_sample=False,
        )  # [B, 1024]

    # 3. 编码GT子目标图像为tokens（用于计算损失）
    gt_subgoal_code, _ = model.vision_tower.vision_tower.rqvaesiglip.encode_image(
        gt_subgoal_image
    )  # [B, 16, 16, 4]
    gt_subgoal_tokens = gt_subgoal_code.reshape(B, -1)  # [B, 1024]

    # 4. 计算视觉自回归损失
    # 使用LLM的logits计算生成的tokens与GT的交叉熵
    visual_loss = F.cross_entropy(
        model.llm.lm_head(generated_subgoal_embeds).view(-1, vocab_size),
        gt_subgoal_tokens.view(-1)
    )

    # 5. 基于生成的子目标预测动作
    action_pred = model.predict_actions_from_subgoal(generated_subgoal_ids)

    # 6. 计算动作损失
    action_loss = F.l1_loss(action_pred, gt_actions)

    # 7. 联合损失
    total_loss = visual_loss + action_loss

    return total_loss, {
        'visual_loss': visual_loss.item(),
        'action_loss': action_loss.item(),
        'generated_subgoal_ids': generated_subgoal_ids,  # 用于可视化
    }
```

---

## 5. 当前实现的问题

### 5.1 问题1：使用GT子目标tokens（严重错误）

**当前代码**：
```python
# 在collate_fn中直接编码GT子目标
subgoal_images = torch.stack([item["subgoal_images"] for item in batch])
code, _ = self.vision_tower.vision_tower.rqvaesiglip.encode_image(subgoal_images)
subgoal_token_ids = code.reshape(B, -1)  # GT tokens

# 训练时直接拼接GT
input_ids = torch.cat([prompt_ids, subgoal_ids, action_input_ids])
```

**问题**：
- ❌ 模型没有学习生成子目标的能力
- ❌ 推理时无法生成子目标（因为训练时从未生成过）
- ❌ 不符合论文的自回归生成方案

### 5.2 问题2：缺少视觉损失

**当前代码**：
```python
# 只计算了动作损失
loss = F.cross_entropy(shift_logits, shift_labels)  # 这是什么损失？
```

**问题**：
- ❌ 没有计算子目标生成的损失
- ❌ 模型无法学习视觉推理

### 5.3 问题3：缺少可视化功能

**当前代码**：
- ❌ 没有解码子目标tokens为图像的功能
- ❌ 无法验证模型是否学会了视觉推理
- ❌ 无法调试训练过程

---

## 6. 实现优先级

### P0 - 核心功能（必须实现）

1. **修改训练流程**：
   - [ ] 移除collate_fn中的GT子目标编码
   - [ ] 实现模型自回归生成子目标tokens
   - [ ] 实现正确的视觉损失计算
   - [ ] 实现联合损失（视觉 + 动作）

2. **修改数据加载**：
   - [ ] 保留GT子目标图像（用于计算损失）
   - [ ] 不要预先编码为tokens

3. **实现推理接口**：
   - [ ] `generate_with_visual_cot()` - 生成子目标和动作
   - [ ] 支持因果注意力（子目标生成）
   - [ ] 支持全注意力（动作预测）

### P1 - 可视化功能（重要）

4. **实现子目标图像解码**：
   - [ ] `decode_subgoal_tokens()` - tokens → 图像
   - [ ] 训练时定期保存可视化样本
   - [ ] 推理时保存子目标图像

5. **实现评估可视化**：
   - [ ] 批量生成子目标图像
   - [ ] 保存对比图（观测 | 生成 | GT）
   - [ ] 生成HTML报告

### P2 - 优化和扩展（可选）

6. **训练优化**：
   - [ ] 支持两阶段训练（预训练 + 微调）
   - [ ] 支持无动作视频数据
   - [ ] 支持多步子目标生成

7. **评估指标**：
   - [ ] 子目标图像质量（PSNR, SSIM, LPIPS）
   - [ ] 子目标语义准确性
   - [ ] 动作预测准确性

---

## 7. 验收标准

### 7.1 功能验收

- [ ] 训练时模型自回归生成子目标tokens（不使用GT）
- [ ] 视觉损失正常下降
- [ ] 动作损失正常下降
- [ ] 推理时能生成合理的子目标图像
- [ ] 子目标图像可视化功能正常

### 7.2 性能验收

- [ ] 在LIBERO-Spatial上成功率 > Phase 3基线 + 10%
- [ ] 子目标图像质量：PSNR > 20dB
- [ ] 推理速度：< 200ms（包含子目标生成）

### 7.3 可解释性验收

- [ ] 生成的子目标图像与GT在语义上一致
- [ ] 子目标图像能反映任务的中间状态
- [ ] 失败案例可通过子目标图像分析原因

---

## 8. 技术风险

### 8.1 训练稳定性

**风险**：自回归生成子目标可能不稳定
**应对**：
- 先在小数据集上验证
- 使用teacher forcing schedule（逐步减少GT使用）
- 调整视觉损失和动作损失的权重

### 8.2 生成质量

**风险**：生成的子目标图像质量差
**应对**：
- 先在视频数据上预训练视觉生成
- 使用更大的子目标horizon范围
- 增加视觉损失权重

### 8.3 显存占用

**风险**：生成1024个tokens显存占用大
**应对**：
- 减小batch size
- 使用gradient checkpointing
- 分阶段生成（先生成子目标，再预测动作）

---

## 9. 参考资料

### 9.1 论文

- CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models (NVIDIA & Stanford, 2025)

### 9.2 代码参考

- VILA-U图像生成：`vila_u_arch.py:generate_image_content()`
- RQ-VAE编码：`rqvaesiglip.encode_image()`
- RQ-VAE解码：`rqvaesiglip.decode()`

### 9.3 相关文档

- `docs/archive/CoT-VLA-modification-plan.md` - 原始改造方案
- `docs/phase4/PHASE4_FINAL.md` - Phase 4总结
- `docs/phase4/PHASE4_CORRECTION.md` - 错误修正说明

---

## 10. 总结

Phase 4的核心是让模型**真正学会自回归生成子目标图像**，而不是使用ground truth。这需要：

1. ✅ 修改训练流程，让模型自己生成子目标
2. ✅ 计算视觉自回归损失
3. ✅ 实现子目标图像解码和可视化
4. ✅ 验证生成的子目标图像质量

**当前实现不符合论文要求，需要重新实现训练逻辑。**

**好消息是VILA-U已经支持图像生成和解码，技术上完全可行！**
