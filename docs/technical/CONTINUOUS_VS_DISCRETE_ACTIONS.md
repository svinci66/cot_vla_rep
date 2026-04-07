# 动作预测方法对比：连续回归 vs 离散分类

## 🎯 当前实现 vs CoT-VLA 论文方法

### 当前实现：连续值回归

```python
# vila_u_arch.py
class VILAUMetaForCausalLM:
    def init_vlm(self, config):
        if getattr(config, "use_action_prediction", False):
            action_out_dim = config.action_chunk_size * config.action_dim
            self.action_head = nn.Linear(
                config.hidden_size,
                action_out_dim,  # 10 * 7 = 70
                bias=True
            )

    def predict_actions(self, hidden_states):
        raw = self.action_head(act_hidden)  # [B, 70]
        actions = raw.view(B, 10, 7)
        actions = torch.tanh(actions)  # 限制到 [-1, 1]
        return actions

# train_action_prediction.py
loss = nn.functional.l1_loss(action_pred, action_labels)
```

**特点**：
- 输出：连续值 [-1, 1]
- 损失：L1 Loss（回归）
- 动作头：独立的 Linear 层

---

### CoT-VLA 论文方法：离散化分类

```python
# CoT-VLA 的做法
class CoTVLAModel:
    def __init__(self):
        self.num_bins = 256  # 每个动作维度离散化为256个bins
        self.action_tokens = self.get_action_tokens()  # 从词表中选256个token

    def get_action_tokens(self):
        """选择词表中最不常用的256个token作为动作token"""
        # 例如：token_ids [32000, 32001, ..., 32255]
        return list(range(32000, 32256))

    def tokenize_action(self, action):
        """
        将连续动作值 [-1, 1] 离散化为 token

        Args:
            action: [B, 7] 连续值 [-1, 1]
        Returns:
            action_tokens: [B, 7] token IDs
        """
        # 1. 归一化到 [0, 1]
        action_normalized = (action + 1) / 2  # [-1, 1] -> [0, 1]

        # 2. 离散化到 [0, 255]
        action_bins = (action_normalized * 255).long()  # [B, 7]

        # 3. 映射到实际的 token IDs
        action_tokens = self.action_tokens[action_bins]  # [B, 7]

        return action_tokens

    def detokenize_action(self, action_tokens):
        """
        将 token 转换回连续动作值

        Args:
            action_tokens: [B, 7] token IDs
        Returns:
            action: [B, 7] 连续值 [-1, 1]
        """
        # 1. 找到 token 在 action_tokens 中的索引
        action_bins = torch.tensor([
            self.action_tokens.index(t) for t in action_tokens
        ])  # [B, 7] 范围 [0, 255]

        # 2. 反归一化
        action_normalized = action_bins.float() / 255  # [0, 1]
        action = action_normalized * 2 - 1  # [-1, 1]

        return action

    def forward(self, input_ids, images, labels=None):
        """
        使用 LLM 的 lm_head 进行动作预测
        """
        # 1. 前向传播
        outputs = self.llm(
            input_ids=input_ids,
            images=images,
            output_hidden_states=True,
        )

        # 2. 使用 LLM 的 lm_head（不需要额外的 action_head）
        logits = outputs.logits  # [B, seq_len, vocab_size]

        # 3. 提取动作预测位置的 logits
        action_logits = logits[:, -7:, :]  # 最后7个位置

        # 4. 只关注动作 token 的概率
        action_logits = action_logits[:, :, self.action_tokens]  # [B, 7, 256]

        if labels is not None:
            # 训练：计算交叉熵损失
            action_token_labels = self.tokenize_action(labels)  # [B, 7]
            loss = F.cross_entropy(
                action_logits.view(-1, 256),
                action_token_labels.view(-1)
            )
            return loss
        else:
            # 推理：生成动作 token
            action_tokens = torch.argmax(action_logits, dim=-1)  # [B, 7]
            actions = self.detokenize_action(action_tokens)
            return actions

# 训练
model = CoTVLAModel()
loss = model(input_ids, images, labels=action_labels)

# 推理
actions = model(input_ids, images)
```

---

## 📊 两种方法的详细对比

### 1. 动作表示

| 方法 | 动作表示 | 输出维度 | 值域 |
|------|---------|---------|------|
| **当前实现** | 连续值 | [B, 10, 7] | [-1, 1] |
| **CoT-VLA** | 离散token | [B, 7] (token IDs) | [32000, 32255] |

### 2. 模型架构

| 组件 | 当前实现 | CoT-VLA |
|------|---------|---------|
| 动作头 | 独立的 `nn.Linear(hidden_size, 70)` | 使用 LLM 的 `lm_head` |
| 激活函数 | Tanh | Softmax (隐式在交叉熵中) |
| 输出层 | 70维连续值 | vocab_size 维 logits |

### 3. 损失函数

```python
# 当前实现：L1 Loss (回归)
loss = F.l1_loss(action_pred, action_labels)
# 或 MSE Loss
loss = F.mse_loss(action_pred, action_labels)

# CoT-VLA：Cross-Entropy Loss (分类)
loss = F.cross_entropy(action_logits, action_token_labels)
```

### 4. 训练过程

#### 当前实现
```python
# 1. 前向传播
outputs = model(input_ids, images, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]

# 2. 动作预测
action_pred = model.predict_actions(hidden_states)  # [B, 10, 7]

# 3. 计算损失
loss = F.l1_loss(action_pred, action_labels)
```

#### CoT-VLA
```python
# 1. 将动作标签离散化
action_token_labels = tokenize_action(action_labels)  # [B, 7]

# 2. 构建输入序列（包含动作 token 位置）
# 输入格式：<image> instruction <action_1> <action_2> ... <action_7>
input_ids = build_input_with_action_slots(instruction)

# 3. 前向传播
logits = model(input_ids, images)  # [B, seq_len, vocab_size]

# 4. 提取动作位置的 logits
action_logits = logits[:, action_positions, :]  # [B, 7, vocab_size]
action_logits = action_logits[:, :, action_token_ids]  # [B, 7, 256]

# 5. 计算交叉熵损失
loss = F.cross_entropy(action_logits.view(-1, 256), action_token_labels.view(-1))
```

---

## 🔍 深入分析：为什么 CoT-VLA 使用离散化？

### 优势

#### 1. 利用预训练 LLM 的能力
```python
# LLM 在预训练时学到的模式：
# "The cat sat on the ___" → "mat" (token 预测)

# 类似地，在动作空间：
# "<image> pick up the bowl <action_1> <action_2> ___" → <action_3> (token 预测)
```

LLM 天然擅长 next-token prediction，可以直接利用这个能力。

#### 2. 训练稳定
- 分类问题通常比回归问题更稳定
- 交叉熵损失有更好的梯度特性
- 不需要仔细调整学习率

#### 3. 统一的建模框架
```python
# 所有输出都是 token 预测
text_output = model.generate(...)      # 文本 token
action_output = model.generate(...)    # 动作 token
```

#### 4. 可以使用 LLM 的生成策略
```python
# 可以使用 temperature、top-k、top-p 等采样策略
actions = model.generate(
    input_ids,
    images,
    temperature=0.7,
    top_k=50,
)
```

### 劣势

#### 1. 精度损失
```python
# 256 bins 的精度
precision = 2.0 / 256 = 0.0078  # 约 0.78%

# 例如：真实动作 = 0.5234
# 离散化后：bin = int((0.5234 + 1) / 2 * 255) = 194
# 恢复后：action = 194 / 255 * 2 - 1 = 0.5216
# 误差：0.5234 - 0.5216 = 0.0018
```

#### 2. 推理速度慢
```python
# 需要自回归生成 7 个 token
for dim in range(7):
    token = model.generate_next_token()  # 7 次前向传播
```

#### 3. 需要修改词表
```python
# 需要选择或添加 256 个动作 token
# 可能与现有 token 冲突
```

---

## 💻 如何实现 CoT-VLA 的方法

### 完整实现示例

```python
# vila_u/model/vila_u_arch.py

class VILAUMetaForCausalLM(ABC):
    def init_vlm(self, config):
        # ... 现有代码 ...

        if getattr(config, "use_action_prediction", False):
            # 设置动作离散化参数
            self.num_action_bins = getattr(config, "num_action_bins", 256)
            self.action_dim = getattr(config, "action_dim", 7)

            # 选择动作 token（使用词表中最不常用的 token）
            vocab_size = self.llm.config.vocab_size
            self.action_token_start = vocab_size - self.num_action_bins
            self.action_token_ids = list(range(
                self.action_token_start,
                vocab_size
            ))

            # 不需要额外的 action_head，直接使用 lm_head

    def tokenize_action(self, action):
        """
        将连续动作 [-1, 1] 转换为 token IDs

        Args:
            action: [B, action_dim] 或 [B, chunk_size, action_dim]
        Returns:
            action_tokens: [B, action_dim] 或 [B, chunk_size, action_dim]
        """
        # 归一化到 [0, 1]
        action_normalized = (action + 1) / 2

        # 离散化到 [0, num_bins-1]
        action_bins = (action_normalized * (self.num_action_bins - 1)).long()
        action_bins = torch.clamp(action_bins, 0, self.num_action_bins - 1)

        # 映射到实际的 token IDs
        action_tokens = action_bins + self.action_token_start

        return action_tokens

    def detokenize_action(self, action_tokens):
        """
        将 token IDs 转换回连续动作 [-1, 1]

        Args:
            action_tokens: [B, action_dim] 或 [B, chunk_size, action_dim]
        Returns:
            action: [B, action_dim] 或 [B, chunk_size, action_dim]
        """
        # 转换为 bins
        action_bins = action_tokens - self.action_token_start

        # 反归一化
        action_normalized = action_bins.float() / (self.num_action_bins - 1)
        action = action_normalized * 2 - 1

        return action

    @torch.no_grad()
    def predict_action_discrete(self, image, instruction, image_processor=None):
        """
        使用离散化方法预测动作
        """
        # 1. 预处理图像
        device = next(self.parameters()).device
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(device, dtype=self.dtype)

        # 2. 构建提示（添加动作 token 占位符）
        from vila_u.constants import DEFAULT_IMAGE_TOKEN

        # 为每个动作维度添加占位符
        action_placeholders = " ".join([f"<action_{i}>" for i in range(self.action_dim)])
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{instruction}\nAction: {action_placeholders}"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # 3. 自回归生成动作 token
        action_tokens = []

        for dim in range(self.action_dim):
            # 前向传播
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor if dim == 0 else None,  # 只在第一次传图像
                output_hidden_states=True,
                return_dict=True,
            )

            # 获取最后一个位置的 logits
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # 只考虑动作 token 的概率
            action_logits = logits[:, self.action_token_ids]  # [B, 256]

            # 选择概率最高的 token（贪心）
            action_token = torch.argmax(action_logits, dim=-1)  # [B]
            action_token = action_token + self.action_token_start

            action_tokens.append(action_token)

            # 将预测的 token 加入输入序列
            input_ids = torch.cat([input_ids, action_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=device)
            ], dim=-1)

        # 4. 转换为连续动作
        action_tokens = torch.stack(action_tokens, dim=-1)  # [B, action_dim]
        actions = self.detokenize_action(action_tokens)

        return actions.squeeze(0)  # [action_dim]


# vila_u/train/train_action_prediction_discrete.py

class ActionPredictionTrainerDiscrete:
    def compute_loss(self, model, batch):
        """使用离散化方法计算损失"""
        observations = batch['observations'].to(self.device)
        instructions = batch['instructions']
        action_labels = batch['action_labels'].to(self.device)  # [B, 10, 7]

        # 1. 将动作标签离散化为 token
        action_token_labels = model.tokenize_action(action_labels)  # [B, 10, 7]

        # 2. 构建输入序列（包含动作 token）
        from vila_u.constants import DEFAULT_IMAGE_TOKEN

        prompts = []
        for inst in instructions:
            # 添加动作 token 占位符
            action_placeholders = " ".join([f"<action_{i}>" for i in range(7)])
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{inst}\nAction: {action_placeholders}"
            prompts.append(prompt)

        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # 3. 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=observations,
            output_hidden_states=True,
            return_dict=True,
        )

        # 4. 提取动作位置的 logits
        logits = outputs.logits  # [B, seq_len, vocab_size]

        # 找到动作 token 的位置（最后 7 个位置）
        action_logits = logits[:, -7:, :]  # [B, 7, vocab_size]

        # 只关注动作 token 的概率
        action_logits = action_logits[:, :, model.action_token_ids]  # [B, 7, 256]

        # 5. 计算交叉熵损失
        # 只使用第一步的动作（如果要预测10步，需要修改）
        action_token_labels_first = action_token_labels[:, 0, :]  # [B, 7]

        loss = F.cross_entropy(
            action_logits.reshape(-1, model.num_action_bins),
            (action_token_labels_first - model.action_token_start).reshape(-1)
        )

        return loss, action_logits
```

---

## 🎯 实现建议

### 方案1：保持当前的连续回归方法
**适用场景**：
- 需要快速推理（实时控制）
- 任务相对简单
- 快速原型验证

**优点**：
- 推理快（1次前向传播）
- 精度高（无离散化误差）
- 实现简单

**缺点**：
- 不能充分利用 LLM 预训练能力
- 训练可能不如分类稳定

### 方案2：切换到离散化分类方法（CoT-VLA）
**适用场景**：
- 想充分利用 LLM 能力
- 有足够的训练数据
- 可以接受较慢的推理速度

**优点**：
- 利用 LLM 的 next-token prediction 能力
- 训练更稳定
- 统一的建模框架

**缺点**：
- 推理慢（7次前向传播）
- 精度损失（256 bins）
- 实现复杂

### 方案3：混合方法
```python
# 第一阶段：使用离散化训练（利用 LLM 能力）
model_discrete.train()

# 第二阶段：添加连续回归头进行微调（提高精度）
model.add_continuous_head()
model.finetune()
```

---

## 📝 总结

| 特性 | 当前实现（连续回归） | CoT-VLA（离散分类） |
|------|-------------------|-------------------|
| 动作表示 | 连续值 [-1, 1] | 离散 token |
| 损失函数 | L1 Loss | Cross-Entropy |
| 推理速度 | 快（1次） | 慢（7次） |
| 精度 | 高（无离散化） | 中（256 bins） |
| 训练稳定性 | 中等 | 高 |
| 利用预训练 | 部分 | 充分 |
| 实现复杂度 | 低 | 高 |

**建议**：
1. **短期**：保持当前的连续回归方法，先验证基本效果
2. **中期**：如果效果不理想，尝试离散化方法
3. **长期**：探索混合方法，结合两者优势
