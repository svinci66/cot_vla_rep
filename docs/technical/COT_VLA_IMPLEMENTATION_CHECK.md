# CoT-VLA / VILA-U 当前实现检查报告

本文档从 `scripts/train/train_phase4_visual_cot_8gpu.sh` 入口出发，按 `COT_VLA_PAPER_VILA_U_MODIFICATIONS.md` 的检查项追踪当前代码从训练脚本、数据集、collator、模型 forward/loss、推理到在线评估的完整链路。

为便于迁移到其他代码版本，本文档只引用文件和功能模块，不使用具体行号。

## 1. 训练入口链路

入口脚本：

- `scripts/train/train_phase4_visual_cot_8gpu.sh`
- `scripts/train/train_action_prediction.sh`
- `vila_u/train/train_action_prediction_mem.py`
- `vila_u/train/train_action_prediction_main.py`

当前 Phase 4 入口会设置以下关键开关：

- `USE_HYBRID_ATTENTION=True`
- `USE_VISUAL_COT=True`
- `USE_VISUAL_COT_LOSS=True`
- `SUBGOAL_MIN_OFFSET=1`
- `SUBGOAL_MAX_OFFSET=ACTION_CHUNK_SIZE`，默认等于 `10`
- `SUBGOAL_SAMPLING_STRATEGY=uniform`
- `ATTN_IMPLEMENTATION=eager`
- `ACTION_CHUNK_SIZE=10`
- `ACTION_DIM=7`

结论：

- Phase 4 训练入口确实走到了当前主训练实现 `train_action_prediction_main.py`。
- 入口默认开启 Visual CoT、Visual CoT loss、hybrid attention。
- hybrid attention 强制使用 eager 4D mask，这是功能上可行但效率上偏重的选择。
- 当前默认数据源只指向 LIBERO-Goal，因此更接近“下载的已预训练 base VILA-U 权重直接在目标 robot demonstrations 上 fine-tune”的设置，而不是论文完整 CoT-VLA pretraining。

## 2. 数据集与样本构造

相关文件：

- `vila_u/data/libero_dataset_v2.py`
- `vila_u/train/train_action_prediction_main.py`

当前数据集实现：

- 读取 LIBERO `.hdf5` 演示数据。
- 使用 `obs/agentview_rgb` 作为当前视觉观测。
- 根据语言 instruction 构造文本输入。
- 返回 `action_labels`，形状为 `[action_chunk_size, 7]`。
- 在 `USE_VISUAL_COT=True` 时返回未来帧 `subgoal_images`。
- 子目标帧 offset 支持 `uniform` 或 `fixed`，默认在 `[SUBGOAL_MIN_OFFSET, SUBGOAL_MAX_OFFSET]` 内随机采样。
- 图像通过 VILA-U image processor 处理为 `256 x 256`。

与论文一致的部分：

- 支持 `(l, s_t, s_{t+n})` 的子目标图像样本。
- 支持 `(l, s_t, s_{t+n}, a_t...a_{t+m})` 的动作训练样本。
- action dim 为 `7`。
- action chunk size 默认为 `10`。
- 子目标 horizon 支持区间采样。

偏差和风险：

- 当前只接入 LIBERO robot demonstrations，没有接入 action-less videos，例如 EPIC-KITCHENS 或 Something-Something V2。
- action-less video 只用于 visual loss 的训练机制没有形成完整数据混合管线。
- pause removal 会先过滤动作序列，再用原始 timestep 取观测和未来 subgoal；这会让 action chunk 的时间间隔和图像 subgoal offset 的语义不完全一致。
- action 只做 `[-1, 1]` clip，没有按训练集动作分布计算 1%-99% percentile。

## 3. Action Tokenization

相关文件：

- `vila_u/utils/action_tokenizer.py`
- `vila_u/constants.py`
- `vila_u/train/train_action_prediction_main.py`

当前实现：

- 使用 `256` 个 action bins。
- 将连续 action 离散化为 action token。
- action token id 由 tokenizer 词表尾部候选 token 选出。
- hybrid attention 训练时，输入 action block 使用 action slot token，占位位置的 label 仍是真实 action token。
- loss 只在 256 个候选 action token 上计算 cross entropy。
- 推理时将预测 token bins 反离散化回连续 action。

与论文一致的部分：

- 使用 256 bins。
- 每个 7-DoF action 维度对应一个 token。
- 使用 action token cross entropy，而不是连续动作 L1/MSE 回归。
- 复用文本 tokenizer 词表中的 token 作为 action bin token。

偏差和风险：

- 论文要求 bin 范围来自训练集动作分布的 1st-99th percentile；当前使用全局固定 `ACTION_MIN=-1.0`、`ACTION_MAX=1.0`。
- 论文描述复用 256 个低频 token；当前使用“词表尾部普通 token”作为启发式近似，并没有真实统计 token frequency。
- percentile 参数没有保存进 checkpoint config，跨数据集或跨 embodiment 时 detokenization 可能不一致。

## 4. Hybrid Attention

相关文件：

- `vila_u/utils/hybrid_attention.py`
- `vila_u/train/train_action_prediction_main.py`
- `vila_u/model/vila_u_arch.py`

当前实现：

- 构造 4D additive attention mask。
- 默认对有效 token 使用 causal attention。
- 最后 `action_chunk_size * action_dim` 个有效 token 被识别为 action slots。
- action slots 可以 full-attend 到同一样本内所有有效 token。
- 非 action token 仍保持 causal mask。

与论文一致的部分：

- 图文区域保持 causal attention。
- action token 区域使用 full attention。
- 动作 token 之间可以相互交互。

偏差和风险：

- 当前 full attention 只将 action query 升级为可看全部有效 token；非 action token 不能看未来 action token。这符合“保护图文自回归”的目标。
- 4D eager mask 会禁用 flash attention 路径，训练和推理都会有明显效率损失。
- mask 构造中存在按 batch 循环和方阵分配，长序列和大 batch 下开销较高。

## 5. Visual CoT 训练

相关文件：

- `vila_u/train/train_action_prediction_main.py`
- `vila_u/model/vila_u_arch.py`
- `vila_u/model/multimodal_encoder/rqvaesigliptransformer_encoder.py`
- `vila_u/model/multimodal_encoder/rqvaesigliptransformer/rqtransformer/modeling_rqtransformer.py`

当前实现：

- 训练时先编码 GT future-frame subgoal image，得到 subgoal embeddings 和 residual codes。
- 将 subgoal embeddings 插入到 action block 前。
- 对 action block 计算离散 action cross entropy。
- 通过 RQTransformer 对 subgoal residual codes 计算 visual CoT cross entropy。
- 总 loss 为 `action_loss_weight * action_loss + visual_loss_weight * visual_loss`。

与论文一致的部分：

- 同时训练动作 token loss 和视觉 token loss。
- visual loss 使用 VILA-U 的 RQTransformer/residual code 预测形式。
- 动作预测以当前观测、语言和子目标 embedding 为条件。
- 训练时使用 GT future frame 作为子目标监督。

偏差和风险：

- 训练动作时使用的是 GT subgoal embedding 条件动作，而不是模型在线生成的 subgoal embedding 条件动作；这会带来 train/inference exposure gap。
- 当前没有 action-less video 数据，因此 visual CoT loss 只来自 robot demonstrations。
- 当前 implementation 没有显式实现论文图中 `[x]`、`[theta]`、`[g]` 这类独立特殊 token，而是用 action slot token 占位。
- visual loss 使用的 RQTransformer 是否可训练取决于冻结策略；当前默认配置下存在和论文不一致的风险，见下一节。

## 6. 可训练参数与冻结策略

相关文件：

- `scripts/train/train_action_prediction.sh`
- `vila_u/train/train_action_prediction_main.py`
- `vila_u/model/vila_u_arch.py`

当前训练脚本默认：

- `tune_language_model=True`
- `tune_mm_projector=True`
- `tune_vision_tower=False`

当前模型训练设置：

- LLM backbone 可训练。
- multimodal projector 可训练。
- vision tower 冻结。
- RQVAE 明确设为 eval。
- RQTransformer 的 `requires_grad` 跟随 `tune_vision_tower`。

与论文一致的部分：

- 冻结 vision tower。
- 优化 LLM backbone。
- 优化 projector。

与论文不一致或需确认的部分：

- 论文说优化 depth transformer，同时冻结 vision tower。
- 当前代码中 RQTransformer/depth transformer 被包含在 vision tower 下，且默认跟随 `tune_vision_tower=False` 冻结。
- 如果当前 RQTransformer 就是论文所说 depth transformer，则 Phase 4 默认训练没有优化 depth transformer，这和论文不一致。

建议：

- 将 RQVAE/SigLIP encoder 的冻结和 RQTransformer/depth transformer 的可训练状态拆开。
- 新增独立开关，例如 `TUNE_DEPTH_TRANSFORMER=True`，默认训练 RQTransformer，但保持 RQVAE/SigLIP vision encoder 冻结。

## 7. 推理路径

相关文件：

- `vila_u/model/vila_u_arch.py`
- `vila_u/eval/trajectory_generator.py`
- `scripts/check_phase4_oracle_subgoal.py`
- `scripts/visualize_phase4_subgoals.py`
- `scripts/vila_zmq_model_worker.py`

当前推理支持三种模式：

- `none`：不使用 subgoal，直接基于当前图像和语言预测动作 token。
- `oracle`：使用 GT future-frame subgoal image 编码后的 embedding 条件动作。
- `generated`：先生成 subgoal embedding，再条件动作预测。

与论文一致的部分：

- 存在“生成子目标图像/embedding -> 条件动作预测”的接口。
- `TrajectoryGenerator` 支持 action chunk 队列，能够执行完整 chunk 后再重新预测。
- ZMQ 在线评估已支持 action chunk 队列：worker 在队列为空时预测一个 chunk，后续环境请求直接弹出队列动作。
- 推理生成 subgoal 时使用 causal attention。
- 动作预测阶段使用 hybrid attention 和 full action block。

偏差和风险：

- ZMQ 环境侧仍是逐步请求，但 worker 侧已缓存 chunk；这降低了重复推理开销，并更接近论文 Algorithm 1 的 chunk 执行方式。
- generated-subgoal 模式仍然需要在每个新 chunk 前生成一次 subgoal；即使有 action queue，单个 task 的在线评估仍明显慢于 `none` 模式。
- generated subgoal 推理返回的是生成 embedding 条件动作；只有 `return_subgoal=True` 时才 decode image，常规控制路径不会真的使用 decoded image。
- 当前 generated subgoal 过程没有使用 KV cache，每生成一个图像 token 都重新跑完整 LLM。
- 当前 debug 模式可以保存每个 chunk 的 action、action token ids、predicted bins、generated subgoal codes 和 decoded subgoal image，用于定位 token 预测和连续动作解码问题。

## 8. 保存与加载

相关文件：

- `vila_u/model/vila_u_arch.py`
- `vila_u/model/builder.py`
- `vila_u/train/train_action_prediction_main.py`

当前保存：

- 保存 LLM。
- 保存 vision tower。
- 保存 mm_projector。
- 保存 config。
- regression mode 下保存 `action_head.bin`。
- discrete action mode 依赖 config 中的 `action_token_ids`、`action_slot_token_id` 和 LLM/mm_projector/vision_tower 权重。

当前加载：

- `load_pretrained_model` 从 checkpoint config 恢复 VILA-U。
- 设置 `resume_path` 后由各 builder 加载对应子模块。
- 推理依赖 config 中保留的 discrete action 相关字段。

风险：

- 如果 future 改为 percentile bin，需要将每个 action dimension 的 bin min/max 或 percentile edges 保存进 config。
- 如果新增独立 depth transformer 训练开关，需要确保保存/加载路径包含对应权重。

## 9. 功能覆盖矩阵

| 检查项 | 当前状态 | 说明 |
|---|---|---|
| Phase 4 入口开启 Visual CoT | 已实现 | 默认开启 `USE_VISUAL_COT` 和 `USE_VISUAL_COT_LOSS` |
| 7-DoF action chunk | 已实现 | 默认 `10 x 7` |
| 离散 action token | 已实现 | 256 bins，cross entropy |
| 1%-99% percentile bins | 未实现 | 当前固定 `[-1, 1]` |
| 低频 tokenizer token | 部分实现 | 当前用词表尾部启发式 |
| subgoal future-frame 采样 | 已实现 | 支持 uniform/fixed |
| visual token loss | 已实现 | 基于 RQTransformer residual codes |
| hybrid attention | 已实现 | 4D eager mask |
| action-less video 预训练 | 未实现 | 仅 LIBERO robot demonstrations |
| depth transformer 可训练 | 可能未实现 | 默认随 vision tower 冻结 |
| generated subgoal 推理 | 已实现 | 但效率仍较低 |
| chunk 级闭环执行 | 部分实现 | `TrajectoryGenerator` 和 ZMQ worker 支持 chunk queue |
| 在线 action token/bin debug | 已实现 | debug 模式可保存 predicted bins 和 token ids |

## 10. 效率问题与优化建议

优先级从高到低：

1. 优化 generated subgoal 推理。
   当前每生成一个图像 token 都重新跑完整 LLM，并且不使用 KV cache。建议改成标准自回归 cache 路径，至少缓存 prompt 和已生成 token 的 KV。

2. 避免在线评估每步重新生成 subgoal。
   ZMQ worker 已维护 action queue：一次请求生成 action chunk，后续请求直接弹出队列动作，队列空了再请求模型。该优化应保留。后续重点不再是“每步重复动作预测”，而是降低每个 generated subgoal chunk 的生成成本。

3. 拆分 vision tower 冻结和 depth transformer 训练。
   保持 RQVAE/SigLIP encoder 冻结，但允许 RQTransformer/depth transformer 训练，避免 visual loss 只能更新 LLM/projector。

4. 向量化 hybrid attention mask 构造。
   当前 mask 构造有 Python batch 循环和完整 `[B, 1, L, L]` 分配。短期可以缓存固定形状 causal mask；中期可以构造更少的覆盖 mask；长期可以接入支持 block mask 的 attention backend。

5. 减少数据读取开销。
   当前每个样本都打开 HDF5 文件并做 PIL/image processor 预处理。建议按 worker 缓存 HDF5 handle，或预导出 LMDB/WebDataset/缓存后的 tensor。

6. action token logits 已经做了局部投影。
   当前只对 256 个 action token 计算 logits，这是正确的效率优化，应保留。

7. 避免不必要的 decoded subgoal image。
   控制路径只需要 subgoal embeddings；除可视化外不应 decode image。

## 11. 建议修复顺序

建议将修复目标分成两层：先对齐论文的 no-pretraining ablation，再追完整 CoT-VLA pretraining。否则当前失败无法判断是缺少预训练造成，还是核心实现偏差造成。

这里的“pretraining”特指 CoT-VLA 论文中的机器人/视频预训练阶段，即 OpenX robot demonstrations + action-less videos。当前使用的 VILA-U 本身已经带有原始多模态预训练权重，不是随机初始化模型。

### 11.1 先对齐论文 no-pretraining 设置

这一层不要求接入 OpenX、EPIC-KITCHENS 或 Something-Something V2，只要求在目标 robot demonstrations 上 fine-tune 时，CoT-VLA 方法本身和论文一致。

1. 修正 action discretization：实现训练集每维 1%-99% percentile 统计、保存 bin edges、训练和推理使用同一套 edges。
2. 拆出 depth transformer/RQTransformer 的独立训练开关：保持 RQVAE/SigLIP vision encoder 冻结，但允许 depth transformer 训练。
3. 确认 visual CoT loss 能更新 subgoal generation 相关参数，而不是只更新 LLM/projector。
4. 使用在线 debug 中保存的 predicted bins/action token ids 回归验证动作尺度，防止 token 解码和训练统计不一致。
5. 保留 ZMQ action queue，并把 task-level summary 路径、debug 日志路径等评估输出做成可配置，避免不同实验覆盖或误读结果。
6. 补充 tests：percentile tokenizer、config 保存/加载 bin edges、depth transformer requires_grad、ZMQ chunk queue、action token/bin debug。

完成这一层后，当前训练才可以较合理地和论文的 no-pretraining ablation 对比。

### 11.2 再追完整 CoT-VLA pretraining

这一层用于接近论文完整方法，而不是只做目标任务 fine-tuning。

1. 增加 action-less video 数据接口，使 EPIC-KITCHENS、Something-Something V2 或同类 captioned videos 能只进入 visual/subgoal loss。
2. 增加 OpenX-style robot demonstration 数据接口，使 robot demonstrations 同时提供 visual loss 和 action loss。
3. 支持按 dataset 配置 subgoal horizon 范围，避免所有数据集都使用同一个固定 offset 范围。
4. 优化 generated subgoal 自回归缓存，降低每个 chunk 的 subgoal 生成成本。
5. 补充 generated-vs-oracle subgoal 评估，用于区分“动作策略问题”和“视觉目标生成问题”。

## 12. 测试结果

已使用 `/home/ubuntu/Downloads/vila_env_fixed/bin/python` 运行轻量 Phase 4 测试：

```text
tests/test_phase4_visual_cot.py
```

测试通过，覆盖：

- causal 4D attention mask。
- subgoal embedding 插入 action block 前。
- visual CoT loss 基本形状与 offset-code 路径。

## 13. 在线隔离实验结论

在 `libero_goal` task 0 上做了两个隔离实验，用于判断当前失败来自 subgoal 质量、在线评估效率，还是动作表示本身。

实验设置：

- `generated` 模式：使用 generated subgoal + action queue。
- `none` 模式：不使用 subgoal，只用当前观测和语言预测动作。
- 两种模式都保存逐步 action；新的 debug 模式还保存每个 chunk 的 predicted bins 和 action token ids。

观察结果：

- `generated` 模式在完整 300 step 在线评估中失败，reward 为 0。
- `none` 模式在完整 300 step 在线评估中也失败，reward 为 0。
- `none` 模式耗时约数秒，`generated` 模式耗时约数百秒到十几分钟；因此 generated subgoal 是主要效率瓶颈。
- 即使不使用 generated subgoal，task 0 仍失败，说明失败不只是 subgoal 生成质量问题，基础 action prediction 或动作解码也存在问题。
- 在线 rollout 的平移动作范数显著小于原始 LIBERO demo；模型输出的动作更接近“微小移动/原地调整”，不足以完成抽屉打开。
- debug 保存的 predicted bins 大量集中在中间 bin 附近，对应固定 `[-1, 1]` 解码后的近零动作。
- 原始 LIBERO demo 中不同动作维度的分布差异明显，尤其平移、旋转和 gripper 的尺度不同；当前统一固定 `[-1, 1]` bins 会放大动作表示不匹配风险。

由实验支持的修复判断：

- action queue 已解决“每步重复预测 chunk”的主要低效问题，应继续保留。
- generated subgoal 仍需要 KV cache 或更轻量生成路径，否则在线评估会非常慢。
- 当前最高优先级仍是 action discretization：需要按训练数据统计每维 percentile/bin edges，并在训练、保存、加载、推理中使用同一套动作解码参数。
- 在 action discretization 修复前，仅优化在线评估效率不太可能提升成功率。

## 14. 论文 No-Pretraining 与当前训练的关系

论文中的 no-pretraining ablation 不是从随机初始化训练，也不是去掉 CoT-VLA 方法本身。它表示：从已经完成原始 VILA-U 多模态预训练的 base VILA-U 出发，跳过 OpenX + action-less videos 的 CoT-VLA 机器人/视频预训练，直接在目标 robot demonstrations 上做 task-specific fine-tuning。

论文 no-pretraining 仍然应包含完整 CoT-VLA 方法设置：

- visual CoT subgoal generation loss。
- action token prediction loss。
- hybrid attention。
- action chunking。
- LLM backbone、projector、depth transformer 可训练。
- vision tower 冻结。
- 每维 1%-99% percentile action bins。
- 正确保存并加载 action tokenization/detokenization 参数。

当前训练与论文 no-pretraining 的相似点：

- 从下载的、已有原始多模态预训练权重的 base VILA-U checkpoint 出发。
- 默认数据源只包含目标 LIBERO robot demonstrations。
- 没有 OpenX CoT-VLA 机器人 demonstration 预训练。
- 没有 EPIC-KITCHENS / Something-Something V2 action-less video 视觉 CoT 预训练。

当前训练与论文 no-pretraining 的关键差异：

- action tokenizer 使用固定 `[-1, 1]` bins，而不是训练集每维 percentile bins。
- depth transformer/RQTransformer 可能随 vision tower 一起被冻结。
- visual CoT 只来自 LIBERO robot demonstrations，缺少 action-less video 预训练数据。
- generated subgoal 在线质量较差，且和训练时 GT subgoal 条件动作之间存在 exposure gap。
- 当前在线评估已通过 action queue 降低重复预测开销，但 generated subgoal 每个 chunk 仍很慢。

因此，当前结果不能简单解释为“论文 no-pretraining 表现”。更准确的判断是：当前是“用已有 VILA-U 预训练权重，只在 LIBERO 上 fine-tune 的弱化 CoT-VLA 实现”。在 action discretization、depth transformer 训练和 visual CoT 更新路径修正前，LIBERO-Goal 在线成功率为 0 不能只归因于缺少 OpenX/action-less video 这一级 CoT-VLA 预训练。
