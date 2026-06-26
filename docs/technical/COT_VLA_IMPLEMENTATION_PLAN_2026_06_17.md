# CoT-VLA Visual CoT Implementation Plan

本文档是后续实现和验证 Visual CoT 的执行准则。依据包括：

- 本地论文 PDF：`/home/ubuntu/Downloads/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.pdf`
- 现有检查文档：`docs/technical/COT_VLA_IMPLEMENTATION_CHECK.md`
- 在线评测文档：`docs/eval/LIBERO_ZMQ_ONLINE_EVAL.md`
- 当前主训练入口：`scripts/train/train_action_prediction.sh`
- 当前训练实现：`vila_u/train/train_action_prediction_main.py`
- 当前生成式 subgoal 推理：`vila_u/model/vila_u_arch.py`
- 当前 ZMQ 在线 worker：`scripts/vila_zmq_model_worker.py`
- 当前 Phase4 单元测试：`tests/test_phase4_visual_cot.py`

本文档的核心原则：先保护当前 action-only 基线，再逐步打开 Visual CoT。不要一上来直接跑 generated-subgoal 全量在线评测。

## 1. 当前基线

### 1.1 当前最可信的 action-only 结果

当前 action-only 全 task、chunk=10 的在线基线应优先参考 6_16 alltask 约 epoch40 的结果，而不是 ckp60：

| checkpoint | online episodes | success |
| --- | ---: | ---: |
| `checkpoints/6_16_alltask` epoch40 左右 | 500 | 392/500 = 78.4% |
| `checkpoints/6_16_alltask/ckp60` | 500 | 380/500 = 76.0% |

ckp60 比 epoch40 下降，说明当前任务里“训练更久”不等于“在线更好”。后续 checkpoint 选择必须以在线成功率为准，offline token accuracy 只能作为诊断指标。

### 1.2 当前 action-only 推荐配置

action-only 训练时：

- `ACTION_CHUNK_SIZE=10`
- `USE_ACTION_PERCENTILE_BINS=True`
- `ACTION_BIN_LOW_PERCENTILE=1.0`
- `ACTION_BIN_HIGH_PERCENTILE=99.0`
- `USE_HYBRID_ATTENTION=True`
- `RANK_SLICE_AFTER_SHUFFLE=True`
- `RANK_PARAMETER_CHECK=True`
- `TUNE_DEPTH_TRANSFORMER=False`

这里 `TUNE_DEPTH_TRANSFORMER=False` 是 action-only 的关键保护项。因为 action-only 没有 visual loss，depth transformer 如果保持 trainable，会出现 DDP unused parameter 或 rank 参数不一致问题。

## 2. 论文方法拆解

论文 CoT-VLA 的关键不是“多加一张图”这么简单，而是两个阶段：

1. 视觉推理：根据当前图像 `s_t` 和语言 `l`，预测未来子目标图像 `s_{t+n}`。
2. 动作生成：根据当前图像、语言和子目标图像，预测动作序列 `a_t ... a_{t+m}`。

论文训练目标是：

- `L_visual`：预测 subgoal image 的 residual visual tokens。
- `L_action`：预测 action tokens。
- 总 loss：`L_action + L_visual`。

论文中的动作表示：

- 每个动作 7 维。
- 每维离散化成 256 个 bins。
- bins 按训练集动作分布 1%-99% percentile 划分。
- action chunk size 使用 10。
- action tokens 使用 full attention，图像和文本 token 使用 causal attention。

论文中的视觉表示：

- 使用 VILA-U 256x256 图像。
- 每张图像编码为 `16 x 16 x 4` residual tokens。
- 每个 spatial token 有 depth=4 的 residual code。
- depth transformer 负责预测 residual tokens。

论文训练策略背景：

- 预训练阶段使用 robot demonstrations 和 action-less videos。
- action-less videos 只提供语言和图像序列，不提供动作，用于增强 subgoal image generation。
- 下游适配时 fine-tune LLM、projector、depth transformer，冻结 vision tower。

本项目当前执行策略：

- 不进行 OpenX / EPIC-KITCHENS / Something-Something V2 预训练。
- 不新增 action-less video 数据管线。
- 直接从已有 VILA-U 7B 256 基座和当前 action-only 训练经验出发，在 LIBERO demonstrations 上做 Visual CoT fine-tune。
- 评估重点放在 LIBERO online success，而不是复现论文完整预训练流程。

论文部署策略：

- 每次在线控制先生成 subgoal image。
- 再生成一段 action chunk。
- 执行完整 chunk 后重新观测、重新生成 subgoal 和动作。

## 3. 当前仓库已经具备的能力

### 3.1 训练入口

`scripts/train/train_action_prediction.sh` 已经支持以下 Phase4 开关：

- `USE_VISUAL_COT`
- `USE_VISUAL_COT_LOSS`
- `TUNE_DEPTH_TRANSFORMER`
- `VISUAL_LOSS_WEIGHT`
- `ACTION_LOSS_WEIGHT`
- `SUBGOAL_MIN_OFFSET`
- `SUBGOAL_MAX_OFFSET`
- `SUBGOAL_SAMPLING_STRATEGY`
- `USE_ACTION_PERCENTILE_BINS`
- `USE_HYBRID_ATTENTION`
- `RANK_SLICE_AFTER_SHUFFLE`
- `SAMPLER_DEBUG`
- `RANK_PARAMETER_CHECK`

### 3.2 训练实现

`vila_u/train/train_action_prediction_main.py` 当前已经实现：

- LIBERO 样本中返回 `subgoal_images`。
- 训练时编码 GT future-frame subgoal image。
- 将 subgoal embeddings 插入 action block 前。
- action loss 使用离散 token cross entropy。
- visual CoT loss 通过 `compute_visual_cot_loss` 调用 RQTransformer。
- percentile action bin edges 会写入 checkpoint config。
- `TUNE_DEPTH_TRANSFORMER=True` 时可以只训练 depth transformer，同时保持 RQVAE/SigLIP 冻结。
- `rank_parameter_check` 可在 epoch 末检查 DDP 各 rank 参数是否一致。

### 3.3 推理实现

`vila_u/model/vila_u_arch.py` 当前已经实现：

- `generate_visual_cot_subgoal(...)`
- `predict_action_with_generated_subgoal(...)`

`scripts/vila_zmq_model_worker.py` 当前在线支持：

- `--subgoal-mode none`
- `--subgoal-mode generated`
- `--replan-every-step`
- `--debug-jsonl`

注意：当前正式 worker 没有 `oracle` 模式。oracle subgoal 目前只能作为离线或临时诊断逻辑，不应写进正式在线流程，除非后续明确补 worker 支持。

### 3.4 已有测试

`tests/test_phase4_visual_cot.py` 已经覆盖：

- causal attention mask。
- subgoal embedding 插入 action block 前。
- visual CoT loss shape。
- visual CoT loss offset-code 路径。
- depth transformer 可以独立于 RQVAE 训练。
- visual loss 会更新 RQTransformer 参数。
- freeze patch 保持 depth transformer train mode。

后续修改 CoT 相关代码后，必须先跑这组测试。

## 4. 当前缺口和风险

### 4.1 明确不做 action-less video 预训练

论文完整 CoT-VLA 使用 OpenX robot demonstrations 加 EPIC-KITCHENS / Something-Something V2 action-less videos。当前策略明确不走这条路线，原因是当前首要问题不是扩大预训练数据，而是验证 Visual CoT 在现有 LIBERO 任务上是否真的带来在线收益。

因此后续执行只围绕 LIBERO demonstrations：

- LIBERO-only 的 Visual CoT training 是否能稳定训练。
- generated subgoal 是否能在线带来收益。
- visual loss 和 action loss 是否能同时工作。
- generated subgoal 的额外推理成本是否可接受。

不要把“缺少 action-less video”作为当前失败的第一解释。先用同一批 LIBERO demonstrations 把 action-only、GT future-frame subgoal 条件训练、generated-subgoal 在线推理这三件事分清楚。

### 4.2 GT subgoal 训练和 generated subgoal 在线有 exposure gap

训练动作时使用 GT future-frame subgoal embedding；在线 generated 模式使用模型自己生成的 subgoal embedding。两者分布可能不同。

因此必须分阶段判断：

- action-only 是否正常。
- GT subgoal 条件下动作预测是否正常。
- generated subgoal 本身是否合理。
- generated subgoal 在线是否真的提升成功率。

不能只看训练 loss 下降就认为 CoT 生效。

### 4.3 generated subgoal 推理慢

论文报告：生成 256 个 image tokens 后再预测 action，在 chunk=10 下平均约 7 倍变慢。

当前代码里 `generate_visual_cot_subgoal` 每个 visual token 都重新跑一次 LLM，且 `use_cache=False`，实际可能更慢。因此 generated-subgoal 在线评测必须先小规模测试，不要直接跑 500 demos。

### 4.4 DDP unused parameter 风险

action-only：

- `USE_VISUAL_COT=False`
- `USE_VISUAL_COT_LOSS=False`
- `TUNE_DEPTH_TRANSFORMER` 必须设为 `False`

Visual CoT：

- `USE_VISUAL_COT=True`
- `USE_VISUAL_COT_LOSS=True`
- `TUNE_DEPTH_TRANSFORMER=True`

只有 visual loss 打开时，depth transformer 才应参与 loss。否则 DDP 会认为这部分 trainable 参数没被用到。

### 4.5 在线评测是最终标准

offline 指标用途：

- 快速确认 checkpoint 是否加载正确。
- 看 token accuracy、MAE、gripper close recall、transition recall。
- 做 per-task/per-demo/semantic sensitivity 诊断。

online 指标用途：

- 决定 checkpoint 是否真正有效。
- 决定是否继续训练、早停、切换策略。

当前经验已经证明：loss 更低或 epoch 更多，不必然带来 online success 更高。

## 5. 执行路线

### Stage 0: 保护 action-only 基线

目标：

- 任何 CoT 修改前，都能复现 action-only 加载、离线评测、在线评测。
- 保证 percentile bins、rank sampler、checkpoint config 没被破坏。

通过标准：

- `TUNE_DEPTH_TRANSFORMER=False` 的 action-only 训练不会 DDP 报错。
- rank sampler 每个 rank 都能看到 50 demos 或全 demos。
- `rank_parameter_check` 通过。
- offline eval 能读到 checkpoint 中保存的 action bin edges。
- online worker `--subgoal-mode none` 正常。

### Stage 1: Phase4 LIBERO-only smoke training

目标：

- 不追求高成功率，只验证 Visual CoT 训练链路真的通。
- 确认 visual loss 非 0。
- 确认 action loss 和 visual loss 都被记录。
- 确认 depth transformer 参与训练。
- 确认 DDP rank 参数一致。

建议先跑 1 epoch、全 task、50 demos、8 卡。

通过标准：

- 训练完成 1 epoch。
- 日志中能看到 `Use Visual CoT Loss: True`。
- 日志中能看到 `Tune Depth Transformer: True`。
- `rank_parameter_check` 通过。
- 没有 DDP unused parameter error。
- 保存出的 checkpoint 包含 `llm/`、`mm_projector/`，并且 depth transformer 相关可训练权重被保存或能从完整目录恢复。

### Stage 2: Phase4 小规模正式训练

目标：

- 在 smoke 通过后，训练一个可用于 offline/online 的 Phase4 checkpoint。
- 不直接大幅拉 epoch。

建议：

- 先 10 或 20 epochs。
- 每 5 epochs 保存轻量 eval checkpoint。
- 优先在线选择 checkpoint，而不是选最后一个。

通过标准：

- 训练过程中 visual loss 没有塌成 NaN。
- action loss 仍能下降。
- epoch 末 rank parameter check 通过。
- 至少保存 2 个 eval checkpoint 用于比较。

### Stage 3: generated subgoal 离线/可视化诊断

目标：

- 在进入在线前确认 generated subgoal 不是空图、纯噪声或明显错位。
- 保存若干 generated subgoal image/codes/debug jsonl。

建议先用：

- 训练集 task1 的 3-5 个 demo。
- 每个 demo 取开头、中段、接近成功前几个时间点。
- 比较当前图像、GT future frame、generated subgoal。

通过标准：

- generated subgoal 能正常生成，不报 shape 或 dtype 错误。
- 生成图像和当前任务有基本相关性。
- 生成耗时可以接受，至少不会卡死在线流程。

### Stage 4: generated subgoal 小规模在线

目标：

- 只测 very small sample，确认在线链路可运行。

推荐顺序：

1. 单 task 3 demos。
2. 单 task 10 demos。
3. 每 task 5 demos。
4. 每 task 10 demos。

通过标准：

- worker 不崩。
- env server 不超时。
- debug jsonl 能写出 action chunks 和 generated codes。
- 成功率不明显低于 action-only 同 checkpoint/同 demos。

### Stage 5: generated subgoal 全量在线

目标：

- 只有 Stage 4 有正向迹象时，才跑 500 demos。

通过标准：

- 至少和 action-only epoch40 基线接近。
- 如果低于 action-only 明显，应先分析 subgoal generation，而不是继续加 epoch。

### Stage 6: 暂不执行预训练扩展

目标：

- 当前阶段不做 OpenX、EPIC-KITCHENS、Something-Something V2。
- 当前阶段不实现 action-less video dataloader。
- 当前阶段不做 robot/action-less mixed pretraining。

只有在以下条件全部满足后，才重新讨论预训练扩展：

- LIBERO-only Visual CoT 训练稳定。
- generated subgoal 在线至少接近 action-only 基线。
- 已确认主要瓶颈是 subgoal generation 泛化，而不是 DDP、action token、gripper、checkpoint 选择或在线环境问题。
- 有足够时间和算力做数据接入、预处理和长训练。

在当前计划里，Stage 6 是“暂停项”，不是后续默认步骤。

## 6. 必跑检查命令

### 6.1 Phase4 单元测试

本地：

```bash
cd /home/ubuntu/sj/cot_vla_rep && \
  PYTHONPATH=/home/ubuntu/sj/cot_vla_rep \
  /home/ubuntu/Downloads/vila_env_fixed/bin/python tests/test_phase4_visual_cot.py
```

服务器：

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && \
  export PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed/bin:$PATH" && \
  export PYTHONPATH="/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep" && \
  python tests/test_phase4_visual_cot.py
```

### 6.2 action-only 离线基线

用于确认当前 checkpoint 的动作预测和 gripper 指标，不用于判断最终成功率。

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && \
  export PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed/bin:$PATH" && \
  export HF_HOME="/data/share/1919650160032350208/sj/hf_cache_shared" && \
  export HF_HUB_OFFLINE=1 && \
  export TRANSFORMERS_OFFLINE=1 && \
  export PYTHONUNBUFFERED=1 && \
  python scripts/run_action_offline_eval_suite.py \
    --model-path ./checkpoints/6_16_alltask \
    --max-demos-per-task 50 \
    --batch-size 8 \
    --num-workers 4 \
    --output-dir outputs/offline_action_eval
```

## 7. Phase4 训练命令

### 7.1 8 卡 smoke train

先跑这个。不要直接跑 60 epoch。

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && \
  export PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed/bin:$PATH" && \
  export HF_HOME="/data/share/1919650160032350208/sj/hf_cache_shared" && \
  export HF_ENDPOINT="https://hf-mirror.com" && \
  export PYTHONUNBUFFERED=1 && \
  export CONDA_ENV_NAME="vila_env_fixed" && \
  which python && \
  which torchrun && \
  python -c "import sys; print(sys.executable)" && \
  OUTPUT_DIR="./checkpoints/vila-u-cot-libero-goal-10task-50demo-8gpu-pergpu16-lr-2e-5-chunk10-phase4-smoke" \
  SINGLE_GPU_MODE=False \
  NUM_GPUS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NUM_EPOCHS=1 \
  LEARNING_RATE=2e-5 \
  LR_SCHEDULER_TYPE=constant \
  WARMUP_RATIO=0.0 \
  BATCH_SIZE=128 \
  ACTION_CHUNK_SIZE=10 \
  MAX_DEMOS_PER_TASK=50 \
  USE_ACTION_PERCENTILE_BINS=True \
  ACTION_BIN_LOW_PERCENTILE=1.0 \
  ACTION_BIN_HIGH_PERCENTILE=99.0 \
  USE_VISUAL_COT=True \
  USE_VISUAL_COT_LOSS=True \
  TUNE_DEPTH_TRANSFORMER=True \
  TUNE_VISION_TOWER=False \
  TUNE_LANGUAGE_MODEL=True \
  TUNE_MM_PROJECTOR=True \
  USE_HYBRID_ATTENTION=True \
  VISUAL_LOSS_WEIGHT=1.0 \
  ACTION_LOSS_WEIGHT=1.0 \
  XYZ_LOSS_WEIGHT=1.5 \
  GRIPPER_CLOSE_LOSS_WEIGHT=2.0 \
  GRIPPER_TRANSITION_LOSS_WEIGHT=4.0 \
  SUBGOAL_MIN_OFFSET=1 \
  SUBGOAL_MAX_OFFSET=10 \
  SUBGOAL_SAMPLING_STRATEGY=uniform \
  LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS=0 \
  REPORT_TO=none \
  WANDB_DISABLED=true \
  DATALOADER_NUM_WORKERS=8 \
  RANK_SLICE_AFTER_SHUFFLE=True \
  SAMPLER_DEBUG=True \
  RANK_PARAMETER_CHECK=True \
  MASTER_PORT=25005 \
  bash scripts/train/train_action_prediction.sh
```

smoke train 重点看：

- `Use Visual CoT Subgoal Sampling: True`
- `Use Visual CoT Loss: True`
- `Tune Depth Transformer: True`
- `Action Percentile Bins: True (1.0-99.0)`
- `Rank Slice After Shuffle: True`
- 每个 rank 的 sampler debug 是否覆盖 demos。
- epoch 结束是否通过 `rank_parameter_check`。

### 7.2 8 卡 Phase4 小规模正式训练

smoke 通过后再跑。先保存多个 checkpoint 方便在线选点。

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && \
  export PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed/bin:$PATH" && \
  export HF_HOME="/data/share/1919650160032350208/sj/hf_cache_shared" && \
  export HF_ENDPOINT="https://hf-mirror.com" && \
  export PYTHONUNBUFFERED=1 && \
  export CONDA_ENV_NAME="vila_env_fixed" && \
  OUTPUT_DIR="./checkpoints/vila-u-cot-libero-goal-10task-50demo-8gpu-pergpu16-lr-2e-5-chunk10-phase4-20ep" \
  SINGLE_GPU_MODE=False \
  NUM_GPUS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NUM_EPOCHS=20 \
  LEARNING_RATE=2e-5 \
  LR_SCHEDULER_TYPE=constant \
  WARMUP_RATIO=0.0 \
  BATCH_SIZE=128 \
  ACTION_CHUNK_SIZE=10 \
  MAX_DEMOS_PER_TASK=50 \
  USE_ACTION_PERCENTILE_BINS=True \
  ACTION_BIN_LOW_PERCENTILE=1.0 \
  ACTION_BIN_HIGH_PERCENTILE=99.0 \
  USE_VISUAL_COT=True \
  USE_VISUAL_COT_LOSS=True \
  TUNE_DEPTH_TRANSFORMER=True \
  TUNE_VISION_TOWER=False \
  TUNE_LANGUAGE_MODEL=True \
  TUNE_MM_PROJECTOR=True \
  USE_HYBRID_ATTENTION=True \
  VISUAL_LOSS_WEIGHT=1.0 \
  ACTION_LOSS_WEIGHT=1.0 \
  XYZ_LOSS_WEIGHT=1.5 \
  GRIPPER_CLOSE_LOSS_WEIGHT=2.0 \
  GRIPPER_TRANSITION_LOSS_WEIGHT=4.0 \
  SUBGOAL_MIN_OFFSET=1 \
  SUBGOAL_MAX_OFFSET=10 \
  SUBGOAL_SAMPLING_STRATEGY=uniform \
  LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS=5 \
  REPORT_TO=none \
  WANDB_DISABLED=true \
  DATALOADER_NUM_WORKERS=8 \
  RANK_SLICE_AFTER_SHUFFLE=True \
  SAMPLER_DEBUG=False \
  RANK_PARAMETER_CHECK=True \
  MASTER_PORT=25006 \
  bash scripts/train/train_action_prediction.sh
```

如果 20 epoch 的中间 checkpoint 已经在线变差，不要继续盲目拉到 60 epoch。先分析 generated subgoal 和 per-task 失败分布。

## 8. 在线评测流程

### 8.1 worker: action-only / baseline

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && \
  export PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed/bin:$PATH" && \
  export HF_HOME="/data/share/1919650160032350208/sj/hf_cache_shared" && \
  export HF_HUB_OFFLINE=1 && \
  export TRANSFORMERS_OFFLINE=1 && \
  export PYTHONUNBUFFERED=1 && \
  PYTHONPATH=/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep \
  python scripts/vila_zmq_model_worker.py \
    --model-path ./checkpoints/6_16_alltask \
    --bind tcp://127.0.0.1:5555 \
    --device cuda \
    --subgoal-mode none \
    --debug-jsonl outputs/online_debug/action_only_worker.jsonl
```

### 8.2 worker: generated-subgoal

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && \
  export PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed/bin:$PATH" && \
  export HF_HOME="/data/share/1919650160032350208/sj/hf_cache_shared" && \
  export HF_HUB_OFFLINE=1 && \
  export TRANSFORMERS_OFFLINE=1 && \
  export PYTHONUNBUFFERED=1 && \
  PYTHONPATH=/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep \
  python scripts/vila_zmq_model_worker.py \
    --model-path ./checkpoints/vila-u-cot-libero-goal-10task-50demo-8gpu-pergpu16-lr-2e-5-chunk10-phase4-20ep/eval-checkpoint-epoch-5 \
    --bind tcp://127.0.0.1:5555 \
    --device cuda \
    --subgoal-mode generated \
    --cfg 3.0 \
    --debug-jsonl outputs/online_debug/generated_subgoal_worker.jsonl
```

### 8.3 LIBERO env server

env server 必须使用 LIBERO 环境，不要和 VILA 模型放在同一个 Python 进程。

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && \
  export PYTHONNOUSERSITE=1 && \
  export PYTHONPATH="/data/share/1919650160032350208/sj/LIBERO:/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep" && \
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && \
  export MUJOCO_GL=egl && \
  /data/share/1919650160032350208/sj/conda_pkgs/libero/bin/python scripts/libero_zmq_env_server.py \
    --suite libero_goal \
    --task-id 0 \
    --episodes 10 \
    --max-steps 300 \
    --host 127.0.0.1 \
    --port 5555 \
    --output-json outputs/online_phase4_task0_first10.json \
    --output-dir outputs/online_phase4_task0_first10 \
    --save-failures
```

如果服务器上的 LIBERO Python 路径不同，使用之前已经验证过的：

```bash
/home/ubuntu/miniconda3/envs/libero/bin/python
```

或服务器实际 conda env 中的 `python`。关键是：VILA worker 用 `vila_env_fixed`，LIBERO env server 用 LIBERO 环境。

## 9. 评测顺序

Phase4 checkpoint 的评测顺序固定如下：

1. `tests/test_phase4_visual_cot.py`
2. smoke train 1 epoch
3. 检查 smoke 日志中的 visual loss、rank sampler、rank parameter check
4. 小规模正式训练，保存 epoch 5/10/15/20
5. offline action eval，确认 action token 和 gripper 指标没有明显崩
6. generated subgoal 可视化或 debug jsonl
7. task0 前 3 demos generated 在线
8. task0 前 10 demos generated 在线
9. 每 task 5 demos generated 在线
10. 每 task 10 demos generated 在线
11. 全 500 generated 在线

任何一步失败，都不要继续扩大规模。

## 10. 必须记录的结果

每个 checkpoint 至少记录：

- checkpoint path
- 训练配置
- epoch
- train loss
- action loss
- visual loss
- offline token accuracy
- offline MAE
- gripper close recall
- gripper transition change recall
- online total success
- per-task online success
- failed demos 列表
- generated-subgoal worker debug jsonl 路径
- 是否使用 `--replan-every-step`
- 是否使用 `--subgoal-mode generated`

推荐记录格式：

```text
checkpoint:
train:
  chunk:
  epoch:
  lr:
  visual_cot:
  visual_loss:
  tune_depth_transformer:
offline:
  token_accuracy:
  mae:
  gripper_close_recall:
  gripper_transition_change_recall:
online:
  total:
  task0:
  task1:
  task2:
  task3:
  task4:
  task5:
  task6:
  task7:
  task8:
  task9:
notes:
```

## 11. 判断标准

### 11.1 训练是否有效

不能只看总 loss。必须同时满足：

- action loss 下降。
- visual loss 存在且为有限值。
- depth transformer 有梯度。
- rank parameter check 通过。
- offline action 指标没有明显低于 action-only。
- 小规模 online 不明显崩。

### 11.2 CoT 是否有效

CoT 有效的最低证据：

- generated 模式在线成功率接近或超过 `none` 模式。
- 失败 task 的成功率有改善，尤其是对 long-horizon 或目标状态更明确的 task。
- generated subgoal 与任务目标有可解释相关性。
- 不是单纯依靠更多 epoch 或更低 loss。

### 11.3 什么时候停止当前方向

满足任一情况应暂停扩大训练：

- generated 模式明显慢到无法完成基本在线评测。
- generated subgoal 视觉上和任务无关。
- generated 模式显著低于 action-only。
- visual loss 下降但 action online success 下降。
- ckp 后期持续变差，类似 ckp60 低于 epoch40 的情况。

## 12. 后续可能需要的代码改动

以下不是 Stage 1 必须项，但可能是后续提升点。

### 12.1 补正式 oracle worker 模式

用途：

- 区分“动作能否利用正确 subgoal”和“模型能否生成正确 subgoal”。

需要：

- env server 向 worker 传 demo id 和当前 step，或 worker 能读取对应 hdf5。
- worker 编码 GT future frame 作为 subgoal embedding。
- 与 `none/generated` 使用同一动作预测接口。

### 12.2 保存 generated subgoal image

当前 worker debug jsonl 主要保存 action chunk、bins、codes。后续应支持：

- 每个 chunk 保存 generated subgoal image。
- 保存当前 observation。
- 可选保存 GT future frame。

用途：

- 快速判断 CoT 是否真的在“想未来状态”。

### 12.3 加速 generated subgoal

当前 `generate_visual_cot_subgoal` 每生成一个 visual token 都跑 LLM 且 `use_cache=False`。后续可优化：

- KV cache。
- batch 内并行。
- 减少生成 token 数。
- 只生成 embedding/code，不 decode image。

### 12.4 暂不做 action-less video 数据管线

当前不接：

- OpenX。
- EPIC-KITCHENS。
- Something-Something V2。
- 其他 action-less video clips。

如果未来重新打开该方向，需要另立文档设计：

- batch 可以没有 action labels。
- 只计算 visual loss。
- mixed dataloader 控制 robot/action-less 比例。
- 每个数据集单独设置 subgoal horizon。
- 不同数据集的图像预处理、语言描述和时间间隔对齐。
- 训练成本和在线收益的评估标准。

## 13. 当前推荐下一步

下一步只做 Stage 1：

1. 跑 `tests/test_phase4_visual_cot.py`。
2. 跑 8 卡 1 epoch smoke train。
3. 检查 visual loss、depth transformer、DDP rank parameter check。
4. smoke 通过后再启动 20 epoch Phase4 小规模正式训练。

暂时不要：

- 直接跑 60 epoch。
- 直接跑 generated-subgoal 500 demos。
- 接 OpenX/action-less video 预训练。
- 用训练 loss 直接判断 CoT 成功。
